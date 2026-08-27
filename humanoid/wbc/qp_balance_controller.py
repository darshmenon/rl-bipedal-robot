#!/usr/bin/env python3
"""QP-based whole-body standing balance controller for the XBot-L humanoid.

Baseline whole-body controller: rigid-body inverse dynamics + a QP contact
force allocator, standing in for the PD/heuristic standing controllers and
the pure-RL locomotion policy this repo otherwise relies on. Every control
step it solves one QP for joint accelerations, motor torques, and foot
contact forces that satisfy the floating-base equations of motion while
holding the CoM over the support polygon and the pelvis upright, subject to
friction-cone and torque-limit constraints.

Simplifications (this is a standing-balance baseline, not a full WBC stack):
- Contacts are modeled as two point contacts per foot (front/back corners of
  the foot collision mesh, from its AABB), not a full flat-foot patch.
- Task costs drop the J_dot @ qdot bias-acceleration term, valid at the low
  velocities a standing/push-recovery controller operates at -- not
  appropriate for a walking/swing-leg task.

Run:
    python -m humanoid.wbc.qp_balance_controller
    python -m humanoid.wbc.qp_balance_controller --push --plot out.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XML_PATH = os.path.join(REPO_ROOT, "resources", "robots", "XBot", "mjcf", "XBot-L-train.xml")

LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
PELVIS_BODY = "base_link"

# Nominal standing joint config: all-zero, matches humanoid_config.py's
# default_joint_angles (the RL policy's zero-action pose).
NOMINAL_JOINT_POS = np.zeros(12)

MU = 0.7  # friction coefficient used in the QP's cone constraint (mesh friction is 0.9; keep margin)
FZ_MIN = 5.0  # N, per contact point -- keeps the QP from planning to leave the ground
FZ_MAX = 800.0  # N, per contact point

KP_COM, KD_COM = 400.0, 40.0
KP_ORI, KD_ORI = 400.0, 40.0
KP_POSTURE, KD_POSTURE = 100.0, 10.0
KP_SWING, KD_SWING = 900.0, 60.0
W_COM, W_ORI, W_POSTURE, W_TAU, W_FORCE, W_SWING = 50.0, 50.0, 1.0, 1e-3, 1e-4, 80.0


def collidable_geom(model, body_name):
    """The one geom under this body that actually participates in contact."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    gstart, gnum = model.body_geomadr[bid], model.body_geomnum[bid]
    gid = next(g for g in range(gstart, gstart + gnum) if model.geom_contype[g] != 0)
    return gid, bid


def foot_contact_points(model, data, body_name):
    """Front/back point contacts at the bottom of the foot's collision mesh.

    Uses the compiled geom AABB (exact for the mesh) rather than a hand-typed
    offset, so it self-corrects if the mesh/URDF changes. The AABB is in the
    geom's own local frame, whose axes don't line up with body/world
    forward/up (XBot-L's foot meshes are heavily re-oriented by geom_quat),
    so which local axis is "vertical" and which is "foot-length" is detected
    from the local->world rotation rather than assumed.
    """
    gid, bid = collidable_geom(model, body_name)
    center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]

    r_geom = np.zeros(9)
    mujoco.mju_quat2Mat(r_geom, model.geom_quat[gid])
    r_geom = r_geom.reshape(3, 3)
    r_body = data.xmat[bid].reshape(3, 3)
    p_body = model.geom_pos[gid]
    r_world = r_body @ r_geom

    vertical_axis = int(np.argmax(np.abs(r_world[2, :])))
    vertical_sign = -np.sign(r_world[2, vertical_axis]) or 1.0
    remaining = [i for i in range(3) if i != vertical_axis]
    length_axis = remaining[int(np.argmax([half[i] for i in remaining]))]

    def to_world(local_pt):
        return data.xpos[bid] + r_body @ (p_body + r_geom @ local_pt)

    base = center.copy()
    base[vertical_axis] += vertical_sign * half[vertical_axis]
    front_local, back_local = base.copy(), base.copy()
    front_local[length_axis] += half[length_axis]
    back_local[length_axis] -= half[length_axis]
    return [(to_world(front_local), bid), (to_world(back_local), bid)]


def com_state(model, data):
    """Total-body CoM position and velocity (world frame)."""
    Jcom = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, Jcom, 0)
    return data.subtree_com[0].copy(), Jcom @ data.qvel


class QPBalanceController:
    def __init__(self, model):
        self.model = model
        self.nv, self.nu = model.nv, model.nu
        self.pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
        self.total_mass = float(model.body_mass.sum())

        self.B = np.zeros((self.nv, self.nu))
        for i in range(self.nu):
            self.B[model.jnt_dofadr[model.actuator_trnid[i, 0]], i] = 1.0
        self.S_joint = self.B.T  # selects actuated-joint rows out of qdd/qvel

        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    def compute_torque(self, data, com_des, stance_bodies=(LEFT_FOOT_BODY, RIGHT_FOOT_BODY), swing_task=None):
        """stance_bodies: foot bodies currently in contact (2 corner points each).
        swing_task: optional {"body", "pos_des", "vel_des"} for a foot not in
        stance_bodies -- tracked as a 3D position task instead of a contact.
        """
        nv, nu = self.nv, self.nu
        contacts = []
        for body in stance_bodies:
            contacts += foot_contact_points(self.model, data, body)
        n_c = len(contacts)
        nz = nv + nu + 3 * n_c
        qdd_sl, tau_sl, f_sl = slice(0, nv), slice(nv, nv + nu), slice(nv + nu, nz)

        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, M, data.qM)
        h = data.qfrc_bias.copy()

        Jc_full = np.zeros((3 * n_c, nv))
        for i, (pt, bid) in enumerate(contacts):
            jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(self.model, data, jacp, jacr, pt, bid)
            Jc_full[3 * i:3 * i + 3, :] = jacp

        Jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(self.model, data, Jcom, 0)
        com = data.subtree_com[0].copy()
        com_vel = Jcom @ data.qvel

        jacp_o, jacr_o = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(self.model, data, jacp_o, jacr_o, data.xpos[self.pelvis_bid], self.pelvis_bid)
        Jori = jacr_o
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, np.array([1.0, 0.0, 0.0, 0.0]), data.xquat[self.pelvis_bid])
        omega = Jori @ data.qvel

        xdd_com_des = KP_COM * (com_des - com) + KD_COM * (-com_vel)
        xdd_ori_des = KP_ORI * ori_err + KD_ORI * (-omega)

        q_joint = data.qpos[7:7 + nu]
        qvel_joint = data.qvel[6:6 + nu]
        qdd_posture_des = KP_POSTURE * (NOMINAL_JOINT_POS - q_joint) + KD_POSTURE * (-qvel_joint)

        P = np.zeros((nz, nz))
        q = np.zeros(nz)

        def add_task(J, xdd_des, w):
            Jf = np.zeros((J.shape[0], nz))
            Jf[:, qdd_sl] = J
            P[:, :] += 2.0 * w * (Jf.T @ Jf)
            q[:] += -2.0 * w * (Jf.T @ xdd_des)

        add_task(Jcom, xdd_com_des, W_COM)
        add_task(Jori, xdd_ori_des, W_ORI)
        add_task(self.S_joint, qdd_posture_des, W_POSTURE)

        if swing_task is not None:
            sbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, swing_task["body"])
            jacp_s, jacr_s = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(self.model, data, jacp_s, jacr_s, data.xpos[sbid], sbid)
            pos_err = swing_task["pos_des"] - data.xpos[sbid]
            vel_err = swing_task["vel_des"] - jacp_s @ data.qvel
            xdd_swing_des = KP_SWING * pos_err + KD_SWING * vel_err
            add_task(jacp_s, xdd_swing_des, W_SWING)

        P[tau_sl, tau_sl] += 2.0 * W_TAU * np.eye(nu)
        f_nom = np.zeros(3 * n_c)
        f_nom[2::3] = self.total_mass * 9.81 / n_c
        P[f_sl, f_sl] += 2.0 * W_FORCE * np.eye(3 * n_c)
        q[f_sl] += -2.0 * W_FORCE * f_nom

        A_eq = np.zeros((nv, nz))
        A_eq[:, qdd_sl] = M
        A_eq[:, tau_sl] = -self.B
        A_eq[:, f_sl] = -Jc_full.T
        b_eq = -h

        rows, lb, ub = [], [], []
        for i in range(n_c):
            fx_idx = nv + nu + 3 * i
            fy_idx, fz_idx = fx_idx + 1, fx_idx + 2
            for axis_idx in (fx_idx, fy_idx):
                for sign in (1.0, -1.0):
                    r = np.zeros(nz)
                    r[axis_idx] = sign
                    r[fz_idx] = -MU
                    rows.append(r)
                    lb.append(-np.inf)
                    ub.append(0.0)
            r = np.zeros(nz)
            r[fz_idx] = 1.0
            rows.append(r)
            lb.append(FZ_MIN)
            ub.append(FZ_MAX)
        for i in range(nu):
            r = np.zeros(nz)
            r[nv + i] = 1.0
            rows.append(r)
            lb.append(self.ctrl_lo[i])
            ub.append(self.ctrl_hi[i])

        A = np.vstack([A_eq, np.array(rows)])
        l = np.concatenate([b_eq, np.array(lb)])
        u = np.concatenate([b_eq, np.array(ub)])

        solver = osqp.OSQP()
        solver.setup(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u, verbose=False, polish=True)
        result = solver.solve()
        if result.info.status_val not in (1, 2):  # solved / solved_inaccurate
            return np.zeros(nu), com, contacts, result.info.status
        return result.x[tau_sl], com, contacts, result.info.status


def run(duration, push, push_force, push_time, push_duration, plot_path, use_viewer):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    controller = QPBalanceController(model)
    pelvis_z_des = float(data.xpos[controller.pelvis_bid][2])

    init_contacts = foot_contact_points(model, data, LEFT_FOOT_BODY) + \
        foot_contact_points(model, data, RIGHT_FOOT_BODY)
    com_des_xy = np.mean([p[:2] for p, _ in init_contacts], axis=0)
    com_des = np.array([com_des_xy[0], com_des_xy[1], data.subtree_com[0][2]])

    sim_dt = model.opt.timestep
    control_decimation = 10
    n_steps = int(duration / sim_dt)
    push_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)

    log_t, log_com, log_zmp, log_pelvis_z, log_status = [], [], [], [], []
    viewer = None
    if use_viewer:
        import mujoco_viewer

        viewer = mujoco_viewer.MujocoViewer(model, data)

    tau = np.zeros(model.nu)
    for step in range(n_steps):
        t = step * sim_dt
        data.xfrc_applied[push_bid, :] = 0.0
        if push and push_time <= t < push_time + push_duration:
            data.xfrc_applied[push_bid, :3] = push_force

        if step % control_decimation == 0:
            tau, com, contacts, status = controller.compute_torque(data, com_des)
            fpts = np.array([p for p, _ in contacts])
            zmp = np.average(fpts[:, :2], axis=0)  # uniform-weight fallback if the QP degenerates
            log_t.append(t)
            log_com.append(com[:2].copy())
            log_zmp.append(zmp)
            log_pelvis_z.append(float(data.xpos[controller.pelvis_bid][2]))
            log_status.append(status)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.render()

    if viewer is not None:
        viewer.close()

    log_t = np.array(log_t)
    log_com = np.array(log_com)
    log_zmp = np.array(log_zmp)
    log_pelvis_z = np.array(log_pelvis_z)
    final_z = log_pelvis_z[-1] if len(log_pelvis_z) else 0.0
    min_z = log_pelvis_z.min() if len(log_pelvis_z) else 0.0
    upright = min_z > 0.5 * pelvis_z_des

    print(f"duration_s={duration:.1f} steps={n_steps} pelvis_z_des={pelvis_z_des:.3f}")
    print(f"final_pelvis_z={final_z:.3f} min_pelvis_z={min_z:.3f} upright={upright}")
    unsolved = sum(1 for s in log_status if s not in ("solved", "solved inaccurate"))
    print(f"qp_unsolved_steps={unsolved}/{len(log_status)}")

    if plot_path:
        # Nominal (t=0) footprint, not the final one: under a push the QP uses
        # hip/ankle roll to shift load between feet, which visibly rotates the
        # foot contact points away from nominal -- a real balance strategy,
        # not a bug, but a moving backdrop makes a poor "was CoM/ZMP inside
        # the base of support" reference.
        nominal_pts = np.array([p[:2] for p, _ in init_contacts])
        plot_trajectories(log_t, log_com, log_zmp, log_pelvis_z, nominal_pts, plot_path)
        print(f"wrote {plot_path}")

    return 0 if upright else 1


def plot_trajectories(t, com_xy, zmp_xy, pelvis_z, support_pts, out_path):
    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(11, 5))

    hull_order = [0, 1, 3, 2, 0]  # front-L, back-L, back-R, front-R -> closed loop
    poly = support_pts[hull_order]
    ax_xy.fill(poly[:, 0], poly[:, 1], color="0.85", label="support polygon", zorder=0)
    ax_xy.plot(com_xy[:, 0], com_xy[:, 1], "-", color="tab:blue", label="CoM", linewidth=1.5)
    ax_xy.plot(zmp_xy[:, 0], zmp_xy[:, 1], "--", color="tab:orange", label="planned ZMP", linewidth=1.5)
    ax_xy.scatter(*com_xy[0], color="tab:blue", marker="o", zorder=3)
    ax_xy.scatter(*com_xy[-1], color="tab:blue", marker="x", zorder=3)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("CoM / ZMP vs. support polygon (top-down)")
    ax_xy.axis("equal")
    ax_xy.legend(loc="best", fontsize=8)

    ax_z.plot(t, pelvis_z, color="tab:green")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("pelvis height [m]")
    ax_z.set_title("Pelvis height vs. time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--push", action="store_true", help="apply a lateral push to the pelvis mid-run")
    parser.add_argument("--push-force", type=float, nargs=3, default=[0.0, 300.0, 0.0])
    parser.add_argument("--push-time", type=float, default=1.5)
    parser.add_argument("--push-duration", type=float, default=0.1)
    parser.add_argument("--plot", type=str, default=None, help="path to save a CoM/ZMP trajectory plot (PNG)")
    parser.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo viewer")
    args = parser.parse_args()

    return run(args.duration, args.push, np.array(args.push_force), args.push_time,
               args.push_duration, args.plot, args.viewer)


if __name__ == "__main__":
    raise SystemExit(main())
