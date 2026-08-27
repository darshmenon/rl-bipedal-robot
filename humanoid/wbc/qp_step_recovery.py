#!/usr/bin/env python3
"""Capture-point stepping extension of the QP standing balance controller.

qp_balance_controller.py rejects pushes via ankle/hip strategy on a fixed
footprint -- it works, but only up to the point where the capture point (the
Linear-Inverted-Pendulum estimate of "where the CoM needs a foot under it to
stop") leaves the support polygon. Past that, no torque at a fixed foot can
recover balance; the only option is to step.

This module adds that: each control step, while double-support, it checks
whether the instantaneous capture point (ICP)

    xcp = com_xy + com_vel_xy / omega0,   omega0 = sqrt(g / com_height)

has left a safety-shrunk support polygon. If so, it swings the foot farther
from the ICP out to the ICP location (clipped to a max step length and a
minimum foot separation) along a minimum-jerk trajectory, tracked as a QP
position task on the swing foot while the QP's contact set drops to the
single stance foot. On landing it returns to double support and re-checks --
so a big enough push produces multiple steps, not just one.

Following Pratt et al.'s capture-point push recovery and the Atlas step
timing/location adjustment work (arXiv:1703.00477), landing is
contact-triggered rather than a fixed timer, and the step target is
re-solved from the current ICP every control step during the swing (not
frozen at trigger time) -- so the foot lands under wherever the ICP actually
ends up, not wherever it was when the step was first triggered.

Simplifications (still a baseline, not a production stepper):
- Step target reachability is checked only via a max-step-length and
  min-separation clamp, not full kinematic/self-collision limits.
- No swing-speed-up: swing duration itself doesn't shorten under continued
  divergence, only the landing point re-targets and touchdown ends the swing
  as soon as it's detected (rather than waiting out a full fixed duration).
- Flat ground assumed for the landing height.

Run:
    python -m humanoid.wbc.qp_step_recovery --push --push-force 0 400 0 --plot out.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from humanoid.wbc.qp_balance_controller import (
    LEFT_FOOT_BODY,
    PELVIS_BODY,
    RIGHT_FOOT_BODY,
    XML_PATH,
    QPBalanceController,
    collidable_geom,
    com_state,
    foot_contact_points,
)

SWING_DURATION = 0.35  # s, nominal -- actual landing is contact-triggered and may end sooner
MIN_SWING_TIME = 0.12  # s, grace period before touchdown is allowed to end the swing early
SWING_HEIGHT = 0.06  # m, peak foot clearance
MAX_STEP_LENGTH = 0.45  # m
MIN_FOOT_SEPARATION = 0.15  # m, lateral distance kept from the stance foot
SUPPORT_MARGIN = 0.75  # shrink the support polygon toward its centroid by this factor for the ICP-inside test
HULL_ORDER = [0, 1, 3, 2]  # front, back of foot A then foot B -> polygon boundary order


def minimum_jerk(s):
    """Blend factor and its time-derivative-scale for s in [0, 1]."""
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    dblend = 30 * s**2 - 60 * s**3 + 30 * s**4
    return blend, dblend


def point_in_convex_polygon(pt, poly):
    n = len(poly)
    sign = None
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(cross) < 1e-9:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def support_polygon(model, data, bodies):
    pts = []
    for body in bodies:
        pts += [p[:2] for p, _ in foot_contact_points(model, data, body)]
    return np.array(pts)


class CaptureStepController:
    """Wraps QPBalanceController with a double-support / single-support-swing
    state machine driven by the instantaneous capture point."""

    def __init__(self, model, com_height):
        self.model = model
        self.qp = QPBalanceController(model)
        self.com_height = com_height
        self.omega0 = np.sqrt(9.81 / com_height)
        self.ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.phase = "stand"
        self.swing_body = None
        self.swing_gid = None
        self.swing_start = None
        self.swing_goal = None
        self.swing_t0 = None
        self.pre_swing_com_xy = None
        self.n_steps_taken = 0

    def _double_support_com_xy(self, data):
        pts = support_polygon(self.model, data, (LEFT_FOOT_BODY, RIGHT_FOOT_BODY))
        return pts.mean(axis=0)

    def _stance_body(self):
        return RIGHT_FOOT_BODY if self.swing_body == LEFT_FOOT_BODY else LEFT_FOOT_BODY

    def _clipped_target(self, cp, start_xy, stance_c):
        """ICP, clamped to a reachable/collision-safe landing spot."""
        target_xy = cp.copy()
        delta = target_xy - start_xy
        dist = np.linalg.norm(delta)
        if dist > MAX_STEP_LENGTH:
            target_xy = start_xy + delta / dist * MAX_STEP_LENGTH
        lateral = target_xy[1] - stance_c[1]
        if abs(lateral) < MIN_FOOT_SEPARATION:
            sign = np.sign(lateral) if lateral != 0 else np.sign(start_xy[1] - stance_c[1]) or 1.0
            target_xy[1] = stance_c[1] + sign * MIN_FOOT_SEPARATION
        return target_xy

    def _touched_down(self, data):
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == self.swing_gid or c.geom2 == self.swing_gid:
                return True
        return False

    def _maybe_trigger_step(self, data, t):
        com, com_vel = com_state(self.model, data)
        cp = com[:2] + com_vel[:2] / self.omega0

        poly = support_polygon(self.model, data, (LEFT_FOOT_BODY, RIGHT_FOOT_BODY))[HULL_ORDER]
        centroid = poly.mean(axis=0)
        margin_poly = centroid + SUPPORT_MARGIN * (poly - centroid)
        if point_in_convex_polygon(cp, margin_poly):
            return False

        left_c = support_polygon(self.model, data, (LEFT_FOOT_BODY,)).mean(axis=0)
        right_c = support_polygon(self.model, data, (RIGHT_FOOT_BODY,)).mean(axis=0)
        self.swing_body = LEFT_FOOT_BODY if np.linalg.norm(left_c - cp) > np.linalg.norm(right_c - cp) else RIGHT_FOOT_BODY
        stance_c = right_c if self.swing_body == LEFT_FOOT_BODY else left_c

        self.swing_gid, swing_bid = collidable_geom(self.model, self.swing_body)
        start = data.xpos[swing_bid].copy()
        target_xy = self._clipped_target(cp, start[:2], stance_c)

        self.swing_start = start
        self.swing_goal = np.array([target_xy[0], target_xy[1], start[2]])
        self.swing_t0 = t
        self.pre_swing_com_xy = self._double_support_com_xy(data)
        self.phase = "swing"
        self.n_steps_taken += 1
        return True

    def step(self, data, t):
        """Returns (com_des_xy, stance_bodies, swing_task)."""
        if self.phase == "stand":
            self._maybe_trigger_step(data, t)

        if self.phase == "swing":
            elapsed = t - self.swing_t0
            if elapsed >= MIN_SWING_TIME and self._touched_down(data):
                s = 1.0
            else:
                s = np.clip(elapsed / SWING_DURATION, 0.0, 1.0)

            # Step-location adjustment (Pratt et al. / Atlas step timing & location):
            # re-solve the landing target from the *current* ICP every control
            # step, instead of freezing it at trigger time, so the foot lands
            # under wherever the CoM actually ends up.
            if s < 1.0:
                com, com_vel = com_state(self.model, data)
                cp_now = com[:2] + com_vel[:2] / self.omega0
                stance_c = support_polygon(self.model, data, (self._stance_body(),)).mean(axis=0)
                self.swing_goal[:2] = self._clipped_target(cp_now, self.swing_start[:2], stance_c)

            blend, dblend = minimum_jerk(s)
            xy = self.swing_start[:2] + blend * (self.swing_goal[:2] - self.swing_start[:2])
            z = self.swing_start[2] + blend * (self.swing_goal[2] - self.swing_start[2]) + SWING_HEIGHT * np.sin(np.pi * s)
            vel_xy = dblend / SWING_DURATION * (self.swing_goal[:2] - self.swing_start[:2])
            vel_z = dblend / SWING_DURATION * (self.swing_goal[2] - self.swing_start[2]) + \
                SWING_HEIGHT * (np.pi / SWING_DURATION) * np.cos(np.pi * s)
            pos_des = np.array([xy[0], xy[1], z])
            vel_des = np.array([vel_xy[0], vel_xy[1], vel_z])
            swing_task = {"body": self.swing_body, "pos_des": pos_des, "vel_des": vel_des}

            stance_body = self._stance_body()
            stance_c = support_polygon(self.model, data, (stance_body,)).mean(axis=0)
            com_des_xy = self.pre_swing_com_xy + blend * (stance_c - self.pre_swing_com_xy)

            if s >= 1.0:
                self.phase = "stand"
                self.swing_body = None
                return com_des_xy, (stance_body,), None
            return com_des_xy, (stance_body,), swing_task

        return self._double_support_com_xy(data), (LEFT_FOOT_BODY, RIGHT_FOOT_BODY), None


def run(duration, push, push_force, push_time, push_duration, plot_path, use_viewer):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    com0, _ = com_state(model, data)
    pelvis_z_des = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)][2])
    stepper = CaptureStepController(model, com_height=float(com0[2]))
    push_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)

    sim_dt = model.opt.timestep
    control_decimation = 10
    n_steps = int(duration / sim_dt)

    log_t, log_com, log_pelvis_z, log_phase = [], [], [], []
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
            com_des_xy, stance_bodies, swing_task = stepper.step(data, t)
            com_des = np.array([com_des_xy[0], com_des_xy[1], stepper.com_height])
            tau, com, _, status = stepper.qp.compute_torque(data, com_des, stance_bodies, swing_task)
            log_t.append(t)
            log_com.append(com[:2].copy())
            log_pelvis_z.append(float(data.xpos[stepper.qp.pelvis_bid][2]))
            log_phase.append(stepper.phase)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.render()

    if viewer is not None:
        viewer.close()

    log_pelvis_z = np.array(log_pelvis_z)
    final_z = log_pelvis_z[-1] if len(log_pelvis_z) else 0.0
    min_z = log_pelvis_z.min() if len(log_pelvis_z) else 0.0
    upright = min_z > 0.5 * pelvis_z_des

    print(f"duration_s={duration:.1f} steps={n_steps} pelvis_z_des={pelvis_z_des:.3f}")
    print(f"final_pelvis_z={final_z:.3f} min_pelvis_z={min_z:.3f} upright={upright}")
    print(f"steps_taken={stepper.n_steps_taken} final_phase={stepper.phase}")

    if plot_path:
        plot_trajectory(np.array(log_t), np.array(log_com), log_pelvis_z, log_phase, plot_path)
        print(f"wrote {plot_path}")

    return 0 if upright else 1


def plot_trajectory(t, com_xy, pelvis_z, phase, out_path):
    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(11, 5))

    swinging = np.array([p == "swing" for p in phase])
    ax_xy.plot(com_xy[:, 0], com_xy[:, 1], "-", color="tab:blue", linewidth=1.2, label="CoM")
    if swinging.any():
        ax_xy.scatter(com_xy[swinging, 0], com_xy[swinging, 1], color="tab:red", s=6, zorder=3, label="stepping")
    ax_xy.scatter(*com_xy[0], color="tab:blue", marker="o", zorder=4)
    ax_xy.scatter(*com_xy[-1], color="tab:blue", marker="x", zorder=4)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("CoM trajectory (red = actively stepping)")
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
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-force", type=float, nargs=3, default=[0.0, 400.0, 0.0])
    parser.add_argument("--push-time", type=float, default=1.0)
    parser.add_argument("--push-duration", type=float, default=0.1)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    return run(args.duration, args.push, np.array(args.push_force), args.push_time,
               args.push_duration, args.plot, args.viewer)


if __name__ == "__main__":
    raise SystemExit(main())
