#!/usr/bin/env python3
"""Headless MuJoCo test of Unitree pretrained H1 walking policy."""

import os
import sys
import time

import mujoco
import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unitree_rl_gym"))
sys.path.insert(0, ROOT)

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    duration = float(os.environ.get("H1_WALK_DURATION", "20"))
    config_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "deploy", "deploy_mujoco", "configs", "h1.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = torch.jit.load(policy_path)
    policy.eval()

    n_steps = int(duration / simulation_dt)
    counter = 0
    heights = []
    xs = []
    start = time.time()

    for _ in range(n_steps):
        tau = pd_control(
            target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds
        )
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1

        if counter % control_decimation == 0:
            qj = (d.qpos[7:] - default_angles) * dof_pos_scale
            dqj = d.qvel[6:] * dof_vel_scale
            quat = d.qpos[3:7]
            omega = d.qvel[3:6] * ang_vel_scale
            gravity_orientation = get_gravity_orientation(quat)

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
                [sin_phase, cos_phase]
            )
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        if counter % int(0.5 / simulation_dt) == 0:
            heights.append(float(d.qpos[2]))
            xs.append(float(d.qpos[0]))

    elapsed = time.time() - start
    final_z = float(d.qpos[2])
    final_x = float(d.qpos[0])
    min_z = min(heights) if heights else final_z
    dist = final_x - xs[0] if xs else 0.0
    upright = min_z > 0.6 and final_z > 0.7
    walked = dist > 0.5

    print(f"duration_s={duration:.1f} wall_s={elapsed:.1f} steps={n_steps}")
    print(f"final_xyz=({d.qpos[0]:.3f},{d.qpos[1]:.3f},{d.qpos[2]:.3f})")
    print(f"min_z={min_z:.3f} forward_m={dist:.3f}")
    print(f"upright={upright} walked={walked}")
    if upright and walked:
        print("RESULT=PASS")
        return 0
    print("RESULT=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
