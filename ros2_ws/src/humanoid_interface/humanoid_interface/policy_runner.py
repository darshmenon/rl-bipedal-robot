"""H1 RL walk policy wrapper (unitree_rl_gym pretrained motion.pt)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import yaml

from humanoid_interface.joint_names import HOLD_JOINTS, LEG_JOINTS


def find_gym_root() -> str:
    env = os.environ.get("UNITREE_RL_GYM_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "..", "..", "..", "..", "unitree_rl_gym"),
        os.path.join(os.path.expanduser("~"), "rl-bipedal-walking", "unitree_rl_gym"),
        "/home/asimov/rl-bipedal-walking/unitree_rl_gym",
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(os.path.join(path, "deploy", "pre_train", "h1", "motion.pt")):
            return path
    raise FileNotFoundError(
        "unitree_rl_gym not found; set UNITREE_RL_GYM_ROOT or clone next to ros2_ws"
    )


def gravity_orientation(qx, qy, qz, qw):
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


class H1PolicyRunner:
    def __init__(self):
        gym_root = find_gym_root()
        sys.path.insert(0, gym_root)

        config_path = os.path.join(gym_root, "deploy", "deploy_mujoco", "configs", "h1.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", gym_root)
        self.default = np.array(config["default_angles"], dtype=np.float32)
        self.ang_vel_scale = float(config["ang_vel_scale"])
        self.dof_pos_scale = float(config["dof_pos_scale"])
        self.dof_vel_scale = float(config["dof_vel_scale"])
        self.action_scale = float(config["action_scale"])
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.num_actions = int(config["num_actions"])
        self.num_obs = int(config["num_obs"])
        self.cmd = np.array(config["cmd_init"], dtype=np.float32)

        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        self.policy_path = policy_path

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.q = self.default.copy()
        self.dq = np.zeros(self.num_actions, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.have_joints = False
        self.have_odom = False
        self.phase_t = 0.0
        self.dt = 0.02
        self.ticks = 0
        self.warmup_ticks = 2
        self.policy_enabled = False

    def set_cmd(self, cmd: np.ndarray):
        self.cmd = np.array(cmd, dtype=np.float32)

    def update_joints(self, names, positions, velocities):
        name_to_i = {n: i for i, n in enumerate(names)}
        try:
            for i, joint in enumerate(LEG_JOINTS):
                j = name_to_i[joint]
                self.q[i] = float(positions[j])
                if velocities:
                    self.dq[i] = float(velocities[j])
            self.have_joints = True
        except KeyError:
            pass

    def update_odom(self, quat, omega):
        self.quat[:] = quat
        self.omega[:] = omega
        self.have_odom = True

    def step(self) -> dict[str, float]:
        targets = dict(HOLD_JOINTS)
        for i, name in enumerate(LEG_JOINTS):
            targets[name] = float(self.default[i])

        self.ticks += 1
        if self.ticks == self.warmup_ticks:
            self.policy_enabled = True

        upright_enough = False
        gravity = None
        if self.have_odom:
            gravity = gravity_orientation(*self.quat)
            upright_enough = gravity[2] < -0.5

        if self.policy_enabled and self.have_joints and self.have_odom and upright_enough:
            qj = (self.q - self.default) * self.dof_pos_scale
            dqj = self.dq * self.dof_vel_scale
            omega = self.omega * self.ang_vel_scale
            if gravity is None:
                gravity = gravity_orientation(*self.quat)

            self.phase_t += self.dt
            period = 0.8
            phase = self.phase_t % period / period
            sin_phase = math.sin(2 * math.pi * phase)
            cos_phase = math.cos(2 * math.pi * phase)

            obs = self.obs
            obs[:3] = omega
            obs[3:6] = gravity
            obs[6:9] = self.cmd * self.cmd_scale
            obs[9 : 9 + self.num_actions] = qj
            obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj
            obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
            obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = (
                sin_phase,
                cos_phase,
            )
            with torch.no_grad():
                self.action = (
                    self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
                )
            target = self.action * self.action_scale + self.default
            for i, name in enumerate(LEG_JOINTS):
                targets[name] = float(target[i])

        return targets
