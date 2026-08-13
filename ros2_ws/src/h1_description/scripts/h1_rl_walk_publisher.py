#!/usr/bin/env python3
"""Drive Unitree H1 in Gazebo with the official unitree_rl_gym pretrained walk policy.

Publishes position targets to the same /model/h1/joint/<name>/cmd_pos topics used
by standing_pose_publisher.py. Observations come from bridged joint states +
/model/h1/odometry (not a full sim2sim match — Gazebo gains/physics differ).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import rclpy
import torch
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

LEG_JOINTS = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
]

HOLD_JOINTS = {
    "torso_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
}


def _find_gym_root() -> str:
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


def get_gravity_orientation(qx, qy, qz, qw):
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


class H1RlWalkPublisher(Node):
    def __init__(self):
        super().__init__("h1_rl_walk_publisher")
        gym_root = _find_gym_root()
        sys.path.insert(0, gym_root)

        config_path = os.path.join(gym_root, "deploy", "deploy_mujoco", "configs", "h1.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", gym_root)
        self._kps = np.array(config["kps"], dtype=np.float32)  # unused; Gazebo PD owns gains
        self._default = np.array(config["default_angles"], dtype=np.float32)
        self._ang_vel_scale = float(config["ang_vel_scale"])
        self._dof_pos_scale = float(config["dof_pos_scale"])
        self._dof_vel_scale = float(config["dof_vel_scale"])
        self._action_scale = float(config["action_scale"])
        self._cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self._num_actions = int(config["num_actions"])
        self._num_obs = int(config["num_obs"])
        self._cmd = np.array(config["cmd_init"], dtype=np.float32)

        self._policy = torch.jit.load(policy_path)
        self._policy.eval()
        self._action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs = np.zeros(self._num_obs, dtype=np.float32)
        self._q = self._default.copy()
        self._dq = np.zeros(self._num_actions, dtype=np.float32)
        self._omega = np.zeros(3, dtype=np.float32)
        self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # x y z w
        self._have_joints = False
        self._have_odom = False
        self._phase_t = 0.0
        self._dt = 0.02
        self._ticks = 0
        # A long warmup was tried first (hold the default squat, then enable
        # the policy) but that made things worse: the static squat has no
        # feedback at all and reliably topples in Gazebo within ~2-3s (logged:
        # pelvis down and pitched ~70deg by t=2.8s), so the policy -- which
        # is actually trained to balance -- never got a chance to run before
        # it was already fallen. Enable it as soon as we have real data
        # instead of waiting for an arbitrary settle time.
        self._warmup_ticks = 2
        self._policy_enabled = False

        self._pubs = {
            name: self.create_publisher(Float64, f"/model/h1/joint/{name}/cmd_pos", 10)
            for name in LEG_JOINTS + list(HOLD_JOINTS)
        }
        # Bridged GZ joint states are remapped to /joint_states in h1_gazebo.launch.py
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(Odometry, "/model/h1/odometry", self._odom_cb, 10)
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f"H1 RL walk ready (policy={policy_path}, cmd={self._cmd.tolist()}, "
            f"warmup_s={self._warmup_ticks * self._dt:.1f})"
        )

    def _joint_cb(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        try:
            for i, joint in enumerate(LEG_JOINTS):
                j = name_to_i[joint]
                self._q[i] = float(msg.position[j])
                if msg.velocity:
                    self._dq[i] = float(msg.velocity[j])
            self._have_joints = True
        except KeyError:
            pass

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self._quat[:] = (q.x, q.y, q.z, q.w)
        w = msg.twist.twist.angular
        self._omega[:] = (w.x, w.y, w.z)
        self._have_odom = True

    def _tick(self):
        targets = dict(HOLD_JOINTS)
        for i, name in enumerate(LEG_JOINTS):
            targets[name] = float(self._default[i])

        self._ticks += 1
        if self._ticks == self._warmup_ticks:
            self._policy_enabled = True
            self.get_logger().info("Warmup done — enabling RL walk policy")

        z = None
        if self._have_odom:
            # Projected gravity in body frame is ~[0,0,-1] when upright
            # (same convention as unitree_rl_gym's MuJoCo deploy).
            gravity = get_gravity_orientation(*self._quat)
            upright_enough = gravity[2] < -0.5
        else:
            upright_enough = False
            gravity = None

        if (
            self._policy_enabled
            and self._have_joints
            and self._have_odom
            and upright_enough
        ):
            qj = (self._q - self._default) * self._dof_pos_scale
            dqj = self._dq * self._dof_vel_scale
            omega = self._omega * self._ang_vel_scale
            if gravity is None:
                gravity = get_gravity_orientation(*self._quat)

            self._phase_t += self._dt
            period = 0.8
            phase = self._phase_t % period / period
            sin_phase = math.sin(2 * math.pi * phase)
            cos_phase = math.cos(2 * math.pi * phase)

            obs = self._obs
            obs[:3] = omega
            obs[3:6] = gravity
            obs[6:9] = self._cmd * self._cmd_scale
            obs[9 : 9 + self._num_actions] = qj
            obs[9 + self._num_actions : 9 + 2 * self._num_actions] = dqj
            obs[9 + 2 * self._num_actions : 9 + 3 * self._num_actions] = self._action
            obs[9 + 3 * self._num_actions : 9 + 3 * self._num_actions + 2] = (
                sin_phase,
                cos_phase,
            )
            with torch.no_grad():
                self._action = (
                    self._policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
                )
            target = self._action * self._action_scale + self._default
            for i, name in enumerate(LEG_JOINTS):
                targets[name] = float(target[i])

        for name, value in targets.items():
            msg = Float64()
            msg.data = value
            self._pubs[name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = H1RlWalkPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
