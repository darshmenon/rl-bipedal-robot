#!/usr/bin/env python3
"""Kinematic walking-pattern generator + analytic leg IK for the bipedal robot.

Publishes position targets to /leg_position_controller/commands, the topic
the JointGroupPositionController (ros2_controllers.yaml) actually listens on.

Structurally inspired by classic footstep -> swing/CoM trajectory -> inverse
kinematics walking-pattern-generator pipelines (e.g. LIPM-based generators
such as open-rdc/ROS2_Walking_Pattern_Generator), reimplemented independently
here with no code or package dependency on that or any other project.

Only sagittal-plane (pitch) motion is generated: the hip/knee/ankle joints in
bipedal.urdf all rotate about the pitch axis only, so there is no roll DOF for
lateral CoM shifting. This is an open-loop trapezoidal gait, not a full 3D
LIPM balance controller.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Leg geometry, from bipedal.urdf (thigh: hip->knee, shin: knee->ankle).
THIGH_LENGTH = 0.4
SHIN_LENGTH = 0.35

# Joint order expected by leg_position_controller (ros2_controllers.yaml).
JOINT_ORDER = [
    "left_hip_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_joint", "right_knee_joint", "right_ankle_joint",
]

# Per-joint limits, from bipedal.urdf. The swing foot's ankle briefly wants to
# exceed its +-0.5 rad limit mid-swing to stay level; that's harmless in the
# air, so all commands are clamped here rather than relying on the sim to do it.
JOINT_LIMITS = {
    "left_hip_joint": (-1.57, 1.57), "left_knee_joint": (-2.0, 0.0), "left_ankle_joint": (-0.5, 0.5),
    "right_hip_joint": (-1.57, 1.57), "right_knee_joint": (-2.0, 0.0), "right_ankle_joint": (-0.5, 0.5),
}

STANCE_HEIGHT = 0.71     # constant hip-to-ankle height (m); LIPM-style fixed CoM height
                         # (chosen so ankle_pitch stays within its +-0.5 rad joint limit
                         # across the step-length range below)
SWING_CLEARANCE = 0.06   # peak foot lift during swing (m)
STEP_TIME = 0.35         # single-leg swing duration (s); full L+R cycle = 2x this
STEP_GAIN = 0.5          # step length (m) per (m/s) of commanded forward speed
MAX_STEP_LENGTH = 0.18
CONTROL_RATE_HZ = 100.0


def leg_ik(x, z, l1=THIGH_LENGTH, l2=SHIN_LENGTH):
    """Analytic 2-link planar IK.

    Ankle target (x forward, z down, both hip-relative) -> (hip_pitch,
    knee_pitch, ankle_pitch) with the ankle angle chosen to keep the foot
    level.
    """
    y = -z
    r2 = x * x + y * y
    d = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    d = max(-1.0, min(1.0, d))
    knee_rel = math.atan2(math.sqrt(max(0.0, 1.0 - d * d)), d)
    hip_ref = math.atan2(l2 * math.sin(knee_rel), l1 + l2 * math.cos(knee_rel))
    hip = hip_ref - math.atan2(x, y)
    knee = -knee_rel
    ankle = -(hip + knee)
    return hip, knee, ankle


def smootherstep(t):
    """6t^5-15t^4+10t^3: zero velocity and acceleration at both t=0 and t=1."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def swing_clearance(tau):
    """Foot-lift profile over a swing phase tau in [0, 1]: 0 -> peak -> 0,
    built from two mirrored smootherstep halves so the peak at tau=0.5 also
    has zero vertical velocity (no foot-strike jerk)."""
    if tau <= 0.5:
        return SWING_CLEARANCE * smootherstep(tau / 0.5)
    return SWING_CLEARANCE * smootherstep((1.0 - tau) / 0.5)


class WalkingPatternGenerator(Node):
    """Drives an open-loop, cmd_vel-scaled trapezoidal walking gait."""

    def __init__(self):
        super().__init__("walking_pattern_generator")

        self._cmd_pub = self.create_publisher(
            Float64MultiArray, "/leg_position_controller/commands", 10
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

        self._forward_speed = 0.0
        self._t0 = self.get_clock().now()
        self.create_timer(1.0 / CONTROL_RATE_HZ, self._step)
        self.get_logger().info(
            "Walking pattern generator ready (open-loop, sagittal-plane only)"
        )

    def _cmd_vel_cb(self, msg):
        self._forward_speed = float(msg.linear.x)

    def _step(self):
        t = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        cycle_time = 2.0 * STEP_TIME
        phase = math.fmod(t, cycle_time) / cycle_time  # [0, 1)
        left_swinging = phase < 0.5
        tau = (phase / 0.5) if left_swinging else ((phase - 0.5) / 0.5)

        step_length = max(
            -MAX_STEP_LENGTH, min(MAX_STEP_LENGTH, STEP_GAIN * self._forward_speed)
        )
        eased = smootherstep(tau)
        swing_x = -step_length / 2.0 + step_length * eased
        stance_x = step_length / 2.0 - step_length * eased
        swing_z = -(STANCE_HEIGHT - swing_clearance(tau))
        stance_z = -STANCE_HEIGHT

        if left_swinging:
            left_x, left_z, right_x, right_z = swing_x, swing_z, stance_x, stance_z
        else:
            right_x, right_z, left_x, left_z = swing_x, swing_z, stance_x, stance_z

        l_hip, l_knee, l_ankle = leg_ik(left_x, left_z)
        r_hip, r_knee, r_ankle = leg_ik(right_x, right_z)

        targets = dict(zip(
            JOINT_ORDER, [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
        ))
        msg = Float64MultiArray()
        msg.data = [
            max(JOINT_LIMITS[name][0], min(JOINT_LIMITS[name][1], targets[name]))
            for name in JOINT_ORDER
        ]
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WalkingPatternGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
