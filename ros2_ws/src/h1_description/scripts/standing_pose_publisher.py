#!/usr/bin/env python3
"""Holds Unitree H1 upright via the native gz-sim JointPositionController
plugins (see h1.urdf), so a 3D lidar mounted on it can be used to exercise
SLAM without a real walking controller yet.

Publishes position targets to /model/h1/joint/<name>/cmd_pos (see h1.urdf's
per-joint <topic> override) for all 19 actuated joints. The neutral target is
a static stance; the leg targets are biased by a pelvis-tilt feedback term
driven by /model/h1/odometry (h1.urdf's OdometryPublisher plugin, bridged to
ROS in h1_gazebo.launch.py), so this isn't pure open-loop pose-holding.

An earlier version of this feedback term read ground truth by shelling out
to `gz topic -e -n 1` inside the publish timer -- subprocess spawn overhead
alone could exceed its own 20ms timeout, so it fed the controller stale or
missing attitude most cycles. This version subscribes to the bridged
Odometry topic directly, which also gives real angular velocity (twist) for
the derivative term instead of finite-differencing noisy polled samples.
"""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64

# This is still a deliberately simple sim-only stabilizer, not a whole-body
# controller: it's per-joint position holds biased by pelvis tilt, not
# estimated CoM/ZMP/contact state. p_gain/d_gain/cmd_max (see h1.urdf's
# JointPositionController overrides) match Unitree G1's public real-hardware
# standing-controller values, H1's closest published cousin.
#
# In this gz-sim URDF/sign convention, the slight-squat pose used by some H1
# RL configs (-0.2 hip, +0.4 knee, -0.2 ankle) drives this model into a fast
# forward fall. hip_pitch=+0.05 / ankle=-0.05 was the best open-loop neutral
# found during earlier hand tuning, so the feedback loop below stabilizes
# around that pose instead of fighting a bad nominal target.
STANDING_POSE = {
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": 0.05,
    "left_knee_joint": 0.0,
    "left_ankle_joint": -0.05,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.05,
    "right_knee_joint": 0.0,
    "right_ankle_joint": -0.05,
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

PUBLISH_RATE_HZ = 100.0

SAGITTAL_KP = 1.2
SAGITTAL_KD = 0.35
MAX_SAGITTAL_CORRECTION = 0.35

LATERAL_KP = 0.35
LATERAL_KD = 0.08
MAX_LATERAL_CORRECTION = 0.08


def _roll_pitch_from_quat(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return roll, pitch


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class StandingPosePublisher(Node):
    def __init__(self):
        super().__init__("standing_pose_publisher")
        self._pubs = {
            name: self.create_publisher(
                Float64, f"/model/h1/joint/{name}/cmd_pos", 10
            )
            for name in STANDING_POSE
        }
        self._roll = 0.0
        self._pitch = 0.0
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._have_odom = False

        self.create_subscription(Odometry, "/model/h1/odometry", self._odom_cb, 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish)
        self._last_feedback_log_time = -10.0
        self.get_logger().info(
            f"Standing stabilizer ready ({len(STANDING_POSE)} joints, "
            f"{PUBLISH_RATE_HZ:.0f} Hz)"
        )

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self._roll, self._pitch = _roll_pitch_from_quat(q.x, q.y, q.z, q.w)
        self._roll_rate = msg.twist.twist.angular.x
        self._pitch_rate = msg.twist.twist.angular.y
        self._have_odom = True

    def _publish(self):
        targets = dict(STANDING_POSE)

        if self._have_odom:
            sagittal = _clamp(
                -(SAGITTAL_KP * self._pitch + SAGITTAL_KD * self._pitch_rate),
                MAX_SAGITTAL_CORRECTION,
            )
            lateral = _clamp(
                LATERAL_KP * self._roll + LATERAL_KD * self._roll_rate,
                MAX_LATERAL_CORRECTION,
            )

            # Pitch correction keeps each foot approximately level while
            # moving the support reaction in the direction of the fall.
            targets["left_hip_pitch_joint"] += sagittal
            targets["right_hip_pitch_joint"] += sagittal
            targets["left_ankle_joint"] -= sagittal
            targets["right_ankle_joint"] -= sagittal

            # H1's simplified ankle joint here is pitch-only, so lateral
            # correction has to come from symmetric hip-roll bias.
            targets["left_hip_roll_joint"] -= lateral
            targets["right_hip_roll_joint"] -= lateral

            t = self.get_clock().now().nanoseconds * 1e-9
            if t - self._last_feedback_log_time > 2.0:
                self.get_logger().info(
                    "feedback roll={:.3f} pitch={:.3f} lateral={:.3f} "
                    "sagittal={:.3f}".format(
                        self._roll, self._pitch, lateral, sagittal
                    )
                )
                self._last_feedback_log_time = t

        for name, target in targets.items():
            msg = Float64()
            msg.data = target
            self._pubs[name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StandingPosePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
