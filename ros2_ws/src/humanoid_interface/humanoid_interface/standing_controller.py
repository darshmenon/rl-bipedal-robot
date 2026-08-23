"""Pelvis-tilt feedback standing stabilizer for H1."""

from __future__ import annotations

import math

from humanoid_interface.joint_names import STANDING_POSE

SAGITTAL_KP = 1.2
SAGITTAL_KD = 0.35
MAX_SAGITTAL_CORRECTION = 0.35
LATERAL_KP = 0.35
LATERAL_KD = 0.08
MAX_LATERAL_CORRECTION = 0.08


def roll_pitch_from_quat(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def clamp(value, limit):
    return max(-limit, min(limit, value))


class StandingController:
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.have_odom = False

    def update_odom(self, quat, omega):
        self.roll, self.pitch = roll_pitch_from_quat(*quat)
        self.roll_rate = omega[0]
        self.pitch_rate = omega[1]
        self.have_odom = True

    def compute_targets(self) -> dict[str, float]:
        targets = dict(STANDING_POSE)
        if not self.have_odom:
            return targets

        sagittal = clamp(
            -(SAGITTAL_KP * self.pitch + SAGITTAL_KD * self.pitch_rate),
            MAX_SAGITTAL_CORRECTION,
        )
        lateral = clamp(
            LATERAL_KP * self.roll + LATERAL_KD * self.roll_rate,
            MAX_LATERAL_CORRECTION,
        )

        targets["left_hip_pitch_joint"] += sagittal
        targets["right_hip_pitch_joint"] += sagittal
        targets["left_ankle_joint"] -= sagittal
        targets["right_ankle_joint"] -= sagittal
        targets["left_hip_roll_joint"] -= lateral
        targets["right_hip_roll_joint"] -= lateral
        return targets
