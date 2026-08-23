"""Scripted right-arm wave trajectory for H1."""

from __future__ import annotations

import math

from humanoid_interface.joint_names import STANDING_POSE


class WaveController:
    """Simple sinusoidal shoulder/elbow wave while legs hold standing pose."""

    PERIOD_S = 1.2
    DURATION_S = 4.0

    def __init__(self):
        self.elapsed = 0.0
        self.active = False

    def start(self):
        self.elapsed = 0.0
        self.active = True

    def stop(self):
        self.active = False
        self.elapsed = 0.0

    def step(self, dt: float) -> tuple[dict[str, float], float, bool]:
        targets = dict(STANDING_POSE)
        if not self.active:
            return targets, 0.0, True

        self.elapsed += dt
        progress = min(1.0, self.elapsed / self.DURATION_S)
        phase = (self.elapsed % self.PERIOD_S) / self.PERIOD_S
        wave = 0.5 * (1.0 - math.cos(2 * math.pi * phase))

        targets["right_shoulder_pitch_joint"] = -0.8 * wave
        targets["right_shoulder_roll_joint"] = -0.3
        targets["right_elbow_joint"] = 1.2 * wave

        done = self.elapsed >= self.DURATION_S
        if done:
            self.active = False
        return targets, progress, done
