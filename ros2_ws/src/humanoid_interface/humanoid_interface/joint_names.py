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

ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

ALL_JOINTS = LEG_JOINTS + ["torso_joint"] + ARM_JOINTS

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

BEHAVIOR_CMDS = {
    "stand": [0.0, 0.0, 0.0],
    "stop": [0.0, 0.0, 0.0],
    "walk_forward": [0.5, 0.0, 0.0],
    "walk_backward": [-0.3, 0.0, 0.0],
    "turn_left": [0.2, 0.0, 0.5],
    "turn_right": [0.2, 0.0, -0.5],
}

VALID_BEHAVIORS = list(BEHAVIOR_CMDS) + ["wave_hand"]
