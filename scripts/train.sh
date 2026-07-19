#!/bin/bash
# Script to start RL training (MuJoCo-native path; see train_mujoco.py docstring
# for why this replaces the discontinued Isaac Gym path in humanoid/scripts/train.py)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "Starting RL training..."
cd "$REPO_ROOT"
python humanoid/scripts/train_mujoco.py --run_name v1 --num_envs 16 --total_timesteps 2000000 "$@"
