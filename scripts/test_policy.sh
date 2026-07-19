#!/bin/bash
# Script to sim-to-sim validate the most recently exported policy in MuJoCo
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

LATEST_POLICY=$(ls -t "$REPO_ROOT"/logs/*/exported/policies/policy_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_POLICY" ]; then
    echo "No exported policy found under logs/*/exported/policies/"
    exit 1
fi

echo "Validating policy: $LATEST_POLICY"
cd "$REPO_ROOT"
python humanoid/scripts/sim2sim.py --load_model "$LATEST_POLICY" "$@"
