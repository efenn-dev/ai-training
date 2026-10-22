#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/projects/ai-training-lab/unsloth
export AI_TRAINING_UI_HOST="${AI_TRAINING_UI_HOST:-0.0.0.0}"
export AI_TRAINING_UI_PORT="${AI_TRAINING_UI_PORT:-5196}"
export AI_TRAINING_ALLOW_WSL_HOST_BIND="${AI_TRAINING_ALLOW_WSL_HOST_BIND:-1}"
exec python3 -m control_panel.server
