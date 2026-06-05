#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/projects/ai-training
export AI_TRAINING_UI_HOST="${AI_TRAINING_UI_HOST:-127.0.0.1}"
export AI_TRAINING_UI_PORT="${AI_TRAINING_UI_PORT:-5196}"
exec python3 -m control_panel.server
