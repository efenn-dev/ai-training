#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=${HF_HOME:-/mnt/s/AITraining/hf-cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/s/AITraining/hf-cache/transformers}
source /mnt/s/AITraining/unsloth/.venv/bin/activate
cd /mnt/s/AITraining/unsloth
RUN_ID="vlm-clip-router-smoke-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="/mnt/s/AITraining/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
python /mnt/c/projects/ai-training-lab/unsloth/scripts/train_vlm_clip_router_smoke.py 2>&1 | tee "$RUN_DIR/train.log"
