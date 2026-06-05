#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=${HF_HOME:-/mnt/s/AITraining/hf-cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/s/AITraining/hf-cache/transformers}
source /mnt/s/AITraining/unsloth/.venv/bin/activate
cd /mnt/s/AITraining/unsloth
export RUN_ID="${RUN_ID:-gemma-orchestrator-smoke-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="/mnt/s/AITraining/runs/${RUN_ID}"
export OUT_DIR="${OUT_DIR:-/mnt/s/AITraining/adapters/${RUN_ID}}"
mkdir -p "$RUN_DIR" "$OUT_DIR"
python /mnt/c/projects/ai-training/scripts/train_gemma_orchestrator_smoke.py 2>&1 | tee "$RUN_DIR/train.log"
