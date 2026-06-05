#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/s/AITraining"
UNSLOTH_DIR="$ROOT/unsloth"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$UNSLOTH_DIR" "$ROOT/datasets" "$ROOT/runs" "$ROOT/adapters" "$ROOT/merged-models"
cd "$UNSLOTH_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv to ~/.local/bin..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating Unsloth venv with $PYTHON_BIN..."
  uv venv .venv --python "$PYTHON_BIN" --seed
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# RTX 50-series / Blackwell works best with CUDA 12.8 PyTorch wheels. uv's
# torch-backend flag is preferred, but fall back to explicit PyTorch index if needed.
echo "Installing/updating Unsloth stack..."
if ! uv pip install unsloth --torch-backend=cu128; then
  echo "uv torch-backend install failed; retrying explicit cu128 PyTorch then Unsloth..."
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  uv pip install unsloth
fi

cat > "$UNSLOTH_DIR/activate.sh" <<'ACTIVATE'
#!/usr/bin/env bash
source /mnt/s/AITraining/unsloth/.venv/bin/activate
cd /mnt/s/AITraining/unsloth
ACTIVATE
chmod +x "$UNSLOTH_DIR/activate.sh"

echo "DONE. Activate with: source /mnt/s/AITraining/unsloth/activate.sh"
