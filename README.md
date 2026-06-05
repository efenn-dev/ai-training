# AI Training / Unsloth Workspace

Purpose: isolated local fine-tuning workspace for Unsloth experiments on Emmitt's RTX 5070 Ti from WSL2.

## Layout

Code/control files live here:

- `C:\projects\ai-training`
- WSL: `/mnt/c/projects/ai-training`

Hot training data and environments live on the fast S: SSD:

- `S:\AITraining\unsloth` / `/mnt/s/AITraining/unsloth` — Unsloth virtualenv and scripts
- `S:\AITraining\datasets` / `/mnt/s/AITraining/datasets` — JSONL datasets
- `S:\AITraining\runs` / `/mnt/s/AITraining/runs` — training outputs/logs
- `S:\AITraining\adapters` / `/mnt/s/AITraining/adapters` — LoRA adapters
- `S:\AITraining\merged-models` / `/mnt/s/AITraining/merged-models` — merged/exported models

## Environment

The setup uses `uv` plus a dedicated Python 3.12 venv:

```bash
/mnt/c/projects/ai-training/scripts/setup_unsloth.sh
/mnt/c/projects/ai-training/scripts/verify_unsloth.sh
```

Activation:

```bash
source /mnt/s/AITraining/unsloth/.venv/bin/activate
cd /mnt/s/AITraining/unsloth
```

## Dashboard launcher

Mission Control app name: `ai-training`

Commands:

- `setup` — run/repair the Unsloth environment install
- `verify` — check GPU, torch, and Unsloth imports
- `shell` — open a Windows Terminal tab into WSL with the env activated

## First experiments

Start small. Recommended first LoRA adapter:

- base model: Qwen2.5/Qwen3 1.5B-3B or Llama 3.2 3B, depending current Unsloth support
- dataset: clean chat-format JSONL under `S:\AITraining\datasets`
- goal: George tone/style adapter or ClipLab edit-decision advisor

Do not train on secrets, credentials, raw private chat logs, or stale task artifacts.
