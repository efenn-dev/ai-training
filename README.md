# AI Training / Unsloth Workspace

Purpose: isolated local fine-tuning workspace for Unsloth experiments on Emmitt's RTX 5070 Ti from WSL2.

## Layout

Code/control files live here:

- `C:\projects\ai-training-lab\unsloth`
- WSL: `/mnt/c/projects/ai-training-lab/unsloth`

Hot training data and environments live on the fast S: SSD:

- `S:\AITraining\unsloth` / `/mnt/s/AITraining/unsloth` — Unsloth virtualenv and scripts
- `S:\AITraining\datasets` / `/mnt/s/AITraining/datasets` — JSONL datasets
- `S:\AITraining\runs` / `/mnt/s/AITraining/runs` — training outputs/logs
- `S:\AITraining\adapters` / `/mnt/s/AITraining/adapters` — LoRA adapters
- `S:\AITraining\merged-models` / `/mnt/s/AITraining/merged-models` — merged/exported models
- `S:\AITraining\traces` / `/mnt/s/AITraining/traces` — agent trace capture, imports, source registry, and training-data prep artifacts

## Agent Trace Lab

The dashboard includes a Trace Lab tab for Fable-like agent training data:

- automatic sidecar trace capture for Agent Arena candidate runs under `S:\AITraining\traces\raw`
- Fable Phase 1 import from `/mnt/c/projects/fable-trace-lora` into `S:\AITraining\traces\imports\fable-claude-code`
- recommended public trace source registry at `S:\AITraining\traces\registry\recommended_sources.json`
- status/metrics surfaced through `/api/traces/summary` and the Trace Lab UI

Useful commands:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/trace/import_fable_phase1.py
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/trace/write_recommended_sources.py
```

## Environment

The setup uses `uv` plus a dedicated Python 3.12 venv:

```bash
/mnt/c/projects/ai-training-lab/unsloth/scripts/setup_unsloth.sh
/mnt/c/projects/ai-training-lab/unsloth/scripts/verify_unsloth.sh
```

Activation:

```bash
source /mnt/s/AITraining/unsloth/.venv/bin/activate
cd /mnt/s/AITraining/unsloth
```

## Dashboard launcher

Mission Control app name: `ai-training`

Commands:

- `ui` — start the local Unsloth control panel on port `5196` and open it in the browser
- `setup` — run/repair the Unsloth environment install
- `verify` — check GPU, torch, and Unsloth imports
- `shell` — open a Windows Terminal tab into WSL with the env activated

On Windows, the `ui` command should use `scripts\launch_control_panel.ps1`. It starts the WSL server, repairs an old loopback-only listener if needed, and opens the reachable WSL VM URL automatically.

The control panel is now arena-first and uses existing CLI scripts as the source of truth for setup/training. It provides:

- an Agent Arena for self-improvement tasks, candidate patch capture, deterministic scoring, manual winner selection, and preference JSONL export under `S:\AITraining\agent-arena`
- a guarded Candidate Batch Runner that creates isolated git worktrees, runs scripted-smoke/Codex/Claude workers, captures diffs/test logs, and imports candidates back into the Arena
- GPU/environment status
- guarded setup/verify buttons
- smoke-training launch
- job logs
- a read-only artifact browser for `S:\AITraining`

Manual launch from WSL:

```bash
cd /mnt/c/projects/ai-training-lab/unsloth
./scripts/run_control_panel.sh
```

If Windows cannot reach `127.0.0.1:5196`, open the WSL IP printed by the server, or run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/launch_control_panel.ps1` from `C:\projects\ai-training-lab\unsloth`.

## First experiments

Start small. Recommended first LoRA adapter:

- base model: Qwen2.5/Qwen3 1.5B-3B or Llama 3.2 3B, depending current Unsloth support
- dataset: clean chat-format JSONL under `S:\AITraining\datasets`
- goal: George tone/style adapter or ClipLab edit-decision advisor

Do not train on secrets, credentials, raw private chat logs, or stale task artifacts.

## ClipLab Edit Director LoRA

V0.1 trains on ClipLab AI Review accept/reject feedback for edit-director decisions. It does not train on raw video, and unlabeled local artifacts are parked in `unlabeled_candidates.jsonl` until a human labels them.

Dataset curation:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/curate_cliplab_edit_director_v01.py
```

Local smoke train after labels exist:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/train_cliplab_edit_director.py \
  --smoke-model \
  --max-train-rows 20 \
  --max-steps 20
```

Compare an adapter against the current pipeline baseline:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/compare_cliplab_edit_director.py \
  --adapter-dir /mnt/s/AITraining/adapters/<run-id>/adapter
```

Training now runs the comparison gate by default after saving an adapter. The gate writes `comparison-report.md`/`.json`, stages a Mission Control Automation Review artifact, and only recommends `shadow_mode_candidate` or `do_not_promote`. Review approval authorizes shadow-mode only; live production routing/editing requires a separate integration task. Use `--skip-eval-gate` only for environment/debug smoke work where no adapter promotion is possible.

For the Mission Control general-ops router lane, use:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/compare_gemma_orchestrator_adapter.py \
  --adapter-dir /mnt/s/AITraining/adapters/<run-id>/adapter
```

That report includes strict schema validity, route/task accuracy, manual-gate misses, fake-verification/hallucination checks, and the same Automation Review shadow-mode-only gate.

Export a promoted adapter as a merged GGUF:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/export_cliplab_edit_director_gguf.py \
  --adapter-dir /mnt/s/AITraining/adapters/<run-id>/adapter
```

## Compact frontier review packets

Use `scripts/generate_frontier_review_packet.py` when a frontier reviewer needs a small packet instead of raw logs/transcripts. It supports `gaming-pipeline` and `general-ops-router` lanes and writes paired Markdown/JSON artifacts with queue counts, changed examples, low-confidence drafts, disagreements, errors, eval metrics, and raw artifact pointers.

See `docs/frontier-review-packets.md` for usage and the checklist for when frontier review is actually required.

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Emmitt Fennessey.
