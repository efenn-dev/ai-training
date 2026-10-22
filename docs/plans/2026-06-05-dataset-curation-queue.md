# AI Training Dataset Curation Queue Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make AI-training useful as George's repeatable dataset curation workbench, not a one-off log pile.

**Architecture:** Keep curated data on `S:\AITraining` and process/control code in `C:\projects\ai-training-lab\unsloth`. Use explicit source registries, sanitized JSONL examples, split manifests, eval gates, and Automation Review cards before any adapter promotion. Training scripts remain CLI/source-of-truth; the control panel is only status + guarded launch.

**Tech Stack:** Python stdlib scripts, JSON/JSONL manifests, existing Unsloth scripts, Mission Control task/review records.

---

## Operating Rules

1. No raw secrets, tokens, API keys, private chat dumps, or credential material in datasets.
2. Every example must have provenance, sanitizer status, label owner/source, and intended split.
3. Holdout examples are never trained on.
4. Manual-gate/safety examples are treated as regression tests first, training data second.
5. Every adapter gets a comparison report and Automation Review card.
6. Review approval can authorize shadow-mode integration planning; live routing always requires a separate task/review.

---

## Dataset Lane Priority

### Active lane: Mission Control / Gemma route adapter

Why active: we have real sanitized shadow cases and measurable eval gates.

Current best:
- Dataset: `/mnt/s/AITraining/datasets/gemma-orchestrator-v0.2-real-shadow-reviewed`
- Adapter: `/mnt/s/AITraining/adapters/gemma-orchestrator-real-shadow-v02-20260604-234911/adapter`
- Review: `review_20260605051453_1bzyyw`
- Status: approved for guarded shadow-mode planning only.

Next data target:
- Collect 50-100 more sanitized real Mission Control route decisions.
- Emphasize disagreements, manual_gate, review_first, close_stale, and parked/blocked tasks.
- Keep deterministic safety overrides outside the model and in eval.

### Parked lane: ClipLab Edit Director LoRA

Why parked: not enough human labels yet.

Resume only when we have:
- Keep/reject labels
- Corrected start/end trims
- Reasons for highlight/reject
- Caption/edit notes
- Platform fit
- Family-safe notes

---

## Files / Artifacts to Create

### Task 1: Create curation queue directory

**Objective:** Establish a durable location for source registry and review queue artifacts.

**Files:**
- Create: `/mnt/s/AITraining/curation/README.md`
- Create: `/mnt/s/AITraining/curation/source-registry.json`
- Create: `/mnt/s/AITraining/curation/queue.jsonl`
- Create: `/mnt/s/AITraining/curation/label-schema.json`

**Verification:**

Run:

```bash
python3 -m json.tool /mnt/s/AITraining/curation/source-registry.json >/dev/null
python3 -m json.tool /mnt/s/AITraining/curation/label-schema.json >/dev/null
python3 - <<'PY'
from pathlib import Path
for p in ['README.md','source-registry.json','queue.jsonl','label-schema.json']:
    assert (Path('/mnt/s/AITraining/curation') / p).exists(), p
print('curation scaffold ok')
PY
```

### Task 2: Add curation helper script

**Objective:** Provide a repeatable audit/check command before training.

**Files:**
- Create: `scripts/curation_check.py`

**Behavior:**
- Validate JSON syntax.
- Count queue items by status/lane/split.
- Refuse obvious secret-like strings in `input`, `notes`, and `label` fields.
- Verify required keys: `id`, `lane`, `source`, `status`, `sanitized`, `split`, `input`, `label`.

**Verification:**

Run:

```bash
python3 scripts/curation_check.py --queue /mnt/s/AITraining/curation/queue.jsonl
```

Expected: prints counts and exits 0 for empty/valid queue.

### Task 3: Add Mission Control route curation intake

**Objective:** Convert real shadow-routing outcomes into queued candidate examples.

**Files:**
- Create: `scripts/curate_mission_control_routes.py`

**Behavior:**
- Reads sanitized source JSONL or existing route shadow evidence.
- Emits queued examples to `/mnt/s/AITraining/curation/queue.jsonl`.
- Does not include raw task descriptions if they contain secrets/credentials/payments/legal/private data; instead writes sanitized summaries.

**Verification:**
- Dry-run mode prints candidate counts.
- No write unless `--write` is passed.

### Task 4: Add split promotion command

**Objective:** Move reviewed queue items into a dataset version.

**Files:**
- Create: `scripts/promote_curation_to_dataset.py`

**Behavior:**
- Selects `status=reviewed` items for a target lane.
- Writes `train.jsonl`, `eval.jsonl`, and `holdout_real_shadow.jsonl` under a new versioned dataset dir.
- Refuses to overwrite an existing dataset without `--force`.
- Writes a `MANIFEST.json` with counts and source IDs.

**Verification:**
- Dry-run shows counts by split.
- Manifest paths exist after write.

### Task 5: Improve control-panel summary

**Objective:** Make the UI tell the operator the state without log spelunking.

**Files:**
- Modify: `control_panel/server.py`
- Modify: `control_panel/static/index.html`
- Modify: `control_panel/static/app.js`
- Modify: `control_panel/static/style.css`

**Behavior:**
- `/api/status` includes `training_state`:
  - `state`: `idle | verifying | training | setup | failed | unknown`
  - `active_job_id`
  - `last_training_adapter`
  - `last_training_run`
  - `latest_eval_gate_summary`
  - `next_recommended_action`
- UI renders a top banner:
  - “No training running”
  - “Verify running”
  - “Training running: <run>”
  - “Last completed adapter: <path>”

**Verification:**

Run:

```bash
python3 -m py_compile control_panel/server.py
python3 - <<'PY'
from control_panel.server import status_payload
s = status_payload()
assert 'training_state' in s
print(s['training_state'])
PY
```

### Task 6: Create review artifact after each dataset/adaptor change

**Objective:** Preserve operator decision gates.

**Files:**
- Update scripts as needed to write comparison/report markdown.

**Behavior:**
- Every adapter has:
  - dataset path
  - adapter path
  - train log
  - baseline metrics
  - adapter metrics
  - gate pass/fail
  - recommendation
- Mission Control review is created for human/operator decision.

---

## Done Definition

- Curation scaffold exists on `/mnt/s/AITraining/curation`.
- Helper checks pass.
- Active route-adapter lane has an explicit next data target.
- ClipLab lane remains parked with clear label requirements.
- Control panel summary task is ready/assigned.
- No live model routing is enabled by this plan.
