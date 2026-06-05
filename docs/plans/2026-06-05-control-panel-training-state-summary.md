# Unsloth Control Panel Training-State Summary Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make the Unsloth control panel answer “what is happening?” without requiring Caleb to read logs.

**Architecture:** Add a computed `training_state` object to the existing stdlib backend `/api/status`, derived from job metadata, logs, known run directories, comparison reports, and adapter artifacts. Render it as a top-level UI banner with next recommended action.

**Tech Stack:** Python stdlib HTTP backend, vanilla HTML/CSS/JS.

---

## Desired Operator Copy

The UI should show one plain sentence:

- `No training running. Last adapter passed eval: <adapter path>`
- `Verify running. Training is not running.`
- `Training running: <run id>. GPU active.`
- `Last training failed. Open log: <path>`
- `Setup/repair running. Do not start training yet.`

---

## Backend Task

**Files:**
- Modify: `control_panel/server.py`

**Add function:** `training_state_payload()`

**Derive:**
- active job from `_jobs` where `status == "running"`
- verify/setup/training from `job.command`
- last adapter from newest `/mnt/s/AITraining/adapters/*/adapter` directory
- latest comparison report from newest `/mnt/s/AITraining/runs/**/comparison-report.json`
- latest training log from newest `/mnt/s/AITraining/runs/**/*train*.log`

**Return shape:**

```json
{
  "state": "idle|verifying|training|setup|failed|unknown",
  "summary": "No training running...",
  "active_job_id": null,
  "active_command": null,
  "last_training_run": "/mnt/s/...",
  "last_training_log": "/mnt/s/...",
  "last_training_adapter": "/mnt/s/.../adapter",
  "latest_eval": {
    "path": "/mnt/s/.../comparison-report.json",
    "promotion_ready": true,
    "gates": {"strict_route_accuracy": true}
  },
  "next_recommended_action": "Approve shadow-mode task or collect more labels."
}
```

**Verification:**

```bash
python3 -m py_compile control_panel/server.py
python3 - <<'PY'
from control_panel.server import status_payload
s = status_payload()
assert 'training_state' in s
assert 'summary' in s['training_state']
print(s['training_state'])
PY
```

---

## UI Task

**Files:**
- Modify: `control_panel/static/index.html`
- Modify: `control_panel/static/app.js`
- Modify: `control_panel/static/style.css`

**Add:**
- One large card/banner directly under the header.
- Fields:
  - State
  - Summary
  - Last adapter
  - Latest eval status
  - Next action

**Verification:**

Start server and open:

```bash
./scripts/run_control_panel.sh
```

Then verify:

```bash
python3 - <<'PY'
import json, urllib.request
s=json.load(urllib.request.urlopen('http://127.0.0.1:5196/api/status', timeout=5))
assert 'training_state' in s
print(s['training_state']['summary'])
PY
```

Use browser console/snapshot if available to confirm the summary renders.

---

## Done Definition

- Caleb can tell whether training is running without reading logs.
- The UI names the latest successful adapter and eval status.
- Verify jobs are clearly distinguished from training jobs.
- No new arbitrary file/shell controls are added.
