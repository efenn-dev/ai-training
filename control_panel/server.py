#!/usr/bin/env python3
"""Local Unsloth control panel.

A deliberately thin stdlib HTTP server over the existing ai-training CLI scripts.
It binds localhost only and exposes only allowlisted actions/artifact roots.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path("/mnt/c/projects/ai-training-lab/unsloth").resolve()
S_ROOT = Path("/mnt/s/AITraining")
RUNS_ROOT = S_ROOT / "runs"
CONTROL_ROOT = RUNS_ROOT / "control-panel"
ARENA_ROOT = S_ROOT / "agent-arena"
WORKTREES_ROOT = ARENA_ROOT / "worktrees"
TRACE_ROOT = S_ROOT / "traces"
STATIC_ROOT = REPO_ROOT / "control_panel" / "static"
JOBS_JSON = CONTROL_ROOT / "jobs.json"
LOCAL_FIRST_TELEMETRY_JSONL = TRACE_ROOT / "metrics" / "local-first-telemetry.jsonl"
LOCAL_FIRST_SUMMARY_JSON = TRACE_ROOT / "metrics" / "local-first-summary.json"
HOST = os.environ.get("AI_TRAINING_UI_HOST", "127.0.0.1")
PORT = int(os.environ.get("AI_TRAINING_UI_PORT", "5196"))
MAX_LOG_BYTES = 500_000
MAX_PREVIEW_BYTES = 300_000
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".md", ".yaml", ".yml", ".csv", ".py", ".sh"}

ARTIFACT_ROOTS = {
    "datasets": S_ROOT / "datasets",
    "runs": RUNS_ROOT,
    "adapters": S_ROOT / "adapters",
    "merged-models": S_ROOT / "merged-models",
    "agent-arena": ARENA_ROOT,
    "traces": TRACE_ROOT,
}

COMMANDS: dict[str, dict[str, Any]] = {
    "verify": {
        "label": "Run Verify",
        "description": "Check GPU, torch/CUDA, Unsloth import, and HF stack imports.",
        "argv": [str(REPO_ROOT / "scripts" / "verify_unsloth.sh")],
        "risk": "low",
        "timeout_seconds": 900,
        "env_keys": {"VERIFY_TIMEOUT_SECONDS"},
        "log_prefix": "verify",
    },
    "setup": {
        "label": "Run Setup / Repair",
        "description": "Install/update the Unsloth environment with uv/pip. This may download packages.",
        "argv": [str(REPO_ROOT / "scripts" / "setup_unsloth.sh")],
        "risk": "high",
        "timeout_seconds": 7200,
        "env_keys": set(),
        "log_prefix": "setup",
    },
    "smoke-train": {
        "label": "Gemma Orchestrator Smoke Train",
        "description": "Launch the existing small smoke-training script with an allowlisted RUN_ID.",
        "argv": [str(REPO_ROOT / "scripts" / "train_gemma_orchestrator_smoke.sh")],
        "risk": "medium",
        "timeout_seconds": 7200,
        "env_keys": {"RUN_ID"},
        "log_prefix": "smoke-train",
    },
    "local-first-smoke": {
        "label": "Local-first Router Smoke",
        "description": "Run safe local-first router smoke cases and generate frontier packets for failures.",
        "argv": [str(REPO_ROOT / "scripts" / "local_first_runner.py"), "--smoke", "--log-shadow", "--timeout", "45", "--max-tokens", "300"],
        "risk": "low",
        "timeout_seconds": 420,
        "env_keys": set(),
        "log_prefix": "local-first-smoke",
    },
    "trace-import-fable": {
        "label": "Import Fable Phase 1 Traces",
        "description": "Copy processed Fable trace artifacts into the AI Training trace store with a manifest.",
        "argv": [str(REPO_ROOT / "scripts" / "trace" / "import_fable_phase1.py")],
        "risk": "low",
        "timeout_seconds": 900,
        "env_keys": set(),
        "log_prefix": "trace-import-fable",
    },
    "trace-import-autoresearch": {
        "label": "Download AutoResearch Repo",
        "description": "Clone the Karpathy AutoResearch repository into the trace store for review and downstream integration.",
        "argv": [str(REPO_ROOT / "scripts" / "trace" / "import_autoresearch.py")],
        "risk": "low",
        "timeout_seconds": 1200,
        "env_keys": set(),
        "log_prefix": "trace-import-autoresearch",
    },
    "trace-normalize-autoresearch": {
        "label": "Normalize AutoResearch Traces",
        "description": "Sanitize the latest AutoResearch import into the training-sanitized trace lane.",
        "argv": [str(REPO_ROOT / "scripts" / "trace" / "normalize_autoresearch.py")],
        "risk": "low",
        "timeout_seconds": 1200,
        "env_keys": set(),
        "log_prefix": "trace-normalize-autoresearch",
    },
    "trace-write-sources": {
        "label": "Write Recommended Trace Sources",
        "description": "Write the current recommended public trace source registry for review and later normalization.",
        "argv": [str(REPO_ROOT / "scripts" / "trace" / "write_recommended_sources.py")],
        "risk": "low",
        "timeout_seconds": 120,
        "env_keys": set(),
        "log_prefix": "trace-write-sources",
    },
}

@dataclass
class Job:
    id: str
    command: str
    label: str
    status: str
    argv: list[str]
    pid: int | None
    started_at: str
    ended_at: str | None
    returncode: int | None
    log_path: str
    risk: str
    env: dict[str, str]
    error: str | None = None

_live: dict[str, subprocess.Popen[str]] = {}
_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    for root in ARTIFACT_ROOTS.values():
        root.mkdir(parents=True, exist_ok=True)
    for child in ["tasks", "candidates", "reviews", "winners", "datasets", "evals", "worktrees"]:
        (ARENA_ROOT / child).mkdir(parents=True, exist_ok=True)
    for child in ["raw", "imports", "sanitized", "normalized", "metrics", "exports", "registry"]:
        (TRACE_ROOT / child).mkdir(parents=True, exist_ok=True)


def load_jobs() -> None:
    ensure_dirs()
    if not JOBS_JSON.exists():
        return
    try:
        data = json.loads(JOBS_JSON.read_text(encoding="utf-8"))
        for item in data[-100:]:
            job = Job(**item)
            if job.status == "running":
                job.status = "unknown"
                job.ended_at = job.ended_at or now_iso()
                job.error = job.error or "Server restarted before this job finished; inspect log/artifacts."
            _jobs[job.id] = job
    except Exception:
        # Keep server usable even if metadata is corrupt.
        pass


def save_jobs() -> None:
    ensure_dirs()
    data = [asdict(job) for job in sorted(_jobs.values(), key=lambda j: j.started_at)[-100:]]
    tmp = JOBS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(JOBS_JSON)


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
    body = text.encode("utf-8", errors="replace")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    json_response(handler, {"ok": False, "error": message}, status)


def read_json(path: Path) -> Any | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc), "path": str(path)}
    return None


def parse_gpu_csv(stdout: str) -> list[dict[str, str]]:
    keys = ["name", "memory_total_mb", "memory_used_mb", "memory_free_mb", "driver", "temperature_c", "utilization_percent"]
    rows = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(keys):
            rows.append(dict(zip(keys, parts[: len(keys)])))
    return rows


def gpu_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "gpus": parse_gpu_csv(proc.stdout),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "gpus": []}


def disk_status(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return {
            "ok": True,
            "total_gb": round(usage.total / 1024**3, 2),
            "used_gb": round(usage.used / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def newest_path(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def newest_rglob(root: Path, pattern: str) -> Path | None:
    try:
        if not root.exists():
            return None
        return newest_path([path for path in root.rglob(pattern) if path.is_file()])
    except OSError:
        return None


def newest_adapter_dir() -> Path | None:
    root = ARTIFACT_ROOTS["adapters"]
    try:
        if not root.exists():
            return None
        candidates = [path for path in root.glob("*/adapter") if path.is_dir()]
        return newest_path(candidates)
    except OSError:
        return None


def latest_eval_summary(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        return {"path": str(path), "error": "comparison report is not a JSON object"}
    gates = data.get("promotion_gates")
    gate_summary: dict[str, Any] = {}
    if isinstance(gates, dict):
        for name, payload in gates.items():
            if isinstance(payload, dict):
                gate_summary[name] = {
                    "passed": payload.get("passed"),
                    "actual": payload.get("actual"),
                    "threshold": payload.get("threshold"),
                }
    adapter_metrics = data.get("adapter", {}).get("metrics", {}) if isinstance(data.get("adapter"), dict) else {}
    return {
        "path": str(path),
        "promotion_ready": data.get("promotion_ready"),
        "gates": gate_summary,
        "adapter_metrics": adapter_metrics if isinstance(adapter_metrics, dict) else {},
    }


def training_state_payload() -> dict[str, Any]:
    with _lock:
        jobs_sorted = sorted(_jobs.values(), key=lambda job: job.started_at, reverse=True)
        active = next((job for job in jobs_sorted if job.status == "running"), None)
        latest_job = jobs_sorted[0] if jobs_sorted else None

    adapter = newest_adapter_dir()
    latest_train_log = newest_rglob(RUNS_ROOT, "*train*.log")
    latest_eval_path = newest_rglob(RUNS_ROOT, "comparison-report.json")
    latest_eval = latest_eval_summary(latest_eval_path)

    base: dict[str, Any] = {
        "state": "idle",
        "summary": "No training running.",
        "active_job_id": None,
        "active_command": None,
        "last_training_run": str(latest_train_log.parent) if latest_train_log else None,
        "last_training_log": str(latest_train_log) if latest_train_log else None,
        "last_training_adapter": str(adapter) if adapter else None,
        "latest_eval": latest_eval,
        "next_recommended_action": "Collect/review more curated examples before starting another adapter.",
    }

    if active:
        base["active_job_id"] = active.id
        base["active_command"] = active.command
        if active.command == "verify":
            base["state"] = "verifying"
            base["summary"] = "Verify running. Training is not running."
            base["next_recommended_action"] = "Wait for verify to finish; only train if CUDA, Unsloth, and HF imports pass."
        elif active.command == "setup":
            base["state"] = "setup"
            base["summary"] = "Setup/repair running. Do not start training yet."
            base["next_recommended_action"] = "Wait for setup to finish, then run verify."
        elif active.command == "smoke-train":
            run_id = active.env.get("RUN_ID") or active.id
            base["state"] = "training"
            base["summary"] = f"Training running: {run_id}. GPU may be active."
            base["next_recommended_action"] = "Let the run finish, then review its adapter/eval artifact before promotion."
        else:
            base["state"] = "unknown"
            base["summary"] = f"Job running: {active.command}. Inspect logs if this was unexpected."
        return base

    if latest_job and latest_job.command == "smoke-train" and latest_job.status == "failed":
        base["state"] = "failed"
        base["summary"] = f"Last training failed. Open log: {latest_job.log_path}"
        base["next_recommended_action"] = "Inspect the failed training log before launching another run."
        return base

    if latest_eval and latest_eval.get("promotion_ready") is True and adapter:
        base["summary"] = f"No training running. Last adapter passed eval: {adapter}"
        base["next_recommended_action"] = "Proceed with guarded shadow-mode integration planning, not live routing."
    elif adapter:
        base["summary"] = f"No training running. Latest adapter found: {adapter}"
    return base


def status_payload() -> dict[str, Any]:
    latest_verify = read_json(RUNS_ROOT / "verify-latest.json")
    roots = {name: {"path": str(path), "exists": path.exists()} for name, path in ARTIFACT_ROOTS.items()}
    venv_python = S_ROOT / "unsloth" / ".venv" / "bin" / "python"
    with _lock:
        jobs = [asdict(job) for job in sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)[:10]]
    return {
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "s_root": str(S_ROOT),
        "venv_python": str(venv_python),
        "venv_exists": venv_python.exists(),
        "roots": roots,
        "disk": disk_status(S_ROOT),
        "gpu": gpu_status(),
        "latest_verify": latest_verify,
        "training_state": training_state_payload(),
        "trace_lab": trace_lab_payload(),
        "jobs": jobs,
        "commands": public_commands(),
        "time": now_iso(),
    }


def trace_common_module():
    trace_scripts = REPO_ROOT / "scripts" / "trace"
    if str(trace_scripts) not in sys.path:
        sys.path.insert(0, str(trace_scripts))
    import trace_common  # type: ignore[import-not-found]

    return trace_common


def trace_lab_payload() -> dict[str, Any]:
    try:
        module = trace_common_module()
        payload = module.summarize_trace_store()
        payload["local_first"] = local_first_summary()
        return payload
    except Exception as exc:  # noqa: BLE001 - status payload should stay usable
        return {"root": str(TRACE_ROOT), "error": repr(exc), "updated_at": now_iso()}


def iter_jsonl(path: Path, limit: int = 5000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows[-limit:]


def safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def local_first_summary() -> dict[str, Any]:
    ensure_dirs()
    rows = iter_jsonl(LOCAL_FIRST_TELEMETRY_JSONL)
    split = {"local": 0, "frontier": 0, "human": 0, "unknown": 0}
    audit = {"pass": 0, "fail": 0, "not_run": 0, "unknown": 0}
    avoided = 0
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    recent: list[dict[str, Any]] = []
    for row in rows:
        worker_class = str(row.get("worker_class") or "unknown")
        split[worker_class if worker_class in split else "unknown"] += 1
        if row.get("frontier_call_avoided") is True:
            avoided += 1
            input_tokens += int(safe_float(row.get("estimated_input_tokens")))
            output_tokens += int(safe_float(row.get("estimated_output_tokens")))
            cost_usd += safe_float(row.get("estimated_cost_usd"))
        qa_raw = row.get("quality_audit_result")
        qa: dict[str, Any] = qa_raw if isinstance(qa_raw, dict) else {}
        if qa.get("passed") is True:
            audit["pass"] += 1
        elif qa.get("passed") is False:
            audit["fail"] += 1
        else:
            status = str(qa.get("status") or "not_run")
            audit[status if status in audit else "unknown"] += 1
        recent.append(
            {
                "recorded_at": row.get("recorded_at"),
                "task_type": row.get("task_type"),
                "worker_class": worker_class,
                "route_mode": row.get("route_mode"),
                "frontier_call_avoided": row.get("frontier_call_avoided"),
                "model_used": row.get("model_used"),
                "quality_audit_result": qa,
                "escalation_reason": row.get("escalation_reason"),
            }
        )
    summary = {
        "schema_version": "local-first-summary-v1",
        "path": str(LOCAL_FIRST_TELEMETRY_JSONL),
        "record_count": len(rows),
        "local_frontier_split": split,
        "frontier_calls_avoided": avoided,
        "estimated_tokens_saved": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
        "estimated_cost_saved_usd": round(cost_usd, 6),
        "quality_audit": audit,
        "recent": recent[-10:][::-1],
        "privacy": "Raw prompts, descriptions, secrets, and prompt text are not stored; router logs keep only task metadata and text digests.",
        "updated_at": now_iso(),
    }
    LOCAL_FIRST_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_FIRST_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def trace_sources_payload() -> list[dict[str, Any]]:
    module = trace_common_module()
    module.write_recommended_sources()
    return module.PUBLIC_TRACE_SOURCES


def local_first_runner_module():
    scripts_root = REPO_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    import local_first_runner  # type: ignore[import-not-found]

    return local_first_runner


def local_first_run_payload(body: dict[str, Any]) -> dict[str, Any]:
    module = local_first_runner_module()
    cfg = module.load_config(REPO_ROOT / "config" / "local_frontier_router_policy.json")
    text = str(body.get("text") or "").strip()
    if not text:
        raise ValueError("Request text is required")
    if len(text) > 20_000:
        raise ValueError("Request text is too long for the dashboard launcher")
    task_type = str(body.get("taskType") or body.get("task_type") or "summarization")
    risk = str(body.get("risk") or "low")
    app = str(body.get("app") or "ai-training")
    result = module.run_one(
        {
            "task_type": task_type,
            "risk": risk,
            "text": text,
            "app": app,
            "source": "control_panel_local_first_launcher",
        },
        cfg,
        timeout=float(body.get("timeout") or 45),
        max_tokens=int(body.get("maxTokens") or body.get("max_tokens") or 500),
        temperature=float(body.get("temperature") or 0.1),
        log_shadow=True,
        packet_root=module.DEFAULT_PACKET_ROOT,
    )
    local_first_summary()
    return result



def public_commands() -> list[dict[str, Any]]:
    return [
        {"id": cid, "label": cfg["label"], "description": cfg["description"], "risk": cfg["risk"]}
        for cid, cfg in COMMANDS.items()
    ]


def safe_env(command_id: str, body: dict[str, Any]) -> dict[str, str]:
    cfg = COMMANDS[command_id]
    env: dict[str, str] = {}
    requested_raw = body.get("env")
    requested: dict[str, Any] = requested_raw if isinstance(requested_raw, dict) else {}

    if command_id == "verify":
        timeout = requested.get("VERIFY_TIMEOUT_SECONDS", body.get("verifyTimeoutSeconds"))
        if timeout is not None:
            try:
                timeout_i = int(timeout)
            except Exception as exc:
                raise ValueError("VERIFY_TIMEOUT_SECONDS must be an integer") from exc
            if not 5 <= timeout_i <= 600:
                raise ValueError("VERIFY_TIMEOUT_SECONDS must be between 5 and 600")
            env["VERIFY_TIMEOUT_SECONDS"] = str(timeout_i)

    if command_id == "smoke-train":
        run_id = requested.get("RUN_ID") or body.get("runId")
        if not run_id:
            run_id = "gemma-orchestrator-smoke-ui-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if not isinstance(run_id, str) or not SAFE_RUN_ID.match(run_id):
            raise ValueError("RUN_ID must be a safe slug: letters, numbers, dot, underscore, hyphen; max 80 chars")
        env["RUN_ID"] = run_id

    unknown = set(requested.keys()) - set(cfg["env_keys"])
    if unknown:
        raise ValueError("Unsupported env override(s): " + ", ".join(sorted(unknown)))
    return env


def safe_slug(raw: str, prefix: str = "item") -> str:
    """Return a filesystem-safe short slug for arena task/candidate IDs."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip()).strip(".-_").lower()
    if not base:
        base = prefix
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base[:48]}-{stamp}"


def arena_task_path(task_id: str) -> Path:
    if not SAFE_RUN_ID.match(task_id):
        raise ValueError("Invalid task id")
    return ARENA_ROOT / "tasks" / f"{task_id}.json"


def arena_candidates_dir(task_id: str) -> Path:
    if not SAFE_RUN_ID.match(task_id):
        raise ValueError("Invalid task id")
    return ARENA_ROOT / "candidates" / task_id


def arena_candidate_path(task_id: str, candidate_id: str) -> Path:
    if not SAFE_RUN_ID.match(candidate_id):
        raise ValueError("Invalid candidate id")
    return arena_candidates_dir(task_id) / f"{candidate_id}.json"


def arena_read_json(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def arena_list_tasks() -> list[dict[str, Any]]:
    ensure_dirs()
    tasks: list[dict[str, Any]] = []
    for path in sorted((ARENA_ROOT / "tasks").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        task = arena_read_json(path)
        if not task:
            continue
        task_id = str(task.get("id") or path.stem)
        candidate_dir = arena_candidates_dir(task_id)
        candidate_count = len(list(candidate_dir.glob("*.json"))) if candidate_dir.exists() else 0
        winner_path = ARENA_ROOT / "winners" / f"{task_id}.json"
        tasks.append({**task, "candidate_count": candidate_count, "has_winner": winner_path.exists()})
    return tasks[:100]


def arena_get_task(task_id: str) -> dict[str, Any]:
    task = arena_read_json(arena_task_path(task_id))
    if not task:
        raise FileNotFoundError("Unknown arena task")
    candidate_dir = arena_candidates_dir(task_id)
    candidates = []
    if candidate_dir.exists():
        for path in sorted(candidate_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate = arena_read_json(path)
            if candidate:
                candidates.append(candidate)
    review = read_json(ARENA_ROOT / "reviews" / f"{task_id}.json")
    winner = read_json(ARENA_ROOT / "winners" / f"{task_id}.json")
    return {"task": task, "candidates": candidates, "review": review, "winner": winner}


def arena_create_task(body: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs()
    title = str(body.get("title") or "").strip()
    prompt = str(body.get("prompt") or "").strip()
    if not title or not prompt:
        raise ValueError("Task title and prompt are required")
    task_id = safe_slug(title, "task")
    task = {
        "id": task_id,
        "title": title[:180],
        "prompt": prompt[:20000],
        "acceptance_criteria": str(body.get("acceptanceCriteria") or "").strip()[:12000],
        "repo_path": str(body.get("repoPath") or REPO_ROOT).strip()[:500],
        "status": "collecting_candidates",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "promotion_gate": "Candidate must pass tests/review and needs manual approval before merge or training promotion.",
    }
    arena_task_path(task_id).write_text(json.dumps(task, indent=2), encoding="utf-8")
    return task


def arena_score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    diff = str(candidate.get("diff") or "")
    test_log = str(candidate.get("test_log") or "")
    review_notes = str(candidate.get("review_notes") or "")
    combined = "\n".join([diff, test_log, review_notes]).lower()
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    pass_terms = ["passed", "all tests pass", "ok", "success", "successful", "0 failed", "returncode: 0"]
    fail_patterns = [r"(?<!0 )failed", r"traceback", r"error:", r"exception", r"[1-9][0-9]* failed", r"leaked secret", r"exposed token"]
    score = 50
    reasons: list[str] = []
    if any(term in combined for term in pass_terms):
        score += 20
        reasons.append("Evidence suggests tests/checks passed")
    if any(re.search(pattern, combined) for pattern in fail_patterns):
        score -= 30
        reasons.append("Failure/security keywords found in logs or review")
    if diff.strip():
        score += 10
        reasons.append("Candidate includes a concrete diff")
    if added + removed > 700:
        score -= 10
        reasons.append("Large diff: review carefully")
    if not test_log.strip():
        score -= 10
        reasons.append("No test log supplied")
    score = max(0, min(100, score))
    if not reasons:
        reasons.append("Insufficient evidence; manual review required")
    return {"score": score, "added_lines": added, "removed_lines": removed, "reasons": reasons}


def arena_add_candidate(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    task_path = arena_task_path(task_id)
    if not task_path.exists():
        raise FileNotFoundError("Unknown arena task")
    label = str(body.get("label") or "Candidate").strip()[:120]
    candidate_id = safe_slug(label, "candidate")
    candidate: dict[str, Any] = {
        "id": candidate_id,
        "task_id": task_id,
        "label": label,
        "summary": str(body.get("summary") or "").strip()[:4000],
        "diff": str(body.get("diff") or "").strip()[:120000],
        "test_log": str(body.get("testLog") or "").strip()[:50000],
        "review_notes": str(body.get("reviewNotes") or "").strip()[:20000],
        "worker": str(body.get("worker") or "manual").strip()[:80],
        "worktree_path": str(body.get("worktreePath") or "").strip()[:500],
        "branch": str(body.get("branch") or "").strip()[:180],
        "agent_log_path": str(body.get("agentLogPath") or "").strip()[:500],
        "test_command": str(body.get("testCommand") or "").strip()[:500],
        "created_at": now_iso(),
    }
    candidate["scorecard"] = arena_score_candidate(candidate)
    path = arena_candidate_path(task_id, candidate_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    task = arena_read_json(task_path)
    task["updated_at"] = now_iso()
    task["status"] = "review_candidates"
    task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    return candidate


def arena_score_task(task_id: str) -> dict[str, Any]:
    payload = arena_get_task(task_id)
    candidates = payload["candidates"]
    scored = []
    for candidate in candidates:
        candidate["scorecard"] = arena_score_candidate(candidate)
        arena_candidate_path(task_id, candidate["id"]).write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        scored.append(candidate)
    scored.sort(key=lambda item: item.get("scorecard", {}).get("score", 0), reverse=True)
    review = {
        "task_id": task_id,
        "created_at": now_iso(),
        "candidate_count": len(scored),
        "ranking": [
            {"id": c["id"], "label": c.get("label"), "score": c.get("scorecard", {}).get("score"), "reasons": c.get("scorecard", {}).get("reasons", [])}
            for c in scored
        ],
        "recommendation": "Select a winner only after manual review of the top diff and test log.",
    }
    (ARENA_ROOT / "reviews" / f"{task_id}.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    return review


def arena_select_winner(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(body.get("candidateId") or "").strip()
    if not candidate_id:
        raise ValueError("candidateId is required")
    task = arena_get_task(task_id)["task"]
    candidate = arena_read_json(arena_candidate_path(task_id, candidate_id))
    if not candidate:
        raise FileNotFoundError("Unknown candidate")
    winner = {
        "task_id": task_id,
        "candidate_id": candidate_id,
        "selected_at": now_iso(),
        "manual_note": str(body.get("note") or "").strip()[:4000],
        "task": task,
        "winner": candidate,
        "promotion_status": "manual_gate_required",
    }
    (ARENA_ROOT / "winners" / f"{task_id}.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    dataset_path = ARENA_ROOT / "datasets" / "code-edit-preference.jsonl"
    candidates = arena_get_task(task_id)["candidates"]
    losers = [c for c in candidates if c.get("id") != candidate_id]
    with dataset_path.open("a", encoding="utf-8") as fh:
        if losers:
            for loser in losers:
                fh.write(json.dumps({"task": task, "chosen": candidate, "rejected": loser, "selected_at": winner["selected_at"]}) + "\n")
        else:
            fh.write(json.dumps({"task": task, "chosen": candidate, "rejected": None, "selected_at": winner["selected_at"]}) + "\n")
    task_path = arena_task_path(task_id)
    task["status"] = "winner_selected_manual_gate"
    task["updated_at"] = now_iso()
    task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    return {**winner, "dataset_path": str(dataset_path)}


def run_text_command(argv: list[str], cwd: Path, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)


def resolve_repo_path(raw: str) -> Path:
    repo = Path(raw or REPO_ROOT).expanduser().resolve()
    allowed_roots = [Path("/mnt/c/projects").resolve(), Path("/mnt/s/AITraining").resolve()]
    if not any(repo == root or root in repo.parents for root in allowed_roots):
        raise ValueError("Repo path must be under /mnt/c/projects or /mnt/s/AITraining")
    proc = run_text_command(["git", "rev-parse", "--show-toplevel"], repo, timeout=20)
    if proc.returncode != 0:
        raise ValueError(f"Repo path is not inside a git repository: {proc.stderr.strip()}")
    return Path(proc.stdout.strip()).resolve()


def shellish_command(raw: str) -> list[str]:
    return ["bash", "-lc", raw]


def candidate_agent_command(worker: str, worktree: Path, prompt: str) -> list[str]:
    if worker == "scripted-smoke":
        quoted = json.dumps(prompt)
        return shellish_command(
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"prompt = {quoted}\n"
            "Path('ARENA_CANDIDATE_NOTES.md').write_text('# Arena scripted candidate\\n\\n' + prompt + '\\n', encoding='utf-8')\n"
            "print('scripted candidate wrote ARENA_CANDIDATE_NOTES.md')\n"
            "PY"
        )
    if worker == "codex":
        return ["codex", "exec", "-C", str(worktree), "-s", "workspace-write", "-a", "never", prompt]
    if worker == "claude":
        return [
            "claude",
            "-p",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Read,Write,Edit,MultiEdit,Bash",
            "--max-budget-usd",
            "1",
            prompt,
        ]
    raise ValueError("Unsupported worker. Use scripted-smoke, codex, or claude.")


def build_candidate_prompt(task: dict[str, Any], worker_label: str) -> str:
    return "\n".join(
        [
            "You are running as a candidate implementation worker inside an isolated git worktree.",
            "Make the smallest safe code change that satisfies the task. Do not commit, push, install global packages, or touch secrets.",
            "If tests are needed and available, run focused tests and summarize the result.",
            f"Candidate label: {worker_label}",
            f"Task title: {task.get('title')}",
            "Task prompt:",
            str(task.get("prompt") or ""),
            "Acceptance criteria:",
            str(task.get("acceptance_criteria") or ""),
            "Return a concise summary of files changed, tests run, and remaining risks.",
        ]
    )


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def git_output(repo: Path, args: list[str], timeout: int = 120) -> str:
    proc = run_text_command(["git", *args], repo, timeout=timeout)
    return (proc.stdout + proc.stderr).strip()


def arena_start_candidate_batch(task_id: str, body: dict[str, Any]) -> Job:
    payload = arena_get_task(task_id)
    task = payload["task"]
    repo = resolve_repo_path(str(body.get("repoPath") or task.get("repo_path") or REPO_ROOT))
    worker = str(body.get("worker") or "scripted-smoke").strip()
    count = int(body.get("count") or 3)
    if count < 1 or count > 3:
        raise ValueError("Candidate count must be between 1 and 3")
    test_command = str(body.get("testCommand") or "").strip()[:500]
    run_id = safe_slug(f"batch-{task_id}", "batch")
    log_path = ARENA_ROOT / "runs" / f"{run_id}.log"
    job_id = f"candidate-batch-{run_id}-{uuid.uuid4().hex[:6]}"
    job = Job(
        id=job_id,
        command="candidate-batch",
        label=f"Candidate Batch: {task.get('title')}",
        status="running",
        argv=["internal:candidate-batch", task_id, worker],
        pid=None,
        started_at=now_iso(),
        ended_at=None,
        returncode=None,
        log_path=str(log_path),
        risk="medium",
        env={"TASK_ID": task_id, "WORKER": worker, "COUNT": str(count)},
    )
    with _lock:
        running = [j for j in _jobs.values() if j.status == "running" and j.command == "candidate-batch"]
        if running:
            raise RuntimeError(f"Another candidate batch is already running: {running[0].id}")
        _jobs[job_id] = job
        save_jobs()

    def worker_thread() -> None:
        rc = 0
        try:
            append_log(log_path, f"[{now_iso()}] candidate batch start task={task_id} repo={repo} worker={worker} count={count}")
            base_ref = git_output(repo, ["rev-parse", "HEAD"])
            for index in range(1, count + 1):
                label = f"{worker}-{index}"
                branch = f"arena/{task_id[:36]}/{label}-{datetime.now().strftime('%H%M%S')}"
                worktree = WORKTREES_ROOT / task_id / label
                append_log(log_path, f"\n[{now_iso()}] creating worktree {worktree} at {base_ref}")
                if worktree.exists():
                    remove_proc = run_text_command(["git", "worktree", "remove", "--force", str(worktree)], repo, timeout=120)
                    append_log(log_path, remove_proc.stdout + remove_proc.stderr)
                    if worktree.exists():
                        shutil.rmtree(worktree)
                run_text_command(["git", "worktree", "prune"], repo, timeout=120)
                proc = run_text_command(["git", "worktree", "add", "-B", branch, str(worktree), base_ref], repo, timeout=120)
                append_log(log_path, proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    rc = 1
                    continue
                prompt = build_candidate_prompt(task, label)
                agent_log = ARENA_ROOT / "runs" / f"{run_id}-{label}.log"
                command = candidate_agent_command(worker, worktree, prompt)
                append_log(log_path, f"[{now_iso()}] running {label}: {' '.join(command[:4])} ...")
                try:
                    proc = run_text_command(command, worktree, timeout=2400)
                    agent_text = proc.stdout + proc.stderr
                    append_log(agent_log, agent_text or "(agent produced no output)")
                    append_log(log_path, f"[{now_iso()}] {label} agent rc={proc.returncode} log={agent_log}")
                    if proc.returncode != 0:
                        rc = 1
                except subprocess.TimeoutExpired as exc:
                    agent_text = f"TIMEOUT after {exc.timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}"
                    append_log(agent_log, agent_text)
                    append_log(log_path, f"[{now_iso()}] {label} timed out")
                    rc = 1
                test_log = ""
                if test_command:
                    append_log(log_path, f"[{now_iso()}] running tests for {label}: {test_command}")
                    try:
                        test_proc = run_text_command(shellish_command(test_command), worktree, timeout=1200)
                        test_log = f"[test returncode: {test_proc.returncode}]\n" + (test_proc.stdout + test_proc.stderr)
                        if test_proc.returncode == 0 and not (test_proc.stdout or test_proc.stderr):
                            test_log += "test command completed successfully\n"
                        append_log(agent_log, "\n--- TEST LOG ---\n" + (test_log or "(empty test output)"))
                        if test_proc.returncode != 0:
                            rc = 1
                    except subprocess.TimeoutExpired as exc:
                        test_log = f"TEST TIMEOUT after {exc.timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}"
                        append_log(agent_log, "\n--- TEST LOG ---\n" + test_log)
                        rc = 1
                run_text_command(["git", "add", "-N", "."], worktree, timeout=120)
                diff = git_output(worktree, ["diff", "--no-ext-diff"], timeout=120)
                status = git_output(worktree, ["status", "--short"], timeout=120)
                summary = f"Automated {worker} candidate. Agent log: {agent_log}. Worktree: {worktree}. Git status:\n{status or '(clean)'}"
                candidate_record = arena_add_candidate(
                    task_id,
                    {
                        "label": label,
                        "summary": summary,
                        "diff": diff,
                        "testLog": test_log,
                        "reviewNotes": f"Candidate batch worker={worker}; branch={branch}; base={base_ref}; status={status or '(clean)'}",
                        "worker": worker,
                        "worktreePath": str(worktree),
                        "branch": branch,
                        "agentLogPath": str(agent_log),
                        "testCommand": test_command,
                    },
                )
                try:
                    trace_common_module().append_trace_event(
                        worker,
                        {
                            "kind": "arena_candidate_attempt",
                            "task_id": task_id,
                            "candidate": candidate_record,
                            "branch": branch,
                            "base_ref": base_ref,
                            "agent_returncode": proc.returncode,
                            "test_command": test_command,
                            "git_status": status,
                        },
                        run_id=run_id,
                    )
                except Exception as exc:  # noqa: BLE001 - trace capture must not break arena runs
                    append_log(log_path, f"[{now_iso()}] trace capture warning for {label}: {repr(exc)}")
            review = arena_score_task(task_id)
            append_log(log_path, f"[{now_iso()}] wrote review ranking with {review['candidate_count']} candidates")
        except Exception as exc:  # noqa: BLE001 - record batch failure
            rc = 1
            append_log(log_path, f"[{now_iso()}] ERROR: {repr(exc)}")
        finally:
            task_path = arena_task_path(task_id)
            task_latest = arena_read_json(task_path)
            if task_latest:
                task_latest["status"] = "review_candidates"
                task_latest["updated_at"] = now_iso()
                task_path.write_text(json.dumps(task_latest, indent=2), encoding="utf-8")
            with _lock:
                saved = _jobs.get(job_id)
                if saved:
                    saved.status = "completed" if rc == 0 else "failed"
                    saved.returncode = rc
                    saved.ended_at = now_iso()
                save_jobs()
            append_log(log_path, f"[{now_iso()}] candidate batch finished rc={rc}")

    task["status"] = "batch_running"
    task["updated_at"] = now_iso()
    arena_task_path(task_id).write_text(json.dumps(task, indent=2), encoding="utf-8")
    threading.Thread(target=worker_thread, daemon=True).start()
    return job


def run_job(command_id: str, body: dict[str, Any]) -> Job:
    if command_id not in COMMANDS:
        raise ValueError("Unknown command")
    cfg = COMMANDS[command_id]
    argv = [str(Path(arg).resolve()) if arg.startswith("/") else arg for arg in cfg["argv"]]
    script = Path(argv[0]).resolve()
    scripts_root = (REPO_ROOT / "scripts").resolve()
    if not script.exists() or not script.is_file():
        raise ValueError(f"Script does not exist: {script}")
    if scripts_root not in [script.parent, *script.parents]:
        raise ValueError("Script path escaped allowlisted scripts directory")
    if script.is_symlink():
        raise ValueError("Refusing to run symlinked script")

    # Keep GPU/env mutations single-flight for v1.
    with _lock:
        running = [j for j in _jobs.values() if j.status == "running" and j.command in {"setup", "smoke-train", "verify"}]
        if running:
            raise RuntimeError(f"Another AI-training job is already running: {running[0].id} ({running[0].command})")

    env_overrides = safe_env(command_id, body)
    ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{cfg['log_prefix']}-{stamp}-{uuid.uuid4().hex[:6]}"
    log_path = CONTROL_ROOT / f"{job_id}.log"
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "HOME": os.environ.get("HOME", "/home/efenn"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        **env_overrides,
    }
    if command_id == "smoke-train":
        run_id = child_env["RUN_ID"]
        log_path = RUNS_ROOT / run_id / "control-panel.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    log_file.write(f"[{now_iso()}] starting {command_id}: {' '.join(argv)}\n")
    if env_overrides:
        log_file.write(f"[{now_iso()}] env overrides: {json.dumps(env_overrides)}\n")

    proc = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        env=child_env,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job = Job(
        id=job_id,
        command=command_id,
        label=cfg["label"],
        status="running",
        argv=argv,
        pid=proc.pid,
        started_at=now_iso(),
        ended_at=None,
        returncode=None,
        log_path=str(log_path),
        risk=cfg["risk"],
        env=env_overrides,
    )
    with _lock:
        _live[job_id] = proc
        _jobs[job_id] = job
        save_jobs()

    def monitor() -> None:
        try:
            rc = proc.wait(timeout=cfg.get("timeout_seconds", 7200))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                rc = proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                rc = proc.poll()
            end_status = "timeout"
        else:
            end_status = "completed" if rc == 0 else "failed"
        finally:
            log_file.write(f"[{now_iso()}] finished {command_id} rc={proc.returncode}\n")
            log_file.close()
        with _lock:
            saved = _jobs.get(job_id)
            if saved:
                saved.status = end_status
                saved.ended_at = now_iso()
                saved.returncode = proc.returncode
            _live.pop(job_id, None)
            save_jobs()

    threading.Thread(target=monitor, daemon=True).start()
    return job


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        return [asdict(job) for job in sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)]


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def cancel_job(job_id: str) -> Job:
    with _lock:
        proc = _live.get(job_id)
        job = _jobs.get(job_id)
    if not job:
        raise ValueError("Unknown job")
    if not proc or proc.poll() is not None:
        return job
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        job.status = "cancelling"
        save_jobs()
    except Exception as exc:
        job.error = repr(exc)
    return job


def read_log(job: Job, offset: int = 0) -> dict[str, Any]:
    path = Path(job.log_path)
    if not path.exists():
        return {"text": "", "offset": offset, "next_offset": offset, "eof": True, "size": 0}
    size = path.stat().st_size
    if offset < 0:
        offset = max(0, size - MAX_LOG_BYTES)
    with path.open("rb") as fh:
        fh.seek(min(offset, size))
        data = fh.read(MAX_LOG_BYTES)
    return {
        "text": data.decode("utf-8", errors="replace"),
        "offset": offset,
        "next_offset": min(size, offset + len(data)),
        "eof": offset + len(data) >= size,
        "size": size,
        "path": str(path),
    }


def resolve_artifact(root_name: str, rel: str = "") -> Path:
    if root_name not in ARTIFACT_ROOTS:
        raise ValueError("Unknown artifact root")
    root = ARTIFACT_ROOTS[root_name].resolve()
    rel = unquote(rel or "").replace("\\", "/").lstrip("/")
    if "\x00" in rel:
        raise ValueError("Invalid path")
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escaped artifact root")
    return candidate


def list_artifacts(root_name: str, rel: str = "") -> dict[str, Any]:
    path = resolve_artifact(root_name, rel)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_dir():
        raise ValueError("Artifact path is not a directory")
    items = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:500]:
        try:
            st = child.stat()
            items.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(ARTIFACT_ROOTS[root_name].resolve())),
                    "type": "dir" if child.is_dir() else "file",
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "previewable": child.is_file() and child.suffix.lower() in TEXT_SUFFIXES and st.st_size <= MAX_PREVIEW_BYTES,
                }
            )
        except OSError:
            continue
    return {"root": root_name, "path": rel, "abs_path": str(path), "items": items}


def preview_artifact(root_name: str, rel: str) -> dict[str, Any]:
    path = resolve_artifact(root_name, rel)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError("Preview supports only text/log/json/markdown-like files")
    size = path.stat().st_size
    if size > MAX_PREVIEW_BYTES:
        raise ValueError(f"File too large to preview ({size} bytes)")
    return {"path": rel, "abs_path": str(path), "size": size, "text": path.read_text(encoding="utf-8", errors="replace")}


class Handler(BaseHTTPRequestHandler):
    server_version = "AITrainingControlPanel/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib override name
        sys.stderr.write("[%s] %s\n" % (now_iso(), format % args))

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if path == "/api/status":
                return json_response(self, status_payload())
            if path == "/api/commands":
                return json_response(self, public_commands())
            if path == "/api/traces/summary":
                return json_response(self, trace_lab_payload())
            if path == "/api/local-first/summary":
                return json_response(self, local_first_summary())
            if path == "/api/traces/sources":
                return json_response(self, trace_sources_payload())
            if path == "/api/jobs":
                return json_response(self, list_jobs())
            if path.startswith("/api/jobs/") and path.endswith("/log"):
                job_id = path.split("/")[3]
                job = get_job(job_id)
                if not job:
                    return error(self, 404, "Unknown job")
                offset = int(qs.get("offset", ["-1"])[0])
                return json_response(self, read_log(job, offset))
            if path.startswith("/api/jobs/"):
                job_id = path.split("/")[3]
                job = get_job(job_id)
                if not job:
                    return error(self, 404, "Unknown job")
                return json_response(self, asdict(job))
            if path == "/api/artifacts/roots":
                return json_response(self, {name: str(root) for name, root in ARTIFACT_ROOTS.items()})
            if path == "/api/arena/tasks":
                return json_response(self, arena_list_tasks())
            if path.startswith("/api/arena/tasks/"):
                task_id = path.split("/")[4]
                return json_response(self, arena_get_task(task_id))
            if path == "/api/artifacts":
                return json_response(self, list_artifacts(qs.get("root", ["runs"])[0], qs.get("path", [""])[0]))
            if path == "/api/artifacts/preview":
                return json_response(self, preview_artifact(qs.get("root", ["runs"])[0], qs.get("path", [""])[0]))
            return self.serve_static(path)
        except Exception as exc:  # noqa: BLE001 - local diagnostic server
            return error(self, 500, repr(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            body_raw = self.rfile.read(min(length, 65536)) if length else b"{}"
            body = json.loads(body_raw.decode("utf-8") or "{}")
            if parsed.path.startswith("/api/commands/") and parsed.path.endswith("/run"):
                command_id = parsed.path.split("/")[3]
                if command_id == "setup" and not body.get("confirm"):
                    return error(self, 400, "Setup requires confirm=true because it installs/updates packages")
                job = run_job(command_id, body)
                return json_response(self, asdict(job), 201)
            if parsed.path == "/api/traces/import-fable":
                return json_response(self, asdict(run_job("trace-import-fable", body)), 201)
            if parsed.path == "/api/traces/import-autoresearch":
                return json_response(self, asdict(run_job("trace-import-autoresearch", body)), 201)
            if parsed.path == "/api/traces/normalize-autoresearch":
                return json_response(self, asdict(run_job("trace-normalize-autoresearch", body)), 201)
            if parsed.path == "/api/traces/write-sources":
                return json_response(self, asdict(run_job("trace-write-sources", body)), 201)
            if parsed.path == "/api/local-first/smoke":
                return json_response(self, asdict(run_job("local-first-smoke", body)), 201)
            if parsed.path == "/api/local-first/run":
                return json_response(self, local_first_run_payload(body))
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.split("/")[3]
                job = cancel_job(job_id)
                return json_response(self, asdict(job))
            if parsed.path == "/api/arena/tasks":
                return json_response(self, arena_create_task(body), 201)
            if parsed.path.startswith("/api/arena/tasks/") and parsed.path.endswith("/batch"):
                task_id = parsed.path.split("/")[4]
                return json_response(self, asdict(arena_start_candidate_batch(task_id, body)), 201)
            if parsed.path.startswith("/api/arena/tasks/") and parsed.path.endswith("/candidates"):
                task_id = parsed.path.split("/")[4]
                return json_response(self, arena_add_candidate(task_id, body), 201)
            if parsed.path.startswith("/api/arena/tasks/") and parsed.path.endswith("/score"):
                task_id = parsed.path.split("/")[4]
                return json_response(self, arena_score_task(task_id))
            if parsed.path.startswith("/api/arena/tasks/") and parsed.path.endswith("/winner"):
                task_id = parsed.path.split("/")[4]
                return json_response(self, arena_select_winner(task_id, body))
            return error(self, 404, "Not found")
        except ValueError as exc:
            return error(self, 400, str(exc))
        except RuntimeError as exc:
            return error(self, 409, str(exc))
        except Exception as exc:  # noqa: BLE001
            return error(self, 500, repr(exc))

    def serve_static(self, request_path: str) -> None:
        rel = request_path.lstrip("/") or "index.html"
        target = (STATIC_ROOT / rel).resolve()
        if target != STATIC_ROOT and STATIC_ROOT not in target.parents:
            return error(self, 403, "Forbidden")
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            return error(self, 404, "Not found")
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
        }
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(target.suffix, "application/octet-stream"))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ensure_dirs()
    load_jobs()
    allow_wsl_host_bind = os.environ.get("AI_TRAINING_ALLOW_WSL_HOST_BIND", "").lower() in {"1", "true", "yes"}
    if HOST not in {"127.0.0.1", "localhost", "::1"} and not (allow_wsl_host_bind and HOST in {"0.0.0.0", "::"}):
        raise SystemExit(
            "Refusing to bind non-local host; set AI_TRAINING_UI_HOST=127.0.0.1 "
            "or AI_TRAINING_ALLOW_WSL_HOST_BIND=1 for the Windows WSL launcher"
        )
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    urls = [f"http://{HOST}:{PORT}/"]
    if HOST == "0.0.0.0":
        try:
            proc = subprocess.run(["hostname", "-I"], text=True, capture_output=True, timeout=2)
            wsl_ip = (proc.stdout.split() or [""])[0]
            if wsl_ip:
                urls.append(f"http://{wsl_ip}:{PORT}/")
        except Exception:
            pass
    print("AI Training / Unsloth control panel: " + " | ".join(urls), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
