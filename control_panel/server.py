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

REPO_ROOT = Path("/mnt/c/projects/ai-training").resolve()
S_ROOT = Path("/mnt/s/AITraining")
RUNS_ROOT = S_ROOT / "runs"
CONTROL_ROOT = RUNS_ROOT / "control-panel"
STATIC_ROOT = REPO_ROOT / "control_panel" / "static"
JOBS_JSON = CONTROL_ROOT / "jobs.json"
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
        "jobs": jobs,
        "commands": public_commands(),
        "time": now_iso(),
    }


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
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.split("/")[3]
                job = cancel_job(job_id)
                return json_response(self, asdict(job))
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
    if HOST not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing to bind non-local host; set AI_TRAINING_UI_HOST=127.0.0.1")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"AI Training / Unsloth control panel: http://{HOST}:{PORT}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
