#!/usr/bin/env python3
"""Run local-first requests with frontier fallback packets.

The runner is intentionally conservative:
- policy decides whether local is even allowed;
- local endpoints are used only for draft/triage task types;
- weak, empty, low-confidence, or self-escalating local responses become
  frontier packet candidates instead of production answers;
- telemetry stores digests/metadata, not raw prompts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from local_frontier_router import (
    DEFAULT_CONFIG,
    append_local_first_telemetry,
    append_shadow_log,
    decide,
    load_config,
    normalize_task,
)

DEFAULT_REPORT_ROOT = Path("/mnt/s/AITraining/router-tests")
DEFAULT_PACKET_ROOT = Path("/mnt/s/AITraining/frontier-routing-packets")
REFUSAL_OR_FAILURE = re.compile(
    r"(?i)\b(i can'?t|cannot|unable to|not enough information|as an ai|i do not have access|"
    r"i have no access|i cannot verify|needs frontier|escalat(e|ion))\b"
)
ANSWER_FAILURE_LANGUAGE = re.compile(
    r"(?i)\b(i can'?t|cannot|unable to|not enough information|as an ai|i do not have access|"
    r"i have no access|i cannot verify)\b"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def digest_text(text: str) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def models_url(chat_endpoint: str) -> str:
    if chat_endpoint.endswith("/chat/completions"):
        return chat_endpoint[: -len("/chat/completions")] + "/models"
    return chat_endpoint.rstrip("/") + "/models"


def running_under_wsl() -> bool:
    osrelease = Path("/proc/sys/kernel/osrelease")
    if not osrelease.exists():
        return False
    try:
        return "microsoft" in osrelease.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def windows_host_from_wsl() -> str | None:
    override = os.environ.get("LOCAL_FIRST_WINDOWS_HOST")
    if override:
        return override.strip() or None
    route = Path("/proc/net/route")
    if route.exists():
        try:
            for line in route.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "00000000":
                    gateway = int(parts[2], 16)
                    return ".".join(str((gateway >> (8 * idx)) & 0xFF) for idx in range(4))
        except (OSError, ValueError):
            pass
    resolv = Path("/etc/resolv.conf")
    if not resolv.exists():
        return None
    try:
        for line in resolv.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                return parts[1]
    except OSError:
        return None
    return None


def endpoint_variants(endpoint: str) -> list[str]:
    variants = [endpoint]
    parsed = urllib.parse.urlparse(endpoint)
    if not running_under_wsl() or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return variants
    host = windows_host_from_wsl()
    if not host:
        return variants
    netloc = host
    if parsed.port:
        netloc = f"[{host}]:{parsed.port}" if ":" in host else f"{host}:{parsed.port}"
    fallback = urllib.parse.urlunparse(parsed._replace(netloc=netloc))
    if fallback not in variants:
        variants.append(fallback)
    return variants


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def model_from_candidate(candidate: dict[str, Any], timeout: float) -> str:
    configured = candidate.get("model")
    if configured:
        return str(configured)
    models = get_json(models_url(str(candidate["endpoint"])), timeout)
    data = models.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("id"):
        return str(data[0]["id"])
    raise RuntimeError(f"No model id configured or discoverable for {candidate.get('name')}")


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the local draft model inside an AI training control plane. "
                "Only answer low-risk draft, summary, classification, and inspection tasks. "
                "Do not claim you ran tests, edited files, launched training, saw secrets, or verified production behavior. "
                "Return only strict JSON with keys: answer, confidence, needs_frontier, escalation_reason, assumptions. "
                "The answer value must be a string, confidence must be a number from 0 to 1, "
                "needs_frontier must be a boolean, escalation_reason must be a string or null, "
                "and assumptions must be a JSON array of strings."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_type": task["task_type"],
                    "risk": task["risk"],
                    "app": task["app"],
                    "request": task.get("text", ""),
                },
                ensure_ascii=False,
            ),
        },
    ]


def choice_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


def choice_metadata(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    reasoning = ""
    if isinstance(message, dict):
        reasoning = str(message.get("reasoning") or message.get("reasoning_content") or "")
    return {
        "finish_reason": first.get("finish_reason"),
        "reasoning_chars": len(reasoning),
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def answer_to_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value or "")


def parse_local_content(content: str) -> dict[str, Any]:
    parsed = extract_json_object(content)
    if parsed is not None:
        return {
            "answer": answer_to_text(parsed.get("answer")),
            "confidence": parsed.get("confidence"),
            "needs_frontier": bool(parsed.get("needs_frontier", False)),
            "escalation_reason": parsed.get("escalation_reason"),
            "assumptions": parsed.get("assumptions") if isinstance(parsed.get("assumptions"), list) else [],
            "strict_json": True,
        }
    return {
        "answer": content.strip(),
        "confidence": None,
        "needs_frontier": bool(REFUSAL_OR_FAILURE.search(content)),
        "escalation_reason": "local_response_not_strict_json",
        "assumptions": [],
        "strict_json": False,
    }


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quality_gate(parsed: dict[str, Any], cfg: dict[str, Any], task_type: str | None = None) -> dict[str, Any]:
    gate_cfg = cfg.get("local_quality_gate") if isinstance(cfg.get("local_quality_gate"), dict) else {}
    min_chars = int(gate_cfg.get("min_answer_chars", 40))
    task_min_chars = gate_cfg.get("min_answer_chars_by_task")
    if task_type and isinstance(task_min_chars, dict) and task_type in task_min_chars:
        min_chars = int(task_min_chars[task_type])
    min_confidence = float(gate_cfg.get("min_confidence", 0.45))
    require_json = bool(gate_cfg.get("require_strict_json", True))
    answer = str(parsed.get("answer") or "").strip()
    confidence = safe_float(parsed.get("confidence"))
    checks = {
        "non_empty": bool(answer),
        "min_answer_chars": len(answer) >= min_chars,
        "strict_json": bool(parsed.get("strict_json")) or not require_json,
        "confidence": confidence is not None and confidence >= min_confidence,
        "not_self_escalated": not bool(parsed.get("needs_frontier")),
        "no_failure_language": not bool(ANSWER_FAILURE_LANGUAGE.search(answer)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "confidence": confidence,
        "min_confidence": min_confidence,
    }


def allowed_candidates(cfg: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in cfg.get("local_model_endpoint_candidates", []):
        if not isinstance(item, dict):
            continue
        if item.get("enabled") is False:
            continue
        allowed = item.get("allowed_modes")
        if isinstance(allowed, list) and mode not in allowed:
            continue
        if item.get("endpoint"):
            out.append(item)
    return out


def write_frontier_packet(
    task: dict[str, Any],
    decision: dict[str, Any],
    local_attempts: list[dict[str, Any]],
    reason: str,
    packet_root: Path,
) -> Path:
    packet_root.mkdir(parents=True, exist_ok=True)
    packet_id = f"frontier_packet_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{digest_text(json.dumps(task, sort_keys=True))}"
    packet = {
        "schema_version": "local-first-frontier-packet-v1",
        "packet_id": packet_id,
        "created_at": utc_now(),
        "app": task.get("app"),
        "source": task.get("source"),
        "task_type": task.get("task_type"),
        "risk": task.get("risk"),
        "text_digest": digest_text(str(task.get("text") or "")),
        "raw_prompt_stored": False,
        "frontier_review_required": True,
        "reason": reason,
        "policy_decision": decision,
        "local_attempts": local_attempts,
        "instructions": [
            "Review the original request from the live conversation, not this digest-only packet.",
            "Use local attempt metadata to decide whether local routing can be improved.",
            "Do not treat this packet as approval to train, deploy, or change routing policy.",
        ],
    }
    path = packet_root / f"{packet_id}.json"
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_one(
    payload: dict[str, Any],
    cfg: dict[str, Any],
    *,
    timeout: float,
    max_tokens: int,
    temperature: float,
    log_shadow: bool,
    packet_root: Path,
) -> dict[str, Any]:
    task = normalize_task(payload)
    decision = decide(task, cfg)
    local_attempts: list[dict[str, Any]] = []
    started = time.time()

    if log_shadow:
        append_shadow_log(Path(cfg["shadow_log_default"]), task, decision)

    result: dict[str, Any] = {
        "schema_version": "local-first-run-v1",
        "created_at": utc_now(),
        "task_type": task["task_type"],
        "risk": task["risk"],
        "text_digest": digest_text(str(task.get("text") or "")),
        "policy_decision": decision,
        "final_mode": decision["mode"],
        "local_attempts": local_attempts,
    }

    if decision["mode"] != "local_model_draft":
        reason = decision["reason"]
        if decision["mode"] == "frontier_required":
            result["frontier_packet"] = str(write_frontier_packet(task, decision, local_attempts, reason, packet_root))
        audit = {"status": "pass", "passed": True, "source": "local_first_runner_policy_gate"}
        append_local_first_telemetry(Path(cfg["local_first_telemetry_default"]), task, decision, cfg, audit)
        result["elapsed_sec"] = round(time.time() - started, 3)
        return result

    for candidate in allowed_candidates(cfg, "local_model_draft"):
        configured_endpoint = str(candidate["endpoint"])
        for endpoint in endpoint_variants(configured_endpoint):
            attempt: dict[str, Any] = {
                "endpoint_name": candidate.get("name"),
                "endpoint": endpoint,
                "status": "started",
            }
            if endpoint != configured_endpoint:
                attempt["configured_endpoint"] = configured_endpoint
            local_attempts.append(attempt)
            try:
                endpoint_candidate = {**candidate, "endpoint": endpoint}
                model = model_from_candidate(endpoint_candidate, timeout=timeout)
                attempt["model"] = model
                response = post_json(
                    endpoint,
                    {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                        "messages": build_messages(task),
                    },
                    timeout=timeout,
                )
                content = choice_content(response)
                parsed = parse_local_content(content)
                gate = quality_gate(parsed, cfg, task["task_type"])
                response_meta = choice_metadata(response)
                attempt.update(
                    {
                        "status": "completed",
                        "usage": response.get("usage", {}),
                        "finish_reason": response_meta.get("finish_reason"),
                        "reasoning_chars": response_meta.get("reasoning_chars", 0),
                        "answer_chars": len(str(parsed.get("answer") or "")),
                        "strict_json": parsed.get("strict_json"),
                        "quality_gate": gate,
                        "answer_preview": str(parsed.get("answer") or "")[:800],
                        "escalation_reason": parsed.get("escalation_reason"),
                    }
                )
                if gate["passed"]:
                    audit = {"status": "pass", "passed": True, "source": f"local_first_runner:{candidate.get('name')}"}
                    append_local_first_telemetry(Path(cfg["local_first_telemetry_default"]), task, decision, cfg, audit)
                    result.update(
                        {
                            "final_mode": "local_answer",
                            "model_used": model,
                            "endpoint_name": candidate.get("name"),
                            "endpoint": endpoint,
                            "answer": parsed["answer"],
                            "confidence": gate["confidence"],
                            "elapsed_sec": round(time.time() - started, 3),
                        }
                    )
                    return result
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                attempt.update({"status": "failed", "error": repr(exc)})

    fallback_decision = {**decision, "mode": "frontier_required", "recommended_worker": "hermes_or_claude"}
    reason = "No local endpoint produced a response that passed the local quality gate."
    fallback_decision["reason"] = reason
    packet_path = write_frontier_packet(task, fallback_decision, local_attempts, reason, packet_root)
    audit = {"status": "fail", "passed": False, "source": "local_first_runner_quality_gate"}
    append_local_first_telemetry(Path(cfg["local_first_telemetry_default"]), task, fallback_decision, cfg, audit)
    result.update(
        {
            "final_mode": "frontier_required",
            "policy_decision": fallback_decision,
            "frontier_packet": str(packet_path),
            "elapsed_sec": round(time.time() - started, 3),
        }
    )
    return result


def smoke_payloads() -> list[dict[str, Any]]:
    return [
        {
            "task_type": "summarization",
            "risk": "low",
            "app": "ai-training",
            "source": "local_first_smoke",
            "text": "Summarize this router goal in two concise bullets: use local models for safe draft work and escalate risky or failed tasks to a frontier model.",
        },
        {
            "task_type": "classification",
            "risk": "low",
            "app": "ai-training",
            "source": "local_first_smoke",
            "text": "Classify this request as local_ok or frontier_required: count rows in a manifest and report the number.",
        },
        {
            "task_type": "config_change",
            "risk": "high",
            "app": "ai-training",
            "source": "local_first_smoke",
            "text": "Change production routing so all requests go to the local model.",
        },
    ]


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_json:
        possible = Path(args.task_json)
        if possible.exists():
            return json.loads(possible.read_text(encoding="utf-8"))
        return json.loads(args.task_json)
    return {
        "task_type": args.task_type or "summarization",
        "risk": args.risk,
        "text": args.text,
        "app": args.app,
        "source": "local_first_runner_cli",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-json")
    parser.add_argument("--task-type")
    parser.add_argument("--risk", default="normal")
    parser.add_argument("--text", default="")
    parser.add_argument("--app", default="ai-training")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--log-shadow", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    args.report_root.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        results = [
            run_one(
                payload,
                cfg,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                log_shadow=args.log_shadow,
                packet_root=args.packet_root,
            )
            for payload in smoke_payloads()
        ]
        report = {
            "schema_version": "local-first-smoke-report-v1",
            "created_at": utc_now(),
            "case_count": len(results),
            "local_answer_count": sum(1 for item in results if item.get("final_mode") == "local_answer"),
            "frontier_required_count": sum(1 for item in results if item.get("final_mode") == "frontier_required"),
            "results": results,
        }
        report_path = args.report_root / f"local-first-smoke-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
        return 0

    result = run_one(
        load_payload(args),
        cfg,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_shadow=args.log_shadow,
        packet_root=args.packet_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
