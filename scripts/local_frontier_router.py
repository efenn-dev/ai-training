#!/usr/bin/env python3
"""Shadow-mode local-vs-frontier router for AI Training.

This script is deliberately advisory. It does not launch models, edit production
routing, train adapters, or approve review items. It emits a deterministic route
recommendation and can append shadow decisions for future eval/curation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "local_frontier_router_policy.json"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_task(payload: dict[str, Any]) -> dict[str, Any]:
    text_parts = [str(payload.get(k, "")) for k in ("title", "summary", "description", "text")]
    text = " ".join(part for part in text_parts if part).strip()
    return {
        "task_type": str(payload.get("task_type") or payload.get("kind") or "low_risk_inspection"),
        "risk": str(payload.get("risk") or "normal"),
        "text": text,
        "app": payload.get("app") or payload.get("appName") or "ai-training",
        "source": payload.get("source") or "cli",
    }


def any_pattern(patterns: list[str], text: str) -> str | None:
    lowered = text.lower()
    for pattern in patterns:
        cleaned = pattern.lower().strip()
        if re.search(rf"(?<!\w){re.escape(cleaned)}(?!\w)", lowered):
            return pattern
    return None


def decide(task: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    task_type = task["task_type"]
    risk = task["risk"].lower()
    text = " ".join([task_type, risk, task.get("text", "")])

    production_enabled = bool(cfg.get("production_routing_enabled", False))
    human_match = any_pattern(cfg.get("human_required_patterns", []), text)
    frontier_match = any_pattern(cfg.get("frontier_required_patterns", []), text)

    if task_type in cfg.get("human_required_task_types", []) or human_match:
        mode = "human_required"
        reason = f"Human/operator gate required{f' by pattern {human_match!r}' if human_match else ''}."
    elif risk in {"high", "critical", "production", "secrets", "deploy"}:
        mode = "frontier_required"
        reason = f"Risk={risk!r} is above local shadow-only authority."
    elif task_type in cfg.get("frontier_required_task_types", []) or frontier_match:
        mode = "frontier_required"
        reason = f"Frontier review required{f' by pattern {frontier_match!r}' if frontier_match else ''}."
    elif task_type in cfg.get("no_model_script_task_types", []):
        mode = "no_model_script"
        reason = "Deterministic script should handle this before any model call."
    elif task_type in cfg.get("local_model_draft_task_types", []):
        mode = "local_model_draft"
        reason = "Low-risk advisory draft/triage is allowed locally in shadow mode."
    else:
        mode = "frontier_required"
        reason = "Unknown task type defaults to frontier review."

    return {
        "policy_version": cfg["policy_version"],
        "production_routing_enabled": production_enabled,
        "shadow_mode": True,
        "mode": mode,
        "recommended_worker": {
            "no_model_script": "local",
            "local_model_draft": "local",
            "frontier_required": "hermes_or_claude",
            "human_required": "manual",
        }[mode],
        "task_type": task_type,
        "risk": task["risk"],
        "reason": reason,
        "allowed_local_actions": [] if mode in {"frontier_required", "human_required"} else cfg.get("local_required_outputs", []),
        "forbidden_local_actions": cfg.get("local_forbidden_actions", []),
        "requires_frontier_packet": mode == "frontier_required",
        "requires_human_review": mode == "human_required",
    }


def append_shadow_log(path: Path, task: dict[str, Any], decision: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps({"task": task, "decision": decision}, sort_keys=True).encode()).hexdigest()[:16]
    row = {
        "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision_id": f"router_shadow_{digest}",
        "task": task,
        "decision": decision,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def estimate_tokens(text: str) -> int:
    """Return a rough token estimate without persisting private prompt text."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, round(len(stripped) / 4))


def model_name_for_decision(decision: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    if decision.get("mode") != "local_model_draft":
        return None
    candidates = cfg.get("local_model_endpoint_candidates", [])
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            return str(first.get("name") or "local_model")
    return "local_model"


def estimated_frontier_cost(task: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    input_tokens = estimate_tokens(str(task.get("text") or ""))
    # Output tokens are unknown in shadow mode; use a small draft-review allowance.
    output_tokens = 0 if input_tokens == 0 else min(2000, max(256, round(input_tokens * 0.35)))
    rates = cfg.get("frontier_cost_estimates_usd_per_1k_tokens", {})
    input_rate = float(rates.get("default_input", 0) or 0) if isinstance(rates, dict) else 0.0
    output_rate = float(rates.get("default_output", 0) or 0) if isinstance(rates, dict) else 0.0
    cost = (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)
    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 6) if cost else None,
    }


def append_local_first_telemetry(
    path: Path,
    task: dict[str, Any],
    decision: dict[str, Any],
    cfg: dict[str, Any],
    quality_audit_result: dict[str, Any] | None = None,
) -> None:
    """Append privacy-safe local-first telemetry; never stores raw prompt/description text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    worker_class = {
        "no_model_script": "local",
        "local_model_draft": "local",
        "frontier_required": "frontier",
        "human_required": "human",
    }.get(str(decision.get("mode")), "unknown")
    estimates = estimated_frontier_cost(task, cfg)
    text = str(task.get("text") or "")
    frontier_call_avoided = worker_class == "local"
    quality_audit = quality_audit_result or {
        "status": "not_run",
        "passed": None,
        "source": "router_shadow_decision",
    }
    row = {
        "schema_version": "local-first-telemetry-v1",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": task.get("source") or "cli",
        "app": task.get("app") or "ai-training",
        "task_type": task.get("task_type"),
        "risk": task.get("risk"),
        "route_mode": decision.get("mode"),
        "worker_class": worker_class,
        "frontier_call_avoided": frontier_call_avoided,
        "model_used": model_name_for_decision(decision, cfg),
        "routing_decision": {
            "policy_version": decision.get("policy_version"),
            "shadow_mode": decision.get("shadow_mode"),
            "recommended_worker": decision.get("recommended_worker"),
        },
        "quality_audit_result": quality_audit,
        "escalation_reason": None if frontier_call_avoided else decision.get("reason"),
        "text_digest": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16] if text else None,
        **estimates,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--task-json", help="JSON object or path to a JSON file describing the task")
    ap.add_argument("--task-type", help="Task type, e.g. draft_label or training_promotion_decision")
    ap.add_argument("--risk", default="normal")
    ap.add_argument("--text", default="")
    ap.add_argument("--app", default="ai-training")
    ap.add_argument("--log-shadow", action="store_true")
    ap.add_argument("--shadow-log", help="Override shadow JSONL path")
    ap.add_argument("--telemetry-log", help="Override local-first telemetry JSONL path")
    ap.add_argument("--quality-audit-result", choices=["pass", "fail", "not_run"], default="not_run")
    ap.add_argument("--quality-audit-source", default="router_shadow_decision")
    return ap.parse_args()


def payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.task_json:
        return {"task_type": args.task_type or "low_risk_inspection", "risk": args.risk, "text": args.text, "app": args.app}
    possible_path = Path(args.task_json)
    if possible_path.exists():
        return json.loads(possible_path.read_text(encoding="utf-8"))
    return json.loads(args.task_json)


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    task = normalize_task(payload_from_args(args))
    decision = decide(task, cfg)
    if args.log_shadow:
        append_shadow_log(Path(args.shadow_log or cfg["shadow_log_default"]), task, decision)
        telemetry_default = cfg.get("local_first_telemetry_default") or "/mnt/s/AITraining/traces/metrics/local-first-telemetry.jsonl"
        audit_passed = {"pass": True, "fail": False}.get(args.quality_audit_result)
        audit = {"status": args.quality_audit_result, "passed": audit_passed, "source": args.quality_audit_source[:120]}
        append_local_first_telemetry(Path(args.telemetry_log or telemetry_default), task, decision, cfg, audit)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
