#!/usr/bin/env python3
"""Validate the TradeStar v0.1 dataset scaffold and JSONL rows."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "tradestar.v0.1"
SYSTEM_PROMPT = (
    "You are TradeStar's trading research assistant. Return strict JSON only. "
    "Provide analysis and risk labels, not buy/sell instructions."
)
DEFAULT_DATASET_DIR = Path("/mnt/s/AITraining/datasets/tradestar-v0.1")
REQUIRED_FILES = (
    "schema.json",
    "MANIFEST.json",
    "seed_examples.jsonl",
    "train.jsonl",
    "eval.jsonl",
    "holdout.jsonl",
    "README.md",
    "public-data-plan.md",
    "leakage-and-safety-checklist.md",
)
JSONL_SPLITS = {
    "seed_examples.jsonl": "seed",
    "train.jsonl": "train",
    "eval.jsonl": "eval",
    "holdout.jsonl": "holdout",
}
HIGH_RISK_REVIEW_LABELS = {"high_risk_lotto", "avoid_event_risk", "avoid_liquidity"}
FUTURE_OUTCOME_KEYS = (
    "future_outcomes",
    "future_outcomes_for_training_only",
    "forward_outcomes",
    "forward_return",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "max_drawdown",
    "benchmark_relative",
    "post_event",
)
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]
UNSAFE_PATTERNS = [
    re.compile(r"(?i)\b(buy|sell|short|hold)\s+(now|today|immediately|this|the|it)\b"),
    re.compile(r"(?i)\b(go long|go short|enter this trade|take this trade|execute (this|the)|place (an? )?order)\b"),
    re.compile(r"(?i)\b(set|use)\s+(your\s+)?position size\b"),
    re.compile(r"(?i)\b(risk|allocate)\s+\d+(\.\d+)?%\s+(of\s+)?(your\s+)?(account|portfolio)\b"),
    re.compile(r"(?i)\b(guaranteed|risk[- ]free|will definitely profit|cannot lose)\b"),
]
PRIVATE_ACCOUNT_PATTERNS = [
    re.compile(r"(?i)\b(account balance|brokerage login|brokerage password|routing number|margin account)\b"),
    re.compile(r"(?i)\b(suitability|fiduciary|personal financial situation)\b"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--strict-warnings", action="store_true", help="Treat warnings as failures")
    parser.add_argument(
        "--require-nonempty-train-eval",
        action="store_true",
        help="Promotion gate: fail if train/eval are empty",
    )
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in negative smoke tests for the validator itself",
    )
    return parser.parse_args()


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    return "" if value is None else str(value)


def add_limited(items: list[str], message: str, max_items: int) -> None:
    if len(items) < max_items:
        items.append(message)
    elif len(items) == max_items:
        items.append(f"additional messages suppressed after {max_items}")


def load_json(path: Path, errors: list[str], max_errors: int) -> Any | None:
    if not path.exists():
        add_limited(errors, f"missing JSON file: {path}", max_errors)
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        add_limited(errors, f"{path}: invalid JSON: {exc}", max_errors)
    return None


def parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def has_future_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in FUTURE_OUTCOME_KEYS):
                return True
            if has_future_key(item):
                return True
    elif isinstance(value, list):
        return any(has_future_key(item) for item in value)
    return False


def validate_text_safety(text: str, where: str, errors: list[str], max_errors: int) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            add_limited(errors, f"{where}: possible secret-like content matched {pattern.pattern}", max_errors)
    for pattern in PRIVATE_ACCOUNT_PATTERNS:
        if pattern.search(text):
            add_limited(errors, f"{where}: private/personal account content matched {pattern.pattern}", max_errors)
    for pattern in UNSAFE_PATTERNS:
        if pattern.search(text):
            add_limited(errors, f"{where}: unsafe trading instruction/certainty matched {pattern.pattern}", max_errors)


def validate_schema_contract(schema: dict[str, Any], errors: list[str], max_errors: int) -> None:
    allowed = schema.get("allowed_values")
    if not isinstance(allowed, dict):
        add_limited(errors, "schema.json: allowed_values must be an object", max_errors)
        return
    for key in ("task_type", "decision_class", "risk_label"):
        values = allowed.get(key)
        if not isinstance(values, list) or not values:
            add_limited(errors, f"schema.json: allowed_values.{key} must be a non-empty list", max_errors)
    fmt = schema.get("format")
    if not isinstance(fmt, dict):
        add_limited(errors, "schema.json: format must be an object", max_errors)
        return
    for key in ("assistant_required_keys", "risk_required_keys", "evidence_item_required_keys"):
        values = fmt.get(key)
        if not isinstance(values, list) or not values:
            add_limited(errors, f"schema.json: format.{key} must be a non-empty list", max_errors)


def read_jsonl(path: Path, errors: list[str], max_errors: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    if not path.exists():
        add_limited(errors, f"missing JSONL file: {path}", max_errors)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                add_limited(errors, f"{path}:{line_no}: invalid JSON: {exc}", max_errors)
                continue
            if not isinstance(item, dict):
                add_limited(errors, f"{path}:{line_no}: row must be an object", max_errors)
                continue
            rows.append((line_no, item))
    return rows


def validate_row(
    *,
    item: dict[str, Any],
    path: Path,
    line_no: int,
    expected_split: str,
    schema: dict[str, Any],
    seen_ids: set[str],
    errors: list[str],
    warnings: list[str],
    max_errors: int,
) -> tuple[str | None, str | None, str | None]:
    loc = f"{path}:{line_no}"
    allowed = schema.get("allowed_values", {}) if isinstance(schema.get("allowed_values"), dict) else {}
    task_types = set(allowed.get("task_type", []))
    decisions = set(allowed.get("decision_class", []))
    risk_labels = set(allowed.get("risk_label", []))
    fmt = schema.get("format", {}) if isinstance(schema.get("format"), dict) else {}
    assistant_required = set(fmt.get("assistant_required_keys", []))
    risk_required = set(fmt.get("risk_required_keys", []))
    evidence_required = set(fmt.get("evidence_item_required_keys", []))

    sample_id = item.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        add_limited(errors, f"{loc}: missing non-empty sample_id", max_errors)
    elif sample_id in seen_ids:
        add_limited(errors, f"{loc}: duplicate sample_id {sample_id}", max_errors)
    else:
        seen_ids.add(sample_id)

    if item.get("schema_version") != SCHEMA_VERSION:
        add_limited(errors, f"{loc}: schema_version must be {SCHEMA_VERSION!r}", max_errors)
    if item.get("split") != expected_split:
        add_limited(errors, f"{loc}: split must be {expected_split!r} for {path.name}", max_errors)
    if not isinstance(item.get("source_status"), str) or not item.get("source_status"):
        add_limited(errors, f"{loc}: missing source_status", max_errors)

    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        add_limited(errors, f"{loc}: messages must be exactly [system, user, assistant]", max_errors)
        return None, None, None
    roles = [msg.get("role") if isinstance(msg, dict) else None for msg in messages]
    if roles != ["system", "user", "assistant"]:
        add_limited(errors, f"{loc}: message roles must be ['system', 'user', 'assistant']; got {roles!r}", max_errors)

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            add_limited(errors, f"{loc}: message {idx} must be object", max_errors)
            return None, None, None
        if not isinstance(msg.get("content"), str):
            add_limited(errors, f"{loc}: message {idx} content must be string", max_errors)
            return None, None, None

    system_content = messages[0]["content"]
    if system_content != SYSTEM_PROMPT:
        add_limited(warnings, f"{loc}: system prompt differs from expected TradeStar prompt", max_errors)

    try:
        user_payload = json.loads(messages[1]["content"])
    except json.JSONDecodeError as exc:
        add_limited(errors, f"{loc}: user content is not strict JSON: {exc}", max_errors)
        return None, None, None
    try:
        assistant_payload = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        add_limited(errors, f"{loc}: assistant content is not strict JSON: {exc}", max_errors)
        return None, None, None
    if not isinstance(user_payload, dict):
        add_limited(errors, f"{loc}: user content JSON must be object", max_errors)
        return None, None, None
    if not isinstance(assistant_payload, dict):
        add_limited(errors, f"{loc}: assistant content JSON must be object", max_errors)
        return None, None, None

    validate_text_safety(flatten_text(user_payload), f"{loc}: user", errors, max_errors)
    validate_text_safety(flatten_text(assistant_payload), f"{loc}: assistant", errors, max_errors)

    missing = assistant_required - set(assistant_payload)
    if missing:
        add_limited(errors, f"{loc}: assistant missing required keys: {', '.join(sorted(missing))}", max_errors)
    if assistant_payload.get("schema_version") != SCHEMA_VERSION:
        add_limited(errors, f"{loc}: assistant schema_version must be {SCHEMA_VERSION!r}", max_errors)

    user_task = user_payload.get("task_type")
    assistant_task = assistant_payload.get("task_type")
    if user_task not in task_types:
        add_limited(errors, f"{loc}: invalid user task_type {user_task!r}", max_errors)
    if assistant_task not in task_types:
        add_limited(errors, f"{loc}: invalid assistant task_type {assistant_task!r}", max_errors)
    if user_task and assistant_task and user_task != assistant_task:
        add_limited(errors, f"{loc}: assistant task_type {assistant_task!r} does not match user task_type {user_task!r}", max_errors)

    decision = assistant_payload.get("decision_class")
    if decision not in decisions:
        add_limited(errors, f"{loc}: invalid decision_class {decision!r}", max_errors)

    confidence = assistant_payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        add_limited(errors, f"{loc}: confidence must be numeric in [0, 1]", max_errors)
    if assistant_payload.get("not_financial_advice") is not True:
        add_limited(errors, f"{loc}: not_financial_advice must be true", max_errors)
    if not isinstance(assistant_payload.get("missing_data"), list):
        add_limited(errors, f"{loc}: missing_data must be a list", max_errors)

    timestamp = user_payload.get("timestamp_available")
    if not isinstance(timestamp, str) or not timestamp:
        add_limited(errors, f"{loc}: user timestamp_available is required", max_errors)
    elif parse_iso_timestamp(timestamp) is None:
        add_limited(warnings, f"{loc}: timestamp_available is not parseable ISO-8601: {timestamp!r}", max_errors)

    if user_payload.get("symbol") is None and user_task in {
        "fundamental_reasoning",
        "setup_classification",
        "risk_label",
        "journal_review",
        "outcome_eval",
    }:
        add_limited(errors, f"{loc}: symbol is required for market/journal task_type {user_task!r}", max_errors)

    user_has_future_key = has_future_key(user_payload)
    assistant_has_future_key = has_future_key(assistant_payload)
    if user_has_future_key or assistant_has_future_key:
        if expected_split in {"train", "eval", "holdout"}:
            add_limited(errors, f"{loc}: promoted split payload contains future/outcome leakage fields", max_errors)
        elif user_task == "outcome_eval" and user_has_future_key and not assistant_has_future_key:
            add_limited(warnings, f"{loc}: seed outcome_eval includes future outcomes; keep offline-only", max_errors)
        else:
            add_limited(errors, f"{loc}: payload contains future/outcome fields outside allowed offline outcome_eval seed", max_errors)

    evidence = assistant_payload.get("evidence")
    if not isinstance(evidence, list):
        add_limited(errors, f"{loc}: evidence must be a list", max_errors)
    elif not evidence and assistant_payload.get("decision_class") != "skip_insufficient_data":
        add_limited(errors, f"{loc}: evidence should be non-empty unless skipping for insufficient data", max_errors)
    else:
        for idx, ev in enumerate(evidence):
            if not isinstance(ev, dict):
                add_limited(errors, f"{loc}: evidence[{idx}] must be object", max_errors)
                continue
            ev_missing = evidence_required - set(ev)
            if ev_missing:
                add_limited(errors, f"{loc}: evidence[{idx}] missing keys: {', '.join(sorted(ev_missing))}", max_errors)

    risk = assistant_payload.get("risk")
    risk_label = None
    if not isinstance(risk, dict):
        add_limited(errors, f"{loc}: risk must be object", max_errors)
    else:
        risk_missing = risk_required - set(risk)
        if risk_missing:
            add_limited(errors, f"{loc}: risk missing keys: {', '.join(sorted(risk_missing))}", max_errors)
        risk_label = risk.get("risk_label")
        if risk_label not in risk_labels:
            add_limited(errors, f"{loc}: invalid risk_label {risk_label!r}", max_errors)
        if not isinstance(risk.get("risk_factors"), list):
            add_limited(errors, f"{loc}: risk.risk_factors must be a list", max_errors)
        if not isinstance(risk.get("invalidation_conditions"), list):
            add_limited(errors, f"{loc}: risk.invalidation_conditions must be a list", max_errors)
        if not isinstance(risk.get("human_review_required"), bool):
            add_limited(errors, f"{loc}: risk.human_review_required must be boolean", max_errors)
        if risk_label in HIGH_RISK_REVIEW_LABELS and risk.get("human_review_required") is not True:
            add_limited(errors, f"{loc}: {risk_label} requires human_review_required=true", max_errors)

    return str(user_task) if isinstance(user_task, str) else None, str(decision) if isinstance(decision, str) else None, str(risk_label) if isinstance(risk_label, str) else None


def validate_dataset(dataset_dir: Path, *, max_errors: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    if not dataset_dir.exists():
        add_limited(errors, f"dataset_dir does not exist: {dataset_dir}", max_errors)
        return {
            "dataset_dir": str(dataset_dir),
            "files": files,
            "counts": {},
            "task_types": {},
            "decision_classes": {},
            "risk_labels": {},
            "warnings": warnings,
            "errors": errors,
            "ok": False,
        }

    for name in REQUIRED_FILES:
        path = dataset_dir / name
        files[name] = {"exists": path.exists()}
        if not path.exists():
            add_limited(errors, f"missing required file: {path}", max_errors)

    schema = load_json(dataset_dir / "schema.json", errors, max_errors)
    manifest = load_json(dataset_dir / "MANIFEST.json", errors, max_errors)
    if not isinstance(schema, dict):
        schema = {}
    if schema.get("schema_version") != SCHEMA_VERSION:
        add_limited(errors, f"schema.json: schema_version must be {SCHEMA_VERSION!r}", max_errors)
    validate_schema_contract(schema, errors, max_errors)
    if not isinstance(manifest, dict):
        manifest = {}
    elif manifest.get("status") != "scaffold_seed_only_no_training" and not any(
        (dataset_dir / name).stat().st_size for name in ("train.jsonl", "eval.jsonl", "holdout.jsonl") if (dataset_dir / name).exists()
    ):
        add_limited(warnings, "MANIFEST status is not scaffold_seed_only_no_training but promoted splits are empty", max_errors)

    for name, expected_split in JSONL_SPLITS.items():
        path = dataset_dir / name
        rows = read_jsonl(path, errors, max_errors)
        files.setdefault(name, {"exists": path.exists()})["rows"] = len(rows)
        counts[expected_split] += len(rows)
        for line_no, item in rows:
            task, decision, risk = validate_row(
                item=item,
                path=path,
                line_no=line_no,
                expected_split=expected_split,
                schema=schema,
                seen_ids=seen_ids,
                errors=errors,
                warnings=warnings,
                max_errors=max_errors,
            )
            if task:
                task_counts[task] += 1
            if decision:
                decision_counts[decision] += 1
            if risk:
                risk_counts[risk] += 1

    return {
        "dataset_dir": str(dataset_dir),
        "files": files,
        "counts": dict(counts),
        "task_types": dict(task_counts),
        "decision_classes": dict(decision_counts),
        "risk_labels": dict(risk_counts),
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }


def _copy_default_scaffold(base: Path) -> None:
    source = DEFAULT_DATASET_DIR
    for name in REQUIRED_FILES:
        src = source / name
        dst = base / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            dst.write_text("" if name.endswith(".jsonl") else "{}\n", encoding="utf-8")


def _first_seed_row(base: Path) -> dict[str, Any]:
    lines = (base / "seed_examples.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("default seed_examples.jsonl is empty; cannot run self-test")
    return json.loads(lines[0])


def run_self_test() -> bool:
    checks: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="tradestar-validator-") as tmp:
        base = Path(tmp)
        _copy_default_scaffold(base)
        row = _first_seed_row(base)
        assistant = json.loads(row["messages"][2]["content"])
        assistant["analysis_notes"] = "Buy now and place an order. This is guaranteed."
        row["messages"][2]["content"] = json.dumps(assistant)
        (base / "seed_examples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = validate_dataset(base, max_errors=100)
        checks.append((not result["ok"]) and any("unsafe trading" in err for err in result["errors"]))

    with tempfile.TemporaryDirectory(prefix="tradestar-validator-") as tmp:
        base = Path(tmp)
        _copy_default_scaffold(base)
        row = _first_seed_row(base)
        row["split"] = "train"
        assistant = json.loads(row["messages"][2]["content"])
        assistant["future_outcomes_for_training_only"] = {"ret_5d": 0.12}
        row["messages"][2]["content"] = json.dumps(assistant)
        (base / "seed_examples.jsonl").write_text("", encoding="utf-8")
        (base / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = validate_dataset(base, max_errors=100)
        checks.append((not result["ok"]) and any("future/outcome leakage" in err for err in result["errors"]))

    with tempfile.TemporaryDirectory(prefix="tradestar-validator-") as tmp:
        base = Path(tmp)
        _copy_default_scaffold(base)
        row = _first_seed_row(base)
        user = json.loads(row["messages"][1]["content"])
        user["brokerage_password"] = "secret=superprivate123"
        row["messages"][1]["content"] = json.dumps(user)
        (base / "seed_examples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = validate_dataset(base, max_errors=100)
        checks.append((not result["ok"]) and any("secret-like" in err for err in result["errors"]))

    with tempfile.TemporaryDirectory(prefix="tradestar-validator-") as tmp:
        base = Path(tmp)
        _copy_default_scaffold(base)
        row = _first_seed_row(base)
        assistant = json.loads(row["messages"][2]["content"])
        assistant["schema_version"] = "WRONG"
        row["messages"][2]["content"] = json.dumps(assistant)
        (base / "seed_examples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = validate_dataset(base, max_errors=100)
        checks.append((not result["ok"]) and any("assistant schema_version" in err for err in result["errors"]))

    return all(checks)


def main() -> int:
    args = parse_args()
    if args.self_test:
        ok = run_self_test()
        print(json.dumps({"self_test": "negative_smoke_cases", "ok": ok}, indent=2))
        return 0 if ok else 1

    result = validate_dataset(args.dataset_dir, max_errors=args.max_errors)
    if args.require_nonempty_train_eval:
        counts = result.get("counts", {})
        if int(counts.get("train", 0)) == 0:
            result["errors"].append("promotion gate: train split is empty")
        if int(counts.get("eval", 0)) == 0:
            result["errors"].append("promotion gate: eval split is empty")
    if args.strict_warnings and result.get("warnings"):
        result["errors"].append("strict-warnings: warnings present")
    result["ok"] = not result.get("errors")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
