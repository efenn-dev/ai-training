#!/usr/bin/env python3
"""Validate and summarize AI-training curation queue JSONL."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED = {"id", "lane", "source", "status", "sanitized", "split", "input", "label"}
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    return ""


def validate_item(item: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED - set(item)
    if missing:
        errors.append(f"line {line_no}: missing required keys: {', '.join(sorted(missing))}")
    if item.get("sanitized") is not True:
        errors.append(f"line {line_no}: sanitized must be true before queue admission")
    if item.get("status") not in {"candidate", "needs_label", "reviewed", "rejected", "parked"}:
        errors.append(f"line {line_no}: invalid status {item.get('status')!r}")
    if item.get("split") not in {"train", "eval", "holdout", "holdout_real_shadow", "unsplit"}:
        errors.append(f"line {line_no}: invalid split {item.get('split')!r}")
    text = flatten_text({"input": item.get("input"), "label": item.get("label"), "notes": item.get("notes")})
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"line {line_no}: possible secret-like content matched {pattern.pattern}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="/mnt/s/AITraining/curation/queue.jsonl")
    args = parser.parse_args()

    queue = Path(args.queue)
    if not queue.exists():
        raise SystemExit(f"queue does not exist: {queue}")

    errors: list[str] = []
    lane_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0

    with queue.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(item, dict):
                errors.append(f"line {line_no}: item must be object")
                continue
            item_id = str(item.get("id", ""))
            if item_id in seen:
                errors.append(f"line {line_no}: duplicate id {item_id}")
            seen.add(item_id)
            errors.extend(validate_item(item, line_no))
            lane_counts[str(item.get("lane", "missing"))] += 1
            status_counts[str(item.get("status", "missing"))] += 1
            split_counts[str(item.get("split", "missing"))] += 1

    print(json.dumps({
        "queue": str(queue),
        "total": total,
        "lanes": dict(lane_counts),
        "statuses": dict(status_counts),
        "splits": dict(split_counts),
        "errors": errors,
        "ok": not errors,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
