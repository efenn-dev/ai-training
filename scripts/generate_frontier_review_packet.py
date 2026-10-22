#!/usr/bin/env python3
"""Generate compact frontier review packets from AI-training state.

The packet intentionally stores counts, summaries, small examples, and artifact
pointers. It does not dump raw transcripts, full logs, or full datasets unless
explicitly asked with --include-raw-snippets.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

SUPPORTED_LANES = {"gaming-pipeline", "general-ops-router"}
DEFAULT_QUEUE = Path("/mnt/s/AITraining/curation/queue.jsonl")
DEFAULT_OUT_DIR = Path("/mnt/s/AITraining/review-packets")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]
TEXT_KEYS = ("title", "summary", "description", "reason", "notes", "text", "content")
QUEUE_REQUIRED = {"id", "lane", "source", "status", "sanitized", "split", "input", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(SUPPORTED_LANES))
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE, help="Current queue JSONL")
    parser.add_argument("--baseline-queue", type=Path, help="Optional prior queue JSONL for changed-example summary")
    parser.add_argument("--dataset-dir", type=Path, help="Optional dataset directory containing JSONL splits")
    parser.add_argument(
        "--eval-report",
        type=Path,
        action="append",
        default=[],
        help="Eval/report JSON to summarize; repeat for baseline/adapter/etc.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Raw artifact pointer to include without embedding contents; repeat as needed.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--packet-id", help="Override output packet id")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-markdown-bytes", type=int, default=32_000)
    parser.add_argument("--include-raw-snippets", action="store_true", help="Include short raw snippets; default is pointer-only")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def stable_digest(value: Any, length: int = 12) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    return ""


def trim_text(value: Any, limit: int = 220) -> str | None:
    text = flatten_text(value).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def compact_obj(value: Any, *, include_raw: bool, text_limit: int = 220) -> Any:
    """Return a small, review-safe representation of an arbitrary value."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return trim_text(value, text_limit) if include_raw else {"text_digest": stable_digest(value), "chars": len(value)}
    if isinstance(value, list):
        return [compact_obj(v, include_raw=include_raw, text_limit=text_limit) for v in value[:3]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"raw", "transcript", "log", "logs", "decoded", "prompt", "completion", "messages"} and not include_raw:
                out[key] = {"omitted": "raw_text", "digest": stable_digest(item), "chars": len(flatten_text(item))}
            elif key in TEXT_KEYS:
                out[key] = trim_text(item, text_limit) if include_raw else compact_obj(item, include_raw=False, text_limit=text_limit)
            elif isinstance(item, (str, list, dict)):
                out[key] = compact_obj(item, include_raw=include_raw, text_limit=text_limit)
            else:
                out[key] = item
        return out
    return str(value)


def compact_example(item: dict[str, Any], *, include_raw: bool) -> dict[str, Any]:
    keys = [
        "id",
        "lane",
        "source",
        "status",
        "split",
        "sanitized",
        "confidence",
        "model",
        "updated_at",
        "created_at",
    ]
    out = {k: item.get(k) for k in keys if k in item}
    if "input" in item:
        out["input"] = compact_obj(item.get("input"), include_raw=include_raw)
    if "label" in item:
        out["label"] = compact_obj(item.get("label"), include_raw=include_raw)
    for key in ("draft", "local_draft", "frontier_label", "human_label", "error", "errors", "notes"):
        if key in item:
            out[key] = compact_obj(item.get(key), include_raw=include_raw)
    out["row_digest"] = stable_digest(item)
    return out


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [f"missing file: {path}"]
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(item, dict):
                errors.append(f"{path}:{line_no}: row is not an object")
                continue
            rows.append(item)
    return rows, errors


def validate_queue_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(rows, 1):
        item_id = str(item.get("id") or f"row_{index}")
        missing = QUEUE_REQUIRED - set(item)
        if missing:
            errors.append(f"queue row {item_id}: missing {', '.join(sorted(missing))}")
        if item_id in seen:
            errors.append(f"queue row {item_id}: duplicate id")
        seen.add(item_id)
        if item.get("sanitized") is not True:
            errors.append(f"queue row {item_id}: sanitized is not true")
        text = flatten_text({"input": item.get("input"), "label": item.get("label"), "notes": item.get("notes")})
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"queue row {item_id}: possible secret-like content matched {pattern.pattern}")
    return errors


def queue_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "total": len(rows),
        "lanes": dict(Counter(str(r.get("lane", "missing")) for r in rows)),
        "statuses": dict(Counter(str(r.get("status", "missing")) for r in rows)),
        "splits": dict(Counter(str(r.get("split", "missing")) for r in rows)),
        "sources": dict(Counter(str(r.get("source", "missing")) for r in rows)),
    }


def row_key(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("id") or row.get("example_id") or row.get("decision_id") or f"row_{fallback}")


def changed_examples(
    current: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    max_examples: int,
    include_raw: bool,
) -> dict[str, Any]:
    current_by_id = {row_key(row, i): row for i, row in enumerate(current, 1)}
    baseline_by_id = {row_key(row, i): row for i, row in enumerate(baseline, 1)}
    added = [k for k in current_by_id if k not in baseline_by_id]
    removed = [k for k in baseline_by_id if k not in current_by_id]
    modified = [
        k
        for k in current_by_id.keys() & baseline_by_id.keys()
        if stable_digest(current_by_id[k]) != stable_digest(baseline_by_id[k])
    ]
    return {
        "counts": {"added": len(added), "removed": len(removed), "modified": len(modified)},
        "added_examples": [compact_example(current_by_id[k], include_raw=include_raw) for k in added[:max_examples]],
        "removed_ids": removed[:max_examples],
        "modified_examples": [compact_example(current_by_id[k], include_raw=include_raw) for k in modified[:max_examples]],
    }


def coerce_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        try:
            num = float(cleaned)
        except ValueError:
            return None
        return num / 100.0 if num > 1 else num
    return None


def confidence_from_item(item: dict[str, Any]) -> tuple[float | None, str | None]:
    candidates: list[tuple[str, Any]] = [
        ("confidence", item.get("confidence")),
        ("score", item.get("score")),
    ]
    for parent in ("draft", "local_draft", "local", "model_output", "label"):
        obj = item.get(parent)
        if isinstance(obj, dict):
            for key in ("confidence", "score", "probability"):
                candidates.append((f"{parent}.{key}", obj.get(key)))
    for key, value in candidates:
        conf = coerce_confidence(value)
        if conf is not None:
            return conf, key
    return None, None


def low_confidence_examples(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    max_examples: int,
    include_raw: bool,
) -> dict[str, Any]:
    low: list[dict[str, Any]] = []
    seen_with_confidence = 0
    for row in rows:
        conf, source_key = confidence_from_item(row)
        if conf is None:
            continue
        seen_with_confidence += 1
        source_text = " ".join(str(row.get(k, "")) for k in ("source", "model", "lane", "status")).lower()
        has_local_draft = "local" in source_text or "draft" in row or "local_draft" in row
        if has_local_draft and conf < threshold:
            item = compact_example(row, include_raw=include_raw)
            item["confidence"] = conf
            item["confidence_source"] = source_key
            low.append(item)
    return {"threshold": threshold, "rows_with_confidence": seen_with_confidence, "count": len(low), "examples": low[:max_examples]}


def value_at_path(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def queue_disagreements(rows: list[dict[str, Any]], *, max_examples: int, include_raw: bool) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    pairs = [
        ("local_label", "human_label"),
        ("local_label", "frontier_label"),
        ("draft.label", "label"),
        ("draft.decision", "label.decision"),
        ("local_draft.label", "label"),
        ("local_draft.decision", "label.decision"),
    ]
    for row in rows:
        reasons: list[str] = []
        if row.get("disagreement") or row.get("disagreements"):
            reasons.append("explicit_disagreement_flag")
        for left_path, right_path in pairs:
            left = value_at_path(row, left_path)
            right = value_at_path(row, right_path)
            if left is not None and right is not None and stable_digest(left) != stable_digest(right):
                reasons.append(f"{left_path}!={right_path}")
        if reasons:
            item = compact_example(row, include_raw=include_raw)
            item["disagreement_reasons"] = sorted(set(reasons))
            candidates.append(item)
    return {"count": len(candidates), "examples": candidates[:max_examples]}


def dataset_summary(dataset_dir: Path | None) -> dict[str, Any] | None:
    if dataset_dir is None:
        return None
    if not dataset_dir.exists():
        return {"path": str(dataset_dir), "exists": False, "files": {}, "errors": [f"missing dataset dir: {dataset_dir}"]}
    files: dict[str, Any] = {}
    errors: list[str] = []
    for file in sorted(dataset_dir.glob("*.jsonl")):
        count = 0
        bad = 0
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                count += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
        files[file.name] = {"rows": count, "invalid_json_rows": bad, "bytes": file.stat().st_size, "path": str(file)}
        if bad:
            errors.append(f"{file}: {bad} invalid JSON row(s)")
    return {"path": str(dataset_dir), "exists": True, "files": files, "errors": errors}


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"could not parse {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"{path}: top-level JSON is not an object"
    return data, None


def report_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("errors", "error", "parse_error"):
        value = report.get(key)
        if isinstance(value, list):
            errors.extend(str(v) for v in value if v)
        elif value:
            errors.append(str(value))
    rows = report.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                for key in ("parse_error", "raw_parse_error", "error"):
                    if row.get(key):
                        errors.append(f"row {row.get('index', row.get('id', '?'))}: {row[key]}")
    return errors


def report_disagreements(report: dict[str, Any], *, max_examples: int, include_raw: bool) -> list[dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        reasons: list[str] = []
        for key, ok_key in (
            ("lane", "lane_ok"),
            ("priority", "priority_ok"),
            ("worker", "worker_ok"),
            ("decision", "decision_ok"),
            ("schema", "schema_ok"),
        ):
            if row.get(ok_key) is False:
                reasons.append(f"{key}_mismatch")
        if row.get("expected") is not None and row.get("parsed") is not None and stable_digest(row["expected"]) != stable_digest(row["parsed"]):
            reasons.append("expected_vs_parsed")
        if reasons:
            out.append({"index": row.get("index"), "reasons": sorted(set(reasons)), "row": compact_obj(row, include_raw=include_raw)})
    return out[:max_examples]


def eval_summaries(paths: list[Path], *, max_examples: int, include_raw: bool) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    disagreements: list[dict[str, Any]] = []
    for path in paths:
        report, err = load_json(path)
        if err:
            errors.append(err)
            continue
        assert report is not None
        raw_metrics = report.get("metrics")
        metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
        raw_rows = report.get("rows")
        rows: list[Any] = raw_rows if isinstance(raw_rows, list) else []
        summary = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "base_model": report.get("base_model"),
            "dataset_file": report.get("dataset_file"),
            "split": report.get("split"),
            "num_examples": report.get("num_examples") or metrics.get("num_examples") or len(rows),
            "metrics": metrics,
            "gate_report": report.get("gate_report") or report.get("gates"),
        }
        summaries.append(summary)
        errors.extend(report_errors(report))
        disagreements.extend(report_disagreements(report, max_examples=max_examples, include_raw=include_raw))
    return summaries, errors, disagreements[:max_examples]


def artifact_pointers(paths: list[Path]) -> list[dict[str, Any]]:
    pointers: list[dict[str, Any]] = []
    for path in paths:
        pointers.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "kind": "directory" if path.exists() and path.is_dir() else "file",
                "note": "Pointer only; contents intentionally omitted from packet.",
            }
        )
    return pointers


def frontier_required_checklist(lane: str, packet: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = packet.get("eval_reports", [])
    metric_gate_risk = False
    for report in metrics:
        m = report.get("metrics") or {}
        if any(str(k).endswith("misses") and v for k, v in m.items()):
            metric_gate_risk = True
        for key in ("strict_route_accuracy", "strict_schema_valid_rate", "lane_accuracy"):
            value = m.get(key)
            if isinstance(value, (int, float)) and value < 0.98:
                metric_gate_risk = True
    checks = [
        {
            "item": "Training/adapters, production routing, config/secrets, publishing, or irreversible state changes are in scope.",
            "frontier_required": any(word in json.dumps(packet).lower() for word in ["adapter", "production", "secret", "credential", "promotion", "routing_change"]),
        },
        {
            "item": "Local draft confidence is below threshold or missing for examples being promoted.",
            "frontier_required": packet["low_confidence_local_drafts"]["count"] > 0,
        },
        {
            "item": "Local/frontier/human labels disagree or eval rows disagree with expected outputs.",
            "frontier_required": packet["disagreements"]["count"] > 0,
        },
        {
            "item": "Schema, parse, sanitization, or secret-like-content errors are present.",
            "frontier_required": packet["errors"]["count"] > 0,
        },
        {
            "item": "Eval metrics miss strict schema/route/safety gates.",
            "frontier_required": metric_gate_risk,
        },
        {
            "item": "The decision is subjective/taste-based or requires operator acceptance.",
            "frontier_required": lane == "gaming-pipeline" and packet["queue"].get("statuses", {}).get("needs_label", 0) > 0,
        },
    ]
    return checks


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    current_rows, queue_read_errors = read_jsonl(args.queue)
    baseline_rows: list[dict[str, Any]] = []
    baseline_errors: list[str] = []
    if args.baseline_queue:
        baseline_rows, baseline_errors = read_jsonl(args.baseline_queue)

    queue_errors = queue_read_errors + validate_queue_rows(current_rows)
    changes = (
        changed_examples(current_rows, baseline_rows, max_examples=args.max_examples, include_raw=args.include_raw_snippets)
        if args.baseline_queue
        else {"counts": {"added": 0, "removed": 0, "modified": 0}, "note": "No --baseline-queue supplied."}
    )
    dataset = dataset_summary(args.dataset_dir)
    evals, eval_errors, eval_disagreements = eval_summaries(
        args.eval_report,
        max_examples=args.max_examples,
        include_raw=args.include_raw_snippets,
    )
    queue_disagree = queue_disagreements(current_rows, max_examples=args.max_examples, include_raw=args.include_raw_snippets)
    all_disagreements = {
        "count": queue_disagree["count"] + len(eval_disagreements),
        "queue_examples": queue_disagree["examples"],
        "eval_examples": eval_disagreements,
    }
    all_errors = queue_errors + baseline_errors + eval_errors + ((dataset or {}).get("errors", []) if dataset else [])
    packet_id = args.packet_id or f"{args.lane}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{stable_digest({'queue': str(args.queue), 'eval': [str(p) for p in args.eval_report]}, 8)}"
    packet: dict[str, Any] = {
        "packet_id": packet_id,
        "generated_at": utc_now(),
        "lane": args.lane,
        "compact_policy": {
            "raw_transcripts_logs_embedded": bool(args.include_raw_snippets),
            "default_behavior": "omit raw transcript/log/message dumps; include counts, short compact examples, digests, and pointers",
            "max_examples_per_section": args.max_examples,
        },
        "queue": queue_summary(args.queue, current_rows),
        "dataset": dataset,
        "changed_examples": changes,
        "low_confidence_local_drafts": low_confidence_examples(
            current_rows,
            threshold=args.low_confidence_threshold,
            max_examples=args.max_examples,
            include_raw=args.include_raw_snippets,
        ),
        "disagreements": all_disagreements,
        "errors": {"count": len(all_errors), "items": all_errors[: args.max_examples * 2]},
        "eval_reports": evals,
        "raw_artifact_pointers": artifact_pointers([args.queue] + ([args.baseline_queue] if args.baseline_queue else []) + (args.eval_report or []) + (args.artifact or [])),
    }
    packet["frontier_review_checklist"] = frontier_required_checklist(args.lane, packet)
    packet["frontier_review_required"] = any(item["frontier_required"] for item in packet["frontier_review_checklist"])
    return packet


def fmt_counter(counter: dict[str, Any] | None) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def render_markdown(packet: dict[str, Any]) -> str:
    queue = packet["queue"]
    dataset = packet.get("dataset") or {}
    dataset_lines: list[str] = []
    for name, info in (dataset.get("files") or {}).items():
        dataset_lines.append(f"- `{name}`: {info['rows']} rows, {info['invalid_json_rows']} invalid JSON rows, {info['bytes']} bytes")
    if not dataset_lines:
        dataset_lines.append("- No dataset directory supplied or no JSONL splits found.")

    eval_lines: list[str] = []
    for report in packet.get("eval_reports", []):
        metrics = report.get("metrics") or {}
        compact_metrics = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items())[:12]) or "no metrics"
        eval_lines.append(f"- `{report['path']}`: examples={report.get('num_examples')}; {compact_metrics}")
    if not eval_lines:
        eval_lines.append("- No eval reports supplied.")

    checklist = []
    for item in packet["frontier_review_checklist"]:
        mark = "x" if item["frontier_required"] else " "
        checklist.append(f"- [{mark}] {item['item']}")

    pointers = [f"- `{p['path']}` ({p['kind']}, exists={p['exists']}, bytes={p['bytes']})" for p in packet["raw_artifact_pointers"]]

    md = f"""# Frontier Review Packet: {packet['packet_id']}

Generated: `{packet['generated_at']}`  
Lane: `{packet['lane']}`  
Frontier review required: **{packet['frontier_review_required']}**

## Compact packet policy

- Raw transcript/log/message dumps embedded: `{packet['compact_policy']['raw_transcripts_logs_embedded']}`
- Default: {packet['compact_policy']['default_behavior']}.
- Max examples per section: {packet['compact_policy']['max_examples_per_section']}

## Queue summary

- Queue: `{queue['path']}`
- Total rows: {queue['total']}
- Lanes: {fmt_counter(queue.get('lanes'))}
- Statuses: {fmt_counter(queue.get('statuses'))}
- Splits: {fmt_counter(queue.get('splits'))}
- Sources: {fmt_counter(queue.get('sources'))}

## Dataset summary

{chr(10).join(dataset_lines)}

## Changed examples

- Added: {packet['changed_examples']['counts']['added']}
- Removed: {packet['changed_examples']['counts']['removed']}
- Modified: {packet['changed_examples']['counts']['modified']}

See JSON packet for compact changed-example digests/samples.

## Low-confidence local drafts

- Threshold: {packet['low_confidence_local_drafts']['threshold']}
- Rows with confidence: {packet['low_confidence_local_drafts']['rows_with_confidence']}
- Low-confidence count: {packet['low_confidence_local_drafts']['count']}

## Disagreements and errors

- Disagreements: {packet['disagreements']['count']}
- Errors: {packet['errors']['count']}

## Eval metrics

{chr(10).join(eval_lines)}

## Frontier review checklist

{chr(10).join(checklist)}

## Raw artifact pointers

{chr(10).join(pointers)}
"""
    return md


def write_outputs(packet: dict[str, Any], out_dir: Path, max_markdown_bytes: int) -> tuple[Path, Path, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{packet['packet_id']}.json"
    md_path = out_dir / f"{packet['packet_id']}.md"
    json_path.write_text(json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(packet)
    warnings: list[str] = []
    md_bytes = len(md.encode("utf-8"))
    if md_bytes > max_markdown_bytes:
        warnings.append(f"markdown packet is {md_bytes} bytes, above budget {max_markdown_bytes}")
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path, warnings


def main() -> int:
    args = parse_args()
    packet = build_packet(args)
    json_path, md_path, warnings = write_outputs(packet, args.out_dir, args.max_markdown_bytes)
    result = {
        "packet_id": packet["packet_id"],
        "json": str(json_path),
        "markdown": str(md_path),
        "frontier_review_required": packet["frontier_review_required"],
        "errors": packet["errors"]["count"],
        "warnings": warnings,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not warnings else 2


if __name__ == "__main__":
    raise SystemExit(main())
