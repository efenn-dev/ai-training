#!/usr/bin/env python3
"""Import ClipLab desktop learning contributions into the canonical AI-training curation queue.

The ClipLab app stores JSON-only alpha learning payloads in Electron userData:
  C:/Users/<user>/AppData/Roaming/cliplabpro-dev/learning-contributions/queue.json

AI Training watches the canonical curation queue:
  /mnt/s/AITraining/curation/queue.jsonl

This bridge keeps the privacy boundary intact: it imports only the sanitized hashes,
structured settings, scores, timing deltas, decisions, and reason chips already present
in the learning-contribution payload. It never reads/copies video, audio, transcripts,
filenames, or local media paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
import os
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path(os.environ.get(
    "CLIPLAB_LEARNING_QUEUE",
    "/mnt/c/Users/USERNAME/AppData/Roaming/cliplabpro-dev/learning-contributions/queue.json",
))
DEFAULT_QUEUE = Path("/mnt/s/AITraining/curation/queue.jsonl")
SCHEMA_VERSION = "cliplab_edit_director.v0.1"
LANE = "cliplab-edit-director"
SOURCE = "cliplab_learning_contributions"


def stable_hash(value: Any, length: int = 16) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"source queue does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"source queue must be a JSON object: {path}")
    return data


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("id"):
                ids.add(str(item["id"]))
    return ids


def deterministic_split(run_id_hash: str, clip_id_hash: str) -> str:
    bucket = int(stable_hash(f"{run_id_hash}:{clip_id_hash}", 8), 16) / 0xFFFFFFFF
    if bucket < 0.80:
        return "train"
    if bucket < 0.95:
        return "eval"
    return "holdout"


def clean_list(value: Any, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip().lower().replace("-", "_")
        text = "".join(ch for ch in text if ch.isalnum() or ch in "_:." )[:80]
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return round(out, 3)


def assistant_label(run_id_hash: str, clip: dict[str, Any]) -> dict[str, Any]:
    decision_raw = str(clip.get("decision") or "unreviewed").lower()
    decision = "keep" if decision_raw == "accepted" else "reject"
    reasons = clean_list(clip.get("reason_chips"))
    if not reasons:
        reasons = ["accepted_by_user" if decision == "keep" else "rejected_by_user"]

    label_clip: dict[str, Any] = {
        "clip_id": str(clip.get("clip_id_hash") or ""),
        "decision": decision,
        "reasons": reasons,
        "platform_fit": "ok" if decision == "keep" else "weak",
        "safety_notes": clean_list(clip.get("risk_flags")),
        "metadata_title": "",
        "metadata_notes": "",
    }
    start = num(clip.get("final_start_sec"))
    end = num(clip.get("final_end_sec"))
    if start is not None:
        label_clip["trim_start_sec"] = start
    if end is not None:
        label_clip["trim_end_sec"] = max(end, start if start is not None else end)

    return {
        "schema_version": SCHEMA_VERSION,
        "clips": [label_clip],
        "run_notes": f"Imported from ClipLab learning contribution run {run_id_hash[:12]}",
        "safety": {"child_safe": True, "sponsor_safe": True},
    }


def row_from_clip(payload: dict[str, Any], clip: dict[str, Any]) -> dict[str, Any] | None:
    run_id_hash = str(payload.get("run_id_hash") or "").lower()
    clip_id_hash = str(clip.get("clip_id_hash") or "").lower()
    if len(run_id_hash) < 16 or len(clip_id_hash) < 16:
        return None
    decision = str(clip.get("decision") or "").lower()
    if decision not in {"accepted", "rejected"}:
        return None

    row_id = f"cliplab-learning-{stable_hash({'run': run_id_hash, 'clip': clip_id_hash}, 20)}"
    input_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id_hash": run_id_hash,
        "clip_id_hash": clip_id_hash,
        "content_type": payload.get("content_type"),
        "run_mode": payload.get("run_mode"),
        "app_version": payload.get("app_version"),
        "settings_summary": payload.get("settings") or {},
        "run_metrics": payload.get("run_metrics") or {},
        "model_versions": payload.get("model_versions") or {},
        "candidate_clip": {
            "queue_index": clip.get("queue_index"),
            "suggested_start_sec": num(clip.get("suggested_start_sec")),
            "suggested_end_sec": num(clip.get("suggested_end_sec")),
            "final_start_sec": num(clip.get("final_start_sec")),
            "final_end_sec": num(clip.get("final_end_sec")),
            "start_delta_sec": num(clip.get("start_delta_sec")),
            "end_delta_sec": num(clip.get("end_delta_sec")),
            "duration_delta_sec": num(clip.get("duration_delta_sec")),
            "ai_scores": clip.get("ai_scores") or {},
            "risk_flags": clean_list(clip.get("risk_flags")),
            "reason_chips": clean_list(clip.get("reason_chips")),
            "keyword_count": clip.get("keyword_count"),
            "score_reason_count": clip.get("score_reason_count"),
        },
        "privacy": {
            "raw_media_included": False,
            "raw_paths_included": False,
            "transcripts_included": False,
            "source_payload": "ClipLab Electron learning-contributions queue",
        },
    }
    return {
        "id": row_id,
        "lane": LANE,
        "source": SOURCE,
        "status": "reviewed",
        "sanitized": True,
        "split": deterministic_split(run_id_hash, clip_id_hash),
        "input": input_payload,
        "label": assistant_label(run_id_hash, clip),
        "notes": {
            "importer": "import_cliplab_learning_queue.py",
            "source_run_id_hash": run_id_hash,
            "source_clip_id_hash": clip_id_hash,
        },
    }


def build_rows(source: Path) -> list[dict[str, Any]]:
    data = load_json(source)
    rows: list[dict[str, Any]] = []
    pending = data.get("pending") or {}
    if not isinstance(pending, dict):
        raise SystemExit("source queue pending field must be an object")
    for payload in pending.values():
        if not isinstance(payload, dict):
            continue
        for clip in payload.get("clips") or []:
            if isinstance(clip, dict):
                row = row_from_clip(payload, clip)
                if row:
                    rows.append(row)
    return rows


def source_summary(source: Path) -> dict[str, Any]:
    """Return visibility counts for the ClipLab source queue without changing import policy."""
    data = load_json(source)
    pending = data.get("pending") or {}
    if not isinstance(pending, dict):
        raise SystemExit("source queue pending field must be an object")
    payloads = 0
    clips = 0
    decisions: Counter[str] = Counter()
    importable = 0
    for payload in pending.values():
        if not isinstance(payload, dict):
            continue
        payloads += 1
        for clip in payload.get("clips") or []:
            if not isinstance(clip, dict):
                continue
            clips += 1
            decision = str(clip.get("decision") or "missing").lower()
            decisions[decision] += 1
            if decision in {"accepted", "rejected"}:
                importable += 1
    return {
        "pending_payloads": payloads,
        "source_clips": clips,
        "source_clip_decisions": dict(decisions),
        "importable_reviewed_clips": importable,
        "skipped_unlabeled_clips": clips - importable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args.source)
    existing = load_existing_ids(args.queue)
    new_rows = [row for row in rows if row["id"] not in existing]

    if not args.dry_run and new_rows:
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        with args.queue.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = source_summary(args.source)
    print(json.dumps({
        "source": str(args.source),
        "queue": str(args.queue),
        **summary,
        "source_rows": len(rows),
        "existing_matches": len(rows) - len(new_rows),
        "appended": 0 if args.dry_run else len(new_rows),
        "dry_run_new_rows": len(new_rows) if args.dry_run else 0,
        "dry_run": args.dry_run,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
