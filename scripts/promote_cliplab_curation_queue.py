#!/usr/bin/env python3
"""Promote reviewed ClipLab curation queue rows into the ClipLab Edit Director dataset.

Input queue rows are expected to come from import_cliplab_learning_queue.py and use
AI Training's canonical curation queue schema. Output rows use the chat-format
schema consumed by train_cliplab_edit_director.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cliplab_edit_director_common import (
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    compact_json,
    default_dataset_dir,
    validate_dataset_dir,
    write_json,
    write_jsonl,
)

DEFAULT_QUEUE = Path("/mnt/s/AITraining/curation/queue.jsonl")
LANE = "cliplab-edit-director"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(item)
    return rows


def dataset_split(queue_split: str) -> str:
    if queue_split == "holdout":
        return "holdout_real_runs"
    if queue_split in {"train", "eval", "holdout_real_runs"}:
        return queue_split
    return "train"


def row_from_queue_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("lane") != LANE:
        return None
    if item.get("status") != "reviewed":
        return None
    if item.get("sanitized") is not True:
        return None
    label = item.get("label")
    input_payload = item.get("input")
    if not isinstance(label, dict) or not isinstance(input_payload, dict):
        return None

    raw_notes = item.get("notes")
    notes: dict[str, Any] = raw_notes if isinstance(raw_notes, dict) else {}
    run_id = str(notes.get("source_run_id_hash") or input_payload.get("run_id_hash") or item.get("id") or "")
    clip_id = str(notes.get("source_clip_id_hash") or input_payload.get("clip_id_hash") or item.get("id") or "")
    split = dataset_split(str(item.get("split") or "train"))
    label_clip = (label.get("clips") or [{}])[0] if isinstance(label.get("clips"), list) else {}
    has_trim = label_clip.get("trim_start_sec") is not None and label_clip.get("trim_end_sec") is not None
    row_id = str(item.get("id") or f"cliplab-learning-{run_id[:12]}-{clip_id[:12]}").replace("cliplab-learning-", "")

    return {
        "id": f"cliplab-edit-director-{row_id}",
        "run_id": run_id,
        "clip_id": clip_id,
        "split": split,
        "source": str(item.get("source") or "curation_queue"),
        "schema_version": SCHEMA_VERSION,
        "has_human_label": True,
        "has_human_trim_label": bool(has_trim),
        "label_source": "learning_contribution",
        "feedback_run_id": run_id,
        "feedback_timestamp": "",
        "feedback_match": "learning_contribution_queue",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": compact_json(input_payload)},
            {"role": "assistant", "content": compact_json(label)},
        ],
    }


def build_dataset_rows(queue: Path) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(queue):
        row = row_from_queue_item(item)
        if row:
            by_id[row["id"]] = row
    splits = {"train": [], "eval": [], "holdout_real_runs": []}
    for row in sorted(by_id.values(), key=lambda r: r["id"]):
        splits[dataset_split(str(row.get("split")))].append(row)
    return splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = build_dataset_rows(args.queue)
    counts = {name: len(rows) for name, rows in splits.items()}
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "train.jsonl", splits["train"])
        write_jsonl(args.output_dir / "eval.jsonl", splits["eval"])
        write_jsonl(args.output_dir / "holdout_real_runs.jsonl", splits["holdout_real_runs"])
        unlabeled_path = args.output_dir / "unlabeled_candidates.jsonl"
        if not unlabeled_path.exists():
            write_jsonl(unlabeled_path, [])
        validation = validate_dataset_dir(args.output_dir)
        counts = dict(validation.get("counts", counts))
        manifest = {
            "name": "cliplab-edit-director-v0.1",
            "schema_version": SCHEMA_VERSION,
            "source_queue": str(args.queue),
            "counts": counts,
            "decisions": validation.get("decisions", {}),
            "unlabeled_statuses": validation.get("unlabeled_statuses", {}),
            "validation": validation,
            "privacy": {
                "raw_video_uploaded": False,
                "raw_paths_in_messages": False,
                "source": "sanitized ClipLab learning-contribution queue",
                "unlabeled_rows_used_for_training": False,
            },
        }
        write_json(args.output_dir / "MANIFEST.json", manifest)
    else:
        validation = {"dry_run": True, "errors": []}

    decision_counts: Counter[str] = Counter()
    for rows in splits.values():
        for row in rows:
            label = json.loads(row["messages"][2]["content"])
            for clip in label.get("clips", []):
                decision_counts[str(clip.get("decision"))] += 1

    print(json.dumps({
        "queue": str(args.queue),
        "output_dir": str(args.output_dir),
        "counts": counts,
        "decisions": dict(decision_counts),
        "validation_errors": validation.get("errors", []),
        "dry_run": args.dry_run,
    }, indent=2))
    return 1 if validation.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
