#!/usr/bin/env python3
"""Curate ClipLab Edit Director v0.1 examples from local artifacts and feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cliplab_edit_director_common import (
    DEFAULT_DATASET_NAME,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    basename_ref,
    clean_list,
    clean_text,
    compact_json,
    default_dataset_dir,
    default_pipeline_root,
    detect_sensitive_text,
    read_json,
    schema_document,
    stable_hash,
    to_float,
    utc_now,
    validate_dataset_dir,
    validate_edit_director_response,
    write_json,
    write_jsonl,
)

log = logging.getLogger(__name__)

FEEDBACK_DECISIONS = {"accepted", "rejected"}
SCORE_KEYS = (
    "effective_score",
    "combined_score",
    "score",
    "audio_score",
    "visual_score",
    "motion_score",
    "vlm_score",
    "priority_score",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=default_pipeline_root())
    parser.add_argument("--output-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--feedback-dir", type=Path, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=None,
        help="Directory or manifest JSON file to scan. Can be passed multiple times.",
    )
    parser.add_argument("--train-pct", type=float, default=0.70)
    parser.add_argument("--eval-pct", type=float, default=0.15)
    parser.add_argument("--holdout-pct", type=float, default=0.15)
    parser.add_argument("--target-platform", default="youtube")
    parser.add_argument("--family-safe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-without-labels", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_json_safe(path: Path, default: Any) -> Any:
    """Read JSON with a warning on malformed files."""
    try:
        return read_json(path, default)
    except Exception as exc:  # noqa: BLE001
        log.warning("Skipping malformed JSON %s: %s", path, exc)
        return default


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping malformed lines with warnings."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSONL %s:%d: %s", path, line_no, exc)
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def safe_clip_feedback_key(value: Any) -> str:
    """Match ClipForge's electron-side clip feedback key generation."""
    raw = str(value or "").replace("\x00", "").strip()
    if not raw:
        return ""
    clean = raw.replace("\\", "/").split("/")[-1]
    clean = re.sub(r"[^A-Za-z0-9._:-]", "_", clean)
    clean = re.sub(r"_+", "_", clean)[:120]
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{clean}_{digest}" if clean else f"clip_{digest}"


def path_hash(value: Any) -> str:
    """Return the same path hash used in prompt payloads."""
    raw = str(value or "").replace("\x00", "").strip()
    return stable_hash(raw) if raw else ""


def feedback_tokens(key: str, item: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return strong and weak match tokens for one feedback item."""
    strong = {key}
    weak = {key}
    for field in ("clip_path", "clip_id", "title"):
        value = str(item.get(field) or "").strip()
        if not value:
            continue
        weak.add(value)
        weak.add(value.replace("\\", "/").split("/")[-1])
        safe = safe_clip_feedback_key(value)
        if safe:
            weak.add(safe)
            if field == "clip_path":
                strong.add(safe)
        if field == "clip_path":
            strong.add(path_hash(value))
    return {t for t in strong if t}, {t for t in weak if t}


def candidate_tokens(candidate: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return strong and weak match tokens for one candidate."""
    raw = candidate.get("_raw", {})
    strong: set[str] = set()
    weak: set[str] = set()
    for value in (
        raw.get("clip_path"),
        raw.get("clip_id"),
        raw.get("title"),
        candidate.get("clip_id"),
        candidate.get("clip_basename"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        weak.add(text)
        weak.add(text.replace("\\", "/").split("/")[-1])
        safe = safe_clip_feedback_key(text)
        if safe:
            weak.add(safe)
            if text == raw.get("clip_path"):
                strong.add(safe)
        if text == raw.get("clip_path"):
            strong.add(path_hash(text))
    return {t for t in strong if t}, {t for t in weak if t}


def load_feedback_records(feedback_dir: Path) -> list[dict[str, Any]]:
    """Load feedback JSON files written by ClipForge."""
    if not feedback_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(feedback_dir.glob("*.json")):
        data = read_json_safe(path, {})
        if not isinstance(data, dict):
            continue
        data["_path"] = path
        data["run_id"] = str(data.get("run_id") or path.stem)
        records.append(data)
    return records


def recent_feedback_summary(records: list[dict[str, Any]], current_run_id: str, limit: int = 5) -> dict[str, Any]:
    """Summarize prior feedback without leaking the current clip label."""
    prior = [r for r in records if str(r.get("run_id")) != str(current_run_id)]
    prior = sorted(prior, key=lambda r: str(r.get("timestamp") or ""))[-limit:]
    ratings: list[int] = []
    tag_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    notes: list[str] = []
    for record in prior:
        try:
            rating = int(record.get("rating") or 0)
        except (TypeError, ValueError):
            rating = 0
        if rating > 0:
            ratings.append(rating)
        tag_counts.update(clean_list(record.get("tags"), limit=20, item_limit=80))
        note = clean_text(record.get("notes"), 180)
        if note:
            notes.append(note)
        clip_feedback = record.get("clip_feedback")
        if not isinstance(clip_feedback, dict):
            continue
        for item in clip_feedback.values():
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision") or "").lower()
            if decision in FEEDBACK_DECISIONS:
                decision_counts[decision] += 1
            reason_counts.update(clean_list(item.get("reasons"), limit=8, item_limit=80))
    return {
        "prior_runs": len(prior),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
        "tag_counts": dict(tag_counts.most_common(8)),
        "clip_decisions": dict(decision_counts),
        "clip_reason_counts": dict(reason_counts.most_common(8)),
        "notes": notes[-3:],
    }


def find_feedback_match(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    """Return the feedback record/item matching a candidate, if any."""
    cand_strong, cand_weak = candidate_tokens(candidate)
    candidate_run = str(candidate.get("run_id") or "")
    for record in records:
        clip_feedback = record.get("clip_feedback")
        if not isinstance(clip_feedback, dict):
            continue
        same_run = str(record.get("run_id") or "") == candidate_run
        for key, item in clip_feedback.items():
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision") or "").strip().lower()
            if decision not in FEEDBACK_DECISIONS:
                continue
            strong, weak = feedback_tokens(str(key), item)
            if cand_strong & strong:
                return record, item, "strong_path_or_key"
            if same_run and cand_weak & weak:
                return record, item, "same_run_token"
    return None


def vl_by_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Group VLM timeline entries by clip index."""
    data = read_json_safe(path, [])
    if not isinstance(data, list):
        return {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = int(to_float(item.get("clip_index"), -1))
        if idx < 0:
            continue
        grouped.setdefault(idx, []).append(item)
    return grouped


def scoring_by_index(path: Path) -> dict[int, dict[str, Any]]:
    """Load scoring trace rows keyed by candidate index."""
    out: dict[int, dict[str, Any]] = {}
    for item in load_jsonl_objects(path):
        idx = int(to_float(item.get("index"), -1))
        if idx >= 0:
            out[idx] = item
    return out


def artifact_refs(paths: list[Path]) -> list[dict[str, str]]:
    """Return privacy-safe artifact references for manifest/provenance."""
    refs: list[dict[str, str]] = []
    for path in paths:
        if path.exists():
            ref = basename_ref(path)
            refs.append({"file": ref["basename"], "path_hash": ref["path_hash"]})
    return refs


def compact_ai_review(value: Any) -> dict[str, Any]:
    """Keep the useful parts of aiReview without dumping huge text."""
    if not isinstance(value, dict):
        return {}
    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    return {
        "overallScore": to_float(value.get("overallScore"), 0.0),
        "scores": value.get("scores") if isinstance(value.get("scores"), dict) else {},
        "verdict": clean_text(value.get("verdict"), 40),
        "reason": clean_text(value.get("reason"), 260),
        "riskFlags": clean_list(value.get("riskFlags"), limit=12, item_limit=80),
        "evidence": {
            "transcriptExcerpt": clean_text(evidence.get("transcriptExcerpt"), 420),
            "dialogueContext": clean_text(evidence.get("dialogueContext"), 420),
            "visualSummary": clean_text(evidence.get("visualSummary"), 420),
            "eventTags": clean_list(evidence.get("eventTags"), limit=12, item_limit=80),
            "keywords": clean_list(evidence.get("keywords"), limit=16, item_limit=80),
            "scoreReasons": clean_list(evidence.get("scoreReasons"), limit=12, item_limit=140),
        },
    }


def compact_vl_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact VLM timeline snippets for prompt input."""
    out: list[dict[str, Any]] = []
    safe_entries = entries if isinstance(entries, list) else []
    for entry in safe_entries[:5]:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "timestamp_sec": round(to_float(entry.get("timestamp_sec")), 3),
                "window_end_sec": round(to_float(entry.get("window_end_sec")), 3),
                "source_start_sec": round(to_float(entry.get("source_start_sec")), 3),
                "source_end_sec": round(to_float(entry.get("source_end_sec")), 3),
                "description": clean_text(entry.get("description"), 280),
                "dialogue": clean_text(entry.get("dialogue"), 220),
            }
        )
    return out


def segment_clip_id(segment: dict[str, Any], fallback_index: int) -> str:
    """Return a stable clip id from segment metadata."""
    for key in ("clip_id", "uuid", "id"):
        value = clean_text(segment.get(key), 120)
        if value:
            return value
    clip_path = str(segment.get("clip_path") or segment.get("source_clip_path") or "").strip()
    if clip_path:
        return clean_text(clip_path.replace("\\", "/").split("/")[-1], 120)
    return f"clip_{fallback_index:03d}"


def build_candidate(
    *,
    root: Path,
    run_id: str,
    segment: dict[str, Any],
    index: int,
    content_type: str,
    target_platform: str,
    family_safe: bool,
    vl_entries: list[dict[str, Any]],
    scoring: dict[str, Any] | None,
    source_artifacts: list[dict[str, str]],
    feedback_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one privacy-safe prompt candidate."""
    clip_path = segment.get("clip_path") or segment.get("source_clip_path") or ""
    source_video = segment.get("source_video") or segment.get("source") or ""
    raw_events = segment.get("events") if isinstance(segment.get("events"), list) else []
    raw_semantic_labels = segment.get("semantic_labels") if isinstance(segment.get("semantic_labels"), list) else []
    clip_ref = basename_ref(clip_path)
    source_ref = basename_ref(source_video)
    clip_id = segment_clip_id(segment, index)
    start = to_float(segment.get("start_sec"), to_float(segment.get("trim_start_sec"), 0.0))
    end = to_float(segment.get("end_sec"), to_float(segment.get("trim_end_sec"), start))
    duration = to_float(segment.get("duration_sec"), max(0.0, end - start))
    scores = {key: round(to_float(segment.get(key)), 3) for key in SCORE_KEYS if segment.get(key) is not None}
    scoring_compact = {}
    if scoring:
        scoring_compact = {
            "action": clean_text(scoring.get("action"), 40),
            "reason": clean_text(scoring.get("reason"), 240),
            "assembled": bool(scoring.get("assembled")),
            "thresholds": {
                "keep": round(to_float(scoring.get("keep_threshold")), 3),
                "compress": round(to_float(scoring.get("compress_threshold")), 3),
                "boring": round(to_float(scoring.get("boring_threshold")), 3),
            },
            "score_reasons": clean_list(scoring.get("score_reasons"), limit=12, item_limit=140),
        }
    candidate = {
        "clip_id": clip_id,
        "index": index,
        "title": clean_text(segment.get("title") or segment.get("segment_title"), 160),
        "source_video": source_ref,
        "clip": clip_ref,
        "clip_basename": clip_ref["basename"],
        "window_sec": {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
        },
        "pipeline_action": clean_text(segment.get("action"), 40),
        "pipeline_reason": clean_text(segment.get("reason"), 260),
        "scores": scores,
        "score_reasons": clean_list(segment.get("score_reasons"), limit=12, item_limit=140),
        "description": clean_text(segment.get("description"), 360),
        "keywords": clean_list(segment.get("keywords"), limit=16, item_limit=80),
        "event_tags": clean_list([segment.get("event_tag"), *raw_events], limit=12, item_limit=80),
        "audio_tags": clean_list(segment.get("audio_tags"), limit=12, item_limit=80),
        "vision_facts_summary": clean_text(segment.get("vision_facts_summary"), 300),
        "ocr_text": clean_text(segment.get("ocr_text"), 220),
        "semantic_labels": [
            {
                "label": clean_text(item.get("label"), 80),
                "score": round(to_float(item.get("score")), 4),
            }
            for item in raw_semantic_labels[:8]
            if isinstance(item, dict)
        ],
        "game_parameters": {
            clean_text(k, 60): clean_text(v, 80)
            for k, v in (segment.get("game_parameters") or {}).items()
            if clean_text(k, 60)
        }
        if isinstance(segment.get("game_parameters"), dict)
        else {},
        "ai_review": compact_ai_review(segment.get("aiReview") or segment.get("ai_review")),
        "vlm_summaries": compact_vl_entries(vl_entries),
        "scoring_trace": scoring_compact,
    }
    return {
        "run_id": run_id,
        "clip_id": clip_id,
        "content_type": clean_text(content_type, 80),
        "target_platform": clean_text(target_platform, 80),
        "safety_mode": {
            "family_safe": bool(family_safe),
            "child_safe": bool(family_safe),
            "sponsor_safe": True,
        },
        "recent_user_feedback": recent_feedback_summary(feedback_records, run_id),
        "candidate_clips": [candidate],
        "source_artifacts": source_artifacts,
        "_raw": {
            "clip_path": str(clip_path or ""),
            "clip_id": clip_id,
            "title": str(segment.get("title") or ""),
            "root": str(root),
        },
    }


def label_from_feedback(candidate: dict[str, Any], item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Convert ClipForge feedback into strict assistant JSON."""
    decision = "keep" if str(item.get("decision")).lower() == "accepted" else "reject"
    clip = candidate["candidate_clips"][0]
    window = clip.get("window_sec") or {}
    trim_start = item.get("trim_start_sec")
    trim_end = item.get("trim_end_sec")
    trim = item.get("trim")
    if isinstance(trim, dict):
        trim_start = trim.get("start_sec", trim.get("start", trim_start))
        trim_end = trim.get("end_sec", trim.get("end", trim_end))
    has_human_trim = trim_start is not None and trim_end is not None
    start = to_float(trim_start, to_float(window.get("start"), 0.0))
    end = to_float(trim_end, to_float(window.get("end"), start))
    if end < start:
        end = start
    reasons = clean_list(item.get("reasons"), limit=8, item_limit=80)
    if not reasons:
        reasons = ["accepted_by_user" if decision == "keep" else "rejected_by_user"]
    note = clean_text(item.get("note"), 500)
    note_l = note.lower()
    child_safe = not any(word in note_l for word in ("nsfw", "explicit", "adult", "unsafe for kids"))
    sponsor_safe = not any(word in note_l for word in ("brand unsafe", "sponsor unsafe", "copyright strike"))
    assistant = {
        "schema_version": SCHEMA_VERSION,
        "clips": [
            {
                "clip_id": str(candidate["clip_id"]),
                "decision": decision,
                "trim_start_sec": round(start, 3),
                "trim_end_sec": round(end, 3),
                "reasons": reasons,
                "platform_fit": "ok" if decision == "keep" else "weak",
                "safety_notes": [],
                "metadata_title": clean_text(item.get("title") or clip.get("title"), 180),
                "metadata_notes": note,
            }
        ],
        "run_notes": "",
        "safety": {
            "child_safe": child_safe,
            "sponsor_safe": sponsor_safe,
        },
    }
    error = validate_edit_director_response(assistant)
    if error:
        raise ValueError(error)
    return assistant, has_human_trim


def split_for(run_id: str, clip_id: str, train_pct: float, eval_pct: float, holdout_pct: float) -> str:
    """Deterministically assign a labeled row to train/eval/holdout."""
    total = train_pct + eval_pct + holdout_pct
    if total <= 0:
        raise ValueError("split percentages must sum to a positive number")
    train_cut = train_pct / total
    eval_cut = train_cut + eval_pct / total
    bucket = int(stable_hash(f"{run_id}:{clip_id}", 8), 16) / 0xFFFFFFFF
    if bucket < train_cut:
        return "train"
    if bucket < eval_cut:
        return "eval"
    return "holdout_real_runs"


def row_from_candidate(
    candidate: dict[str, Any],
    assistant: dict[str, Any],
    split: str,
    match_kind: str,
    feedback_record: dict[str, Any],
    has_human_trim: bool,
) -> dict[str, Any]:
    """Build one training/eval row."""
    user_payload = {k: v for k, v in candidate.items() if k != "_raw"}
    row_id = stable_hash({"run_id": candidate["run_id"], "clip_id": candidate["clip_id"], "split": split}, 16)
    return {
        "id": f"cliplab-edit-director-{row_id}",
        "run_id": candidate["run_id"],
        "clip_id": candidate["clip_id"],
        "split": split,
        "source": "cliplab_local_artifacts",
        "schema_version": SCHEMA_VERSION,
        "has_human_label": True,
        "has_human_trim_label": bool(has_human_trim),
        "label_source": "clip_feedback",
        "feedback_run_id": str(feedback_record.get("run_id") or ""),
        "feedback_timestamp": str(feedback_record.get("timestamp") or ""),
        "feedback_match": match_kind,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": compact_json(user_payload)},
            {"role": "assistant", "content": compact_json(assistant)},
        ],
    }


def unlabeled_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Build one unlabeled curation candidate row."""
    user_payload = {k: v for k, v in candidate.items() if k != "_raw"}
    row_id = stable_hash({"run_id": candidate["run_id"], "clip_id": candidate["clip_id"]}, 16)
    return {
        "id": f"cliplab-edit-director-unlabeled-{row_id}",
        "run_id": candidate["run_id"],
        "clip_id": candidate["clip_id"],
        "status": "needs_label",
        "source": "cliplab_local_artifacts",
        "schema_version": SCHEMA_VERSION,
        "input": user_payload,
    }


def discover_artifact_roots(pipeline_root: Path, explicit_roots: list[Path] | None) -> list[Path]:
    """Resolve artifact roots to scan."""
    if explicit_roots:
        return explicit_roots
    roots: list[Path] = []
    processed = pipeline_root / "processed" / "highlights"
    if processed.exists():
        roots.append(processed)
    demo_runs = pipeline_root / "demo_runs"
    if demo_runs.exists():
        for child in sorted(demo_runs.iterdir()):
            if (child / "manifest.json").exists():
                roots.append(child / "manifest.json")
    return roots


def run_id_from_artifact(root: Path, orchestrator_path: Path | None, payload: dict[str, Any]) -> str:
    """Return a stable run id for an artifact root."""
    if isinstance(payload, dict) and payload.get("run_id"):
        return clean_text(payload.get("run_id"), 100)
    if orchestrator_path and orchestrator_path.name.startswith("orchestrator_result_"):
        suffix = orchestrator_path.stem.removeprefix("orchestrator_result_")
        if suffix and suffix != "latest":
            return clean_text(suffix, 100)
    return f"{root.stem}-{stable_hash(str(root), 8)}"


def segments_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert manifest clips into segment-like records."""
    clips = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(clips, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, clip in enumerate(clips):
        if not isinstance(clip, dict):
            continue
        rows.append(
            {
                "index": idx,
                "clip_id": clip.get("uuid") or clip.get("id"),
                "title": clip.get("name") or clip.get("title"),
                "clip_path": clip.get("src") or clip.get("path") or clip.get("source_clip_path"),
                "start_sec": clip.get("start") or clip.get("start_sec") or 0,
                "end_sec": clip.get("end") or clip.get("end_sec") or clip.get("duration") or 0,
                "duration_sec": clip.get("duration") or clip.get("duration_sec") or 0,
                "score": clip.get("score"),
                "description": clip.get("summary") or clip.get("description"),
                "reason": clip.get("reason") or clip.get("verdictReason"),
                "aiReview": clip.get("aiReview") or clip.get("ai_review"),
            }
        )
    return rows


def collect_candidates_from_root(
    root: Path,
    feedback_records: list[dict[str, Any]],
    target_platform: str,
    family_safe: bool,
) -> list[dict[str, Any]]:
    """Collect candidate clips from one local artifact root or manifest file."""
    root = Path(root)
    manifest_file = root if root.is_file() else root / "manifest.json"
    artifact_dir = root.parent if root.is_file() else root
    highlights = read_json_safe(artifact_dir / "highlights.json", {})
    edit_decisions = read_json_safe(artifact_dir / "edit_decisions.json", {})
    manifest = read_json_safe(manifest_file, {}) if manifest_file.exists() else {}
    orchestrator_files = sorted(artifact_dir.glob("orchestrator_result_*.json"))
    orchestrator_path = None
    if artifact_dir / "orchestrator_result_latest.json" in orchestrator_files:
        orchestrator_path = artifact_dir / "orchestrator_result_latest.json"
    elif orchestrator_files:
        orchestrator_path = orchestrator_files[-1]
    orchestrator = read_json_safe(orchestrator_path, {}) if orchestrator_path else {}
    run_id = run_id_from_artifact(artifact_dir, orchestrator_path, orchestrator or edit_decisions or manifest)
    content_json = read_json_safe(artifact_dir / "content_type.json", {})
    content_type = (
        clean_text(content_json.get("content_type"), 80)
        if isinstance(content_json, dict)
        else ""
    ) or clean_text(highlights.get("content_type") if isinstance(highlights, dict) else "", 80)
    source_artifacts = artifact_refs(
        [
            artifact_dir / "highlights.json",
            artifact_dir / "edit_decisions.json",
            artifact_dir / "vl_timeline.json",
            artifact_dir / "scoring_trace.jsonl",
            artifact_dir / "orchestrator_result_latest.json",
            manifest_file,
        ]
    )
    vl_lookup = vl_by_index(artifact_dir / "vl_timeline.json")
    scoring_lookup = scoring_by_index(artifact_dir / "scoring_trace.jsonl")
    segments = []
    if isinstance(edit_decisions, dict):
        segments = ((edit_decisions.get("decisions") or {}).get("segments") or [])
    if not segments and isinstance(orchestrator, dict):
        segments = ((orchestrator.get("decisions") or {}).get("segments") or [])
    if not segments and isinstance(manifest, dict):
        segments = segments_from_manifest(manifest)
    if not segments and isinstance(highlights, dict):
        segments = highlights.get("highlights") or []
    if not isinstance(segments, list):
        return []

    candidates: list[dict[str, Any]] = []
    for fallback_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        index = int(to_float(segment.get("index"), fallback_index))
        candidate = build_candidate(
            root=artifact_dir,
            run_id=run_id,
            segment=segment,
            index=index,
            content_type=content_type,
            target_platform=target_platform,
            family_safe=family_safe,
            vl_entries=vl_lookup.get(index, []),
            scoring=scoring_lookup.get(index),
            source_artifacts=source_artifacts,
            feedback_records=feedback_records,
        )
        for issue in detect_sensitive_text({k: v for k, v in candidate.items() if k != "_raw"}):
            log.warning("Dropping candidate %s/%s: %s", run_id, candidate["clip_id"], issue)
            break
        else:
            candidates.append(candidate)
    return candidates


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Build dataset artifacts and return a manifest."""
    feedback_dir = args.feedback_dir or (args.pipeline_root / "data" / "feedback")
    feedback_records = load_feedback_records(feedback_dir)
    roots = discover_artifact_roots(args.pipeline_root, args.artifact_root)
    candidates: list[dict[str, Any]] = []
    for root in roots:
        candidates.extend(
            collect_candidates_from_root(
                root,
                feedback_records,
                target_platform=args.target_platform,
                family_safe=args.family_safe,
            )
        )

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        raw_path = candidate.get("_raw", {}).get("clip_path", "")
        key = (str(candidate.get("run_id")), str(candidate.get("clip_id")), path_hash(raw_path))
        deduped.setdefault(key, candidate)
    candidates = list(deduped.values())

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "eval": [],
        "holdout_real_runs": [],
    }
    unlabeled: list[dict[str, Any]] = []
    dropped_labels: list[dict[str, str]] = []
    label_completeness: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()

    for candidate in candidates:
        match = find_feedback_match(candidate, feedback_records)
        if not match:
            unlabeled.append(unlabeled_row(candidate))
            continue
        record, item, match_kind = match
        try:
            assistant, has_human_trim = label_from_feedback(candidate, item)
        except Exception as exc:  # noqa: BLE001
            dropped_labels.append(
                {
                    "run_id": str(candidate.get("run_id")),
                    "clip_id": str(candidate.get("clip_id")),
                    "error": str(exc),
                }
            )
            continue
        split = split_for(candidate["run_id"], candidate["clip_id"], args.train_pct, args.eval_pct, args.holdout_pct)
        row = row_from_candidate(candidate, assistant, split, match_kind, record, has_human_trim)
        splits[split].append(row)
        match_counts[match_kind] += 1
        label_completeness["clip_decision"] += 1
        if has_human_trim:
            label_completeness["trim"] += 1
        if assistant["clips"][0]["reasons"]:
            label_completeness["reasons"] += 1

    manifest = {
        "name": DEFAULT_DATASET_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "pipeline_root": str(args.pipeline_root),
        "feedback_dir": str(feedback_dir),
        "artifact_roots": [str(root) for root in roots],
        "privacy": {
            "raw_video_uploaded": False,
            "raw_paths_in_messages": False,
            "unlabeled_rows_used_for_training": False,
        },
        "counts": {
            "candidates": len(candidates),
            "feedback_records": len(feedback_records),
            "train": len(splits["train"]),
            "eval": len(splits["eval"]),
            "holdout_real_runs": len(splits["holdout_real_runs"]),
            "unlabeled_candidates": len(unlabeled),
            "dropped_labels": len(dropped_labels),
        },
        "label_completeness": dict(label_completeness),
        "feedback_match_counts": dict(match_counts),
        "dropped_labels": dropped_labels[:50],
        "warnings": [],
    }
    if manifest["counts"]["train"] == 0:
        manifest["warnings"].append("No train rows: collect explicit AI Review accept/reject labels before training.")
    if manifest["counts"]["eval"] == 0:
        manifest["warnings"].append("No eval rows: promotion gates cannot run until labeled eval examples exist.")
    if label_completeness.get("trim", 0) == 0:
        manifest["warnings"].append("No human trim labels found; trim-boundary promotion gate will remain blocked.")

    if args.fail_without_labels and sum(len(rows) for rows in splits.values()) == 0:
        raise SystemExit("No labeled examples found.")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "schema.json", schema_document())
        write_jsonl(args.output_dir / "train.jsonl", splits["train"])
        write_jsonl(args.output_dir / "eval.jsonl", splits["eval"])
        write_jsonl(args.output_dir / "holdout_real_runs.jsonl", splits["holdout_real_runs"])
        write_jsonl(args.output_dir / "unlabeled_candidates.jsonl", unlabeled)
        validation = validate_dataset_dir(args.output_dir)
        manifest["validation"] = validation
        write_json(args.output_dir / "MANIFEST.json", manifest)
        readme = f"""# ClipLab Edit Director LoRA v0.1

This dataset lane is generated from local ClipLab artifacts and explicit AI Review feedback.

- Train rows: `{manifest["counts"]["train"]}`
- Eval rows: `{manifest["counts"]["eval"]}`
- Holdout real-run rows: `{manifest["counts"]["holdout_real_runs"]}`
- Unlabeled candidates parked for review: `{manifest["counts"]["unlabeled_candidates"]}`

Privacy rules:

- Raw video is not copied or uploaded.
- Prompt messages use basenames and hashes instead of full local paths.
- Transcript/dialogue fields are capped before writing.
- Rows without human labels are written to `unlabeled_candidates.jsonl` and are not used for training.

Useful commands:

```bash
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/curate_cliplab_edit_director_v01.py
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/train_cliplab_edit_director.py --base-model unsloth/gemma-4-E2B-it-unsloth-bnb-4bit --max-steps 20
python3 /mnt/c/projects/ai-training-lab/unsloth/scripts/compare_cliplab_edit_director.py --adapter-dir /mnt/s/AITraining/adapters/<run>/adapter
```
"""
        (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    manifest = build_dataset(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not args.dry_run and manifest.get("validation", {}).get("errors"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
