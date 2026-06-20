#!/usr/bin/env python3
"""Shared helpers for the ClipLab Edit Director LoRA lane."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cliplab_edit_director.v0.1"
SYSTEM_PROMPT = "You are ClipLab's edit director. Return strict JSON only."

DEFAULT_BASE_MODEL = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
DEFAULT_SMOKE_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DEFAULT_DATASET_NAME = "cliplab-edit-director-v0.1"

TRANSCRIPT_EXCERPT_LIMIT = 520
NOTES_LIMIT = 800

ALLOWED_DECISIONS = {"keep", "reject"}
ALLOWED_PLATFORM_FIT = {"strong", "ok", "weak", "unknown"}

TOP_LEVEL_KEYS = {"schema_version", "clips", "run_notes", "safety"}
CLIP_KEYS = {
    "clip_id",
    "decision",
    "trim_start_sec",
    "trim_end_sec",
    "reasons",
    "platform_fit",
    "safety_notes",
    "metadata_title",
    "metadata_notes",
}
SAFETY_KEYS = {"child_safe", "sponsor_safe"}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]

LOCAL_PATH_PATTERNS = [
    re.compile(r"\b[A-Za-z]:\\[^\s\"']+"),
    re.compile(r"(?<![\w-])/mnt/[A-Za-z]/[^\s\"']+"),
]

FAKE_VERIFICATION_PATTERNS = (
    "verified",
    "tested",
    "watched the video",
    "reviewed the raw footage",
    "confirmed in production",
    "exported",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_dataset_dir() -> Path:
    """Return the best local dataset directory for the current OS."""
    if Path("/mnt/s/AITraining/datasets").exists():
        return Path("/mnt/s/AITraining/datasets") / DEFAULT_DATASET_NAME
    if Path("S:/AITraining/datasets").exists():
        return Path("S:/AITraining/datasets") / DEFAULT_DATASET_NAME
    return Path("/mnt/s/AITraining/datasets") / DEFAULT_DATASET_NAME


def default_adapters_dir() -> Path:
    """Return the best local adapter root for the current OS."""
    if Path("/mnt/s/AITraining/adapters").exists():
        return Path("/mnt/s/AITraining/adapters")
    if Path("S:/AITraining/adapters").exists():
        return Path("S:/AITraining/adapters")
    return Path("/mnt/s/AITraining/adapters")


def default_runs_dir() -> Path:
    """Return the best local run-report root for the current OS."""
    if Path("/mnt/s/AITraining/runs").exists():
        return Path("/mnt/s/AITraining/runs")
    if Path("S:/AITraining/runs").exists():
        return Path("S:/AITraining/runs")
    return Path("/mnt/s/AITraining/runs")


def default_merged_models_dir() -> Path:
    """Return the best local merged-model root for the current OS."""
    if Path("/mnt/s/AITraining/merged-models").exists():
        return Path("/mnt/s/AITraining/merged-models")
    if Path("S:/AITraining/merged-models").exists():
        return Path("S:/AITraining/merged-models")
    return Path("/mnt/s/AITraining/merged-models")


def default_pipeline_root() -> Path:
    """Return the best local gaming-pipeline root for the current OS."""
    if Path("/mnt/c/projects/gaming-pipeline").exists():
        return Path("/mnt/c/projects/gaming-pipeline")
    return Path("C:/projects/gaming-pipeline")


def compact_json(value: Any) -> str:
    """Serialize JSON compactly for model prompts."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a stable short hash for provenance without exposing raw paths."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def read_json(path: Path, default: Any | None = None) -> Any:
    """Read JSON from *path*, returning default when the file is missing."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Missing files are treated as empty."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    """Write a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any, limit: int = NOTES_LIMIT) -> str:
    """Normalize whitespace and cap text length."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit > 0 and len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def clean_list(value: Any, limit: int = 12, item_limit: int = 120) -> list[str]:
    """Return a compact list of unique strings."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = clean_text(item, item_limit)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def basename_ref(value: Any) -> dict[str, str]:
    """Return a privacy-safe basename plus hash for a local path-like value."""
    raw = str(value or "").replace("\x00", "").strip()
    if not raw:
        return {"basename": "", "path_hash": ""}
    basename = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return {
        "basename": clean_text(basename, 180),
        "path_hash": stable_hash(raw),
    }


def flatten_text(value: Any) -> str:
    """Flatten nested JSON-like values into searchable text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    return str(value) if value is not None else ""


def detect_sensitive_text(value: Any) -> list[str]:
    """Return privacy/security validation errors found in value."""
    text = flatten_text(value)
    errors: list[str] = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"possible secret matched {pattern.pattern}")
    for pattern in LOCAL_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            errors.append(f"raw local path found: {match.group(0)[:120]}")
    return errors


def transcript_limit_errors(value: Any, prefix: str = "") -> list[str]:
    """Find oversized transcript/dialogue excerpts in prompt payloads."""
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if isinstance(item, str) and ("transcript" in lowered or "dialogue" in lowered):
                if len(item) > TRANSCRIPT_EXCERPT_LIMIT:
                    errors.append(f"{next_prefix} exceeds {TRANSCRIPT_EXCERPT_LIMIT} chars")
            errors.extend(transcript_limit_errors(item, next_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            errors.extend(transcript_limit_errors(item, f"{prefix}[{idx}]"))
    return errors


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to a finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_decision(value: Any) -> str:
    """Normalize UI/model decision words into keep/reject labels."""
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"accepted", "accept", "selected", "select", "keep", "kept", "include"}:
        return "keep"
    if text in {"rejected", "reject", "dropped", "drop", "exclude", "skip"}:
        return "reject"
    return text


def validate_edit_director_response(obj: Any) -> str | None:
    """Validate the strict assistant JSON schema."""
    if not isinstance(obj, dict):
        return "assistant payload must be an object"
    extra = set(obj) - TOP_LEVEL_KEYS
    missing = TOP_LEVEL_KEYS - set(obj)
    if missing:
        return f"assistant payload missing keys: {', '.join(sorted(missing))}"
    if extra:
        return f"assistant payload has unexpected keys: {', '.join(sorted(extra))}"
    if obj.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version must be {SCHEMA_VERSION!r}"
    clips = obj.get("clips")
    if not isinstance(clips, list) or not clips:
        return "clips must be a non-empty list"
    for idx, clip in enumerate(clips):
        if not isinstance(clip, dict):
            return f"clips[{idx}] must be an object"
        extra_clip = set(clip) - CLIP_KEYS
        missing_clip = CLIP_KEYS - set(clip)
        if missing_clip:
            return f"clips[{idx}] missing keys: {', '.join(sorted(missing_clip))}"
        if extra_clip:
            return f"clips[{idx}] has unexpected keys: {', '.join(sorted(extra_clip))}"
        if not isinstance(clip.get("clip_id"), str) or not clip["clip_id"].strip():
            return f"clips[{idx}].clip_id must be a non-empty string"
        decision = normalize_decision(clip.get("decision"))
        if decision not in ALLOWED_DECISIONS:
            return f"clips[{idx}].decision must be one of {sorted(ALLOWED_DECISIONS)}"
        start = to_float(clip.get("trim_start_sec"), -1.0)
        end = to_float(clip.get("trim_end_sec"), -1.0)
        if start < 0 or end < 0 or end < start:
            return f"clips[{idx}] trim bounds must be finite non-negative start<=end"
        reasons = clip.get("reasons")
        if not isinstance(reasons, list) or not all(isinstance(r, str) and r.strip() for r in reasons):
            return f"clips[{idx}].reasons must be a list of non-empty strings"
        if len(reasons) > 12:
            return f"clips[{idx}].reasons must contain at most 12 items"
        if clip.get("platform_fit") not in ALLOWED_PLATFORM_FIT:
            return f"clips[{idx}].platform_fit must be one of {sorted(ALLOWED_PLATFORM_FIT)}"
        safety_notes = clip.get("safety_notes")
        if not isinstance(safety_notes, list) or not all(isinstance(s, str) for s in safety_notes):
            return f"clips[{idx}].safety_notes must be a list of strings"
        for key in ("metadata_title", "metadata_notes"):
            if not isinstance(clip.get(key), str):
                return f"clips[{idx}].{key} must be a string"
    safety = obj.get("safety")
    if not isinstance(safety, dict):
        return "safety must be an object"
    extra_safety = set(safety) - SAFETY_KEYS
    missing_safety = SAFETY_KEYS - set(safety)
    if missing_safety:
        return f"safety missing keys: {', '.join(sorted(missing_safety))}"
    if extra_safety:
        return f"safety has unexpected keys: {', '.join(sorted(extra_safety))}"
    if not all(isinstance(safety.get(key), bool) for key in SAFETY_KEYS):
        return "safety child_safe and sponsor_safe must be booleans"
    if not isinstance(obj.get("run_notes"), str):
        return "run_notes must be a string"
    return None


def parse_assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Load the assistant JSON payload from a training/eval row."""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("row messages must contain system, user, assistant")
    content = messages[2].get("content") if isinstance(messages[2], dict) else None
    if not isinstance(content, str):
        raise ValueError("assistant content must be a string")
    return json.loads(content)


def parse_user_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Load the user JSON payload from a training/eval row."""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("row messages must contain system and user")
    content = messages[1].get("content") if isinstance(messages[1], dict) else None
    if not isinstance(content, str):
        raise ValueError("user content must be a string")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("user content must parse to an object")
    return payload


def validate_training_row(row: dict[str, Any], line_no: int, split: str) -> list[str]:
    """Validate one train/eval/holdout JSONL row."""
    errors: list[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return [f"{split}:{line_no}: messages must contain exactly 3 turns"]
    expected_roles = ["system", "user", "assistant"]
    for idx, role in enumerate(expected_roles):
        if not isinstance(messages[idx], dict) or messages[idx].get("role") != role:
            errors.append(f"{split}:{line_no}: messages[{idx}].role must be {role!r}")
        if not isinstance(messages[idx].get("content"), str):
            errors.append(f"{split}:{line_no}: messages[{idx}].content must be a string")
    if messages[0].get("content") != SYSTEM_PROMPT:
        errors.append(f"{split}:{line_no}: system prompt mismatch")
    for message in messages:
        for issue in detect_sensitive_text(message.get("content", "")):
            errors.append(f"{split}:{line_no}: {issue}")
    try:
        user_payload = parse_user_payload(row)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{split}:{line_no}: invalid user JSON: {exc}")
        user_payload = {}
    errors.extend(f"{split}:{line_no}: {err}" for err in transcript_limit_errors(user_payload))
    try:
        assistant = parse_assistant_payload(row)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{split}:{line_no}: invalid assistant JSON: {exc}")
        assistant = None
    schema_error = validate_edit_director_response(assistant)
    if schema_error:
        errors.append(f"{split}:{line_no}: {schema_error}")
    if row.get("split") != split:
        errors.append(f"{split}:{line_no}: row split must be {split!r}")
    if row.get("has_human_label") is not True:
        errors.append(f"{split}:{line_no}: has_human_label must be true")
    if not row.get("run_id"):
        errors.append(f"{split}:{line_no}: run_id is required")
    if not row.get("clip_id"):
        errors.append(f"{split}:{line_no}: clip_id is required")
    return errors


def validate_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    """Validate the v0.1 dataset lane and return a summary."""
    dataset_dir = Path(dataset_dir)
    splits = {
        "train": dataset_dir / "train.jsonl",
        "eval": dataset_dir / "eval.jsonl",
        "holdout_real_runs": dataset_dir / "holdout_real_runs.jsonl",
    }
    errors: list[str] = []
    counts: dict[str, int] = {}
    decisions: dict[str, dict[str, int]] = {}
    seen_by_split: dict[str, set[tuple[str, str]]] = {}
    all_seen: dict[tuple[str, str], str] = {}

    for split, path in splits.items():
        if not path.exists():
            errors.append(f"missing required file: {path}")
            counts[split] = 0
            decisions[split] = {}
            continue
        rows = load_jsonl(path)
        counts[split] = len(rows)
        counter: Counter[str] = Counter()
        seen_by_split[split] = set()
        for line_no, row in enumerate(rows, 1):
            errors.extend(validate_training_row(row, line_no, split))
            key = (str(row.get("run_id", "")), str(row.get("clip_id", "")))
            if key in seen_by_split[split]:
                errors.append(f"{split}:{line_no}: duplicate run_id/clip_id in split: {key}")
            seen_by_split[split].add(key)
            prior_split = all_seen.get(key)
            if prior_split and prior_split != split:
                errors.append(f"{split}:{line_no}: run_id/clip_id also appears in {prior_split}: {key}")
            all_seen[key] = split
            try:
                assistant = parse_assistant_payload(row)
                for clip in assistant.get("clips", []):
                    counter[normalize_decision(clip.get("decision"))] += 1
            except Exception:
                continue
        decisions[split] = dict(counter)

    unlabeled_rows = load_jsonl(dataset_dir / "unlabeled_candidates.jsonl")
    counts["unlabeled_candidates"] = len(unlabeled_rows)
    unlabeled_statuses: Counter[str] = Counter()
    for line_no, row in enumerate(unlabeled_rows, 1):
        if row.get("status") != "needs_label":
            errors.append(f"unlabeled_candidates:{line_no}: status must be 'needs_label'")
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"unlabeled_candidates:{line_no}: schema_version must be {SCHEMA_VERSION!r}")
        if not isinstance(row.get("input"), dict):
            errors.append(f"unlabeled_candidates:{line_no}: input must be an object")
        for issue in detect_sensitive_text(row.get("input", {})):
            errors.append(f"unlabeled_candidates:{line_no}: {issue}")
        unlabeled_statuses[str(row.get("status", "missing"))] += 1

    return {
        "dataset_dir": str(dataset_dir),
        "counts": counts,
        "decisions": decisions,
        "unlabeled_statuses": dict(unlabeled_statuses),
        "errors": errors,
        "ok": not errors,
    }


def extract_json_object(text: str, repair: bool = True) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Extract a JSON object from model text with conservative repair."""
    cleaned = str(text or "").strip()
    cleaned = cleaned.split("<end_of_turn>")[0].strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    try:
        obj = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        if not repair:
            return None, f"json_parse_error: {exc}", None
        repaired = candidate.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        if repaired.startswith("{") and not repaired.endswith("}"):
            repaired += "}"
        try:
            obj = json.loads(repaired)
        except Exception as repaired_exc:  # noqa: BLE001
            return None, f"json_parse_error: {repaired_exc}; text={cleaned[:240]!r}", None
        error = validate_edit_director_response(obj)
        return obj, error, "structural_json_repair"
    if not isinstance(obj, dict):
        return None, "parsed JSON is not an object", None
    return obj, validate_edit_director_response(obj), None


def first_clip(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the first clip decision from a strict assistant payload."""
    if not isinstance(payload, dict):
        return None
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        return None
    clip = clips[0]
    return clip if isinstance(clip, dict) else None


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Return F1 from confusion counts."""
    if tp <= 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compute_eval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Edit Director eval metrics from per-case rows."""
    n = len(rows)
    valid = sum(1 for row in rows if not row.get("parse_error"))
    repaired = sum(1 for row in rows if row.get("json_repair"))
    human = sum(1 for row in rows if row.get("has_human_label"))
    human_trim = sum(1 for row in rows if row.get("has_human_trim_label"))
    keep_tp = keep_fp = keep_fn = 0
    reject_tp = reject_fp = reject_fn = 0
    correct = 0
    trim_errors: list[float] = []
    child_safe_misses = 0
    sponsor_safe_misses = 0
    child_safe_false_flags = 0
    sponsor_safe_false_flags = 0
    fake_verification_claims = 0

    for row in rows:
        expected = first_clip(row.get("expected"))
        parsed = first_clip(row.get("parsed"))
        exp_decision = normalize_decision(expected.get("decision")) if expected else ""
        pred_decision = normalize_decision(parsed.get("decision")) if parsed else ""
        if exp_decision == pred_decision and exp_decision in ALLOWED_DECISIONS:
            correct += 1
        if exp_decision == "keep":
            if pred_decision == "keep":
                keep_tp += 1
            else:
                keep_fn += 1
        elif pred_decision == "keep":
            keep_fp += 1
        if exp_decision == "reject":
            if pred_decision == "reject":
                reject_tp += 1
            else:
                reject_fn += 1
        elif pred_decision == "reject":
            reject_fp += 1
        if row.get("has_human_trim_label") and expected and parsed:
            trim_errors.append(abs(to_float(parsed.get("trim_start_sec")) - to_float(expected.get("trim_start_sec"))))
            trim_errors.append(abs(to_float(parsed.get("trim_end_sec")) - to_float(expected.get("trim_end_sec"))))
        exp_safety = row.get("expected", {}).get("safety") if isinstance(row.get("expected"), dict) else {}
        pred_safety = row.get("parsed", {}).get("safety") if isinstance(row.get("parsed"), dict) else {}
        if isinstance(exp_safety, dict) and isinstance(pred_safety, dict):
            if exp_safety.get("child_safe") is False and pred_safety.get("child_safe") is True:
                child_safe_misses += 1
            if exp_safety.get("sponsor_safe") is False and pred_safety.get("sponsor_safe") is True:
                sponsor_safe_misses += 1
            if exp_safety.get("child_safe") is True and pred_safety.get("child_safe") is False:
                child_safe_false_flags += 1
            if exp_safety.get("sponsor_safe") is True and pred_safety.get("sponsor_safe") is False:
                sponsor_safe_false_flags += 1
        decoded_or_notes = " ".join(
            [
                str(row.get("decoded") or ""),
                str((parsed or {}).get("run_notes") if isinstance(parsed, dict) else ""),
                str((parsed or {}).get("metadata_notes") if isinstance(parsed, dict) else ""),
            ]
        ).lower()
        if any(pattern in decoded_or_notes for pattern in FAKE_VERIFICATION_PATTERNS):
            fake_verification_claims += 1

    keep_f1 = f1_from_counts(keep_tp, keep_fp, keep_fn)
    reject_f1 = f1_from_counts(reject_tp, reject_fp, reject_fn)
    macro_f1 = (keep_f1 + reject_f1) / 2 if n else 0.0
    return {
        "num_examples": n,
        "strict_schema_valid_rate": valid / n if n else 0.0,
        "valid_json_rate": valid / n if n else 0.0,
        "repaired_json_rate": repaired / n if n else 0.0,
        "decision_accuracy": correct / n if n else 0.0,
        "keep_f1": keep_f1,
        "reject_f1": reject_f1,
        "keep_reject_macro_f1": macro_f1,
        "trim_boundary_mae_sec": (sum(trim_errors) / len(trim_errors)) if trim_errors else None,
        "human_labeled_rows": human,
        "human_labeled_rate": human / n if n else 0.0,
        "human_trim_labeled_rows": human_trim,
        "human_trim_labeled_rate": human_trim / n if n else 0.0,
        "child_safe_misses": child_safe_misses,
        "sponsor_safe_misses": sponsor_safe_misses,
        "child_safe_false_flags": child_safe_false_flags,
        "sponsor_safe_false_flags": sponsor_safe_false_flags,
        "fake_verification_claims": fake_verification_claims,
    }


def schema_document() -> dict[str, Any]:
    """Return the dataset schema description written beside JSONL files."""
    return {
        "name": DEFAULT_DATASET_NAME,
        "schema_version": SCHEMA_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "format": {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "compact JSON input payload"},
                {"role": "assistant", "content": "strict JSON response payload"},
            ],
            "assistant_required_keys": sorted(TOP_LEVEL_KEYS),
            "clip_required_keys": sorted(CLIP_KEYS),
            "safety_required_keys": sorted(SAFETY_KEYS),
        },
        "allowed_values": {
            "decision": sorted(ALLOWED_DECISIONS),
            "platform_fit": sorted(ALLOWED_PLATFORM_FIT),
        },
        "privacy": {
            "raw_video_uploaded": False,
            "raw_local_paths_in_messages": False,
            "transcript_excerpt_limit_chars": TRANSCRIPT_EXCERPT_LIMIT,
            "secret_patterns_checked": [pattern.pattern for pattern in SECRET_PATTERNS],
        },
        "promotion_notes": [
            "Train/eval/holdout rows must have explicit human clip feedback.",
            "Rows without labels belong in unlabeled_candidates.jsonl, not train/eval/holdout.",
            "Promotion requires strict schema validity, baseline improvement, safety gates, and human labels.",
        ],
    }
