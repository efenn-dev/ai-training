#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

S_ROOT = Path(r"S:\AITraining")
TRACE_ROOT = S_ROOT / "traces"
DEFAULT_REVIEW_ROOT = TRACE_ROOT / "review-packs"
TEXT_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".log",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".py",
    ".sh",
    ".ps1",
    ".bat",
    ".rst",
}
MAX_SCRUB_CHARS = 120_000

MASK_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("api_key_assignment", re.compile(r"(OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN)=([^\s]+)"), r"\1=[REDACTED_SECRET]"),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "[REDACTED_SECRET]"),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED_SECRET]"),
    ("gh_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "[REDACTED_SECRET]"),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "[REDACTED_PRIVATE_KEY]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("windows_user_path", re.compile(r"[A-Za-z]:\\Users\\([^\\\s]+)"), r"C:\\Users\\USER"),
    ("wsl_user_path", re.compile(r"/home/([^/\s]+)"), "/home/USER"),
    ("mnt_user_path", re.compile(r"/mnt/[a-z]/Users/([^/\s]+)"), "/mnt/DRIVE/Users/USER"),
]


@dataclass
class FileReviewStats:
    file_count: int = 0
    file_count_modified: int = 0
    text_files: int = 0
    json_files: int = 0
    jsonl_files: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    rows_in: int = 0
    rows_out: int = 0
    redaction_count: int = 0
    redaction_by_pattern: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.redaction_by_pattern is None:
            self.redaction_by_pattern = {}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def scrub_text(text: str, max_chars: int = MAX_SCRUB_CHARS) -> tuple[str, int, dict[str, int]]:
    redactions = 0
    counts: dict[str, int] = {}
    scrubbed = text
    if len(scrubbed) > max_chars:
        scrubbed = scrubbed[:max_chars] + "\n[REDACTED_LONG_PAYLOAD]"
        redactions += 1
        counts["max_chars"] = counts.get("max_chars", 0) + 1

    for name, pattern, replacement in MASK_PATTERNS:
        matches = pattern.findall(scrubbed)
        if not matches:
            continue
        try:
            scrubbed, count = pattern.subn(replacement, scrubbed)
        except TypeError:
            # Some patterns are built with capture groups and custom replacement lambdas.
            scrubbed, count = pattern.subn(lambda _m, repl=replacement: repl, scrubbed)
        if count:
            redactions += count
            counts[name] = counts.get(name, 0) + count

    if "base64" in scrubbed and len(text) > 2000:
        scrubbed = "[REDACTED_LARGE_BASE64_PAYLOAD]"
        redactions += 1
        counts["large_base64"] = counts.get("large_base64", 0) + 1

    return scrubbed, redactions, counts


def scrub_obj(obj: Any) -> tuple[Any, int, dict[str, int]]:
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, list):
        total = 0
        pattern_counts: dict[str, int] = {}
        out: list[Any] = []
        for item in obj:
            cleaned, count, counts = scrub_obj(item)
            out.append(cleaned)
            total += count
            for key, value in counts.items():
                pattern_counts[key] = pattern_counts.get(key, 0) + value
        return out, total, pattern_counts
    if isinstance(obj, dict):
        total = 0
        pattern_counts = {}
        out: dict[str, Any] = {}
        for key, value in obj.items():
            cleaned, count, counts = scrub_obj(value)
            out[key] = cleaned
            total += count
            for k, v in counts.items():
                pattern_counts[k] = pattern_counts.get(k, 0) + v
        return out, total, pattern_counts
    return obj, 0, {}


def scrub_jsonl(src: Path, dst: Path) -> tuple[int, int, dict[str, int]]:
    redactions = 0
    pattern_counts: dict[str, int] = {}
    rows_in = 0
    rows_out = 0
    lines: list[str] = []
    with src.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                lines.append(line if line.endswith("\n") else "\n")
                continue
            rows_in += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                cleaned, text_count, counts = scrub_text(line.rstrip("\n"), max_chars=MAX_SCRUB_CHARS)
            else:
                cleaned, text_count, counts = scrub_obj(payload)
                cleaned = json.dumps(cleaned, ensure_ascii=False)
            redactions += text_count
            for key, value in counts.items():
                pattern_counts[key] = pattern_counts.get(key, 0) + value
            rows_out += 1
            lines.append(f"{cleaned}\n")

    dst.write_text("".join(lines), encoding="utf-8")
    return rows_in, rows_out, redactions, pattern_counts


def scrub_json(src: Path, dst: Path) -> tuple[int, int, int, dict[str, int]]:
    with src.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    cleaned, redactions, counts = scrub_obj(payload)
    rows_in = len(payload) if isinstance(payload, list) else 1
    rows_out = len(cleaned) if isinstance(cleaned, list) else 1
    dst.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows_in, rows_out, redactions, counts


def scrub_text_file(src: Path, dst: Path) -> tuple[int, int, dict[str, int]]:
    text = src.read_text(encoding="utf-8", errors="replace")
    scrubbed, redactions, counts = scrub_text(text)
    dst.write_text(scrubbed, encoding="utf-8")
    return redactions, 0, counts


def copy_and_scrub(src_root: Path, dst_root: Path) -> FileReviewStats:
    stats = FileReviewStats()
    stats.redaction_by_pattern = {}
    for child in src_root.rglob("*"):
        if child.is_dir():
            continue
        relative = child.relative_to(src_root)
        target = dst_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        bytes_in = child.stat().st_size
        stats.file_count += 1
        stats.bytes_in += bytes_in

        modified = False
        if child.suffix.lower() == ".jsonl":
            stats.jsonl_files += 1
            rows_in, rows_out, row_redactions, counts = scrub_jsonl(child, target)
            stats.rows_in += rows_in
            stats.rows_out += rows_out
            if row_redactions:
                modified = True
            for key, value in counts.items():
                stats.redaction_by_pattern[key] = stats.redaction_by_pattern.get(key, 0) + value
            stats.redaction_count += row_redactions
            stats.bytes_out += target.stat().st_size
        elif child.suffix.lower() == ".json":
            stats.json_files += 1
            rows_in, rows_out, redactions, counts = scrub_json(child, target)
            stats.rows_in += rows_in
            stats.rows_out += rows_out
            if redactions:
                modified = True
            for key, value in counts.items():
                stats.redaction_by_pattern[key] = stats.redaction_by_pattern.get(key, 0) + value
            stats.redaction_count += redactions
            stats.bytes_out += target.stat().st_size
        elif child.suffix.lower() in TEXT_EXTENSIONS:
            stats.text_files += 1
            redactions, _rows_out, counts = scrub_text_file(child, target)
            if redactions:
                modified = True
            for key, value in counts.items():
                stats.redaction_by_pattern[key] = stats.redaction_by_pattern.get(key, 0) + value
            stats.redaction_count += redactions
            stats.bytes_out += target.stat().st_size
        else:
            shutil.copy2(child, target)
            stats.bytes_out += target.stat().st_size

        if modified:
            stats.file_count_modified += 1

    return stats


def import_manifest_paths(manifest_path: Path) -> Path | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    source = payload.get("source_root", "")
    return Path(str(source).replace("/mnt/", ""))


def latest_manifest(import_root: Path) -> Path | None:
    manifests = sorted(import_root.rglob("IMPORT_MANIFEST.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def license_review(source_alias: str, manifest: dict[str, Any]) -> dict[str, Any]:
    source = str(manifest.get("source", "")).lower()
    source_root = str(manifest.get("source_root", ""))
    if "claude-fable-5-claude-code" in source or "fable-claude-code" in source_alias:
        return {
            "status": "blocked",
            "license": "agpl-3.0",
            "risk": "commercial_reuse_blocked",
            "notes": "AGPL source used in this archive requires explicit commercial/public approval and legal review.",
        }
    if "fable-trace-lora" in source_alias.lower() or "fable-trace-lora" in source_root.lower():
        return {
            "status": "manual",
            "license": "unknown",
            "risk": "no_license_file_detected",
            "notes": "Local fable-trace-lora workspace archive. Verify repo license + model/data terms before use.",
        }
    return {
        "status": "manual",
        "license": "unknown",
        "risk": "unknown_source",
        "notes": "No explicit license mapping found in this review run.",
    }


def privacy_review(stats: FileReviewStats) -> dict[str, Any]:
    if stats.redaction_count == 0:
        decision = "clear_no_redactions"
        blocked = False
    elif stats.redaction_count < 5:
        decision = "low_redactions"
        blocked = False
    else:
        decision = "needs_human_review"
        blocked = True
    return {
        "risk": "review_required" if blocked else "low",
        "decision": decision,
        "redaction_count": stats.redaction_count,
        "modified_file_count": stats.file_count_modified,
        "patterns": stats.redaction_by_pattern,
    }


def source_display_name(path: Path) -> str:
    return path.relative_to(TRACE_ROOT / "imports").as_posix()


def build_review_report(sources: list[dict[str, Any]], run_root: Path, include_raw_input: bool) -> dict[str, Any]:
    total_redactions = sum(source["redaction_count"] for source in sources)
    total_files = sum(source["file_count"] for source in sources)
    return {
        "schema_version": "fable-license-privacy-review-v1",
        "review_id": run_root.name,
        "started_at": now_iso(),
        "completed_at": now_iso(),
        "source_root": str(run_root),
        "summary": {
            "total_sources": len(sources),
            "total_files_copied": total_files,
            "total_rows_copied": sum(source["rows_in"] for source in sources),
            "total_rows_out": sum(source["rows_out"] for source in sources),
            "total_redactions": total_redactions,
            "total_files_modified": sum(source["modified_files"] for source in sources),
            "input_mode": "copy_only_no_raw_mutation",
            "license_manual_review_required": any(source["license_review"]["status"] == "manual" or source["license_review"]["status"] == "blocked" for source in sources),
            "privacy_review_required": any(source["privacy_review"]["risk"] == "review_required" for source in sources),
            "source": "manual_fable_review",
        },
        "sources": sources,
        "notes": [
            "No source data was modified. This is a working copy scrub pass only.",
            "Run a legal/commercial review before moving any rows to sanitized/normalized training lanes.",
        ],
        "include_raw_inputs": include_raw_input,
    }


def run_for_sources(review_root: Path, source_names: Iterable[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source_name in source_names:
        source_root = TRACE_ROOT / "imports" / source_name
        if not source_root.exists():
            logging.warning("Source import not found: %s", source_root)
            continue
        manifest_path = latest_manifest(source_root)
        if not manifest_path:
            logging.warning("No import manifest for %s", source_root)
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        source_root_path = manifest_path.parent
        destination = review_root / source_name / manifest_path.parent.name
        if destination.exists():
            shutil.rmtree(destination)
        logging.info("Creating scrubbed review copy: %s -> %s", source_root_path, destination)
        stats = copy_and_scrub(source_root_path, destination)
        source_stats = {
            "source_name": source_name,
            "source_type": manifest.get("schema_version", "unknown"),
            "import_manifest": str(manifest_path),
            "source_root": str(source_root_path),
            "destination_root": str(destination),
            "license_review": license_review(source_name, manifest),
            "privacy_review": privacy_review(stats),
            "file_count": stats.file_count,
            "modified_files": stats.file_count_modified,
            "json_files": stats.json_files,
            "jsonl_files": stats.jsonl_files,
            "text_files": stats.text_files,
            "rows_in": stats.rows_in,
            "rows_out": stats.rows_out,
            "bytes_in": stats.bytes_in,
            "bytes_out": stats.bytes_out,
            "redaction_count": stats.redaction_count,
            "redaction_by_pattern": stats.redaction_by_pattern,
        }
        results.append(source_stats)
        logging.info(
            "Source %s reviewed: files=%s modified=%s rows=%s/%s redactions=%s",
            source_name,
            stats.file_count,
            stats.file_count_modified,
            stats.rows_in,
            stats.rows_out,
            stats.redaction_count,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start license/privacy/masking review by creating scrubbed copies.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["fable-claude-code", "fable-trace-lora-sessions"],
        help="Import folders under S:/AITraining/traces/imports to review.",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=DEFAULT_REVIEW_ROOT,
        help="Destination root for scrubbed review copies.",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Optional manual timestamp label used for destination folder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    stamp = args.stamp or time.strftime("%Y%m%d-%H%M%S")
    review_root = args.review_root / f"fable-review-{stamp}"
    review_root.mkdir(parents=True, exist_ok=True)
    results = run_for_sources(review_root, args.sources)
    report_path = review_root / "LICENSE_PRIVACY_REVIEW_MANIFEST.json"
    summary = build_review_report(results, review_root, include_raw_input=False)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Review report written: %s", report_path)


if __name__ == "__main__":
    main()
