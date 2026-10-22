#!/usr/bin/env python3
"""Run a lightweight quality audit for local-curator lane outputs.

The audit is intentionally deterministic and read-only. It scores existing
curator-lane artifacts for schema validity, required-field completeness,
grounding/provenance pointers, uncertainty/escalation handling, privacy, and
final-approval safety language. It does not call a model, approve labels, promote
adapters, change holdouts, or enable production routing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_OUT_DIR = Path("/mnt/s/AITraining/reports/local-curator-quality-audit")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]
FINAL_APPROVAL_TERMS = re.compile(
    r"(?i)\b(approved|final approved|production ready|promote(?:d)?|reviewed|verified)\b"
)
SAFE_FINAL_CONTEXT = re.compile(
    r"(?i)(not approved|requires|human|frontier|operator|approval|review_required|approved review item|quality_audit|review_status|frontier-reviewed|human/frontier-reviewed|reviewed_labels_seen|training_allowed|forbidden|set status=)"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> tuple[Any | None, list[str]]:
    if not path.exists():
        return None, [f"missing file: {path}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON in {path}: {exc}"]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing file: {path}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
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


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    return ""


def has_secret_like_text(value: Any) -> list[str]:
    text = flatten_text(value)
    return [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]


def approval_language_warnings(value: Any) -> list[str]:
    text = flatten_text(value)
    warnings: list[str] = []
    for match in FINAL_APPROVAL_TERMS.finditer(text):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        context = text[start:end]
        if not SAFE_FINAL_CONTEXT.search(context):
            warnings.append(f"approval-like term without nearby gate context: {match.group(0)!r}")
    return warnings[:5]


@dataclass
class AuditCase:
    case_id: str
    description: str
    path: Path
    validator: Callable[[Path], dict[str, Any]]


@dataclass
class CaseResult:
    case_id: str
    description: str
    path: str
    passed: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "path": self.path,
            "passed": self.passed,
            "score": round(self.score, 3),
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def score_checks(checks: dict[str, bool], *, hard_fail: bool = False) -> tuple[bool, float]:
    if not checks:
        return False, 0.0
    score = sum(1 for ok in checks.values() if ok) / len(checks)
    return (not hard_fail and all(checks.values())), score


def validate_frontier_packet(path: Path) -> dict[str, Any]:
    data, errors = read_json(path)
    checks = {
        "json_valid": data is not None,
        "has_schema_sections": False,
        "has_queue_summary": False,
        "has_review_checklist": False,
        "has_artifact_pointers": False,
        "has_uncertainty_or_gate": False,
        "privacy_raw_omitted": False,
        "no_secret_like_text": False,
        "no_unsafe_final_approval_language": False,
    }
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    if isinstance(data, dict):
        checks["has_schema_sections"] = all(k in data for k in ("lane", "packet_id", "generated_at", "frontier_review_required"))
        checks["has_queue_summary"] = isinstance(data.get("queue"), dict) and "total" in data["queue"]
        checklist = data.get("frontier_review_checklist")
        checks["has_review_checklist"] = isinstance(checklist, list) and len(checklist) >= 4
        pointers = data.get("raw_artifact_pointers")
        checks["has_artifact_pointers"] = isinstance(pointers, list) and bool(pointers)
        checks["has_uncertainty_or_gate"] = "frontier_review_required" in data and isinstance(data.get("frontier_review_required"), bool)
        policy_obj = data.get("compact_policy")
        policy = policy_obj if isinstance(policy_obj, dict) else {}
        checks["privacy_raw_omitted"] = policy.get("raw_transcripts_logs_embedded") is False
        secrets = has_secret_like_text(data)
        approvals = approval_language_warnings(data)
        checks["no_secret_like_text"] = not secrets
        checks["no_unsafe_final_approval_language"] = not approvals
        warnings.extend([f"secret-like pattern present: {item}" for item in secrets])
        warnings.extend(approvals)
        metrics = {
            "lane": data.get("lane"),
            "frontier_review_required": data.get("frontier_review_required"),
            "queue_total": data.get("queue", {}).get("total") if isinstance(data.get("queue"), dict) else None,
            "errors_count": data.get("errors", {}).get("count") if isinstance(data.get("errors"), dict) else None,
            "checklist_items": len(checklist) if isinstance(checklist, list) else 0,
        }
    passed, score = score_checks(checks, hard_fail=bool(errors or warnings))
    return {"passed": passed, "score": score, "checks": checks, "errors": errors, "warnings": warnings, "metrics": metrics}


def validate_telemetry_summary(path: Path) -> dict[str, Any]:
    data, errors = read_json(path)
    checks = {
        "json_valid": data is not None,
        "schema_version_present": False,
        "has_local_frontier_split": False,
        "has_savings_estimates": False,
        "has_quality_audit_counts": False,
        "privacy_statement_present": False,
        "recent_records_no_raw_text": False,
        "no_secret_like_text": False,
        "no_unsafe_final_approval_language": False,
    }
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    if isinstance(data, dict):
        checks["schema_version_present"] = data.get("schema_version") == "local-first-summary-v1"
        checks["has_local_frontier_split"] = isinstance(data.get("local_frontier_split"), dict)
        checks["has_savings_estimates"] = isinstance(data.get("estimated_tokens_saved"), dict) and "estimated_cost_saved_usd" in data
        checks["has_quality_audit_counts"] = isinstance(data.get("quality_audit"), dict)
        checks["privacy_statement_present"] = "Raw prompts" in str(data.get("privacy", ""))
        recent = data.get("recent")
        checks["recent_records_no_raw_text"] = isinstance(recent, list) and all(
            "text" not in row and "description" not in row and "summary" not in row for row in recent if isinstance(row, dict)
        )
        secrets = has_secret_like_text(data)
        approvals = approval_language_warnings(data)
        checks["no_secret_like_text"] = not secrets
        checks["no_unsafe_final_approval_language"] = not approvals
        warnings.extend([f"secret-like pattern present: {item}" for item in secrets])
        warnings.extend(approvals)
        metrics = {
            "record_count": data.get("record_count"),
            "frontier_calls_avoided": data.get("frontier_calls_avoided"),
            "quality_audit": data.get("quality_audit"),
        }
    passed, score = score_checks(checks, hard_fail=bool(errors or warnings))
    return {"passed": passed, "score": score, "checks": checks, "errors": errors, "warnings": warnings, "metrics": metrics}


def validate_opsqa_dataset_report(path: Path) -> dict[str, Any]:
    data, errors = read_json(path)
    checks = {
        "json_valid": data is not None,
        "schema_versions_present": False,
        "counts_present": False,
        "privacy_policy_present": False,
        "golden_or_reviewed_rows_present": False,
        "no_rejected_rows_exported": False,
        "no_secret_like_text": False,
        "no_unsafe_final_approval_language": False,
    }
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    if isinstance(data, dict):
        counts_obj = data.get("counts")
        counts = counts_obj if isinstance(counts_obj, dict) else {}
        checks["schema_versions_present"] = bool(data.get("schema_version") and data.get("run_packet_schema_version"))
        checks["counts_present"] = bool(counts) and "exported_rows" in counts
        checks["privacy_policy_present"] = isinstance(data.get("privacy_policy"), dict)
        checks["golden_or_reviewed_rows_present"] = int(counts.get("golden", 0) or 0) + int(counts.get("reviewed_labels_seen", 0) or 0) > 0
        checks["no_rejected_rows_exported"] = int(counts.get("rejected_rows", 0) or 0) == 0
        secrets = has_secret_like_text(data)
        approvals = approval_language_warnings(data)
        checks["no_secret_like_text"] = not secrets
        checks["no_unsafe_final_approval_language"] = not approvals
        warnings.extend([f"secret-like pattern present: {item}" for item in secrets])
        warnings.extend(approvals)
        metrics = {"counts": counts, "dataset_name": data.get("dataset_name")}
    passed, score = score_checks(checks, hard_fail=bool(errors or warnings))
    return {"passed": passed, "score": score, "checks": checks, "errors": errors, "warnings": warnings, "metrics": metrics}


def validate_router_shadow_log(path: Path) -> dict[str, Any]:
    rows, errors = read_jsonl(path)
    checks = {
        "jsonl_valid": not errors,
        "has_rows": bool(rows),
        "required_fields_present": False,
        "shadow_mode_only": False,
        "has_policy_version": False,
        "has_escalation_or_reason": False,
        "no_raw_prompt_text": False,
        "no_secret_like_text": False,
        "no_unsafe_final_approval_language": False,
    }
    warnings: list[str] = []
    if rows:
        checks["required_fields_present"] = all(
            all(k in row for k in ("logged_at", "decision_id", "task", "decision")) for row in rows
        )
        decisions = [decision for row in rows if isinstance((decision := row.get("decision")), dict)]
        checks["shadow_mode_only"] = bool(decisions) and all(decision.get("shadow_mode") is True for decision in decisions)
        checks["has_policy_version"] = bool(decisions) and all(bool(decision.get("policy_version")) for decision in decisions)
        checks["has_escalation_or_reason"] = bool(decisions) and all(bool(decision.get("reason")) for decision in decisions)
        # Router shadow logs may keep compact task text for reproducibility, but must not include prompts/messages/raw logs.
        checks["no_raw_prompt_text"] = all(
            not any(key in row for key in ("prompt", "messages", "raw", "transcript")) for row in rows
        )
        secrets = has_secret_like_text(rows)
        approvals = approval_language_warnings(rows)
        checks["no_secret_like_text"] = not secrets
        checks["no_unsafe_final_approval_language"] = not approvals
        warnings.extend([f"secret-like pattern present: {item}" for item in secrets])
        warnings.extend(approvals)
    passed, score = score_checks(checks, hard_fail=bool(errors or warnings))
    metrics = {
        "row_count": len(rows),
        "modes": sorted({str(row.get("decision", {}).get("mode")) for row in rows if isinstance(row.get("decision"), dict)}),
    }
    return {"passed": passed, "score": score, "checks": checks, "errors": errors, "warnings": warnings, "metrics": metrics}


def build_cases() -> list[AuditCase]:
    return [
        AuditCase(
            "gaming_frontier_packet",
            "Gaming-pipeline frontier packet has compact schema, provenance, gate, and privacy fields.",
            Path("/mnt/s/AITraining/review-packets/gaming-pipeline-current-smoke.json"),
            validate_frontier_packet,
        ),
        AuditCase(
            "general_ops_frontier_packet",
            "General-ops-router frontier packet has compact schema, provenance, gate, and privacy fields.",
            Path("/mnt/s/AITraining/review-packets/general-ops-router-current-smoke.json"),
            validate_frontier_packet,
        ),
        AuditCase(
            "local_first_telemetry_summary",
            "Local-first telemetry summary tracks route split, avoided frontier calls, quality audit counts, and privacy posture.",
            Path("/mnt/s/AITraining/traces/metrics/local-first-summary.json"),
            validate_telemetry_summary,
        ),
        AuditCase(
            "opsqa_dataset_report",
            "ClipLab ops QA dataset report exposes schema versions, counts, and privacy export rules.",
            Path("/mnt/s/AITraining/datasets/cliplab-opsqa-v0.1/dataset_report.json"),
            validate_opsqa_dataset_report,
        ),
        AuditCase(
            "router_shadow_log",
            "Router shadow decisions are explicit shadow-mode records with policy versions and reasons.",
            Path("/mnt/s/AITraining/router-shadow/decisions.jsonl"),
            validate_router_shadow_log,
        ),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local curator quality audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Overall: {'PASS' if report['passed'] else 'FAIL'} ({report['passed_cases']}/{report['case_count']} cases passed, average score {report['average_score']:.3f})",
        "",
        "Scope: deterministic read-only audit of existing AI Training curator-lane artifacts. No model was launched, no labels/adapters were approved, no holdouts were changed, and production routing remains disabled.",
        "",
        "## Cases",
    ]
    for case in report["cases"]:
        lines.extend([
            "",
            f"### {case['case_id']} — {'PASS' if case['passed'] else 'FAIL'} ({case['score']:.3f})",
            "",
            case["description"],
            "",
            f"Path: `{case['path']}`",
            "",
            "Checks:",
        ])
        for name, ok in case["checks"].items():
            lines.append(f"- {'PASS' if ok else 'FAIL'} {name}")
        if case["errors"]:
            lines.append("Errors:")
            lines.extend(f"- {item}" for item in case["errors"])
        if case["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in case["warnings"])
        if case["metrics"]:
            lines.append("Metrics:")
            lines.append("```json")
            lines.append(json.dumps(case["metrics"], indent=2, sort_keys=True))
            lines.append("```")
    lines.extend([
        "",
        "## Local-curator gate interpretation",
        "",
        "- Passed cases are suitable as local draft/triage evidence only.",
        "- This audit does not mark examples reviewed/approved and does not approve adapters or routing changes.",
        "- Any promotion, holdout change, production routing enablement, or subjective acceptance still needs human/frontier review.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-id", default=f"audit-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args()

    results: list[CaseResult] = []
    for case in build_cases():
        outcome = case.validator(case.path)
        results.append(
            CaseResult(
                case_id=case.case_id,
                description=case.description,
                path=str(case.path),
                passed=bool(outcome["passed"]),
                score=float(outcome["score"]),
                checks=outcome["checks"],
                errors=outcome["errors"],
                warnings=outcome["warnings"],
                metrics=outcome["metrics"],
            )
        )

    report = {
        "schema_version": "local-curator-quality-audit-v1",
        "generated_at": utc_now(),
        "report_id": args.report_id,
        "profile_under_audit": "ai-training-local-curator",
        "mode": "deterministic_read_only_artifact_audit",
        "model_launched": False,
        "case_count": len(results),
        "passed_cases": sum(1 for result in results if result.passed),
        "average_score": round(sum(result.score for result in results) / len(results), 3) if results else 0.0,
        "passed": all(result.passed for result in results),
        "cases": [result.to_json() for result in results],
        "safety_notes": [
            "Local curator output remains draft/triage only.",
            "No final approval, adapter promotion, holdout changes, production routing, or training launch occurred.",
            "Frontier/human gate remains required for high-impact decisions.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.report_id}.json"
    md_path = args.out_dir / f"{args.report_id}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"ok": report["passed"], "json": str(json_path), "markdown": str(md_path), "summary": {"case_count": report["case_count"], "passed_cases": report["passed_cases"], "average_score": report["average_score"]}}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
