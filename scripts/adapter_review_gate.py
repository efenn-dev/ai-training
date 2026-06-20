#!/usr/bin/env python3
"""Stage adapter eval reports and upsert a Mission Control Automation Review.

This script intentionally authorizes shadow-mode only. Approval of the generated
review card is a human signal that the adapter may be tried in shadow mode; live
routing/editing still requires a separate integration task.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

APP_MANAGER_DATA_DIR = Path("/mnt/c/projects/app-manager-mcp/data")
STATE_PATH = APP_MANAGER_DATA_DIR / "dashboard-state.json"
REVIEW_ARTIFACT_ROOT = APP_MANAGER_DATA_DIR / "review-artifacts"
TASK_ID = "task_20260618083557_rveiid"
SHADOW_ONLY_NOTE = (
    "Approval authorizes shadow-mode evaluation only. It does not authorize live "
    "production routing/editing, adapter export, or integration without a separate integration task."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=["gaming-pipeline-edit-director", "general-ops-router"])
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--promotion-ready", required=True, choices=["true", "false"])
    parser.add_argument("--summary", default="")
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--artifact-root", type=Path, default=REVIEW_ARTIFACT_ROOT)
    return parser.parse_args()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def review_id() -> str:
    return f"review_{time.strftime('%Y%m%d%H%M%S', time.gmtime())}_{str(time.time_ns())[-6:]}"


def safe_slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)[:90]
    return "-".join(part for part in clean.split("-") if part) or "adapter-eval"


def copy_if_exists(src: Path, dest: Path) -> str | None:
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)


def load_report_summary(report_json: Path) -> dict[str, Any]:
    report = load_json(report_json, {})
    gates = report.get("promotion_gates") if isinstance(report, dict) else {}
    gate_results = gates if isinstance(gates, dict) else {}
    failed = [name for name, gate in gate_results.items() if isinstance(gate, dict) and not gate.get("passed")]
    return {
        "promotion_ready": bool(report.get("promotion_ready")),
        "failed_gates": failed,
        "baseline_metrics": (report.get("baseline") or {}).get("metrics", {}) if isinstance(report, dict) else {},
        "adapter_metrics": (report.get("adapter") or {}).get("metrics", {}) if isinstance(report, dict) else {},
        "holdout": report.get("holdout") if isinstance(report, dict) else None,
    }


def stage_artifact(args: argparse.Namespace, key: str) -> tuple[Path, Path]:
    artifact_dir = args.artifact_root / f"{safe_slug(key)}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    review_md = artifact_dir / "REVIEW.md"
    summary_json = artifact_dir / "summary.json"
    copy_if_exists(args.report_md, review_md)
    copy_if_exists(args.report_json, artifact_dir / args.report_json.name)
    summary = {
        "lane": args.lane,
        "adapter_dir": args.adapter_dir,
        "run_dir": str(args.run_dir),
        "report_md": str(args.report_md),
        "report_json": str(args.report_json),
        "promotion_ready": args.promotion_ready == "true",
        "shadow_only": True,
        "shadow_only_note": SHADOW_ONLY_NOTE,
        "created_at": utc_now(),
        "report_summary": load_report_summary(args.report_json),
    }
    write_json(summary_json, summary)
    log_path = artifact_dir / "run.log"
    log_path.write_text(
        "Adapter eval gate staged for Mission Control Automation Review.\n"
        f"Lane: {args.lane}\nAdapter: {args.adapter_dir}\nRun dir: {args.run_dir}\nReport: {args.report_md}\n{SHADOW_ONLY_NOTE}\n",
        encoding="utf-8",
    )
    return artifact_dir, log_path


def find_existing_review(reviews: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    marker = f"adapter_eval_key={key}"
    for review in reviews:
        notes = str(review.get("notes") or "")
        labels = [str(label) for label in review.get("labels") or []]
        if marker in notes or key in labels:
            if review.get("status") in {"approved", "rejected"}:
                continue
            return review
    return None


def upsert_review(args: argparse.Namespace, artifact_dir: Path, log_path: Path, key: str) -> dict[str, Any]:
    state = load_json(args.state_path, {})
    now = utc_now()
    if not isinstance(state, dict):
        state = {}
    reviews = state.get("runReviews")
    if not isinstance(reviews, list):
        reviews = []
    promotion_ready = args.promotion_ready == "true"
    title = f"Adapter eval gate: {args.lane}"
    summary = args.summary or (
        "Adapter comparison passed all shadow-mode gates; operator approval may allow shadow-mode only."
        if promotion_ready
        else "Adapter comparison failed one or more gates; do not promote. Review report before any shadow-mode trial."
    )
    labels = [
        "adapter-eval-gate",
        args.lane,
        "shadow-mode-only",
        "promotion-ready" if promotion_ready else "do-not-promote",
        key,
    ]
    notes = "\n".join([
        f"adapter_eval_key={key}",
        f"adapter_dir={args.adapter_dir}",
        f"run_dir={args.run_dir}",
        SHADOW_ONLY_NOTE,
    ])
    review = find_existing_review(reviews, key)
    if review:
        review.update({
            "title": title,
            "runType": "experiment",
            "risk": "medium" if promotion_ready else "high",
            "status": "needs_review",
            "summary": summary,
            "artifactPath": str(artifact_dir),
            "logPath": str(log_path),
            "labels": labels[:12],
            "notes": notes,
            "taskId": TASK_ID,
            "updatedAt": now,
        })
    else:
        review = {
            "id": review_id(),
            "appName": "ai-training",
            "title": title,
            "runType": "experiment",
            "risk": "medium" if promotion_ready else "high",
            "status": "needs_review",
            "summary": summary,
            "artifactPath": str(artifact_dir),
            "logPath": str(log_path),
            "labels": labels[:12],
            "notes": notes,
            "taskId": TASK_ID,
            "createdAt": now,
            "updatedAt": now,
        }
        reviews.insert(0, review)
    state["runReviews"] = reviews[:250]
    state["schemaVersion"] = max(int(state.get("schemaVersion") or 0), 2)
    state["updatedAt"] = now
    write_json(args.state_path, state)
    return review


def main() -> int:
    args = parse_args()
    key = f"adapter-eval-{args.lane}-{safe_slug(args.adapter_dir)}"
    artifact_dir, log_path = stage_artifact(args, key)
    review = upsert_review(args, artifact_dir, log_path, key)
    print(json.dumps({"review_id": review["id"], "artifact_dir": str(artifact_dir), "status": review["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
