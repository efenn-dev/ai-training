#!/usr/bin/env python3
"""Compare ClipLab Edit Director adapter eval against the current pipeline baseline."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cliplab_edit_director_common import (
    DEFAULT_BASE_MODEL,
    default_dataset_dir,
    default_runs_dir,
    write_json,
)

log = logging.getLogger(__name__)

STRICT_SCHEMA_GATE = 0.98
SAFETY_MISSES_GATE = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--split", default="eval", choices=["eval", "holdout_real_runs", "train"])
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-holdout", action="store_true", help="Skip optional holdout_real_runs comparison even when rows exist.")
    parser.add_argument("--no-review", action="store_true", help="Do not create/update the Mission Control Automation Review gate.")
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def jsonl_has_rows(path: Path) -> bool:
    """Return whether a JSONL file exists and contains at least one non-empty row."""
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def run_eval(
    *,
    label: str,
    script: Path,
    out: Path,
    dataset_dir: Path,
    split: str,
    base_model: str,
    max_seq_length: int,
    max_new_tokens: int,
    limit: int,
    adapter_dir: Path | None = None,
    heuristic: bool = False,
) -> dict[str, Any]:
    """Run eval script and return the JSON report."""
    cmd = [
        sys.executable,
        str(script),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        split,
        "--base-model",
        base_model,
        "--out",
        str(out),
        "--label",
        label,
        "--max-seq-length",
        str(max_seq_length),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if heuristic:
        cmd.append("--heuristic-baseline")
    if adapter_dir:
        cmd.extend(["--adapter-dir", str(adapter_dir)])
    log.info("Running %s eval: %s", label, " ".join(cmd))
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def metric(report: dict[str, Any], key: str, default: Any = 0.0) -> Any:
    """Return one metric from an eval report."""
    return (report.get("metrics") or {}).get(key, default)


def gate_report(baseline: dict[str, Any], adapter: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build promotion gate results."""
    baseline_f1 = float(metric(baseline, "keep_reject_macro_f1", 0.0) or 0.0)
    adapter_f1 = float(metric(adapter, "keep_reject_macro_f1", 0.0) or 0.0)
    baseline_trim = metric(baseline, "trim_boundary_mae_sec", None)
    adapter_trim = metric(adapter, "trim_boundary_mae_sec", None)
    trim_gate_passed = baseline_trim is not None and adapter_trim is not None and float(adapter_trim) < float(baseline_trim)
    human_rows = int(metric(adapter, "human_labeled_rows", 0) or 0)
    num_examples = int(metric(adapter, "num_examples", 0) or 0)
    human_trim_rows = int(metric(adapter, "human_trim_labeled_rows", 0) or 0)
    gates = {
        "strict_schema_valid_rate": {
            "threshold": STRICT_SCHEMA_GATE,
            "actual": float(metric(adapter, "strict_schema_valid_rate", 0.0) or 0.0),
            "passed": float(metric(adapter, "strict_schema_valid_rate", 0.0) or 0.0) >= STRICT_SCHEMA_GATE,
        },
        "keep_reject_macro_f1_beats_baseline": {
            "baseline": baseline_f1,
            "actual": adapter_f1,
            "passed": adapter_f1 > baseline_f1,
        },
        "trim_boundary_mae_beats_baseline": {
            "baseline": baseline_trim,
            "actual": adapter_trim,
            "passed": trim_gate_passed,
            "note": "Requires corrected human trim labels; blocked when trim MAE is unavailable or tied.",
        },
        "child_safe_misses": {
            "threshold": SAFETY_MISSES_GATE,
            "actual": int(metric(adapter, "child_safe_misses", 0) or 0),
            "passed": int(metric(adapter, "child_safe_misses", 0) or 0) == SAFETY_MISSES_GATE,
        },
        "sponsor_safe_misses": {
            "threshold": SAFETY_MISSES_GATE,
            "actual": int(metric(adapter, "sponsor_safe_misses", 0) or 0),
            "passed": int(metric(adapter, "sponsor_safe_misses", 0) or 0) == SAFETY_MISSES_GATE,
        },
        "human_labeled_eval": {
            "threshold": num_examples,
            "actual": human_rows,
            "passed": num_examples > 0 and human_rows == num_examples,
            "note": "No promotion if eval is synthetic/current-pipeline-only without human labels.",
        },
        "human_trim_labeled_eval": {
            "threshold": num_examples,
            "actual": human_trim_rows,
            "passed": num_examples > 0 and human_trim_rows == num_examples,
            "note": "Needed before claiming trim-boundary improvement.",
        },
        "fake_verification_claims": {
            "threshold": 0,
            "actual": int(metric(adapter, "fake_verification_claims", 0) or 0),
            "passed": int(metric(adapter, "fake_verification_claims", 0) or 0) == 0,
        },
    }
    return gates


def holdout_gate_report(baseline: dict[str, Any], adapter: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build holdout-specific gates without requiring trim labels to be present."""
    baseline_f1 = float(metric(baseline, "keep_reject_macro_f1", 0.0) or 0.0)
    adapter_f1 = float(metric(adapter, "keep_reject_macro_f1", 0.0) or 0.0)
    return {
        "holdout_macro_f1_matches_or_beats_baseline": {
            "baseline": baseline_f1,
            "actual": adapter_f1,
            "passed": adapter_f1 >= baseline_f1,
        },
        "holdout_child_safe_misses": {
            "threshold": SAFETY_MISSES_GATE,
            "actual": int(metric(adapter, "child_safe_misses", 0) or 0),
            "passed": int(metric(adapter, "child_safe_misses", 0) or 0) == SAFETY_MISSES_GATE,
        },
        "holdout_sponsor_safe_misses": {
            "threshold": SAFETY_MISSES_GATE,
            "actual": int(metric(adapter, "sponsor_safe_misses", 0) or 0),
            "passed": int(metric(adapter, "sponsor_safe_misses", 0) or 0) == SAFETY_MISSES_GATE,
        },
    }


def fmt(value: Any) -> str:
    """Format metrics for markdown."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    adapter: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    promotion_ready: bool,
) -> None:
    """Write a human-readable comparison report."""
    metric_names = [
        "strict_schema_valid_rate",
        "keep_reject_macro_f1",
        "decision_accuracy",
        "trim_boundary_mae_sec",
        "child_safe_misses",
        "sponsor_safe_misses",
        "human_labeled_rows",
        "human_trim_labeled_rows",
        "fake_verification_claims",
    ]
    rows = []
    for name in metric_names:
        rows.append(f"| {name} | {fmt(metric(baseline, name, None))} | {fmt(metric(adapter, name, None))} |")
    gate_lines = []
    for name, gate in gates.items():
        required = gate.get("threshold", gate.get("baseline"))
        gate_lines.append(
            f"- {name}: actual `{fmt(gate.get('actual'))}`, required/baseline `{fmt(required)}` -> "
            f"{'PASS' if gate.get('passed') else 'FAIL'}"
        )
        if gate.get("note"):
            gate_lines.append(f"  {gate['note']}")
    md = f"""# ClipLab Edit Director Adapter Comparison

Dataset: `{args.dataset_dir}` (`{args.split}`)

Adapter: `{args.adapter_dir}`

## Metrics

| Metric | Pipeline baseline | Adapter |
| --- | ---: | ---: |
{chr(10).join(rows)}

## Promotion Gates

{chr(10).join(gate_lines)}

Promotion decision: **{'ready_for_promotion' if promotion_ready else 'do_not_promote'}**

## Notes

- Baseline is the current deterministic ClipLab pipeline/orchestrator decision derived from the same candidate artifacts.
- Promotion requires strict schema validity >= {STRICT_SCHEMA_GATE:.2f}, better keep/reject macro F1, better trim-boundary MAE, zero child/sponsor safety misses, zero fake-verification claims, and fully human-labeled eval rows.
- If trim labels are absent, this report intentionally blocks promotion even if keep/reject improves.
"""
    path.write_text(md, encoding="utf-8")


def append_holdout_and_shadow_note(path: Path, holdout: dict[str, Any] | None, promotion_ready: bool) -> None:
    """Append holdout and shadow-mode-only recommendation details to an existing report."""
    if holdout:
        metric_names = [
            "strict_schema_valid_rate",
            "keep_reject_macro_f1",
            "decision_accuracy",
            "trim_boundary_mae_sec",
            "child_safe_misses",
            "sponsor_safe_misses",
        ]
        rows = [
            f"| {name} | {fmt(metric(holdout['baseline'], name, None))} | {fmt(metric(holdout['adapter'], name, None))} |"
            for name in metric_names
        ]
        gate_lines = []
        for name, gate in holdout["gates"].items():
            required = gate.get("threshold", gate.get("baseline"))
            gate_lines.append(
                f"- {name}: actual `{fmt(gate.get('actual'))}`, required/baseline `{fmt(required)}` -> "
                f"{'PASS' if gate.get('passed') else 'FAIL'}"
            )
        holdout_section = (
            "\n## Holdout Performance\n\n"
            "| Metric | Pipeline baseline | Adapter |\n"
            "| --- | ---: | ---: |\n"
            f"{chr(10).join(rows)}\n\n"
            f"Holdout gates:\n\n{chr(10).join(gate_lines)}\n"
        )
    else:
        holdout_section = "\n## Holdout Performance\n\nNo holdout_real_runs rows were evaluated.\n"
    recommendation = "shadow_mode_candidate" if promotion_ready else "do_not_promote"
    shadow_note = (
        "\n## Mission Control Review Gate\n\n"
        f"Recommendation: **{recommendation}**. Approval authorizes shadow-mode only; live production "
        "routing/editing requires a separate integration task.\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(holdout_section + shadow_note)


def upsert_review(args: argparse.Namespace, run_dir: Path, promotion_ready: bool) -> None:
    """Create or update the Mission Control Automation Review gate."""
    if args.no_review:
        return
    helper = Path(__file__).with_name("adapter_review_gate.py")
    summary = (
        "ClipLab edit-director adapter passed eval gates; approval allows shadow-mode only."
        if promotion_ready
        else "ClipLab edit-director adapter failed eval gates; do not promote."
    )
    cmd = [
        sys.executable,
        str(helper),
        "--lane",
        "gaming-pipeline-edit-director",
        "--adapter-dir",
        str(args.adapter_dir),
        "--run-dir",
        str(run_dir),
        "--report-md",
        str(run_dir / "comparison-report.md"),
        "--report-json",
        str(run_dir / "comparison-report.json"),
        "--promotion-ready",
        "true" if promotion_ready else "false",
        "--summary",
        summary,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    run_id = args.adapter_dir.parent.name if args.adapter_dir else f"cliplab-edit-director-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.run_dir or (default_runs_dir() / f"{run_id}-comparison-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("eval_cliplab_edit_director.py")

    baseline = run_eval(
        label="pipeline-baseline",
        script=script,
        out=run_dir / "eval-pipeline-baseline.json",
        dataset_dir=args.dataset_dir,
        split=args.split,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        heuristic=True,
    )
    adapter = run_eval(
        label="adapter",
        script=script,
        out=run_dir / "eval-adapter.json",
        dataset_dir=args.dataset_dir,
        split=args.split,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )

    gates = gate_report(baseline, adapter)
    holdout: dict[str, Any] | None = None
    holdout_file = args.dataset_dir / "holdout_real_runs.jsonl"
    if not args.no_holdout and args.split != "holdout_real_runs" and jsonl_has_rows(holdout_file):
        holdout_baseline = run_eval(
            label="pipeline-baseline-holdout",
            script=script,
            out=run_dir / "eval-pipeline-baseline-holdout.json",
            dataset_dir=args.dataset_dir,
            split="holdout_real_runs",
            base_model=args.base_model,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            heuristic=True,
        )
        holdout_adapter = run_eval(
            label="adapter-holdout",
            script=script,
            out=run_dir / "eval-adapter-holdout.json",
            dataset_dir=args.dataset_dir,
            split="holdout_real_runs",
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
        holdout = {
            "baseline": holdout_baseline,
            "adapter": holdout_adapter,
            "gates": holdout_gate_report(holdout_baseline, holdout_adapter),
        }
    promotion_ready = all(gate["passed"] for gate in gates.values())
    if holdout:
        promotion_ready = promotion_ready and all(gate["passed"] for gate in holdout["gates"].values())
    comparison = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "run_dir": str(run_dir),
        "elapsed_sec": round(time.time() - t0, 3),
        "promotion_gates": gates,
        "holdout": {
            "baseline": {"path": str(run_dir / "eval-pipeline-baseline-holdout.json"), "metrics": holdout["baseline"].get("metrics", {})},
            "adapter": {"path": str(run_dir / "eval-adapter-holdout.json"), "metrics": holdout["adapter"].get("metrics", {})},
            "gates": holdout["gates"],
        }
        if holdout
        else None,
        "promotion_ready": promotion_ready,
        "baseline": {
            "path": str(run_dir / "eval-pipeline-baseline.json"),
            "metrics": baseline.get("metrics", {}),
        },
        "adapter": {
            "path": str(run_dir / "eval-adapter.json"),
            "metrics": adapter.get("metrics", {}),
        },
    }
    write_json(run_dir / "comparison-report.json", comparison)
    write_markdown(run_dir / "comparison-report.md", args, baseline, adapter, gates, promotion_ready)
    append_holdout_and_shadow_note(run_dir / "comparison-report.md", holdout, promotion_ready)
    upsert_review(args, run_dir, promotion_ready)
    log.info("Wrote comparison report to %s", run_dir)
    if args.enforce_gates and not promotion_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
