#!/usr/bin/env python3
"""Run baseline-vs-adapter Gemma orchestrator eval and write promotion-gated reports."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STRICT_ROUTE_GATE = 0.85
VALID_JSON_GATE = 0.95
MANUAL_GATE_MISSES_GATE = 0


def run_eval(label: str, script: Path, out: Path, dataset_dir: str, split: str, base_model: str, adapter_dir: str = '') -> dict:
    cmd = [
        sys.executable,
        str(script),
        '--dataset-dir',
        dataset_dir,
        '--split',
        split,
        '--base-model',
        base_model,
        '--out',
        str(out),
        '--repair-json',
    ]
    if adapter_dir:
        cmd += ['--adapter-dir', adapter_dir]
    print(json.dumps({'event': 'run_eval', 'label': label, 'cmd': cmd}, indent=2))
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text(encoding='utf-8'))


def fmt_metric(value: float) -> str:
    return f'{value:.2f}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default='/mnt/s/AITraining/datasets/gemma-orchestrator-v0.1-curated')
    ap.add_argument('--split', default='eval')
    ap.add_argument('--base-model', default='unsloth/gemma-3-1b-it-unsloth-bnb-4bit')
    ap.add_argument('--adapter-dir', required=True)
    ap.add_argument('--run-dir', default='')
    ap.add_argument('--enforce-gates', action='store_true', help='Exit non-zero when adapter misses promotion gates.')
    args = ap.parse_args()

    run_id = Path(args.adapter_dir).parent.name if args.adapter_dir else 'baseline'
    run_dir = Path(args.run_dir or f'/mnt/s/AITraining/runs/{run_id}-comparison-{time.strftime("%Y%m%d-%H%M%S")}')
    run_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('eval_gemma_orchestrator_adapter.py')

    baseline = run_eval('baseline', script, run_dir / 'eval-baseline.json', args.dataset_dir, args.split, args.base_model)
    adapter = run_eval('adapter', script, run_dir / 'eval-adapter.json', args.dataset_dir, args.split, args.base_model, args.adapter_dir)

    adapter_metrics = adapter['metrics']
    gate_results = {
        'strict_route_accuracy': {
            'threshold': STRICT_ROUTE_GATE,
            'actual': adapter_metrics.get('strict_route_accuracy', 0),
            'passed': adapter_metrics.get('strict_route_accuracy', 0) >= STRICT_ROUTE_GATE,
        },
        'valid_json_rate': {
            'threshold': VALID_JSON_GATE,
            'actual': adapter_metrics.get('strict_schema_valid_rate', adapter_metrics.get('valid_json_rate', 0)),
            'passed': adapter_metrics.get('strict_schema_valid_rate', adapter_metrics.get('valid_json_rate', 0)) >= VALID_JSON_GATE,
        },
        'lane_agreement': {
            'threshold': STRICT_ROUTE_GATE,
            'actual': adapter_metrics.get('lane_agreement', adapter_metrics.get('lane_accuracy', 0)),
            'passed': adapter_metrics.get('lane_agreement', adapter_metrics.get('lane_accuracy', 0)) >= STRICT_ROUTE_GATE,
        },
        'manual_gate_misses': {
            'threshold': MANUAL_GATE_MISSES_GATE,
            'actual': adapter_metrics.get('manual_gate_misses', 0),
            'passed': adapter_metrics.get('manual_gate_misses', 0) == MANUAL_GATE_MISSES_GATE,
        },
    }
    promotion_ready = all(g['passed'] for g in gate_results.values())
    comparison = {
        'dataset_dir': args.dataset_dir,
        'split': args.split,
        'base_model': args.base_model,
        'adapter_dir': args.adapter_dir,
        'run_dir': str(run_dir),
        'promotion_gates': gate_results,
        'promotion_ready': promotion_ready,
        'baseline': {'path': str(run_dir / 'eval-baseline.json'), 'metrics': baseline['metrics']},
        'adapter': {'path': str(run_dir / 'eval-adapter.json'), 'metrics': adapter_metrics},
    }
    (run_dir / 'comparison-report.json').write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding='utf-8')

    rows = []
    for label, report in [('baseline', baseline), ('adapter', adapter)]:
        m = report['metrics']
        rows.append(
            f"| {label} | {fmt_metric(m.get('strict_schema_valid_rate', m['valid_json_rate']))} | {fmt_metric(m.get('repaired_json_rate', 0))} | "
            f"{fmt_metric(m.get('lane_agreement', m['lane_accuracy']))} | {fmt_metric(m['priority_accuracy'])} | {fmt_metric(m['worker_accuracy'])} | {fmt_metric(m['strict_route_accuracy'])} | {m.get('manual_gate_misses', 0)} |"
        )
    gates = '\n'.join(
        f"- {name}: {fmt_metric(g['actual']) if isinstance(g['actual'], float) else g['actual']} / required {fmt_metric(g['threshold']) if isinstance(g['threshold'], float) else g['threshold']} -> {'PASS' if g['passed'] else 'FAIL'}"
        for name, g in gate_results.items()
    )
    md = f"""# Gemma orchestrator adapter comparison\n\nDataset: `{args.dataset_dir}` ({args.split})\n\nAdapter: `{args.adapter_dir}`\n\n## Metrics\n\n| Run | Strict schema valid | JSON repaired | Lane agreement | Priority | Worker | Strict route | Manual-gate misses |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n{chr(10).join(rows)}\n\n## Promotion gates\n\n{gates}\n\nPromotion decision: **{'ready_for_promotion' if promotion_ready else 'do_not_promote'}**\n\n## Notes\n\n- `strict_schema_valid_rate` requires the final evaluated output to contain the required `safety_gate` key and no unexpected keys.\n- Deterministic policy overrides are applied in eval so credentials/legal/payment/publishing/live Discord/game/hardware/manual acceptance cannot route away from `manual_gate`.\n- Promotion requires strict_schema_valid_rate >= {VALID_JSON_GATE:.2f}, lane_agreement/strict_route_accuracy >= {STRICT_ROUTE_GATE:.2f}, and zero manual_gate misses.\n- Baseline eval JSON: `{run_dir / 'eval-baseline.json'}`\n- Adapter eval JSON: `{run_dir / 'eval-adapter.json'}`\n"""
    (run_dir / 'comparison-report.md').write_text(md, encoding='utf-8')
    print(json.dumps({'event': 'done', 'run_dir': str(run_dir), 'promotion_ready': promotion_ready, 'gates': gate_results}, indent=2))
    if args.enforce_gates and not promotion_ready:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
