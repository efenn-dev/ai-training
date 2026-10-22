#!/usr/bin/env python3
"""Small Hermes-style CLI wrapper around the local-first runner."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from local_first_runner import DEFAULT_PACKET_ROOT, DEFAULT_REPORT_ROOT, load_config, run_one, smoke_payloads, utc_now

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "local_frontier_router_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route one request through the local-first/Hermes fallback policy.")
    parser.add_argument("text", nargs="*", help="Request text. If omitted, stdin is used.")
    parser.add_argument("--task-type", default="summarization")
    parser.add_argument("--risk", default="low")
    parser.add_argument("--app", default="ai-training")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="Print the full JSON result.")
    parser.add_argument("--smoke", action="store_true", help="Run the built-in local-first smoke suite.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(DEFAULT_CONFIG)
    if args.smoke:
        DEFAULT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        results = [
            run_one(
                payload,
                cfg,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                log_shadow=True,
                packet_root=DEFAULT_PACKET_ROOT,
            )
            for payload in smoke_payloads()
        ]
        report = {
            "schema_version": "local-first-smoke-report-v1",
            "created_at": utc_now(),
            "case_count": len(results),
            "local_answer_count": sum(1 for item in results if item.get("final_mode") == "local_answer"),
            "frontier_required_count": sum(1 for item in results if item.get("final_mode") == "frontier_required"),
            "results": results,
        }
        report_path = DEFAULT_REPORT_ROOT / f"local-first-smoke-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
        else:
            print(f"report: {report_path}")
            print(f"local_answer_count: {report['local_answer_count']}")
            print(f"frontier_required_count: {report['frontier_required_count']}")
        return 0

    text = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
    if not text:
        raise SystemExit("Request text is required.")
    result = run_one(
        {
            "task_type": args.task_type,
            "risk": args.risk,
            "text": text,
            "app": args.app,
            "source": "hermes_local_first_cli",
        },
        cfg,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_shadow=True,
        packet_root=DEFAULT_PACKET_ROOT,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"mode: {result.get('final_mode')}")
    if result.get("model_used"):
        print(f"model: {result.get('model_used')}")
    if result.get("answer"):
        print("\n" + str(result["answer"]))
    elif result.get("frontier_packet"):
        print(f"frontier_packet: {result['frontier_packet']}")
        print(f"reason: {result.get('policy_decision', {}).get('reason')}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
