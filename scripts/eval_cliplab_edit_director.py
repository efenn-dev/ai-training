#!/usr/bin/env python3
"""Evaluate ClipLab Edit Director baseline or adapter outputs."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from cliplab_edit_director_common import (
    DEFAULT_BASE_MODEL,
    SCHEMA_VERSION,
    clean_list,
    clean_text,
    compact_json,
    compute_eval_metrics,
    default_dataset_dir,
    extract_json_object,
    first_clip,
    load_jsonl,
    normalize_decision,
    parse_assistant_payload,
    parse_user_payload,
    to_float,
    validate_edit_director_response,
    write_json,
)
from train_cliplab_edit_director import gpu_snapshot, load_unsloth_model

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--split", default="eval", choices=["train", "eval", "holdout_real_runs"])
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--heuristic-baseline", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repair-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render system+user messages into an inference prompt."""
    prompt_messages = messages[:2]
    try:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        system = prompt_messages[0]["content"]
        user = prompt_messages[1]["content"]
        return "<start_of_turn>user\n" + system + "\n\n" + user + "<end_of_turn>\n<start_of_turn>model\n"


def heuristic_prediction(row: dict[str, Any]) -> dict[str, Any]:
    """Predict using the current deterministic pipeline/orchestrator signals."""
    payload = parse_user_payload(row)
    clips = payload.get("candidate_clips") if isinstance(payload.get("candidate_clips"), list) else []
    clip = clips[0] if clips and isinstance(clips[0], dict) else {}
    window = clip.get("window_sec") if isinstance(clip.get("window_sec"), dict) else {}
    ai_review = clip.get("ai_review") if isinstance(clip.get("ai_review"), dict) else {}
    risk_flags = clean_list(ai_review.get("riskFlags"), limit=12, item_limit=80)
    action = clean_text(clip.get("pipeline_action") or (clip.get("scoring_trace") or {}).get("action"), 40).lower()
    verdict = clean_text(ai_review.get("verdict"), 40).lower()
    overall = to_float(ai_review.get("overallScore"), 0.0)
    reject_risks = {"idle_menu_risk", "low_payoff", "weak_title", "game_state_conflict"}
    if action in {"drop", "reject", "skip"}:
        decision = "reject"
    elif verdict == "weak" and (overall < 48 or reject_risks.intersection({r.lower() for r in risk_flags})):
        decision = "reject"
    else:
        decision = "keep"
    start = to_float(window.get("start"), 0.0)
    end = to_float(window.get("end"), start)
    safety_mode = payload.get("safety_mode") if isinstance(payload.get("safety_mode"), dict) else {}
    reasons = clean_list(
        [
            action or "pipeline_decision",
            clip.get("pipeline_reason"),
            *risk_flags,
        ],
        limit=10,
        item_limit=100,
    )
    if not reasons:
        reasons = ["pipeline_baseline"]
    prediction = {
        "schema_version": SCHEMA_VERSION,
        "clips": [
            {
                "clip_id": str(row.get("clip_id") or clip.get("clip_id") or ""),
                "decision": decision,
                "trim_start_sec": round(start, 3),
                "trim_end_sec": round(max(start, end), 3),
                "reasons": reasons,
                "platform_fit": "ok" if decision == "keep" else "weak",
                "safety_notes": [],
                "metadata_title": clean_text(clip.get("title"), 180),
                "metadata_notes": "",
            }
        ],
        "run_notes": "",
        "safety": {
            "child_safe": bool(safety_mode.get("child_safe", True)),
            "sponsor_safe": bool(safety_mode.get("sponsor_safe", True)),
        },
    }
    error = validate_edit_director_response(prediction)
    if error:
        raise ValueError(error)
    return prediction


def load_eval_model(args: argparse.Namespace) -> tuple[Any, Any, Any, str, dict[str, Any]]:
    """Load base model and optional adapter for eval."""
    import torch
    from peft import PeftModel

    loader, model, tokenizer, loader_name = load_unsloth_model(args.base_model, args.max_seq_length)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    try:
        loader.for_inference(model)
    except Exception:  # noqa: BLE001
        model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, torch, loader_name, gpu_snapshot(torch)


def evaluate_heuristic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate the deterministic baseline."""
    eval_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected = parse_assistant_payload(row)
        parsed = heuristic_prediction(row)
        expected_clip = first_clip(expected) or {}
        parsed_clip = first_clip(parsed) or {}
        eval_rows.append(
            {
                "index": index,
                "id": row.get("id"),
                "run_id": row.get("run_id"),
                "clip_id": row.get("clip_id"),
                "expected": expected,
                "parsed": parsed,
                "decoded": compact_json(parsed),
                "parse_error": None,
                "json_repair": None,
                "expected_decision": normalize_decision(expected_clip.get("decision")),
                "predicted_decision": normalize_decision(parsed_clip.get("decision")),
                "has_human_label": bool(row.get("has_human_label")),
                "has_human_trim_label": bool(row.get("has_human_trim_label")),
            }
        )
    return eval_rows


def evaluate_model(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a model or adapter."""
    import torch

    model, tokenizer, torch_module, loader_name, gpu_after_load = load_eval_model(args)
    eval_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt = render_prompt(tokenizer, row["messages"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        parsed, error, repair_note = extract_json_object(decoded, repair=args.repair_json)
        expected = parse_assistant_payload(row)
        expected_clip = first_clip(expected) or {}
        parsed_clip = first_clip(parsed) or {}
        eval_rows.append(
            {
                "index": index,
                "id": row.get("id"),
                "run_id": row.get("run_id"),
                "clip_id": row.get("clip_id"),
                "expected": expected,
                "parsed": parsed,
                "decoded": decoded,
                "parse_error": error,
                "json_repair": repair_note,
                "expected_decision": normalize_decision(expected_clip.get("decision")),
                "predicted_decision": normalize_decision(parsed_clip.get("decision")),
                "has_human_label": bool(row.get("has_human_label")),
                "has_human_trim_label": bool(row.get("has_human_trim_label")),
            }
        )
        log.info(
            "case %d/%d expected=%s predicted=%s parse_error=%s",
            index + 1,
            len(rows),
            eval_rows[-1]["expected_decision"],
            eval_rows[-1]["predicted_decision"],
            error,
        )
    telemetry = {
        "unsloth_loader": loader_name,
        "gpu_after_load": gpu_after_load,
        "gpu_final": gpu_snapshot(torch_module),
    }
    return eval_rows, telemetry


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    dataset_file = args.dataset_dir / f"{args.split}.jsonl"
    rows = load_jsonl(dataset_file)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found in {dataset_file}")

    if args.heuristic_baseline:
        eval_rows = evaluate_heuristic(rows)
        extra_telemetry: dict[str, Any] = {"heuristic_baseline": True}
    else:
        eval_rows, extra_telemetry = evaluate_model(rows, args)

    metrics = compute_eval_metrics(eval_rows)
    report = {
        "label": args.label or ("heuristic-baseline" if args.heuristic_baseline else "adapter" if args.adapter_dir else "base-model"),
        "dataset_file": str(dataset_file),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "row_count": len(rows),
        "elapsed_sec": round(time.time() - t0, 3),
        "metrics": metrics,
        "telemetry": extra_telemetry,
        "rows": eval_rows,
    }
    write_json(args.out, report)
    log.info("Wrote eval report to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
