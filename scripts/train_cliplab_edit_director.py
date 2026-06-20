#!/usr/bin/env python3
"""Train the ClipLab Edit Director LoRA with Unsloth SFT/QLoRA."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cliplab_edit_director_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SMOKE_MODEL,
    default_adapters_dir,
    default_dataset_dir,
    validate_dataset_dir,
    write_json,
)

log = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--smoke-model", action="store_true", help=f"Use {DEFAULT_SMOKE_MODEL}.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60, help="Use -1 for epoch-based training.")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-response-only", action="store_true")
    parser.add_argument("--report-to", default="none", help="none, tensorboard, trackio, or comma-separated list.")
    parser.add_argument("--skip-eval-gate", action="store_true", help="Skip post-train adapter comparison and Mission Control review gate.")
    parser.add_argument("--eval-gate-limit", type=int, default=0, help="Optional row limit for the post-train eval gate.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render a row's chat messages into trainer text."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:  # noqa: BLE001
        system = messages[0]["content"]
        user = messages[1]["content"]
        assistant = messages[2]["content"]
        return (
            "<start_of_turn>user\n"
            + system
            + "\n\n"
            + user
            + "<end_of_turn>\n<start_of_turn>model\n"
            + assistant
            + "<end_of_turn>"
        )


def gpu_snapshot(torch_module: Any) -> dict[str, Any]:
    """Return current and peak CUDA memory telemetry."""
    if not torch_module.cuda.is_available():
        return {"cuda_available": False}
    device = torch_module.cuda.current_device()
    props = torch_module.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "total_vram_gb": round(props.total_memory / (1024**3), 3),
        "allocated_gb": round(torch_module.cuda.memory_allocated(device) / (1024**3), 3),
        "reserved_gb": round(torch_module.cuda.memory_reserved(device) / (1024**3), 3),
        "peak_allocated_gb": round(torch_module.cuda.max_memory_allocated(device) / (1024**3), 3),
        "peak_reserved_gb": round(torch_module.cuda.max_memory_reserved(device) / (1024**3), 3),
    }


def load_unsloth_model(model_name: str, max_seq_length: int) -> tuple[Any, Any, Any, str]:
    """Load a model through Unsloth, preferring the Gemma 4 FastModel path."""
    import unsloth  # noqa: F401 - apply Unsloth patches before transformers/peft

    load_errors: list[str] = []
    try:
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            full_finetuning=False,
        )
        return FastModel, model, tokenizer, "FastModel"
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"FastModel: {exc}")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        return FastLanguageModel, model, tokenizer, "FastLanguageModel"
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"FastLanguageModel: {exc}")
    raise RuntimeError("Could not load model with Unsloth: " + " | ".join(load_errors))


def build_sft_config(SFTConfig: Any, kwargs: dict[str, Any]) -> Any:
    """Build SFTConfig with compatibility fallbacks across TRL versions."""
    attempts = [dict(kwargs)]
    if "eval_strategy" in kwargs:
        alt = dict(kwargs)
        alt["evaluation_strategy"] = alt.pop("eval_strategy")
        attempts.append(alt)
    minimal = dict(kwargs)
    for key in ("max_length", "eval_strategy", "evaluation_strategy", "eval_steps"):
        minimal.pop(key, None)
    attempts.append(minimal)
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return SFTConfig(**attempt)
        except TypeError as exc:
            last_error = exc
    raise last_error or TypeError("could not construct SFTConfig")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    model_name = DEFAULT_SMOKE_MODEL if args.smoke_model else args.base_model
    run_id = args.run_id or f"cliplab-edit-director-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.out_dir or (default_adapters_dir() / run_id)
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_dataset_dir(args.dataset_dir)
    if validation["errors"]:
        write_json(out_dir / "DATASET_VALIDATION.json", validation)
        raise SystemExit(f"dataset validation failed; see {out_dir / 'DATASET_VALIDATION.json'}")
    row_counts = validation["counts"]
    if row_counts.get("train", 0) <= 0:
        raise SystemExit("No train rows found. Run curation after collecting AI Review accept/reject labels.")

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run inside the WSL Unsloth environment on a CUDA GPU.")
    torch.cuda.reset_peak_memory_stats()

    telemetry: dict[str, Any] = {
        "event": "train_cliplab_edit_director",
        "run_id": run_id,
        "model": model_name,
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(out_dir),
        "adapter_dir": str(adapter_dir),
        "row_counts": row_counts,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "num_epochs": args.num_epochs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log.info("Loading model %s", model_name)
    loader, model, tokenizer, loader_name = load_unsloth_model(model_name, args.max_seq_length)
    telemetry["unsloth_loader"] = loader_name
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    model = loader.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    telemetry["gpu_after_load"] = gpu_snapshot(torch)

    log.info("Loading dataset from %s", args.dataset_dir)
    data_files = {"train": str(args.dataset_dir / "train.jsonl")}
    if not args.no_eval and row_counts.get("eval", 0) > 0:
        data_files["eval"] = str(args.dataset_dir / "eval.jsonl")
    raw = load_dataset("json", data_files=data_files)
    if row_counts.get("unlabeled_candidates", 0) > 0:
        log.info(
            "Ignoring %s unlabeled candidate rows parked in %s; supervised SFT uses only train/eval labels.",
            row_counts["unlabeled_candidates"],
            args.dataset_dir / "unlabeled_candidates.jsonl",
        )
    train_ds = raw["train"]
    eval_ds = raw.get("eval")
    if args.max_train_rows and args.max_train_rows > 0:
        train_ds = train_ds.select(range(min(args.max_train_rows, len(train_ds))))
    if eval_ds is not None and args.max_eval_rows and args.max_eval_rows > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_rows, len(eval_ds))))

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        return {"text": render_messages(tokenizer, example["messages"])}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(format_example, remove_columns=eval_ds.column_names)
    telemetry["effective_rows"] = {
        "train": len(train_ds),
        "eval": len(eval_ds) if eval_ds is not None else 0,
    }

    report_to = [item.strip() for item in args.report_to.split(",") if item.strip() and item.strip() != "none"]
    save_steps = args.save_steps or (args.max_steps if args.max_steps and args.max_steps > 0 else 50)
    cfg_kwargs = {
        "output_dir": str(out_dir),
        "dataset_text_field": "text",
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "report_to": report_to or "none",
        "run_name": run_id,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "max_length": args.max_seq_length,
    }
    if eval_ds is not None:
        cfg_kwargs["eval_strategy"] = "steps"
        cfg_kwargs["eval_steps"] = max(1, args.eval_steps)
    training_args = build_sft_config(SFTConfig, cfg_kwargs)

    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
        )

    if not args.no_response_only:
        try:
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
            telemetry["response_only_masking"] = True
        except Exception as exc:  # noqa: BLE001
            telemetry["response_only_masking"] = False
            telemetry["response_only_masking_error"] = str(exc)
            log.warning("Could not enable response-only masking: %s", exc)

    log.info("Starting training: rows=%s max_steps=%s", len(train_ds), args.max_steps)
    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start
    train_metrics = getattr(train_result, "metrics", {}) or {}

    log.info("Saving adapter to %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    telemetry.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_sec": round(time.time() - t0, 3),
            "train_wall_time_sec": round(train_elapsed, 3),
            "final_loss": train_metrics.get("train_loss"),
            "train_metrics": train_metrics,
            "gpu_final": gpu_snapshot(torch),
            "adapter_path": str(adapter_dir),
        }
    )
    write_json(out_dir / "RUN_SUMMARY.json", telemetry)
    if not args.skip_eval_gate:
        compare_script = Path(__file__).with_name("compare_cliplab_edit_director.py")
        gate_run_dir = out_dir / "eval-gate"
        cmd = [
            sys.executable,
            str(compare_script),
            "--dataset-dir",
            str(args.dataset_dir),
            "--base-model",
            model_name,
            "--adapter-dir",
            str(adapter_dir),
            "--run-dir",
            str(gate_run_dir),
        ]
        if args.eval_gate_limit > 0:
            cmd.extend(["--limit", str(args.eval_gate_limit)])
        log.info("Running post-train eval gate: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            telemetry["eval_gate_report"] = str(gate_run_dir / "comparison-report.md")
        except subprocess.CalledProcessError as exc:
            telemetry["eval_gate_error"] = str(exc)
            write_json(out_dir / "RUN_SUMMARY.json", telemetry)
            raise
        write_json(out_dir / "RUN_SUMMARY.json", telemetry)
    log.info("Training complete. Summary: %s", out_dir / "RUN_SUMMARY.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
