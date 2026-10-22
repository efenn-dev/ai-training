#!/usr/bin/env python3
"""Merge a ClipLab Edit Director LoRA and export a q4_k_m GGUF."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from cliplab_edit_director_common import (
    DEFAULT_BASE_MODEL,
    default_merged_models_dir,
    write_json,
)
from train_cliplab_edit_director import gpu_snapshot, load_unsloth_model

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--quantization", default="q4_k_m")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_export_model(base_model: str, adapter_dir: Path, max_seq_length: int) -> tuple[Any, Any, Any, str, str]:
    """Load an adapter in the form Unsloth can merge/export."""
    import unsloth  # noqa: F401 - apply Unsloth patches
    from peft import PeftModel

    load_errors: list[str] = []
    for model_name, mode in ((str(adapter_dir), "adapter_direct"), (base_model, "base_plus_peft")):
        try:
            loader, model, tokenizer, loader_name = load_unsloth_model(model_name, max_seq_length)
            if mode == "base_plus_peft":
                model = PeftModel.from_pretrained(model, str(adapter_dir))
            return model, tokenizer, loader, loader_name, mode
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{mode}: {exc}")
    raise RuntimeError("Could not load adapter for export: " + " | ".join(load_errors))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    run_name = args.adapter_dir.parent.name if args.adapter_dir.name == "adapter" else args.adapter_dir.name
    output_dir = args.output_dir or (default_merged_models_dir() / f"{run_name}-{args.quantization}")
    merged_dir = output_dir / "merged_16bit"
    gguf_dir = output_dir / f"gguf_{args.quantization}"
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run inside the WSL Unsloth environment on a CUDA GPU.")
    torch.cuda.reset_peak_memory_stats()

    telemetry: dict[str, Any] = {
        "event": "export_cliplab_edit_director_gguf",
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "output_dir": str(output_dir),
        "merged_dir": str(merged_dir),
        "gguf_dir": str(gguf_dir),
        "quantization": args.quantization,
        "max_seq_length": args.max_seq_length,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    log.info("Loading adapter %s", args.adapter_dir)
    model, tokenizer, _loader, loader_name, load_mode = load_export_model(args.base_model, args.adapter_dir, args.max_seq_length)
    telemetry["unsloth_loader"] = loader_name
    telemetry["load_mode"] = load_mode
    telemetry["gpu_after_load"] = gpu_snapshot(torch)

    if not hasattr(model, "save_pretrained_merged"):
        raise SystemExit(
            "Loaded model does not expose save_pretrained_merged. "
            "Use an adapter saved by train_cliplab_edit_director.py in the Unsloth environment."
        )
    if not hasattr(model, "save_pretrained_gguf"):
        raise SystemExit(
            "Loaded model does not expose save_pretrained_gguf. "
            "Update/repair Unsloth before exporting GGUF."
        )

    log.info("Saving merged 16-bit model to %s", merged_dir)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    log.info("Saving GGUF %s to %s", args.quantization, gguf_dir)
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=args.quantization)

    telemetry.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_sec": round(time.time() - t0, 3),
            "gpu_final": gpu_snapshot(torch),
            "export_path": str(gguf_dir),
        }
    )
    write_json(output_dir / "EXPORT_SUMMARY.json", telemetry)
    log.info("Export complete. Summary: %s", output_dir / "EXPORT_SUMMARY.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
