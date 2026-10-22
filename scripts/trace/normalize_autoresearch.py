#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_common import sanitize_autoresearch_import, summarize_trace_store, write_recommended_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize AutoResearch imports for training review.")
    parser.add_argument("--import-root", type=Path, default=Path("/mnt/s/AITraining/traces/imports/karpathy-autoresearch"))
    parser.add_argument("--sanitized-root", type=Path, default=Path("/mnt/s/AITraining/traces/sanitized/karpathy-autoresearch"))
    parser.add_argument("--import-stamp", default=None, help="Optional import stamp folder under import-root (e.g., 20260622-022933).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = sanitize_autoresearch_import(
        import_stamp=args.import_stamp,
        import_root=args.import_root,
        sanitized_root=args.sanitized_root,
    )
    write_recommended_sources()
    print(json.dumps({"manifest": manifest, "summary": summarize_trace_store()}, indent=2))


if __name__ == "__main__":
    main()
