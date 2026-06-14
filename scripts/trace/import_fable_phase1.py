#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_common import import_fable_phase1, summarize_trace_store, write_recommended_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Import processed Fable Phase 1 trace artifacts into S:/AITraining/traces.")
    parser.add_argument("--source-root", default="/mnt/c/projects/fable-trace-lora")
    args = parser.parse_args()
    manifest = import_fable_phase1(Path(args.source_root))
    write_recommended_sources()
    print(json.dumps({"manifest": manifest, "summary": summarize_trace_store()}, indent=2))


if __name__ == "__main__":
    main()
