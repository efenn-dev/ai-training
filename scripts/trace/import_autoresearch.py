#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from trace_common import import_autoresearch, summarize_trace_store, write_recommended_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and import the Karpathy AutoResearch repo into the AI-training trace store."
    )
    parser.add_argument("--repo-url", default="https://github.com/karpathy/autoresearch")
    parser.add_argument("--branch", default=None)
    args = parser.parse_args()

    manifest = import_autoresearch(
        repo_url=args.repo_url,
        branch=args.branch,
    )
    write_recommended_sources()
    print(json.dumps({"manifest": manifest, "summary": summarize_trace_store()}, indent=2))


if __name__ == "__main__":
    main()
