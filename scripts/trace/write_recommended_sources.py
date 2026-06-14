#!/usr/bin/env python3
from __future__ import annotations

import json

from trace_common import summarize_trace_store, write_recommended_sources


def main() -> None:
    path = write_recommended_sources()
    print(json.dumps({"wrote": str(path), "summary": summarize_trace_store()}, indent=2))


if __name__ == "__main__":
    main()
