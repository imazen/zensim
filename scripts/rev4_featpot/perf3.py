#!/usr/bin/env python3
"""PERF3 adapter over PERF2 and the unchanged SPEEDQ measurement owners."""
from pathlib import Path
import os
os.environ.setdefault("ZENSIM_PERF_ROOT", str(Path.home() / "tmp/perf3-20261018"))
os.environ.setdefault("ZENSIM_PERF_OWNER", "codex-perf3")
from perf2 import main
if __name__ == "__main__":
    main()
