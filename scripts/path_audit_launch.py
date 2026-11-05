#!/usr/bin/env python3
"""Register supervising Linux threads with zenbench, then execute the benchmark.

zenbench 0.1.11 excludes ancestor leaders but not their task IDs in its rival
benchmark scan. Register only verified live ancestor tasks; unrelated benchmark
processes remain visible to admission and all resource gates remain enabled.
"""
import os
from pathlib import Path
import sys


def main():
    excluded = set(filter(None, os.environ.get("ZENBENCH_LAUNCHER_PIDS", "").split(",")))
    pid = os.getppid()
    visited = set()
    while pid and pid not in visited:
        visited.add(pid)
        proc = Path("/proc") / str(pid)
        try:
            excluded.update(p.name for p in (proc / "task").iterdir())
            status = (proc / "status").read_text()
        except FileNotFoundError:
            break
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    os.environ["ZENBENCH_LAUNCHER_PIDS"] = ",".join(sorted(excluded, key=int))
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
