"""Follow fleet harvest: per-cell reference-clustered CIs for fleet-era MLP cells as they land.

POTENTIAL - ceiling, not a model score. A cell is fleet-era iff its directory has a
`fleet_receipt.json` (written by the fleet executor and kept by harvest_fit_cells.py). Dev-box
AVX-512 cells never get a CI from here and are never mixed into fleet aggregates. Each cell runs
`mlp_ci.py` (P0) or `p2_mlp_ci.py` (P2) under the shared heavy lock. When every one of the
960 P0 (or 960 P2) cells is fleet-era and has a CI, it also runs `summarize.py` and
`check_core_gates.py` on the harvested tree into /var/tmp/rev4-featpot/fleet_era/ (P0 only;
P2 aggregation stays with its own registered scripts). Idempotent; loops until stopped.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/var/tmp/rev4-featpot")
REPO = Path(__file__).resolve().parents[2]
HEAVY = str(Path.home() / "tmp/devin/heavy")
ENV = {**os.environ, "ZEN_PANEL_BIN": "/var/tmp/rev4-featpot/target/debug/panel",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
LOG = ROOT / "fleet_follow.log"
CELL = re.compile(r"POT_(?P<set>[a-z0-9_]+?)_(?P<arm>r0|minus_basic|p2_perm|p2)_mlp(?P<h>32|128)$")


def log(text: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} {text}"
    print(line, flush=True)
    with LOG.open("a") as stream:
        stream.write(line + "\n")


def cells(base: Path):
    for group in sorted(base.glob("POT_*_mlp*")):
        m = CELL.match(group.name)
        if not m:
            continue
        for cell in sorted(group.iterdir()):
            t = re.match(r"(?P<view>full|o[0-4])_r(?P<rep>[0-4])$", cell.name)
            if t and (cell / "result.json").is_file():
                yield m.groupdict(), t.groupdict(), cell


def run_ci(g: dict, v: dict, cell: Path) -> bool:
    script = "p2_mlp_ci.py" if g["arm"].startswith("p2") else "mlp_ci.py"
    cmd = [HEAVY, "--mem", "16G", "--jobs", "8", "--", "python", f"scripts/rev4_featpot/{script}",
           "--set", g["set"], "--arm", g["arm"], "--hidden", g["h"], "--rep", v["rep"]]
    if v["view"] != "full":
        cmd += ["--outer", v["view"][1]]
    proc = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
    (ROOT / f"fleet_ci_{cell.parent.name}_{cell.name}.log").write_text(proc.stdout + proc.stderr)
    return proc.returncode == 0 and (cell / "ci.json").is_file()


def main() -> None:
    while True:
        done = failed = 0
        counts = {"p0_fleet": 0, "p2_fleet": 0}
        for base, key in ((ROOT / "fits", "p0_fleet"), (ROOT / "p2/mlp", "p2_fleet")):
            for g, v, cell in cells(base):
                if not (cell / "fleet_receipt.json").is_file():
                    continue
                counts[key] += 1
                if (cell / "ci.json").is_file():
                    continue
                if run_ci(g, v, cell):
                    done += 1
                else:
                    failed += 1
                    log(f"CI FAILED {cell}")
        log(f"heartbeat fleet-era cells {counts} new CIs {done} failed {failed}")
        if counts["p0_fleet"] == 960 and not (ROOT / "fleet_era/p0_summary.json").exists():
            (ROOT / "fleet_era").mkdir(exist_ok=True)
            out = ROOT / "fleet_era/p0_summary.json"
            for cmd in ([HEAVY, "--mem", "16G", "--jobs", "8", "--", "python",
                         "scripts/rev4_featpot/summarize.py", "--out", str(out)],
                        [HEAVY, "--mem", "16G", "--jobs", "8", "--", "python",
                         "scripts/rev4_featpot/check_core_gates.py", "--summary", str(out)]):
                proc = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
                (ROOT / "fleet_era" / f"{Path(cmd[6]).stem}.log").write_text(proc.stdout + proc.stderr)
                log(f"{Path(cmd[6]).name} rc={proc.returncode}")
        time.sleep(600)


if __name__ == "__main__":
    sys.exit(main())
