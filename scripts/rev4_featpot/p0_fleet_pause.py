"""Record the exact completed P0 cell boundary when fleet GO pauses a runner."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from linear_probe import ROOT, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=["serial", "accelerator", "handoff"], required=True)
    args = parser.parse_args()
    receipts = sorted((ROOT / "fits").glob("POT_*_mlp*/**/result.json"),
                      key=lambda path: (path.stat().st_mtime_ns, str(path)))
    last = receipts[-1] if receipts else None
    record = {"schema": "rev4-featpot-p0-fleet-pause-v1", "runner": args.runner,
              "completed_fit_cells": len(receipts),
              "last_completed_cell": str(last) if last else None,
              "last_completed_cell_sha256": sha(last) if last else None,
              "note": "fit result boundary; importance and CI may follow separately"}
    output = ROOT / f"p0_fleet_pause_{args.runner}.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    if args.runner == "handoff":
        processes = subprocess.check_output(["ps", "-ww", "-eo", "args="], text=True)
        active = [line for line in processes.splitlines()
                  if ("python scripts/rev4_featpot/mlp_probe.py" in line or
                      "python scripts/rev4_featpot/accelerate_mlp.py --batch" in line)
                  and "ps -ww" not in line]
        if active:
            raise RuntimeError(f"local P0 fit processes still active: {active}")
        report = ROOT / "p0_fleet_pause_handoff.json"
        paused = (Path.home() / "tmp/zensim-paper/rev4/FLEET_FITS_PAUSED.md")
        paused.write_text(
            "# Feature-potential P0 local MLP fleet handoff\n\n"
            f"Paused at {datetime.now(timezone.utc).isoformat()}.\n\n"
            f"Completed P0 fit receipts: **{len(receipts)}/960**.\n\n"
            f"Last completed cell: `{last}`; SHA-256 `{sha(last) if last else 'none'}`.\n\n"
            f"Handoff receipt: `{report}`; SHA-256 `{sha(report)}`.\n\n"
            "Both local P0 runner shells exited and no local P0 fit or accelerator process remained "
            "at this check. Importance and CI receipts may still be pending for completed fits.\n")
        manifest = (Path.home() / "tmp/devin/rev4_featbank-potential_manifest.tsv")
        with manifest.open("a") as stream:
            stream.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\tcreated\t{paused}\n")
    print(json.dumps({**record, "receipt": str(output), "receipt_sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
