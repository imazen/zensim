"""Build the confirmatory pin (`rev4-featpot-v2c-confirm-pin-v2`) from the frozen root and the canon exploratory results.

R2.1 short-list rule, recomputed here (never typed in by hand): eligible = (family, head) whose canon exploratory verdict passes V1
on >= 2 sources with no regression (`V1` in the v2_compare output); if more than 6, keep the 6 with the largest mean permutation excess
across the five exploratory sources (ties: head N first, then fewer added columns). None eligible -> refuse: no arm is read.

  python3 v2c_pin.py --root ROOT --compare-dir ROOT/compare --calibration ROOT/compare/calibration_h2.json --human-weight 2 \
      --program-sha H --data-sha H --labels LABELS.json --out PIN.json

LABELS.json = {set: v2c_labels spec} for the five sealed sets and the TERMINAL guard (original manifests, sha256 given).
Needs ZEN_PANEL_BIN (its sha256 is pinned) and a frozen root (`v2c_wide.py freeze`).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import v2_common
from v2_common import HEADS, load_frozen, sha

PIN_SCHEMA = "rev4-featpot-v2c-confirm-pin-v2"
MAX_SHORTLIST = 6


def mean_excess(rec: dict) -> float:
    return float(np.mean([v["excess"] for v in rec["sources"].values()]))


def shortlist(compare_dir: Path, arms: list[str], tag: str = "") -> tuple[list[dict], dict]:
    """(entries, provenance): eligible entries ranked by the R2.1 rule, capped at 6, with the compare files they came from.
    `tag` is v2_compare's weight suffix (`_h<w>` under amendment R3), so the files read are the ones it wrote."""
    found, prov = [], {}
    for arm in arms:
        for head in HEADS:
            path = compare_dir / f"{arm}_{head}{tag}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text())
            prov[f"{arm}_{head}"] = {"path": str(path), "sha256": sha(path)}
            if rec.get("status") == "OK" and len([s for s, v in rec["sources"].items() if v.get("v1_pass")]) >= 2 and not rec["regressions"]:
                width = len(v2_common.arm_columns(arm)[2]) - 944
                found.append({"arm": arm, "head": head, "mean_excess": mean_excess(rec), "added_columns": width})
    found.sort(key=lambda e: (-round(e["mean_excess"], 12), e["head"] != "N", e["added_columns"]))
    return found[:MAX_SHORTLIST], prov


def build(args) -> dict:
    root = Path(args.root)
    frozen, frozen_sha = load_frozen(root)
    tag = "" if args.human_weight is None else f"_h{args.human_weight:g}"
    entries, prov = shortlist(Path(args.compare_dir), args.arm, tag)
    if not entries:
        raise SystemExit("no eligible (family, head): no arm is read on the sealed sets (R2.1)")
    panel = os.environ.get("ZEN_PANEL_BIN")
    if not panel:
        raise SystemExit("ZEN_PANEL_BIN must be set")
    here = Path(__file__).resolve().parent
    code = {f"rev4_featpot/{n}": sha(here / n) for n in ("v2_confirm_read.py", "v2_compare.py", "v2_common.py", "v2c_labels.py", "v2c_wide.py")}
    code["lib/zen_stats.py"] = sha(here.parent / "lib" / "zen_stats.py")
    arms = sorted({e["arm"] for e in entries})
    return {"schema": PIN_SCHEMA, "reference": "r0", "candidates": arms, "heads": list(HEADS),
            "shortlist": [{"arm": e["arm"], "head": e["head"]} for e in entries],
            "shortlist_rank": [{k: e[k] for k in ("arm", "head", "mean_excess", "added_columns")} for e in entries],
            "human_weight": args.human_weight, "frozen_sha256": frozen_sha, "wide_receipts": frozen["wide_receipts"],
            "confirm_receipt_sha256": frozen["confirm_receipt_sha256"], "keep_lists_sha256": frozen["keep_lists_sha256"],
            "binaries": {"panel": sha(Path(panel)), **dict(b.split("=", 1) for b in args.binary)},
            "program_sha": args.program_sha, "data_sha": args.data_sha, "code": code,
            "shortlist_provenance": {"calibration": {"path": str(args.calibration), "sha256": sha(Path(args.calibration))}, "compare": prov},
            "labels": json.loads(Path(args.labels).read_text()),
            "table_code_note": "wide/confirm receipts record the (rewritten) build commit and a dirty working copy; script hashes are in table_code"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--compare-dir", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--arm", action="append", default=[], help="candidate arms considered (every arm with a compare file)")
    ap.add_argument("--human-weight", type=float, default=None)
    ap.add_argument("--program-sha", required=True)
    ap.add_argument("--data-sha", required=True)
    ap.add_argument("--binary", action="append", default=[], help="NAME=sha256 for zensim_mlp_train and bake_dial_refit")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.write_text(json.dumps(build(args), indent=1) + "\n")
    print(json.dumps({"pin": str(args.out), "sha256": sha(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
