#!/usr/bin/env python3
"""The near-lossless HUMAN axis, in the column shape `bake_verdict` publishes.

The exam's near-lossless corpora are ssim2 self-targets (see
`hfnl944_reachability.py`), so no model can beat the opponent on them. This
builds the non-circular replacement: the near-lossless ZONE of the gold human
corpus — CID22 rows whose human MOS falls in the top merged-decile band
(`merged-decile-2026-08-06`, lo = 0.80, open above; n = 1425 over 49
references) — for every arm the exam scores, INCLUDING `peer_ssim2`.

Every statistic comes from the `panel` binary (`--per-group` = the canonical
`zenstats::per_group_srocc`, the same quantity `bake_verdict` publishes as
`per_ref_mean` / `per_ref_n` / `frac_negative`). This script writes TSVs and
reads panel's JSON; it computes nothing. The paired confidence intervals come
from `paired_perref_boot.py` in this directory (`BAND_LO=0.8`), which is the
owner of the reference-clustered resample.

    hfnl944_band_table.py --out-dir /mnt/v/output/zensim/hfnl944-2026-09-01
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

O = Path(os.environ.get("O", "/mnt/v/output/zensim/ssim2-bar-2026-08-31"))
RM = Path("/mnt/v/output/zensim/reports/refmetrics")
REPO = Path(__file__).resolve().parents[2]
PANEL = Path(os.environ.get("ZEN_PANEL_BIN", REPO / "target/release/panel"))
# The top band of the committed scheme; `lo` is read from a board cell rather
# than hardcoded so a scheme change cannot silently move this axis.
BAND_SCHEME = "merged-decile-2026-08-06"
ARMS = ["ssim2", "B", "ADD156", "W10L9P", "W10L9PH", "Q7b"]


def band_lo_from_board(board: Path) -> float:
    d = json.loads(board.read_text())
    blk = d["rank"]["cid22"]
    assert blk["band_scheme"]["name"] == BAND_SCHEME, blk["band_scheme"]["name"]
    top = blk["bands"][-1]
    assert top["hi"] is None, "top band must be open above"
    return float(top["lo"])


def rows_for(arm: str):
    if arm == "ssim2":
        r = list(csv.DictReader(open(RM / "cid22_ssim2.tsv"), delimiter="\t"))
        return [(float(x["ssim2"]), float(x["MCOS"]) / 100.0,
                 os.path.basename(x["ref_path"])) for x in r]
    lines = [l.rstrip("\n").split("\t") for l in open(O / f"pp_{arm}_cid22.tsv") if l.strip()]
    i = {n: k for k, n in enumerate(lines[0])}
    return [(float(r[i["pred"]]), float(r[i["human"]]), r[i["ref"]]) for r in lines[1:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--board-dir", type=Path,
                    default=Path("/mnt/v/output/zensim/reports/fulleval"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lo = band_lo_from_board(a.board_dir / "W10L9PH_s4004_packed.fulleval.json")

    data = {arm: rows_for(arm) for arm in ARMS}
    n = len(data["ssim2"])
    base = [t for _, t, _ in data["ssim2"]]
    for k, v in data.items():
        assert len(v) == n, f"{k}: {len(v)} != {n}"
        d = max(abs(t - b) for (_, t, _), b in zip(v, base))
        assert d == 0.0, f"{k}: targets differ index-wise by {d} — pairing is a fiction"
    keep = [i for i in range(n) if base[i] >= lo]
    print(f"# band lo={lo} (open above, scheme {BAND_SCHEME}): {len(keep)} of {n} pairs, "
          f"span {max(base[i] for i in keep) - min(base[i] for i in keep):.6f}")

    out = {}
    for arm in ARMS:
        tsv = a.out_dir / f"band_{arm}_cid22hi.tsv"
        with open(tsv, "w") as f:
            f.write("predicted\ttarget\tband\n")
            for i in keep:
                p, t, r = data[arm][i]
                f.write(f"{p}\t{t}\t{r}\n")
        j = subprocess.run([str(PANEL), "--input", str(tsv), "--per-group", "--json"],
                           capture_output=True, text=True, check=True).stdout
        # NOTE (owner defect, reported not fixed by this lane): `panel --json
        # --per-group` writes TWO concatenated JSON documents — the `groups`
        # object and the `per_group` object — so the stream is not itself valid
        # JSON. `zen_stats.panel` never passes `--per-group`, which is why
        # nothing has tripped on it. Read both with raw_decode.
        dec, docs, k = json.JSONDecoder(), [], 0
        while k < len(j):
            while k < len(j) and j[k].isspace():
                k += 1
            if k >= len(j):
                break
            obj, k = dec.raw_decode(j, k)
            docs.append(obj)
        groups = next(d["groups"] for d in docs if "groups" in d)
        pg = next((d["per_group"] for d in docs if "per_group" in d), {})
        agg = next(g for g in groups if g["label"] == "ALL")
        out[arm] = {
            "n": agg["n"],
            "srocc": agg["srocc"],
            "plcc": agg.get("plcc"),
            "krocc": agg.get("krocc"),
            "pwrc": agg.get("pwrc"),
            "z_rmse": agg.get("z_rmse"),
            "per_ref_mean": pg.get("mean"),
            "per_ref_median": pg.get("median"),
            "per_ref_n": pg.get("n_groups"),
            "frac_negative": pg.get("frac_negative"),
        }
    # `panel --input` reports |SROCC| only; the SIGNED pooled value comes from
    # the batch path's `full` stats (the same owner, its other entry point). A
    # band is exactly where the sign matters (campaign appendix V), so it is
    # carried explicitly rather than assumed positive.
    sys.path.insert(0, str(REPO / "scripts" / "lib"))
    from zen_stats import panel_batch  # noqa: E402
    signed = panel_batch(
        [(f"sgn_{arm}", [data[arm][i][0] for i in keep], [base[i] for i in keep])
         for arm in ARMS], stats="full")
    for arm, r in zip(ARMS, signed):
        assert abs(abs(r["srocc_signed"]) - out[arm]["srocc"]) < 1e-9, arm
        out[arm]["srocc_signed"] = r["srocc_signed"]

    (a.out_dir / "cid22_hfnl_band_panel.json").write_text(
        json.dumps({"band": {"scheme": BAND_SCHEME, "lo": lo, "hi": None,
                             "n": len(keep), "corpus": "cid22"},
                    "arms": out}, indent=2, sort_keys=True) + "\n")

    cols = ["n", "srocc_signed", "per_ref_mean", "per_ref_n", "frac_negative"]
    print("arm\t" + "\t".join(cols))
    for arm in ARMS:
        print(arm + "\t" + "\t".join(
            ("" if out[arm][c] is None else
             (f"{out[arm][c]:.6f}" if isinstance(out[arm][c], float) else str(out[arm][c])))
            for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
