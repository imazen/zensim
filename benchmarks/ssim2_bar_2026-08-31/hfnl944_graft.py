#!/usr/bin/env python3
"""Land the near-lossless HUMAN axis onto the board, through the owner.

Builds one same-bake source verdict per arm carrying `rank.hfnl_cid22band` —
the CID22 top-MOS band measured by `hfnl944_band_table.py` (statistics) and
`paired_perref_boot.py` with `BAND_LO` (paired reference-clustered CIs) — and
hands each to `scripts/promote_fulleval.py --graft-into … --graft-rank …`,
which is sha-gated and asserts every other key byte-identical. Nothing is
recomputed here and no board file is written by this script.

Why a NEW axis name rather than a fill of `hf_nearlossless`: that corpus is an
ssim2 self-target (`hfnl944_reachability.py`), so a value on it is agreement,
never a win. `hfnl_cid22band` is the same zone measured against people, on the
identical 4,292-pair CID22 population every arm already shares, so it is the
axis W2's near-lossless clause can actually be decided on.

An arm is grafted only where its exam per-pair dump is provably the board
cell's own prediction vector (`--evidence`), which the caller states per arm.

    hfnl944_graft.py --panel <cid22_hfnl_band_panel.json> \
        --boot <cid22_hfnl_band_paired_boot.txt> --out-dir <scratch> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FE = Path("/mnt/v/output/zensim/reports/fulleval")

# arm -> (board cell stem, how the arm was proved to be that cell's vector)
TARGETS = {
    "ssim2": ("peer_ssim2",
              "pred vector max|delta| = 0.0 vs reports/refmetrics/cid22_ssim2.tsv, "
              "the table the peer cell was built from"),
    "B": ("b_sdr_linear_cid80_inclwinsor_dense_dial@cur372",
          "pred vector max|delta| = 0.0 vs the cell's stored per_pair.cid22.pred"),
    "ADD156": ("ADD156_safesyn_only_raw_lasso@cur372",
               "cell per-pair is stripped; pooled CID22 SROCC is BIT-equal "
               "(0.8633799667492866) to this dump's, and differs from the "
               "stored-era cell (0.8632968920382094)"),
    "W10L9P": ("W10L9P_s4005_packed", "pred vector max|delta| = 0.0"),
    "W10L9PH": ("W10L9PH_s4004_packed", "pred vector max|delta| = 0.0"),
    "Q7b": ("Q7b_pools_g0.2_a0.2_b0.97", "pred vector max|delta| = 0.0"),
}


def parse_boot(path: Path) -> dict:
    """Pull the two delta tables out of the paired-bootstrap record."""
    out: dict[str, dict] = {}
    section = None
    for line in path.read_text().splitlines():
        if line.startswith("candidate\tpooled"):
            section = "pooled"
            continue
        if line.startswith("candidate\tmean"):
            section = "within_image"
            continue
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if section and len(f) == 7 and f[0] in TARGETS:
            out.setdefault(f[0], {})[section] = {
                "arm": float(f[1]), "ssim2": float(f[2]), "delta": float(f[3]),
                "ci95_lo": float(f[4]), "ci95_hi": float(f[5]),
                "p_arm_gt_ssim2": float(f[6]),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--boot", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    panel = json.loads(a.panel.read_text())
    boot = parse_boot(a.boot)
    band = panel["band"]

    for arm, (cell, evidence) in TARGETS.items():
        board = FE / f"{cell}.fulleval.json"
        if not board.exists():
            print(f"SKIP {arm}: board cell missing {board}")
            continue
        bdoc = json.loads(board.read_text())
        blk = dict(panel["arms"][arm])
        blk["band"] = band
        blk["display"] = ("CID22 near-lossless zone (top merged-decile MOS band) "
                          "— human labels, non-circular")
        blk["vs_peer_ssim2"] = boot.get(arm, {}) if arm != "ssim2" else {
            "note": "this IS the opponent row"}
        blk["provenance"] = {
            "axis": "hfnl_cid22band",
            "why": "the exam's near-lossless corpora (hf_nearlossless, hfnlproxy) are "
                   "ssim2 SELF-TARGETS — peer_ssim2 scores 1.0 on them by construction "
                   "at any feature width — so W2's near-lossless clause cannot be "
                   "decided there. This is the same zone on human MOS.",
            "corpus": "cid22 (validation-only, 4292 pairs / 49 refs) restricted to the "
                      f"top band of scheme {band['scheme']}",
            "statistics": "panel --per-group (zenstats::per_group_srocc) — the same "
                          "quantity bake_verdict publishes as per_ref_mean/"
                          "per_ref_n/frac_negative",
            "intervals": "benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py "
                         "BAND_LO=0.8, reference-clustered paired bootstrap "
                         "B=10000 seed=20260901",
            "pairing_evidence": evidence,
            "lane": "benchmarks/ssim2_replacement_bar_2026-08-31.md APPENDIX A "
                    "(hfnl944, 2026-09-01)",
        }
        src = a.out_dir / f"graftsrc_{cell}.json"
        src.write_text(json.dumps(
            {"name": bdoc.get("name"), "bake_sha256": bdoc.get("bake_sha256"),
             "rank": {"hfnl_cid22band": blk}}, indent=2, sort_keys=True) + "\n")
        cmd = [sys.executable, str(REPO / "scripts/promote_fulleval.py"),
               "--graft-into", str(board), "--verdict", str(src),
               "--graft-rank", "hfnl_cid22band"]
        if a.dry_run:
            cmd.append("--dry-run")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[{arm} -> {cell}] rc={r.returncode} {r.stdout.strip()}{r.stderr.strip()}")
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
