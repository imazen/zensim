#!/usr/bin/env python3
"""Score every stimulus of every arm with every candidate bake + ssim2.

Owners only: features come from `v2_ab_extract` CSVs (the canonical extractor),
bake forward passes from `predict_features_with_bake`, SSIMULACRA2 from
`zenmetrics batch --metric ssim2` (the CPU implementation the metrics crate
owns).  This file marshals bytes between them and computes nothing itself.

Regime purity: each bake is scored ONLY on the extraction whose regime it was
trained/evaluated at -- 372 bakes on `__v1`, 944 bakes on `__foldapp2`, the
W-LIN pools bake on `__foldapp2pools`.  Mixing them is the `--regime 944`
mis-scoring bug in zensim/CLAUDE.md; the map below is the guard.
"""
from __future__ import annotations
import argparse, csv, json, os, struct, subprocess, sys
from pathlib import Path

# name -> (bake path, extraction regime suffix)
BAKES = {
    "W10L9PH_s4004_packed": ("/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin", "foldapp2"),
    "W10L9P_s4005_packed": ("/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9P_s4005_packed.bin", "foldapp2"),
    "Q7b_pools_g0.2_a0.2_b0.97": ("/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin", "foldapp2pools"),
    "B": ("WEIGHTS/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin", "v1"),
    "ADD156": ("/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin", "v1"),
}


def read_features(csv_path: Path):
    with open(csv_path) as f:
        r = csv.reader(f)
        hdr = next(r)
        fi = [i for i, c in enumerate(hdr) if c.startswith("f") and c[1:].isdigit()]
        rows = [[float(row[i]) for i in fi] for row in r]
    return rows


def run_bake(bin_path: str, bake: str, rows, post: str, scratch: Path) -> list[float]:
    n_rows, n_feat = len(rows), len(rows[0])
    blob = scratch / "feat.bin"
    with open(blob, "wb") as f:
        f.write(struct.pack("<II", n_feat, n_rows))
        for row in rows:
            f.write(struct.pack(f"<{n_feat}f", *row))
    p = subprocess.run([bin_path, "--bake", bake, "--bake-post", post,
                        "--features-file", str(blob)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"predict_features_with_bake failed for {bake}:\n{p.stderr[-4000:]}")
    out = [float(x) for x in p.stdout.split()]
    assert len(out) == n_rows, f"{bake}: {len(out)} scores for {n_rows} rows"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="hfhuman output dir")
    ap.add_argument("--predict-bin", required=True)
    ap.add_argument("--zenmetrics-bin", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--arms", default="ptc_native,btc_displayed,btc_native")
    a = ap.parse_args()
    d = Path(a.dir); scratch = Path(a.scratch); scratch.mkdir(parents=True, exist_ok=True)
    prov: dict = {"bakes": {}, "arms": {}}

    for arm in a.arms.split(","):
        idx = list(csv.DictReader(open(d / f"{arm}_index.tsv"), delimiter="\t"))
        cols: dict[str, list[float]] = {}
        # ---- ssim2 (CPU, the zenmetrics owner) --------------------------
        s2out = d / f"{arm}_ssim2.tsv"
        if not s2out.exists():
            p = subprocess.run([a.zenmetrics_bin, "batch", "--metric", "ssim2",
                                "--pairs", str(d / f"{arm}_pairs.tsv"), "--output", str(s2out)],
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit(f"zenmetrics ssim2 failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
        srows = list(csv.DictReader(open(s2out), delimiter="\t"))
        s2col = next(c for c in srows[0] if "ssim" in c.lower())
        assert len(srows) == len(idx), f"{arm}: ssim2 {len(srows)} vs index {len(idx)}"
        # zenmetrics preserves the pairs order; assert the paths line up anyway
        for i, (sr, ir) in enumerate(zip(srows, idx)):
            for k in ("ref_path", "dist_path"):
                if k in sr:
                    assert os.path.abspath(sr[k]) == os.path.abspath(ir[k]), f"{arm} row {i} {k} mismatch"
        cols["peer_ssim2"] = [float(r[s2col]) for r in srows]

        # ---- bakes ------------------------------------------------------
        feats: dict[str, list] = {}
        for name, (bake, regime) in BAKES.items():
            bake = bake.replace("WEIGHTS", a.weights_dir)
            if regime not in feats:
                feats[regime] = read_features(d / "features" / f"{arm}__{regime}.csv")
                assert len(feats[regime]) == len(idx)
            for post in ("clamp", "raw"):
                key = name if post == "clamp" else f"{name}#raw"
                cols[key] = run_bake(a.predict_bin, bake, feats[regime], post, scratch)
            prov["bakes"][name] = {"bake": bake, "regime": regime,
                                   "n_features": len(feats[regime][0])}

        out = d / f"{arm}_scores.tsv"
        keys = list(cols)
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["row", "stimulus"] + keys)
            for i, ir in enumerate(idx):
                w.writerow([i, ir["stimulus"]] + [f"{cols[k][i]:.9g}" for k in keys])
        prov["arms"][arm] = {"n_rows": len(idx), "scores_tsv": str(out), "columns": keys}
        print(f"{arm}: {len(idx)} stimuli x {len(keys)} scorers -> {out}")

    (d / "score_provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
