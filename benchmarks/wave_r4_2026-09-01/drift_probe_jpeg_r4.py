#!/usr/bin/env python3
"""wave-r4 A6 drift probe: does a DECODED-PNG distorted side reproduce the
pixels the direct-bitstream extraction saw?

The 196k big leg is heterogeneous by construction: its 111,068 JPEG rows point
at `.jpg` bitstreams that `v2_ab_extract` reads directly through `zen_io`
(zenjpeg), while its 85,018 avif/jxl/webp rows point at PNGs this wave decoded
through `zencodec`. If those two routes disagree on pixels, the leg would mix
two feature vintages -- exactly the hazard zensim/CLAUDE.md flags ("re-decoded
pixels will NOT byte-match the canonical parquets for those codecs ... if
exactness matters, re-extract ALL corpora through one decoder").

We cannot test the avif/jxl/webp route directly (no surviving reference
extraction of those rows -- that is the whole reason this leg is being built).
What we CAN test is the ROUTE ITSELF, on JPEG, where both sides exist:

    route A (reference): v2_ab_extract reads  q<N>.jpg          via zen_io
    route B (this wave): v2_ab_extract reads  decoded q<N>.png  via zen_io,
                         where the PNG came from zencodec-decoding that same
                         q<N>.jpg through the decode owner

Route A's output is already on disk as `ext_safesyn_full.parquet` (111,068
rows, row-aligned with `safesyn_jpeg_FULL_pairs_ab.tsv` -- MEASURED
111068/111068 on ref_basename AND human_score exact). So we re-extract a sample
through route B and diff f0..f943 against the stored route-A rows.

A clean result means the decode step is pixel-transparent and the 196k leg is
ONE vintage. A drifting result must be reported loudly, not averaged away.

Usage:  drift_probe_jpeg_r4.py [N]     (default 200)
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01")
PARENT_PARQUET = ROOT / "ext_safesyn_full.parquet"
PARENT_TSV = Path("/mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv")
DECODER = Path(
    "/mnt/v/zen/cargo-targets/waver4-decode/release/examples/verify_bitstream_decode"
)
EXTRACTOR = Path("/mnt/v/zen/cargo-targets/waver4/release/examples/v2_ab_extract")
WORK = Path("/mnt/v/output/zensim/waver4-run-2026-09-01/driftprobe")
N_DEFAULT = 200
SEED = 20260901


def die(m: str) -> None:
    print(f"ABORT: {m}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    for p in (PARENT_PARQUET, PARENT_TSV, DECODER, EXTRACTOR):
        if not p.exists():
            die(f"missing {p}")
    links = WORK / "links"
    decoded = WORK / "decoded"
    for d in (links, decoded):
        d.mkdir(parents=True, exist_ok=True)

    # ---- route-A reference rows -----------------------------------------
    refs: list[str] = []
    dists: list[str] = []
    hs: list[str] = []
    with open(PARENT_TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            refs.append(row["ref_path"])
            dists.append(row["dist_path"])
            hs.append(row["human_score"])

    tbl = pq.read_table(PARENT_PARQUET)
    if tbl.num_rows != len(refs):
        die(f"parquet {tbl.num_rows} vs tsv {len(refs)}")
    rb = tbl.column("ref_basename").to_pylist()
    ok = sum(
        1
        for i in range(len(refs))
        if os.path.splitext(os.path.basename(refs[i]))[0] == rb[i]
    )
    if ok != len(refs):
        die(f"route-A row alignment gate {ok}/{len(refs)}")

    # ---- sample, stratified across the three JPEG families --------------
    by_codec: dict[str, list[int]] = {}
    for i, d in enumerate(dists):
        by_codec.setdefault(d.split("/")[-2], []).append(i)
    rng = random.Random(SEED)
    per = max(1, n // len(by_codec))
    idx: list[int] = []
    for c in sorted(by_codec):
        idx.extend(rng.sample(by_codec[c], min(per, len(by_codec[c]))))
    idx = sorted(idx)[:n]

    # ---- route B: decode those same .jpg through the decode owner -------
    listing = WORK / "decode_list.tsv"
    finals: list[Path] = []
    with open(listing, "w") as f:
        for i in idx:
            dp = dists[i]
            parts = dp.split("/")
            uniq = f"{parts[-3]}__{parts[-2]}__{os.path.splitext(parts[-1])[0]}"
            link = links / f"{uniq}.jpg"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(dp)
            f.write(f"{link}\n")
            finals.append(decoded / f"{uniq}.png")

    r = subprocess.run(
        [str(DECODER), "--decode-list", str(listing), "--out-dir", str(decoded), "--jobs", "8"],
        capture_output=True,
        text=True,
    )
    print(r.stderr.strip()[-800:])
    if r.returncode != 0:
        die(f"decode owner rc={r.returncode}")
    nmiss = sum(1 for p in finals if not p.is_file() or p.stat().st_size == 0)
    if nmiss:
        die(f"{nmiss} decoded PNGs missing/empty")

    pairs = WORK / "pairs_probe.tsv"
    with open(pairs, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for k, i in enumerate(idx):
            f.write(f"{refs[i]}\t{finals[k]}\t{hs[i]}\n")

    out_csv = WORK / "probe_routeB.csv"
    env = dict(os.environ, ZENSIM_AB_MODE="foldapp2pools")
    r = subprocess.run(
        [str(EXTRACTOR), str(pairs), str(out_csv)], capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        die(f"extractor rc={r.returncode}")

    # ---- compare f0..f943 ------------------------------------------------
    with open(out_csv, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = list(rd)
    if len(rows) != len(idx):
        die(f"route-B rows {len(rows)} != {len(idx)}")
    fcols = [c for c in hdr if c.startswith("f")]
    if len(fcols) != 944:
        die(f"route-B has {len(fcols)} feature cols, expected 944")
    fpos = {c: hdr.index(c) for c in fcols}

    ref_cols = {c: tbl.column(c).to_pylist() for c in fcols}

    max_abs = 0.0
    max_rel = 0.0
    max_abs_at = ""
    max_rel_at = ""
    n_cells = 0
    n_diff = 0
    per_feat_diff: dict[str, int] = {}
    for k, i in enumerate(idx):
        row = rows[k]
        for c in fcols:
            a = float(row[fpos[c]])
            b = ref_cols[c][i]
            n_cells += 1
            d = abs(a - b)
            if d != 0.0:
                n_diff += 1
                per_feat_diff[c] = per_feat_diff.get(c, 0) + 1
            if d > max_abs:
                max_abs, max_abs_at = d, f"{c}@row{i}(a={a!r},b={b!r})"
            scale = max(abs(a), abs(b))
            if scale > 1e-12:
                rl = d / scale
                if rl > max_rel:
                    max_rel, max_rel_at = rl, f"{c}@row{i}(a={a!r},b={b!r})"

    res = {
        "probe": "wave-r4 A6 JPEG decode-route drift",
        "question": "does zencodec-decode->PNG->zen_io match zen_io direct .jpg read?",
        "n_rows": len(idx),
        "n_feature_cols": len(fcols),
        "n_cells": n_cells,
        "n_cells_differing": n_diff,
        "max_abs": max_abs,
        "max_abs_at": max_abs_at,
        "max_rel": max_rel,
        "max_rel_at": max_rel_at,
        "n_features_with_any_diff": len(per_feat_diff),
        "top_features_by_diff_count": sorted(
            per_feat_diff.items(), key=lambda x: -x[1]
        )[:15],
        "verdict": "BYTE-IDENTICAL" if n_diff == 0 else "DRIFT",
        "route_a": "ext_safesyn_full.parquet (v2_ab_extract read q<N>.jpg via zen_io)",
        "route_b": "v2_ab_extract read decoded q<N>.png (zencodec decode of that same .jpg)",
    }
    (WORK / "drift_probe_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
