#!/usr/bin/env python3
"""Rebuild the 372-wide G-ADDR instruments (dial grid + identity + negative-tail
probe) at a chosen extractor era, from surviving PIXELS.

WHY THIS EXISTS (measured 2026-09-05, `benchmarks/d_peaks_372_postC_2026-09-05.md`).
The shipped runtime is one extraction era ahead of every stored 372 artifact
(option C, `56bbcda2`), so a G-ADDR reading taken on the stored instruments does
not describe the dial a user gets. Rebuilding needs the PIXELS, and the three
instruments are in three different states:

* **dial grid** — the canonical grid's OWN pixels are GONE. Its build list
  `/mnt/v/output/zensim/eval_panels_2026-05-29/qsweep_372_grid.tsv` points every
  distorted cell at `/mnt/v/input/zensim/images/<ref>/<codec>/q<N>.png`, the
  decode cache DELETED 2026-06-22 (`CLAUDE.md`), and **0 of 2,560 of those paths
  exist**. The surviving `dial-grid-pixels-2026-07-27/` set is a DIFFERENT
  encode of the same (image, codec, q) lattice — `zenjpeg` where the 2026-05-29
  jpeg leg was `mozjpeg-rs-420-e4`. So a rebuild here is a NEW instrument on the
  same key set, never a re-read of the old one; it is registered separately and
  the old grid is not touched.
  **It is the better-matched instrument for the bars**: the registered
  `peer_ssim2` grid pins ARE ssim2 on the 2026-07-27 pixels — measured, in
  `dialcells_ssim2_qv2grid.tsv`, which is `dialcells_ssim2_944grid.tsv`
  restricted to these 4,424 keys (4,424 of 4,424 exactly equal) and whose
  min/max/p5/p95 reproduce the registry row to the last digit.
* **identity probe** — era-INVARIANT at w372 and measured so: a fresh HEAD
  extraction of the same 38 `ref == dist` pairs is the zero vector on all
  14,136 cells, 0 differing from the stored probe. Rebuilt anyway so the
  property stays gated rather than assumed.
* **negative-tail probe** — rebuildable. Its 2,000 rows join uniquely (2,000 of
  2,000, 0 ambiguous) to `kadis700k_canonical_gpu_2026-07-01.parquet` on
  `(feat_0, feat_1, feat_2, score_ssim2_gpu)`, which supplies `distorted_url`
  (R2-persisted PNG) and `source_filename` (local under
  `/mnt/v/datasets/kadis700k/refs/`, 2,000 of 2,000 present).

Extraction is `extract_features_372col --corpus pairs-tsv` — THE owner. Nothing
here computes a feature.

Usage:
  build_dial372_instruments.py --extractor <bin> --out-dir <dir> --tag postC \\
      [--negtail-pixels <dir>] [--what grid,identity,negtail]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

CANON_GRID = ("/mnt/v/output/zensim/eval_panels_2026-05-29/"
              "dial_grid_372col_2026-05-29_quarantined_v2.parquet")
PAIRS_TSV = "/mnt/v/output/zensim/reports/refmetrics/dialgrid_pairs.tsv"
IDENTITY_PAIRS = "/mnt/v/output/zensim/dialgate-2026-09-04/build/identity_pairs.tsv"
NEGTAIL_PROBE = ("/mnt/v/output/zensim/dialgate-2026-09-04/"
                 "negtail_probe_372_2026-09-04.parquet")
KADIS_CANON = ("/mnt/v/datasets/kadis700k/canonical/"
               "kadis700k_canonical_gpu_2026-07-01.parquet")
KADIS_REFS = "/mnt/v/datasets/kadis700k/refs"
NF = 372
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif", ".jxl")


def stem(name: str) -> str:
    low = name.lower()
    for e in EXTS:
        if low.endswith(e):
            return name[: -len(e)]
    return name


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pairs(path: str, cols: list[str], rows: list[list]) -> None:
    """Write a pairs TSV carrying a `row_id` the extractor hands back.

    THE ORDER TRAP (found the hard way 2026-09-05, see `extract`)."""
    with open(path, "w") as f:
        f.write("\t".join(cols + ["row_id"]) + "\n")
        for i, r in enumerate(rows):
            f.write("\t".join([str(x) for x in r] + [str(i)]) + "\n")


def extract(extractor: str, pairs_tsv: str, out_csv: str) -> list[list[float]]:
    """Run THE extractor over a pairs TSV and return features IN INPUT ORDER.

    **`extract_features_372col` does NOT emit rows in input order.** It ends
    with `rows.sort_by(|a, b| a.0.cmp(&b.0))` — a stable sort on
    `ref_basename` — so groups come out alphabetically by reference. Attaching
    key columns positionally after it is therefore WRONG for any input not
    already sorted that way, and it fails SILENTLY: the row count is right,
    every key is present exactly once, and only the pairing is scrambled.

    MEASURED cost of not knowing this: the first build of the 2026-09-05 dial
    grid attached (image_id, codec, q) positionally, which shuffled features
    across the 38 references. Every scorer's ladder monotonicity collapsed
    (shipped D 0.9847 -> 0.5611) and jpeg q0 read a dial of 75.6 on an image
    whose ssim2 truth is -8.03. Nothing errored.

    The fix needs no change to the extractor (whose sort other callers'
    positional comparisons against stored tables depend on): the pairs TSV
    carries a numeric `row_id`, `load_pairs_tsv` forwards every numeric extra
    column, and we sort back by it. The permutation is then CHECKED, not
    assumed."""
    r = subprocess.run([extractor, "--corpus", "pairs-tsv", "--path", pairs_tsv,
                        "--out", out_csv], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"extractor failed rc={r.returncode}")
    rows = list(csv.DictReader(open(out_csv)))
    n = sum(1 for _ in open(pairs_tsv)) - 1
    if len(rows) != n:
        raise SystemExit(f"extractor emitted {len(rows)} rows for {n} pairs")
    if "row_id" not in rows[0]:
        raise SystemExit("the extractor did not carry `row_id` back — refusing to "
                         "attach key columns positionally (see this function's docstring)")
    out: list[list[float] | None] = [None] * n
    for row in rows:
        i = int(round(float(row["row_id"])))
        if not 0 <= i < n or out[i] is not None:
            raise SystemExit(f"row_id {i} is out of range or duplicated — refusing")
        out[i] = [float(row[f"f{j}"]) for j in range(NF)]
    if any(x is None for x in out):
        raise SystemExit("some row_id never came back — refusing")
    return out  # type: ignore[return-value]


def grid_pair_order() -> tuple[list[dict], dict]:
    """Map every canonical-grid row to its 2026-07-27 pixel pair. Refuses on any
    miss — a partially-mapped grid is a different instrument, not a smaller one."""
    t = pq.read_table(CANON_GRID, columns=["image_id", "codec", "q", "codec_param",
                                           "param_kind"]).to_pydict()
    rows = list(csv.DictReader(open(PAIRS_TSV), delimiter="\t"))
    idx = {}
    for i, r in enumerate(rows):
        knob = json.loads(r["knob_tuple_json"] or "{}")
        p = float(knob["distance"]) if "distance" in knob else float(r["q"])
        k = (stem(os.path.basename(r["ref_path"])), r["codec"].replace("zen", "", 1),
             round(p, 6))
        idx.setdefault(k, i)
    out, miss = [], []
    for i in range(len(t["image_id"])):
        k = (t["image_id"][i], t["codec"][i], round(float(t["codec_param"][i]), 6))
        j = idx.get(k)
        if j is None:
            miss.append(k)
        else:
            out.append(rows[j])
    if miss:
        raise SystemExit(f"{len(miss)} canonical grid cells have no surviving pixel pair, "
                         f"first {miss[0]} — refusing")
    return out, t


def build_grid(args, work: str) -> dict:
    pairs, meta = grid_pair_order()
    tsv = os.path.join(work, "grid_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score"],
                [[r["ref_path"], r["dist_path"], f"{float(p):.6f}"]
                 for r, p in zip(pairs, meta["codec_param"])])
    feats = extract(args.extractor, tsv, os.path.join(work, "grid.csv"))
    assert len(feats) == len(pairs)
    cols = {"image_id": meta["image_id"], "codec": meta["codec"], "q": meta["q"],
            "codec_param": meta["codec_param"], "param_kind": meta["param_kind"]}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"dial_grid_372col_{args.tag}_2026-09-05.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(feats), "sha256": sha256(dst),
            "pairs_tsv": tsv, "refs": len({r["ref_path"] for r in pairs})}


def build_identity(args, work: str) -> dict:
    src = list(csv.DictReader(open(IDENTITY_PAIRS), delimiter="\t"))
    entries = [os.path.basename(r["ref_path"]) for r in src]
    tsv = os.path.join(work, "identity_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score"],
                [[r["ref_path"], r["dist_path"], r["human_score"]] for r in src])
    feats = extract(args.extractor, tsv, os.path.join(work, "identity.csv"))
    nonzero = sum(1 for row in feats for v in row if v != 0.0)
    cols = {"entry": entries}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"identity_probe_372_{args.tag}_2026-09-05.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(feats), "sha256": sha256(dst),
            "nonzero_cells": nonzero,
            "note": "the identity feature vector is the ZERO vector at w372; nonzero_cells "
                    "MUST be 0 (measured, not assumed — the 944 layout is NOT zero here)"}


def negtail_rows() -> list[dict]:
    """Join the stored probe's 2,000 rows back to KADIS pixels. Refuses on any
    miss or ambiguity — a probe rebuilt from a subset is a different probe."""
    import numpy as np  # noqa: F401  (kept local; only this path needs it)
    kc = pq.read_table(KADIS_CANON, columns=["feat_0", "feat_1", "feat_2",
                                             "score_ssim2_gpu", "distorted_url",
                                             "source_filename", "source_id"]).to_pydict()
    key: dict = {}
    for i in range(len(kc["feat_0"])):
        key.setdefault((kc["feat_0"][i], kc["feat_1"][i], kc["feat_2"][i],
                        kc["score_ssim2_gpu"][i]), []).append(i)
    pr = pq.read_table(NEGTAIL_PROBE, columns=["f0", "f1", "f2", "ssim2_gpu",
                                               "human_score", "entry"]).to_pydict()
    out, amb, miss = [], 0, 0
    for k in range(len(pr["f0"])):
        v = key.get((pr["f0"][k], pr["f1"][k], pr["f2"][k], pr["ssim2_gpu"][k]))
        if v is None:
            miss += 1
            continue
        if len(v) > 1:
            amb += 1
        i = v[0]
        out.append({"entry": pr["entry"][k], "ssim2_gpu": pr["ssim2_gpu"][k],
                    "human_score": pr["human_score"][k],
                    "url": kc["distorted_url"][i],
                    "ref": os.path.join(KADIS_REFS, kc["source_filename"][i]),
                    "source_id": kc["source_id"][i]})
    if miss or amb:
        raise SystemExit(f"negtail join: {miss} unmatched, {amb} ambiguous — refusing")
    return out


def build_negtail(args, work: str) -> dict:
    rows = negtail_rows()
    px = args.negtail_pixels
    missing = [r for r in rows if not os.path.exists(os.path.join(px, os.path.basename(r["url"])))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} negtail PNGs absent under {px}. "
                         "Fetch them first:\n"
                         "  set -a; . ~/.config/cloudflare/r2-credentials; set +a\n"
                         "  s5cmd --endpoint-url https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com "
                         "run <cp commands built from distorted_url>")
    tsv = os.path.join(work, "negtail_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score", "ssim2_gpu"],
                [[r["ref"], os.path.join(px, os.path.basename(r["url"])),
                  repr(r["human_score"]), repr(r["ssim2_gpu"])] for r in rows])
    feats = extract(args.extractor, tsv, os.path.join(work, "negtail.csv"))
    cols = {"entry": [r["entry"] for r in rows],
            "ssim2_gpu": [r["ssim2_gpu"] for r in rows],
            "human_score": [r["human_score"] for r in rows]}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"negtail_probe_372_{args.tag}_2026-09-05.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(feats), "sha256": sha256(dst),
            "sources": len({r["source_id"] for r in rows}),
            "pixels_dir": px}


# ── the LADDER instrument (2026-09-05) ────────────────────────────────────────
# Built from a `build_ladder_grid.sh` run rather than from the canonical grid's
# key set. Two things make it a DIFFERENT instrument, both deliberate:
#
#  1. **The floor is the codec's own lowest DISTINCT settings.** MEASURED: zenjpeg
#     emits ONE bitstream for every q in 0..10, so the previous grid's bottom three
#     steps (q 0/5/10) were a single setting sampled three times and the mentor's
#     jpeg floor bar was a vacuous 0.0000. Every codec has its own plateau
#     (avif/svt-rs ties pairwise because quality 0..100 maps onto QP 0..63), so the
#     grid is dense at the floor and duplicates are removed by ENCODE HASH, never
#     by a guessed per-codec step table.
#  2. **The two AVIF backends are two ladders**, `avif-svt` and `avif-rav1e`, because
#     `FloorRepresentabilityRule` groups by `(image_id, codec)` and they are
#     different encoders with different quantizer mappings.
#
# TWO tables come out, and the difference matters:
#   * `<tag>_full.parquet`  — EVERY encoded step, with `saturated` / `encode_sha` /
#     every metric variant. The archive; nothing is thrown away.
#   * `dial_grid_372col_<tag>.parquet` — DISTINCT settings only, in the canonical
#     grid's schema. THE instrument. Emitting only distinct rows here is what lets
#     `dial_addressability.rs` stay unchanged: its "bottom K steps" are then the
#     bottom K *configurable* settings by construction.
# The canonical grid's own (distance -> q) curve. JXL is parameterised by
# butteraugli DISTANCE, but `q` must stay quality-ASCENDING on every codec so the
# floor rule needs no per-codec direction switch (gate doc 16.2, pinned by a test).
# The canonical grid already fixed that convention — q=0 <-> distance 25.0, up to
# q=99.8 <-> distance 0.05 — so this instrument REUSES it rather than inventing a
# second one, and interpolates linearly in q for the two distances this ladder adds
# (24.0 and 22.0, which land at q 4.0 and 12.0 between their canonical neighbours).
CANON_JXL_QD = [(0.0, 25.0), (8.0, 23.0), (16.0, 21.0), (24.0, 19.0), (32.0, 17.0),
                (40.0, 15.0), (48.0, 13.0), (60.0, 10.0), (68.0, 8.0), (74.0, 6.5),
                (80.0, 5.0), (84.0, 4.0)]


def _jxl_q_for_distance(dist: float) -> float:
    """Quality-ascending q for a butteraugli distance, on the canonical curve."""
    t = pq.read_table(CANON_GRID, columns=["codec", "q", "codec_param"]).to_pydict()
    pts = sorted({(float(q), float(cp)) for c, q, cp in
                  zip(t["codec"], t["q"], t["codec_param"]) if c == "jxl"},
                 key=lambda x: -x[1])            # distance DESCENDING == q ascending
    if not pts:
        raise SystemExit("canonical grid carries no jxl rows — cannot reuse its q curve")
    for (q, d) in pts:
        if abs(d - dist) < 1e-9:
            return q
    lo = [p for p in pts if p[1] > dist]
    hi = [p for p in pts if p[1] < dist]
    if not lo:                                    # below the curve's smallest distance
        return pts[-1][0]
    if not hi:                                    # above its largest (== deeper floor)
        return pts[0][0]
    (q0, d0), (q1, d1) = lo[-1], hi[0]
    return q0 + (q1 - q0) * (d0 - dist) / (d0 - d1)


LADDER_LEGS = {
    "jpeg":       ("jpeg",       "q"),
    "webp":       ("webp",       "q"),
    "avif_svt":   ("avif-svt",   "q"),
    "avif_rav1e": ("avif-rav1e", "q"),
    "jxl":        ("jxl",        "distance"),
}


def _legs(args) -> list[str] | None:
    """The explicit leg subset for this run, or None = every leg."""
    v = getattr(args, "legs", "") or ""
    return [x for x in v.split(",") if x] or None


_JXLQ: dict[float, float] = {}


def _ladder_rows(ladder_dir: str, dropped: list | None = None,
                 legs: list[str] | None = None) -> list[dict]:
    """Join each leg's metric TSV to its pairs TSV and hash the persisted bitstream.

    `dropped` collects cells that cannot enter the instrument (decode failures)."""
    out = []
    dropped = dropped if dropped is not None else []
    for leg, (codec_name, param_kind) in LADDER_LEGS.items():
        # An EXPLICIT leg list, never silent tolerance: a run that quietly skipped a
        # missing leg would emit a smaller instrument that looks like a complete one.
        if legs is not None and leg not in legs:
            continue
        tsv = os.path.join(ladder_dir, "tsv", f"{leg}.tsv")
        prs = os.path.join(ladder_dir, "pairs", f"{leg}.tsv")
        if not (os.path.exists(tsv) and os.path.exists(prs)):
            raise SystemExit(f"ladder leg {leg} is missing {tsv} or {prs} — refusing "
                             f"a partially-built instrument")
        key = lambda r: (r["image_path"], r["q"], r["knob_tuple_json"])
        pair = {key(r): r for r in csv.DictReader(open(prs), delimiter="\t")}
        enc_dir = os.path.join(ladder_dir, "encoded", leg)
        for r in csv.DictReader(open(tsv), delimiter="\t"):
            sc = r.get("score_ssim2", "")
            scored = sc not in ("", None) and sc.lower() not in ("nan",)
            pr = pair.get(key(r))
            if pr is None:
                # A cell whose DECODE failed writes no distorted PNG and therefore no
                # pairs row. That is a legitimate absence and is COUNTED. A missing
                # pairs row for a cell that DID score is an inconsistency and still
                # refuses — the two must not be conflated.
                if scored:
                    raise SystemExit(f"{leg}: metric row {key(r)} scored {sc} but has no "
                                     f"pairs row — refusing")
                dropped.append({"leg": leg, "image": os.path.basename(r["image_path"]),
                                "param": (json.loads(r["knob_tuple_json"] or "{}")
                                          .get("distance", r["q"])),
                                "why": "decode-fail (no distorted pixels, no pairs row)"})
                continue
            knob = json.loads(r["knob_tuple_json"] or "{}")
            param = float(knob["distance"]) if param_kind == "distance" else float(r["q"])
            enc = os.path.join(enc_dir, r["encoded_filename"]) if r.get("encoded_filename") else ""
            if not enc or not os.path.exists(enc):
                raise SystemExit(f"{leg}: encoded bitstream missing for {key(r)} — the "
                                 f"whole point of this rebuild is that bytes are kept")
            # A cell whose DECODE failed has no distorted pixels and no score, so it
            # cannot carry features and must not enter the instrument. It is COUNTED
            # and reported, never dropped silently.
            #
            # MEASURED 2026-09-05, and PRE-EXISTING rather than introduced here: the
            # `zenjxl` leg loses exactly 13 images x the 10 largest distances = 130
            # cells, and all 13 are ODD-dimensioned (513x769 / 769x513). The
            # 2026-07-27 run of the same sources shows the identical signature (the
            # same 13 images at distances 10..25). Because those are the LARGEST
            # distances, the loss is precisely each affected ladder's FLOOR, so those
            # jxl ladders cannot be floor-graded at all — which is why this is
            # surfaced rather than absorbed.
            if not scored or not os.path.exists(pr["dist_path"]):
                dropped.append({"leg": leg, "image": os.path.basename(pr["ref_path"]),
                                "param": param,
                                "why": "decode-fail (no distorted pixels)"})
                continue
            qcol = (_JXLQ.setdefault(param, _jxl_q_for_distance(param))
                    if param_kind == "distance" else float(r["q"]))
            row = {"image_id": stem(os.path.basename(pr["ref_path"])),
                   "codec": codec_name, "param_kind": param_kind,
                   "codec_param": param, "q": qcol,
                   "ref_path": pr["ref_path"], "dist_path": pr["dist_path"],
                   "encode_sha": sha256(enc), "encoded_bytes": int(r["encoded_bytes"]),
                   "encode_ms": float(r["encode_ms"]), "decode_ms": float(r["decode_ms"])}
            for k, v in r.items():
                if k.startswith("score_"):
                    row[k] = float(v) if v not in ("", None) else float("nan")
            out.append(row)
    return out


def _mark_saturated(rows: list[dict]) -> list[dict]:
    """Flag every step whose encoded bytes repeat the previous DISTINCT step's.

    Ordered from each codec's FLOOR upward: ascending `q`, but DESCENDING
    `distance` (JXL's floor is its largest distance — 25.0, beyond which the
    encoder saturates: 26/30/40/50 are byte-identical to it, measured)."""
    by = {}
    for r in rows:
        by.setdefault((r["image_id"], r["codec"]), []).append(r)
    for (_img, _c), lad in by.items():
        desc = lad[0]["param_kind"] == "distance"
        lad.sort(key=lambda r: -r["codec_param"] if desc else r["codec_param"])
        last = None
        for r in lad:
            r["saturated"] = (last is not None and r["encode_sha"] == last)
            if not r["saturated"]:
                last = r["encode_sha"]
    return [r for lad in by.values() for r in lad]


def _drop_truncated_floor_ladders(rows: list[dict], dropped: list) -> tuple[list[dict], list]:
    """Remove any ladder whose OWN FLOOR cells are missing.

    A ladder that lost its most-extreme settings cannot answer the question A7r
    asks. Its surviving bottom three are simply the lowest settings that happened
    to decode, several steps up the curve — grading them would flatter the bar
    while looking exactly like a normal measurement.

    CONCRETE: `zenjxl` at distance >= 10 on ODD-dimensioned sources (513x769 /
    769x513) emits a bitstream that fails to decode — 13 images x the 10 largest
    distances, and the 2026-07-27 run of the same sources shows the identical
    signature, so it is pre-existing rather than introduced by this instrument.
    Those are exactly the FLOOR cells, so all 13 ladders are excluded here and
    reported as NOT MEASURABLE rather than silently graded on distance 8."""
    lost: dict = {}
    for d in dropped:
        lost.setdefault((stem(d["image"]), d["leg"]), []).append(d["param"])
    excluded, keep = [], []
    for r in rows:
        # `leg` and `codec` differ (avif_svt vs avif-svt); match on both spellings.
        hit = lost.get((r["image_id"], r["codec"])) or               lost.get((r["image_id"], r["codec"].replace("-", "_")))
        if hit is None:
            keep.append(r)
            continue
        floorward = max(hit) if r["param_kind"] == "distance" else min(hit)
        surviving = r["codec_param"]
        more_extreme = (floorward > surviving) if r["param_kind"] == "distance"             else (floorward < surviving)
        if more_extreme:
            excluded.append((r["image_id"], r["codec"]))
        else:
            keep.append(r)
    return keep, sorted(set(excluded))


def build_ladder(args, work: str) -> dict:
    dropped: list = []
    rows = _mark_saturated(_ladder_rows(args.ladder_dir, dropped, _legs(args)))
    rows, excluded = _drop_truncated_floor_ladders(rows, dropped)
    distinct = [r for r in rows if not r["saturated"]]

    # FULL archive first — it must survive even if extraction later fails.
    full_cols = {k: [r[k] for r in rows] for k in rows[0] if k not in ("ref_path", "dist_path")}
    full = os.path.join(args.out_dir, f"ladder_grid_{args.tag}_full.parquet")
    pq.write_table(pa.table(full_cols), full, compression="zstd")

    tsv = os.path.join(work, "ladder_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score"],
                [[r["ref_path"], r["dist_path"], f"{r['codec_param']:.6f}"] for r in distinct])
    feats = extract(args.extractor, tsv, os.path.join(work, "ladder.csv"))
    cols = {"image_id": [r["image_id"] for r in distinct],
            "codec": [r["codec"] for r in distinct],
            "q": [r["q"] for r in distinct],
            "codec_param": [r["codec_param"] for r in distinct],
            "param_kind": [r["param_kind"] for r in distinct]}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"dial_grid_372col_{args.tag}.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")

    # The MENTOR's own per-cell scores on exactly these cells, in the format
    # `--dial-peer-scores` / `--gaddr-grid-truth` read (image_id, codec, q, pred).
    # Written for the DISTINCT rows only, because those are the instrument.
    truth = os.path.join(args.out_dir, f"dialcells_ssim2_{args.tag}.tsv")
    with open(truth, "w") as f:
        f.write("image_id\tcodec\tq\tpred\n")
        for r in distinct:
            f.write(f"{r['image_id']}\t{r['codec']}\t{r['q']}\t{r['score_ssim2']:.6f}\n")

    per = {}
    for r in rows:
        d = per.setdefault(r["codec"], {"cells": 0, "saturated": 0, "ladders": set()})
        d["cells"] += 1
        d["saturated"] += int(r["saturated"])
        d["ladders"].add(r["image_id"])
    drop_by_leg: dict = {}
    for d in dropped:
        drop_by_leg.setdefault(d["leg"], {"n": 0, "images": set(), "params": set()})
        drop_by_leg[d["leg"]]["n"] += 1
        drop_by_leg[d["leg"]]["images"].add(d["image"])
        drop_by_leg[d["leg"]]["params"].add(d["param"])
    return {"file": dst, "rows": len(distinct), "sha256": sha256(dst),
            "full_table": full, "full_rows": len(rows), "full_sha256": sha256(full),
            "grid_truth_tsv": truth, "grid_truth_sha256": sha256(truth),
            "dropped_cells": len(dropped),
            "excluded_truncated_floor_ladders": [list(e) for e in excluded],
            "n_excluded_ladders": len(excluded),
            "dropped_by_leg": {k: {"n": v["n"], "n_images": len(v["images"]),
                                   "params": sorted(v["params"])}
                               for k, v in sorted(drop_by_leg.items())},
            "pairs_tsv": tsv,
            "per_codec": {c: {"cells": d["cells"], "saturated": d["saturated"],
                              "distinct": d["cells"] - d["saturated"],
                              "ladders": len(d["ladders"])} for c, d in sorted(per.items())}}


def build_anchor(args, work: str) -> dict:
    """The dial-calibration ANCHOR from a ladder run: features + an UNCLAMPED target.

    Two deliberate differences from the shipped anchor
    (`canonical-2026-05-21/train/multiband_anchor_dial100.parquet`) and from the
    2026-09-04 imazen-26 proposal, both measured problems rather than preferences:

    1. **The target is NOT clamped.** Both prior anchors store
       `target_score = max(ssim2, 0)`. On the shipped one that stores 147 of 2,000
       genuinely-negative rows (down to ssim2 -64.16) as a single 0, and
       `fit_spline_knots` then collapses that whole run onto ONE bottom knot —
       which is why shipped Profile B emits `frac_below_zero = 0.0000` on a probe
       whose every row's ssim2 truth is negative. The lever for the floor is the
       CLAMP, so this anchor keeps the sign.
    2. **Target and features are ONE era.** The 2026-09-04 anchor's targets are
       bigcodec's stored ssim2 from its own decode era while its features are
       decoded today; the measured cost of that kind of skew is -3.658 dial points
       on shipped B (DATASET_HISTORY 3.34), and its own 2x2 put anchor ERA at
       +3.9/+4.8/+3.9 against anchor CONTENT at -0.4/-1.0/-0.2. Here both sides
       come from the same fresh sweep.

    Identity rows (`ref == dist`, target 100) are appended so the spline is pinned
    at the top by construction rather than by extrapolation."""
    dropped: list = []
    rows = _mark_saturated(_ladder_rows(args.ladder_dir, dropped, _legs(args)))
    distinct = [r for r in rows if not r["saturated"]]
    ident = sorted({r["ref_path"] for r in rows})

    pair_rows = [[r["ref_path"], r["dist_path"], f"{r['score_ssim2']:.6f}"] for r in distinct]
    pair_rows += [[p, p, "100.000000"] for p in ident]
    tsv = os.path.join(work, "anchor_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score"], pair_rows)
    feats = extract(args.extractor, tsv, os.path.join(work, "anchor.csv"))

    target = [r["score_ssim2"] for r in distinct] + [100.0] * len(ident)
    cols = {
        "image_id": [r["image_id"] for r in distinct] + [stem(os.path.basename(p)) for p in ident],
        "codec": [r["codec"] for r in distinct] + ["identity"] * len(ident),
        "codec_param": [r["codec_param"] for r in distinct] + [0.0] * len(ident),
        # THREE names for one column, because the consumers disagree:
        # `extend-top --target-col target_score`, `fit-lasso --anchor-target ssim2_gpu`,
        # and the parquet loader's `human_score`. Same values; no consumer silently
        # picks up a different quantity.
        "target_score": list(target), "ssim2_gpu": list(target), "human_score": list(target),
    }
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"ladder_anchor_372col_{args.tag}.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    neg = sum(1 for t in target if t < 0.0)
    return {"file": dst, "rows": len(target), "sha256": sha256(dst),
            "identity_rows": len(ident), "distinct_cells": len(distinct),
            "dropped_cells": len(dropped),
            "negative_target_rows": neg,
            "min_target": min(target), "max_target": max(target),
            "note": "target is UNCLAMPED ssim2 (negative rows PRESERVED — the shipped "
                    "anchor's max(ssim2,0) is what collapses the negative tail); identity "
                    "rows carry 100.0"}


def build_ladder944(args, work: str) -> dict:
    """Attach a 944-wide extraction to the ladder instrument's OWN key columns.

    ORDER CONTRACT, verified rather than assumed. `extract_features_372col` sorts
    its output by `ref_basename` (:216) and so scrambles positional attachment;
    **`v2_ab_extract` does NOT** — tested on a deliberately non-alphabetical 6-row
    input, its output order matched its input exactly. This function therefore
    joins POSITIONALLY, and GATES that decision: the 944 CSV's `ref_basename` must
    equal the pairs TSV's ref basename on every row, and the pairs TSV's `row_id`
    must be the identity permutation. Either mismatch refuses.

    Regime: `ZENSIM_AB_MODE=foldapp2pools` — the runtime regime of
    `dial_grid_944col_POOLS_2026-08-30`. Never column-mix regimes.
    """
    pairs = list(csv.DictReader(open(args.ladder944_pairs), delimiter="\t"))
    rows = list(csv.DictReader(open(args.ladder944_csv)))
    if len(rows) != len(pairs):
        raise SystemExit(f"944 csv has {len(rows)} rows for {len(pairs)} pairs — refusing")
    for i, (pr, r) in enumerate(zip(pairs, rows)):
        if int(round(float(pr["row_id"]))) != i:
            raise SystemExit(f"pairs TSV row {i} carries row_id {pr['row_id']} — the 944 "
                             f"join is POSITIONAL and needs the identity permutation")
        if stem(os.path.basename(pr["ref_path"])) != stem(r["ref_basename"]):
            raise SystemExit(f"944 row {i} is {r['ref_basename']!r} but the pairs TSV says "
                             f"{os.path.basename(pr['ref_path'])!r} — order is NOT preserved, "
                             f"refusing to attach keys positionally")
    keys = pq.read_table(args.ladder944_keys,
                         columns=["image_id", "codec", "q", "codec_param",
                                  "param_kind"]).to_pydict()
    if len(keys["image_id"]) != len(rows):
        raise SystemExit("key parquet and 944 csv disagree on row count — refusing")
    for i in range(len(rows)):
        if keys["image_id"][i] != stem(os.path.basename(pairs[i]["ref_path"])):
            raise SystemExit(f"key parquet row {i} disagrees with the pairs TSV — refusing")
    nf = sum(1 for c in rows[0] if c.startswith("f") and c[1:].isdigit())
    cols = dict(keys)
    for j in range(nf):
        cols[f"f{j}"] = [float(r[f"f{j}"]) for r in rows]
    dst = os.path.join(args.out_dir, f"dial_grid_944col_{args.tag}.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(rows), "n_features": nf, "sha256": sha256(dst),
            "regime": "folded720append2pools (ZENSIM_AB_MODE=foldapp2pools)",
            "join": "positional, GATED on ref_basename equality and an identity row_id"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True,
                    help="path to extract_features_372col built at the target era")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="era tag, e.g. postC / preC")
    ap.add_argument("--build-commit", default="", help="the extractor's tree commit")
    ap.add_argument("--negtail-pixels",
                    default="/mnt/v/output/zensim/dpeaks372-2026-09-05/negtail_pixels")
    ap.add_argument("--ladder944-csv", default="", help="v2_ab_extract output")
    ap.add_argument("--ladder944-pairs", default="", help="the pairs TSV it was run on")
    ap.add_argument("--ladder944-keys", default="", help="the 372 ladder instrument (keys)")
    ap.add_argument("--legs", default="",
                    help="comma-separated leg subset (jpeg,webp,avif_svt,avif_rav1e,jxl). "
                         "Empty = all. EXPLICIT so a missing leg cannot silently shrink "
                         "the instrument.")
    ap.add_argument("--ladder-dir", default="",
                    help="a build_ladder_grid.sh output dir; required for --what ladder")
    ap.add_argument("--what", default="grid,identity,negtail",
                    help="grid,identity,negtail (canonical key set) and/or `ladder` "
                         "(a floor-dense per-codec ladder run — see build_ladder)")
    ap.add_argument("--work", default=os.path.expanduser("~/tmp/dial372_instruments"))
    args = ap.parse_args()
    work = os.path.join(args.work, args.tag)
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    want = set(args.what.split(","))
    man = {"built_utc": datetime.now(timezone.utc).isoformat(),
           "extractor": args.extractor, "build_commit": args.build_commit,
           "tag": args.tag, "instruments": {}}
    if "grid" in want:
        man["instruments"]["dial_grid"] = build_grid(args, work)
    if "identity" in want:
        man["instruments"]["identity_probe"] = build_identity(args, work)
    if "negtail" in want:
        man["instruments"]["negtail_probe"] = build_negtail(args, work)
    if "ladder" in want:
        if not args.ladder_dir:
            raise SystemExit("--what ladder needs --ladder-dir")
        man["ladder_dir"] = args.ladder_dir
        man["instruments"]["ladder_grid"] = build_ladder(args, work)
    if "ladder944" in want:
        for f in ("ladder944_csv", "ladder944_pairs", "ladder944_keys"):
            if not getattr(args, f):
                raise SystemExit(f"--what ladder944 needs --{f.replace('_','-')}")
        man["instruments"]["ladder_grid_944"] = build_ladder944(args, work)
    if "anchor" in want:
        if not args.ladder_dir:
            raise SystemExit("--what anchor needs --ladder-dir")
        man["ladder_dir"] = args.ladder_dir
        man["instruments"]["ladder_anchor"] = build_anchor(args, work)
    mp = os.path.join(args.out_dir, f"_MANIFEST_{args.tag}.json")
    with open(mp, "w") as f:
        json.dump(man, f, indent=1)
    for k, v in man["instruments"].items():
        print(f"{k:16s} rows={v['rows']:6d} sha={v['sha256'][:16]} {v['file']}")
    print("wrote", mp)


if __name__ == "__main__":
    main()
