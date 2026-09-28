#!/usr/bin/env python3
"""AIC-4 frozen-model read on the PTC crops AND the full-resolution encodes.

T0 eval-only (DATA_SPLITS "September 14 clarification"): frozen models are
SCORED here, nothing is fitted, calibrated or selected on the result, and the
read is recorded in DATA_SPLITS.md's exposure ledger. AIC-4 has FIVE sources,
so the per-source columns are a sanity check (is any one ladder ranked
backwards?), never a selection axis.

Inputs are the outputs of the existing owners, never re-derived here:
  * labels  — `site/data/parquet/aic4_sample.parquet` (committed, 709f4597):
              `human_jnd` = reconstructed JND distance, DISTORTION-oriented
              (declared in check_target_orientation.py EXPECTED_ORIENTATION).
  * scores  — `score_pairs_tuner` parquets (named profiles + BakeScorer
              ensembles) and `peer_metric_pairs` TSVs, one per image leg.
  * the organisers' own metric columns from the same committed parquet
    (crop leg only — they scored the crops).
Every statistic is the canonical Rust `panel` via `scripts/lib/zen_stats`:
|SROCC|/|KROCC| from `panel(..., band=source)`, the sign from
`panel_batch(stats="srocc")`. The ALIGNED sign multiplies by the declared
orientation (-1 for aic4), so +1 means "ranks like the humans".

Usage:
  aic4_refresh_read.py --pairs-crop P.tsv --pairs-full Q.tsv \
      --scores-crop A.parquet --scores-full B.parquet \
      --peer-crop A.tsv --peer-full B.tsv --out-json X.json --out-md X.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.csv as pacsv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.lib.zen_stats import panel, panel_batch  # noqa: E402
from scripts.canonical_corpus.check_target_orientation import (  # noqa: E402
    DISTORTION, EXPECTED_ORIENTATION)

LABELS = REPO / "site" / "data" / "parquet" / "aic4_sample.parquet"
# Organiser-published columns carried in the committed labels parquet.
PUBLISHED = ["score_psnr_y", "score_ssim", "score_ms_ssim", "score_iw_ssim",
             "score_vmaf_neg", "score_ssim2_paper", "score_hdr_vdp_2",
             "score_hdr_vdp_3", "score_cvvdp"]
PEER_COLS = ["ssim2", "butter_max", "butter_p1", "butter_p2", "butter_p3"]
# DECLARED metric polarity (never inferred from the data being read): butteraugli
# is a DISTANCE — it rises with distortion, exactly like the q_jnd target — so its
# aligned sign is flipped once more. Every other column here is quality-shaped.
DISTANCE_METRICS = {"butter_max", "butter_p1", "butter_p2", "butter_p3"}


def stim_key(dist_path: str) -> str:
    """`.../PTC_00002_AVIF_01.png` and `.../00002_AVIF_01.png` -> `00002_AVIF_01`."""
    return Path(dist_path).stem.removeprefix("PTC_")


def read_scores(path: Path, pairs: Path) -> dict[str, dict[str, float]]:
    """{column: {stim_key: value}} for every numeric score column.

    Neither scorer carries the path columns through, but both keep input row
    order and pass `human_score` through, so rows are keyed by the pairs TSV
    they were run on — and that passthrough label must match row for row."""
    if path.suffix == ".parquet":
        t = pq.read_table(path)
    else:
        t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(
            delimiter="\t"), convert_options=pacsv.ConvertOptions(
            column_types={"human_score": "string"}))
    pt = pacsv.read_csv(pairs, parse_options=pacsv.ParseOptions(delimiter="\t"),
                        convert_options=pacsv.ConvertOptions(
                            column_types={"human_score": "string"}))
    want = pt.column("human_score").to_pylist()
    got = t.column("human_score").to_pylist()
    by_label = dict(zip(want, (stim_key(p) for p in pt.column("dist_path").to_pylist())))
    if pt.num_rows != t.num_rows:
        raise SystemExit(f"{path}: {t.num_rows} rows vs {pt.num_rows} in {pairs}")
    if got != want:
        # `peer_metric_pairs` writes in completion order (rayon for_each). The
        # AIC-4 label text is unique per stimulus, so it keys a row exactly —
        # but only when that holds; anything else is refused, never guessed.
        if len(by_label) != len(want) or sorted(got) != sorted(want):
            raise SystemExit(f"{path}: rows cannot be keyed to {pairs}")
    keys = [by_label[g] for g in got]
    out = {}
    for name in t.column_names:
        if name.startswith("score_") or name in PEER_COLS:
            vals = t.column(name).to_pylist()
            out[name] = dict(zip(keys, vals))
    return out


def read_one(label: str, pred_by_key: dict, keys, human, source) -> dict:
    x = [float(pred_by_key[k]) for k in keys]
    agg = panel(x, human, band=source)
    srcs = sorted(set(source))
    jobs = [(label + "|ALL", x, human)]
    for s in srcs:
        idx = [i for i, v in enumerate(source) if v == s]
        jobs.append((f"{label}|{s}", [x[i] for i in idx], [human[i] for i in idx]))
    signed = {r["label"].split("|")[1]: r["srocc_signed"]
              for r in panel_batch(jobs, stats="srocc")}
    sgn = -1.0 if EXPECTED_ORIENTATION["aic4"] == DISTORTION else 1.0
    if label.split(".", 1)[1] in DISTANCE_METRICS:
        sgn = -sgn
    return {
        "n": agg["n"], "srocc": agg["srocc"], "krocc": agg["krocc"],
        "srocc_aligned": sgn * signed["ALL"],
        "per_source": {s: {"srocc": agg["bands"][f"band={s}"]["srocc"],
                           "krocc": agg["bands"][f"band={s}"]["krocc"],
                           "srocc_aligned": sgn * signed[s]} for s in srcs},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-crop", type=Path, required=True)
    ap.add_argument("--scores-full", type=Path, required=True)
    ap.add_argument("--peer-crop", type=Path, required=True)
    ap.add_argument("--peer-full", type=Path, required=True)
    ap.add_argument("--pairs-crop", type=Path, required=True,
                    help="the pairs TSV --scores-crop/--peer-crop were run on")
    ap.add_argument("--pairs-full", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    a = ap.parse_args()

    lab = pq.read_table(LABELS)
    keys = [stim_key(p) for p in lab.column("dist_path").to_pylist()]
    human = [float(v) for v in lab.column("human_jnd").to_pylist()]
    source = lab.column("image_name").to_pylist()
    published = {c: dict(zip(keys, lab.column(c).to_pylist())) for c in PUBLISHED}

    legs = {"crop": {**read_scores(a.scores_crop, a.pairs_crop),
                     **read_scores(a.peer_crop, a.pairs_crop)},
            "full": {**read_scores(a.scores_full, a.pairs_full),
                     **read_scores(a.peer_full, a.pairs_full)}}
    res: dict = {"labels": str(LABELS.relative_to(REPO)), "n_sources": len(set(source)),
                 "orientation": EXPECTED_ORIENTATION["aic4"], "legs": {}}
    for leg, cols in legs.items():
        res["legs"][leg] = {}
        for col, byk in sorted(cols.items()):
            missing = [k for k in keys if k not in byk or byk[k] is None]
            if missing:
                raise SystemExit(f"{leg}/{col}: {len(missing)} stimuli unscored, e.g. {missing[:3]}")
            res["legs"][leg][col] = read_one(f"{leg}.{col}", byk, keys, human, source)
    res["legs"]["crop_published"] = {
        c: read_one(f"pub.{c}", published[c], keys, human, source) for c in PUBLISHED}

    a.out_json.write_text(json.dumps(res, indent=1, sort_keys=True))
    srcs = sorted(set(source))
    lines = ["| model | leg | n | SROCC | KROCC | aligned sign | "
             + " | ".join(f"src {s} SROCC" for s in srcs) + " |",
             "|---|---|--:|--:|--:|--:|" + "--:|" * len(srcs)]
    for leg in ("crop", "full", "crop_published"):
        for col, r in res["legs"][leg].items():
            ps = " | ".join(
                f"{r['per_source'][s]['srocc']:.4f}"
                + ("" if r["per_source"][s]["srocc_aligned"] > 0 else " ⛔")
                for s in srcs)
            lines.append(f"| {col} | {leg} | {r['n']} | {r['srocc']:.4f} | {r['krocc']:.4f} | "
                         f"{'+' if r['srocc_aligned'] > 0 else '−'} | {ps} |")
    a.out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
