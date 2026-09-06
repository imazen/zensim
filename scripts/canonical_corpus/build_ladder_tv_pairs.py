#!/usr/bin/env python3
"""Build a TV pairs file for the WITHIN-LADDER ordering hinge.

THE LOSS ALREADY HAS AN OWNER. `zensim_validate::mlp_train::TvRegularizer`
(`--tv-pairs-file` / `--tv-weight` / `--tv-margin` / `--tv-apply-every` /
`--tv-batch` / `--tv-band-weights`) is a within-ladder pairwise hinge over
adjacent severity levels with an anti-collapse margin, wired on the plain path.
This script does NOT re-implement it — it only decides WHICH pairs the owner
should see. Adding a second `--ladder-hinge` flag would be a duplicate
implementation, which this repo bans.

WHAT IT EMITS. `lo_idx <TAB> hi_idx <TAB> band_id`, header `lo_idx...`, where the
indices are into the **concatenated group feature rows in `--group` order**. The
trainer's loader reads exactly that; as of 2026-09-06 it counts and reports
out-of-range pairs and refuses when every pair is out of range, so a mismatch
here fails loudly instead of silently emptying the ladder term.

`lo` is the LOWER-quality member and `hi` the HIGHER-quality one, matching the
owner's `lo_q` / `hi_q` naming.

THE MATERIALITY RULE. A pair is kept only when the reference metric orders its
two members by **>= 0.5 ssim2 points** — the same `ENCODER_SSIM2_MARGIN_PT` /
`MATERIAL_INV_PT` constant `bake_verdict` grades the dial's monotonicity with.
Below it the two settings are not distinguishable in the units the exam scores,
and supervising them teaches noise.

SATURATION DEDUP. The safesyn q grid starts at 5 and steps by 5, so a codec's
low-q plateau is only partly present and *nothing has verified* that q5 and q10
are distinct settings on every codec. Consecutive steps with identical
`size_bytes` are collapsed before pairs are formed. (The ladder instrument keys
on `encode_sha`; that column does not exist here, so `size_bytes` is the
available proxy and is declared as such.)

THE JOIN IS POSITIONAL, AND IT IS VERIFIED HERE, NOT ASSUMED. The sidecar CSV
and the training parquet are row-aligned; the script checks `ref_basename` on
every row and aborts on the first mismatch.

BUTTERAUGLI AGREEMENT IS **NOT MEASURED** AND IS NOT SILENTLY SKIPPED. The gate's
two-reference rule needs a *named* butteraugli variant (pnorm3 at margin 0.05 is
primary; max at 0.25 is report-only), and this CSV's `gpu_butteraugli` /
`cpu_butteraugli` columns do not say which they are. The script reports their
coverage and stops there; identifying the variant empirically is the work that
would unlock a second arm.

Usage:
    build_ladder_tv_pairs.py --parquet <safesyn.parquet> --sidecar <csv>
                             --out <pairs.tsv> [--margin 0.5] [--low-q 50]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

import pyarrow.parquet as pq

# The dial's own materiality constant, in ssim2 points
# (`zensim_validate::dial_addressability::ENCODER_SSIM2_MARGIN_PT`, and
# `bake_verdict`'s `MATERIAL_INV_PT`).
DEFAULT_MARGIN_PT = 0.5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="the training group's parquet (group 0)")
    ap.add_argument("--sidecar", required=True, help="row-aligned CSV carrying codec/quality/size_bytes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN_PT,
                    help="minimum |delta ssim2| in POINTS for a pair to be material")
    ap.add_argument("--low-q", type=float, default=50.0,
                    help="pairs whose higher member sits at or below this q go to band 0")
    ap.add_argument("--index-offset", type=int, default=0,
                    help="added to every emitted index; use when this group is not --group 0")
    ap.add_argument("--anchor-parquet", default=None,
                    help="a SECOND ladder source carrying image_id/codec/codec_param, appended "
                         "to the same TSV at --anchor-offset. Its ladders reach the encoders' "
                         "TRUE lowest settings, which the primary corpus does not.")
    ap.add_argument("--anchor-offset", type=int, default=0,
                    help="global row index of the anchor group's first row (= the summed row "
                         "counts of every --group that precedes it)")
    ap.add_argument("--anchor-bottom-k", type=int, default=3,
                    help="pairs touching the K lowest settings of an anchor ladder go to "
                         "BAND 3 — the same window --floor-rule resolvable reads. Band 3 is "
                         "its own band ON PURPOSE: sharing band 0 with the primary corpus' "
                         "~80k low-q pairs would dilute the floor signal ~300:1 and make "
                         "--tv-band-weights unable to address it.")
    ap.add_argument("--anchor-repeat", type=int, default=1,
                    help="replicate every anchor pair N times in the pool. The sampler draws "
                         "UNIFORMLY from the pair list, so a band weight alone cannot fix a "
                         "1.9%%-of-pool presence — it scales the gradient of a draw that "
                         "almost never happens.")
    ap.add_argument("--anchor-bottom-repeat", type=int, default=1,
                    help="ADDITIONAL replication for the bottom-window pairs, on top of "
                         "--anchor-repeat")
    args = ap.parse_args()

    tbl = pq.read_table(args.parquet, columns=["ref_basename", "human_score"])
    ref_pq = tbl.column("ref_basename").to_pylist()
    score_pq = tbl.column("human_score").to_pylist()
    n = len(ref_pq)
    print(f"parquet: {n} rows", file=sys.stderr)

    rows = []
    with open(args.sidecar, newline="") as f:
        rd = csv.DictReader(f)
        for i, r in enumerate(rd):
            rows.append(r)
    if len(rows) != n:
        sys.exit(f"FATAL: sidecar has {len(rows)} rows, parquet has {n} — not row-aligned")

    # Verify the positional join on EVERY row rather than sampling it.
    n_ba = 0
    for i, r in enumerate(rows):
        # `ref_basename` in the parquet has no extension; the sidecar's
        # `source_path` is a full path to the .png. Compare the stems.
        base = os.path.splitext(os.path.basename(r["source_path"]))[0]
        if base != ref_pq[i]:
            sys.exit(f"FATAL: row {i} ref mismatch: sidecar {base!r} vs parquet {ref_pq[i]!r}")
        if r.get("gpu_butteraugli") or r.get("cpu_butteraugli"):
            n_ba += 1
    print(f"positional join VERIFIED on all {n} rows", file=sys.stderr)
    print(
        f"butteraugli present on {n_ba}/{n} rows — variant UNIDENTIFIED, so the "
        f"two-reference agreement arm is NOT MEASURED",
        file=sys.stderr,
    )

    # ssim2 in POINTS is the trainer's own target column, scaled.
    ssim2 = [s * 100.0 for s in score_pq]

    ladders = defaultdict(list)
    for i, r in enumerate(rows):
        ladders[(ref_pq[i], r["codec"])].append(i)

    n_ladders = len(ladders)
    pairs = []
    n_adjacent = 0
    n_saturated_collapsed = 0
    n_below_margin = 0
    n_short = 0
    for _key, idxs in ladders.items():
        idxs.sort(key=lambda i: float(rows[i]["quality"]))
        # Saturation dedup: drop a step whose encoded size equals the previous
        # kept step's — it is the same setting sampled twice.
        kept = []
        last_size = None
        for i in idxs:
            sz = rows[i]["size_bytes"]
            if last_size is not None and sz == last_size:
                n_saturated_collapsed += 1
                continue
            kept.append(i)
            last_size = sz
        if len(kept) < 2:
            n_short += 1
            continue
        for a, b in zip(kept, kept[1:]):
            n_adjacent += 1
            # `b` is the higher q. Order the emitted pair by the REFERENCE
            # metric, not by q — a backwards rung is exactly what the hinge is
            # supposed to see, but only when it is materially backwards.
            d = ssim2[b] - ssim2[a]
            if abs(d) < args.margin:
                n_below_margin += 1
                continue
            lo, hi = (a, b) if d > 0 else (b, a)
            q_hi = float(rows[hi]["quality"])
            band = 0 if q_hi <= args.low_q else (1 if q_hi < 85.0 else 2)
            pairs.append((lo + args.index_offset, hi + args.index_offset, band))

    # ── the ANCHOR ladders ────────────────────────────────────────────────
    # Their point is the BOTTOM: `--floor-rule resolvable` grades the three
    # lowest mentor-resolvable settings of each ladder, and the primary corpus
    # has no cells there at all (its q grid starts at 5 and steps by 5). Pairs
    # inside that window go to band 0 so `--tv-band-weights` can up-weight
    # exactly the ordering A7r reads.
    n_anchor = 0
    anchor_bottom = 0
    if args.anchor_parquet:
        at = pq.read_table(args.anchor_parquet,
                           columns=["image_id", "codec", "codec_param", "ssim2_gpu"])
        aid = [str(x) for x in at.column("image_id").to_pylist()]
        acod = at.column("codec").to_pylist()
        aparam = at.column("codec_param").to_pylist()
        assim = at.column("ssim2_gpu").to_pylist()
        alad = defaultdict(list)
        for i in range(at.num_rows):
            alad[(aid[i], acod[i])].append(i)
        for _key, idxs in alad.items():
            idxs.sort(key=lambda i: float(aparam[i]))
            if len(idxs) < 2:
                continue
            bottom = set(idxs[: args.anchor_bottom_k])
            for a, b in zip(idxs, idxs[1:]):
                d = assim[b] - assim[a]
                if abs(d) < args.margin:
                    continue
                lo, hi = (a, b) if d > 0 else (b, a)
                is_bottom = a in bottom or b in bottom
                band = 3 if is_bottom else 1
                reps = args.anchor_repeat * (args.anchor_bottom_repeat if is_bottom else 1)
                for _ in range(reps):
                    pairs.append((lo + args.anchor_offset, hi + args.anchor_offset, band))
                n_anchor += reps
                if is_bottom:
                    anchor_bottom += reps

    with open(args.out, "w") as f:
        f.write("lo_idx\thi_idx\tband_id\n")
        for lo, hi, band in pairs:
            f.write(f"{lo}\t{hi}\t{band}\n")

    band_counts = [0, 0, 0, 0]
    for _, _, b in pairs:
        band_counts[b] += 1

    manifest = {
        "_schema": "ladder-tv-pairs-2026-09-06",
        "built_by": "scripts/canonical_corpus/build_ladder_tv_pairs.py",
        "loss_owner": "zensim_validate::mlp_train::TvRegularizer (--tv-pairs-file/--tv-weight/--tv-margin)",
        "inputs": {
            "parquet": {"path": args.parquet, "rows": n, "sha256": sha256(args.parquet)},
            "sidecar": {"path": args.sidecar, "rows": len(rows), "sha256": sha256(args.sidecar)},
        },
        "join": "POSITIONAL, verified on every row via basename(source_path) == ref_basename",
        "index_space": "concatenated group feature rows in --group order",
        "index_offset": args.index_offset,
        "materiality": {
            "rule": "keep a pair only when |delta ssim2| >= margin POINTS",
            "margin_points": args.margin,
            "constant_source": "ENCODER_SSIM2_MARGIN_PT / MATERIAL_INV_PT = 0.5",
            "reference_metric": "human_score * 100 from the training parquet (the trainer's own target)",
        },
        "saturation_dedup": {
            "key": "size_bytes",
            "note": "encode_sha is not present in this sidecar; size_bytes is the available proxy",
            "steps_collapsed": n_saturated_collapsed,
        },
        "butteraugli_agreement": {
            "state": "NOT MEASURED",
            "reason": "the sidecar's gpu_butteraugli/cpu_butteraugli columns do not name their variant; "
                      "the gate's rule needs pnorm3 (margin 0.05) or max (0.25)",
            "rows_with_a_value": n_ba,
        },
        "anchor": {
            "parquet": args.anchor_parquet,
            "offset": args.anchor_offset,
            "bottom_k": args.anchor_bottom_k,
            "pairs_emitted": n_anchor,
            "of_which_bottom_window_band0": anchor_bottom,
            "note": "the anchor ladders reach the encoders' TRUE lowest settings; the primary "
                    "corpus does not, which is the DATA gap this arm tests",
        } if args.anchor_parquet else None,
        "counts": {
            "ladders": n_ladders,
            "ladders_too_short_after_dedup": n_short,
            "adjacent_pairs_considered": n_adjacent,
            "dropped_below_margin": n_below_margin,
            "pairs_emitted": len(pairs),
            "band_0_low_q": band_counts[0],
            "band_1_mid_q": band_counts[1],
            "band_2_high_q": band_counts[2],
            "band_3_anchor_floor_window": band_counts[3],
        },
        "out": {"path": args.out, "sha256": None},
    }
    manifest["out"]["sha256"] = sha256(args.out)
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
