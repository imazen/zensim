#!/usr/bin/env python3
"""Map a featsub index file (appendix J K-subsets) onto the folded-944
extraction stages that produce each slot — the appendix-J follow-up
question "can a K-small model skip extraction work?".

Layout source of truth: `zensim/src/feature_v2.rs` (verified 2026-08-05):
  f0..f155    v1-fold      base = scale*39 + ch*13   (13 locals/ch/scale)
  f156..f371  STRUCTURAL ZEROS (folded regimes)
  f372..f719  v2-348       372 + scale*87 + ch*29    (29 locals, `mod idx`)
  f720..f923  append-204   720 + scale*51 + ch*17    (17 locals, `idx_append`)
  f924..f943  append2-20   924 + scale*5  + local    (5 locals, Y-only)

Pass attribution (from `finish_channel_scale` / `finish_append` /
append2 finalize — which accumulator supplies each local):
  v1fold  : `fold_v1_basic_bands` (own V-blur over the H-blurred planes)
  dense   : `dense_block_kernel`   (DenseAccum)
  grad    : `gradient_block_kernel`(GradientAccum; + BANDVIS lanes on Y)
  block   : blockiness kernel      (separate accumulator)
  append  : `append_block_kernel`  (AppendAccum; HL bins on HDR route)
  a2final : append2 finalize-only  (LUMA_MEAN_REF — free, from append Σs)

Usage:
  python3 scripts/featsub/k128_stage_map.py benchmarks/featsub/idx/top128.idx \
      [--ranked benchmarks/featsub/idx/ranked.tsv] [--out out.tsv]
"""

import argparse
import sys
from collections import defaultdict

V1_LOCALS = [
    "ssim_p1", "ssim_p4", "ssim_p2",
    "art_p1", "art_p4", "art_p2",
    "det_p1", "det_p4", "det_p2",
    "mse", "hf_loss", "hf_mag_loss", "hf_gain",
]
# all v1 locals come from the one fused v1-fold pass
V1_PASS = {i: "v1fold" for i in range(13)}

V2_LOCALS = [
    "SSIM_MEAN", "SSIM_DEV2", "SSIM_DEV4", "ART", "DET", "MSE",
    "HF_GAIN", "HF_LOSS", "HF_MAG_LOSS",
    "SSIM_SOFT_PEAK", "ART_SOFT_PEAK", "DET_SOFT_PEAK",
    "MASKED_SSIM", "MASKED_ART", "MASKED_DET", "MASKED_MSE",
    "IW_SSIM", "IW_ART", "IW_DET", "IW_MSE",
    "PJND_TRANSDUCER", "PJND_FRAGILITY", "GMS",
    "PJND_TRANSDUCER_LOW_K", "PJND_TRANSDUCER_HIGH_K",
    "BLOCKINESS", "RINGING", "BANDING", "EDGE_WIDTH_CHANGE",
]
V2_PASS = {}
for i in range(29):
    V2_PASS[i] = "dense"
for i in (21, 22, 26, 27, 28):  # PJND_FRAGILITY, GMS, RINGING, BANDING, EWC
    V2_PASS[i] = "grad"
V2_PASS[25] = "block"  # BLOCKINESS

APP_LOCALS = [
    "XMASK_TRANSDUCER", "LUM_TRANSDUCER",
    "LUM_DARK_ERR", "LUM_MID_ERR", "LUM_BRIGHT_ERR",
    "MSCN_DIFF_MEAN", "MSCN_DIFF_L2",
    "CONTRAST_GAIN", "CONTRAST_LOSS", "TEXTURE_DISSIM",
    "GMS_DEV2", "ART_DEV2", "DET_DEV2",
    "GLOBAL_DMEAN", "GLOBAL_CGAIN", "GLOBAL_CLOSS", "GRAD_SRC_MEAN",
]
APP_PASS = {i: "append" for i in range(17)}
APP_PASS[10] = "grad"  # GMS_DEV2: sum_gms2 accumulates in GradientAccum
APP_PASS[16] = "grad"  # GRAD_SRC_MEAN: grad.sum_grad_src
# ART_DEV2/DET_DEV2 need BOTH (dense first moment + append second moment);
# attribute to append (the marginal pass) — noted in the report.

A2_LOCALS = ["BANDVIS_GAIN", "BANDVIS_LOSS", "LUMA_MEAN_REF", "HL_BIN1", "HL_BIN2"]
A2_PASS = {0: "grad", 1: "grad", 2: "a2final", 3: "append", 4: "append"}

CH = ["X", "Y", "B"]

# --- the OTHER live layout: v1-372 (the pre-fold `compute_extended_features`
# vector the 372-era corpora and bakes use). Verified against
# `zensim/src/metric.rs` passes 1-4 (2026-08-06): the f156-371 block that the
# folded regimes zero is REAL here, and it is three 72-wide sub-blocks, each
# 6 locals/channel/scale (base = off + scale*18 + ch*6). Appendix T needs this
# to NAME the survivors of a 372-root additive fit; the folded decoder above
# would mislabel every one of them as a structural zero.
V1_PEAK_LOCALS = ["ssim_max", "art_max", "det_max", "ssim_p95", "art_p95", "det_p95"]
V1_MASK_LOCALS = ["masked_ssim_p1", "masked_ssim_p4", "masked_ssim_p2",
                  "masked_art_4th", "masked_det_4th", "masked_mse"]
V1_IW_LOCALS = ["iw_ssim_p1", "iw_ssim_p4", "iw_ssim_p2",
                "iw_art_4th", "iw_det_4th", "iw_mse"]


def decode_v1_372(i: int):
    """v1-372 layout -> (block, scale, ch, local_name, pass)."""
    if i < 156:
        s, r = divmod(i, 39)
        c, l = divmod(r, 13)
        return ("v1basic156", s, CH[c], V1_LOCALS[l], "v1basic")
    for off, blk, locals_, pas in (
        (156, "peak72", V1_PEAK_LOCALS, "peak"),
        (228, "masked72", V1_MASK_LOCALS, "masked"),
        (300, "iw72", V1_IW_LOCALS, "iwpool"),
    ):
        if i < off + 72:
            j = i - off
            s, r = divmod(j, 18)
            c, l = divmod(r, 6)
            return (blk, s, CH[c], locals_[l], pas)
    raise ValueError(f"index {i} out of 372 range")


def decode(i: int):
    """-> (block, scale, ch, local_name, pass) — ch is None for append2/zero."""
    if i < 156:
        s, r = divmod(i, 39)
        c, l = divmod(r, 13)
        return ("v1fold156", s, CH[c], V1_LOCALS[l], V1_PASS[l])
    if i < 372:
        return ("zeros156_371", None, None, "STRUCTURAL_ZERO", "none")
    if i < 720:
        j = i - 372
        s, r = divmod(j, 87)
        c, l = divmod(r, 29)
        return ("v2_348", s, CH[c], V2_LOCALS[l], V2_PASS[l])
    if i < 924:
        j = i - 720
        s, r = divmod(j, 51)
        c, l = divmod(r, 17)
        return ("append204", s, CH[c], APP_LOCALS[l], APP_PASS[l])
    if i < 944:
        j = i - 924
        s, l = divmod(j, 5)
        return ("append2_20", s, "Y", A2_LOCALS[l], A2_PASS[l])
    raise ValueError(f"index {i} out of 944 range")


def load_idx(path: str):
    txt = open(path).read()
    body = " ".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    return sorted({int(t) for t in body.replace(",", " ").split() if t.strip()})


def load_ranked(path: str):
    """ranked.tsv columns: rank idx family mean_abs cum_share dead."""
    w = {}
    for ln in open(path):
        if ln.startswith("#") or ln.startswith("rank") or not ln.strip():
            continue
        parts = ln.split()
        w[int(parts[1])] = float(parts[3])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idx_file")
    ap.add_argument("--ranked", default=None, help="ranked.tsv for mean_abs weights")
    ap.add_argument("--out", default=None, help="write per-index TSV here")
    ap.add_argument("--layout", default="folded944", choices=("folded944", "v1_372"),
                    help="feature layout of the model whose indices these are")
    args = ap.parse_args()
    dec = decode if args.layout == "folded944" else decode_v1_372

    ids = load_idx(args.idx_file)
    weights = load_ranked(args.ranked) if args.ranked else {}

    rows = []
    for i in ids:
        blk, s, c, name, pas = dec(i)
        rows.append((i, blk, s, c, name, pas, weights.get(i)))

    out = open(args.out, "w") if args.out else sys.stdout
    out.write("feat_idx\tblock\tscale\tchannel\tlocal\tpass\tmean_abs\n")
    for i, blk, s, c, name, pas, w in rows:
        ws = f"{w:.6g}" if w is not None else "-"
        out.write(f"{i}\t{blk}\t{s if s is not None else '-'}\t{c or '-'}\t{name}\t{pas}\t{ws}\n")
    if args.out:
        out.close()

    # ---- coverage matrices ----
    def emit(title, key):
        agg = defaultdict(int)
        for _, blk, s, c, name, pas, _ in rows:
            agg[key(blk, s, c, pas)] += 1
        print(f"\n== {title} ==", file=sys.stderr)
        for k in sorted(agg, key=str):
            print(f"  {k}: {agg[k]}", file=sys.stderr)

    emit("by block", lambda b, s, c, p: b)
    emit("by scale", lambda b, s, c, p: s)
    emit("by channel", lambda b, s, c, p: c)
    emit("by pass", lambda b, s, c, p: p)
    emit("by (scale, channel)", lambda b, s, c, p: (s, c))
    emit("by (pass, scale, channel)", lambda b, s, c, p: (p, s, c))

    # The extraction-skip analysis below is folded-944-SPECIFIC (its pass
    # universe and the APPEND_SKIP_B_SCALE0 carve-out are that layout's). It is
    # meaningless for --layout v1_372, whose passes are v1basic/peak/masked/
    # iwpool, so it is skipped rather than printed wrong.
    if args.layout != "folded944":
        return

    # untouched (pass, scale, ch) cells over the full ACTIVE grid.
    # append (0, B) is excluded: APPEND_SKIP_B_SCALE0 already skips it in
    # the shipped extractor (its 17 slots are wired zeros).
    all_cells = set()
    for p in ("v1fold", "dense", "grad", "block", "append"):
        for s in range(4):
            for c in CH:
                if p == "append" and (s, c) == (0, "B"):
                    continue
                all_cells.add((p, s, c))
    hit = {(p, s, c) for i, b, s, c, _n, p, _w in rows if s is not None}
    untouched = sorted(all_cells - hit, key=str)
    # pixel-share weighting: scale s carries 4^-s of the base plane's
    # pixels; a (pass, scale, ch) cell's work is ~proportional to that.
    px = {s: 0.25**s for s in range(4)}
    tot = {p: sum(px[s] for (pp, s, c) in all_cells if pp == p) for p in
           ("v1fold", "dense", "grad", "block", "append")}
    skip = defaultdict(float)
    for (p, s, c) in untouched:
        skip[p] += px[s]
    print(f"\n== untouched (pass, scale, channel) cells: {len(untouched)} "
          f"of {len(all_cells)} active ==", file=sys.stderr)
    for u in untouched:
        print(f"  {u}", file=sys.stderr)
    print("\n== pixel-weighted skippable share per pass ==", file=sys.stderr)
    for p in sorted(tot):
        print(f"  {p}: {skip[p]/tot[p]*100:.2f}% of the pass's pixel work",
              file=sys.stderr)


if __name__ == "__main__":
    main()
