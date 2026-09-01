#!/usr/bin/env python3
"""Classify all 944 feature slots by MARGINAL COST GIVEN THE v1-ONLY (156) WALK.

The classification is READ FROM SOURCE (zensim/src/{fused,feature_v2}.rs); this
file encodes it once, emits the per-slot TSV, and prints the counts the doc
quotes.  It is the table generator for
`benchmarks/free_features_2026-09-01.md`, nothing else — no statistics, no
model, no measurement.

Classes (the brief's taxonomy, with one split the source forced):

  A   already-computed, EMIT-ONLY. The accumulator fires unconditionally in
      the 156 walk; emitting costs an f64 store.
  A0  structurally CONSTANT on this route (correct-0). Free to emit and worth
      nothing — a model reads a column of zeros.
  B   finalize-only from sums the 156 walk already keeps.
  C   in-register cheap: a new accumulator over quantities ALREADY LIVE in the
      fused kernel, no new plane / load / pass. Split by what it reads:
        C-raw   reads only `s` / `d`  -> the value is the 944 table's value,
                because no blur is involved.
        C-blur  reads the V-blurred `mu1/mu2/ssq/s12` too. Those ARE in the
                fused kernel's registers, but a v1 band re-inits its V-blur
                recurrence at the band buffer top while phase A runs it over
                the whole strip window, so the value would NOT be the 944
                table's value. Computable free; NOT the same feature.
        C-xch   reads a plane from ANOTHER channel (`ref_y`). No new compute,
                but a new stream into a per-channel kernel.
  D   needs a plane the 156 walk does not make (`activity`, `bs2`, `act_x/b`).
  E   needs its own pass — no new plane, but a traversal the fused kernel does
      not have (a 4-neighbour gradient stencil, the blockiness lattice).

Usage: python3 scripts/freefeats_classify.py [OUT.tsv]
"""
import sys
from collections import Counter

N_SCALES = 4
V1_PER = 31          # 13 basic + 6 peaks + 6 masked + 6 iw
V2_PER = 29
APP_PER = 17
APP2_PER = 5
V1_TOTAL = N_SCALES * 3 * V1_PER          # 372
V2_TOTAL = N_SCALES * 3 * V2_PER          # 348
APP_TOTAL = N_SCALES * 3 * APP_PER        # 204
APP2_TOTAL = N_SCALES * APP2_PER          # 20
APPEND0 = V1_TOTAL + V2_TOTAL             # 720
APPEND2_0 = APPEND0 + APP_TOTAL           # 924
TOTAL = APPEND2_0 + APP2_TOTAL            # 944

# ---- v1 block: basic (computed) / peaks (free) / masked+IW (own pass group)
V1_BASIC = ["ssim_mean", "ssim_l4", "ssim_l2", "art_mean", "art_l4", "art_l2",
            "det_mean", "det_l4", "det_l2", "mse", "hf_var_loss", "hf_mad_loss",
            "hf_var_gain"]
V1_PEAK = ["ssim_max", "art_max", "det_max", "ssim_l8", "art_l8", "det_l8"]
V1_POOL = ["ssim_mean", "ssim_4th", "ssim_2nd", "art_4th", "det_4th", "mse"]

# ---- v2-348, per (scale, channel): (name, class, reads)
V2 = [
    ("SSIM_MEAN",           "C-blur", "mu1,mu2,ssq,s12"),
    ("SSIM_DEV2",           "C-blur", "mu1,mu2,ssq,s12"),
    ("SSIM_DEV4",           "C-blur", "mu1,mu2,ssq,s12"),
    ("ART",                 "C-blur", "mu1,mu2,s,d"),
    ("DET",                 "C-blur", "mu1,mu2,s,d"),
    ("MSE",                 "C-raw",  "s,d"),
    ("HF_GAIN",             "C-blur", "mu1,mu2,s,d"),
    ("HF_LOSS",             "C-blur", "mu1,mu2,s,d"),
    ("HF_MAG_LOSS",         "C-blur", "mu1,mu2,s,d"),
    ("SSIM_SOFT_PEAK",      "C-blur", "mu1,mu2,ssq,s12"),
    ("ART_SOFT_PEAK",       "C-blur", "mu1,mu2,s,d"),
    ("DET_SOFT_PEAK",       "C-blur", "mu1,mu2,s,d"),
    ("MASKED_SSIM",         "D",      "activity"),
    ("MASKED_ART",          "D",      "activity"),
    ("MASKED_DET",          "D",      "activity"),
    ("MASKED_MSE",          "D",      "activity"),
    ("IW_SSIM",             "D",      "activity"),
    ("IW_ART",              "D",      "activity"),
    ("IW_DET",              "D",      "activity"),
    ("IW_MSE",              "D",      "activity"),
    ("PJND_TRANSDUCER",     "D",      "activity"),
    ("PJND_FRAGILITY",      "E",      "grad stencil"),
    ("GMS",                 "E",      "grad stencil"),
    ("PJND_TRANSDUCER_LOW_K",  "D",   "activity"),
    ("PJND_TRANSDUCER_HIGH_K", "D",   "activity"),
    ("BLOCKINESS",          "E",      "lattice pass"),
    ("RINGING",             "D",      "activity+grad"),
    ("BANDING",             "E",      "grad stencil"),
    ("EDGE_WIDTH_CHANGE",   "E",      "grad stencil (2 scales)"),
]
assert len(V2) == V2_PER

# ---- append-204, per (scale, channel)
APP = [
    ("XMASK_TRANSDUCER",  "D",      "act_x,act_b (Y only)"),
    ("LUM_TRANSDUCER",    "D",      "activity,ref_y (Y only)"),
    ("LUM_DARK_ERR",      "C-xch",  "s,d,ref_y"),
    ("LUM_MID_ERR",       "C-xch",  "s,d,ref_y"),
    ("LUM_BRIGHT_ERR",    "C-xch",  "s,d,ref_y"),
    ("MSCN_DIFF_MEAN",    "D",      "bs2"),
    ("MSCN_DIFF_L2",      "D",      "bs2"),
    ("CONTRAST_GAIN",     "D",      "bs2"),
    ("CONTRAST_LOSS",     "D",      "bs2"),
    ("TEXTURE_DISSIM",    "D",      "bs2"),
    ("GMS_DEV2",          "E",      "grad stencil"),
    ("ART_DEV2",          "C-blur", "mu1,mu2,s,d"),
    ("DET_DEV2",          "C-blur", "mu1,mu2,s,d"),
    ("GLOBAL_DMEAN",      "C-raw",  "s,d"),
    ("GLOBAL_CGAIN",      "C-raw",  "s,d"),
    ("GLOBAL_CLOSS",      "C-raw",  "s,d"),
    ("GRAD_SRC_MEAN",     "E",      "grad stencil"),
]
assert len(APP) == APP_PER

APP2 = [
    ("BANDVIS_GAIN",  "D", "activity + 2nd-difference stencil (Y)"),
    ("BANDVIS_LOSS",  "D", "activity + 2nd-difference stencil (Y)"),
    ("LUMA_MEAN_REF", "C-raw", "s (Y)"),
    ("HL_BIN1",       "A0", "HDR-route only -> 0 on SDR"),
    ("HL_BIN2",       "A0", "HDR-route only -> 0 on SDR"),
]
assert len(APP2) == APP2_PER

APPEND_SKIP_B_SCALE0 = True   # feature_v2.rs `append_cell_active`


def rows():
    out = []
    for scale in range(N_SCALES):
        for ch in range(3):
            b = (scale * 3 + ch) * 13
            for i, n in enumerate(V1_BASIC):
                out.append((b + i, "v1_basic", scale, ch, n, "computed",
                            "the 156 walk IS this"))
            c = (scale * 3 + ch) * 6
            for i, n in enumerate(V1_PEAK):
                out.append((N_SCALES * 39 + c + i, "v1_peaks", scale, ch, n, "A",
                            "unconditional in fused_vblur_features_ssim"))
            for i, n in enumerate(V1_POOL):
                out.append((N_SCALES * 57 + c + i, "v1_masked", scale, ch, n, "D",
                            "activity chain + store_sigma + *_inline_both"))
            for i, n in enumerate(V1_POOL):
                out.append((N_SCALES * 75 + c + i, "v1_iw", scale, ch, n, "D",
                            "same pass group as masked"))
    for scale in range(N_SCALES):
        for ch in range(3):
            b = V1_TOTAL + (scale * 3 + ch) * V2_PER
            for i, (n, cl, rd) in enumerate(V2):
                out.append((b + i, "v2_348", scale, ch, n, cl, rd))
            skipped = APPEND_SKIP_B_SCALE0 and ch == 2 and scale == 0
            b = APPEND0 + scale * 3 * APP_PER + ch * APP_PER
            for i, (n, cl, rd) in enumerate(APP):
                if skipped:
                    cl, rd = "A0", "APPEND_SKIP_B_SCALE0 -> 0 in the 944 tables too"
                elif ch != 1 and n in ("XMASK_TRANSDUCER", "LUM_TRANSDUCER"):
                    cl, rd = "A0", "cross-channel cell, 0 for ch != Y"
                out.append((b + i, "append_204", scale, ch, n, cl, rd))
    for scale in range(N_SCALES):
        b = APPEND2_0 + scale * APP2_PER
        for i, (n, cl, rd) in enumerate(APP2):
            out.append((b + i, "append2_20", scale, 1, n, cl, rd))
    out.sort()
    assert [r[0] for r in out] == list(range(TOTAL)), "slot coverage is not 0..944"
    return out


def main():
    rs = rows()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    hdr = "slot\tblock\tscale\tchannel\tname\tclass\treads\n"
    body = "".join("\t".join(map(str, r)) + "\n" for r in rs)
    if path:
        open(path, "w").write(hdr + body)
        print(f"wrote {path}")
    counts = Counter(r[5] for r in rs)
    print(f"{'class':10s} {'slots':>6s}")
    for k in ("computed", "A", "A0", "B", "C-raw", "C-blur", "C-xch", "D", "E"):
        print(f"{k:10s} {counts.get(k, 0):6d}")
    print(f"{'TOTAL':10s} {sum(counts.values()):6d}")
    print()
    print("SHIPPED FREE SET (class A + C-raw excluding the division-bearing v2 MSE):")
    free = [r for r in rs if r[5] == "A"
            or (r[5] == "C-raw" and r[4] in ("GLOBAL_DMEAN", "GLOBAL_CGAIN",
                                             "GLOBAL_CLOSS", "LUMA_MEAN_REF"))]
    print(f"  {len(free)} slots: peaks {sum(1 for r in free if r[5]=='A')} "
          f"+ raw-moment {sum(1 for r in free if r[5]=='C-raw')}")
    print("  raw-moment slot indices:",
          ",".join(str(r[0]) for r in free if r[5] == "C-raw"))


if __name__ == "__main__":
    main()
