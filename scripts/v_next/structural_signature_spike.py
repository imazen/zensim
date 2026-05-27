#!/usr/bin/env python3
"""Approach-B spike (task #33): is there a CONTENT-ROBUST structural-defect
signal that separates a BROKEN localized decode from HONEST compression on
SCREEN content, where perceptual tiling (tile-min / gap) failed?

Hypothesis (error-vs-activity decorrelation): honest compression error
correlates with local source activity — busy/edge regions quantize harder,
flat regions stay clean. A structural decoder bug (channel swap/zero, block
zero/garbage, chroma boundary) injects error that is NOT explained by the
source's local activity: huge error in a tile whose source activity doesn't
justify it. So the discriminator is, per tile, how far the tile's error lies
ABOVE the honest error-vs-activity cloud — which is relative to the source,
hence content-robust.

We compute per 64x64 tile (stride 32) vs the CLEAN ref:
  - luma error (RMSE), chroma error (RMSE)  [BT.601 YCbCr]
  - source luma activity (std of luma in the tile)
Then per image take the WORST tile and test several discriminators against the
honest q20 anchor.

Usage: python3 scripts/v_next/structural_signature_spike.py <corruption_dir> <ref.png> [label]
"""
import sys, os, glob, re, json
import numpy as np
from PIL import Image

TILE = 64
STRIDE = 32

# families that are genuine DECODER BUGS (should rank below honest lq),
# localized to a sub-region (where tile-min failed on screen).
STRUCTURAL = {
    "channel_invert", "channel_swap_rb", "channel_swap_rg", "channel_swap_gb",
    "channel_zero_r", "channel_zero_g", "channel_zero_b", "channel_max_r",
    "block_zero", "block_garbage", "block_gray", "block_copy_wrong",
    "block_repeat_neighbor", "chroma_boundary",
    "composite_premul_as_straight", "composite_wrong_bg_black",
    "composite_wrong_bg_white", "overlay_glyph", "overlay_line", "overlay_rect",
}
LOCAL_REGIONS = {"sq8", "sq16", "sq64", "frac4"}


def rgb_ycc(im):
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _blocks(a, t):
    h, w = a.shape
    hh, ww = (h // t) * t, (w // t) * t
    return a[:hh, :ww].reshape(hh // t, t, ww // t, t)


def _block_reduce_rmse(sq, t):
    """mean over non-overlapping txt blocks of a squared-error map → sqrt."""
    return np.sqrt(_blocks(sq, t).mean(axis=(1, 3)))


def _block_reduce_max(a, t):
    """max over non-overlapping txt blocks. Catches thin/hard defects
    (a 1px line) that tile-MEAN dilutes away."""
    return _blocks(a, t).max(axis=(1, 3))


def _block_reduce_std(y, t):
    """per-block std of luma over non-overlapping txt blocks."""
    yb = _blocks(y, t)
    m = yb.mean(axis=(1, 3), keepdims=True)
    return np.sqrt(((yb - m) ** 2).mean(axis=(1, 3)))


# cache ref decode (it's the same for every dist)
_REF_CACHE = {}


# tile_stats columns
C_LUMA, C_CHROMA, C_SLSTD, C_MAXPIX, C_SCSTD = range(5)


def tile_stats(ref_path, dist_path, tile=TILE):
    """Vectorized per-(non-overlapping tile×tile)-tile stats. Returns array
    (n_tiles, 5): luma_rmse, chroma_rmse, src_luma_std, maxpix_err,
    src_chroma_std.
    - maxpix_err = max over tile pixels of (|Δluma| + |Δchroma|) — the
      thin/hard-defect channel the tile-mean RMSE dilutes away.
    - src_chroma_std = the source tile's chroma activity (the right 'activity'
      baseline for the chroma channel: a chroma defect in a source region that
      is achromatic, hence chroma-flat, is anomalous)."""
    if ref_path not in _REF_CACHE:
        _REF_CACHE[ref_path] = rgb_ycc(Image.open(ref_path))
    ry, rcb, rcr = _REF_CACHE[ref_path]
    dy, dcb, dcr = rgb_ycc(Image.open(dist_path))
    if ry.shape != dy.shape:
        return np.empty((0, 5))
    chroma_e = np.sqrt(((rcb - dcb) ** 2 + (rcr - dcr) ** 2) / 2)
    luma = _block_reduce_rmse((ry - dy) ** 2, tile)
    chroma = _block_reduce_rmse(chroma_e ** 2, tile)
    std = _block_reduce_std(ry, tile)
    maxpix = _block_reduce_max(np.abs(ry - dy) + chroma_e, tile)
    # source chroma activity: std of source chroma magnitude relative to neutral
    src_chroma_mag = np.sqrt(((rcb - 128) ** 2 + (rcr - 128) ** 2) / 2)
    scstd = _block_reduce_std(src_chroma_mag, tile)
    return np.column_stack([luma.ravel(), chroma.ravel(), std.ravel(),
                            maxpix.ravel(), scstd.ravel()])


def parse_name(path):
    name = os.path.basename(path)[:-len("__corruption.png")]
    m = re.match(r".*?__([a-z_0-9]+?)__(whole|frac2|frac4|sq64|sq16|sq8)__(op\d+)$", name)
    if m:
        return name, m.group(1), m.group(2), m.group(3)
    return name, name, "?", "?"


def make_bins(vals):
    b = np.quantile(vals, np.linspace(0, 1, 9))
    b[-1] += 1e-6
    return b


def build_scale(ref, honest_paths, tile):
    """For one tile size: build the per-activity-bin honest p95 bars and return
    a `sig(dist_path) -> (me, pe, ce)` excess function + the (mean,maxpix,chroma)
    bars. me/pe binned by source LUMA activity; ce by source CHROMA activity."""
    honest = [tile_stats(ref, q, tile) for q in honest_paths]
    ha = np.vstack([h for h in honest if h.size])
    lbins, cbins = make_bins(ha[:, C_SLSTD]), make_bins(ha[:, C_SCSTD])
    nlb, ncb = len(lbins) - 1, len(cbins) - 1
    lidx = np.clip(np.digitize(ha[:, C_SLSTD], lbins) - 1, 0, nlb - 1)
    cidx = np.clip(np.digitize(ha[:, C_SCSTD], cbins) - 1, 0, ncb - 1)

    def p95(vals, idx, n):
        return np.array([np.quantile(vals[idx == b], 0.95) if np.any(idx == b)
                         else np.inf for b in range(n)])

    mean_p95 = p95(ha[:, C_LUMA] + ha[:, C_CHROMA], lidx, nlb)
    maxp_p95 = p95(ha[:, C_MAXPIX], lidx, nlb)
    chrm_p95 = p95(ha[:, C_CHROMA], cidx, ncb)

    def sig_arr(ts):
        if ts.size == 0:
            return None
        lb = np.clip(np.digitize(ts[:, C_SLSTD], lbins) - 1, 0, nlb - 1)
        cb = np.clip(np.digitize(ts[:, C_SCSTD], cbins) - 1, 0, ncb - 1)
        return (float(((ts[:, C_LUMA] + ts[:, C_CHROMA]) - mean_p95[lb]).max()),
                float((ts[:, C_MAXPIX] - maxp_p95[lb]).max()),
                float((ts[:, C_CHROMA] - chrm_p95[cb]).max()))

    hsig = [s for s in (sig_arr(h) for h in honest) if s is not None]
    bars = tuple(sorted(s[i] for s in hsig)[int(len(hsig) * 0.95)] for i in range(3))
    return (lambda dist: sig_arr(tile_stats(ref, dist, tile))), bars


def main():
    out_dir, ref = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(out_dir)
    scales = [int(x) for x in os.environ.get("SCALES", "64,16").split(",")]
    corruptions = sorted(glob.glob(os.path.join(out_dir, "*__corruption.png")))

    # Build per-scale honest bars (the defect-vs-honest bar is relative to the
    # source's own local activity at that scale → content-robust). Multi-scale
    # closes the 8x8-defect-vs-64px-tile dilution: 16px catches small defects,
    # 64px keeps honest screen text-ringing from flooding the bar.
    q20s = sorted(glob.glob(os.path.join(out_dir, "*__q20.png")))
    honest_paths = q20s[::10]
    scale_fns = {}
    for t in scales:
        sig, bars = build_scale(ref, honest_paths, t)
        scale_fns[t] = (sig, bars)
        print(f"[{label}] scale {t:3d} honest-q20 bars: "
              f"mean={bars[0]:.2f} maxpix={bars[1]:.2f} chroma={bars[2]:.2f}")

    # Score each defect at every scale; a defect is FLAGGED if any channel at
    # any scale clears that scale's honest bar.
    rows = []
    for c in corruptions:
        name, fam, region, op = parse_name(c)
        if fam not in STRUCTURAL or region not in LOCAL_REGIONS:
            continue
        per_scale = {}
        for t in scales:
            sig, bars = scale_fns[t]
            s = sig(c)
            if s is None:
                break
            per_scale[t] = (s, bars)
        else:
            rows.append((fam, region, op, per_scale))

    def flagged_at(per_scale, t):
        s, bars = per_scale[t]
        return any(s[i] > bars[i] for i in range(3))

    def flagged(per_scale):
        return any(flagged_at(per_scale, t) for t in scales)

    # per-scale and combined gate
    for t in scales:
        p = [r for r in rows if flagged_at(r[3], t)]
        print(f"[{label}] gate scale {t:3d}: {len(p):3d}/{len(rows)} = "
              f"{len(p)/max(len(rows),1)*100:5.1f}%")
    pc = [r for r in rows if flagged(r[3])]
    print(f"[{label}] gate MULTI-SCALE {scales}: {len(pc):3d}/{len(rows)} = "
          f"{len(pc)/max(len(rows),1)*100:5.1f}%")

    # op-level stratified on the multi-scale gate.
    print(f"[{label}] op-stratified (multi-scale gate):")
    for op in ("op100", "op50", "op20"):
        v = [flagged(r[3]) for r in rows if r[2] == op]
        if v:
            print(f"  {op:6s} {sum(v):3d}/{len(v):3d}  {sum(v)/len(v)*100:5.1f}%")

    # per-family on the multi-scale gate
    fams = {}
    for r in rows:
        fams.setdefault(r[0], []).append(flagged(r[3]))
    print("per-family pass (localized regions, multi-scale gate):")
    for fam in sorted(fams):
        v = fams[fam]
        print(f"  {fam:32s} {sum(v):2d}/{len(v):2d}  {sum(v)/len(v)*100:5.1f}%")

    # op100 misses only — the real residual gaps (faint blends excluded)
    miss100 = [r for r in rows if r[2] == "op100" and not flagged(r[3])]
    print(f"\n{len(miss100)} op100 misses (full-strength defect undetected) — the real gaps:")
    for fam, region, op, per_scale in sorted(miss100, key=lambda r: (r[0], r[1])):
        detail = " ".join(f"t{t}({per_scale[t][0][0]:.1f},{per_scale[t][0][1]:.1f},"
                          f"{per_scale[t][0][2]:.1f})" for t in scales)
        print(f"  {fam:26s} {region:6s}  {detail}")


if __name__ == "__main__":
    main()
