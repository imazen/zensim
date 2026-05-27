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


def _block_reduce_rmse(sq):
    """mean over non-overlapping TILExTILE blocks of a squared-error map → sqrt."""
    h, w = sq.shape
    hh, ww = (h // TILE) * TILE, (w // TILE) * TILE
    sq = sq[:hh, :ww].reshape(hh // TILE, TILE, ww // TILE, TILE)
    return np.sqrt(sq.mean(axis=(1, 3)))  # (nty, ntx)


def _block_reduce_max(a):
    """max over non-overlapping TILExTILE blocks. Catches thin/hard defects
    (a 1px line) that tile-MEAN dilutes away."""
    h, w = a.shape
    hh, ww = (h // TILE) * TILE, (w // TILE) * TILE
    a = a[:hh, :ww].reshape(hh // TILE, TILE, ww // TILE, TILE)
    return a.max(axis=(1, 3))


def _block_reduce_std(y):
    """per-block std of luma over non-overlapping TILExTILE blocks."""
    h, w = y.shape
    hh, ww = (h // TILE) * TILE, (w // TILE) * TILE
    yb = y[:hh, :ww].reshape(hh // TILE, TILE, ww // TILE, TILE)
    m = yb.mean(axis=(1, 3), keepdims=True)
    return np.sqrt(((yb - m) ** 2).mean(axis=(1, 3)))


# cache ref decode (it's the same for every dist)
_REF_CACHE = {}


def tile_stats(ref_path, dist_path):
    """Vectorized per-(non-overlapping 64x64)-tile stats. Returns array
    (n_tiles, 4): luma_rmse, chroma_rmse, src_luma_std, maxpix_err.
    maxpix_err = max over the tile's pixels of (|Δluma| + |Δchroma|) — the
    thin/hard-defect channel that the tile-mean RMSE dilutes away."""
    if ref_path not in _REF_CACHE:
        _REF_CACHE[ref_path] = rgb_ycc(Image.open(ref_path))
    ry, rcb, rcr = _REF_CACHE[ref_path]
    dy, dcb, dcr = rgb_ycc(Image.open(dist_path))
    if ry.shape != dy.shape:
        return np.empty((0, 4))
    luma = _block_reduce_rmse((ry - dy) ** 2)
    chroma = _block_reduce_rmse(((rcb - dcb) ** 2 + (rcr - dcr) ** 2) / 2)
    std = _block_reduce_std(ry)
    pix = np.abs(ry - dy) + np.sqrt(((rcb - dcb) ** 2 + (rcr - dcr) ** 2) / 2)
    maxpix = _block_reduce_max(pix)
    return np.column_stack([luma.ravel(), chroma.ravel(), std.ravel(), maxpix.ravel()])


def parse_name(path):
    name = os.path.basename(path)[:-len("__corruption.png")]
    m = re.match(r".*?__([a-z_0-9]+?)__(whole|frac2|frac4|sq64|sq16|sq8)__(op\d+)$", name)
    if m:
        return name, m.group(1), m.group(2), m.group(3)
    return name, name, "?", "?"


def main():
    out_dir, ref = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(out_dir)
    corruptions = sorted(glob.glob(os.path.join(out_dir, "*__corruption.png")))

    # 1) Build the HONEST error-vs-activity cloud from a sample of q20 anchors'
    #    tiles. Decode each sampled q20 ONCE and reuse for both the cloud and
    #    the per-image bar distribution. For each activity bin, the honest p95
    #    luma+chroma error is the bar a structural defect must exceed.
    q20s = sorted(glob.glob(os.path.join(out_dir, "*__q20.png")))
    sample = q20s[::8]  # spike sample for the cloud + bar
    honest = [tile_stats(ref, q) for q in sample]
    ha = np.vstack([h for h in honest if h.size])
    if ha.size == 0:
        print("no honest tiles", file=sys.stderr); return
    act = ha[:, 2]  # src_luma_std
    bins = np.quantile(act, np.linspace(0, 1, 9))
    bins[-1] += 1e-6
    bin_idx = np.clip(np.digitize(act, bins) - 1, 0, len(bins) - 2)

    def per_bin_p95(vals):
        return np.array([
            np.quantile(vals[bin_idx == b], 0.95) if np.any(bin_idx == b) else np.inf
            for b in range(len(bins) - 1)
        ])

    # two activity-binned honest bars: tile-MEAN error and tile-MAX-pixel error.
    mean_p95 = per_bin_p95(ha[:, 0] + ha[:, 1])       # rmse luma+chroma
    maxp_p95 = per_bin_p95(ha[:, 3])                   # max-pixel |Δluma|+|Δchroma|

    def signals(ts):
        """(mean-excess, maxpix-excess) = max over tiles of each channel's
        (value − honest p95 for that tile's activity bin). Content-robust:
        every bar is relative to the source's own local activity."""
        if ts.size == 0:
            return None
        b = np.clip(np.digitize(ts[:, 2], bins) - 1, 0, len(bins) - 2)
        me = ((ts[:, 0] + ts[:, 1]) - mean_p95[b]).max()
        pe = (ts[:, 3] - maxp_p95[b]).max()
        return float(me), float(pe)

    # honest q20 bars: p95 of each signal across the honest sample (reuse tiles).
    hsig = [signals(h) for h in honest]
    hsig = [s for s in hsig if s is not None]
    hm = sorted(s[0] for s in hsig); hp = sorted(s[1] for s in hsig)
    mean_bar = hm[int(len(hm) * 0.95)]
    maxp_bar = hp[int(len(hp) * 0.95)]
    print(f"[{label}] honest-q20 bars: mean-excess p95={mean_bar:.2f}  "
          f"maxpix-excess p95={maxp_bar:.2f}  (defects must clear EITHER)")

    # 2) Score localized structural defects on BOTH signals.
    rows = []
    for c in corruptions:
        name, fam, region, op = parse_name(c)
        if fam not in STRUCTURAL or region not in LOCAL_REGIONS:
            continue
        s = signals(tile_stats(ref, c))
        if s is None:
            continue
        rows.append((fam, region, op, s[0], s[1]))

    def gate(r, which):
        me, pe = r[3], r[4]
        if which == "mean":
            return me > mean_bar
        if which == "maxpix":
            return pe > maxp_bar
        return me > mean_bar or pe > maxp_bar   # "either" — the combined gate

    for which in ("mean", "maxpix", "either"):
        p = [r for r in rows if gate(r, which)]
        print(f"[{label}] gate ({which:6s}): {len(p):3d}/{len(rows)} = "
              f"{len(p)/max(len(rows),1)*100:5.1f}%")

    # per-family on the combined ("either") gate
    fams = {}
    for r in rows:
        fams.setdefault(r[0], []).append(gate(r, "either"))
    print("per-family pass (localized regions, combined gate):")
    for fam in sorted(fams):
        v = fams[fam]
        print(f"  {fam:32s} {sum(v):2d}/{len(v):2d}  {sum(v)/len(v)*100:5.1f}%")

    # worst misses on the combined gate (neither signal cleared its bar)
    miss = [r for r in rows if not gate(r, "either")]
    miss = sorted(miss, key=lambda r: max(r[3] - mean_bar, r[4] - maxp_bar))
    print(f"\n{len(miss)} misses (combined gate). by op-level:")
    op_miss = {}
    for r in miss:
        op_miss.setdefault(r[2], 0)
        op_miss[r[2]] += 1
    for op in sorted(op_miss):
        print(f"  {op}: {op_miss[op]} misses")
    print("closest 12 misses (fam region op  mean-ex maxpix-ex):")
    for fam, region, op, me, pe in miss[-12:]:
        print(f"  {fam:28s} {region:6s} {op:6s}  {me:7.2f} {pe:7.2f}")


if __name__ == "__main__":
    main()
