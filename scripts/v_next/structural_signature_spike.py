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


# tile_stats columns
C_LUMA, C_CHROMA, C_SLSTD, C_MAXPIX, C_SCSTD = range(5)


def tile_stats(ref_path, dist_path):
    """Vectorized per-(non-overlapping 64x64)-tile stats. Returns array
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
    luma = _block_reduce_rmse((ry - dy) ** 2)
    chroma = _block_reduce_rmse(chroma_e ** 2)
    std = _block_reduce_std(ry)
    maxpix = _block_reduce_max(np.abs(ry - dy) + chroma_e)
    # source chroma activity: std of source chroma magnitude relative to neutral
    src_chroma_mag = np.sqrt(((rcb - 128) ** 2 + (rcr - 128) ** 2) / 2)
    scstd = _block_reduce_std(src_chroma_mag)
    return np.column_stack([luma.ravel(), chroma.ravel(), std.ravel(),
                            maxpix.ravel(), scstd.ravel()])


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
    # Two activity axes: src LUMA std (for the luma+maxpix channels) and src
    # CHROMA std (for the chroma channel — a chroma defect in a chroma-flat
    # source region is the anomaly to catch). Each gets its own quantile bins.
    def make_bins(vals):
        b = np.quantile(vals, np.linspace(0, 1, 9))
        b[-1] += 1e-6
        return b

    lbins = make_bins(ha[:, C_SLSTD])
    cbins = make_bins(ha[:, C_SCSTD])
    lidx = np.clip(np.digitize(ha[:, C_SLSTD], lbins) - 1, 0, len(lbins) - 2)
    cidx = np.clip(np.digitize(ha[:, C_SCSTD], cbins) - 1, 0, len(cbins) - 2)

    def per_bin_p95(vals, idx, n):
        return np.array([
            np.quantile(vals[idx == b], 0.95) if np.any(idx == b) else np.inf
            for b in range(n)
        ])

    nlb, ncb = len(lbins) - 1, len(cbins) - 1
    mean_p95 = per_bin_p95(ha[:, C_LUMA] + ha[:, C_CHROMA], lidx, nlb)  # luma+chroma rmse
    maxp_p95 = per_bin_p95(ha[:, C_MAXPIX], lidx, nlb)                  # max-pixel
    chrm_p95 = per_bin_p95(ha[:, C_CHROMA], cidx, ncb)                  # chroma-only rmse

    def signals(ts):
        """(mean-excess, maxpix-excess, chroma-excess) = max over tiles of each
        channel's (value − honest p95 for that tile's activity bin). Each bar is
        relative to the source's own local activity → content-robust."""
        if ts.size == 0:
            return None
        lb = np.clip(np.digitize(ts[:, C_SLSTD], lbins) - 1, 0, nlb - 1)
        cb = np.clip(np.digitize(ts[:, C_SCSTD], cbins) - 1, 0, ncb - 1)
        me = ((ts[:, C_LUMA] + ts[:, C_CHROMA]) - mean_p95[lb]).max()
        pe = (ts[:, C_MAXPIX] - maxp_p95[lb]).max()
        ce = (ts[:, C_CHROMA] - chrm_p95[cb]).max()
        return float(me), float(pe), float(ce)

    # honest q20 bars: p95 of each signal across the honest sample (reuse tiles).
    hsig = [s for s in (signals(h) for h in honest) if s is not None]
    def bar95(i):
        v = sorted(s[i] for s in hsig)
        return v[int(len(v) * 0.95)]
    mean_bar, maxp_bar, chrm_bar = bar95(0), bar95(1), bar95(2)
    print(f"[{label}] honest-q20 bars: mean={mean_bar:.2f}  maxpix={maxp_bar:.2f}  "
          f"chroma={chrm_bar:.2f}  (defect clears ANY → flagged)")

    # 2) Score localized structural defects on all THREE signals.
    rows = []
    for c in corruptions:
        name, fam, region, op = parse_name(c)
        if fam not in STRUCTURAL or region not in LOCAL_REGIONS:
            continue
        s = signals(tile_stats(ref, c))
        if s is None:
            continue
        rows.append((fam, region, op, s[0], s[1], s[2]))

    def gate(r, which):
        me, pe, ce = r[3], r[4], r[5]
        return {
            "mean": me > mean_bar,
            "maxpix": pe > maxp_bar,
            "chroma": ce > chrm_bar,
            "any": me > mean_bar or pe > maxp_bar or ce > chrm_bar,
        }[which]

    for which in ("mean", "maxpix", "chroma", "any"):
        p = [r for r in rows if gate(r, which)]
        print(f"[{label}] gate ({which:6s}): {len(p):3d}/{len(rows)} = "
              f"{len(p)/max(len(rows),1)*100:5.1f}%")

    # op-level stratified on the combined gate: op100 = full-strength defect
    # (the number that matters); op20 = 20%-opacity faint blend (near-imperceptible).
    print(f"[{label}] op-stratified (combined 'any' gate):")
    for op in ("op100", "op50", "op20"):
        v = [gate(r, "any") for r in rows if r[2] == op]
        if v:
            print(f"  {op:6s} {sum(v):3d}/{len(v):3d}  {sum(v)/len(v)*100:5.1f}%")

    # per-family on the combined gate
    fams = {}
    for r in rows:
        fams.setdefault(r[0], []).append(gate(r, "any"))
    print("per-family pass (localized regions, combined gate):")
    for fam in sorted(fams):
        v = fams[fam]
        print(f"  {fam:32s} {sum(v):2d}/{len(v):2d}  {sum(v)/len(v)*100:5.1f}%")

    # op100 misses only — the real residual gaps (faint blends excluded)
    miss100 = [r for r in rows if r[2] == "op100" and not gate(r, "any")]
    print(f"\n{len(miss100)} op100 misses (full-strength defect undetected) — the real gaps:")
    for fam, region, op, me, pe, ce in sorted(miss100, key=lambda r: (r[0], r[1])):
        print(f"  {fam:28s} {region:6s}  mean={me:7.2f} maxpix={pe:7.2f} chroma={ce:7.2f}")


if __name__ == "__main__":
    main()
