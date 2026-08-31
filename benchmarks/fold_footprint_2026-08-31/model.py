#!/usr/bin/env python3
"""Analytic footprint model for the buffered and fold scoring walks, checked
against the measured peak-RSS sweep (`rss_before.tsv` / `rss_after.tsv`).

Every term is derived from source, not fitted; the ONLY fitted quantity is the
per-arm process baseline `P0`, taken as the mean residual over the 128x128 row
(where every image-dependent term is under 4 MiB) and then held fixed for the
other seven sizes.

  ./model.py rss_before.tsv before
  ./model.py rss_after.tsv  after

`fold_model` also returns `strip_alloc`, which the RSS prediction deliberately
does NOT use: under stock glibc the ScratchV2Strip planes a walk never writes
are demand-zero mmap pages that never fault, so the RESIDENT term is the written
set. `strip_alloc` is the allocation-side figure to compare against heaptrack.
"""
import csv, sys

NUM_SCALES, STRIP_ROWS, HALO_P = 4, 128, 10
CAP_SLACK = 32                       # feature_v2_stream::scale_capacity_rows
V1_BAND_ROWS, V1_BAND_OVERLAP = 32, 5
V1_BANDS_PER_STRIP = STRIP_ROWS // V1_BAND_ROWS      # 4
BAND_ROWS = V1_BAND_ROWS + 2 * V1_BAND_OVERLAP       # 42
WIDE = STRIP_ROWS + 2 * HALO_P                       # 148
CHUNK = 64                                           # DEFAULT_CONVERT_CHUNK_ROWS
ADVANCE_MAX = 256
F32 = 4


def advance_rows(t, h, era):
    if era == "before":
        return ADVANCE_MAX
    want = min(max(t * (CHUNK // 2), CHUNK), ADVANCE_MAX)
    a = -(-want // CHUNK) * CHUNK
    return min(a, -(-max(h, 1) // CHUNK) * CHUNK)


def rolling(w, h, adv):
    """24 * sum_s W_s * cap_s -- 2 sides x 3 channels of f32 rolling windows."""
    tot = 0
    for s in range(NUM_SCALES):
        ws, hs = w >> s, h >> s
        step = max(adv >> s, 2)
        cap = min(STRIP_ROWS + 2 * HALO_P + step + CAP_SLACK, max(hs, 1) + 2 * HALO_P)
        tot += ws * cap
    return 2 * 3 * F32 * tot


def fold_model(w, h, t, era):
    adv = advance_rows(t, h, era)
    slots = V1_BANDS_PER_STRIP if era == "before" else min(V1_BANDS_PER_STRIP, t)
    # FoldPoolScratch: 6 `ensure` planes + 4 band-local H planes, per (channel,
    # slot). Before the fix slot 0 was Vec-doubled 37 -> 74 rows.
    if era == "before":
        pool_rows = 3 * BAND_ROWS + 2 * BAND_ROWS   # 3 slots at 42 + slot 0 at 74
    else:
        pool_rows = slots * BAND_ROWS
    pool = 3 * 10 * pool_rows * w * F32
    # ScratchV2Strip: 14 planes allocated before the fix, 2 for a v1_only+Full
    # score after. RESIDENT is the written set either way under stock glibc.
    strip_alloc = (14 if era == "before" else 2) * 3 * WIDE * w * F32
    strip_res = 2 * 3 * WIDE * w * F32
    # The producer interleaves conversion with the band phase (it converts on
    # demand mid-walk), so unlike buffered these DO coexist.
    conv = min(t, max(2 * adv // CHUNK, 1)) * CHUNK * w * 3
    return dict(rolling=rolling(w, h, adv), pool=pool, strip=strip_res,
                strip_alloc=strip_alloc, conv=conv, advance=adv, slots=slots)


def buf_model(w, h, t):
    planes = 2 * 3 * w * h * F32                       # 24*W*H, in-place pyramid
    bands = min(t, max(-(-h // V1_BAND_ROWS), 1)) * 7 * BAND_ROWS * w * F32
    conv = min(t, max(-(-h // CHUNK), 1)) * CHUNK * w * 3
    # The conversion completes before the first ScaleBuffers exists, so the two
    # phases are disjoint in time and the allocator reuses the pages: peak is
    # the MAX, not the sum. (Using the sum over-predicts 16T by 9-14%.)
    return dict(planes=planes, bands=max(bands, conv))


def main():
    path, era = sys.argv[1], sys.argv[2]
    m = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        m[(r["arm"], int(r["size"]), int(r["threads"]))] = int(r["workingset_kib"])
    kib = lambda b: b / 1024.0
    sizes = sorted({k[1] for k in m})
    threads = sorted({k[2] for k in m})

    # P0 from the 128x128 row only.
    p0 = {}
    for arm, fn in (("score_buffered", lambda w, h, t: sum(buf_model(w, h, t).values())),
                    ("score_fold", lambda w, h, t: sum(v for k, v in fold_model(w, h, t, era).items()
                                                       if k in ("rolling", "pool", "strip", "conv")))):
        res = [m[(arm, 128, t)] - kib(fn(128, 128, t)) for t in threads if (arm, 128, t) in m]
        p0[arm] = sum(res) / len(res)
    print(f"# era={era}   fitted P0: buffered {p0['score_buffered']:.0f} KiB, "
          f"fold {p0['score_fold']:.0f} KiB")
    print(f"{'size':>5} {'T':>3} {'arm':>15} | {'pred':>8} {'meas':>8} {'err%':>7} | terms (KiB)")
    for size in sizes:
        if size == 128:
            continue
        for t in threads:
            b = buf_model(size, size, t)
            pb = kib(sum(b.values())) + p0["score_buffered"]
            mb = m[("score_buffered", size, t)]
            print(f"{size:>5} {t:>3} {'score_buffered':>15} | {pb:>8.0f} {mb:>8} "
                  f"{100*(pb-mb)/mb:>6.1f}% | planes {kib(b['planes']):.0f} "
                  f"bands {kib(b['bands']):.0f}")
            f = fold_model(size, size, t, era)
            pf = kib(f["rolling"] + f["pool"] + f["strip"] + f["conv"]) + p0["score_fold"]
            mf = m[("score_fold", size, t)]
            print(f"{size:>5} {t:>3} {'score_fold':>15} | {pf:>8.0f} {mf:>8} "
                  f"{100*(pf-mf)/mf:>6.1f}% | roll {kib(f['rolling']):.0f} "
                  f"pool {kib(f['pool']):.0f} strip {kib(f['strip']):.0f} "
                  f"conv {kib(f['conv']):.0f} (adv {f['advance']} slots {f['slots']})")


def crossover(t, era, lo=128, hi=8192):
    """Smallest square side at which the fold model is <= the buffered model.

    Bisection on the square W=H family, terms only (P0 cancels: it is the same
    process either way and the two arms differ by <300 KiB on the 128 row).
    """
    def fold_minus_buf(w):
        f = fold_model(w, w, t, era)
        b = buf_model(w, w, t)
        return (f["rolling"] + f["pool"] + f["strip"] + f["conv"]) - sum(b.values())
    if fold_minus_buf(hi) > 0:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fold_minus_buf(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "crossover":
        print(f"{'threads':>7} {'before W':>9} {'before MP':>10} "
              f"{'after W':>8} {'after MP':>9}")
        for t in (1, 2, 4, 8, 16):
            wb, wa = crossover(t, "before"), crossover(t, "after")
            print(f"{t:>7} {wb:>9} {wb*wb/1e6:>10.2f} {wa:>8} {wa*wa/1e6:>9.2f}")
        sys.exit(0)
    main()
