#!/usr/bin/env python3
"""Reduce w4_speed.sh's logs to the W4 table. Reads zenbench's own per-arm
means; forms only min-over-starts and ratios. Validates each start at
collection time (see w4_speed.sh's protocol note): a start whose stable
`fast_ssim2` arm reads below a plausible floor for its size is DISCARDED as
harness degeneration rather than selected by min()."""
import re, sys, glob, os, collections
FLOOR = {576: 5.0, 1152: 20.0, 2304: 80.0}   # ms; fast_ssim2 cannot beat these
LINE = re.compile(r'^\s*(\S+)\s+.*?([0-9]+\.[0-9]+)\s*ms', re.I)

def parse(path):
    starts, cur = [], {}
    for ln in open(path, errors="replace"):
        if "ssim2_speed_bar:" in ln:
            if cur: starts.append(cur)
            cur = {}
            continue
        m = LINE.match(ln)
        if m:
            name, ms = m.group(1), float(m.group(2))
            cur.setdefault(name, ms)
    if cur: starts.append(cur)
    return starts

def main(outdir):
    print(f"{'cell':16s} {'arm':26s} {'min_ms':>9s} {'n_ok':>5s} {'n_bad':>6s} "
          f"{'x_fast_ssim2':>13s} {'x_add156':>9s}")
    for f in sorted(glob.glob(os.path.join(outdir, "w4_t*_s*.log"))):
        m = re.search(r'w4_t(\d+)_s(\d+)\.log', f)
        t, size = int(m.group(1)), int(m.group(2))
        starts = parse(f)
        floor = FLOOR.get(size, 0.0)
        ok = [s for s in starts if s.get("fast_ssim2", 0.0) >= floor]
        bad = len(starts) - len(ok)
        if not ok:
            print(f"t{t}/{size:<12d} ALL {len(starts)} STARTS DISCARDED (fast_ssim2 below {floor} ms floor)")
            continue
        mins = {}
        for s in ok:
            for k, v in s.items():
                mins[k] = min(mins.get(k, 1e18), v)
        base_ss = mins.get("fast_ssim2")
        base_add = mins.get("add156_156basic")
        for arm in sorted(mins):
            r1 = f"{base_ss/mins[arm]:.2f}x" if base_ss else "-"
            r2 = f"{mins[arm]/base_add:.4f}" if base_add else "-"
            print(f"t{t}/{size:<12d} {arm:26s} {mins[arm]:9.3f} {len(ok):5d} {bad:6d} {r1:>13s} {r2:>9s}")
        print()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/mnt/v/output/zensim/fastclass2-2026-09-05/speed")
