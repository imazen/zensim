#!/usr/bin/env python3
"""
Prune SSIM-related weights below an absolute threshold from the V0_2 weight set.

The goal is to drive small SSIM weights to literal zero so the runtime
`active_channels` decision flips `need_ssim` to false for those cells. That
swaps the kernel from the heavy 4-plane H-blur path to the cheap 2-plane
H-blur path for those channel-at-scale combinations.

SSIM-related weight indices per (scale, channel) cell:
- basic[0..3] : ssim_mean, ssim_4th, ssim_2nd
- peak[0]     : ssim_max
- peak[3]     : ssim_p95

Usage:
    python3 prune_ssim.py THRESHOLD > pruned.txt
    python3 prune_ssim.py THRESHOLD --report   # print which cells flip
"""
import re
import sys

REPO_ROOT = "/home/lilith/work/zen/zensim--zero-weight-elide"
PROFILE_RS = f"{REPO_ROOT}/zensim/src/profile.rs"

def load_v0_2():
    src = open(PROFILE_RS).read()
    m = re.search(r'WEIGHTS_PREVIEW_V0_2: \[f64; 228\] = \[(.*?)\];', src, re.S)
    if not m:
        sys.exit("WEIGHTS_PREVIEW_V0_2 not found")
    nums = [float(x) for x in re.findall(r'(-?\d+\.\d+(?:[eE][+-]?\d+)?)', m.group(1))]
    if len(nums) != 228:
        sys.exit(f"Expected 228 weights, got {len(nums)}")
    return nums

def ssim_indices(scale, ch):
    base = scale * 3 * 13 + ch * 13
    peak = 156 + scale * 3 * 6 + ch * 6
    return [base + 0, base + 1, base + 2, peak + 0, peak + 3]

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    threshold = float(sys.argv[1])
    report = "--report" in sys.argv

    weights = load_v0_2()
    pruned = list(weights)

    flips = []
    pruned_count = 0
    for s in range(4):
        for c in range(3):
            idx = ssim_indices(s, c)
            cell_ssim_max = max(abs(weights[i]) for i in idx)
            if cell_ssim_max < threshold:
                # Flip the whole cell's SSIM block to zero
                for i in idx:
                    if abs(pruned[i]) > 0:
                        pruned_count += 1
                    pruned[i] = 0.0
                flips.append((s, "XYB"[c], cell_ssim_max))

    if report:
        print(f"Threshold: {threshold}", file=sys.stderr)
        print(f"Cells with all SSIM weights < threshold (will flip to need_ssim=false): {len(flips)}/12",
              file=sys.stderr)
        for s, ch, m in flips:
            print(f"  scale {s}, channel {ch}  (max abs SSIM weight: {m:.6f})", file=sys.stderr)
        print(f"Total weights pruned: {pruned_count}", file=sys.stderr)
        print(f"Active weights: orig={sum(1 for w in weights if abs(w) > 1e-6)} -> "
              f"pruned={sum(1 for w in pruned if abs(w) > 1e-6)}",
              file=sys.stderr)

    for w in pruned:
        print(f"{w:.10f}")

if __name__ == "__main__":
    main()
