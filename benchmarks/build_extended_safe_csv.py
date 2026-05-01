#!/usr/bin/env python3
"""Build training_safe_synthetic_extended.csv from training.csv post-fill.

Pipeline (mirrors what produced training_safe_synthetic.csv from
training_concordant.csv):

1. Read full training.csv.
2. Group rows by source_path.
3. For each source group, run pairwise concordance filtering between
   ssim2 and dssim. Iteratively drop the row with the highest discordant
   count until all remaining pairs are concordant.
4. Drop rows whose source stem matches any of the 49 CID22 validation
   stems (the 41 hardcoded blocklist + 8 additional contaminated stems
   identified empirically by comparing existing safe vs full).
5. Write rows in original order to extended.csv.

Designed to be ledger-stable: the new e1 rows in training.csv get
filtered through concordance + CID22 ban exactly like the original
codecs were.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

CID22_BLOCKLIST = set([
    "1025469", "1044329", "1189261", "1279330", "1418519", "1420710",
    "1475938", "1531677", "1544947", "159550", "1624487", "162520",
    "164595", "2079234", "2190188", "225228", "2253934", "2389166",
    "2670327", "2736139", "2775196", "2887497", "2936831", "297394",
    "3156482", "3316926", "3637739", "3653963", "373965", "3762075",
    "382297", "4215100", "5055743", "5458393", "6078297", "6292444",
    "70497", "7062219", "7552578", "792079", "844297",
])

# Empirically identified: stems present in training.csv but absent from
# training_safe_synthetic.csv beyond the 41-stem blocklist. Treated as
# additional contaminated CID22 sources (per CLAUDE.md note: "475
# CID22-contaminated pairs removed (7 unblocked CID22 stems × ~68 pairs
# each)" — close to the 8 we observed).
CID22_EXTRA = set([
    "21169144185_3f7977cb5a_o",
    "3316926_opo25u",
    "adriankierman-report-page",
    "pexels-photo-1933873",
    "pexels-photo-2686358",
    "pexels-photo-2802032",
    "pexels-photo-4210863",
    "ularapi_Semarang_City_Logo",
])

CID22_BAN = CID22_BLOCKLIST | CID22_EXTRA

def stem_for_path(p: str) -> str:
    """Extract canonical source stem from source_path.

    Source paths look like:
      /mnt/v/input/zensim/sources/<stem>_<bucket>.png
    or with multi-bucket suffixes like 1022x818_512sq. We want the
    leading <stem> token before the first _<bucket> suffix.
    """
    name = Path(p).stem
    # Strip every trailing _<bucket> suffix.
    BUCKETS = ("512sq", "1024sq", "769x513", "513x769", "1022x818", "818x1022")
    while True:
        stripped = False
        for b in BUCKETS:
            suf = f"_{b}"
            if name.endswith(suf):
                name = name[: -len(suf)]
                stripped = True
                break
        if not stripped:
            return name

def is_banned(source_path: str) -> bool:
    s = stem_for_path(source_path)
    return s in CID22_BAN

def concordant(a, b):
    # a, b are (ssim2, dssim) tuples; ssim2 higher = better, dssim lower = better
    s_sign = 0 if a[0] == b[0] else (1 if a[0] > b[0] else -1)
    d_sign = 0 if a[1] == b[1] else (1 if a[1] < b[1] else -1)
    if s_sign == 0 or d_sign == 0:
        return True
    return s_sign == d_sign

def filter_group(rows):
    """rows: list of (orig_idx, ssim2, dssim). Returns set of orig_idx kept."""
    n = len(rows)
    if n <= 1:
        return {r[0] for r in rows}
    # Build discordant counts.
    disc = [0] * n
    pairs = []  # (i, j, is_concordant)
    for i in range(n):
        for j in range(i + 1, n):
            c = concordant((rows[i][1], rows[i][2]), (rows[j][1], rows[j][2]))
            if not c:
                disc[i] += 1
                disc[j] += 1
                pairs.append((i, j))
    if not pairs:
        return {r[0] for r in rows}
    # Iteratively drop the row with the highest discordant count.
    alive = set(range(n))
    while True:
        # Recompute discordant counts among alive rows.
        d = [0] * n
        for i, j in pairs:
            if i in alive and j in alive:
                d[i] += 1
                d[j] += 1
        worst = max((d[i], i) for i in alive)
        if worst[0] == 0:
            break
        alive.discard(worst[1])
        if len(alive) <= 1:
            break
    return {rows[i][0] for i in alive}

def main():
    if len(sys.argv) < 3:
        print("Usage: build_extended_safe_csv.py <input.csv> <output.csv>")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    src_idx = header.index("source_path")
    ssim2_idx = header.index("gpu_ssimulacra2")
    dssim_idx = header.index("dssim")

    # Group by source_path; track original indices.
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        try:
            ssim2 = float(row[ssim2_idx])
            dssim = float(row[dssim_idx])
        except (ValueError, IndexError):
            continue  # Skip rows with bad metrics.
        if is_banned(row[src_idx]):
            continue  # Skip CID22-contaminated rows.
        groups[row[src_idx]].append((i, ssim2, dssim))

    keep = set()
    n_total = 0
    for src, group_rows in groups.items():
        kept = filter_group(group_rows)
        keep.update(kept)
        n_total += len(group_rows)

    out_rows = [row for i, row in enumerate(rows) if i in keep]
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in out_rows:
            w.writerow(row)
    print(f"Read {len(rows)} rows from {in_path}")
    print(f"  After CID22 ban + bad-metric drop: {n_total} rows in {len(groups)} groups")
    print(f"  After concordance filter: {len(out_rows)} rows kept")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
