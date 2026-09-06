#!/usr/bin/env python3
"""Reduce the W4 per-cell logs to the W4 table.

Reads zenbench's own per-arm numbers; forms only min-over-starts and ratios.

TWO PARSER DEFECTS FIXED 2026-09-06 — both were mine, and together they made the
first collation emit a header and nothing else:

  1. THE GLOB WAS STALE. It looked for `w4_t*_s*.log`, but
     `scripts/fastclass2_w4_deferred.sh` writes `w4_<tier>_t<T>_s<S>.log` — the
     tier was added to the filename and the glob was never updated, so ZERO
     files matched and the loop body never ran. That is why the table had no
     rows and also no "ALL STARTS DISCARDED" line: there was nothing to discard.
  2. THE ARM-NAME REGEX CAPTURED THE BOX-DRAWING PREFIX. zenbench's summary
     rows read `  ├─ fast_ssim2   24.7 ±0.0ms`, and `^\\s*(\\S+)` matches `├─`,
     not the arm. It then matched `0.0` (from `±0.0ms`) as the time, because
     `24.7` is not directly followed by `ms`. Every start would have parsed as
     `{"├─": 0.0}` — a silent, plausible-looking wrong answer, which is worse
     than the empty table defect 1 produced.

Now it parses the BAR-CHART block instead (`  free156_peaks_raw  ████ 6.20ms`),
which carries one more significant figure than the summary table and has no
prefix to confuse the name with. The summary table is the fallback, with an
anchored name pattern.

VALIDATION AT COLLECTION, per CLAUDE.md's zenbench-degeneration rule: a start
whose STABLE control (`fast_ssim2`) reads below a plausible floor for its size
is DISCARDED. `min()` is only safe against noise that is one-directional, and a
harness under a tight wall budget can report a spuriously LOW mean for every arm
at once — which `min()` would then happily select.
"""

import glob
import os
import re
import statistics as st
import sys

# ms; fast_ssim2 physically cannot beat these at the given size on this box.
FLOOR = {576: 5.0, 1152: 20.0, 2304: 80.0}
BAR = re.compile(r"^\s{2,}([A-Za-z][A-Za-z0-9_]*)\s+[█░▓]+\s*([0-9]+\.[0-9]+)ms\s*$")
ROW = re.compile(r"^\s*[├╰│─└┬-]*\s*([A-Za-z][A-Za-z0-9_]*)\s+([0-9]+\.[0-9]+)\s*±")
START = re.compile(r"ssim2_speed_bar:")
ROUNDS = re.compile(r"(\d+)\s+rounds\s*×")


def parse(path):
    """-> [ {arm: ms}, ... ] one dict per process start, plus rounds seen."""
    starts, cur, rounds = [], {}, []
    for ln in open(path, errors="replace"):
        if START.search(ln):
            if cur:
                starts.append(cur)
            cur = {}
            continue
        m = ROUNDS.search(ln)
        if m:
            rounds.append(int(m.group(1)))
        m = BAR.match(ln) or ROW.match(ln)
        if m:
            cur.setdefault(m.group(1), float(m.group(2)))
    if cur:
        starts.append(cur)
    return starts, rounds


def main(outdir):
    files = sorted(glob.glob(os.path.join(outdir, "w4_*_t*_s*.log")))
    if not files:
        # Loud, because an empty match set is exactly what defect 1 looked like.
        print(f"NO LOGS MATCHED {outdir}/w4_*_t*_s*.log — nothing to collate", file=sys.stderr)
        return 2
    print(f"{'cell':18s} {'arm':24s} {'min_ms':>9s} {'med_ms':>9s} {'n_ok':>5s} "
          f"{'n_bad':>6s} {'x_fast_ssim2':>13s} {'/add156':>9s}")
    for f in files:
        m = re.search(r"w4_([a-z0-9]+)_t(\d+)_s(\d+)\.log$", os.path.basename(f))
        tier, t, size = m.group(1), int(m.group(2)), int(m.group(3))
        starts, rounds = parse(f)
        floor = FLOOR.get(size, 0.0)
        ok = [s for s in starts if s.get("fast_ssim2", 0.0) >= floor]
        bad = len(starts) - len(ok)
        cell = f"{tier}/t{t}/{size}"
        if not ok:
            print(f"{cell:18s} ALL {len(starts)} STARTS DISCARDED "
                  f"(fast_ssim2 below the {floor} ms floor for {size}²)")
            continue
        mins, meds = {}, {}
        for arm in sorted({k for s in ok for k in s}):
            vs = [s[arm] for s in ok if arm in s]
            mins[arm], meds[arm] = min(vs), st.median(vs)
        base_ss, base_add = mins.get("fast_ssim2"), mins.get("add156_156basic")
        rtxt = f"rounds {min(rounds)}-{max(rounds)}" if rounds else "rounds ?"
        print(f"# {cell}: {len(ok)} start(s) kept, {bad} discarded, {rtxt}")
        for arm in sorted(mins):
            r1 = f"{base_ss / mins[arm]:.2f}x" if base_ss else "-"
            r2 = f"{mins[arm] / base_add:.4f}" if base_add else "-"
            print(f"{cell:18s} {arm:24s} {mins[arm]:9.3f} {meds[arm]:9.3f} "
                  f"{len(ok):5d} {bad:6d} {r1:>13s} {r2:>9s}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "/mnt/v/output/zensim/fastclass2-2026-09-05/speed"))
