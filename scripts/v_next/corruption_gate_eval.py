#!/usr/bin/env python3
"""Corruption-corpus gate: for every structural-corruption entry, check
score(ref, corruption) < score(ref, q20-honest-lq) through a zensim bake.

The gate is the property zensim's negative tail must satisfy for the
regression-test use case (codec-corpus#7): a structurally-broken decode must
rank BELOW an honestly-lossy encode, so a test catches the bug instead of
passing it.

Usage:
  python3 scripts/v_next/corruption_gate_eval.py <bake.bin> <corruption_out_dir> <ref.png> [label]
"""
import sys, os, glob, subprocess, re
from concurrent.futures import ThreadPoolExecutor

SCORE = "./target/release/score_pair_with_bake"
TILE = "./target/release/score_tiles_with_bake"
TILE_MIN = os.environ.get("TILE_MIN") == "1"  # task #33: tile-min pooling


def score(bake, ref, dist):
    if TILE_MIN:
        # tile-min: localized-defect signal. Output cols: global min p2 p5 median n.
        r = subprocess.run([TILE, "--bake", bake, "--bake-post", "clamp", "--ref", ref,
                            "--dist", dist, "--tile", os.environ.get("TILE_SIZE", "64"),
                            "--overlap", "0.5"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return None
        try:
            return float(r.stdout.strip().split()[1])  # min tile
        except (ValueError, IndexError):
            return None
    r = subprocess.run([SCORE, "--bake", bake, "--bake-post", "raw", "--ref", ref, "--dist", dist],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return None


def main():
    bake, out_dir, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(bake)
    corruptions = sorted(glob.glob(os.path.join(out_dir, "*__corruption.png")))
    print(f"[{label}] {len(corruptions)} corruption entries vs q20/q10 anchors", file=sys.stderr)

    def work(cpath):
        key = cpath[:-len("__corruption.png")]
        q20p, q10p = key + "__q20.png", key + "__q10.png"
        name = os.path.basename(key)
        m = re.match(r".*?__([a-z_0-9]+?)__(whole|frac2|frac4|sq64|sq16|sq8)__(op20|op50|op100)$", name)
        fam, region, sev = (m.group(1), m.group(2), m.group(3)) if m else (name, "?", "?")
        sc = score(bake, ref, cpath)
        sq20 = score(bake, ref, q20p) if os.path.exists(q20p) else None
        sq10 = score(bake, ref, q10p) if os.path.exists(q10p) else None
        return dict(name=name, fam=fam, region=region, sev=sev, sc=sc, sq20=sq20, sq10=sq10)

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(work, corruptions))

    rows = [r for r in rows if r["sc"] is not None and r["sq20"] is not None]
    n = len(rows)
    passed = [r for r in rows if r["sc"] < r["sq20"]]
    print(f"\n[{label}] GATE score(corruption) < score(q20): {len(passed)}/{n} = {len(passed)/n*100:.1f}% PASS")
    # also vs q10 (more aggressive anchor)
    rows10 = [r for r in rows if r["sq10"] is not None]
    p10 = [r for r in rows10 if r["sc"] < r["sq10"]]
    print(f"[{label}] GATE score(corruption) < score(q10): {len(p10)}/{len(rows10)} = {len(p10)/max(len(rows10),1)*100:.1f}% PASS")

    # per-family pass rate
    fams = {}
    for r in rows:
        fams.setdefault(r["fam"], []).append(r["sc"] < r["sq20"])
    print("\nper-family gate(vs q20) pass rate:")
    for fam in sorted(fams):
        v = fams[fam]
        print(f"  {fam:34s} {sum(v):3d}/{len(v):3d}  {sum(v)/len(v)*100:5.1f}%")

    # per-region (subtlety axis: smaller region = harder)
    regs = {}
    for r in rows:
        regs.setdefault(r["region"], []).append(r["sc"] < r["sq20"])
    print("\nper-region gate(vs q20) pass rate (smaller = harder/subtler):")
    for reg in ["whole", "frac2", "frac4", "sq64", "sq16", "sq8"]:
        if reg in regs:
            v = regs[reg]
            print(f"  {reg:8s} {sum(v):3d}/{len(v):3d}  {sum(v)/len(v)*100:5.1f}%")

    # worst FAILURES (corruption scored ABOVE q20 = metric let a bug pass)
    fails = sorted([r for r in rows if r["sc"] >= r["sq20"]], key=lambda r: r["sc"] - r["sq20"], reverse=True)
    print(f"\n{len(fails)} FAILURES (corruption >= q20). worst 20:")
    print(f"  {'name':52s} {'corr':>8s} {'q20':>8s}")
    for r in fails[:20]:
        print(f"  {r['name']:52s} {r['sc']:8.1f} {r['sq20']:8.1f}")


if __name__ == '__main__':
    main()
