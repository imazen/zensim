#!/usr/bin/env python3
"""Batch RD block-selection eval: does the zensim diffmap pick better blocks to
spend bits on than SSE, across many codec pairs, judged independently?

Orchestration only — runs the Rust `rd_block_selection` example (which owns the
diffmap + block logic) and the `zenmetrics` judge (which owns the perceptual
stats). Python here just fans out over pairs and aggregates; it computes no
metric of its own.

Per image it produces `sse` / `zensim` / `random` refined variants (same block
count = same rate) and scores each vs the reference with an INDEPENDENT judge
(butteraugli, a different metric family from zensim). The RD verdict is: does
refining the ZENSIM-selected blocks beat refining the SSE-selected blocks at the
same rate?

Usage:
  python3 scripts/v_next/rd_diffmap_eval.py <pairs.tsv> <out-dir> [--block 32] [--frac 0.25] [--n 40]

pairs.tsv: `ref_path<TAB>dist_path` per line (dist may be a .jpg/.png encode).
"""
import argparse
import pathlib
import statistics
import subprocess
import sys

EXAMPLE = "./target/release/examples/rd_block_selection"
ZM = str(pathlib.Path.home() / "work/zen/zenmetrics/target/release/zenmetrics")


def judge(metric, ref, dist):
    """One independent perceptual score (butteraugli lower=better / ssim2 higher)."""
    try:
        out = subprocess.run(
            [ZM, "score", "--metric", metric, "--reference", ref, "--distorted", dist],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception:
        return None
    import re
    m = re.search(r"[-+]?\d+\.\d+", out)
    return float(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--metric", default="butteraugli")
    a = ap.parse_args()

    lower_better = a.metric in ("butteraugli", "dssim")
    pairs = [ln.split("\t")[:2] for ln in pathlib.Path(a.pairs).read_text().splitlines()
             if ln.strip() and not ln.startswith("ref")][: a.n]
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # per strategy: list of (refined_score − baseline_score) improvements
    improve = {"sse": [], "zensim": [], "butteraugli": [], "random": []}
    z_beats_sse = 0
    n_ok = 0
    overlaps = []
    for ref, dist in pairs:
        stem = pathlib.Path(dist).stem
        wd = outdir / stem
        r = subprocess.run(
            [EXAMPLE, ref, dist, str(wd), "--block", str(a.block), "--frac", str(a.frac)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  skip {stem}: {r.stderr[:80]}", file=sys.stderr)
            continue
        # overlap % printed on stdout
        import re
        mo = re.search(r"(\d+)% agree", r.stdout)
        if mo:
            overlaps.append(int(mo.group(1)))
        base = judge(a.metric, ref, dist)
        if base is None:
            continue
        row = {"base": base}
        for strat in ("sse", "zensim", "butteraugli", "random"):
            vp = wd / f"{stem}__{strat}.png"
            if not vp.exists():
                continue
            s = judge(a.metric, ref, str(vp))
            if s is None:
                break
            # improvement = how much better than baseline (positive = better)
            row[strat] = (base - s) if lower_better else (s - base)
            improve[strat].append(row[strat])
        else:
            n_ok += 1
            if row["zensim"] > row["sse"]:
                z_beats_sse += 1
            print(f"  {stem[:34]:36} base={base:.3f} "
                  f"sse+{row['sse']:+.3f} zensim+{row['zensim']:+.3f} "
                  f"rand+{row['random']:+.3f}  {'ZEN>SSE' if row['zensim']>row['sse'] else ''}")

    print(f"\n=== RD block-selection: {n_ok} pairs, block={a.block} frac={a.frac} judge={a.metric} ===")
    if n_ok:
        for strat in ("sse", "zensim", "butteraugli", "random"):
            v = improve[strat]
            print(f"  {strat:8} mean perceptual improvement (refine {int(a.frac*100)}% of blocks): "
                  f"{statistics.mean(v):+.4f} ± {statistics.pstdev(v):.4f}")
        print(f"\n  zensim beats sse on {z_beats_sse}/{n_ok} pairs "
              f"({100*z_beats_sse/n_ok:.0f}%)")
        if overlaps:
            print(f"  sse∩zensim block overlap: {statistics.mean(overlaps):.0f}% "
                  f"(low = the diffmap makes genuinely different RD choices)")
        mz = statistics.mean(improve["zensim"])
        ms = statistics.mean(improve["sse"])
        mb = statistics.mean(improve["butteraugli"]) if improve["butteraugli"] else None
        print(f"\n  VERDICT (mean perceptual improvement at matched rate, higher=better):")
        print(f"    zensim {mz:+.4f}  vs  sse {ms:+.4f}  => zensim {'WINS' if mz>ms else 'loses'} by {mz-ms:+.4f}")
        if mb is not None:
            # the headline: zensim vs butteraugli (the deployed jxl-encoder RD driver)
            zb = statistics.mean(iz - ib for iz, ib in zip(improve["zensim"], improve["butteraugli"]))
            print(f"    zensim {mz:+.4f}  vs  butteraugli {mb:+.4f}  => "
                  f"{'zensim WINS' if mz>mb else 'butteraugli wins'} by {mz-mb:+.4f} "
                  f"(paired mean Δ {zb:+.4f})")
            print(f"    both vs sse: zensim {mz-ms:+.4f}, butteraugli {mb-ms:+.4f} "
                  f"(a perceptual diffmap should beat SSE; the question is by how much, and vs each other)")


if __name__ == "__main__":
    main()
