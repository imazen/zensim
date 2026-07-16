#!/usr/bin/env python3
"""RD block-selection matrix: which distortion signal, used to pick WHICH blocks
a codec spends bits on, produces the best encode — across multiple judges?

Orchestration only. Runs the Rust `rd_block_selection` example (which owns the
zensim + butteraugli diffmaps and the block logic) and the `zenmetrics` judges
(which own the perceptual stats). Python fans out + aggregates; no metric here.

SELECTORS (the RD "driver" — which blocks get the refinement budget):
  sse         — Σ(R−D)² per block (the PSNR-optimal codec default)
  zensim      — Σ zensim-diffmap per block (candidate)
  butteraugli — Σ butteraugli-diffmap per block (jxl-encoder's deployed driver)
  random      — control

JUDGES (score the refined result vs reference — every selector under every judge):
  butter_max    butteraugli max-norm     (lower=better)
  butter_3norm  butteraugli 3-norm/pnorm3 (lower=better; the Cloudinary CID22 agg)
  ssim2         SSIMULACRA2               (higher=better)
  zensim_B      shipped zensim Profile B  (higher=better)

CIRCULARITY IS EXPLICIT: a selector scored by its OWN metric is home-turf
(butteraugli-selector under butter_max/3norm; zensim-selector under zensim_B) —
those cells are marked. The fair reads are (a) does each perceptual selector beat
`sse` under EVERY judge, and (b) zensim vs butteraugli under the OTHER's judge.

Usage: rd_diffmap_eval.py <pairs.tsv> <out-dir> [--block 32] [--frac 0.25] [--n 50]
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys

EXAMPLE = "./target/release/examples/rd_block_selection"
ZM = str(pathlib.Path.home() / "work/zen/zenmetrics/target/release/zenmetrics")

SELECTORS = ["sse", "zensim", "butteraugli", "random"]
# judge key -> (zenmetrics --metric, json score field, lower_is_better, home_selector)
JUDGES = {
    "butter_max": ("butteraugli", "butteraugli_max", True, "butteraugli"),
    "butter_3norm": ("butteraugli", "butteraugli_pnorm3", True, "butteraugli"),
    "ssim2": ("ssim2", "ssim2", False, None),
    "zensim_B": ("zensim", "zensim", False, "zensim"),
}


def zscore(metric, ref, dist):
    """Return the full zenmetrics score dict for a pair (one call, maybe 2 keys)."""
    try:
        out = subprocess.run(
            [ZM, "score", "--metric", metric, "--reference", ref,
             "--distorted", dist, "--output", "json"],
            capture_output=True, text=True, timeout=180,
        ).stdout
        return json.loads(out).get("scores", {})
    except Exception:
        return {}


def score_all(ref, dist):
    """All judge fields for one image, minimizing zenmetrics calls."""
    vals = {}
    for zmetric in {"butteraugli", "ssim2", "zensim"}:
        vals.update(zscore(zmetric, ref, dist))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    pairs = [ln.split("\t")[:2] for ln in pathlib.Path(a.pairs).read_text().splitlines()
             if ln.strip() and not ln.startswith("ref")][: a.n]
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # improve[selector][judge] = list of per-image (improvement over baseline,
    # oriented higher=better regardless of the judge's native direction).
    improve = {s: {j: [] for j in JUDGES} for s in SELECTORS}
    overlaps = {"sse_zen": [], "zen_butter": [], "sse_butter": []}
    n_ok = 0

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
        import re
        for key, pat in [("sse_zen", r"sse∩zensim=(\d+)%"),
                         ("zen_butter", r"zensim∩butter=(\d+)%"),
                         ("sse_butter", r"sse∩butter=(\d+)%")]:
            m = re.search(pat, r.stdout)
            if m:
                overlaps[key].append(int(m.group(1)))

        base = score_all(ref, dist)
        if not all(JUDGES[j][1] in base for j in JUDGES):
            continue
        row_ok = True
        pending = {}
        for sel in SELECTORS:
            vp = wd / f"{stem}__{sel}.png"
            if not vp.exists():
                row_ok = False
                break
            sc = score_all(ref, str(vp))
            for j, (_, field, lower, _home) in JUDGES.items():
                if field not in sc:
                    row_ok = False
                    break
                d = (base[field] - sc[field]) if lower else (sc[field] - base[field])
                pending.setdefault(sel, {})[j] = d
            if not row_ok:
                break
        if not row_ok:
            continue
        n_ok += 1
        for sel in SELECTORS:
            for j in JUDGES:
                improve[sel][j].append(pending[sel][j])
        print(f"  {stem[:30]:32} " + "  ".join(
            f"{sel[:4]}:{pending[sel]['ssim2']:+.2f}" for sel in SELECTORS))

    if not n_ok:
        sys.exit("no pairs scored")

    # --- the matrix ---
    print(f"\n=== RD selector × judge matrix ({n_ok} pairs, block={a.block}, "
          f"refine {int(a.frac*100)}% of blocks) ===")
    print("mean perceptual improvement over the unrefined encode (higher=better; "
          "same block count = same rate)\n")
    hdr = f"{'selector':12}" + "".join(f"{j:>14}" for j in JUDGES)
    print(hdr)
    print("-" * len(hdr))
    means = {s: {j: statistics.mean(improve[s][j]) for j in JUDGES} for s in SELECTORS}
    for sel in SELECTORS:
        cells = []
        for j in JUDGES:
            home = JUDGES[j][3] == sel
            cells.append(f"{means[sel][j]:+.3f}{'*' if home else ' '}")
        print(f"{sel:12}" + "".join(f"{c:>14}" for c in cells))
    print("\n* = home-turf (selector judged by its own metric — expected to win; not a fair cell)")

    # --- the fair verdicts ---
    print("\nFAIR READS:")
    for j in JUDGES:
        z, s, b = means["zensim"][j], means["sse"][j], means["butteraugli"][j]
        beats_sse = "zensim>sse" if z > s else "zensim<sse"
        vs_b = "zensim>butter" if z > b else "zensim<butter"
        tag = "  (butter home)" if JUDGES[j][3] == "butteraugli" else \
              ("  (zensim home)" if JUDGES[j][3] == "zensim" else "  (neutral)")
        print(f"  under {j:13}: {beats_sse} (Δ{z-s:+.3f}), {vs_b} (Δ{z-b:+.3f}){tag}")
    for k, label in [("sse_zen", "sse∩zensim"), ("zen_butter", "zensim∩butter"),
                     ("sse_butter", "sse∩butter")]:
        if overlaps[k]:
            print(f"  block overlap {label:14}: {statistics.mean(overlaps[k]):.0f}%")


if __name__ == "__main__":
    main()
