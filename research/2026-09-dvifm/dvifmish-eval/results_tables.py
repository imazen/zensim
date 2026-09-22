#!/usr/bin/env python3
"""Markdown tables for the dvifmish RESULTS page from the final run's outputs.

  results_tables.py aic2026 <repro work dir> [preset ...]
      Spearman of each preset's E with the organisers' three DVIFM columns,
      crops / full resolution, plus per-source and per-codec means.
  results_tables.py corruption <repro work dir> <peers dir> [preset ...]
      §6: below-anchor shares and matched-FP detection, validation / training
      sources, DVIFM presets (float path) beside the peers.
  results_tables.py types <repro work dir> <set> <preset>
      SROCC within each distortion type (TID2013 / KADID-10k full sets), with
      the datasets' own type names (TID2013 readme; KADID-10k database page).
"""
import json
import sys
from pathlib import Path

COLS = ("proposal-DVIFM", "proposal-DVIFM-0.2", "proposal-DVIFM-0.2-use_chroma")
TID_TYPES = ["Additive Gaussian noise", "Additive noise in colour components", "Spatially correlated noise",
             "Masked noise", "High frequency noise", "Impulse noise", "Quantization noise", "Gaussian blur",
             "Image denoising", "JPEG compression", "JPEG2000 compression", "JPEG transmission errors",
             "JPEG2000 transmission errors", "Non eccentricity pattern noise",
             "Local block-wise distortions of different intensity", "Mean shift (intensity shift)",
             "Contrast change", "Change of color saturation", "Multiplicative Gaussian noise", "Comfort noise",
             "Lossy compression of noisy images", "Image color quantization with dither",
             "Chromatic aberrations", "Sparse sampling and reconstruction"]
KADID_TYPES = ["Gaussian blur", "Lens blur", "Motion blur", "Color diffusion", "Color shift",
               "Color quantization", "Color saturation 1", "Color saturation 2", "JPEG2000", "JPEG",
               "White noise", "White noise in color component", "Impulse noise", "Multiplicative noise",
               "Denoise", "Brighten", "Darken", "Mean shift", "Jitter", "Non-eccentricity patch", "Pixelate",
               "Quantization", "Color block", "High sharpen", "Contrast change"]


def aic2026(work, presets):
    r = {k: json.loads((work / "results" / f"aic2026_{k}__float.json").read_text())["presets"]
         for k in ("crop", "full")}
    names = presets or sorted(r["crop"])
    print("| Preset | " + " | ".join(f"`{c}`" for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for n in names:
        cells = [f"{r['crop'][n][c]['srocc']:.3f} / {r['full'][n][c]['srocc']:.3f}" for c in COLS]
        print(f"| `{n}` | " + " | ".join(cells) + " |")
    print()
    print("Within sources / within codecs (mean SROCC, crops):")
    for n in names:
        print(f"- `{n}`: " + "; ".join(
            f"{c}: {r['crop'][n][c]['per_source']['srocc']:.2f} / {r['crop'][n][c]['per_codec']['srocc']:.2f}"
            for c in COLS))


def corruption(work, peers, presets):
    rows = []
    for label, path, key in (("dvifm", work / "corruption", "dvifm_{s}_float.json"),
                             ("peer", peers, "peers_{s}.json")):
        v = json.loads((path / key.format(s="validate")).read_text())["models"]
        t = json.loads((path / key.format(s="train")).read_text())["models"]
        for name in (presets if label == "dvifm" and presets else sorted(v)):
            rows.append((label, name, v[name], t[name]))
    print("| Metric | below q20 anchor | below q10 anchor | detection at 1% FP | detection at 5% FP |")
    print("|---|---|---|---|---|")
    for label, name, v, t in rows:
        shown = f"`{name}`" if label == "dvifm" else name
        cells = [f"{v[k]:.3f} / {t[k]:.3f}" for k in ("pass_q20", "pass_q10", "detect_at_fp1", "detect_at_fp5")]
        print(f"| {shown} | " + " | ".join(cells) + " |")


def types(work, set_, preset):
    import csv
    from collections import defaultdict
    import scipy.stats
    pairs = list(csv.DictReader(open(work / "pairs" / f"{set_}.tsv"), delimiter="\t"))
    sc = list(csv.DictReader(open(work / "scores" / set_ / "float" / f"{preset}.tsv"), delimiter="\t"))
    assert len(pairs) == len(sc)
    g = defaultdict(list)
    for r, x in zip(pairs, sc):
        g[r["codec"]].append((float(r["human_score"]), -float(x["distortion"])))
    names = TID_TYPES if set_.startswith("tid") else KADID_TYPES
    out = []
    for k, v in g.items():
        n = int("".join(ch for ch in k if ch.isdigit()))
        out.append((scipy.stats.spearmanr([a for a, _ in v], [b for _, b in v]).statistic, n, names[n - 1]))
    for rho, n, name in sorted(out):
        print(f"{rho:.3f}  #{n:02d} {name}")


def main():
    mode, work = sys.argv[1], Path(sys.argv[2])
    if mode == "aic2026":
        aic2026(work, sys.argv[3:])
    elif mode == "corruption":
        corruption(work, Path(sys.argv[3]), sys.argv[4:])
    elif mode == "types":
        types(work, sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
