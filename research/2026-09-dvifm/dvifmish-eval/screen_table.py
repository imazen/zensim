#!/usr/bin/env python3
"""Markdown tables of a screen decision (`screen_decide.py --json` output).

  screen_table.py decision.json record    # the benchmark-record table (per-seed composites)
  screen_table.py decision.json constants # the docs/CONSTANTS.md table (preset names)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    d = json.load(open(sys.argv[1]))
    mode = sys.argv[2]
    if mode == "record":
        print(f"σ = {d['sigma']:.4f}, bar {d['bar']:.4f}, best `{d['best']}`:\n")
        print("| arm | composite (mean of 3 seeds) | seed SD | per-seed | CID22-A | KonFiG val "
              "| codec_dev (teacher) | safesyn_dev (teacher) | survives |")
        print("|---|---|---|---|---|---|---|---|---|")
        for v in d["arms"]:
            print(f"| {v['arm']} | {v['composite_mean']:.4f} | {v['composite_sd']:.4f} | "
                  + " / ".join(f"{x:.4f}" for x in v["per_seed"])
                  + f" | {v['cid22a']:.4f} | {v['konfig_val']:.4f} | {v['codec_dev']:.4f} | "
                  f"{v['safesyn_dev']:.4f} | {'yes' if v['survives'] else 'no'} |")
    else:
        import os
        os.environ.setdefault("DVIFMISH_R2_BASE", "vis-curve-fit")
        import screen
        print("| Arm | Preset (seed 1) | Composite | Seed SD | CID22-A | KonFiG | codec, teacher "
              "| SafeSyn, teacher |")
        print("|---|---|---|---|---|---|---|---|")
        for v in d["arms"]:
            best = v["composite_mean"] >= d["bar"]
            comp = f"**{v['composite_mean']:.4f}**" if best else f"{v['composite_mean']:.4f}"
            print(f"| {v['arm']} | `{screen.PRESET.get(v['arm'], v['arm'])}` | {comp} | "
                  f"{v['composite_sd']:.4f} | {v['cid22a']:.4f} | {v['konfig_val']:.4f} | "
                  f"{v['codec_dev']:.4f} | {v['safesyn_dev']:.4f} |")


if __name__ == "__main__":
    main()
