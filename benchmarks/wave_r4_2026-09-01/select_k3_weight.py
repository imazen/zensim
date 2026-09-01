#!/usr/bin/env python3
"""a4bkon lane §24.3: apply the MECHANICAL, pre-registered K3 kon-weight
selection rule to K1's two weights' scored fullevals. Frozen before K1 had
any result (wave_r4_2026-09-01.md §24.3) -- this script only APPLIES that
already-written rule; it does not choose a new one.

Rule:
  1. A weight QUALIFIES if its mean-over-seeds CID22 stays within delta_cid22
     = 0.010 of K4 (A4b-control)'s own CID22.
  2. If both qualify: take the higher mean KonJND.
  3. If only one qualifies: take that one, regardless of KonJND.
  4. If neither qualifies: K3 uses A4b's own default weight (1.2) -- i.e.
     K3 degenerates to "K2 alone" -- stated, not silently substituted.
"""
import json
import sys

O = "/mnt/v/output/zensim/a4bkon-2026-09-01"
DELTA_CID22 = 0.010


def read(name):
    with open(f"{O}/{name}.fulleval.json") as fh:
        d = json.load(fh)
    cid22 = d["rank"]["cid22"]["srocc"]
    konjnd = d["rank"]["konjnd"]["srocc"] if "konjnd" in d["rank"] else d["rank"].get("konjnd1k", {}).get("srocc")
    return cid22, konjnd


def mean(a, b):
    return (a + b) / 2.0


def main():
    k4_cid22, k4_kon = read("K4")
    print(f"K4 (control) CID22={k4_cid22:.4f} KonJND={k4_kon}")

    weights = {}
    for w in ("1.8", "2.4"):
        c4, k4v = read(f"K1_w{w}_s4004")
        c5, k5v = read(f"K1_w{w}_s4005")
        mc, mk = mean(c4, c5), mean(k4v, k5v)
        qualifies = abs(mc - k4_cid22) <= DELTA_CID22
        weights[w] = {"mean_cid22": mc, "mean_konjnd": mk, "qualifies": qualifies}
        print(f"K1 w={w}: mean CID22={mc:.4f} (Δ={mc - k4_cid22:+.4f}) "
              f"mean KonJND={mk:.4f} qualifies={qualifies}")

    qualified = {w: v for w, v in weights.items() if v["qualifies"]}
    if len(qualified) == 2:
        winner = max(qualified, key=lambda w: qualified[w]["mean_konjnd"])
        reason = "both qualify, higher mean KonJND"
    elif len(qualified) == 1:
        winner = next(iter(qualified))
        reason = "only one qualifies"
    else:
        winner = None
        reason = "neither qualifies -- K3 uses A4b's own default 1.2 (K3 = K2 alone)"

    print(f"\nSELECTED: {winner or '1.2 (default, K3=K2 alone)'} ({reason})")
    with open(f"{O}/k3_selection.json", "w") as fh:
        json.dump({"weights": weights, "selected": winner or "1.2", "reason": reason}, fh, indent=1)
    print(f"wrote {O}/k3_selection.json")


if __name__ == "__main__":
    sys.exit(main())
