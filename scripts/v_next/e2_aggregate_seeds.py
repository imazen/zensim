#!/usr/bin/env python3
"""Aggregate the E2 twin-arm verdicts across seeds {1,7,13} into the
pre-registered decision: per-corpus mean±σ Δ(720−372) + the WIN bands + σ(CID22).

Reads the bake_verdict markdown summary tables at
  /home/lilith/tmp/verdict_{ext720,v1372}_s{1,7,13}.md
(seed 1 has no explicit suffix mismatch — its files are verdict_{arm}_s1.md).

Usage: python3 scripts/v_next/e2_aggregate_seeds.py
"""
import re
import statistics as st

TMP = "/home/lilith/tmp"
SEEDS = [1, 7, 13]
ARMS = ["v1372", "ext720"]
ORDER = ["CID22", "KADIK10k", "TID2013", "CSIQ", "LIVE-R2", "KonJND-1k (full)",
         "AIC-3 CTC", "AIC-4 sample", "imazen-26 non-photo (held-out)",
         "imazen-26 real-codec (held-out, ssim2)"]
# clean holdouts (never trained) vs train guards
TRAIN_GUARD = {"KADIK10k", "TID2013"}
COMP_HOLDOUT = {"CID22", "AIC-3 CTC", "AIC-4 sample",
                "imazen-26 real-codec (held-out, ssim2)"}


def rows(path):
    d = {}
    try:
        for ln in open(path):
            m = re.match(r'\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', ln)
            if m:
                name = m.group(1).strip()
                if name and not name.startswith(("Corpus", "Band")):
                    d[name] = float(m.group(3))
    except FileNotFoundError:
        return None
    return d


def main():
    # data[arm][seed] = {corpus: srocc}
    data = {a: {} for a in ARMS}
    have = []
    for a in ARMS:
        for s in SEEDS:
            r = rows(f"{TMP}/verdict_{a}_s{s}.md")
            if r:
                data[a][s] = r
                have.append((a, s))
    seeds_ok = sorted({s for _, s in have})
    print(f"seeds present: {seeds_ok}  (arms×seeds loaded: {len(have)}/{len(ARMS)*len(SEEDS)})")
    if not seeds_ok:
        raise SystemExit("no verdict files found yet")

    # per-corpus mean±σ of Δ across the seeds where BOTH arms are present
    paired = [s for s in seeds_ok if s in data["v1372"] and s in data["ext720"]]
    print(f"paired seeds (both arms): {paired}\n")
    print(f"{'corpus':40s} {'class':12s} {'meanΔ':>8s} {'σΔ':>7s}  per-seed Δ")
    comp_means = []
    worst = (1.0, "")
    cid22_ext = []
    for k in ORDER:
        deltas = []
        for s in paired:
            if k in data["v1372"][s] and k in data["ext720"][s]:
                deltas.append(data["ext720"][s][k] - data["v1372"][s][k])
        if not deltas:
            continue
        mean = st.mean(deltas)
        sd = st.pstdev(deltas) if len(deltas) > 1 else 0.0
        cls = "TRAIN-GUARD" if k in TRAIN_GUARD else ("comp-hold" if k in COMP_HOLDOUT else "clean-FR")
        per = " ".join(f"{d:+.4f}" for d in deltas)
        star = "  <<<" if abs(mean) >= 0.010 and k not in TRAIN_GUARD else ""
        print(f"{k:40s} {cls:12s} {mean:+8.4f} {sd:7.4f}  [{per}]{star}")
        if k in COMP_HOLDOUT:
            comp_means.append(mean)
        if k not in TRAIN_GUARD:
            if mean < worst[0]:
                worst = (mean, k)

    # σ(CID22) across seeds, per arm (the pre-registered instrument-noise gate)
    for a in ARMS:
        vals = [data[a][s]["CID22"] for s in paired if "CID22" in data[a].get(s, {})]
        if len(vals) > 1:
            print(f"  σ(CID22) {a} = {st.pstdev(vals):.4f}  (values {['%.4f'%v for v in vals]})  "
                  f"gate <0.02 -> {'PASS' if st.pstdev(vals) < 0.02 else 'FAIL'}")

    print("\n=== PRE-REGISTERED E2 BANDS (across paired seeds) ===")
    if comp_means:
        cm = st.mean(comp_means)
        print(f"  compression-holdout mean Δ = {cm:+.4f}  (WIN ≥ -0.010) -> {'PASS' if cm >= -0.010 else 'FAIL'}")
    print(f"  worst clean-corpus mean Δ = {worst[0]:+.4f} ({worst[1]})  (WIN no corpus ≤ -0.030) -> "
          f"{'PASS' if worst[0] > -0.030 else 'FAIL'}")


if __name__ == "__main__":
    main()
