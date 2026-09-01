#!/usr/bin/env python3
"""hybrid_exam_table.py — the amended W1–W7 scorecard over the lane's artifacts.

Reads ONLY what the owners wrote: `bake_verdict --full-json` fullevals (rank +
dial), the exam's `paired_perref_boot.py` outputs (CIs), and the peer_ssim2
board cell. It computes NO statistic — it compares stored numbers to the
registered thresholds and counts clauses.

Thresholds are exam §2.4, verbatim and unchanged:
  delta_corpus 0.010 pooled; delta_cid22-within 0.004; K=2 with >=1 win on
  CID22 or hfnl_cid22band; ladder bar = ssim2's own measured value; S floor 0.85.
"""
import json, re, os, glob, argparse

D_CORPUS, D_WITHIN, K = 0.010, 0.004, 2
POOLED = ["cid22", "csiq", "live", "aic3", "aic4", "konjnd"]
NAMED = {"cid22", "hfnl_cid22band"}

def peer():
    p = "/mnt/v/output/zensim/reports/fulleval/peer_ssim2.fulleval.json"
    r = json.load(open(p))["rank"]
    out = {c: abs(r[c]["srocc_signed"]) for c in POOLED if c in r}
    out["hfnl_cid22band"] = r["hfnl_cid22band"]["srocc_signed"]
    out["per_ref"] = {"hfnl_cid22band": r["hfnl_cid22band"]["per_ref_mean"]}
    return out

ZONE_RE = re.compile(
    r'^\|\s*all\s*\|\s*all\s*\|\s*q>=85\s*\|(?:[^|]*\|){7}\s*(\d+)%\s*\|')

def zone_ends_backwards(verdict_md):
    """Share of q>=85 ladders ENDING backwards, read out of the verdict's own
    zone table (`dial.zones`, scheme ladder-inversion-2026-08-31). Read, never
    recomputed."""
    if not os.path.exists(verdict_md):
        return None
    for line in open(verdict_md):
        m = ZONE_RE.match(line)
        if m:
            return float(m.group(1)) / 100.0
    return None

BOOT_RE = re.compile(r'^(\S+)\t(-?[0-9.]+)\t(-?[0-9.]+)\t([-+][0-9.]+)\t([-+][0-9.]+)\t([-+][0-9.]+)\t([0-9.]+)$')

def read_boot(path, corpus, band=False):
    """-> {arm: {'pooled': (d, lo, hi), 'within': (d, lo, hi)}} from one boot file."""
    out, mode = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path):
        if line.startswith("candidate\tpooled"):
            mode = "pooled"; continue
        if line.startswith("candidate\tmean"):
            mode = "within"; continue
        m = BOOT_RE.match(line.rstrip("\n"))
        if m and mode:
            out.setdefault(m.group(1), {})[mode] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/v/output/zensim/hybrid-2026-09-01")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args()
    P = peer()
    B = {}
    for c in ["cid22", "csiq", "aic3", "live", "aic4"]:
        B[c] = read_boot(f"{a.dir}/boot_{c}.txt", c)
    B["hfnl_cid22band"] = read_boot(f"{a.dir}/boot_cid22_band0.8.txt", "cid22", band=True)
    for extra in (f"{a.dir}/boot_cid22_band0.8_decay.txt",):
        for k, v in read_boot(extra, "cid22", band=True).items():
            B["hfnl_cid22band"].setdefault(k, v)
    names = a.arms or [os.path.basename(p)[:-len(".fulleval.json")]
                       for p in sorted(glob.glob(f"{a.dir}/HY*.fulleval.json"))]
    rows = []
    for n in names:
        f = f"{a.dir}/{n}.fulleval.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); r = d["rank"]; dial = d.get("dial") or {}
        vals = {c: abs(r[c]["srocc_signed"]) for c in POOLED if c in r}
        prf = {c: (r[c].get("per_ref_mean") or 0.0) for c in POOLED if c in r}
        # W1
        w1_pool = {c: vals[c] - P[c] for c in POOLED if c in vals}
        w1_ok = all(v >= -D_CORPUS for v in w1_pool.values())
        # within-image: cid22 at its own tighter delta, others at delta_corpus.
        pr_peer = {"cid22": 0.9613, "csiq": 0.9084, "aic3": 0.9521, "live": 0.9566}
        w1_within = {c: prf[c] - pr_peer[c] for c in pr_peer if c in prf}
        w1_ok = w1_ok and all(
            v >= -(D_WITHIN if c == "cid22" else D_CORPUS) for c, v in w1_within.items())
        # W2 — strict wins with a CI excluding zero
        wins = []
        for c, tbl in B.items():
            e = tbl.get(n)
            if not e:
                continue
            for mode, lab in (("pooled", f"{c}"), ("within", f"{c}·wi")):
                if mode in e and e[mode][1] > 0:
                    wins.append(lab)
        named_hit = any(w.split("·")[0] in NAMED for w in wins)
        w2_ok = len(wins) >= K and named_hit
        # W3 — ssim2's OWN measured ladder values on the SAME (pools) grid,
        # from `bake_verdict --dial-peer-scores` (dial_peer_ssim2_poolsgrid.md):
        # 33 material inversions over 4,702 rung pairs, 0 % of q>=85 ladders
        # ending backwards. The bar is the opponent's number, so it cannot
        # drift; it is exact here rather than the 4-dp printed form, because at
        # 4 dp several arms tie it and a tie must resolve as PASS ('>=').
        SSIM2_MONO = 1.0 - 33.0 / 4702.0        # 0.9929817099106764
        SSIM2_ENDS_BACKWARDS_Q85 = 0.0
        mono = dial.get("mono_pct")
        ends = zone_ends_backwards(f"{a.dir}/{n}.verdict.md")
        w3_ok = (mono is not None and mono >= SSIM2_MONO
                 and ends is not None and ends <= SSIM2_ENDS_BACKWARDS_Q85)
        rows.append(dict(name=n, ends_backwards=ends, w1=w1_ok, w1_pool=w1_pool, w1_within=w1_within,
                         w2=w2_ok, wins=wins, named=named_hit, mono=mono, w3=w3_ok,
                         dyn=dial.get("dynamic_range"), tied=dial.get("tied_pct"),
                         nonphoto=r.get("nonphoto", {}).get("srocc_signed"),
                         imazen26=r.get("imazen26", {}).get("srocc_signed"),
                         cid22=vals.get("cid22"), kon=vals.get("konjnd")))
    hdr = f"{'arm':14s} {'W1':>4s} {'W2':>4s} {'W3':>4s}  {'wins (CI excl. 0)':40s} {'worstW1':>8s}"
    print(hdr); print("-" * len(hdr))
    for x in rows:
        worst = min(list(x["w1_pool"].values()) + list(x["w1_within"].values()))
        print(f"{x['name']:14s} {'PASS' if x['w1'] else 'FAIL':>4s} "
              f"{'PASS' if x['w2'] else 'FAIL':>4s} {'PASS' if x['w3'] else 'FAIL':>4s}  "
              f"{', '.join(x['wins'])[:40]:40s} {worst:+8.4f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
