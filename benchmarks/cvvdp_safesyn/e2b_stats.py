#!/usr/bin/env python3
"""cvvdp-safesyn E2b analysis — display selection on TRAIN human legs.

Statistics contract (prereg benchmarks/cvvdp-safesyn_prereg_2026-09-23.md):
  * pooled SROCC + within-image (per-reference) SROCC per leg x display,
    via zensim_validate::panel (scripts/lib/zen_stats.py — the only stat
    code path; nothing reimplemented here).
  * paired delta = display - standard_4k, reference-clustered bootstrap
    B=2000, seed 20260923, random.Random(20260923) over reference NAMES,
    expanded to row indices (E1 analyze.py convention); the same draw set
    is applied to every arm, so deltas are paired.
  * KonFiG leg: triplet ordering accuracy on q_jnd via
    `panel --pairwise --resample` (within-reference stimulus pairs,
    q_jnd_a != q_jnd_b, choice = side with the larger q_jnd).
  * CI: two-sided percentile at 1 - 0.05/6 = 0.991667 (Bonferroni over the
    6 challenger arms), on the delta resample distribution.
  * selection: beats standard_4k only if pooled delta >= +0.010 AND the
    99% cluster CI excludes 0 on BOTH kadid and tid, AND konfig accuracy
    delta is not significantly negative; among passers take the largest
    mean delta over the two SROCC legs.

Inputs: /var/tmp/cvvdp-safesyn/e2b/scores/{leg}__{display}.tsv written by
run_e2b_scores.sh (zenmetrics batch --group-by-ref; row order may be
grouped — we re-key on ref/dist paths, never on row order).
"""
import csv, json, math, random, subprocess, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "lib"))
import zen_stats

B = 2000
SEED = 20260923
ALPHA = 0.05 / 6          # Bonferroni over the 6 challenger arms
CI_LO, CI_HI = ALPHA / 2, 1 - ALPHA / 2
CTRL = "standard_4k"
DISPLAYS = ["standard_4k", "sdr_4k_30", "standard_fhd", "sdr_fhd_24",
            "standard_phone", "iphone_14_pro", "modern_oled_phone_indoor"]
CHALLENGERS = [d for d in DISPLAYS if d != CTRL]
COL = lambda d: "cvvdp_cpu_imazen_v0_1_0" + ("" if d == "standard_4k" else f"_{d}")
SCORES = "/var/tmp/cvvdp-safesyn/e2b/scores"
SCRATCH = "/var/tmp/cvvdp-safesyn/e2b/stats"
SROCC_LEGS = ["kadid", "tid"]
ALL_LEGS = ["kadid", "tid", "konfig"]


def load_leg(leg):
    """Join the 7 display TSVs on (ref_path, dist_path); verify keys align."""
    joined = None
    for d in DISPLAYS:
        path = f"{SCORES}/{leg}__{d}.tsv"
        rows = list(csv.DictReader(open(path), delimiter="\t"))
        keyed = {(r["ref_path"], r["dist_path"]): r for r in rows}
        assert len(keyed) == len(rows), f"{path}: duplicate (ref,dist) keys"
        if joined is None:
            joined = {k: {"ref": k[0], "dist": k[1],
                          "t": float(v["human_score"]),
                          **({"q": float(v["q_jnd"])} if "q_jnd" in v else {})}
                      for k, v in keyed.items()}
        else:
            assert set(keyed) == set(joined), f"{leg}/{d}: key set mismatch vs {CTRL}"
        col = COL(d)
        for k, v in keyed.items():
            joined[k][d] = float(v[col])
    rows = list(joined.values())
    print(f"  {leg}: {len(rows)} rows, {len({r['ref'] for r in rows})} refs", file=sys.stderr)
    return rows


def draws(refs, B, seed):
    """E1 convention: random.Random(seed) over reference names."""
    rng = random.Random(seed)
    n = len(refs)
    return [[refs[rng.randrange(n)] for _ in range(n)] for _ in range(B)]


def pct(v, q):
    v = sorted(x for x in v if not math.isnan(x))
    if not v:
        return float("nan")
    return v[min(len(v) - 1, max(0, int(q * len(v))))]


def srocc_leg(rows, ref_draws, chunk=400):
    """Point + per-resample pooled SROCC per display (owner: panel --batch)."""
    refs = sorted({r["ref"] for r in rows})
    by_ref = {}
    for j, r in enumerate(rows):
        by_ref.setdefault(r["ref"], []).append(j)
    bases = {"t": [r["t"] for r in rows]}
    for d in DISPLAYS:
        bases[d] = [r[d] for r in rows]
    jobs = [(f"{d}|P", d, "t", None) for d in DISPLAYS]
    sel = []
    for b, dr in enumerate(ref_draws):
        s = [j for r in dr for j in by_ref.get(r, [])]
        sel.append(s)
        for d in DISPLAYS:
            jobs.append((f"{d}|{b}", d, "t", s))
    res = []
    for i in range(0, len(jobs), chunk * len(DISPLAYS)):
        res.extend(zen_stats.panel_batch_indexed(
            bases, jobs[i:i + chunk * len(DISPLAYS)], stats="srocc", timeout=3600))
    out = {d: {"point": float("nan"), "boot": [float("nan")] * len(ref_draws)}
           for d in DISPLAYS}
    for r in res:
        d, k = r["label"].split("|")
        v = r.get("srocc_signed", r.get("srocc"))
        v = float("nan") if v is None else float(v)
        if k == "P":
            out[d]["point"] = v
        else:
            out[d]["boot"][int(k)] = v
    return out


def perref_srocc(leg):
    """Within-image (per-reference) SROCC via panel --input --per-group."""
    out = {}
    for d in DISPLAYS:
        p = subprocess.run(
            [zen_stats._find_panel_bin(), "--input", f"{SCORES}/{leg}__{d}.tsv",
             "--json", "--col-predicted", COL(d), "--col-target", "human_score",
             "--col-band", "ref_path", "--per-group"],
            capture_output=True, text=True, check=True)
        j = json.loads(p.stdout)
        out[d] = (j.get("per_group") or {}).get("mean")
    return out


def konfig_pairwise(rows, ref_draws):
    """Triplet ordering accuracy on q_jnd (owner: panel --pairwise --resample)."""
    os.makedirs(SCRATCH, exist_ok=True)
    by_ref = {}
    for i, r in enumerate(rows):
        by_ref.setdefault(r["ref"], []).append(i)
    pairs = []
    for ref, ii in by_ref.items():
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                qi, qj = rows[i]["q"], rows[j]["q"]
                if qi == qj:
                    continue
                # choice = side with LARGER q_jnd (more distorted).
                pairs.append((ref, i, j, "left" if qi > qj else "right"))
    groups = sorted({p[0] for p in pairs})
    gidx = {g: k for k, g in enumerate(groups)}
    man = f"{SCRATCH}/konfig__resample.tsv"
    with open(man, "w") as f:
        f.write("POINT\t*\n")
        for b, dr in enumerate(ref_draws):
            s = [str(gidx[r]) for r in dr if r in gidx]
            if s:
                f.write(f"B{b}\t{','.join(s)}\n")
    out = {}
    for d in DISPLAYS:
        rf = f"{SCRATCH}/konfig__{d}.tsv"
        with open(rf, "w") as f:
            f.write("group\ts_left\ts_right\tchoice\tweight\n")
            for (ref, i, j, ch) in pairs:
                f.write(f"{ref}\t{rows[i][d]}\t{rows[j][d]}\t{ch}\t1\n")
        p = subprocess.run(
            [zen_stats._find_panel_bin(), "--pairwise", rf, "--resample", man],
            capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"panel --pairwise failed konfig/{d}:\n{p.stderr[-2000:]}")
        lines = p.stdout.strip().split("\n")
        hdr = lines[0].split("\t")
        recs = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
        boot = [float("nan")] * len(ref_draws)
        point = tie = float("nan")
        for rec in recs:
            if rec["label"] == "POINT":
                point = float(rec["acc_response"]); tie = float(rec["tie_rate"])
            else:
                boot[int(rec["label"][1:])] = float(rec["acc_response"])
        out[d] = {"point": point, "boot": boot, "tie_rate": tie}
    return out, len(pairs), len(groups)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    legs = {leg: load_leg(leg) for leg in ALL_LEGS}
    report = {"B": B, "seed": SEED, "ci": [CI_LO, CI_HI], "legs": {}}

    for leg in SROCC_LEGS:
        rows = legs[leg]
        refs = sorted({r["ref"] for r in rows})
        dr = draws(refs, B, SEED)
        st = srocc_leg(rows, dr)
        pr = perref_srocc(leg)
        entry = {"n": len(rows), "refs": len(refs), "arms": {}}
        for d in DISPLAYS:
            boot = st[d]["boot"]
            ctrl = st[CTRL]["boot"]
            delta = [a - c for a, c in zip(boot, ctrl)]
            entry["arms"][d] = {
                "srocc_pooled": st[d]["point"],
                "srocc_perref": pr[d],
                "delta_point": st[d]["point"] - st[CTRL]["point"],
                "delta_ci": [pct(delta, CI_LO), pct(delta, CI_HI)],
                "delta_boot_n": sum(not math.isnan(x) for x in delta),
            }
        report["legs"][leg] = entry
        print(f"  {leg} done", file=sys.stderr)

    # KonFiG triplet leg
    rows = legs["konfig"]
    refs = sorted({r["ref"] for r in rows})
    dr = draws(refs, B, SEED)
    pw, npairs, ngroups = konfig_pairwise(rows, dr)
    entry = {"n": len(rows), "refs": len(refs), "triplets": npairs,
             "triplet_groups": ngroups, "arms": {}}
    for d in DISPLAYS:
        boot = pw[d]["boot"]; ctrl = pw[CTRL]["boot"]
        delta = [a - c for a, c in zip(boot, ctrl)]
        entry["arms"][d] = {
            "acc_point": pw[d]["point"], "tie_rate": pw[d]["tie_rate"],
            "delta_point": pw[d]["point"] - pw[CTRL]["point"],
            "delta_ci": [pct(delta, CI_LO), pct(delta, CI_HI)],
        }
    report["legs"]["konfig"] = entry
    print("  konfig done", file=sys.stderr)

    # ---- selection rule (verbatim from prereg) ----
    sel = {}
    for d in CHALLENGERS:
        kd = report["legs"]["kadid"]["arms"][d]
        td = report["legs"]["tid"]["arms"][d]
        kg = report["legs"]["konfig"]["arms"][d]
        ok_srocc = (kd["delta_point"] >= 0.010 and kd["delta_ci"][0] > 0 and
                    td["delta_point"] >= 0.010 and td["delta_ci"][0] > 0)
        ok_konfig = not (kg["delta_ci"][1] < 0)   # not significantly negative
        sel[d] = {
            "kadid": {"delta": kd["delta_point"], "ci": kd["delta_ci"]},
            "tid": {"delta": td["delta_point"], "ci": td["delta_ci"]},
            "konfig": {"delta": kg["delta_point"], "ci": kg["delta_ci"]},
            "passes": bool(ok_srocc and ok_konfig),
            "mean_srocc_delta": (kd["delta_point"] + td["delta_point"]) / 2,
        }
    passers = [d for d in CHALLENGERS if sel[d]["passes"]]
    selected = max(passers, key=lambda d: sel[d]["mean_srocc_delta"]) if passers else None
    report["selection"] = {"challengers": sel, "selected": selected,
                           "rule": "delta>=+0.010 pooled & 99.1667% CI excl 0 on kadid AND tid; "
                                   "konfig delta not significantly negative; max mean delta"}

    out = "/var/tmp/cvvdp-safesyn/e2b/e2b_stats.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"wrote {out}", file=sys.stderr)
    print(json.dumps({"selected": selected, "passers": passers}))


if __name__ == "__main__":
    main()
