#!/usr/bin/env python3
"""Rev4 E1b — render the committed record (md + compact json) from the full run.

Inputs: /var/tmp/rev4-e1b/e1b_full.json (crosscodec.py) and
/var/tmp/rev4-e1b/direction.json (direction.py, exploratory). No statistic here.
"""
import hashlib
import json
import sys

FULL = "/var/tmp/rev4-e1b/e1b_full.json"
DIR = "/var/tmp/rev4-e1b/direction.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/rev4_e1b_crosscodec_2026-09-23"

NAMES = {"cid22a": "CID22-A(25)", "cid22a_encdir": "CID22-A, E1 encoder-dir codec (sens. iii)",
         "csiq": "CSIQ JPEG+JPEG2000", "aic3": "AIC-3 CTC (decoded PNG, source res.)",
         "aic4crop": "AIC-4 crop (620×800, as shown)", "aic4full": "AIC-4 full res. (secondary)",
         "fc_btc_native": "JPEG-AIC forced choice btc_native", "tid": "TID2013 JPEG+JPEG2000 (TRAIN, descriptive)"}
PEERLAB = {"cvvdp_4k": "cvvdp_4k (not documented display)", "cvvdp_gpu_unrec": "cvvdp GPU (display unrecorded)"}
MODELS = {"B": "B (served default; CSIQ: 07-07 bake/08-30 root; FC: 07-07 bake v1)",
          "C": "C = W10L9PH_s4004_packed", "D": "D (MT914_matched_D; CSIQ D_shipped@dguard2)",
          "R915_fast": "Rev3 fast R915_y60_h32_ens5", "R915_rich": "Rev3 rich R915_basic228_h128_ens5",
          "V0_2": "PreviewV0_2 (CID22/AIC-3 May-era site parquet; AIC-4 pixel read)"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def f(x):
    return f"{x:+.4f}"


def ci(v):
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]" + ("**" if v[2] < 0 else ("*" if v[1] > 0 else ""))


def main():
    d = json.load(open(FULL))
    dr = json.load(open(DIR))
    U = d["units"]
    L = []
    w = L.append
    w("# Rev4 E1b — is zensim's human deficit a cross-codec ordering problem? (2026-09-23)\n")
    w("Preregistered: `benchmarks/rev4_e1b_prereg_2026-09-23.md`. Scripts: `benchmarks/rev4_e1b_2026-09-23/`. "
      f"Full run `{FULL}` (sha256 `{sha(FULL)[:16]}…`), B = {d['B']}, seed {d['seed']}, reference-clustered "
      "(forced choice: source-image-clustered, 5 images). Every accuracy is `acc_response` from `panel --pairwise` "
      "(sha256 `f11857c2…`, E1's binary). `**` CI entirely < 0, `*` CI entirely > 0. Δ = zensim − best peer of that pair class.\n")
    w("**This re-tests E1's exploratory lead on the same data it came from** (prereg §1). A confirmation here would mean the lead "
      "survives a fixed rule, not that it replicates.\n")
    w("## Decision (prereg §6)\n")
    w("| model | primary | family members separate (i) | best peer fixed (ii) | CID22 E1 codec def. (iii) | unit classes (primary) |")
    w("|---|---|---|---|---|---|")
    for z, v in d["decisions"].items():
        p = v["primary"]
        w(f"| {MODELS[z]} | **{p['outcome']}** | {v['sens_members_separate']['outcome']} | {v['sens_fixed_peer']['outcome']} | "
          f"{v['sens_cid22_encdir']['outcome']} | " + ", ".join(f"{k} {x}" for k, x in p["units"].items())
          + "; members " + ", ".join(f"{k} {x}" for k, x in p["members"].items() if k in ("aic3", "aic4crop", "fc_btc_native"))
          + f"; AIC-4 full {v['aic4full_class']}; TID {v['tid_class']} |")
    w("\nCONF = cross deficit (CI < 0) with same-codec tie/win; NONSPEC = same-codec deficit (CI < 0) not demonstrably smaller than the cross one; "
      "NONE = no deficit; MISSING = no rows.\n")
    w("""## Reading

- **B (served default): REFUTED.** On the JPEG-AIC family its deficit is cross-codec-specific: AIC-3 Δx −0.041, AIC-4 crop
  −0.054, forced choice −0.016, all CI < 0, with same-codec ties at the ceiling. It is not cross-specific elsewhere. On CSIQ it
  loses as much on JPEG/JPEG2000 ladders as across them (Δs −0.062 [−0.078, −0.045], Δx −0.061; 07-07 bake era, not the
  served bytes). On CID22-A its only deficit is same-codec (Δs −0.0026 vs IW-SSIM, CI < 0); cross pairs tie SSIMULACRA2.
  With the family members counted separately, B becomes UNRESOLVED (3 CONF vs 2 NONSPEC).
- **C (`W10L9PH_s4004`): UNRESOLVED.** Only CID22-A shows the pattern: Δx −0.0089 [−0.0130, −0.0049] vs SSIMULACRA2, a
  same-codec tie, and DiD −0.0088 (CI < 0). AIC-4 crop vs CVVDP at `standard_fhd` is −0.020 [−0.0415, +0.0020] on 5 references
  and just misses. AIC-3, forced choice and CSIQ tie. On the AIC-4 full-resolution rendering C beats SSIMULACRA2 (+0.0076*).
  **To resolve:** a second multi-codec corpus with enough references (the full AIC-4 set rather than the 5-reference sample),
  CVVDP at its documented display on AIC-3 and CID22-A (E2b), and forced-choice peers other than SSIMULACRA2.
- **Same-codec pairs are near the ceiling on JPEG-AIC.** Every metric except butteraugli max orders ≥ 99.8 % of same-codec
  AIC-3/AIC-4 pairs correctly, and forced-choice same-codec accuracy is identical for all three scorers. The same-codec "tie"
  there cannot tell a model that understands a ladder from one that only orders it monotonically.
- **Descriptive: where the cross-codec errors are.** The deficits sit mostly in small-gap pairs (< 1 JND, or at most the median MOS gap): e.g. C on CID22-A −0.0157 small vs −0.0022 large, B on AIC-3 −0.067 vs −0.012.
  Large-gap pairs are ≥ 0.989 for the best peer on every JND unit. Pairs that involve JPEG dominate: C on CID22-A loses on JPEG vs
  WebP −0.025, AVIF −0.019, HEIC −0.017 and JXL −0.011. On AIC-4 crop, vs CVVDP-fhd, it loses on JPEG-1 vs JPEG-2000 −0.14,
  JPEG-2000 vs JPEG-XL −0.094, JPEG-1 vs VVC −0.086 and AVIF vs JPEG-1 −0.040. C is ahead of CVVDP-fhd on all five JPEG-AI pairs (CI > 0 on four).
- **Exploratory: the JPEG error has opposite signs.** On CID22-A (mozjpeg at web qualities) C over-rates JPEG (484 of 510
  wrong-where-SSIMULACRA2-right pairs). On AIC-3/AIC-4 (JPEG-1 near threshold) every zensim model under-rates JPEG-1
  (C: 85 of 94 on AIC-3 and 162 of 162 on AIC-4 crop). One JPEG offset cannot fix both. Cross-codec supervision needs pairs at
  matched quality in both regimes.
- **Where peers win:** SSIMULACRA2 leads cross pairs on CID22-A (0.9115) and CSIQ (0.9479). CVVDP-fhd leads AIC-4 crop (0.9136
  vs C 0.8936). IW-SSIM leads AIC-3 (0.9400, C 0.9397 ties). butteraugli 3-norm leads TID (TRAIN). zensim was trained on
  SSIMULACRA2 labels, so ties with SSIMULACRA2 are partly distillation.
""")
    w("## Decision inputs: Δx (cross), Δs (same), DiD = Δx − Δs\n")
    w("| unit | model | best peer cross / same | n pairs cross / same (refs) | Δx [CI] | Δs [CI] | DiD [CI] |")
    w("|---|---|---|---|---|---|---|")
    for u, r in U.items():
        cx, cs = r["classes"]["cross"], r["classes"]["same"]
        n = lambda c: c.get("n_pairs", c.get("n_questions"))
        refs = r.get("n_refs", r.get("n_images"))
        for z, v in r["decision_inputs"].items():
            w(f"| {NAMES[u]} | {z} | {PEERLAB.get(cx['best_peer'], cx['best_peer'])} / {cs['best_peer']} | {n(cx)} / {n(cs)} ({refs}) | "
              f"{ci(v['dx'])} | {ci(v['ds'])} | {ci(v['did'])} |")
    main_end = len(L)
    w("# Rev4 E1b — accuracy per metric and pair class (point [95 % CI])\n")
    w("Companion to `benchmarks/rev4_e1b_crosscodec_2026-09-23.md`. Same run and conventions.\n")
    for u, r in U.items():
        cls = list(r["classes"])
        w(f"**{NAMES[u]}** — " + "; ".join(
            f"{k}: n = {r['classes'][k].get('n_pairs', r['classes'][k].get('n_questions'))}"
            + (f", all-metrics-right {r['classes'][k]['frac_all_metrics_right']}" if r['classes'][k].get('frac_all_metrics_right') is not None else "")
            for k in cls) + (f"; exact label ties dropped {r['label_ties_dropped']}" if r.get("label_ties_dropped") else "") + "\n")
        w("| metric | " + " | ".join(cls) + " |")
        w("|---|" + "---|" * len(cls))
        for m in r["classes"][cls[0]]["acc"]:
            w(f"| {PEERLAB.get(m, m)} | " + " | ".join(
                f"{r['classes'][k]['acc'][m]['point']:.4f} [{r['classes'][k]['acc'][m]['ci'][0]:.4f}, {r['classes'][k]['acc'][m]['ci'][1]:.4f}]"
                for k in cls) + " |")
        w("")
    acc_end = len(L)
    w("## Descriptive strata (cross pairs; Δ vs the unit's cross best peer, fixed)\n")
    for u, r in U.items():
        if u == "cid22a_encdir":
            continue
        s = r["strata_cross"]
        bp = r["classes"]["cross"]["best_peer"]
        zs = [z for z in ("B", "C") if z in r["zensim"]]
        w(f"**{NAMES[u]}** — peer {PEERLAB.get(bp, bp)}; gap threshold {s['_gap_threshold']}\n")
        w("| stratum | n | peer acc | " + " | ".join(f"{z} Δ [CI]" for z in zs) + " |")
        w("|---|---|---|" + "---|" * len(zs))
        for k, v in s.items():
            if k.startswith("_"):
                continue
            w(f"| {k} | {v.get('n_pairs', v.get('n_questions'))} | {v['acc'][bp]:.4f} | " + " | ".join(ci(v["d_best"][z]) for z in zs) + " |")
        w("")
    strata_end = len(L)
    w("# Rev4 E1b — error tables: pairs each metric orders wrong where the best cross peer orders right\n")
    w("Companion to `benchmarks/rev4_e1b_crosscodec_2026-09-23.md`. Raw material for cross-codec supervision design.\n")
    w("Cells are `wrong-where-peer-right / right-where-peer-wrong` (metric ties in parentheses when non-zero), per codec pair. "
      "Same-codec rows are `X (same)`. Forced choice: per-question strict majority, split questions excluded. "
      "All zensim models and peers per cell are in the JSON.\n")
    for u, r in U.items():
        if u == "cid22a_encdir" or "error_tables" not in r:
            continue
        et = r["error_tables"]
        zs = [z for z in ("B", "C", "D", "R915_fast", "R915_rich", "V0_2") if z in r["zensim"]]
        w(f"**{NAMES[u]}** — peer {PEERLAB.get(et['peer'], et['peer'])}\n")
        w("| class | codec pair | n | " + " | ".join(zs) + " |")
        w("|---|---|---|" + "---|" * len(zs))
        for k, byc in et["by_class"].items():
            for cp, mm in sorted(byc.items()):
                cell = lambda x: f"{x[1]}/{x[2]}" + (f" ({x[3]})" if x[3] else "")
                w(f"| {k} | {cp} | {mm[zs[0]][0]} | " + " | ".join(cell(mm[z]) for z in zs) + " |")
        w("")
    err_end = len(L)
    w("## EXPLORATORY (not preregistered): which side does zensim over-rate in JPEG-vs-other errors?\n")
    w(f"Source `benchmarks/rev4_e1b_2026-09-23/direction.py` → `{DIR}` (sha256 `{sha(DIR)[:16]}…`). Among cross pairs involving JPEG that the "
      "model orders wrong while the unit's cross best peer orders right: `JPEG over-rated` = humans judged the JPEG stimulus worse; "
      "`JPEG under-rated` = humans judged the other codec worse.\n")
    w("| unit | peer | " + " | ".join(MODELS) + " |")
    w("|---|---|" + "---|" * len(MODELS))
    for u, v in dr.items():
        w(f"| {NAMES[u]} | {v['peer']} | " + " | ".join(
            (f"{v['by_model'][z]['jpeg_overrated']} over / {v['by_model'][z]['jpeg_underrated']} under" if z in v["by_model"] else "—") for z in MODELS) + " |")
    w("")
    main = L[:main_end] + L[acc_end:strata_end] + L[err_end:]
    open(OUT + ".md", "w").write("\n".join(main) + "\n")
    open(OUT + "_acc.md", "w").write("\n".join(L[main_end:acc_end]) + "\n")
    open(OUT + "_errors.md", "w").write("\n".join(L[strata_end:err_end]) + "\n")

    # compact json
    cj = {"prereg": "benchmarks/rev4_e1b_prereg_2026-09-23.md", "full_run": FULL, "full_run_sha256": sha(FULL),
          "direction": DIR, "direction_sha256": sha(DIR), "B": d["B"], "seed": d["seed"],
          "panel_bin_sha256": "f11857c27563bfc0ca242d07ef52ba78526b9545d02a3ab9f0754e4a34d05688",
          "models": MODELS, "decisions": d["decisions"], "units": {}}
    strata = {"note": "cross pairs only; Δ vs the unit's cross best peer (fixed); [point, lo, hi]", "full_run_sha256": sha(FULL)}
    errs = {"note": "zensim models only; peers' tallies are in the full run", "full_run_sha256": sha(FULL)}
    for u, r in U.items():
        cj["units"][u] = {
            "name": NAMES[u], "n_refs": r.get("n_refs", r.get("n_images")),
            "classes": {k: {kk: vv for kk, vv in c.items() if kk in ("n_pairs", "n_questions", "n_refs", "n_images", "best_peer", "frac_all_metrics_right")}
                        | {"acc": {m: [a["point"]] + a["ci"] for m, a in c["acc"].items()}} for k, c in r["classes"].items()},
            "decision_inputs": r["decision_inputs"],
        }
        if u != "cid22a_encdir":
            strata.setdefault("units", {})[u] = {k: ({"n": v.get("n_pairs", v.get("n_questions")), "d_best": v["d_best"]} if isinstance(v, dict) else v)
                                                 for k, v in r["strata_cross"].items()}
        if "error_tables" in r and u != "cid22a_encdir":
            et = r["error_tables"]
            zs = set(r["zensim"])
            errs.setdefault("units", {})[u] = {"peer": et["peer"], "cols": "[n, wrong_peer_right, right_peer_wrong, tie]",
                                              "by_class": {k: {cp: {m: x for m, x in mm.items() if m in zs} for cp, mm in byc.items()}
                                                           for k, byc in et["by_class"].items()}}
    cj["direction_exploratory"] = dr
    json.dump(cj, open(OUT + ".json", "w"), separators=(",", ":"))
    json.dump(errs, open(OUT + "_errors.json", "w"), separators=(",", ":"))
    json.dump(strata, open(OUT + "_strata.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
