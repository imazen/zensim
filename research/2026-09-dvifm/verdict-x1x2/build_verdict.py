#!/usr/bin/env python3
"""Assemble the verdict record: X2 arm comparison + X1 peer tables +
paired bootstrap + CID22-B sealed read into the two benchmark files.

  build_verdict.py <outdir> <repo_benchmarks_dir>

Inputs it expects under <outdir>:
  x2/x2_results.json, x2_decision.json
  scores/<peer>__on__<leg>.csv  (peers + dvifm winner per leg)
  scores/cid22b manifest
Writes:
  <repo>/benchmarks/dvifm_verdict_2026-09-20.{md,json}
"""
import csv
import glob
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

S2D = "/mnt/v/output/zensim/dvifm-screen2d-2026-09-19"
METRICS = f"{S2D}/tools/metrics.py"
DEV_LEGS = ["safesyn_dev", "cid22_dev", "codec_dev", "human_dev"]
EVAL_LEGS = DEV_LEGS + ["kadid135", "konfig_val", "kadid_dev_full"]
PEERS = ["fastssim2", "bake_prof_b", "bake_prof_d",
         "bake_r915_basic228", "bake_r915_y60"]


def read_scores(path):
    ref, tgt, sc, e = [], [], [], []
    for r in csv.DictReader(open(path)):
        ref.append(r["ref_path"]); tgt.append(float(r["target"]))
        sc.append(float(r["score"])); e.append(float(r["E"]))
    return np.array(ref), np.asarray(tgt), np.asarray(sc), np.asarray(e)


def metrics(ref, tgt, sc):
    sro = float(scipy.stats.spearmanr(sc, tgt).statistic)
    kro = float(scipy.stats.kendalltau(sc, tgt).statistic)
    try:
        from scipy.optimize import curve_fit
        def f5(x, a, b, c, d, e):
            return a * (0.5 - 1.0 / (1.0 + np.exp(b * (x - c)))) + d * x + e
        p0 = [tgt.max() - tgt.min(), 0.1, np.median(sc),
              1.0, tgt.min()]
        p, _ = curve_fit(f5, sc, tgt, p0=p0, maxfev=20000)
        pl = float(np.corrcoef(f5(sc, *p), tgt)[0, 1])
    except Exception:
        pl = float(np.corrcoef(sc, tgt)[0, 1])
    return {"srocc": sro, "krocc": kro, "plcc": pl}


def ref_boot_delta(base_csv, cand_csv, boots=2000, seed=20260920):
    """Cluster bootstrap over refs of the per-row SROCC delta cand-base."""
    bref, btgt, bsc, _ = read_scores(base_csv)
    cref, ctgt, csc, _ = read_scores(cand_csv)
    assert np.array_equal(bref, cref) and np.array_equal(btgt, ctgt), \
        (base_csv, cand_csv)
    groups = defaultdict(list)
    for i, r in enumerate(bref):
        groups[r].append(i)
    gkeys = np.array(sorted(groups))
    gidx = [np.asarray(groups[k]) for k in gkeys]
    rng = np.random.default_rng(seed)
    deltas = np.empty(boots)
    for b in range(boots):
        sel = np.concatenate([gidx[i]
                              for i in rng.integers(0, len(gidx),
                                                    len(gidx))])
        deltas[b] = (scipy.stats.spearmanr(csc[sel], ctgt[sel]).statistic
                     - scipy.stats.spearmanr(
                         bsc[sel], btgt[sel]).statistic)
    return {"delta_mean": float(deltas.mean()),
            "ci95": [float(np.quantile(deltas, 0.025)),
                     float(np.quantile(deltas, 0.975))],
            "p_delta_le_0": float((deltas <= 0).mean()),
            "n_boots": boots}


def leg_table(scores_dir, leg, methods):
    out = {}
    for m in methods:
        p = scores_dir / f"{m}__on__{leg}.csv"
        if not p.exists():
            out[m] = None
            continue
        ref, tgt, sc, _ = read_scores(p)
        d = metrics(ref, tgt, sc)
        d["n"] = len(tgt)
        d["n_refs"] = len(set(ref))
        out[m] = d
    return out


def main():
    outdir = Path(sys.argv[1])
    repo = Path(sys.argv[2])
    scores = outdir / "scores"
    x2 = json.loads((outdir / "x2/x2_results.json").read_text())
    dec = json.loads((outdir / "x2_decision.json").read_text())
    winner = dec["decision"]["winner"]

    record = {
        "lane": "verdict", "date": "2026-09-20",
        "winner_form": winner,
        "x2": {"seeds": x2["seeds"], "n_rows": x2["n_rows"],
               "decision": dec["decision"],
               "arm_summary": {a: {
                   "composite_mean": dec["arms"][a]["composite_mean"],
                   "composite_std": dec["arms"][a]["composite_std"],
                   "composite_srocc": dec["arms"][a]["composite_srocc"],
                   "fit_mse": dec["arms"][a]["fit_mse"],
                   "fit_srocc": dec["arms"][a]["fit_srocc"],
                   "wall_s": dec["arms"][a]["wall_s"],
                   "legs": dec["arms"][a]["legs"],
               } for a in dec["arms"]},
               "deltas": dec["deltas"]},
        "x1": {}, "cid22b": {},
    }

    methods = [f"dvifm_{winner}"] + PEERS
    # X1 legs: dev legs + kadid135 + konfig_val
    for leg in EVAL_LEGS:
        t = leg_table(scores, leg, methods)
        record["x1"][leg] = {"metrics": t}
        # paired bootstrap: dvifm vs every peer on this leg
        base_tag = f"dvifm_{winner}"
        deltas = {}
        for peer in PEERS:
            bp = scores / f"{peer}__on__{leg}.csv"
            cp = scores / f"{base_tag}__on__{leg}.csv"
            if bp.exists() and cp.exists():
                deltas[f"{base_tag}-vs-{peer}"] = ref_boot_delta(bp, cp)
        record["x1"][leg]["bootstrap"] = deltas

    # CID22-B sealed read (separate section)
    cb = "cid22b"
    if (scores / f"dvifm_{winner}__on__{cb}.csv").exists():
        t = leg_table(scores, cb, methods)
        deltas = {}
        for peer in PEERS:
            bp = scores / f"{peer}__on__{cb}.csv"
            cp = scores / f"dvifm_{winner}__on__{cb}.csv"
            if bp.exists() and cp.exists():
                deltas[f"dvifm_{winner}-vs-{peer}"] = ref_boot_delta(
                    bp, cp)
        man = {}
        mp = outdir / "pairs/cid22b_unsealed.manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text())
        record["cid22b"] = {"metrics": t, "bootstrap": deltas,
                            "unseal_manifest": man,
                            "note": "single registered read; labels joined "
                                    "after all configs frozen. Bake scores "
                                    "in this table are a CORRECTION of that "
                                    "same single read (recomputed on the "
                                    "w944/ceiling_rev3 features-rev3 "
                                    "ext_cid22val table after the lane's "
                                    "w986 research-path table was found "
                                    "era-mismatched to the bakes) — NOT a "
                                    "second holdout exposure"}

    jp = repo / "benchmarks/dvifm_verdict_2026-09-20.json"
    jp.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {jp}")
    return record


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_md(record, path, missing):
    x2 = record["x2"]
    win = record["winner_form"]
    L = []
    L.append("# DVIFM standalone verdict — 2026-09-20\n")
    L.append("Lane: `verdict`. Question: does the standalone three-plane "
             "Y′CbCr DVIFM block-visibility model predict quality "
             "competitively with `fast-ssim2` and the project's own "
             "models on held-out references — and at what parameter "
             "count?\n")
    # ---- plain verdict ----
    L.append("## Verdict\n")
    L.append("**Standalone DVIFM is NOT competitive with `fast-ssim2` on "
             "real held-out labels.** On the five real-label legs the "
             "frozen gate scores SROCC 0.759–0.921 vs `fast-ssim2` "
             "0.735–0.957. Paired ref-bootstrap deltas (dvifm−ssim2): "
             "significantly below on kadid_dev_full (−0.042, "
             "CI95 [−0.070, −0.017]) and the sealed CID22-B (−0.134, "
             "CI95 [−0.185, −0.089]); nominally below on human_dev "
             "(−0.030, n.s.) and kadid135 (−0.031, n.s.); significantly "
             "above only on konfig_val (+0.024, CI95 [+0.007, "
             "+0.038]).\n")
    L.append("Against the project's own bakes the answer splits by label "
             "kind: on ssim2-pseudo-label development legs (safesyn, "
             "cid22_dev, codec_dev — targets are literally ssim2/100, so "
             "the bakes' 0.97–1.00 there is ssim2-target regression, not "
             "human agreement) dvifm_gate is last or near-last; on real-"
             "label legs it is below every bake on every leg with one "
             "exception (above bake_prof_d on konfig_val, +0.035 sig) — "
             "and on the sealed CID22-B read it is significantly below "
             "all of them (dvifm 0.774 vs B 0.890, D 0.879, "
             "R915_basic228 0.897, R915_y60 0.861, fast-ssim2 0.913; "
             "paired Δ −0.08..−0.13, all P(Δ≤0)≈1.0).\n")
    L.append("**For the record — on real-label legs `R915_basic228` beats "
             "BOTH `fast-ssim2` and `dvifm_gate`:** human_dev 0.931 vs "
             "0.817/0.783; kadid_dev_full 0.945 vs 0.944/0.901; konfig_val "
             "0.839 vs 0.735/0.759; and on the sealed CID22-B read 0.897 "
             "vs 0.913/0.774 (second to fast-ssim2 there, above dvifm).\n")
    L.append("Parameter cost: the X2 winner is the **gate** form — "
             "≤15 fitted knees (one per active plane×level) + head, "
             "≈35 constants total; the fully-fitted smooth curve's "
             "+0.0031 composite edge is under the 2σ seed-noise bar. So "
             "DVIFM's standalone ceiling here is reachable at "
             "near-constant cost — but the ceiling itself is below "
             "`fast-ssim2` on real labels. Per the plan's kill "
             "criterion, X1's first condition is met; any residual value "
             "must come from the X4/X5 transplant and X6 steering lanes "
             "(outside this lane's scope).\n")
    L.append("## MISSING / not done\n")
    L += [f"- {m}" for m in missing] or ["- (none recorded)"]
    L.append("\n## X2 — constant-form comparison\n")
    L.append("Three arms on identical fit rows per seed "
             f"({x2['n_rows']} seeded-uniform rows of the 50,463-row "
             "joint-core-v1 SDR fit domain, seeds "
             f"{x2['seeds']}):\n")
    L.append("- **curve** — fully fitted smooth visibility "
             "v(c)=exp(−softplus(βς(ln c−ln c0))/ς) per (plane,level): "
             "5 constants + head per cell (≈95 constants total).")
    L.append("- **gate** — two-state v(c)=[c≤κ] or uniform (off), "
             "g=P=1: ≤15 knees + head (≤35 constants; "
             "one integer compare per block in a shipped kernel).")
    L.append("- **prior** — no loss-fitted constants: g=1, P=1, β=0.65, "
             "ς=4, c0=TRAIN 10th percentile per cell + head (≤35).")
    rows = []
    for a in ("curve", "gate", "prior"):
        s = x2["arm_summary"].get(a)
        if not s:
            continue
        rows.append([a, fmt(s["composite_mean"]),
                     fmt(s["composite_std"]),
                     fmt(np.mean(s["fit_mse"])),
                     fmt(np.mean(s["fit_srocc"])),
                     fmt(np.mean(s["wall_s"]), 1)])
    L.append("\n" + md_table(
        rows, ["arm", "dev composite SROCC (mean/5 seeds)", "seed std",
               "fit MSE", "fit SROCC", "fit wall s"]))
    L.append("\n\nPer-leg SROCC (mean over seeds):\n")
    legs = list(next(iter(x2["arm_summary"].values()))["legs"].keys())
    rows = []
    for l in legs:
        rows.append([l] + [
            fmt(x2["arm_summary"][a]["legs"][l]["srocc_mean"])
            if a in x2["arm_summary"] and
            l in x2["arm_summary"][a]["legs"] else "—"
            for a in ("curve", "gate", "prior")])
    L.append(md_table(rows, ["leg", "curve", "gate", "prior"]))
    d = x2["deltas"]
    L.append("\n\nPaired-seed deltas (composite):\n")
    L.append(md_table(
        [[k, fmt(v["delta_mean"]), fmt(v["delta_std"]),
          "[" + ", ".join(fmt(x) for x in v["delta_per_seed"]) + "]"]
         for k, v in d.items()],
        ["delta", "mean", "std", "per-seed"]))
    dec = record["x2"]["decision"]
    L.append(f"\n\n**Decision: `{win}`** — {dec['rule']}. Measured: "
             f"gate composite {fmt(dec['gate_composite'])} ± "
             f"{fmt(dec['gate_std'])}, curve−gate "
             f"{fmt(dec['curve_minus_gate'])}.\n")
    L.append("\n## X1 — frozen winner vs peers\n")
    L.append("Frozen artefact: winning form refit on the union of the "
             "five seed subsets (≤5,120 rows), identical machinery. "
             "Peers: `fast-ssim2` (extractor audit channel, same decoded "
             "RGB8 buffers), zensim B/D (`*_byid_2026-09-06` bakes via "
             "`ensemble_score_rows`), both frozen Rev3 ensembles "
             "(R915_basic228_h128_ens5, R915_y60_h32_ens5).\n")
    for leg, blk in record["x1"].items():
        m = blk["metrics"]
        rows = [[t,
                 fmt(v["srocc"]) if v else "—",
                 fmt(v["krocc"]) if v else "—",
                 fmt(v["plcc"]) if v else "—",
                 v["n"] if v else "—"]
                for t, v in m.items()]
        L.append(f"\n### {leg}\n")
        L.append(md_table(rows, ["method", "SROCC", "KROCC", "PLCC", "n"]))
        if blk.get("bootstrap"):
            rows = [[k, fmt(v["delta_mean"]),
                     f"[{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]",
                     fmt(v["p_delta_le_0"])]
                    for k, v in blk["bootstrap"].items()]
            L.append("\n" + md_table(
                rows, ["paired Δ SROCC (cand−base, ref bootstrap)",
                       "mean", "CI95", "P(Δ≤0)"]))
    L.append("\n## CID22-B — the single sealed read\n")
    cb = record.get("cid22b", {})
    if cb.get("metrics"):
        man = cb.get("unseal_manifest", {})
        L.append(f"Unsealed {man.get('unsealed_utc','?')} "
                 f"({man.get('rows','?')} rows, source sha "
                 f"{str(man.get('source_sha256',''))[:16]}…). Scored "
                 "once, post-freeze, no iteration. **Bake columns below "
                 "are the corrected re-issue of that same single read** "
                 "(era-matched w944/`ceiling_rev3` feature tables; see "
                 "Provenance) — not a second exposure.\n")
        m = cb["metrics"]
        rows = [[t, fmt(v["srocc"]) if v else "—",
                 fmt(v["krocc"]) if v else "—",
                 fmt(v["plcc"]) if v else "—",
                 v["n"] if v else "—"] for t, v in m.items()]
        L.append(md_table(rows, ["method", "SROCC", "KROCC", "PLCC", "n"]))
        if cb.get("bootstrap"):
            rows = [[k, fmt(v["delta_mean"]),
                     f"[{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]",
                     fmt(v["p_delta_le_0"])]
                    for k, v in cb["bootstrap"].items()]
            L.append("\n" + md_table(
                rows, ["paired Δ SROCC", "mean", "CI95", "P(Δ≤0)"]))
    else:
        L.append("NOT PERFORMED — see MISSING.\n")
    L.append("\n## Provenance & limitations\n")
    L.append("- Fit domain: joint-core-v1 SDR pairs "
             "(50,463 rows, TRAIN role). The core's permuted-column "
             "feature-screen gate FAILED (~+0.0021 cost vs ~0.003 seed "
             "noise): the core is admissible for MODEL-LEVEL comparisons "
             "(this lane's use) but not for feature screens — recorded "
             "per `benchmarks/joint_core_v1_2026-09-20.md`.")
    L.append("- Cache formats: all scored legs use f16 capped records "
             "(safesyn cap=512/row, all others 1024/row; deterministic "
             "strides, formula rev 3). The 2d uncapped f32 caches were "
             "removed mid-lane by the superseded-cache cleanup, so "
             "kadid/konfig/cid22b were re-extracted into this lane's "
             "cache at cap=1024 — same spec, same extractor build.")
    L.append("- Extractor: `extract_features_372col` sha256 "
             "9c0d5ac453c9d4c7… (prebuilt 2026-09-20 01:29, formula "
             "rev 3, ZENSIM_SAMPLE_DIGEST=1).")
    L.append("- fast-ssim2 enters via the audit channel on identical "
             "decoded RGB8 buffers (not a separate pixel path).")
    L.append("- Dev-leg targets: safesyn_dev, cid22_dev and codec_dev "
             "carry SIGNED fast-ssim2/100 pseudo-labels (verified: label "
             "== audit peer_ssim2.score ÷ 100 to full precision), so "
             "fastssim2 = 1.0000 there is circular by construction and "
             "the bakes' 0.97–0.99 measure ssim2-target regression. "
             "Real human labels: human_dev, kadid*, konfig_val, and the "
             "sealed cid22b.")
    L.append("- KADID dev restricted to refs {I01,I03,I05} per the "
             "lane prompt; konfig is the originsplit validation split "
             "(4 source groups).")
    L.append("- CID22-B bake values were initially corrupted by a "
             "feature-table era mismatch and have been CORRECTED: the "
             "lane's re-extracted peer tables were w986 research-path "
             "(f0..f985) while the bakes consume w944/`ceiling_rev3` "
             "(f0..f943). Supervisor-flagged; diagnosed by scoring B on "
             "the same rows through the pixel path "
             "(`score_pair_with_bake`/`BakeScorer::compute`, SROCC 0.897 "
             "vs labels on a 41-row sample) vs the w986 table path "
             "(0.56), corr(pixel,table) 0.49. Corrected scores use the "
             "historical `rev3-public-human-eval-2026-09-14/"
             "features-rev3` parquets (`w944/ceiling_rev3#b782e349`, "
             "producer surface `BakeScorer::compute`) — verified "
             "pixel≡table (corr 1.0) and joined row-for-row: cid22b "
             "2100/2100 via eval pairs.tsv row_id→(ref,dist), konfig "
             "436/436 positional label-identical, kadid 250/250 via "
             "(ref,type,level) canonical order with exact label match. "
             "Struck w986-era CSVs kept alongside as `*.w986era`. Same "
             "single registered read recomputed — no new holdout "
             "exposure.")
    L.append("- No holdouts read except the single registered CID22-B "
             "read (see its section).")
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    rec = main()
    missing = json.loads(Path(sys.argv[3]).read_text())["missing"] \
        if len(sys.argv) > 3 else ["(record generated before MISSING "
                                   "list was finalised)"]
    write_md(rec, Path(sys.argv[2]) /
             "benchmarks/dvifm_verdict_2026-09-20.md", missing)
    print("wrote md")
