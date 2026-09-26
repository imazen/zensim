"""Refresh the early reviewed P2 section without opening any target file."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from p2_data import ROOT, sha


REPORT = (Path.home() / "tmp/zensim-paper/rev4/FEATBANK_POTENTIAL_DONE.md")
MANIFEST = (Path.home() / "tmp/devin/rev4_featbank-potential_manifest.tsv")
START, END = "<!-- BEGIN P2 -->", "<!-- END P2 -->"
D1 = ("kadid_train", "tid2013", "konfig_train", "cid22_a25", "aic3",
      "kadid_select", "konfig_val")


def fmt(value):
    return f"{value:+.4f}"


def main():
    source = ROOT / "p2/standalone.json"
    if not source.is_file():
        raise FileNotFoundError("standalone GMSD receipt is not complete")
    standalone = json.loads(source.read_text())
    if standalone["schema"] != "rev4-featpot-p2-standalone-v1" or len(standalone["rows"]) != 9:
        raise ValueError("standalone receipt is incomplete")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [START, "## P2 — reviewed exact GMSD/GMSM peer control",
             "", f"Status at {now}: **POTENTIAL — ceiling, not a model score.** ",
             "`REVIEW_GMSBANK.md` says **PROMOTE WITH CORRECTIONS** and independently verifies the peer columns. "
             "The P2 adapter pins the 18-set peer manifest, per-set Parquet and bank-key hashes, joins only admitted "
             "D1/D2 rows by `pair_key`, and preserves collapsed stimulus multiplicity. Peer binary SHA-256 "
             "`2ed8f6765e2f9a7bca5a4ea4a3b8b7b266fb10c57b5b0c9f5563f978c2108ea6`; "
             f"standalone receipt `{source}` SHA-256 `{sha(source)}`. The peer manifest's root-bank hash is stale "
             "after the bank metadata rewrite; the reviewed per-set key hashes still match the pinned bank snapshot.",
             "", "Standalone GMSD is sign-oriented as **−gmsd** against quality labels; GMSM uses **+gmsm**. "
             "KonJND-BPG rows measure agreement with an SSIMULACRA2 oracle (/100), not humans; "
             "CID22-A is human MCOS/100. No raw units are pooled.",
             "", "| Set | Target | Rows | Refs | Standalone −GMSD SROCC [95% reference CI] | +GMSM SROCC [95% reference CI] |",
             "|---|---|---:|---:|---:|---:|"]
    for name, row in standalone["rows"].items():
        g = row["measurements"]["gmsd"]
        m = row["measurements"]["gmsm"]
        ci = g["ci95_srocc"]
        m_ci = m["ci95_srocc"]
        target = "SSIM2 oracle /100" if row["target"] == "ssim2_oracle" else ("human MCOS/100" if name == "cid22_a25" else "human quality")
        lines.append(f"| `{name}` | {target} | {row['rows']} | {row['references']} | "
                     f"{g['score']['srocc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
                     f"{m['score']['srocc']:.4f} [{m_ci[0]:.4f}, {m_ci[1]:.4f}] |")
    lines += ["", "P2 adds the raw two peer columns to P0; its matched control jointly permutes them by unique "
              "`pair_key` within each reference. The D5 bar is stability frequency ≥0.6, nested-CV gain ≥+0.005 "
              "SROCC with paired 95% CI excluding zero on at least two **human** sets, "
              "and no non-negative-head dial regression (the cost clause was removed by amendment 61d3c07a050d). A positive control must establish instrument sensitivity "
              "and the permuted control's incremental CI must include zero. These conditions are evaluated per "
              "dataset and model; the BPG oracle cannot count toward the human-set bar. The completed P0 "
              "basic-ablation positive control is sensitive only on KADID TRAIN BVLS/lasso and KADID SELECT lasso; "
              "other set/model cells cannot support a family conclusion under the registered instrument rule."]
    lines += ["", "Cost is measured and reported, never a gate (amendment `61d3c07a050d`); P2 is a diagnostic "
              "control and no cost number is claimed for the separate exact peer scorer."]
    lines += ["", "The preregistered interpretation table is conditional on the paired controls, "
              "CIs and every D5 gate. P1 and P3 (Part B sidecars now landed) are reported in the arms summary, not here:", "",
              "| Pattern after all gates | Interpretation |",
              "|---|---|",
              "| P1 ≈ P3 > P0 | The bank captured GMSD's useful information. |",
              "| P2 > P1 | The gamma domain or gradient operator matters. |",
              "| Neither P1 nor P2 > P0 | No human-label gain. |"]
    found = []
    for model in ("bvls", "linear"):
        for name in D1:
            path = ROOT / "p2/d1" / f"POT_{name}_{model}_compare.json"
            if path.is_file():
                value = json.loads(path.read_text())
                phase = value["phases"]["nested"]
                found.append((name, model, phase, path))
    if found:
        lines += ["", "### D1 nested-CV P2 comparisons", "",
                  "| Set | Model | Rows / refs | P0 SROCC | P2 SROCC | P2−P0 [paired 95% CI] | Perm−P0 [paired 95% CI] |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, model, phase, _ in found:
            p2, perm = phase["deltas"]["p2"], phase["deltas"]["perm"]
            lines.append(f"| `{name}` | {model} | {phase['rows']} / {phase['references']} | "
                         f"{phase['point_srocc']['p0']:.4f} | "
                         f"{phase['point_srocc']['p2']:.4f} | {fmt(p2['point_delta_srocc'])} "
                         f"[{fmt(p2['ci95'][0])}, {fmt(p2['ci95'][1])}] | "
                         f"{fmt(perm['point_delta_srocc'])} "
                         f"[{fmt(perm['ci95'][0])}, {fmt(perm['ci95'][1])}] |")
        lines.append("Paired bootstrap receipts: " + ", ".join(f"`{p}` SHA-256 `{sha(p)}`" for _, _, _, p in found) + ".")
    bvls_coefficients = []
    for name in D1 + ("konjnd_bpg_train",):
        path = ROOT / "p2/d1" / f"POT_{name}_p2_bvls" / "result.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if (value["set"], value["arm"], value["model"]) != (name, "p2", "bvls") or len(value["outer"]) != 5:
                raise ValueError(f"{path}: P2 BVLS coefficient identity mismatch")
            full = value["in_sample"]["peer_coefficients"]
            outer = [fold["peer_coefficients"] for fold in value["outer"]]
            if len(full) != 2 or any(len(pair) != 2 for pair in outer):
                raise ValueError(f"{path}: missing peer BVLS coefficient")
            bvls_coefficients.append((name, full, outer, path))
    if bvls_coefficients:
        lines += ["", "### P2 BVLS peer coefficients", "",
                  "Coefficients are the fitted standardized-feature weights for raw f944 `gmsd` "
                  "and f945 `gmsm`; both were explicitly free in the registered sign mask. "
                  "These descriptive weights do not establish a P2 gain.", "",
                  "| Set | Full gmsd / gmsm | Five outer-fold gmsd / gmsm pairs |",
                  "|---|---:|---|"]
        for name, full, outer, _ in bvls_coefficients:
            pairs = ", ".join(f"{g:+.4g}/{m:+.4g}" for g, m in outer)
            lines.append(f"| `{name}` | {full[0]:+.4g}/{full[1]:+.4g} | {pairs} |")
        lines.append("Coefficient sources: " + ", ".join(
            f"`{path}` SHA-256 `{sha(path)}`" for _, _, _, path in bvls_coefficients) + ".")
    for model in ("bvls", "linear"):
        path = ROOT / "p2/d2" / f"LODO_{model}_compare.json"
        if path.is_file():
            value = json.loads(path.read_text())
            lines += ["", f"### D2 seven-fold P2 transfer — {model}", "",
                      "| Held out | Evaluation set | Rows / refs | P0 SROCC | P2 SROCC | P2−P0 [paired 95% CI] | Perm−P0 [paired 95% CI] |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for heldout, row in value["folds"].items():
                p2, perm = row["deltas"]["p2"], row["deltas"]["perm"]
                lines.append(f"| `{heldout}` | `{row['set']}` | {row['rows']} / {row['references']} | "
                             f"{row['point_srocc']['p0']:.4f} | "
                             f"{row['point_srocc']['p2']:.4f} | {fmt(p2['point_delta_srocc'])} "
                             f"[{fmt(p2['ci95'][0])}, {fmt(p2['ci95'][1])}] | "
                             f"{fmt(perm['point_delta_srocc'])} "
                             f"[{fmt(perm['ci95'][0])}, {fmt(perm['ci95'][1])}] |")
            lines.append(f"Receipt `{path}` SHA-256 `{sha(path)}`.")
            if model == "bvls":
                fit_path = ROOT / "p2/d2/LODO_p2_bvls/result.json"
                fit = json.loads(fit_path.read_text())
                lines += ["", "P2 BVLS D2 standardized peer weights by held-out source:", "",
                          "| Held out | gmsd | gmsm |",
                          "|---|---:|---:|"]
                for heldout, fold in fit["folds"].items():
                    g, m = fold["peer_coefficients"]
                    lines.append(f"| `{heldout}` | {g:+.4g} | {m:+.4g} |")
                lines.append(f"Coefficient source `{fit_path}` SHA-256 `{sha(fit_path)}`.")
    mlp = []
    for name in D1:
        for hidden in (32, 128):
            path = ROOT / "p2/mlp_compare" / f"POT_{name}_mlp{hidden}.json"
            if path.is_file():
                mlp.append((name, hidden, json.loads(path.read_text()), path))
    if mlp:
        lines += ["", "### D1 five-seed H32/H128 P2 comparisons", "",
                  "The 946-column P2 MLP tables are explicitly admitted as unqualified diagnostic "
                  "composites under a recorded historical-replay reason; no bake is shipped.", "",
                  "| Set | Head | Rows / refs | P0 mean SROCC | P2 mean SROCC | P2−P0 [paired 95% CI] | Perm−P0 [paired 95% CI] |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, hidden, value, _ in mlp:
            p2, perm = value["deltas"]["p2"], value["deltas"]["perm"]
            detail = json.loads((ROOT / "p2/mlp_aggregate" /
                                 f"POT_{name}_p2_mlp{hidden}" / "result.json").read_text())
            lines.append(f"| `{name}` | H{hidden} | {detail['rows']} / {detail['references']} | "
                         f"{value['points']['p0']:.4f} | "
                         f"{value['points']['p2']:.4f} | {fmt(p2['point_delta_srocc'])} "
                         f"[{fmt(p2['ci95'][0])}, {fmt(p2['ci95'][1])}] | "
                         f"{fmt(perm['point_delta_srocc'])} "
                         f"[{fmt(perm['ci95'][0])}, {fmt(perm['ci95'][1])}] |")
        lines.append("Paired five-seed receipts: " + ", ".join(f"`{p}` SHA-256 `{sha(p)}`" for _, _, _, p in mlp) + ".")
        lines += ["", "D1 seed detail (pooled out-of-fold SROCC in fixed replicate order; "
                  "spreads are balanced outer-fold means):", "",
                  "| Set | Head | Arm | Rows / refs | Five nested seeds | Nested min–max | Five full seeds | Full min–max | In-sample mean | Gap | Init spread | Sample spread |",
                  "|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|"]
        for name, hidden, _, _ in mlp:
            for arm, root in (("P0", ROOT / "mlp_aggregate"),
                              ("P2", ROOT / "p2/mlp_aggregate"),
                              ("P2 perm", ROOT / "p2/mlp_aggregate")):
                arm_name = {"P0": "r0", "P2": "p2", "P2 perm": "p2_perm"}[arm]
                path = root / f"POT_{name}_{arm_name}_mlp{hidden}" / "result.json"
                aggregate = json.loads(path.read_text())
                seeds = aggregate["replicates"]
                if len(seeds) != 5 or aggregate["B"] != 2000:
                    raise ValueError(f"{path}: incomplete P2 seed detail")
                seed_text = ", ".join(f"{row['nested_srocc']:.4f}" for row in seeds)
                full_text = ", ".join(f"{row['in_sample_srocc']:.4f}" for row in seeds)
                lines.append(f"| `{name}` | H{hidden} | {arm} | "
                             f"{aggregate['rows']} / {aggregate['references']} | {seed_text} | "
                             f"{aggregate['nested_min_srocc']:.4f}–{aggregate['nested_max_srocc']:.4f} | "
                             f"{full_text} | "
                             f"{aggregate['in_sample_min_srocc']:.4f}–{aggregate['in_sample_max_srocc']:.4f} | "
                             f"{aggregate['in_sample_mean_srocc']:.4f} | "
                             f"{aggregate['gap_mean_srocc']:.4f} | "
                             f"{aggregate['init_spread_fold_mean_srocc']:.4f} | "
                             f"{aggregate['sample_spread_fold_mean_srocc']:.4f} |")
    stability = []
    for name in D1 + ("konjnd_bpg_train",):
        for arm in ("p2", "p2_perm"):
            path = ROOT / "p2/stability" / f"POT_{name}_{arm}_lasso" / "result.json"
            if path.is_file():
                value = json.loads(path.read_text())
                stability.append((name, arm, value["family_frequency"]["peer_gmsd_gmsm"], path))
    if stability:
        lines += ["", "### Peer-column lasso stability", "",
                  "Frequency is the fraction of 200 half-reference draws selecting either peer column "
                  "at the frozen full-inner 1-SE lambda.", "",
                  "| Set | Arm | Peer-family frequency |",
                  "|---|---|---:|"]
        for name, arm, freq, _ in stability:
            lines.append(f"| `{name}` | {arm} | {freq:.3f} |")
    d2_mlp = []
    for heldout in ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
                    "cid22_a25", "aic3", "kadid_select"):
        for hidden in (32, 128):
            path = ROOT / "p2/d2_mlp_compare" / f"LODO_{heldout}_mlp{hidden}.json"
            if path.is_file():
                d2_mlp.append((heldout, hidden, json.loads(path.read_text()), path))
    if d2_mlp:
        lines += ["", "### D2 five-seed H32/H128 transfer", "",
                  "Each model fits the other six D2 sources with equal draw share; KonFiG and "
                  "BPG use their reference-disjoint VAL views for evaluation. BPG remains an oracle fold.", "",
                  "| Held out | Eval | Head | Rows / refs | P0 mean SROCC | P2 mean SROCC | P2−P0 [paired 95% CI] | Perm−P0 [paired 95% CI] |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for heldout, hidden, value, _ in d2_mlp:
            p2, perm = value["deltas"]["p2"], value["deltas"]["p2_perm"]
            lines.append(f"| `{heldout}` | `{value['eval_set']}` | H{hidden} | "
                         f"{value['rows']} / {value['references']} | "
                         f"{value['points_mean_srocc']['r0']:.4f} | "
                         f"{value['points_mean_srocc']['p2']:.4f} | "
                         f"{fmt(p2['point_delta_srocc'])} [{fmt(p2['ci95'][0])}, {fmt(p2['ci95'][1])}] | "
                         f"{fmt(perm['point_delta_srocc'])} "
                         f"[{fmt(perm['ci95'][0])}, {fmt(perm['ci95'][1])}] |")
        lines.append("D2 MLP receipts: " + ", ".join(f"`{p}` SHA-256 `{sha(p)}`" for _, _, _, p in d2_mlp) + ".")
        lines += ["", "D2 seed detail (held-out SROCC in fixed replicate order):", "",
                  "| Held out | Head | Arm | Rows / refs | Five seeds | Min | Max | Mean [95% reference CI] |",
                  "|---|---:|---|---:|---|---:|---:|---:|"]
        for heldout, hidden, value, _ in d2_mlp:
            for arm, label in (("r0", "P0"), ("p2", "P2"), ("p2_perm", "P2 perm")):
                detail = value["arms"][arm]
                seeds = detail["seed_srocc"]
                if len(seeds) != 5:
                    raise ValueError(f"{heldout}/H{hidden}/{arm}: incomplete D2 seeds")
                ci = detail["ci95_mean_srocc"]
                lines.append(f"| `{heldout}` | H{hidden} | {label} | "
                             f"{value['rows']} / {value['references']} | "
                             f"{', '.join(f'{score:.4f}' for score in seeds)} | "
                             f"{detail['seed_min_srocc']:.4f} | "
                             f"{detail['seed_max_srocc']:.4f} | "
                             f"{value['points_mean_srocc'][arm]:.4f} "
                             f"[{ci[0]:.4f}, {ci[1]:.4f}] |")
    missing_d1 = 14 - len(found)
    missing_d2 = sum(not (ROOT / "p2/d2" / f"LODO_{model}_compare.json").is_file()
                     for model in ("bvls", "linear"))
    outer_importance = sum((ROOT / "p2/mlp" / f"POT_{name}_{arm}_mlp{hidden}" /
                            f"o{outer}_r{rep}" / "importance.json").is_file()
                           for name in D1 + ("konjnd_bpg_train",)
                           for arm in ("p2", "p2_perm") for hidden in (32, 128)
                           for outer in range(5) for rep in range(5))
    lines += ["", f"**MISSING for P2:** {missing_d1}/14 D1 paired linear/BVLS comparison cells, "
              f"{missing_d2}/2 D2 seven-fold paired comparison panels, {14-len(mlp)}/14 H32/H128 "
              f"five-seed paired D1 MLP comparisons, {16-len(stability)}/16 stability receipts, "
              f"{800-outer_importance}/800 outer-importance receipts, "
              f"{14-len(d2_mlp)}/14 H32/H128 D2 MLP comparison panels, "
              "and the P2 dial-contract gate (no cost gate: cost is reported, never a gate). "
              "No D5 or P1/P2/P3 interpretation verdict is earned yet.", END]
    section = "\n".join(lines)
    original = REPORT.read_text()
    if START in original and END in original:
        before = original[:original.index(START)].rstrip()
        after = original[original.index(END) + len(END):].lstrip("\n")
        original = before + "\n\n" + after
    title, remainder = original.split("\n", 1)
    title = title.replace("BASELINE RUNNING / CANDIDATE BLOCKED",
                          "BASELINE RUNNING / P2 RUNNING; PART B CANDIDATES WAITING")
    title = title.replace("CORE BASELINE GATES COMPLETE / CANDIDATES BLOCKED",
                          "CORE BASELINE GATES COMPLETE / P2 RUNNING; PART B CANDIDATES WAITING")
    outcome = re.search(r"(?m)^## Outcome(?: to date)?\b", remainder)
    if outcome is None:
        raise ValueError("report has no outcome heading after its MISSING section")
    revised = (title + "\n" + remainder[:outcome.start()] + section + "\n\n" +
               remainder[outcome.start():])
    revised = re.sub(r"- Candidate C1–C4, all-C, P1/P2/P3 GMSD arms[^\n]*", 
                     "- Candidate C1–C4, all-C and P1/P3 still await reviewed Part B sidecars. P2 status is in the early section above.", revised)
    revised = re.sub(r"- C1–C4, all-C, GMSD P1/P2/P3[^\n]*", 
                     "- C1–C4, all-C and GMSD P1/P3 await reviewed Part B sidecars. P2 status is in the early section above.", revised)
    temporary = REPORT.with_suffix(".md.new")
    temporary.write_text(revised)
    temporary.replace(REPORT)
    with MANIFEST.open("a") as stream:
        stream.write(f"{now}\tmodified\t{REPORT}\n")
    print(json.dumps({"output": str(REPORT), "sha256": sha(REPORT),
                      "standalone_sha256": sha(source), "d1_compares": len(found),
                      "d2_compares": 2 - missing_d2}))


if __name__ == "__main__":
    main()
