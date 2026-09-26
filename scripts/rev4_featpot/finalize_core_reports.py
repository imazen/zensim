"""Publish verified P0 core-gate receipts as incomplete Rev4 lane reports.

This runs only after the registered fit, CI, importance, and pooled MLP jobs.
It reads saved receipts, never label files, and leaves LODO/peer/candidate
analyses explicitly missing.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from summarize import ARMS, ROOT, SETS, sha


PAPER = (Path.home() / "tmp/zensim-paper/rev4")
MANIFEST = (Path.home() / "tmp/devin/rev4_featbank-potential_manifest.tsv")


def gather() -> tuple[dict, list[dict], dict]:
    summary_path = ROOT / "baseline_summary.json"
    summary = json.loads(summary_path.read_text())
    core = json.loads((ROOT / "core_gates_done.json").read_text())
    if not core["all_pass"] or core["summary_sha256"] != sha(summary_path):
        raise ValueError("core-gate receipt absent, failed, or stale")
    aggregates = []
    hashes = []
    for name in SETS:
        for arm in ARMS:
            for hidden in (32, 128):
                path = ROOT / "mlp_aggregate" / f"POT_{name}_{arm}_mlp{hidden}" / "result.json"
                row = json.loads(path.read_text())
                if (row["set"], row["arm"], row["hidden"]) != (name, arm, hidden):
                    raise ValueError(f"{path}: aggregate identity mismatch")
                if len(row["replicates"]) != 5 or row["B"] != 2000:
                    raise ValueError(f"{path}: incomplete seed/bootstrap receipt")
                if row["fold_manifest_sha256"] != summary["fold_manifest_sha256"]:
                    raise ValueError(f"{path}: fold manifest mismatch")
                for rep in row["replicates"]:
                    for source in [*rep["outer_sources"], rep["full_source"]]:
                        if sha(Path(source["path"])) != source["sha256"]:
                            raise ValueError(f"{path}: source result changed")
                aggregates.append(row)
                hashes.append(sha(path))
    index_hash = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    counts = {"deterministic_cells": len(summary["cells"]),
              "deterministic_cis": sum(r["ci_sha256"] is not None
                                       for r in summary["cells"]),
              "controls": len(summary["controls"]),
              "shams": len(summary["shams"]),
              "e1_pairwise_cells": len(summary["stats"]),
              "stability_sets": len(summary["stability"]),
              "deterministic_lodo_folds": len(summary["lodo"]),
              "transfer_matrices": len(summary["transfer_matrices"]),
              "transfer_matrix_cells": sum(r["cells"] for r in summary["transfer_matrices"]),
              "mlp_replicates": len(summary["mlp_replicates"]),
              "mlp_cis": sum(r["ci_sha256"] is not None
                             for r in summary["mlp_replicates"]),
              "mlp_outer_importance": sum(r["importance_sha256"] is not None
                                          for r in summary["mlp_replicates"]
                                          if r["outer"] is not None),
              "mlp_pooled_cells": len(aggregates)}
    expected = {"deterministic_cells": 32, "deterministic_cis": 32,
                "controls": 16, "shams": 16, "e1_pairwise_cells": 32,
                "stability_sets": 8, "deterministic_lodo_folds": 14,
                "transfer_matrices": 2, "transfer_matrix_cells": 98,
                "mlp_replicates": 960,
                "mlp_cis": 960, "mlp_outer_importance": 800,
                "mlp_pooled_cells": 32}
    if counts != expected:
        raise ValueError(f"core counts differ: {counts} != {expected}")
    controls = [(r["set"], r["model"]) for r in summary["controls"] if r["sensitive"]]
    receipt = {"summary_sha256": sha(summary_path),
               "core_gates_sha256": sha(ROOT / "core_gates_done.json"),
               "aggregate_index_sha256": index_hash, "counts": counts,
               "sensitive_controls": controls}
    return summary, aggregates, receipt


def table(aggregates: list[dict]) -> str:
    lines = ["| Set | Arm | Head | Rows / refs | Nested mean [95% CI] | Nested min–max | Full mean | Full min–max | Gap | Five nested seeds | Five full seeds | Init spread | Sample spread |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for row in aggregates:
        seeds = ", ".join(f"{r['nested_srocc']:.4f}" for r in row["replicates"])
        full_seeds = ", ".join(f"{r['in_sample_srocc']:.4f}" for r in row["replicates"])
        ci = row["nested_mean_ci95_srocc"]
        lines.append(
            f"| {row['set']} | {row['arm']} | H{row['hidden']} | "
            f"{row['rows']} / {row['references']} | "
            f"{row['nested_mean_srocc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{row['nested_min_srocc']:.4f}–{row['nested_max_srocc']:.4f} | "
            f"{row['in_sample_mean_srocc']:.4f} | "
            f"{row['in_sample_min_srocc']:.4f}–{row['in_sample_max_srocc']:.4f} | "
            f"{row['gap_mean_srocc']:.4f} | {seeds} | {full_seeds} | "
            f"{row['init_spread_fold_mean_srocc']:.4f} | "
            f"{row['sample_spread_fold_mean_srocc']:.4f} |")
    return "\n".join(lines)


def deterministic_table(summary: dict) -> str:
    lines = ["| Set | Model | Arm | Target | Rows / refs | Nested SROCC [95% CI] | In-sample SROCC [95% CI] | Gap |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for row in summary["cells"]:
        target = ("SSIM2 oracle /100" if row["target"] == "ssim2_oracle" else
                  "human MCOS/100" if row["set"] == "cid22_a25" else "human quality")
        nested, full = row["nested_ci95_srocc"], row["in_sample_ci95_srocc"]
        lines.append(f"| {row['set']} | {row['model']} | {row['arm']} | {target} | "
                     f"{row['rows']} / {row['references']} | "
                     f"{row['nested_srocc']:.4f} [{nested[0]:.4f}, {nested[1]:.4f}] | "
                     f"{row['in_sample_srocc']:.4f} [{full[0]:.4f}, {full[1]:.4f}] | "
                     f"{row['gap_srocc']:.4f} |")
    return "\n".join(lines)


def target_scale_table(summary: dict) -> str:
    scales = {row["set"]: row["label_source"]["label_scale"] for row in summary["cells"]}
    scales["konjnd_bpg_val"] = "ssim2/100 (reference-disjoint oracle evaluation)"
    lines = ["| Set | Target scale in its source set |", "|---|---|"]
    lines.extend(f"| {name} | {scale} |" for name, scale in sorted(scales.items()))
    return "\n".join(lines)


def control_table(summary: dict) -> str:
    lines = ["| Set | Model | Rows / refs | R0 − basic Δ SROCC [95% CI] | Sensitive | Zero sham Δ [95% CI] |",
             "|---|---|---:|---:|---|---:|"]
    shams = {(row["set"], row["model"]): row for row in summary["shams"]}
    for row in summary["controls"]:
        ci = row["ci95"]
        sham = shams[(row["set"], row["model"])]
        sci = sham["ci95"]
        lines.append(f"| {row['set']} | {row['model']} | {row['rows']} / {row['references']} | "
                     f"{row['delta_srocc']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{'yes' if row['sensitive'] else 'no'} | "
                     f"{sham['delta_srocc']:+.4f} [{sci[0]:+.4f}, {sci[1]:+.4f}] |")
    return "\n".join(lines)


def lodo_table(summary: dict) -> str:
    lines = ["| Model | Held out | Evaluation set | Target | Rows / refs | SROCC [95% CI] |",
             "|---|---|---|---|---:|---:|"]
    for row in summary["lodo"]:
        target = ("SSIM2 oracle /100" if row["target"] == "ssim2_oracle" else
                  "human MCOS/100" if row["eval_set"] == "cid22_a25" else "human quality")
        ci = row["ci95_srocc"]
        lines.append(f"| {row['model']} | {row['heldout']} | {row['eval_set']} | {target} | "
                     f"{row['rows']} / {row['references']} | "
                     f"{row['srocc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
    return "\n".join(lines)


def transfer_matrix_tables(summary: dict) -> str:
    heldouts = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
                "cid22_a25", "aic3", "kadid_select")
    blocks = []
    for record in summary["transfer_matrices"]:
        model = record["model"]
        path = Path(record["path"])
        if record["cells"] != 49 or sha(path) != record["sha256"]:
            raise ValueError(f"{path}: transfer-matrix summary hash/count mismatch")
        value = json.loads(path.read_text())
        source = ROOT / "lodo" / f"LODO_r0_{model}" / "result.json"
        if (value["schema"] != "rev4-featpot-lodo-cross-matrix-v1"
                or value["model"] != model or value["source_result_sha256"] != sha(source)
                or value["B"] != 2000 or value["seed"] != 20260923
                or value["unit"] != "reference" or set(value["rows"]) != set(heldouts)):
            raise ValueError(f"{path}: transfer-matrix provenance mismatch")
        lines = [f"**{model}**, receipt SHA-256 `{record['sha256']}`.", "",
                 "| Trained without ↓ / evaluated on → | " + " | ".join(heldouts) + " |",
                 "|---|" + "---:|" * len(heldouts)]
        for withheld in heldouts:
            cols = value["rows"][withheld]["columns"]
            if set(cols) != set(heldouts):
                raise ValueError(f"{path}: incomplete row {withheld}")
            cells = []
            for evaluated in heldouts:
                row = cols[evaluated]
                ci = row["ci95_srocc_signed"]
                if row["bootstrap_finite"] < 1900:
                    raise ValueError(f"{path}: incomplete CI {withheld}/{evaluated}")
                cells.append(f"{row['srocc_signed']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
            lines.append(f"| {withheld} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def stability_table(summary: dict) -> str:
    families = ("basic", "peaks", "masked_iw", "v2", "append", "append2")
    lines = ["| Set | Half-reference draws | " + " | ".join(families) + " |",
             "|---|---:|" + "---:|" * len(families)]
    for row in summary["stability"]:
        values = " | ".join(f"{row['family_frequency'][name]:.3f}" for name in families)
        lines.append(f"| {row['set']} | {row['draws']} | {values} |")
    return "\n".join(lines)


def d2_mlp_if_complete() -> list[dict] | None:
    """Read only saved, paired D2 receipts after all fourteen panels exist."""
    heldouts = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
                "cid22_a25", "aic3", "kadid_select")
    eval_substitution = {"konfig_train": "konfig_val",
                         "konjnd_bpg_train": "konjnd_bpg_val"}
    paths = [ROOT / "p2/d2_mlp_compare" / f"LODO_{name}_mlp{hidden}.json"
             for name in heldouts for hidden in (32, 128)]
    if not all(path.is_file() for path in paths):
        return None
    rows = []
    for name in heldouts:
        for hidden in (32, 128):
            path = ROOT / "p2/d2_mlp_compare" / f"LODO_{name}_mlp{hidden}.json"
            value = json.loads(path.read_text())
            expected_eval = eval_substitution.get(name, name)
            if (value["schema"] != "rev4-featpot-p2-d2-mlp-compare-v1" or
                    (value["heldout"], value["eval_set"], value["hidden"]) !=
                    (name, expected_eval, hidden) or value["B"] != 2000 or
                    value["seed"] != 20260923 or value["unit"] != "reference"):
                raise ValueError(f"{path}: D2 panel identity or bootstrap mismatch")
            arm = value["arms"]["r0"]
            if len(arm["seed_srocc"]) != 5 or len(arm["source_sha256"]) != 5:
                raise ValueError(f"{path}: incomplete P0 D2 seed receipt")
            for rep, expected_hash in enumerate(arm["source_sha256"]):
                source = (ROOT / "p2/d2_mlp" / f"LODO_r0_mlp{hidden}" /
                          f"without_{name}_r{rep}" / "result.json")
                if sha(source) != expected_hash:
                    raise ValueError(f"{path}: P0 D2 source {rep} changed")
            rows.append(value)
    return rows


def d2_table(rows: list[dict]) -> str:
    lines = ["| Held out | Evaluation set | Target | Head | Rows / refs | Mean SROCC [95% CI] | Five seeds | Min | Max |",
             "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for row in rows:
        arm = row["arms"]["r0"]
        ci = arm["ci95_mean_srocc"]
        target = "SSIM2 oracle /100" if row["label_source"]["target"] == "ssim2_oracle" else "human quality"
        seeds = ", ".join(f"{score:.4f}" for score in arm["seed_srocc"])
        lines.append(f"| `{row['heldout']}` | `{row['eval_set']}` | {target} | "
                     f"H{row['hidden']} | {row['rows']} / {row['references']} | "
                     f"{row['points_mean_srocc']['r0']:.4f} "
                     f"[{ci[0]:.4f}, {ci[1]:.4f}] | {seeds} | "
                     f"{arm['seed_min_srocc']:.4f} | {arm['seed_max_srocc']:.4f} |")
    return "\n".join(lines)


def lodo_jpeg_if_complete() -> list[dict] | None:
    """Use saved, source-pinned R0 contrasts without opening targets."""
    paths = [ROOT / "lodo_jpeg" / f"LODO_r0_{model}" / "result.json"
             for model in ("bvls", "linear")]
    if not all(path.is_file() for path in paths):
        return None
    rows = []
    heldouts = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
                "cid22_a25", "aic3", "kadid_select")
    for model, path in zip(("bvls", "linear"), paths):
        value = json.loads(path.read_text())
        source = ROOT / "lodo" / f"LODO_r0_{model}" / "result.json"
        if (value["schema"] != "rev4-featpot-lodo-jpeg-cross-contrast-v1"
                or value["model"] != model or value["source_result_sha256"] != sha(source)
                or set(value["folds"]) != set(heldouts)):
            raise ValueError(f"{path}: JPEG contrast provenance mismatch")
        for heldout in heldouts:
            row = value["folds"][heldout]
            if row["status"] == "MEASURED":
                if (row["B"] != 2000 or row["seed"] != 20260923
                        or row["unit"] != "reference" or row["owner"] != "panel --pairwise"
                        or row["bootstrap_finite"] < 1900):
                    raise ValueError(f"{path}: {heldout} bootstrap protocol mismatch")
            rows.append({"model": model, "heldout": heldout, **row})
    return rows


def jpeg_table(rows: list[dict]) -> str:
    lines = ["| Model | Held out | Status | Rows / refs | JPEG cross pairs | Other cross pairs | JPEG − other accuracy [95% CI] |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        pair_counts = row.get("pairs", {})
        delta = (f"{row['jpeg_minus_other']:.4f} "
                 f"[{row['ci95_jpeg_minus_other'][0]:.4f}, {row['ci95_jpeg_minus_other'][1]:.4f}]"
                 if row["status"] == "MEASURED" else "—")
        n = f"{row['rows']} / {row['references']}" if "rows" in row else "—"
        lines.append(f"| {row['model']} | {row['heldout']} | {row['status']} | {n} | "
                     f"{pair_counts.get('jpeg_cross', 0)} | {pair_counts.get('other_cross', 0)} | "
                     f"{delta} |")
    return "\n".join(lines)


def report(title: str, summary: dict, aggregates: list[dict], receipt: dict,
           bookmark: str, utc: str, p2_section: str = "",
           d2_rows: list[dict] | None = None,
           jpeg_rows: list[dict] | None = None) -> str:
    controls = [(r["set"], r["model"]) for r in summary["controls"] if r["sensitive"]]
    command = "python scripts/rev4_featpot/finalize_core_reports.py --verify-only"
    table_command = "python scripts/rev4_featpot/finalize_core_reports.py --table"
    output = json.dumps(receipt, sort_keys=True)
    d2_missing = ("- H32/H128 MLP LODO and incumbent/peer LODO rows, including peer-relative JPEG residuals, remain open. "
                  "The existing BVLS/lasso R0 seven-fold LODO receipts are available but do not substitute for those analyses."
                  if d2_rows is None else
                  "- Incumbent/peer LODO rows and peer-relative JPEG residuals remain open. "
                  "The P0 H32/H128 seven-fold transfer results are reported below.")
    d2_detail = ("" if d2_rows is None else
                 "\n## P0 H32/H128 D2 seven-fold transfer\n\n"
                 "All five fixed seeds and reference-clustered B=2,000 CIs are from paired saved D2 panels. "
                 "KonFiG and BPG use their reference-disjoint VAL views. BPG measures oracle agreement, not human generalization.\n\n"
                 + d2_table(d2_rows) + "\n")
    jpeg_detail = ("" if jpeg_rows is None else
                   "\n## R0 D2 JPEG versus other format contrast\n\n"
                   "This descriptive within-reference ordering comparison uses the E1b codec classifier "
                   "and B=2,000 paired reference bootstrap draws. Unclassifiable views are marked. "
                   "Peer-relative JPEG residuals remain open.\n\n"
                   + jpeg_table(jpeg_rows) + "\n")
    p2_missing = ("P2 is reported in the early section below; P1/P3 and C1–C4/all-C still await "
                  "reviewed Part B keyed sidecars. No candidate-family or adoption verdict is earned."
                  if p2_section else
                  "C1–C4, all-C, GMSD P1/P2/P3 and their matched within-reference permuted controls have not run. "
                  "Check reviewed keyed sidecars and D1/D2 coverage before candidate fits. "
                  "No candidate-family or adoption verdict is earned.")
    return f"""# {title} ({utc})

## MISSING / not done

{d2_missing}
- {p2_missing}
- MCL-JCI was excluded from this baseline; any later fitting use requires the D3 role ruling. Forbidden and secret holdouts remain unread.

{p2_section}

## Outcome

**POTENTIAL — ceiling, not a model score.** The P0 core baseline gates passed against the promoted Rev3 f32 bank, using only role-allowed bank labels. The verified receipt has 32/32 deterministic fits and reference CIs, 16/16 positive controls, 16/16 exact-zero shams, 32/32 E1/pairwise cells, 8/8 200-draw lasso stability sets, 14/14 deterministic R0 LODO folds, two 49-cell transfer matrices, 960/960 H32/H128 fit replicates and reference CIs, 800/800 outer-fold permutation-importance receipts, and 32/32 pooled five-seed MLP estimates. The baseline summary SHA-256 is `{receipt['summary_sha256']}`; the aggregate index SHA-256 is `{receipt['aggregate_index_sha256']}`. The only sensitive basic-ablation instruments are `{controls}`. A failed positive control rules out a family conclusion on that set/model.

CID22-A is human MCOS/100. KonJND-BPG is SSIMULACRA2 oracle/100 and may be negative. CID22 TRAIN and SafeSyn raw 0–100 oracle columns were not pooled with these targets. No held-out `pairs/` or `raw/` target copy was opened. No model was packed into `zensim/weights`, and nothing was pushed.

{target_scale_table(summary)}

The MLP table reports each head's mean across five pooled out-of-fold seed scores, paired reference-bootstrap CI for that mean, mean in-sample score and gap. The last two columns are the ranges of balanced **outer-fold mean** SROCC by init seed and sample-order seed; they are descriptive spreads, not model-selection criteria. Display values are rounded to four decimals; every exact score and resample is in the 32 `/var/tmp/rev4-featpot/mlp_aggregate/POT_*/result.json` receipts.

### Deterministic in-sample and nested-CV cells

{deterministic_table(summary)}

### Positive and zero-column controls

{control_table(summary)}

### Deterministic R0 D2 seven-fold transfer

KonFiG and BPG evaluate on reference-disjoint VAL views. BPG is an oracle target; no raw units are pooled.

{lodo_table(summary)}

The following 7×7 matrices show each fold model (row) against every D2 view (column), with B=2,000 reference CIs. Only the diagonal is held-out transfer. Off-diagonal cells may score a set used in fitting. BPG is an SSIMULACRA2 oracle /100; CID22-A is human MCOS/100; raw units are never pooled.

{transfer_matrix_tables(summary)}

### R0 lasso family stability

The entries are frequencies over 200 half-reference subsamples per set at the frozen 1-SE lambda.

{stability_table(summary)}

### H32/H128 non-negative-head MLP cells

{table(aggregates)}
{d2_detail}
{jpeg_detail}

## One-line recomputations and actual output

From the repository root:

```bash
{command}
```

Actual output: `{output}`.

```bash
{table_command}
```

Actual output is the tables above, including every MLP seed value. This command reads and validates the saved result receipts without opening label files.

## Commits and continuation

Local bookmark `quarantine/codex/featbank-potential` points to `{bookmark}`. Preregistration `d2169f5b` and preread GMSD/C1–C4 addendum `044f00dc043f` precede target reads. Exact commands, logs and source hashes are in the committed `benchmarks/featbank-potential_WORKLOG*.md` files and `/var/tmp/rev4-featpot/`. The remaining peer/candidate analyses require their stated gates and separate review; this report makes no overall family verdict.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    summary, aggregates, receipt = gather()
    d2_rows = d2_mlp_if_complete()
    jpeg_rows = lodo_jpeg_if_complete()
    if args.verify_only:
        print(json.dumps(receipt, sort_keys=True))
        return
    if args.table:
        print(target_scale_table(summary))
        print("\n" + deterministic_table(summary))
        print("\n" + control_table(summary))
        print("\n" + lodo_table(summary))
        print("\n" + transfer_matrix_tables(summary))
        print("\n" + stability_table(summary))
        print("\n" + table(aggregates))
        if d2_rows is not None:
            print("\n" + d2_table(d2_rows))
        if jpeg_rows is not None:
            print("\n" + jpeg_table(jpeg_rows))
        return
    bookmark = subprocess.check_output(
        ["jj", "log", "-r", "quarantine/codex/featbank-potential",
         "--no-graph", "-T", "commit_id.short(8)"], text=True).strip()
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for name, title in (
        ("FEATBANK_POTENTIAL_BASELINE_DONE.md", "FEATBANK POTENTIAL baseline — CORE GATES COMPLETE / LODO OPEN"),
        ("FEATBANK_POTENTIAL_DONE.md", "FEATBANK POTENTIAL — CORE BASELINE GATES COMPLETE / CANDIDATES BLOCKED"),
    ):
        if d2_rows is not None and name == "FEATBANK_POTENTIAL_BASELINE_DONE.md":
            title = title.replace("LODO OPEN", "P0 D2 MLP COMPLETE / PEER LODO OPEN")
        path = PAPER / name
        p2_section = ""
        if name == "FEATBANK_POTENTIAL_DONE.md" and path.is_file():
            previous = path.read_text()
            start, end = "<!-- BEGIN P2 -->", "<!-- END P2 -->"
            if start in previous and end in previous:
                p2_section = previous[previous.index(start):previous.index(end) + len(end)]
                title = title.replace("CANDIDATES BLOCKED",
                                      "P2 RUNNING; PART B CANDIDATES WAITING")
        if path.is_file():
            backup = ROOT / "report_snapshots" / f"{path.stem}_{utc}.md"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        temporary = path.with_suffix(path.suffix + ".new")
        temporary.write_text(report(title, summary, aggregates, receipt, bookmark, utc,
                                    p2_section, d2_rows, jpeg_rows))
        temporary.replace(path)
        with MANIFEST.open("a") as stream:
            stream.write(f"{utc}\tmodified\t{path}\n")
        print(json.dumps({"output": str(path), "sha256": sha(path)}))


if __name__ == "__main__":
    main()
