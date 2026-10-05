#!/usr/bin/env python3
"""HDR-route gate panel (amended form, registered in
benchmarks/hdr944_retrain_wave_2026-08-28.md). Computes, for each bake, on
the mc944 t1 VAL leg (census-clean, never trained):
  - per-codec swing FIDELITY = model p50 swing / target p50 swing over the
    pooled q-ladder (bar 0.65..1.5)
  - HG-mono = per-(rendition,codec) fraction of non-decreasing adjacent
    steps, on codecs whose target swing >= 25 (bar >= 0.93)
Forward = predict_features_with_bake (owner); no stats re-derived.

usage: hdr_route_panel.py <bake.bin> [<bake2.bin> ...] [--parquet P]
Metric-only HDRTEACH (no bake, feature read or fit):
  hdr_route_panel.py --teacher-parquet TRAIN --teacher-parquet VAL --teacher-output-dir DIR
"""
import argparse, json, math, os, struct, subprocess, sys, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]


def _teacher_panel(paths, output_dir):
    """Descriptive HDRTEACH agreement using canonical signed SROCC, no fit."""
    sys.path.insert(0, str(REPO))
    from scripts.lib.zen_stats import panel_batch_indexed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, populations = {}, {}
    for path in paths:
        table = pq.read_table(path)
        footer = table.schema.metadata or {}
        if footer.get(b"zensim.hdrteach.study") != b"HDRTEACH-2026-10-04":
            raise ValueError("not an admitted HDRTEACH table")
        role = footer[b"zensim.hdrteach.role"].decode()
        if role not in {"train", "val"} or role in results:
            raise ValueError("unexpected/duplicate role")
        rows = table.to_pylist()
        if not rows or any(x["role"] != role for x in rows):
            raise ValueError("empty/mixed-role teacher table")
        groups, families = defaultdict(list), defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["ref_path"]].append(i)
            families[row["source_family"]].append(i)
        hv = [x["hdrvdp3_q_jod"] for x in rows]
        cv = [x["cvvdp_jod"] for x in rows]
        if not all(math.isfinite(x) for x in hv + cv):
            raise ValueError("nonfinite teacher")
        bases = {"hdrvdp3": hv, "cvvdp": cv}
        jobs = [("ALL", "hdrvdp3", "cvvdp", None)]
        jobs += [(ref, "hdrvdp3", "cvvdp", ids) for ref, ids in groups.items()]
        panels = panel_batch_indexed(bases, jobs, stats="srocc")
        if any(p["n_dropped"] for p in panels):
            raise ValueError("canonical panel dropped admitted rows")
        per_ref = []
        for (ref, ids), panel in zip(groups.items(), panels[1:]):
            absres = [abs(rows[i]["rank_residual_normalized"]) for i in ids]
            per_ref.append(dict(ref_path=ref, source_family=rows[ids[0]]["source_family"], rows=len(ids),
                                srocc_signed=panel["srocc_signed"], mean_abs_rank_residual_normalized=float(np.mean(absres)),
                                max_abs_rank_residual_normalized=max(absres),
                                agree_count=sum(rows[i]["agree"] for i in ids)))
        per_family = []
        for family, ids in families.items():
            refs = [x for x in per_ref if x["source_family"] == family]
            rho = [x["srocc_signed"] for x in refs if math.isfinite(x["srocc_signed"])]
            per_family.append(dict(source_family=family, rows=len(ids), references=len(refs),
                                   within_reference_srocc_mean=float(np.mean(rho)) if rho else None,
                                   mean_abs_rank_residual_normalized=float(np.mean([abs(rows[i]["rank_residual_normalized"]) for i in ids])),
                                   max_abs_rank_residual_normalized=max(abs(rows[i]["rank_residual_normalized"]) for i in ids),
                                   agree_count=sum(rows[i]["agree"] for i in ids)))
        rhos = [x["srocc_signed"] for x in per_ref if math.isfinite(x["srocc_signed"])]
        rank_sorted = lambda xs: sorted(xs, key=lambda x: (-x["mean_abs_rank_residual_normalized"], str(x.get("ref_path", x.get("source_family")))))
        worst_rows = sorted(rows, key=lambda x: (-abs(x["rank_residual_normalized"]), x["row_id"]))
        worst_pooled = sorted(rows, key=lambda x: (-abs(x["rank_residual_pooled_normalized"]), x["row_id"]))
        # Report example identities and scores without repeating every constant provenance field.
        fields = ["row_id", "source_family", "ref_path", "dist_path", "q", "hdrvdp3_q_jod", "cvvdp_jod",
                  "rank_hdrvdp3_within_ref", "rank_cvvdp_within_ref", "rank_residual_positions",
                  "rank_residual_normalized", "rank_residual_pooled_normalized", "agree"]
        examples = lambda xs: [{k: x[k] for k in fields} for x in xs[:20]]
        result = dict(rows=len(rows), references=len(groups), source_families=len(families),
                      cvvdp_label_era=rows[0]["cvvdp_label_era"], pooled_srocc_signed=panels[0]["srocc_signed"],
                      within_reference=dict(defined=len(rhos), undefined=len(per_ref)-len(rhos),
                                            mean=float(np.mean(rhos)) if rhos else None,
                                            median=float(np.median(rhos)) if rhos else None,
                                            min=min(rhos) if rhos else None, max=max(rhos) if rhos else None,
                                            negative=sum(x < 0 for x in rhos)),
                      agree_count=sum(x["agree"] for x in rows), agree_fraction=sum(x["agree"] for x in rows)/len(rows),
                      hdrvdp3_range=dict(min=min(hv), max=max(hv), negative=sum(x < 0 for x in hv), exactly10=sum(x == 10 for x in hv)),
                      cvvdp_range=dict(min=min(cv), max=max(cv)),
                      mean_abs_rank_residual_normalized=float(np.mean([abs(x["rank_residual_normalized"]) for x in rows])),
                      worst_rows_within_reference=examples(worst_rows), worst_rows_pooled=examples(worst_pooled),
                      worst_references=rank_sorted(per_ref)[:20], worst_families=rank_sorted(per_family),
                      per_reference=per_ref, per_family=per_family)
        if role == "val":
            mix = [x["historic_cvvdp_mix"] for x in rows]
            jobs = [("ALL", "hdrvdp3", "mix", None)] + [(ref, "hdrvdp3", "mix", ids) for ref, ids in groups.items()]
            aux = panel_batch_indexed({"hdrvdp3": hv, "mix": mix}, jobs, stats="srocc")
            rho = [x["srocc_signed"] for x in aux[1:] if math.isfinite(x["srocc_signed"])]
            result["historic_mix_auxiliary"] = dict(pooled_srocc_signed=aux[0]["srocc_signed"],
                                                      within_reference_mean=float(np.mean(rho)) if rho else None, defined=len(rho))
        results[role] = result
        populations[role] = rows

    if set(results) != {"train", "val"}:
        raise ValueError("both admitted roles are required")
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return value
    result_path = output_dir / "agreement.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(finite_json(results), indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, role in zip(axes, ["train", "val"]):
        rows = populations[role]
        counts, _, _, mount = ax.hist2d([x["hdrvdp3_q_jod"] for x in rows], [x["cvvdp_jod"] for x in rows], bins=70, cmap="magma", cmin=1)
        if int(np.nansum(counts)) != len(rows):
            raise ValueError("density omitted rows")
        ax.set(xlabel="HDR-VDP-3 q_jod (ppd60)", ylabel="CVVDP JOD (" + ("fresh" if role == "train" else "historic") + ")",
               title=f"{role.upper()}: all {len(rows):,} pairs; signed rho={results[role]['pooled_srocc_signed']:.5f}")
        fig.colorbar(mount, ax=ax, label="Pairs / bin")
    fig.savefig(output_dir / "teacher_density.png", dpi=160)
    fig.savefig(output_dir / "teacher_density.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row_axes, role in zip(axes, ["train", "val"]):
        for ax, ref in zip(row_axes, results[role]["worst_references"][:3]):
            ladder = sorted((x for x in populations[role] if x["ref_path"] == ref["ref_path"]), key=lambda x: float(x["q"]))
            q = [float(x["q"]) for x in ladder]
            ax.plot(q, [x["rank_hdrvdp3_within_ref"] for x in ladder], "o-", label="HDR-VDP-3")
            ax.plot(q, [x["rank_cvvdp_within_ref"] for x in ladder], "x-", label="CVVDP")
            geometry = Path(ref["ref_path"]).name.split(".scale")[-1].split(".hdr")[0]
            ax.set(xlabel="Encoder q", ylabel="Within-reference quality midrank",
                   title=f"{role.upper()} origin {ref['source_family']} / {geometry}")
            ax.legend()
    fig.savefig(output_dir / "worst_ladders.png", dpi=150)
    fig.savefig(output_dir / "worst_ladders.pdf")
    plt.close(fig)

    # Format the stored owner results; no alternative statistic/report implementation.
    provenance = json.loads(footer[b"zensim.hdrteach.provenance"])
    train, val = results["train"], results["val"]
    lines = ["# HDRTEACH — fixed-condition native HDR-VDP-3 teacher labels", "",
             "## MISSING first", "",
             "- **Independent human HDR validation: MISSING.** These are two metric teachers, not new human observations. No model is fitted or qualified by this packet.",
             "- **Independent UPIQ testing: unavailable for a student trained on these targets.** The HDR-VDP-3 paper, sections 3 and 5, explicitly states quality calibration/recalibration using UPIQ (>4,000 SDR/HDR images). Publication text alone was read, not UPIQ labels or images. [HDR-VDP-3 paper](https://arxiv.org/pdf/2304.13625), [author documentation](https://hdrvdp.sourceforge.net/wiki/).",
             "- **Fresh native CVVDP VAL labels: MISSING.** VAL retains its registered historic JOD and mixed labels; comparison differences may include judge-era/viewing differences. TRAIN and VAL are reported separately.",
             "- Full MATLAB parity on these image pairs and product/model qualification are not claimed. The existing 21 synthetic VDP3 reference-port golden cases and invalid-input tests passed.", "",
             "## Complete label and teacher-table coverage", "",
             f"Exactly {train['rows']:,} corrected TRAIN pairs ({train['references']} references / {train['source_families']} source families) and {val['rows']:,} registered hdr_v3mix VAL pairs ({val['references']} / {val['source_families']}) are retained under their original roles. All use native declared-PQ PNG/JXL and explicit common-primary conversion to absolute BT.709 RGB nits. There is no 8-bit scoring conversion. Transport is f32 absolute RGB, widened to the owner's f64 VDP3 metric. All distortion variants here are zenjxl.", "",
             "One fixed condition: ppd60, quality task, RGB BT.709 input, led-lcd-srgb spectral emission, surround none, observer age24, reference quality options, PQ EOTF peak10,000 nits, no added ambient reflection. The absolute input is used without an additional configured display-peak override. HDR-VDP-3's native gamut floor remains part of its reference algorithm. Physical screen dimensions/distance are not supplied because ppd is explicit.", "",
             f"zenmetrics source `{provenance['zenmetrics_commit']}`; binary SHA256 `{provenance['zenmetrics_binary_sha256']}`. Each Parquet row carries this provenance, every viewing/default option, exact original role/row ID, reference/distorted paths and hashes, source authority/hash, raw teacher values, rank residuals and boolean `agree`. Original feature columns are not mixed or copied.", "",
             "## Measured agreement", "",
             "Signed SROCC comes from the canonical Rust panel in SROCC-only mode: no logistic calibration or model fit. Within-reference mean gives each reference variant equal weight; undefined constant groups are counted and excluded from that mean. No row is dropped. Source families are original origin IDs, with geometry variants grouped separately for rank calculation.", ""]

    def mdtable(headers, values):
        lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        lines.extend("| " + " | ".join(str(v) for v in value) + " |" for value in values)
        lines.append("")

    def number(value, precision=6):
        return f"{value:.{precision}f}" if value is not None and math.isfinite(value) else "undefined"

    mdtable(["Set / CVVDP era", "Pairs", "Signed SROCC", "Within-ref mean", "Within-ref median", "Undefined refs", "agree"],
            [[role.upper() + (" / fresh" if role == "train" else " / historic"), x["rows"], f"{x['pooled_srocc_signed']:.6f}",
              number(x['within_reference']['mean']), number(x['within_reference']['median']), x["within_reference"]["undefined"],
              f"{x['agree_count']}/{x['rows']} ({100*x['agree_fraction']:.2f}%)"] for role, x in results.items()])
    x = results["val"]["historic_mix_auxiliary"]
    lines.extend([f"VAL historic cvvdp-mix, auxiliary only: signed pooled SROCC **{number(x['pooled_srocc_signed'])}**; within-reference mean **{number(x['within_reference_mean'])}**, {x['defined']} defined references. This mix does not define the agree flag.", ""])
    mdtable(["Set", "HDR-VDP-3 min / max", "Negative / exactly10", "Within-ref rho min / max", "Negative refs"],
            [[role.upper(), f"{x['hdrvdp3_range']['min']:.9f} / {x['hdrvdp3_range']['max']:.9f}",
              f"{x['hdrvdp3_range']['negative']} / {x['hdrvdp3_range']['exactly10']}",
              number(x['within_reference']['min']) + " / " + number(x['within_reference']['max']), x["within_reference"]["negative"]] for role, x in results.items()])
    lines.extend(["## Preregistered agree rule", "",
                  "Before new labels, the rule was frozen: within the exact same reference and original role, rank each teacher's quality in ascending order using averaged tied midranks; agree is true iff the absolute rank difference is <=1.0 position. Groups with fewer than two rows or nonfinite values fail closed. A constant teacher has defined tied midranks but undefined Spearman correlation. Final admission additionally requires finite teacher values for every row.", "",
                  "One position is 1/14 of the TRAIN ladder span and 1/12 of VAL's. The flag measures ordering tolerance, not equality of JOD calibration or perceptual significance. It has not been tuned to results. Both flags and all rows remain in the table; any future fit/filter is separately registered and TRAIN-only.", "",
                  "## Largest disagreements and examples", "",
                  "Within-reference residual = HDR-VDP-3 midrank minus CVVDP midrank, divided by n-1. Reference/family severity is mean absolute normalized residual. Rows sort by absolute normalized residual, ties by original row ID; references/families break ties by path/ID. Positive residual means HDR-VDP-3 places the row higher. Pooled residual uses the same ranks across the entire original set and is reported separately in agreement.json.", ""])
    for role, x in results.items():
        lines.extend([f"### {role.upper()} — worst rows", ""])
        mdtable(["Row ID", "Origin / geometry", "q", "HDR-VDP-3", "CVVDP JOD", "HDR / CV ranks", "Rank residual", "agree"],
                [[row["row_id"], row["source_family"] + " / " + Path(row["ref_path"]).name.split(".scale")[-1].split(".hdr")[0], row["q"],
                  f"{row['hdrvdp3_q_jod']:.9f}", f"{row['cvvdp_jod']:.9f}", f"{row['rank_hdrvdp3_within_ref']:g} / {row['rank_cvvdp_within_ref']:g}",
                  f"{row['rank_residual_positions']:+g}", row["agree"]] for row in x["worst_rows_within_reference"][:10]])
        lines.extend([f"### {role.upper()} — worst references", ""])
        mdtable(["Origin / geometry", "Mean abs normalized residual", "Signed SROCC", "agree / n"],
                [[ref["source_family"] + " / " + Path(ref["ref_path"]).name.split(".scale")[-1].split(".hdr")[0],
                  f"{ref['mean_abs_rank_residual_normalized']:.6f}", number(ref['srocc_signed']), f"{ref['agree_count']}/{ref['rows']}"] for ref in x["worst_references"][:10]])
        lines.extend([f"### {role.upper()} — worst source families", ""])
        mdtable(["Origin ID", "Pairs / refs", "Mean abs normalized residual", "Within-ref rho", "agree / n"],
                [[fam["source_family"], f"{fam['rows']} / {fam['references']}", f"{fam['mean_abs_rank_residual_normalized']:.6f}",
                  number(fam['within_reference_srocc_mean']), f"{fam['agree_count']}/{fam['rows']}"] for fam in x["worst_families"][:10]])
        row = x["worst_rows_within_reference"][0]
        lines.extend([f"Example {role.upper()} row {row['row_id']}: reference `{row['ref_path']}`; distortion `{row['dist_path']}`. Their immutable hashes and exact authority binding are in the teacher table. The worst-ladder plots retain every q point of the three selected reference examples; they do not depict human judgments.", ""])
    lines.extend(["## Artifacts and limits", "",
                  f"agreement.json retains every reference/family panel and separate worst within-reference/pooled rows. teacher_density.png/pdf includes exactly all {train['rows']:,} / {val['rows']:,} raw score pairs; worst_ladders.png/pdf shows the three highest-residual references per role without curve fitting. The Parquet tables retain all rows and all per-row rank residuals, not just the examples.", "",
                  "Teacher agreement cannot establish human accuracy or compare the teachers' scientific merit. Repeated scales from the same origin are dependent; no confidence interval or generalization claim is made. No bake, public API, feature regime, source split, model/default or integrity companion changes. No UPIQ labels/images, secret holdout, model fit, promotion or push.", ""])
    (output_dir / "teacher_agreement.md").write_text("\n".join(lines))
    for role, result in results.items():
        print(role, result["pooled_srocc_signed"], result["within_reference"], result["agree_count"], flush=True)


def _e26_panel(root, control_root, bank, output_dir):
    """E26 frozen native feature cache, production BakeScorer and canonical signed panels."""
    import hashlib
    sys.path.insert(0, str(REPO / "scripts/rev4_featpot"))
    sys.path.insert(0, str(REPO))
    import e24_rev5 as e24
    from e13_teacher import paired
    from v2_common import dense_bake, sha, FITBIN, PANEL
    from rev5_bank import HDR_TEACHER_SHA
    from scripts.lib.zen_stats import panel_batch_indexed

    root, control_root, bank, output_dir = map(Path, (root, control_root, bank, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((bank / "_MANIFEST.json").read_text())
    ids = e24.e21.columns("by_v2fy")
    if (man.get("role") != "val" or man.get("rows") != 3900 or man.get("formula_revision") != "Rev5"
            or man.get("teacher_sha256") != HDR_TEACHER_SHA["val"] or man.get("requested_ids") != ids):
        raise ValueError("E26 VAL bank is not the complete registered native Rev5 population")
    for name, pin in [("keys.parquet", "keys_sha256"), ("features.parquet", "features_parquet_sha256")]:
        if sha(bank / name) != man[pin]:
            raise ValueError("E26 native VAL cache changed")
    keys = pq.read_table(bank / "keys.parquet").to_pylist()
    features = pq.read_table(bank / "features.parquet")
    if features["row_id"].to_pylist() != [r["row_id"] for r in keys] or any(r["role"] != "val" for r in keys):
        raise ValueError("E26 VAL feature/label row join mismatch")
    groups = defaultdict(list)
    for i, row in enumerate(keys):
        groups[row["ref_path"]].append(i)
    if len(groups) != 300 or any(len(v) != 13 for v in groups.values()):
        raise ValueError("E26 VAL reference census differs")
    width = man["feature_width"]
    X = np.column_stack([features[f"f{i}"].to_numpy() for i in range(width)]).astype("<f8")
    for i in range(width):
        if not (np.isfinite(X[:, i]).all() if i in ids else np.isnan(X[:, i]).all()):
            raise ValueError(f"E26 native VAL measured/absent contract f{i}")
    wire = output_dir / "native_val.f64.wire"
    with wire.open("wb") as f:
        f.write(struct.pack("<II", width, len(keys)))
        f.write(X.tobytes())
    tool = os.environ.get("ZL_PREDICT", str(REPO / "target/release/predict_features_with_bake"))
    cells = {}
    for label, base, spec in [("control", control_root, e24.CONTROL),
                             *((f"hd{w}", root, e24.CONTROL+f":hd{w}") for w in (4, 16))]:
        result_cells = []
        for source in e24.SOURCE_ORDER:
            for seed in e24.SEEDS:
                cell = base / "cells" / f"{spec}__N" / f"without_{source}_s{seed}"
                result = json.loads((cell / "result.json").read_text())
                bake = cell / "refit/last.bin"
                if (result["selected_bake_sha256"] != sha(bake) or result["selected_epoch"] != 119
                        or result["epoch_rule"] != "last" or result["heldout"] != source or result["seed_index"] != seed):
                    raise ValueError("E26 cell identity/epoch/bake receipt mismatch")
                dense, binding = e24.e26_bound_bake(cell, output_dir)
                proc = subprocess.run([tool, "--bake", str(dense), "--features-file", str(wire), "--f64-wire", "--production"],
                                      capture_output=True, text=True, check=True)
                pred = np.array([float(v) for v in proc.stdout.split()], dtype=np.float64)
                if len(pred) != 3900 or not np.isfinite(pred).all():
                    raise ValueError("E26 native VAL prediction coverage/nonfinite refusal")
                bases = {"prediction": pred, "hdrvdp3": [r["hdrvdp3_q_jod"] for r in keys],
                         "cvvdp": [r["cvvdp_jod"] for r in keys]}
                jobs = [("ALL", "prediction", target, None) for target in ("hdrvdp3", "cvvdp")]
                jobs += [(ref, "prediction", target, ix) for target in ("hdrvdp3", "cvvdp") for ref, ix in groups.items()]
                panels = panel_batch_indexed(bases, jobs, stats="srocc")
                if any(p["n_dropped"] or not math.isfinite(p["srocc_signed"]) for p in panels):
                    raise ValueError("E26 canonical signed panel dropped/undefined admitted rows")
                means = {t: float(np.mean([p["srocc_signed"] for p in panels[2+j*300:2+(j+1)*300]]))
                         for j, t in enumerate(("hdrvdp3", "cvvdp"))}
                np.save(output_dir / f"{label}_{source}_s{seed}.npy", pred, allow_pickle=False)
                result_cells.append(dict(source=source, seed=seed, bake_sha256=sha(bake), dense_sha256=sha(dense),
                    result_sha256=sha(cell / "result.json"), serving_binding=binding, prediction_sha256=hashlib.sha256(pred.astype("<f8").tobytes()).hexdigest(),
                    within_reference=means, pooled={t: panels[j]["srocc_signed"] for j, t in enumerate(("hdrvdp3", "cvvdp"))},
                    per_reference={t: [p["srocc_signed"] for p in panels[2+j*300:2+(j+1)*300]] for j,t in enumerate(("hdrvdp3", "cvvdp"))}))
        cells[label] = result_cells
        print(f"E26 HDR {label}: {len(result_cells)} complete VAL panels", flush=True)
    sdr = json.loads((root / "compare/e26_sdr_decision.json").read_text())
    arms = {}
    for label in ("hd4", "hd16"):
        if [(c["source"], c["seed"]) for c in cells[label]] != [(c["source"], c["seed"]) for c in cells["control"]]:
            raise ValueError("E26 HDR cell pairing mismatch")
        within = {t: paired([c["within_reference"][t] for c in cells[label]],
                            [c["within_reference"][t] for c in cells["control"]]) for t in ("hdrvdp3", "cvvdp")}
        pooled = {t: paired([c["pooled"][t] for c in cells[label]],
                            [c["pooled"][t] for c in cells["control"]]) for t in ("hdrvdp3", "cvvdp")}
        hdr_pass = within["hdrvdp3"]["delta"] > 2*within["hdrvdp3"]["se"] and within["cvvdp"]["delta"] >= -2*within["cvvdp"]["se"]
        arms[label] = dict(sdr=sdr[label], within_reference=within, pooled_report_only=pooled,
                           hdr_pass=bool(hdr_pass), passes=bool(hdr_pass and sdr[label]["as_good"]))
    decision = dict(schema="e26-registered-decision-v1", rows=3900, references=300, cells_per_arm=50,
                    pairing="all 50 registered (fold,seed) cells", arms=arms,
                    adopt=next((a for a in ("hd4", "hd16") if arms[a]["passes"]), None),
                    val_bank_manifest_sha256=sha(bank / "_MANIFEST.json"), tools={str(p):sha(p) for p in (Path(tool), FITBIN, PANEL)},
                    status="COMPLETE", cells=cells)
    (output_dir / "e26_hdr_decision.json").write_text(json.dumps(decision, indent=1)+"\n")
    print(json.dumps({"adopt":decision["adopt"], "arms":arms}), flush=True)


if "--e26-root" in sys.argv:
    e26_ap = argparse.ArgumentParser(description="Registered E26 native HDR VAL panel")
    e26_ap.add_argument("--e26-root", required=True)
    e26_ap.add_argument("--e26-control-root", required=True)
    e26_ap.add_argument("--e26-val-bank", required=True)
    e26_ap.add_argument("--e26-output-dir", required=True)
    e26_args = e26_ap.parse_args()
    _e26_panel(e26_args.e26_root, e26_args.e26_control_root, e26_args.e26_val_bank, e26_args.e26_output_dir)
    raise SystemExit(0)


if "--teacher-parquet" in sys.argv:
    teacher_ap = argparse.ArgumentParser(description="Pinned HDRTEACH descriptive teacher agreement")
    teacher_ap.add_argument("--teacher-parquet", action="append", required=True)
    teacher_ap.add_argument("--teacher-output-dir", required=True)
    teacher_args = teacher_ap.parse_args()
    _teacher_panel(teacher_args.teacher_parquet, teacher_args.teacher_output_dir)
    raise SystemExit(0)

ap = argparse.ArgumentParser()
ap.add_argument("bakes", nargs="+")
ap.add_argument("--parquet", default="/mnt/v/zen/zensim-training/hdrgrid-mc944-t1-2026-08-27/hdrgrid_mc944_t2_val.parquet")
a = ap.parse_args()

t = pq.read_table(a.parquet)
fcols = [f"feat_{i}" for i in range(944)]
X = np.column_stack([np.asarray(t[c].to_pylist(), np.float32) for c in fcols])
target = np.asarray(t["human_score"].to_pylist(), float) * 100.0
codec = t["codec"].to_pylist(); rend = t["image_path"].to_pylist(); qv = t["q"].to_pylist()
tool = os.environ.get("ZL_PREDICT", str(REPO / "target/release/predict_features_with_bake"))

def swing_and_mono(vals):
    """vals: array aligned to rows. Returns per-codec (swing, mono)."""
    pooled = defaultdict(lambda: defaultdict(list))   # codec -> q -> [v]
    ladders = defaultdict(lambda: defaultdict(list))  # codec -> rendition -> [(q, v)]
    for c, r, q, v in zip(codec, rend, qv, vals):
        pooled[c][q].append(v); ladders[c][r].append((q, v))
    out = {}
    for c, qmap in pooled.items():
        qs = sorted(qmap)
        p50 = {q: float(np.median(qmap[q])) for q in qs}
        swing = p50[qs[-1]] - p50[qs[0]]
        monos = []
        for r, pts in ladders[c].items():
            pts = sorted(pts)
            d = np.diff([v for _, v in pts])
            if len(d): monos.append(float((d >= -0.05).mean()))
        out[c] = (swing, float(np.mean(monos)))
    return out

tgt = swing_and_mono(target)
print("target swings:", {c: round(s, 2) for c, (s, _) in tgt.items()})
for bake in a.bakes:
    with tempfile.NamedTemporaryFile(suffix=".wire", delete=False) as f:
        f.write(struct.pack("<II", X.shape[1], X.shape[0]))
        f.write(X.astype("<f4").tobytes())
        wire = f.name
    try:
        r = subprocess.run([tool, "--bake", bake, "--features-file", wire],
                           capture_output=True, text=True, check=True)
    finally:
        os.unlink(wire)
    pred = np.array([float(v) for v in r.stdout.split()])
    assert len(pred) == len(target), (len(pred), len(target))
    mdl = swing_and_mono(pred)
    cells, ok = [], True
    for c in sorted(tgt):
        ts, _ = tgt[c]; ms, mono = mdl[c]
        fid = ms / ts if ts else float("nan")
        fid_ok = 0.65 <= fid <= 1.5
        mono_ok = mono >= 0.93 if ts >= 25 else None
        ok &= fid_ok and (mono_ok is not False)
        cells.append(f"{c}: fid={fid:.2f}{'✓' if fid_ok else '✗'}"
                     + (f" mono={mono:.3f}{'✓' if mono_ok else '✗'}" if mono_ok is not None else f" mono={mono:.3f}(ungated)"))
    print(f"{os.path.basename(bake):<36} {'PASS' if ok else 'FAIL'}  " + " | ".join(cells))
