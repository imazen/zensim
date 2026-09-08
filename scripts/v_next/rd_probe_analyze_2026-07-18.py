#!/usr/bin/env python3
"""diffmap-RD probe analysis (2026-07-18) — turns the probe matrix into the two
pre-registered verdicts of docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md:

HELPFUL  — bytes saved vs the same codec's no-diffmap baseline at EQUAL
           ACHIEVED JUDGE SCORE (log-bytes interpolation on the baseline's
           per-image score→bytes frontier; never at nominal distance/target).
           Judges: ssim2 + butteraugli + zensim (uniform zenmetrics build) —
           per-judge tables so home-turf cells are visible (#38 convention).
EFFICIENT — passes / encode_ms / |achieved−T| residuals per driver.

Reads:  $RD/jxl/manifest_*.tsv, $RD/zenjpeg/probe.tsv, $RD/judge_{ssim2,butteraugli,zensim}.tsv
Writes: $RD/analysis_summary.md (+ prints it)
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

RD = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/rd-target-eval-2026-07"
CLASS = {"codec_wiki": "screen", "gmessages": "screen"}  # rest = photo


def load_judges():
    """dist_path -> {judge: score}. Butteraugli negated (higher=better uniformly)."""
    out = defaultdict(dict)
    specs = [
        ("ssim2", f"{RD}/judge_ssim2.tsv", ("ssim2", "ssim2_gpu"), +1),
        ("butter", f"{RD}/judge_butteraugli.tsv", ("butteraugli_pnorm3", "butteraugli_max"), -1),
        ("zensim", f"{RD}/judge_zensim.tsv", ("zensim",), +1),
    ]
    for name, path, cols, sign in specs:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                v = next((r[c] for c in cols if c in r and r[c] not in ("", "nan")), None)
                if v is not None:
                    out[r["dist_path"]][name] = sign * float(v)
    return out


def interp_bytes(frontier, score):
    """log-bytes at `score` on a per-image (score, bytes) frontier; None if outside."""
    pts = sorted(frontier)
    if not pts or score < pts[0][0] or score > pts[-1][0]:
        return None
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if s0 <= score <= s1:
            if s1 == s0:
                return b0
            t = (score - s0) / (s1 - s0)
            return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
    return pts[-1][1]


def helpful(cells, baseline_label, judges, codec):
    """cells: list of dicts {image, label, bytes, dist_path}. Returns
    {(judge, driver, class): [bytes_saved_frac,...]}."""
    frontiers = defaultdict(list)  # (judge, image) -> [(score, bytes)]
    for c in cells:
        if c["label"] != baseline_label:
            continue
        for j, s in judges.get(c["dist_path"], {}).items():
            frontiers[(j, c["image"])].append((s, c["bytes"]))
    saved = defaultdict(list)
    for c in cells:
        if c["label"] == baseline_label:
            continue
        for j, s in judges.get(c["dist_path"], {}).items():
            base = interp_bytes(frontiers.get((j, c["image"]), []), s)
            if base and base > 0:
                cls = CLASS.get(c["image"], "photo")
                saved[(j, c["label"], cls)].append(1.0 - c["bytes"] / base)
    return saved


def fmt_saved(saved, drivers, judges=("ssim2", "butter", "zensim")):
    lines = [
        "| driver | class | " + " | ".join(f"{j} med% (n)" for j in judges) + " |",
        "|---|---|" + "---|" * len(judges),
    ]
    for d in drivers:
        for cls in ("photo", "screen"):
            cells = []
            for j in judges:
                v = saved.get((j, d, cls), [])
                cells.append(f"{100*st.median(v):+.1f} ({len(v)})" if v else "—")
            lines.append(f"| {d} | {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def target_loop_main():
    """Read actual demo_matrix outputs; reuse the existing matched-judge owner.

    The sparse q curves are a diagnostic interpolation, not an RDO gain claim.
    No extrapolation; retain out-of-range/unreachable cases in target summaries.
    """
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-loop", type=Path, required=True)
    ap.add_argument("--baseline", default="B")
    ap.add_argument("--screen-source", action="append", type=int, default=[])
    args = ap.parse_args()
    root = args.target_loop
    if not (root / "COMPLETE").is_file():
        raise SystemExit("target-loop matrix incomplete; refusing a success summary")
    rows = [json.loads(line) for line in (root / "measurements.jsonl").read_text().splitlines()]
    inp = json.loads((root / "INPUTS.json").read_text())
    if inp.get("schema") == "reachable-target-v1":
        return reachable_target_summary(root, inp, rows)
    expected = len(inp["sources"]) * len(inp["codecs"]) * len(inp["targets"]) * (2 + len(inp["bakes"]))
    if len(rows) != expected or any("achieved" not in r for r in rows):
        raise SystemExit("missing/failed target-loop cells")
    expected_keys = {(s, c, m, t) for s in range(len(inp["sources"]))
                     for c in inp["codecs"] for m in inp["profiles"] + [Path(b["path"]).stem for b in inp["bakes"]]
                     for t in inp["targets"]}
    keys = [(r["source"], r["codec"], r["model"], r["target"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise SystemExit("duplicate/mismatched target-loop cells")
    CLASS.update({str(i): "screen" for i in args.screen_source})
    result = {"instrument": "rd_probe_analyze target-loop", "baseline": args.baseline,
              "cells": len(rows), "max_iterations": inp["max_iterations"],
              "input_sha256": __import__("hashlib").sha256((root / "INPUTS.json").read_bytes()).hexdigest(),
              "measurement_sha256": __import__("hashlib").sha256((root / "measurements.jsonl").read_bytes()).hexdigest(),
              "summary": [], "matched_judge": [],
              "limitation": "four-source scalar-quality-controller diagnostic; sparse log-byte interpolation without extrapolation; no encoder-RDO intervention or qualified RD improvement"}
    text = ["# Actual codec-target loop", "",
            "Targets use each model's own dial. Independent judges are reported separately.",
            "Sparse matched-quality interpolation is diagnostic; this controller changes only q, not codec RDO.", "",
            "| codec | model | hits / n | median abs error | p95 abs error | median passes | median loop ms | median score ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    rdtext = []
    for codec in inp["codecs"]:
        group = [r for r in rows if r["codec"] == codec]
        models = sorted({r["model"] for r in group})
        if args.baseline not in models:
            raise SystemExit("baseline absent")
        for model in models:
            rr = [r for r in group if r["model"] == model]
            errors = sorted(abs(r["error"]) for r in rr)
            q = {"codec": codec, "model": model, "n": len(rr), "hits": sum(r["converged"] for r in rr),
                 "median_abs_error": st.median(errors), "p95_abs_error": errors[math.ceil(.95*len(errors))-1],
                 "median_passes": st.median(r["passes"] for r in rr),
                 "median_loop_ms": 1000*st.median(r["loop_seconds"] for r in rr),
                 "median_score_ms": 1000*st.median(st.median(r["score_seconds"]) for r in rr),
                 "median_bytes": st.median(r["bytes"] for r in rr),
                 "undershoots_beyond_tolerance": sum(r["error"] < -inp["tolerance"] for r in rr),
                 "overshoots_beyond_tolerance": sum(r["error"] > inp["tolerance"] for r in rr),
                 "targets_outside_observed_score_range": sum(not min(p["score"] for p in r["probes"]) <= r["target"] <= max(p["score"] for p in r["probes"]) for r in rr),
                 "process_peak_rss_kib": max(r["process_peak_rss_kib"] for r in rr),
                 "legacy_three_pass_screen": "pass" if inp["max_iterations"] <= 3 and st.median(errors) <= 2 else "fail" if inp["max_iterations"] <= 3 else "outside_three_pass_gate",
                 "G_TARGET_qualification": "unmeasured: no per-image feasibility bounds; legacy screen superseded 2026-09-08"}
            result["summary"].append(q)
            text.append(f"| {codec} | {model} | {q['hits']}/{q['n']} | {q['median_abs_error']:.3f} | {q['p95_abs_error']:.3f} | {q['median_passes']:.1f} | {q['median_loop_ms']:.2f} | {q['median_score_ms']:.3f} |")
        # Each distinct reconstruction appears once in the RD comparison. Target
        # hit statistics above retain every preregistered requested target.
        distinct = {(r["source"], r["model"], r["encoded_sha256"]): r for r in group}
        cells, judges = [], {}
        for r in distinct.values():
            key = r["encoded"]
            cells.append({"image": str(r["source"]), "label": r["model"], "bytes": r["bytes"], "dist_path": key})
            judges[key] = {"ssim2": r["ssim2"], "butter": -r["butteraugli_pnorm3"], "zensim": r["fixed_b"]}
        saved = helpful(cells, args.baseline, judges, codec)
        rdtext.extend(["", f"## {codec}: diagnostic bytes at matched judge quality versus {args.baseline}",
                     "Out-of-range comparisons are omitted, never extrapolated.",
                     fmt_saved(saved, [m for m in models if m != args.baseline])])
        for (judge, model, cls), values in sorted(saved.items()):
            result["matched_judge"].append({"codec":codec,"judge":judge,"model":model,"class":cls,
                "n":len(values),"median_saved_fraction":st.median(values),"values":values})
    text.extend(rdtext)
    (root / "analysis_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (root / "analysis_summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def reachable_target_summary(root, inp, rows):
    """Summarize witnessed targets only, checking the entire declared matrix.

    The bound oracle never contributes controller probes or seed choices.
    Independent judge values are observations, not a matched-RD qualification.
    """
    import hashlib, json
    if inp["fit_calibration"]:
        raise SystemExit("calibration data cannot be summarized as steering evaluation")
    complete = json.loads((root / "COMPLETE").read_text())
    bounds = [json.loads(line) for line in (root / "bounds.jsonl").read_text().splitlines()]
    expected_bounds = {(s, c, m) for s in range(len(inp["source_manifest"]["sources"]))
                       for c in inp["codecs"] for m in inp["models"]}
    bound_keys = [(b["source"], b["codec"], b["model"]) for b in bounds]
    by_id = {b["id"]: b for b in bounds}
    if set(bound_keys) != expected_bounds or len(bound_keys) != len(expected_bounds) or len(by_id) != len(bounds):
        raise SystemExit("missing/duplicate/mismatched bound cells")
    expected = {(b["id"], t, k, p) for b in bounds for t in b["steering_targets"]
                for k in inp["budgets"] for p in inp["policies"]}
    keys = [(r["bound_id"], r["target"], r["pass_budget"], r["policy"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected or complete["measurements"] != len(rows):
        raise SystemExit("missing/duplicate/mismatched steering cells")
    for b in bounds:
        if len(b["probes"]) != inp["bound_steps"]:
            raise SystemExit("incomplete bound ladder")
        measured = [p["score"] for p in b["probes"]]
        if (min(measured), max(measured)) != (b["attained_min"], b["attained_max"]):
            raise SystemExit("bound extrema mismatch")
        if any(not any(abs(s-t) <= inp["tolerance"] + 1e-5 for s in measured) for t in b["steering_targets"]):
            raise SystemExit("unwitnessed target admitted to steering")
    for r in rows:
        b = by_id[r["bound_id"]]
        if (r["source"],r["codec"],r["model"]) != (b["source"],b["codec"],b["model"]):
            raise SystemExit("steering/bound identity mismatch")
        if r["target_status"] != "witnessed" or not 1 <= r["passes"] <= r["pass_budget"] or len(r["probes"]) != r["passes"]:
            raise SystemExit("invalid feasibility or pass accounting")
        if not all(math.isfinite(r[k]) for k in ("achieved","error","target","loop_seconds","ssim2","butteraugli_pnorm3")):
            raise SystemExit("nonfinite measurement")
        if abs(r["achieved"] - r["target"] - r["error"]) > 1e-4:
            raise SystemExit("error arithmetic mismatch")
    def stats(group):
        errors = sorted(abs(r["error"]) for r in group)
        return {"n":len(group), "sources":len({r["origin"] for r in group}),
                "families":len({r["family"] for r in group}),
                "median_abs_error":st.median(errors),
                "p95_abs_error":errors[math.ceil(.95*len(errors))-1], "worst_abs_error":max(errors),
                "median_signed_error":st.median(r["error"] for r in group),
                "hits":{str(t):sum(abs(r["error"]) <= t for r in group) for t in (.5,1.,2.)},
                "undershoots_beyond_tolerance":sum(r["error"] < -inp["tolerance"] for r in group),
                "median_passes":st.median(r["passes"] for r in group),
                "median_loop_ms":1000*st.median(r["loop_seconds"] for r in group),
                "median_bytes":st.median(r["bytes"] for r in group)}
    groups = defaultdict(list)
    by_class = defaultdict(list)
    by_position = defaultdict(list)
    for r in rows:
        key = (r["codec"],r["model"],r["pass_budget"],r["policy"])
        groups[key].append(r)
        by_class[(*key,r["content_class"])].append(r)
        b = by_id[r["bound_id"]]
        position = ("lower_endpoint" if abs(r["target"]-b["attained_min"]) <= 1e-5
                    else "upper_endpoint" if abs(r["target"]-b["attained_max"]) <= 1e-5 else "interior")
        by_position[(*key,position)].append(r)
    summary = [{"codec":c,"model":m,"budget":k,"policy":p,**stats(g)}
               for (c,m,k,p),g in sorted(groups.items())]
    classes = [{"codec":c,"model":m,"budget":k,"policy":p,"class":cl,**stats(g)}
               for (c,m,k,p,cl),g in sorted(by_class.items())]
    positions = [{"codec":c,"model":m,"budget":k,"policy":p,"position":pos,**stats(g)}
                 for (c,m,k,p,pos),g in sorted(by_position.items())]
    # Paired errors are aggregated within source family first. Repeated target
    # requests are not independent samples and there are too few families in a
    # smoke run to justify a universal uncertainty/qualification statement.
    paired = []
    indexed = {key:r for key,r in zip(keys,rows)}
    for c in inp["codecs"]:
        for m in inp["models"]:
            for k in inp["budgets"]:
                delta = defaultdict(list)
                for r in groups[(c,m,k,"train_curve")]:
                    base = indexed[(r["bound_id"],r["target"],k,"midpoint")]
                    delta[r["family"]].append(abs(r["error"])-abs(base["error"]))
                family_means = {f:st.mean(v) for f,v in delta.items()}
                paired.append({"codec":c,"model":m,"budget":k,
                    "family_mean_error_deltas":family_means,
                    "mean_family_delta":st.mean(family_means.values()),
                    "families_improved":sum(v<0 for v in family_means.values())})
    coverage = defaultdict(lambda:defaultdict(int))
    for b in bounds:
        for r in b["requests"]:
            coverage[(b["codec"],b["model"])][r["status"]] += 1
    result = {"schema":inp["schema"],"cells":len(rows),"bound_cells":len(bounds),
              "input_sha256":hashlib.sha256((root/"INPUTS.json").read_bytes()).hexdigest(),
              "measurement_sha256":hashlib.sha256((root/"measurements.jsonl").read_bytes()).hexdigest(),
              "bounds_sha256":hashlib.sha256((root/"bounds.jsonl").read_bytes()).hexdigest(),
              "summary":summary,"by_content_class":classes,"by_target_position":positions,"paired":paired,
              "fixed_request_coverage":[{"codec":c,"model":m,**counts} for (c,m),counts in sorted(coverage.items())],
              "qualification":"unqualified scalar experiment; native diffmap and full-range coverage remain separate",
              "uncertainty":"source-family paired descriptive deltas; no significance or universal tolerance claim"}
    text = ["# Reachable-target scalar steering", "",
            "Only witnessed attainable targets enter these errors. Fixed requests without a witness are counted separately.",
            "Bounds are offline evaluation work; controllers use only frozen train calibration and their own probes.", "",
            "| codec | model | budget | policy | n | median error | p95 error | hits ±1 | median ms |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        label = s["model"] if not s["model"].startswith("bake:") else s["model"][:17]
        text.append(f"| {s['codec']} | {label} | {s['budget']} | {s['policy']} | {s['n']} | {s['median_abs_error']:.3f} | {s['p95_abs_error']:.3f} | {s['hits']['1.0']} | {s['median_loop_ms']:.2f} |")
    text += ["", "No perceptual tolerance or native diffmap/RD qualification is established by this experiment."]
    (root/"analysis_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (root/"analysis_summary.md").write_text("\n".join(text)+"\n")
    print("\n".join(text))


def main():
    if "--target-loop" in sys.argv:
        return target_loop_main()
    judges = load_judges()
    out = ["# diffmap-RD probe analysis — auto-generated\n"]

    # ---- jxl ----
    jxl = []
    import glob
    for mf in glob.glob(f"{RD}/jxl/manifest_*.tsv"):
        with open(mf) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                jxl.append({
                    "image": r["image"], "label": r["label"],
                    "bytes": int(r["bytes"]), "dist_path": r["dist_path"],
                    "ms": float(r["encode_ms"]), "op": r["distance"],
                })
    if jxl:
        drivers = sorted({c["label"] for c in jxl} - {"none"})
        out.append("## jxl-encoder — bytes saved vs `none` baseline at equal judged score\n")
        out.append(fmt_saved(helpful(jxl, "none", judges, "jxl"), drivers))
        out.append("\n### efficiency (median encode_ms per image class)\n")
        eff = defaultdict(list)
        for c in jxl:
            eff[(c["label"], CLASS.get(c["image"], "photo"))].append(c["ms"])
        out.append("| driver | photo ms | screen ms |")
        out.append("|---|---|---|")
        for d in ["none", *drivers]:
            p = eff.get((d, "photo"), []); s = eff.get((d, "screen"), [])
            out.append(f"| {d} | {st.median(p):.0f} | {st.median(s):.0f} |" if p and s else f"| {d} | — | — |")

    # ---- zenjpeg ----
    zp = f"{RD}/zenjpeg/probe.tsv"
    if os.path.exists(zp):
        zj = []
        with open(zp) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                png = f"{RD}/zenjpeg/{r['driver']}__{r['image']}__t{int(float(r['target']))}.png"
                zj.append({
                    "image": r["image"], "label": r["driver"], "target": float(r["target"]),
                    "bytes": int(r["bytes"]), "achieved": float(r["achieved_score"]),
                    "passes": int(r["passes"]), "ms": float(r["encode_ms"]), "dist_path": png,
                })
        drivers = sorted({c["label"] for c in zj} - {"global"})
        out.append("\n## zenjpeg — bytes saved vs `global` baseline at equal judged score\n")
        out.append(fmt_saved(helpful(zj, "global", judges, "zenjpeg"), drivers))
        out.append("\n### efficiency + targeting residual (each driver on its OWN dial)\n")
        out.append("| driver | med passes | med ms | med abs(achieved−T) | n |")
        out.append("|---|---|---|---|---|")
        for d in sorted({c["label"] for c in zj}):
            v = [c for c in zj if c["label"] == d]
            res = [abs(c["achieved"] - c["target"]) for c in v
                   if not math.isnan(c["achieved"])]
            res_s = f"{st.median(res):.2f}" if res else "n/a (one-shot, no measure)"
            out.append(
                f"| {d} | {st.median([c['passes'] for c in v]):.0f} "
                f"| {st.median([c['ms'] for c in v]):.0f} "
                f"| {res_s} | {len(v)} |"
            )

    text = "\n".join(out) + "\n"
    with open(f"{RD}/analysis_summary.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
