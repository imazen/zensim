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
    if inp.get("schema") in ("native-jxl-target-v1", "native-codec-target-v1"):
        return native_target_summary(root, inp, rows)
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


def native_target_summary(root, inp, rows):
    """Validate native work accounting and reuse matched-judge interpolation.

    These are descriptive family-paired development measurements. Sparse
    witnessed coverage and interpolation do not confer product qualification.
    """
    import hashlib, json
    complete = json.loads((root / "COMPLETE").read_text())
    bound_rows = [json.loads(line) for line in (root / "bounds.jsonl").read_text().splitlines()]
    bounds = [r["bound"] for r in bound_rows]
    coverage = json.loads((root / "coverage.json").read_text())
    sources = {s["origin"]: s for s in inp["sources"]["sources"]}
    arms = inp["arms"]
    tol = inp["tolerance"]
    shared = inp["schema"] == "native-codec-target-v1"
    ladder_size = len(inp["quality_knots"]) if shared else 21
    codec = inp.get("codec", "jxl")
    if shared and codec != "avif":
        raise SystemExit("shared native accounting is only qualified for the AVIF adapter")
    ladders = defaultdict(list)
    for b in bounds:
        ladders[(b["origin"], b["arm"])].append(b)
    if (len(sources) != len(inp["sources"]["sources"])
            or set(ladders) != {(s, a) for s in sources for a in arms}
            or complete["bounds"] != len(bounds)):
        raise SystemExit("missing/duplicate/mismatched native bounds")
    for ladder in ladders.values():
        if (len(ladder) != ladder_size or len({b["knob"] for b in ladder}) != ladder_size
                or not all(math.isfinite(b["score"]) for b in ladder)):
            raise SystemExit("incomplete/nonfinite native ladder")
    if shared:
        for record in bound_rows:
            b, work = record["bound"], record["work"]
            n = inp["bound_encodes"][b["arm"]]
            if (len(work) != n or len(record["bound_probe_scores"]) != n
                    or b["seed_score"] != record["bound_probe_scores"][0]
                    or b["score"] != record["bound_probe_scores"][-1]
                    or b["knob"] not in inp["quality_knots"]
                    or any(w["knob"] != b["knob"] for w in work)):
                raise SystemExit("bound encode/seed-score accounting mismatch")
            count = int(b["arm"] != "scalar")
            for i, w in enumerate(work):
                if (w["internal_reconstructions"] != 0 or w["native_pixel_comparisons"] != count
                        or w["map_evaluations"] != count or not 0 <= w["consumed_maps"] <= int(i > 0 and b["arm"] == "active")):
                    raise SystemExit("bound map accounting mismatch")
            if work[-1]["encoded_sha256"] != b["encoded_sha256"]:
                raise SystemExit("bound emitted state mismatch")
    # Independent judges must see the pixels decoded by the measured backend.
    from PIL import Image
    def verify_pixels(file, expected):
        with Image.open(file) as image:
            if image.mode != "RGB" or hashlib.sha256(image.tobytes()).hexdigest() != expected:
                raise SystemExit("native decoded pixels changed")
    for (origin, arm), ladder in ladders.items():
        for i, b in enumerate(ladder):
            encoded = (root / f"{origin}-{arm}-{i}.{codec}").read_bytes()
            if len(encoded) != b["bytes"] or hashlib.sha256(encoded).hexdigest() != b["encoded_sha256"]:
                raise SystemExit("native bound bytes changed")
            verify_pixels(root / f"{origin}-{arm}-{i}.png", b["decoded_sha256"])
    admitted, coverage_keys, extrema = set(), set(), []
    coverage_counts = defaultdict(lambda: defaultdict(int))
    for c in coverage:
        key = (c["origin"], c["target"])
        if key in coverage_keys or c["origin"] not in sources:
            raise SystemExit("duplicate/unknown native coverage")
        coverage_keys.add(key)
        witnessed = []
        for arm in arms:
            witness = any(abs(b["score"] - c["target"]) <= tol
                          for b in ladders[(c["origin"], arm)])
            if witness != c[arm + "_witnessed"]:
                raise SystemExit("native coverage witness mismatch")
            scores = [b["score"] for b in ladders[(c["origin"], arm)]]
            status = ("witnessed" if witness else "outside_measured_envelope"
                      if c["target"] < min(scores) or c["target"] > max(scores)
                      else "unwitnessed_inside_envelope")
            coverage_counts[arm][status] += 1
            witnessed.append(witness)
        if all(witnessed):
            admitted.add(key)
    expected_coverage = set()
    for origin in sources:
        scores = sorted({b["score"] for b in ladders[(origin, "neutral")]})
        targets = set(inp["fixed_requests"]) | {scores[i * (len(scores)-1)//4] for i in range(5)}
        expected_coverage.update((origin, t) for t in targets)
        for arm in arms:
            ss = [b["score"] for b in ladders[(origin, arm)]]
            extrema.append({"origin":origin, "arm":arm, "attained_min":min(ss), "attained_max":max(ss)})
    if coverage_keys != expected_coverage:
        raise SystemExit("missing native coverage requests")
    expected = {(s, t, a, p, k) for s, t in admitted for a in arms
                for p in inp["policies"] for k in inp["budgets"]}
    keys = [(r["origin"], r["target"], r["arm"], r["policy"], r["budget"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected or complete["cases"] != len(rows):
        raise SystemExit("missing/duplicate/mismatched native steering cells")
    for r in rows:
        n = r["full_encodes"]
        if (not 1 <= n <= r["budget"] or n != len(r["probes"])
                or n != len(r["native_work"]) or n != r["search_pixel_comparisons"]
                or r["terminal_decodes"] != 1 or r["terminal_pixel_comparisons"] != 1):
            raise SystemExit("native complete-encode accounting mismatch")
        for i, (probe, work) in enumerate(zip(r["probes"], r["native_work"])):
            count = 0 if r["arm"] == "scalar" else (1 if shared else 3)
            recon = 0 if shared else count
            if (probe["knob"] != work["knob"] or work["internal_reconstructions"] != recon
                    or work["native_pixel_comparisons"] != count or work["map_evaluations"] != count
                    or (shared and not 0 <= work["consumed_maps"] <= int(i > 0 and r["arm"] == "active"))):
                raise SystemExit("native map/reconstruction engagement mismatch")
        if (not all(math.isfinite(r[k]) for k in ("target", "achieved", "signed_error", "total_seconds"))
                or abs(r["achieved"] - r["target"] - r["signed_error"]) > 1e-4
                or not any(w["encoded_sha256"] == r["encoded_sha256"]
                           and abs(p["score"] - r["achieved"]) <= 1e-5
                           for p, w in zip(r["probes"], r["native_work"]))):
            raise SystemExit("native terminal score/selection mismatch")
        encoded = (root / r["bitstream"]).read_bytes()
        if len(encoded) != r["bytes"] or hashlib.sha256(encoded).hexdigest() != r["encoded_sha256"]:
            raise SystemExit("native emitted bytes changed")
        selected = next(w for w in r["native_work"] if w["encoded_sha256"] == r["encoded_sha256"])
        verify_pixels(root / r["decoded"], selected["decoded_sha256"])
    def stats(group):
        errors = sorted(abs(r["signed_error"]) for r in group)
        return {"n":len(group), "families":len({r["family"] for r in group}),
                "median_abs_error":st.median(errors), "p95_abs_error":errors[math.ceil(.95*len(errors))-1],
                "worst_abs_error":max(errors), "hits":{str(t):sum(e <= t for e in errors) for t in (.25,.5,1.,2.)},
                "undershoots_beyond_tolerance":sum(r["signed_error"] < -tol for r in group),
                "mean_full_encodes":st.mean(r["full_encodes"] for r in group),
                "median_total_ms":1000*st.median(r["total_seconds"] for r in group),
                "median_bytes":st.median(r["bytes"] for r in group),
                "full_encodes":sum(r["full_encodes"] for r in group),
                "native_reconstructions":sum(w["internal_reconstructions"] for r in group for w in r["native_work"]),
                "native_pixel_comparisons":sum(w["native_pixel_comparisons"] for r in group for w in r["native_work"]),
                "map_evaluations":sum(w["map_evaluations"] for r in group for w in r["native_work"]),
                "consumed_non_neutral_maps":sum(w.get("consumed_maps",0) for r in group for w in r["native_work"]),
                "search_pixel_comparisons":sum(r["search_pixel_comparisons"] for r in group),
                "terminal_decodes_and_comparisons":len(group),
                "process_peak_rss_kib":max(r["process_peak_rss_kib"] for r in group)}
    groups, classes = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["arm"], r["policy"], r["budget"])
        groups[key].append(r)
        classes[(*key, r["class"])].append(r)
    summary = [{"arm":a,"policy":p,"budget":k,**stats(g)} for (a,p,k),g in sorted(groups.items())]
    by_class = [{"arm":a,"policy":p,"budget":k,"class":c,**stats(g)} for (a,p,k,c),g in sorted(classes.items())]
    index = dict(zip(keys, rows))
    paired = []
    for base in ("scalar", "neutral"):
        for budget in inp["budgets"]:
            family = defaultdict(list)
            for r in groups[("active", "train_curve", budget)]:
                b = index[(r["origin"],r["target"],base,"train_curve",budget)]
                family[r["family"]].append(abs(r["signed_error"])-abs(b["signed_error"]))
            paired.append({"baseline":base,"budget":budget,"family_mean_abs_error_delta":{f:st.mean(v) for f,v in family.items()}})
    # Judge the exact reference/decoded pairs, including unselected search arms.
    expected_pairs = {str(root / f"{origin}-{arm}-{i}.png"): sources[origin]["path"]
                      for (origin, arm), ladder in ladders.items() for i, _ in enumerate(ladder)}
    expected_pairs.update({str(root / r["decoded"]): sources[r["origin"]]["path"] for r in rows})
    judges = {}
    for name, file, col, sign in (("ssim2","judge_ssim2.tsv","ssim2",1),
                                  ("butter","judge_butteraugli.tsv","butteraugli_pnorm3",-1)):
        if not (root / file).exists():
            continue
        seen_pairs = set()
        with (root / file).open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                path = r["dist_path"]
                if (path not in expected_pairs or path in seen_pairs
                        or r["ref_path"] != expected_pairs[path]):
                    raise SystemExit("independent judge pair identity mismatch")
                seen_pairs.add(path)
                value = float(r[col])
                if not math.isfinite(value):
                    raise SystemExit("nonfinite independent judge")
                judges.setdefault(r["dist_path"], {})[name] = sign * value
        if seen_pairs != set(expected_pairs):
            raise SystemExit("incomplete independent judge pair matrix")
    CLASS.update({s: v["content_class"] for s,v in sources.items()})
    cells = []
    for (origin, arm), ladder in ladders.items():
        for i,b in enumerate(ladder):
            cells.append({"image":origin,"label":arm,"bytes":b["bytes"],"dist_path":str(root/f"{origin}-{arm}-{i}.png")})
    seen = set()
    for r in rows:
        if r["policy"] != "train_curve":
            continue
        key = (r["origin"],r["arm"],r["budget"],r["encoded_sha256"])
        if key not in seen:
            seen.add(key)
            cells.append({"image":r["origin"],"label":f"{r['arm']}-target-{r['budget']}",
                          "bytes":r["bytes"],"dist_path":str(root/r["decoded"])})
    if judges and any(set(judges.get(c["dist_path"], {})) != {"ssim2","butter"} for c in cells):
        raise SystemExit("incomplete independent judge matrix")
    rd = []
    for baseline in ("scalar", "neutral"):
        for (judge, label, cls), values in sorted(helpful(cells, baseline, judges, codec).items()):
            rd.append({"baseline":baseline,"judge":judge,"arm":label,"class":cls,"n":len(values),
                       "median_saved_fraction":st.median(values),"mean_saved_fraction":st.mean(values),"values":values})
    result = {"schema":inp["schema"],"cells":len(rows),"bound_cells":len(bounds),
              "input_sha256":hashlib.sha256((root/"INPUTS.json").read_bytes()).hexdigest(),
              "measurement_sha256":hashlib.sha256((root/"measurements.jsonl").read_bytes()).hexdigest(),
              "summary":summary,"by_content_class":by_class,"paired":paired,"attained_bounds":extrema,
              "requested_targets":len(coverage),"jointly_witnessed_targets":len(admitted),
              "coverage_by_arm":dict(coverage_counts),
              "negative_witnessed_targets":sum(t < 0 for _,t in admitted),"matched_judge":rd,
              "offline_validation_bound_cost":{"full_encodes":sum(len(r["work"]) for r in bound_rows),"scalar_comparisons":sum(len(r["work"]) for r in bound_rows),
                  "internal_reconstructions":sum(w["internal_reconstructions"] for r in bound_rows for w in r["work"]),
                  "native_pixel_comparisons":sum(w["native_pixel_comparisons"] for r in bound_rows for w in r["work"]),
                  "map_evaluations":sum(w["map_evaluations"] for r in bound_rows for w in r["work"])},
              "qualification":"unqualified native development screen; sparse coverage, eight families, no universal perceptual tolerance",
              "cost_scope":"total ms = search plus terminal decode/score, excludes offline bounds/calibration and output PNG writes; map finite feature probes are not extra pixel comparisons",
              "rd_scope":"sparse per-image log-byte interpolation without extrapolation; deduplicated target outputs, independent judges; direct matched-quality confirmation required"}
    text = [f"# Native {codec.upper()} target steering", "", f"{len(admitted)}/{len(coverage)} targets witnessed across all arms; {len(rows)} cases on {len(sources)} validation families.", "",
            "| arm | policy | budget | median error | p95 error | hits ±1 | mean encodes | median total ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        text.append(f"| {s['arm']} | {s['policy']} | {s['budget']} | {s['median_abs_error']:.3f} | {s['p95_abs_error']:.3f} | {s['hits']['1.0']}/{s['n']} | {s['mean_full_encodes']:.2f} | {s['median_total_ms']:.2f} |")
    if rd:
        text += ["", "Independent matched-quality byte savings versus the scalar bound ladder (positive is smaller).",
                 "These sparse interpolations include a sampling effect: even an identical scalar output at a new knob can differ from the ladder interpolation. They do not isolate the causal map effect.", "",
                 "| active output | judge | content | points | median byte savings |",
                 "|---|---|---|---:|---:|"]
        for r in rd:
            if r["baseline"] == "scalar" and r["arm"] in ("active", "active-target-3"):
                text.append(f"| {r['arm']} | {r['judge']} | {r['class']} | {r['n']} | {100*r['median_saved_fraction']:.3f}% |")
    text += ["", result["qualification"], "", result["cost_scope"], "", result["rd_scope"]]
    (root/"analysis_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (root/"analysis_summary.md").write_text("\n".join(text)+"\n")
    print("\n".join(text))


def interventions_main():
    """Validate and summarize the registered native finite-quantizer experiment."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--interventions", type=Path, required=True)
    root = ap.parse_args().interventions.resolve()
    if not __debug__:
        raise SystemExit("intervention validation requires Python assertions enabled")
    inp = json.loads((root / "INPUTS.json").read_text())
    done = json.loads((root / "COMPLETE.json").read_text())
    assert inp["schema"] in ("native-jxl-interventions-v1", "native-jxl-interventions-v2")
    native_png = inp["schema"] == "native-jxl-interventions-v2"
    coarse = native_png and inp["region_mode"] == "coarse4"
    factors = [("up",1.2),("down",0.8)] if coarse else [("up",1.1),("down",0.9)]
    if native_png:
        assert inp["png_io"] == "zenpng-0.1.4-packed-opaque-rgb8-v1"
        assert inp["region_mode"] in ("transform", "coarse4")
        assert np.allclose(inp["raw_q_factors"], [factors[1][1], factors[0][1]], rtol=0, atol=1e-6)
    else:
        # Historical v1 artifacts only. New runs never use foreign image IO.
        from PIL import Image
    assert inp["model_sha256"] == "cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6"
    rows = [json.loads(line) for line in (root / "measurements.jsonl").read_text().splitlines()]
    assert len(rows) == done["full_encodes"] == done["independent_decodes"] == done["ordinary_scalar_pixel_comparisons"]
    sources = {s["origin"]: s for s in inp["sources"]}
    assert len(sources) == len(inp["sources"]) == 4
    assert len({s["family"] for s in sources.values()}) == 4
    assert len({s["content_class"] for s in sources.values()}) == 4
    sha = lambda b: hashlib.sha256(b).hexdigest()
    for source in sources.values():
        assert source["split"] == "train" and sha(Path(source["path"]).read_bytes()) == source["sha256"]
    groups = defaultdict(list)
    pixels, requested, actual, expected_pairs = {}, {}, {}, {}
    for row in rows:
        source = sources[row["origin"]]
        assert row["class"] == source["content_class"]
        key = (row["origin"], row["distance"])
        assert key[1] in (1., 3.)
        cell = root / f"o_{key[0]}-d{key[1]:g}"
        assert Path(row["cell"]).resolve() == cell
        work = row["work"]
        name = work["name"]
        assert Path(name).name == name
        identity = (*key, name)
        assert identity not in pixels
        width, height = row["width"], row["height"]
        rgb = (cell / f"{name}.rgb8").read_bytes()
        assert len(rgb) == width * height * 3 and sha(rgb) == work["decoded_sha256"]
        if native_png:
            assert sha((cell / f"{name}.png").read_bytes()) == work["png_sha256"]
            assert work["png_readback_rgb_sha256"] == sha(rgb)
            assert work["png_roundtrip_decodes"] == 1
            assert math.isfinite(work["png_roundtrip_seconds"]) and work["png_roundtrip_seconds"] >= 0
        else:
            with Image.open(cell / f"{name}.png") as im:
                assert im.mode == "RGB" and im.size == (width, height) and im.tobytes() == rgb
        encoded = (cell / f"{name}.jxl").read_bytes()
        assert len(encoded) == work["bytes"] and sha(encoded) == work["encoded_sha256"]
        for field, suffix, cache in [("requested_q_sha256", "requested-q.u8", requested),
                                     ("actual_q_sha256", "actual-q.u8", actual)]:
            value = (cell / f"{name}.{suffix}").read_bytes()
            assert len(value) == ((width+7)//8) * ((height+7)//8)
            assert min(value) >= 1 and sha(value) == work[field]
            cache[identity] = np.frombuffer(value, dtype=np.uint8)
        pixels[identity] = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
        assert all(math.isfinite(work[k]) for k in ["score", "scale", "inv_scale", "encode_seconds", "decode_seconds", "score_seconds"])
        assert all(work[k] >= 0 for k in ["encode_seconds", "decode_seconds", "score_seconds"])
        assert work["full_encodes"] == work["independent_decodes"] == work["scalar_pixel_comparisons"] == 1
        assert work["internal_reconstructions"] == work["map_evaluations"] == 0
        path = str(cell / f"{name}.png")
        expected_pairs[path] = {"ref_path": source["path"], "dist_path": path,
            "origin": key[0], "arm": name, "bytes": str(work["bytes"])}
        groups[key].append(row)
    assert len(groups) == len(done["cells"]) == done["additional_scored_maps"] == 8
    assert set(groups) == {(o, d) for o in sources for d in (1., 3.)}
    completion = {(v["origin"], v["distance"]): v for v in done["cells"]}
    assert len(completion) == 8
    assert done["internal_reconstructions"] == 0
    if native_png:
        assert done["source_png_decodes"] == 4 and done["png_roundtrip_decodes"] == len(rows)
        for origin, source in sources.items():
            record = json.loads((root/f"source-{origin}.json").read_text())
            assert record["source"] == source and record["source_decodes"] == 1
            rgb = (root/f"source-{origin}.rgb8").read_bytes()
            assert sha(rgb) == record["decoded_sha256"]
            assert len(rgb) == record["width"] * record["height"] * 3
            assert all((r["width"],r["height"]) == (record["width"],record["height"])
                       for r in rows if r["origin"] == origin)
            assert math.isfinite(record["source_decode_seconds"]) and record["source_decode_seconds"] >= 0
    compatibility = json.loads((root / "COMPATIBILITY.json").read_text())
    assert compatibility["version_stdout"].startswith("djxl v0.12.")
    expected_compatibility = {(o, d, n) for o, d in groups
                              for n in ("baseline", "r0-down", "r0-up")}
    assert len(compatibility["decodes"]) == done["libjxl_compatibility_decodes"] == len(expected_compatibility)
    compat_index = {str(root/f"o_{o}-d{d:g}"/f"{n}.jxl"): (o, d, n)
                    for o, d, n in expected_compatibility}
    seen_compatibility = set()
    for record in compatibility["decodes"]:
        identity = compat_index[record["encoded"]]
        assert identity not in seen_compatibility
        seen_compatibility.add(identity)
        assert sha(Path(record["encoded"]).read_bytes()) == record["encoded_sha256"]
        o, d, n = identity
        decoded = root/"compatibility"/f"{o}-d{d:g}-{n}.png"
        assert record["decoded"] == str(decoded)
        primary = pixels[identity]
        if native_png:
            assert done["compatibility_png_decodes"] == 2*len(expected_compatibility)
            assert record["native_png_decodes"] == 2
            assert sha(decoded.read_bytes()) == record["png_sha256"]
            raw = decoded.with_suffix(".rgb8").read_bytes()
            assert len(raw) == primary.size and sha(raw) == record["decoded_sha256"]
            assert (record["width"],record["height"]) == (primary.shape[1],primary.shape[0])
            independent = np.frombuffer(raw,dtype=np.uint8).reshape(primary.shape)
        else:
            with Image.open(decoded) as im:
                assert im.mode == "RGB" and im.size == (primary.shape[1], primary.shape[0])
                assert sha(im.tobytes()) == record["decoded_sha256"]
                independent = np.array(im)
        difference = int(np.max(np.abs(independent.astype(int) - primary.astype(int))))
        assert difference == record["primary_max_abs_rgb8_difference"]
    with (root / "judge_pairs.tsv").open() as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))
    assert len(pairs) == len(expected_pairs) and len({p["dist_path"] for p in pairs}) == len(pairs)
    assert all(p == expected_pairs[p["dist_path"]] for p in pairs)
    judges = {}
    for metric, column, sign in [("ssim2", "ssim2", 1), ("butteraugli", "butteraugli_pnorm3", -1)]:
        with (root / f"judge_{metric}.tsv").open() as f:
            panel = list(csv.DictReader(f, delimiter="\t"))
        assert len(panel) == len(rows) and len({r["dist_path"] for r in panel}) == len(rows)
        assert {r["dist_path"] for r in panel} == set(expected_pairs)
        for r in panel:
            assert all(r[k] == v for k, v in expected_pairs[r["dist_path"]].items())
            assert math.isfinite(float(r[column]))
        judges[metric] = {r["dist_path"]: sign * float(r[column]) for r in panel}

    def corr(xs, ys):
        if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
            return None
        value = float(spearmanr(xs, ys).statistic)
        assert math.isfinite(value)
        return value

    results, samples = [], []
    for key, rr in sorted(groups.items()):
        cell = root / f"o_{key[0]}-d{key[1]:g}"
        by_name = {r["work"]["name"]: r for r in rr}
        assert len(by_name) == len(rr) == completion[key]["full_encodes"]
        base, neutral = by_name["baseline"], by_name["neutral"]
        for field in ["encoded_sha256", "decoded_sha256", "actual_q_sha256", "requested_q_sha256", "score", "global_scale", "scale", "inv_scale"]:
            assert base["work"][field] == neutral["work"][field]
        width, height = base["width"], base["height"]
        bx = (width+7)//8
        regions = json.loads((cell / "regions.json").read_text())
        assert len(regions) == completion[key]["transform_regions"]
        covered_image = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            x0, y0 = region["x"]*8, region["y"]*8
            x1, y1 = min(width, x0+region["blocks_x"]*8), min(height, y0+region["blocks_y"]*8)
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            assert region["area"] == (x1-x0)*(y1-y0)
            assert math.isfinite(region["map_mass"]) and math.isfinite(region["map_density"])
            assert abs(region["map_density"] * region["area"] - region["map_mass"]) < 1e-10
            covered_image[y0:y1, x0:x1] += 1
        assert np.all(covered_image == 1)
        transforms = regions
        if coarse:
            regions = json.loads((cell/"allocation_regions.json").read_text())
            expected = defaultdict(list)
            by = (height+7)//8
            for i,r in enumerate(transforms):
                expected[(4*r["y"]//by)*4+4*r["x"]//bx].append(i)
            assert len(regions) == len(expected)
            for region,(grid_cell,members) in zip(regions,sorted(expected.items())):
                assert region["grid_cell"] == grid_cell and region["transform_indices"] == members
                assert region["area"] == sum(transforms[i]["area"] for i in members)
                assert abs(region["map_mass"]-sum(transforms[i]["map_mass"] for i in members)) < 1e-10
                assert abs(region["map_density"]*region["area"]-region["map_mass"]) < 1e-10
            count = len(regions)
            indices = list(range(count))
        else:
            count = min(16, len(regions))
            indices = [i*(len(regions)-1)//max(1,count-1) for i in range(count)]
        assert completion[key]["sampled_regions"] == count
        assert set(by_name) == {"baseline", "neutral"} | {f"r{i}-{side}" for i in indices for side in ["up", "down"]}
        central, cell_samples = [], []
        baseline_id = (*key, "baseline")
        for index in indices:
            region = regions[index]
            members = [transforms[i] for i in region["transform_indices"]] if coarse else [region]
            covered = {y*bx+x for r in members for y in range(r["y"],r["y"]+r["blocks_y"])
                       for x in range(r["x"],r["x"]+r["blocks_x"])}
            mask = np.array([i in covered for i in range(len(actual[baseline_id]))])
            pixel_mask = np.zeros((height,width),dtype=bool)
            for r in members:
                x0,y0 = r["x"]*8,r["y"]*8
                x1,y1 = min(width,x0+r["blocks_x"]*8),min(height,y0+r["blocks_y"]*8)
                pixel_mask[y0:y1,x0:x1] = True
            assert int(pixel_mask.sum()) == region["area"]
            pair = {}
            for side, factor in factors:
                name = f"r{index}-{side}"
                row, identity = by_name[name], (*key, name)
                work, intervention = row["work"], row["intervention"]
                assert intervention["region_index"] == index and intervention["region"] == region
                assert abs(intervention["factor"]-factor) < 1e-6
                for field in ["global_scale", "scale", "inv_scale"]:
                    assert work[field] == base["work"][field]
                assert np.array_equal(requested[identity][~mask], requested[baseline_id][~mask])
                delta = requested[identity].astype(int) - requested[baseline_id].astype(int)
                assert np.all(delta[mask] >= 0) if side == "up" else np.all(delta[mask] <= 0)
                if native_png:
                    old = requested[baseline_id].astype(int)
                    wanted = np.floor(old.astype(np.float32)*np.float32(factor)+np.float32(0.5)).astype(int)
                    wanted = np.maximum(wanted,old+1) if side == "up" else np.minimum(wanted,old-1)
                    wanted = np.clip(wanted,1,255)
                    assert np.array_equal(requested[identity][mask],wanted[mask])
                changed = actual[identity] != actual[baseline_id]
                assert intervention["changed_inside"] == int(np.sum(changed & mask))
                assert intervention["changed_outside"] == int(np.sum(changed & ~mask))
                log_change = float(np.mean(np.log(actual[identity][mask].astype(float) / actual[baseline_id][mask])))
                assert abs(log_change-intervention["mean_log_actual_q_change"]) < 1e-12
                score_change = work["score"]-base["work"]["score"]
                assert abs(score_change-intervention["score_change"]) < 1e-5
                assert work["bytes"]-base["work"]["bytes"] == intervention["byte_change"]
                changed_pixels = np.any(pixels[identity] != pixels[baseline_id], axis=2)
                inside = int(np.sum(changed_pixels & pixel_mask))
                path, baseline_path = str(cell/f"{name}.png"), str(cell/"baseline.png")
                sample = {"origin":key[0],"distance":key[1],"region_index":index,"side":side,
                    **intervention,"native_score_delta":score_change,"byte_delta":intervention["byte_change"],
                    "pixels_changed_inside":inside,"pixels_changed_outside":int(np.sum(changed_pixels))-inside,
                    **{metric+"_quality_delta":panel[path]-panel[baseline_path] for metric,panel in judges.items()}}
                pair[side] = sample
                cell_samples.append(sample)
            span = pair["up"]["mean_log_actual_q_change"] - pair["down"]["mean_log_actual_q_change"]
            if span > 1e-12:
                derivative = {"region_index":index,"map_mass":region["map_mass"],"map_density":region["map_density"],
                    **{field: (pair["up"][field]-pair["down"][field])/span for field in
                       ["native_score_delta","byte_delta","ssim2_quality_delta","butteraugli_quality_delta"]}}
                central.append(derivative)
        summary = {"origin":key[0],"class":sources[key[0]]["content_class"],"distance":key[1],
            "interventions":len(cell_samples),"central_samples":len(central),
            "actual_quantizer_inert":sum(s["changed_inside"]+s["changed_outside"] == 0 for s in cell_samples),
            "pixel_inert":sum(s["pixels_changed_inside"]+s["pixels_changed_outside"] == 0 for s in cell_samples),
            "captured_quantizer_inert_but_pixels_changed":sum(
                s["changed_inside"]+s["changed_outside"] == 0
                and s["pixels_changed_inside"]+s["pixels_changed_outside"] > 0 for s in cell_samples),
            "quantizer_changes_outside_region":sum(s["changed_outside"] > 0 for s in cell_samples),
            "pixel_changes_outside_region":sum(s["pixels_changed_outside"] > 0 for s in cell_samples),
            "nonpositive_central_byte_derivatives":sum(c["byte_delta"] <= 0 for c in central),
            "rank_associations":{},"direction":{}}
        for metric in ["native_score_delta","ssim2_quality_delta","butteraugli_quality_delta"]:
            signs = [s[metric] * (1 if s["mean_log_actual_q_change"] > 0 else -1)
                     for s in cell_samples if abs(s["mean_log_actual_q_change"]) > 1e-12]
            eps = 1e-5 if metric == "native_score_delta" else 1e-6
            summary["direction"][metric] = {"positive":sum(v>eps for v in signs),
                "negative":sum(v < -eps for v in signs),"flat":sum(abs(v)<=eps for v in signs),"epsilon":eps}
            for predictor in ["map_mass","map_density"]:
                summary["rank_associations"][predictor+"_vs_"+metric] = corr(
                    [c[predictor] for c in central], [c[metric] for c in central])
                positive_rate = [c for c in central if c["byte_delta"] > 0]
                summary["rank_associations"][predictor+"_vs_"+metric+"_per_byte"] = corr(
                    [c[predictor] for c in positive_rate], [c[metric]/c["byte_delta"] for c in positive_rate])
        summary["central_differences"] = central
        results.append(summary)
        samples.extend(cell_samples)
    result = {"schema":"native-jxl-interventions-analysis-v2" if native_png else "native-jxl-interventions-analysis-v1","inputs":inp,"work":done,
        "compatibility":compatibility,
        "judge_pairs_per_metric":len(rows),"verified_neutral_cells":len(groups),
        "cells":results,"interventions":samples,
        "conclusion":"Mechanism evidence only; no model qualification, target-attainment or matched-RD claim."}
    text = ["# Native JXL finite-block interventions", "",
        "All encoded/pixel/quantizer hashes, neutral repeats, region/probe coverage and independent judge identities verified.", "",
        "Central differences divide the up-minus-down score/byte difference by actual mean log-quantizer span. Butteraugli is negated so positive means quality improvement.", "",
        "| Origin | Class | Distance | D direction positive/negative/flat | Mass vs D derivative | Density vs D gain/byte |", "|---|---|---:|---:|---:|---:|"]
    for cell in results:
        direction = cell["direction"]["native_score_delta"]
        values = cell["rank_associations"]
        fmt = lambda v: "undefined" if v is None else f"{v:.3f}"
        text.append(f"| {cell['origin']} | {cell['class']} | {cell['distance']:g} | {direction['positive']}/{direction['negative']}/{direction['flat']} | {fmt(values['map_mass_vs_native_score_delta'])} | {fmt(values['map_density_vs_native_score_delta_per_byte'])} |")
    text += ["", result["conclusion"], "Flat quantizers, nonpositive byte derivatives and changes outside the selected region remain explicit in JSON. Correlations describe this fixed training-family screen; they are not release gates."]
    (root/"analysis_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (root/"analysis_summary.md").write_text("\n".join(text)+"\n")
    print("\n".join(text))


def main():
    if "--interventions" in sys.argv:
        return interventions_main()
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
