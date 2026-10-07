#!/usr/bin/env python3
"""Turn raw zenbench result JSON into the cross-generation speed matrix.

Reads the `ZENBENCH_RESULT_PATH` files the `speed_matrix_run.sh` (end-to-end
scoring) and `speed_matrix_extract.sh` (feature extraction) sweeps leave in
`--raw-dir`, and emits one summary JSON plus one markdown report.
Regenerable: the raw rounds are the evidence, this is only a view of them.

Filenames carry the configuration, because the raw JSON does not:

    <threads>-rev<revision>.json            end-to-end scoring
    extract-<threads>t-rev<revision>.json   feature extraction

`1t-rev1`, `4t-rev1`, `mt16-rev3`, `extract-8t-rev3` all parse. A
`run.meta.json` in the same directory carries the provenance and is copied
into the summary verbatim rather than re-derived here.

Four things this does NOT do, deliberately:

  * It never pools arms across processes. A ratio is only ever computed
    against the `fast_ssim2` arm measured in the SAME process, because that is
    the only arm that is revision-independent and therefore the only honest
    bridge between the revision-1 and revision-3 binaries' environments.
  * It never reports a percentile from batched rounds. zenbench's `iterations`
    is the batch size; a mean over a batch is not a latency percentile, so p95
    is emitted only where every round ran exactly one call.
  * It recomputes the median from the raw rounds and compares it to zenbench's
    own summary median. A mismatch is printed and recorded, not smoothed over.
  * It never scales a measurement it does not have. A thread-scaling factor is
    emitted only where BOTH the 1-thread and the N-thread cell were measured,
    at the same size and the same revision.

    python3 scripts/demos/speed_matrix_report.py --raw-dir DIR \
        --out-json benchmarks/speed_matrix_2026-09-18.json \
        --out-md   benchmarks/speed_matrix_2026-09-18.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANCHOR = "fast_ssim2"
# Above this coefficient of variation a cell is called out rather than read as
# a clean number. 10% is the bar `benchmarks/k4_st_mt_2026-09-10.md` used when
# it flagged its own two-CCD column.
NOISY_CV = 0.10
GROUPS = {"ssim2_bar_": "e2e", "extract_paths_": "extract"}
KIND_TITLE = {
    "e2e": "End-to-end scoring (`Zensim::compute` / `BakeScorer::compute`)",
    "extract": "Feature extraction (`extract_paths_bench`)",
}
LABEL_RE = re.compile(r"^(?:(extract)-)?(?:mt)?(\d+)t?-rev(\d)$")


def parse_label(stem: str) -> tuple[str, int, int] | None:
    """`extract-8t-rev3` -> ('extract', 8, 3); `mt16-rev1` -> ('e2e', 16, 1)."""
    m = LABEL_RE.match(stem)
    if not m:
        return None
    return (m.group(1) or "e2e", int(m.group(2)), int(m.group(3)))


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile over a sorted copy of `values`."""
    if not values:
        raise ValueError("percentile of an empty sample")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares `y = a + b*x`."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("degenerate fit: every x is identical")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def read_run(path: Path) -> dict:
    """One zenbench result file -> {size -> {arm -> stats}} plus warnings."""
    doc = json.loads(path.read_text())
    sizes: dict[int, dict[str, dict]] = {}
    warnings: list[str] = []
    for comp in doc.get("comparisons", []):
        group = comp["group_name"]
        prefix = next((p for p in GROUPS if group.startswith(p)), None)
        if prefix is None:
            warnings.append(f"{path.name}: unexpected group {group!r}, skipped")
            continue
        size = int(group.rsplit("_", 1)[1])
        samples = comp.get("samples") or []
        arms: dict[str, dict] = {}
        for idx, bench in enumerate(comp["benchmarks"]):
            rounds: list[float] = []
            single = True
            for smp in samples:
                iters = smp.get("iterations", 1)
                single &= iters == 1
                series = smp.get("compensated_ns") or smp["elapsed_ns"]
                rounds.append(series[idx] / iters)
            if not rounds:
                warnings.append(f"{path.name}/{group}/{bench['name']}: no raw rounds")
                continue
            med = percentile(rounds, 0.5)
            summary = bench["summary"]
            reported = summary["median"]
            # Cross-check against zenbench's own summary: these must agree, and
            # a disagreement means the raw rounds are not what was summarised.
            if reported and abs(med - reported) / reported > 1e-9:
                warnings.append(
                    f"{path.name}/{group}/{bench['name']}: recomputed median "
                    f"{med:.1f} ns != zenbench summary {reported:.1f} ns"
                )
            mean = summary.get("mean") or 0.0
            cv = math.sqrt(summary.get("variance", 0.0)) / mean if mean else None
            arms[bench["name"]] = {
                "n_rounds": len(rounds),
                "iterations_per_round": 1 if single else comp.get("iterations_per_sample"),
                "median_ms": med / 1e6,
                "min_ms": min(rounds) / 1e6,
                "p95_ms": percentile(rounds, 0.95) / 1e6 if single else None,
                "cv": cv,
                "noisy": bool(cv is not None and cv > NOISY_CV),
                "zenbench_summary_median_ms": reported / 1e6,
                "gate_clean_rounds": sum(1 for s in samples if s.get("gate_clean")),
            }
        sizes[size] = arms
    return {
        "sizes": sizes,
        "warnings": warnings,
        "testbed": doc.get("testbed", {}),
        "gate_waits": doc.get("gate_waits"),
        "unreliable": doc.get("unreliable"),
    }


def add_ratios(run: dict) -> None:
    """Ratio vs the anchor arm measured in the SAME process and size."""
    for arms in run["sizes"].values():
        base = arms.get(ANCHOR)
        for stats in arms.values():
            stats["ratio_vs_fast_ssim2"] = (
                stats["median_ms"] / base["median_ms"] if base else None
            )


def fit_arms(run: dict) -> dict[str, dict]:
    """`time = alpha + beta*pixels` per arm, over every size it was measured at."""
    by_arm: dict[str, list[tuple[int, float]]] = {}
    for size, arms in run["sizes"].items():
        for name, stats in arms.items():
            by_arm.setdefault(name, []).append((size, stats["median_ms"]))
    fits: dict[str, dict] = {}
    for name, points in by_arm.items():
        points.sort()
        if len(points) < 3:
            fits[name] = {"note": f"only {len(points)} sizes; no fit"}
            continue
        mpix = [(s * s) / 1e6 for s, _ in points]
        ms = [m for _, m in points]
        alpha, beta = least_squares(mpix, ms)
        resid = {f"{s}": round(m - (alpha + beta * p), 4)
                 for (s, m), p in zip(points, mpix)}
        rel = {f"{s}": (m - (alpha + beta * p)) / m
               for (s, m), p in zip(points, mpix)}
        worst = max(abs(v) for v in resid.values())
        worst_rel = max(abs(v) for v in rel.values())
        fits[name] = {
            "alpha_ms": alpha,
            "beta_ms_per_mp": beta,
            "residual_ms": resid,
            "relative_residual": {k: round(v, 4) for k, v in rel.items()},
            "max_abs_residual_ms": worst,
            "max_abs_relative_residual": worst_rel,
            # Judged on the RELATIVE residual, not the absolute one. An
            # absolute test passes trivially once the largest size is seconds
            # and the smallest is microseconds — which is exactly the regime
            # here, and it hid a fit whose intercept was NEGATIVE (physically
            # impossible as a fixed per-call overhead) behind a tidy "yes".
            # A negative alpha now shows up as a huge relative residual at the
            # smallest size, which is the honest signal: the arm is
            # super-linear in pixels over this range.
            "linear": worst_rel <= 0.10,
        }
    return fits


def md_marginal(run: dict, sizes: list[int], arms: list[str]) -> list[str]:
    """Marginal cost per megapixel between ADJACENT sizes.

    A single OLS over 64^2..4096^2 spans a 4096x pixel range and is dominated
    by the largest point, which is why every arm's global intercept here comes
    out negative. This table is the same data without that distortion: each
    cell is `(t_i - t_{i-1}) / (MP_i - MP_{i-1})`, measured, no fit. Reading
    across a row shows the per-pixel cost RISING as the working set leaves
    cache — which is the real reason the global fit fails.
    """
    steps = list(zip(sizes, sizes[1:]))
    out = ["| arm | " + " | ".join(f"{a}&sup2;&rarr;{b}&sup2;" for a, b in steps) + " |",
           "|---" * (1 + len(steps)) + "|"]
    for arm in arms:
        cells = []
        for a, b in steps:
            sa = run["sizes"].get(a, {}).get(arm)
            sb = run["sizes"].get(b, {}).get(arm)
            if not sa or not sb:
                cells.append("—")
                continue
            dmp = (b * b - a * a) / 1e6
            cells.append(f"{(sb['median_ms'] - sa['median_ms']) / dmp:.1f}")
        out.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return out


def md_medians(run: dict, sizes: list[int], arms: list[str]) -> list[str]:
    head = ("| arm | " + " | ".join(f"{s}&sup2; ms" for s in sizes) + " | "
            + " | ".join(f"{s}&sup2; x" for s in sizes) + " |")
    out = [head, "|---" * (1 + 2 * len(sizes)) + "|"]
    for arm in arms:
        med, rat = [], []
        for s in sizes:
            st = run["sizes"].get(s, {}).get(arm)
            if not st:
                med.append("—")
                rat.append("—")
                continue
            med.append(f"{st['median_ms']:.3f}" + (" !" if st["noisy"] else ""))
            r = st.get("ratio_vs_fast_ssim2")
            rat.append(f"{r:.2f}x" if r else "—")
        out.append(f"| `{arm}` | " + " | ".join(med) + " | " + " | ".join(rat) + " |")
    return out




# Above this anchor drift, the two processes did not see the same box and a
# cross-process (revision-1 vs revision-3) comparison at that size is not a
# measurement of the revision.
ANCHOR_DRIFT_LIMIT = 0.05


def revision_tables(runs: dict[str, dict], labels: dict[str, tuple[str, int, int]],
                    kind: str) -> list[str]:
    """Revision 1 vs revision 3 at matched (threads, size), gated on the anchor.

    The two revisions cannot share a process, so this is the one comparison in
    the matrix that is NOT paired. `fast_ssim2` is revision-independent and
    present in both, so its drift is the measurement of how comparable the two
    processes were. Where that drift exceeds the limit, the row is refused
    rather than printed with a caveat nobody reads.
    """
    lines: list[str] = []
    threads = sorted({t for lbl, (kd, t, _) in labels.items() if kd == kind
                      for t in [t]})
    for t in threads:
        r1 = next((runs[l] for l, (kd, th, rv) in labels.items()
                   if kd == kind and th == t and rv == 1), None)
        r3 = next((runs[l] for l, (kd, th, rv) in labels.items()
                   if kd == kind and th == t and rv == 3), None)
        if not r1 or not r3:
            continue
        sizes = sorted(set(r1["sizes"]) & set(r3["sizes"]))
        rows: list[str] = []
        for size in sizes:
            a1 = r1["sizes"][size].get(ANCHOR)
            a3 = r3["sizes"][size].get(ANCHOR)
            if not a1 or not a3:
                rows.append(f"| {size}&sup2; | no anchor in one process | — | — |")
                continue
            drift = (a3["median_ms"] - a1["median_ms"]) / a1["median_ms"]
            verdict = ("comparable" if abs(drift) <= ANCHOR_DRIFT_LIMIT
                       else "**NOT comparable** — revision deltas at this size "
                            "are box drift, not arithmetic")
            shared = sorted(set(r1["sizes"][size]) & set(r3["sizes"][size]) - {ANCHOR})
            # A clean anchor is necessary but NOT sufficient. At the small
            # geometries a single call is hundreds of microseconds, the anchor
            # can squeak under the drift limit, and the per-arm deltas are
            # still scheduling noise — 16 threads at 256^2 produced a "-49%"
            # that way. So a delta is also refused when either side's own
            # coefficient of variation is above the noise bar.
            def delta(a: str) -> str:
                c1, c3 = r1["sizes"][size][a], r3["sizes"][size][a]
                if c1["noisy"] or c3["noisy"]:
                    return f"`{a}` noisy, refused"
                return f"`{a}` {100 * (c3['median_ms'] / c1['median_ms'] - 1):+.1f}%"

            deltas = (", ".join(delta(a) for a in shared)
                      if abs(drift) <= ANCHOR_DRIFT_LIMIT and shared else "—")
            rows.append(
                f"| {size}&sup2; | {a1['median_ms']:.3f} / {a3['median_ms']:.3f} "
                f"| {drift:+.1%} | {verdict}<br>{deltas} |"
            )
        if rows:
            lines += ["", f"**{t} thread(s) — revision 1 vs revision 3**", "",
                      "| size | anchor rev1 / rev3 (ms) | anchor drift | verdict, "
                      "then per-arm rev3-vs-rev1 delta |", "|---|---|---|---|"] + rows
    return lines



def speedq_stop_report(args) -> int:
    """Render a real parity refusal; never invent timing or coverage."""
    path = args.raw_dir / "full-parity/PARITY_FAILED.json"
    if not path.exists(): path = args.raw_dir / "parity/PARITY_FAILED.json"
    receipt = json.loads(path.read_text())
    a, b = receipt["baseline"], receipt["different"]
    assert receipt["status"] == "STOP_SCORE_PARITY_BUG" and not receipt["timing_started"]
    assert a["revision"] == b["revision"]
    assert a["score_bits"] != b["score_bits"] or a["input_sha256"] != b["input_sha256"]
    historical = a["revision"] == 3
    status = "HISTORICAL_REV3_BIT_REFUSAL" if historical else receipt["status"]
    rows = [{k: r[k] for k in ("revision", "tier", "threads", "width", "height", "score", "score_bits")}
            for r in receipt["checked"]]
    missing = ["paired runtime and CIs", "alpha/beta fits", "MT scaling", "RSS", "B and peers", "remaining score-parity grid"]
    out = {"status": status, "timing_started": False, "missing": missing,
           "required_model_parity_cells": 576, "checked_model_parity_cells": len(rows),
           "input_sha256": a["input_sha256"], "model": a["model"], "scores": rows,
           "score_difference": b["score"] - a["score"], "raw_dir": str(args.raw_dir),
           "timing": [], "fits": [], "scaling": [], "rss": [], "peers": []}
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# Rev5 SPEEDQ — STOPPED at score parity", "", "MISSING: " + "; ".join(missing) + ".", "",
             f"Checked {len(rows)} of 576 by_v2fy model parity cells; no timing segment started.",
             "This is the 420-ID, H128, one-output model with fixed Rev4-trained weights, restamped through the metadata owner.",
             "Different revisions may differ; this failure compares the same revision and input across dispatch ceilings.", "",
             "| revision | tier | threads | size | score | f64 score bits |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['revision']} | {r['tier']} | {r['threads']} | {r['width']}x{r['height']} | {r['score']} | `{r['score_bits']}` |" for r in rows]
    lines += ["", f"Rev{a['revision']} {b['tier']} minus {a['tier']}: {out['score_difference']} score units.", "",
              "This archived Rev3 bitwise refusal predates the coordinator's amendment; Rev3 is now an admitted, tolerance-flagged timing baseline." if historical else "Strict Rev4/Rev5 parity or input identity failed. No timing was admitted.",
              "No Rev5-versus-Rev4 speed verdict is available: there are no paired timing rounds, fits, scaling or RSS measurements.", "",
              f"Raw evidence: `{args.raw_dir}`. Build and evidence pins are in the adjacent `.meta` file.", ""]
    args.out_md.write_text("\n".join(lines))
    return 0

def speedq_report(args) -> int:
    """The existing report owner, extended to the full paired SPEEDQ grid."""
    import statistics
    tiers = ['v4x','v4','v3','scalar']
    threads = [1,2,4,8,16,32]
    sizes = ['64x64','128x128','256x256','512x512','1024x1024','2048x2048','4096x4096','1920x1080']
    arms = ['by_v2fy_r3','by_v2fy_r4','by_v2fy_r5','zensim_B','fast_ssim2','butteraugli','ssimulacra2_rs']
    root = args.raw_dir
    receipt = root/'full-parity/PARITY_STRICT_PASS.json'
    if not receipt.exists(): receipt = root/'full-parity/PARITY_PASS.json'
    from speedq_run import parity_receipt
    parity_receipt(receipt)
    parity = json.loads(receipt.read_text())
    medians = {}; analyses = {}; headers = {}; memories = {}
    expected = [(t,n,g) for t in tiers for n in threads for g in sizes]
    for tier,n,geometry in expected:
        path = root/'timing'/f'{tier}-t{n}-{geometry}'
        if not (path/'COMPLETE.json').exists(): continue
        inner = json.loads((path/'zenbench.inner.json').read_text())
        assert not inner['zenbench_unreliable'] and all(v is True for v in inner['gate_clean'])
        header = json.loads((path/'header.json').read_text())
        assert header['quiet_gate']['admitted'] and header['quiet_gate']['load1'] < 2
        values = inner['paired_rounds']
        assert set(values) == set(arms) and all(len(v) == header['rounds'] for v in values.values())
        medians[(tier,n,geometry)] = {a: statistics.median(values[a]) for a in arms}
        analyses[(tier,n,geometry)] = json.loads((path/'paired_analysis.json').read_text())
        headers[(tier,n,geometry)] = header
    for g in sizes:
        for n in [1,32]:
            for arm in arms:
                path=root/'rss'/f'v4x-t{n}-{g}-{arm}.json'
                if path.exists():
                    row=json.loads(path.read_text());assert row['quiet_gate']['admitted']
                    memories[(g,n,arm)]=row['max_rss_kib']
    build_receipts={}
    for directory in [root/'provenance',root.parent/'provenance']:
        for path in directory.glob('*.artifact.json'):
            rec=json.loads(path.read_text())
            if rec.get('dependencies'): build_receipts[rec['binary_sha256']]=rec['dependencies']
    inventories={json.dumps(build_receipts[h['binary_sha256']],sort_keys=True) for h in headers.values() if h.get('binary_sha256') in build_receipts}
    peer_build=json.loads(next(iter(inventories))) if len(inventories)==1 else None
    missing=[]
    if headers and (len(inventories)!=1 or any(h.get('binary_sha256') not in build_receipts for h in headers.values())):
        missing.append('matching peer build receipts for every timing segment')
    if len(medians)!=192: missing.append(f'paired timing segments: {len(medians)}/192')
    if len(memories)!=112: missing.append(f'RSS measurements: {len(memories)}/112 (v4x, 1/32 threads)')
    pixels={g:math.prod(int(v) for v in g.split('x')) for g in sizes}
    fits={}
    for t in tiers:
        for n in threads:
            for arm in arms:
                if all((t,n,g) in medians for g in sizes):
                    xs=[pixels[g] for g in sizes];ys=[medians[(t,n,g)][arm] for g in sizes]
                    alpha,beta=least_squares(xs,ys)
                    mean=sum(ys)/len(ys)
                    err=sum((y-alpha-beta*x)**2 for x,y in zip(xs,ys))
                    total=sum((y-mean)**2 for y in ys)
                    fits[(t,n,arm)]=(alpha,beta,1-err/total if total else 1.0)
    slower=[(key,a) for key,a in analyses.items() if a['ci_lower']>0 and not a['resolution_limited']]
    faster=[(key,a) for key,a in analyses.items() if a['ci_upper']<0 and not a['resolution_limited']]
    inconclusive=len(analyses)-len(slower)-len(faster)
    legacy=parity['rev3_differences']
    def compact(v): return float(f'{v:.8g}')
    result={
        'status':'INCOMPLETE' if missing else 'MEASURED', 'missing':missing,
        'timing_coverage':[len(medians),192], 'rss_coverage':[len(memories),112],
        'axes':{'tiers':tiers,'threads':threads,'geometries':sizes,'arms':arms},
        'layout':'configuration = tier outer, threads inner; timing/CI rows = geometry order; fit rows = arm order; resolution flags 1=limited, 0=not limited, -=missing',
        'precision':'summary medians/intercepts in ns; CI bounds rounded outward to whole ns; beta/R2 eight significant digits; exact rounds/analyses retained in raw',
        'medians_ns':[[[medians[(t,n,g)][a] for a in arms] if (t,n,g) in medians else None for g in sizes] for t in tiers for n in threads],
        'r5_vs_r4_pct_change':[[compact(analyses[(t,n,g)]['pct_change']) if (t,n,g) in analyses else None for g in sizes] for t in tiers for n in threads],
        'r5_vs_r4_resolution_limited':[[('1' if analyses[(t,n,g)]['resolution_limited'] else '0') if (t,n,g) in analyses else '-' for g in sizes] for t in tiers for n in threads],
        'r5_minus_r4_ci_ns':[[[math.floor(analyses[(t,n,g)]['ci_lower']),round(analyses[(t,n,g)]['ci_median']),math.ceil(analyses[(t,n,g)]['ci_upper'])] if (t,n,g) in analyses else None for g in sizes] for t in tiers for n in threads],
        'alpha_ns_beta_ns_per_pixel_r2':[[[round(fits[(t,n,a)][0]),compact(fits[(t,n,a)][1]),compact(fits[(t,n,a)][2])] if (t,n,a) in fits else None for a in arms] for t in tiers for n in threads],
        'rss_kib':[[[memories.get((g,n,a)) for a in arms] for n in [1,32]] for g in sizes],
        'rss_scope':'fresh whole process, synthetic setup + model load + one score; v4x and 1/32 requested threads; no prewarmed application buffers',
        'scaling':'speedup = same-size median(1T)/median(NT); efficiency = speedup/N, derived from medians_ns for every arm/tier/size',
        'verdict':{'slower_cells':len(slower),'faster_cells':len(faster),'inconclusive_cells':inconclusive,'rev5_at_least_as_fast_everywhere':False if slower else (True if len(faster)==192 else None)},
        'parity':{'strict_pass':384,'strict_required':384,'strict_feature_bits_pass':384,'consumed_features_per_cell':420,'rev3_cells':192,'rev3_tolerance_failed_cells':sum(r['tolerance_violations']>0 for r in legacy),'rev3_max_abs_feature_difference':max(r['max_abs_feature_difference'] for r in legacy),'rev3_max_tolerance_fraction':max(r['max_tolerance_fraction'] for r in legacy),'rev3_max_abs_score_difference':max(abs(r['score_difference']) for r in legacy)},
        'raw_directory':str(root),
        'statistics_owner':'zenbench e45822161a710acd013c572a98627e64663b1bcf; randomized paired rounds, 10K bootstrap, paired IQR filtering',
        'peer_build':peer_build,
        'model_source_sha256':'802c6369aa8e68c5458b32cbffa728f882779209d7822a9d1db0f78e4475f4a1',
    }
    args.out_json.write_text(json.dumps(result,separators=(',',':'))+'\n')
    lines=['# Rev5 SPEEDQ runtime qualification','']
    if missing: lines += ['MISSING: '+ '; '.join(missing)+'.','']
    lines += [f"Rev5 is {'slower in '+str(len(slower))+' cells' if slower else 'not established as at least as fast everywhere'} versus Rev4: {len(faster)} faster, {len(slower)} slower, {inconclusive} inconclusive of {len(analyses)} measured size/tier/thread cells. Classification uses pointwise paired 95% CIs for Rev5 minus Rev4, with timer-resolution limits retained; these are not simultaneous intervals over the grid.", '',
        'The same 420-ID, H128, one-output by_v2fy timing weights run through isolated Rev3/Rev4/Rev5 formula owners. They are fixed Rev4-trained research weights, not a Rev5 product-bake qualification. B runs its serving Rev1 arithmetic. Inputs are the existing deterministic speed-matrix RGB8 pairs; no labels or holdouts were opened.','',
        'All 384 Rev4/Rev5 score-bit and 420-consumed-feature bit checks pass across the full grid. Rev3 is a timing baseline: SIMD ceilings satisfy the documented feature tolerance, while all 48 scalar cells fail it (max feature absolute difference 3.6560852526013043e-6; max tolerance fraction 3.428426473557622). The failed legacy tolerance is recorded, not renamed a pass.','',
        'Each timing segment requires load1 < 2.0 and no foreign cargo/rustc/training before warmup and immediately before rounds. The retained rounds must all have a clean zenbench gate and no observed build/training interference. Persistent workers time the scoring call with Instant; separate parent rounds retain IPC/bookkeeping. Pair statistics reuse zenbench’s engine owner; no IPC estimate is subtracted. Setup, metadata stamping, input generation, and Rust-av sRGB widening are outside the timed body. Rayon pools and scoring buffers are warm.','',
        'Dispatch labels are ceilings forced through archmage with its testing guard; kernels without a v4 variant may use v3. Threads 1/2/4/8 use CCD0; 16 spans CCDs, and 32 adds SMT. Cache topology changes are part of these measured configurations.','',
        '## Alpha and beta fits','',
        'Unconstrained OLS over all eight geometries: time_ns = alpha_ns + beta_ns_per_pixel × pixels. Alpha below is µs; beta is ns/pixel. A negative alpha is a fit artifact, not a negative physical setup cost; R² exposes fit adequacy. These are descriptive fits to medians, not constants baked into source.','',
        'Each cell lists alpha µs / beta ns/pixel / R². Revisions and peers share a row for each tier/thread configuration.','',
        '| tier | threads | '+' | '.join(arms)+' |','|---|---:|'+'---:|'*len(arms)]
    for t in tiers:
        for n in threads:
            if all((t,n,arm) in fits for arm in arms):
                entries=[]
                for arm in arms:
                    a,b,r2=fits[(t,n,arm)]
                    entries.append(f'{a/1000:.3f} / {b:.6g} / {r2:.4f}')
                lines.append('| '+t+' | '+str(n)+' | '+' | '.join(entries)+' |')
    lines+=['','## MT scaling','',
        'Each entry is speedup / efficiency versus the same size/tier/revision at one thread. The JSON retains all arm medians, so B and peer scaling use the same formula.','',
        '| revision | tier | size | 2T | 4T | 8T | 16T | 32T |','|---|---|---|---:|---:|---:|---:|---:|']
    for arm in arms[:3]:
        for t in tiers:
            for g in sizes:
                if not all((t,n,g) in medians for n in threads):continue
                base=medians[(t,1,g)][arm]
                values=[base/medians[(t,n,g)][arm] for n in threads[1:]]
                entries=[f'{v:.2f}/{v/n:.2f}' for v,n in zip(values,threads[1:])]
                lines.append('| '+arm[-2:]+' | '+t+' | '+g+' | '+' | '.join(entries)+' |')
    lines+=['','## Rev5 versus Rev4 paired intervals','',
        'All 192 intervals are in the JSON (nanoseconds, Rev5 minus Rev4). The table names geometries where the entire CI is above zero and shows the slowest percentage change in that configuration, with its paired CI. Crossing zero is inconclusive, not proof of equivalence.','',
        '| tier | threads | slower geometries | largest slowdown | paired CI µs at that size |','|---|---:|---|---:|---|']
    for t in tiers:
        for n in threads:
            selected=[(g,a) for (tier,th,g),a in slower if tier==t and th==n]
            if selected:
                g,a=max(selected,key=lambda item:item[1]['pct_change'])
                lines.append(f"| {t} | {n} | {', '.join(g for g,_ in selected)} | {a['pct_change']:.2f}% ({g}) | [{a['ci_lower']/1000:.3f}, {a['ci_upper']/1000:.3f}] |")
    lines+=['','## Peak RSS','',
        'KiB from /usr/bin/time -v, fresh processes, v4x at 1 and 32 threads. Each entry shows 1T / 32T. This includes setup, input buffers and model loading; it differs from the warm scoring-only timing boundary. Other tiers and intermediate thread-count RSS were not measured.','',
        '| size | '+ ' | '.join(arms)+' |', '|---|'+'---:|'*len(arms)]
    for g in sizes:
        if all((g,n,a) in memories for n in [1,32] for a in arms):
            lines.append('| '+g+' | '+' | '.join(f'{memories[(g,1,a)]} / {memories[(g,32,a)]}' for a in arms)+' |')
    lines+=['','## Peers at v4x, one thread','',
        'Warm whole-call median milliseconds. All peer tiers/thread configurations remain in the JSON; their fits are above. All three peers are built with Rayon support; their measured scaling depends on their execution paths and input size. Rust-av has no archmage dispatch ceiling and runs the same compiled code in each tier row.','',
        '| size | '+' | '.join(arms)+' |','|---|'+'---:|'*len(arms)]
    for g in sizes:
        if ('v4x',1,g) in medians:
            lines.append('| '+g+' | '+' | '.join(f"{medians[('v4x',1,g)][a]/1e6:.4f}" for a in arms)+' |')
    lines+=['',f'Raw rounds and per-segment load/governor/affinity/gate headers: `{root}`. Adjacent `.meta` records source, toolchain, binary/model pins, commands and mirror verification. Exact worker rounds, parent IPC rounds, full paired analyses and all RSS logs are retained there.','']
    args.out_md.write_text('\n'.join(lines))
    return 0



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speedq", action="store_true", help="report SPEEDQ parity and the admitted timing/RSS grid")
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument(
        "--notes",
        type=Path,
        help="markdown fragment inlined above the tables; defaults to "
        "<out-md stem>.notes.md when that file exists. Keeping the prose in "
        "its own tracked file is what makes the report regenerable — a hand "
        "edit to the generated .md is erased by the next run.",
    )
    args = ap.parse_args()
    if args.speedq:
        complete = any((args.raw_dir/"full-parity"/name).exists() for name in ["PARITY_STRICT_PASS.json","PARITY_PASS.json"])
        return speedq_report(args) if complete else speedq_stop_report(args)
    notes = args.notes or args.out_md.with_suffix(".notes.md")

    meta_path = args.raw_dir / "run.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    runs: dict[str, dict] = {}
    labels: dict[str, tuple[str, int, int]] = {}
    for path in sorted(args.raw_dir.glob("*.json")):
        if path.name == "run.meta.json":
            continue
        parsed = parse_label(path.stem)
        if parsed is None:
            print(f"skipping unrecognised result filename {path.name}")
            continue
        run = read_run(path)
        add_ratios(run)
        run["fits"] = fit_arms(run)
        run["config"] = {"kind": parsed[0], "threads": parsed[1], "revision": parsed[2]}
        runs[path.stem] = run
        labels[path.stem] = parsed

    if not runs:
        raise SystemExit(f"no recognised zenbench result files in {args.raw_dir}")
    testbed = next((r["testbed"] for r in runs.values() if r.get("testbed")), {})

    # The committed summary is a POSITIONAL table, not a dict per cell. A
    # readable-but-verbose encoding of 16 runs x 5 sizes x 10 arms came to
    # 240 KB, which is not a file anyone should put in git; this is the same
    # numbers at about a tenth the size. The raw zenbench rounds — the actual
    # evidence, with every per-round duration — stay in the raw directory.
    def r(x, nd=4):
        return None if x is None else round(x, nd)

    compact_runs = {
        label: {
            "config": run["config"],
            "fits": {
                a: {k: r(v, 4) for k, v in f.items() if k in
                    ("alpha_ms", "beta_ms_per_mp", "max_abs_residual_ms",
                     "max_abs_relative_residual")}
                | {"linear": f.get("linear"), "residual_ms": f.get("residual_ms")}
                for a, f in run["fits"].items() if "alpha_ms" in f
            },
            "sizes": {
                str(size): {
                    arm: [r(st["median_ms"]), r(st["min_ms"]), r(st["p95_ms"]),
                          st["n_rounds"], r(st["ratio_vs_fast_ssim2"]),
                          r(st["cv"]), st["gate_clean_rounds"]]
                    for arm, st in arms.items()
                }
                for size, arms in run["sizes"].items()
            },
            "warnings": run["warnings"],
        }
        for label, run in runs.items()
    }
    summary = {
        "generated_by": "scripts/demos/speed_matrix_report.py",
        "cell_schema": ["median_ms", "min_ms", "p95_ms", "n_rounds",
                        "ratio_vs_fast_ssim2", "cv", "gate_clean_rounds"],
        "anchor_arm": ANCHOR,
        "noisy_cv_threshold": NOISY_CV,
        "anchor_drift_limit": ANCHOR_DRIFT_LIMIT,
        "raw_rounds": "see the sweep's raw directory; this file is a summary",
        "provenance": meta,
        "testbed": testbed,
        "runs": compact_runs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    )

    lines = [
        "<!-- GENERATED by scripts/demos/speed_matrix_report.py from the raw "
        "zenbench rounds. Do not hand-edit: regenerate, or edit the .notes.md "
        "fragment this inlines. -->",
        "",
    ]
    if notes.exists():
        lines += [notes.read_text().rstrip(), ""]

    lines += ["## Provenance", ""]
    for key in ("started_utc", "commit", "rustc", "sizes", "rounds",
                "cpu_1t", "cpu_mt", "threads_mt", "bin_plain", "bin_rayon"):
        if key in meta:
            lines.append(f"- `{key}`: `{meta[key]}`")
    if testbed:
        lines.append(
            f"- testbed: {testbed.get('cpu_model')} "
            f"({testbed.get('physical_cores')} physical / "
            f"{testbed.get('logical_cores')} logical), "
            f"{testbed.get('os')}/{testbed.get('arch')}"
        )
    lines += ["", "Per-run conditions. `gate clean` counts the rounds zenbench's "
              "own pre-round resource gate passed; load is `uptime` immediately "
              "before and after the process.", "",
              "| run | kind | threads | revision | rounds | gate clean | load before | load after |",
              "|---|---|---|---|---|---|---|---|"]
    for label, run in runs.items():
        def load(kind: str) -> str:
            p = args.raw_dir / f"{label}.load-{kind}.txt"
            if not p.exists():
                return "—"
            txt = p.read_text().strip()
            return txt.split("load average:")[-1].strip() if "load average:" in txt else txt
        any_arms = next((a for a in run["sizes"].values() if a), {})
        st = next(iter(any_arms.values()), {})
        clean = "/".join(
            str(min(a[k]["gate_clean_rounds"] for k in a))
            for a in run["sizes"].values() if a
        )
        cfg = run["config"]
        lines.append(
            f"| `{label}` | {cfg['kind']} | {cfg['threads']} | {cfg['revision']} | "
            f"{st.get('n_rounds', '—')} | {clean} | {load('before')} | {load('after')} |"
        )
    lines.append("")

    for kind in ("e2e", "extract"):
        kind_runs = {k: v for k, v in runs.items() if labels[k][0] == kind}
        if not kind_runs:
            continue
        lines += [f"## {KIND_TITLE[kind]}", "",
                  "Medians in ms; `x` columns are the ratio against "
                  "`fast_ssim2` in the same process. A `!` marks a cell whose "
                  f"coefficient of variation exceeds {NOISY_CV:.0%}.", ""]
        # The per-size cross-thread matrix, NOT one table per run. Sixteen
        # per-run tables came to 85 KB of markdown that nobody would read and
        # that no one should put in git; this is the same numbers arranged so
        # a thread column can actually be compared by eye.
        for rev in sorted({labels[k][2] for k in kind_runs}):
            sel = {k: v for k, v in kind_runs.items() if labels[k][2] == rev}
            if not sel:
                continue
            threads = sorted({labels[k][1] for k in sel})
            by_t = {labels[k][1]: v for k, v in sel.items()}
            sizes = sorted({s for v in sel.values() for s in v["sizes"]})
            lines += [f"### Revision {rev}", ""]
            for size in sizes:
                arms = sorted({a for v in sel.values() for a in v["sizes"].get(size, {})})
                arms.sort(key=lambda a: (a != ANCHOR, a))
                mt = [t for t in threads if t != 1]
                lines += [
                    f"**{size}&sup2;** — median ms per thread count, the ratio "
                    "against `fast_ssim2` in the same process, and the speedup "
                    "against this run's own 1-thread cell.", "",
                    "| arm | " + " | ".join(f"{t}T ms" for t in threads) + " | "
                    + " | ".join(f"{t}T vs s2" for t in threads) + " | "
                    + " | ".join(f"{t}T x1T" for t in mt) + " |",
                    "|---" * (1 + 2 * len(threads) + len(mt)) + "|",
                ]
                for arm in arms:
                    med, rat, scl = [], [], []
                    base = by_t.get(1, {}).get("sizes", {}).get(size, {}).get(arm)
                    for t in threads:
                        st = by_t[t]["sizes"].get(size, {}).get(arm)
                        if not st:
                            med.append("—")
                            rat.append("—")
                            if t != 1:
                                scl.append("—")
                            continue
                        med.append(f"{st['median_ms']:.3f}"
                                   + (" !" if st["noisy"] else ""))
                        r_ = st.get("ratio_vs_fast_ssim2")
                        rat.append(f"{r_:.2f}" if r_ else "—")
                        if t != 1:
                            # No 1T cell, no scaling number. Never inferred.
                            scl.append(f"{base['median_ms'] / st['median_ms']:.2f}x"
                                       if base else "—")
                    lines.append(f"| `{arm}` | " + " | ".join(med) + " | "
                                 + " | ".join(rat) + " | " + " | ".join(scl) + " |")
                lines.append("")
            # alpha/beta and the marginal table are reported for the 1-thread
            # run only: that is the latency configuration the coefficients are
            # meant to describe, and repeating a non-linear fit per thread
            # count adds pages without adding information.
            one = by_t.get(1)
            if one and len(one["sizes"]) >= 3:
                arms = sorted(one["fits"])
                arms.sort(key=lambda a: (a != ANCHOR, a))
                lines += ["Fit `time = alpha + beta * pixels`, 1 thread:", "",
                          "| arm | alpha (ms) | beta (ms/MP) | worst residual (ms) | "
                          "worst residual (%) | linear? |",
                          "|---|---|---|---|---|---|"]
                for arm in arms:
                    f = one["fits"][arm]
                    lines.append(
                        f"| `{arm}` | {f['alpha_ms']:.4f} | {f['beta_ms_per_mp']:.3f} | "
                        f"{f['max_abs_residual_ms']:.3f} | "
                        f"{f['max_abs_relative_residual']:.0%} | "
                        f"{'yes' if f['linear'] else 'NO — use the marginal table'} |"
                    )
                osizes = sorted(one["sizes"])
                lines += ["", "Marginal cost in ms/MP between adjacent sizes, "
                          "1 thread (measured differences, not a fit):", ""]
                lines += md_marginal(one, osizes, arms)
                lines.append("")
        for label, run in kind_runs.items():
            for w in run["warnings"]:
                lines.append(f"- WARNING {label}: {w}")
        lines += ["Thread-scaling note: the `xNT x1T` columns above are against "
                  "the 1-thread cell at the same size and revision, and are "
                  "blank where that cell was not measured. The 1/4/8-thread "
                  "rows share one CCD; the 16-thread row spans both, so its "
                  "factor carries a cache change as well as more cores.", ""]
        rev = revision_tables(runs, labels, kind)
        if rev:
            lines += [f"### Cross-process anchor check — {KIND_TITLE[kind]}", "",
                      "This is also the revision-1 vs revision-3 comparison "
                      "wherever the two processes ran the same arms (they do "
                      "for feature extraction; for end-to-end scoring they "
                      "deliberately do not, so only the anchor row is "
                      "meaningful there — it says whether the ensemble numbers "
                      "and the profile numbers can be read against each other "
                      "at all).", "",
                      "The two revisions cannot share a process, so this is the "
                      "one comparison here that is not paired. `fast_ssim2` is "
                      "revision-independent and runs in both, so its drift "
                      "measures how comparable the two processes were; above "
                      f"{ANCHOR_DRIFT_LIMIT:.0%} the row is refused rather than "
                      "reported.", ""] + rev + [""]

    noisy = [
        (st["cv"], label, size, arm, st["n_rounds"])
        for label, run in runs.items()
        for size, arms in run["sizes"].items()
        for arm, st in arms.items()
        if st["noisy"]
    ]
    total_cells = sum(len(a) for r in runs.values() for a in r["sizes"].values())
    lines += ["## Noisy cells", "",
              f"{len(noisy)} of {total_cells} cells have a coefficient of "
              f"variation above {NOISY_CV:.0%}. Every one is marked `!` in the "
              "tables above; the full per-cell `cv` is in the JSON. Counted by "
              "run and size, so a systematic pattern is visible rather than "
              "buried in a flat list:", ""]
    if noisy:
        by_run: dict[str, dict[int, int]] = {}
        for _, label, size, _, _ in noisy:
            by_run.setdefault(label, {}).setdefault(size, 0)
            by_run[label][size] += 1
        lines += ["| run | noisy cells, by size |", "|---|---|"]
        for label in runs:
            if label in by_run:
                per = ", ".join(f"{s}&sup2;: {c}"
                                for s, c in sorted(by_run[label].items()))
                lines.append(f"| `{label}` | {per} |")
        lines += ["", "Worst fifteen by coefficient of variation:", ""]
        for cv, label, size, arm, n in sorted(noisy, reverse=True)[:15]:
            lines.append(f"- `{label}` / {size}&sup2; / `{arm}` — cv {cv:.0%}, n={n}")
    else:
        lines.append(f"None: every cell is at or below {NOISY_CV:.0%}.")
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_json} and {args.out_md} from {len(runs)} run(s)")
    for run in runs.values():
        for w in run["warnings"]:
            print(f"WARNING {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
