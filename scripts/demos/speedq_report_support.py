"""Validation and comparison helpers for the existing SPEEDQ reporter."""
import json
import math
from pathlib import Path

def verify_speedq_batches(path, inner):
    """Bind worker gate flags to the saved parent rounds and batch offsets."""
    manifest=json.loads((path/'zenbench.json').read_text())
    assert manifest['schema']=='speedq-parent-batches-v1'
    assert manifest['round_cap']==64 and manifest['rounds_total']==len(inner['gate_clean'])
    offset=0;flags=[]
    assert manifest['batches']
    for i,batch in enumerate(manifest['batches']):
        assert batch['round_offset']==offset, 'parent batch offsets differ'
        assert batch['warmup_ms']==(20 if i==0 else 0), 'parent batch warmup differs'
        assert Path(batch['path']).name==batch['path'], 'parent batch must be a local member'
        parent=json.loads((path/batch['path']).read_text())
        assert parent['unreliable'] is False, 'unreliable parent batch admitted'
        assert len(parent['comparisons'])==1
        comparison=parent['comparisons'][0]
        samples=comparison['samples']
        assert comparison['completed_rounds']==batch['rounds_total']==len(samples)==batch['rounds_requested']
        assert {b['name'] for b in comparison['benchmarks']}==set(inner['paired_rounds']), 'parent batch arms differ'
        assert all(s['iterations']==1 for s in samples), 'parent rounds must be single calls'
        flags.extend(s['gate_clean'] for s in samples)
        offset+=len(samples)
    assert flags==inner['gate_clean'], 'parent and worker gate flags differ'
    return len(manifest['batches'])


def speedq_comparison(before, after):
    """Align two measured grids; reuse each run's paired CIs without refitting."""
    for result in (before, after):
        assert result['timing_coverage'] == [192,192], 'comparison requires both full timing grids'
        assert result['model_source_sha256'] == before['model_source_sha256'], 'comparison model differs'
        assert result['statistics_owner'] == before['statistics_owner'], 'comparison statistics owner differs'
        assert result['axes'] == before['axes'], 'comparison grid differs'
    rows=[]
    axes=before['axes']
    configurations=[(t,n) for t in axes['tiers'] for n in axes['threads']]
    assert len(configurations)*len(axes['geometries'])==192
    r5=axes['arms'].index('by_v2fy_r5')
    def verdict(ci,limited):
        assert len(ci)==3 and all(math.isfinite(x) for x in ci) and ci[0]<=ci[1]<=ci[2]
        assert limited in ('0','1')
        return 'slower' if ci[0]>0 and limited=='0' else ('faster' if ci[2]<0 and limited=='0' else 'inconclusive')
    def cell_verdict(result,i,j,ci):
        rounded=verdict(ci,result['r5_vs_r4_resolution_limited'][i][j])
        exact=result.get('r5_vs_r4_cell_verdict')
        if exact is None: return rounded
        value=exact[i][j]
        assert value in ('slower','faster','inconclusive')
        # Outward rounding may put a sub-ns bound at zero. Preserve the
        # classification made from the original full-precision paired CI.
        assert rounded=='inconclusive' or rounded==value
        return value
    for i,(tier,n) in enumerate(configurations):
        for j,g in enumerate(axes['geometries']):
            old_ci=before['r5_minus_r4_ci_ns'][i][j]
            new_ci=after['r5_minus_r4_ci_ns'][i][j]
            old_median=before['medians_ns'][i][j][r5]
            new_median=after['medians_ns'][i][j][r5]
            assert old_median>0 and new_median>0
            rows.append(dict(tier=tier,threads=n,geometry=g,
                before_ci_ns=old_ci,after_ci_ns=new_ci,
                before_verdict=cell_verdict(before,i,j,old_ci),
                after_verdict=cell_verdict(after,i,j,new_ci),
                rev5_median_before_ns=old_median,rev5_median_after_ns=new_median,
                rev5_median_change_pct=100*(new_median/old_median-1)))
    for prefix,result in [('before',before),('after',after)]:
        for label in ('slower','faster','inconclusive'):
            assert sum(r[prefix+'_verdict']==label for r in rows)==result['verdict'][label+'_cells'], 'rounded comparison classifications differ from authoritative verdict counts'
    return dict(before_verdict=before['verdict'],after_verdict=after['verdict'],cells=rows,
                interpretation='Each CI compares Rev5 minus Rev4 within its own run; median changes between runs are descriptive, not a paired between-build confidence interval.')


def speedq_comparison_markdown(comparison, baseline_path):
    lines=['# SPEEDQ paired-run comparison','',f'Baseline: `{baseline_path}`.', '',comparison['interpretation'], '',
           'CI bounds below are the outward-rounded pointwise 95% bounds, in milliseconds. Inconclusive includes timer-resolution limits. All 192 grid cells are shown.', '',
           '| tier | threads | size | before CI ms | after CI ms | before | after | Rev5 median change % |',
           '|---|---:|---|---:|---:|---|---|---:|']
    for row in comparison['cells']:
        old=row['before_ci_ns'];new=row['after_ci_ns']
        lines.append(f"| {row['tier']} | {row['threads']} | {row['geometry']} | {old[0]/1e6:.6f} .. {old[2]/1e6:.6f} | {new[0]/1e6:.6f} .. {new[2]/1e6:.6f} | {row['before_verdict']} | {row['after_verdict']} | {row['rev5_median_change_pct']:.4f} |")
    return '\n'.join(lines)+'\n'


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
             "This is the 420-ID, H128, one-output model with a recorded source SHA, restamped through the metadata owner without requantizing its weights.",
             "Different revisions may differ; this failure compares the same revision and input across dispatch ceilings.", "",
             "| revision | tier | threads | size | score | f64 score bits |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['revision']} | {r['tier']} | {r['threads']} | {r['width']}x{r['height']} | {r['score']} | `{r['score_bits']}` |" for r in rows]
    lines += ["", f"Rev{a['revision']} {b['tier']} minus {a['tier']}: {out['score_difference']} score units.", "",
              "This archived Rev3 bitwise refusal predates the coordinator's amendment; Rev3 is now an admitted, tolerance-flagged timing baseline." if historical else "Strict Rev4/Rev5 parity or input identity failed. No timing was admitted.",
              "No Rev5-versus-Rev4 speed verdict is available: there are no paired timing rounds, fits, scaling or RSS measurements.", "",
              f"Raw evidence: `{args.raw_dir}`. Build and evidence pins are in the adjacent `.meta` file.", ""]
    args.out_md.write_text("\n".join(lines))
    return 0

