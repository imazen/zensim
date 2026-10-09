"""COSTCMP view of the existing SPEEDQ report and statistics owners."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
import costcmp_run as cmp
import speedq_run as collection
from speedq_report_support import verify_speedq_batches


def report(args):
    from speed_matrix_report import least_squares
    root=args.raw_dir
    rows=cmp.receipt(root/'parity/PREFLIGHT_PASS.json')
    artifact=json.loads((root/'provenance/instrument.artifact.json').read_text())
    expected_sha=artifact['binary_sha256']
    medians={};analyses={};selections={};batches=0;memories={};packets=[];saved=[]
    for g,t,n in cmp.cells():
        p=root/'timing'/f'{t}-t{n}-{g}'
        done=json.loads((p/'COMPLETE.json').read_text());header=json.loads((p/'header.json').read_text())
        assert done['status']=='PASS' and done['rounds']==32 and done['paired_alignment_verified'] and done['zenbench_gate_clean']
        assert header['quiet_gate']['admitted'] and header['quiet_gate']['load1']<2 and not header['gate_trace']
        assert header['binary_sha256']==expected_sha and header['arms']==cmp.ARMS
        assert header['round_cap']==64 and header['round_rule']==collection.CLEAN_ROUND_RULE
        interference=json.loads((p/'interference.json').read_text());assert interference['admitted'] and not interference['foreign']
        inner=json.loads((p/'zenbench.inner.json').read_text());values,selection=collection.select_clean_rounds(inner)
        assert set(values)==set(cmp.ARMS) and all(done[k]==v for k,v in selection.items())
        for name in cmp.ARMS:cmp.validate_ready(rows,name,g,t,n,inner['workers'][name])
        batches+=verify_speedq_batches(p,inner)
        analysis=json.loads((p/'paired_analysis.json').read_text());assert analysis['baseline_arm']==cmp.ARMS[0]
        assert set(analysis['comparisons'])==set(cmp.ARMS[1:])
        for name,a in analysis['comparisons'].items():
            assert a['n_samples']==32 and a['ci_lower']<=a['ci_median']<=a['ci_upper']
        for name in cmp.ARMS[1:]:
            packets.append(dict(baseline=values[cmp.ARMS[0]],candidate=values[name],iterations=[1]*32,timer_resolution_ns=inner['timer_resolution_ns']))
            saved.append(analysis['comparisons'][name])
        key=f'{t}-t{n}-{g}';medians[key]={a:statistics.median(v) for a,v in values.items()};analyses[key]=analysis;selections[key]=selection
    analyzer=root/'provenance/paired-rounds-analyzer'
    source=json.loads((root/'provenance/source.json').read_text())
    assert hashlib.sha256(analyzer.read_bytes()).hexdigest()==source['paired_analyzer_sha256']
    replay=json.loads(subprocess.check_output([str(analyzer)],input=json.dumps(packets),text=True))
    assert replay==saved, 'paired statistics replay differs from saved analyses'
    for g in collection.GEOMETRIES:
        for n in (1,32):
            for arm in cmp.ARMS:
                row=json.loads((root/'rss'/f'v4x-t{n}-{g}-{arm}.json').read_text())
                assert row['quiet_gate']['admitted'] and row['quiet_gate']['load1']<2
                cmp.validate_ready(rows,arm,g,'v4x',n,row['worker'])
                memories[f'v4x-t{n}-{g}-{arm}']=row['max_rss_kib']
    fits=[]
    for tier in cmp.TIERS:
        for n in cmp.THREADS:
            pixels=[int(g.split('x')[0])*int(g.split('x')[1]) for g in collection.GEOMETRIES]
            for arm in cmp.ARMS:
                times=[medians[f'{tier}-t{n}-{g}'][arm] for g in collection.GEOMETRIES]
                alpha,beta=least_squares(pixels,times);mean=sum(times)/len(times)
                error=sum((y-alpha-beta*x)**2 for x,y in zip(pixels,times));total=sum((y-mean)**2 for y in times)
                fits.append(dict(tier=tier,threads=n,arm=arm,alpha_ns=alpha,beta_ns_per_pixel=beta,r2=1-error/total,
                                 ms_per_mp_1024sq=medians[f'{tier}-t{n}-1024x1024'][arm]/1e6/(1024**2/1e6),
                                 ms_per_mp_4096sq=medians[f'{tier}-t{n}-4096x4096'][arm]/1e6/(4096**2/1e6)))
    result=dict(status='MEASURED',timing_configurations=64,arm_timings=320,rss_observations=80,
                axes=dict(tiers=cmp.TIERS,threads=cmp.THREADS,geometries=collection.GEOMETRIES,arms=cmp.ARMS),
                binary_sha256=expected_sha,model_sha256=collection.SOURCE_SHA,fast_ssim2_main=cmp.FAST_MAIN,
                medians_ns=medians,paired_analyses=analyses,round_selection=selections,validated_parent_batches=batches,paired_analysis_replays=len(replay),
                fits=fits,rss_kib=memories,raw_directory=str(root),peer_build=artifact['dependencies'],
                statistics_owner='frozen zenbench paired_rounds analyzer; SHA and source recorded in metadata',
                interval_scope='pointwise paired 95% CIs, candidate minus production Rev5; not simultaneous grid-wide intervals')
    args.out_json.write_text(json.dumps(result,separators=(',',':'))+'\n')
    lines=['# Production model cost comparison','',
        'All 64 configurations have five arms; all 80 requested fresh-process RSS observations are measured. Inputs are SPEEDQ deterministic RGB8 pairs. This measures scoring cost, not model quality or corpus-wide performance.','',
        f'Production: frozen seed-0 f16 `{collection.SOURCE_SHA}` at Rev5. A and B use their complete named serving routes at their own Rev1 arithmetic. Latest fast-ssim2 main is `{cmp.FAST_MAIN}` (0.9.0); registry 0.8.2 remains a separate continuity arm. Both use RGB8 one-shot scoring with Rayon enabled. Setup/input/model loading is untimed for scoring; scoring allocations and each complete forward/calibration are timed. RSS measures fresh processes including setup.','',
        'All rounds use the original SPEEDQ quiet gate, exclusive segment lock and first-32-clean-of-at-most-64 rule. Intervals reuse the frozen zenbench paired-bootstrap/IQR owner. Pointwise CIs are not simultaneous intervals across the grid. A zero-crossing or resolution-limited interval is inconclusive.','',
        '## Per-cell times and paired intervals','',
        'Each peer entry is its warm median ms, followed by the pointwise 95% interval for peer minus production ms. Positive bounds mean the peer costs more; negative bounds mean it costs less. Resolution-limited entries are marked `limited`.','',
        '| tier | threads | size | production ms | A ms [CI] | B ms [CI] | fast main ms [CI] | fast 0.8.2 ms [CI] |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for g,t,n in cmp.cells():
        k=f'{t}-t{n}-{g}';entries=[f'{medians[k][cmp.ARMS[0]]/1e6:.6f}']
        for a in cmp.ARMS[1:]:
            ci=analyses[k]['comparisons'][a]
            entries.append(f"{medians[k][a]/1e6:.6f} [{ci['ci_lower']/1e6:.6f}, {ci['ci_upper']/1e6:.6f}]"+(' limited' if ci['resolution_limited'] else ''))
        lines.append(f'| {t} | {n} | {g} | '+' | '.join(entries)+' |')
    lines+=['','## Alpha, beta and measured ms/MP','',
            'Unconstrained OLS through the existing reporter owner: time_ns = alpha_ns + beta_ns_per_pixel × pixels over all eight geometries. Negative intercepts are fit artifacts. R² reports adequacy. ms/MP columns divide the actual 1024² and 4096² medians by their exact decimal-MP sizes (1.048576 and 16.777216 MP); they are not extrapolations to an unmeasured one-million-pixel input.','',
            '| tier | threads | arm | alpha µs | beta ns/px | R² | 1024² ms/MP | 4096² ms/MP |','|---|---:|---|---:|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['tier']} | {f['threads']} | {f['arm']} | {f['alpha_ns']/1000:.3f} | {f['beta_ns_per_pixel']:.6f} | {f['r2']:.6f} | {f['ms_per_mp_1024sq']:.6f} | {f['ms_per_mp_4096sq']:.6f} |")
    lines+=['','## Fresh-process peak RSS','',
            'KiB from each `/usr/bin/time -v` log; v4x only, 1/32 requested threads. Each cell shows 1T / 32T. Intermediate threads and v3 RSS are not measured.','',
            '| size | '+' | '.join(cmp.ARMS)+' |','|---|'+'---:|'*len(cmp.ARMS)]
    for g in collection.GEOMETRIES:lines.append('| '+g+' | '+' | '.join(f"{memories[f'v4x-t1-{g}-{a}']} / {memories[f'v4x-t32-{g}-{a}']}" for a in cmp.ARMS)+' |')
    lines+=['',f'Raw evidence: `{root}`. Adjacent metadata and JSON pointer record exact source/model/binary/dependency/analyzer pins, commands, strict production parity and archive verification.','']
    args.out_md.write_text('\n'.join(lines))
    return 0
