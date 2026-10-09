"""Validate REV5PERF4's COSTCMP receipts with the existing statistics owner."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import costcmp_run as cmp
import speedq_run as owner
from speedq_report_support import verify_speedq_batches


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(args):
    root = args.raw_dir
    rows = cmp.receipt(root / 'scaling/parity/PREFLIGHT_PASS.json', scaling=True)
    from rev5perf_report import compare_bits
    parity = compare_bits(root/'before', root/'verified')
    builds = {label: json.loads((root / f'provenance/instrument-{label}.artifact.json').read_text())
              for label in ('before', 'verified')}
    assert builds['before']['dependencies'] == builds['verified']['dependencies']
    for label, artifact in builds.items():
        assert sha(root / f'provenance/instrument-{label}') == artifact['binary_sha256']
    medians = {}; analyses = {}; selections = {}; packets = []; saved = []; batches = 0
    expected_workers = {arm: builds['before' if arm == 'by_v2fy_r5_before' else 'verified']['binary_sha256']
                        for arm in cmp.SCALING_ARMS}
    for g, t, n in cmp.cells(scaling=True):
        tag = f'{t}-t{n}-{g}'
        dest = root / 'scaling/timing' / tag
        done = json.loads((dest / 'COMPLETE.json').read_text())
        header = json.loads((dest / 'header.json').read_text())
        assert done['status'] == 'PASS' and done['rounds'] == 32
        assert done['paired_alignment_verified'] and done['zenbench_gate_clean']
        assert header['quiet_gate']['admitted'] and header['quiet_gate']['load1'] < 2
        assert not header['gate_trace'] and header['cpuset'] == owner.CPUSETS[n]
        assert header['binary_sha256'] == builds['verified']['binary_sha256']
        assert header['worker_binary_sha256'] == expected_workers
        assert header['arms'] == cmp.SCALING_ARMS
        assert header['round_cap'] == 64 and header['round_rule'] == owner.CLEAN_ROUND_RULE
        assert header['model_source_sha256'] == owner.SOURCE_SHA
        interference = json.loads((dest / 'interference.json').read_text())
        assert interference['admitted'] and not interference['foreign']
        inner = json.loads((dest / 'zenbench.inner.json').read_text())
        values, selection = owner.select_clean_rounds(inner, 32)
        assert set(values) == set(cmp.SCALING_ARMS)
        assert all(done[k] == v for k, v in selection.items())
        for name in cmp.SCALING_ARMS:
            cmp.validate_ready(rows, name, g, t, n, inner['workers'][name])
        batches += verify_speedq_batches(dest, inner)
        analysis = json.loads((dest / 'paired_analysis.json').read_text())
        assert analysis['baseline_arm'] == cmp.SCALING_ARMS[0]
        assert set(analysis['comparisons']) == set(cmp.SCALING_ARMS[1:])
        for arm in cmp.SCALING_ARMS[1:]:
            ci = analysis['comparisons'][arm]
            assert ci['n_samples'] == 32 and ci['ci_lower'] <= ci['ci_median'] <= ci['ci_upper']
            packets.append(dict(baseline=values[cmp.SCALING_ARMS[0]], candidate=values[arm],
                                iterations=[1]*32, timer_resolution_ns=inner['timer_resolution_ns']))
            saved.append(ci)
        medians[tag] = {name: statistics.median(v) for name, v in values.items()}
        analyses[tag] = analysis
        selections[tag] = selection
    analyzer = root / 'provenance/paired-rounds-analyzer'
    source = json.loads((root / 'provenance/verified-source.json').read_text())
    assert sha(analyzer) == source['paired_analyzer_sha256']
    replay = json.loads(subprocess.check_output([str(analyzer)], input=json.dumps(packets), text=True))
    assert replay == saved, 'saved paired analyses differ from exact replay'
    memories = {}
    for g in cmp.SCALING_GEOMETRIES:
        for n in cmp.SCALING_THREADS:
            for arm in ('by_v2fy_r5', 'by_v2fy_r5_before'):
                tag = f'v4x-t{n}-{g}-{arm}'
                row = json.loads((root / 'scaling/rss' / (tag + '.json')).read_text())
                assert row['quiet_gate']['admitted'] and row['quiet_gate']['load1'] < 2
                assert row['binary_sha256'] == expected_workers[arm]
                cmp.validate_ready(rows, arm, g, 'v4x', n, row['worker'])
                log = (root / 'scaling/rss' / (tag + '.log')).read_text()
                measured = next(int(line.rsplit(':', 1)[1]) for line in log.splitlines()
                                if 'Maximum resident set size (kbytes)' in line)
                assert row['max_rss_kib'] == measured
                memories[tag] = measured
    verdicts = {}
    for region, threads in [('scaling', [8, 16, 32]), ('low_threads', [1, 2, 4])]:
        verdicts[region] = {}
        for arm in cmp.SCALING_ARMS[1:]:
            counts = dict(after_faster=0, after_slower=0, inconclusive=0)
            for g, t, n in cmp.cells(scaling=True):
                if n not in threads:
                    continue
                ci = analyses[f'{t}-t{n}-{g}']['comparisons'][arm]
                label = ('after_faster' if ci['ci_lower'] > 0 and not ci['resolution_limited'] else
                         'after_slower' if ci['ci_upper'] < 0 and not ci['resolution_limited'] else 'inconclusive')
                counts[label] += 1
            verdicts[region][arm] = counts
    result = dict(status='MEASURED', strict_parity=parity, timing_configurations=48, arm_timings=240,
                  rss_observations=48, medians_ns=medians, paired_analyses=analyses,
                  verdicts=verdicts, round_selection=selections, rss_kib=memories,
                  validated_parent_batches=batches, paired_analysis_replays=len(replay),
                  axes=dict(tiers=cmp.TIERS, threads=cmp.SCALING_THREADS,
                            geometries=cmp.SCALING_GEOMETRIES, arms=cmp.SCALING_ARMS),
                  builds=builds, model_sha256=owner.SOURCE_SHA, raw_directory=str(root),
                  interval_scope='pointwise paired 95% CI, peer minus after Rev5; not simultaneous grid-wide intervals')
    owner.write(args.out_json, result)
    lines = ['# Rev5 ordered strip batch measurements', '',
             'Twenty-four requested scaling cells and twenty-four 1/2/4-thread controls use the COSTCMP collector. Each cell pairs after Rev5, Rev4, serving A/B and the frozen before Rev5 executable on identical SPEEDQ RGB8 inputs. Setup is untimed; complete scoring calls include their allocations. Model bytes and input hashes are checked before measurement.', '',
             'All cells use the exclusive segment lock, original quiet gate and first 32 shared clean rounds from at most 64. Statistics are replayed through the frozen zenbench paired-round analyzer. The intervals below are pointwise bootstrap intervals of IQR-filtered mean paired differences (peer minus after); warm medians use all 32 admitted rounds. Positive intervals mean after costs less. These inputs measure scoring cost; quality and corpus-wide performance are not measured.', '',
             '| tier | threads | size | after ms | before ms [95% CI] | Rev4 ms [95% CI] | A ms [95% CI] | B ms [95% CI] |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for g, t, n in cmp.cells(scaling=True):
        tag = f'{t}-t{n}-{g}'
        entries = [f'{medians[tag][cmp.SCALING_ARMS[0]]/1e6:.6f}']
        for arm in ('by_v2fy_r5_before', 'by_v2fy_r4', 'zensim_A', 'zensim_B'):
            ci = analyses[tag]['comparisons'][arm]
            entries.append(f"{medians[tag][arm]/1e6:.6f} [{ci['ci_lower']/1e6:.6f}, {ci['ci_upper']/1e6:.6f}]" +
                           (' limited' if ci['resolution_limited'] else ''))
        lines.append(f'| {t} | {n} | {g} | ' + ' | '.join(entries) + ' |')
    lines += ['', '## Fresh-process peak RSS', '',
              'KiB measured by `/usr/bin/time -v`; v4x only. Both binaries include identical model/input setup.', '',
              '| threads | size | before KiB | after KiB |', '|---:|---|---:|---:|']
    for g in cmp.SCALING_GEOMETRIES:
        for n in cmp.SCALING_THREADS:
            lines.append(f"| {n} | {g} | {memories[f'v4x-t{n}-{g}-by_v2fy_r5_before']} | {memories[f'v4x-t{n}-{g}-by_v2fy_r5']} |")
    lines += ['', f'Raw evidence: `{root}`. Exact pins, full stdout/stderr, profiles and all admitted/excluded rounds remain in that directory.', '']
    args.out_md.write_text('\n'.join(lines))
    return 0
