#!/usr/bin/env python3
"""MAPPERF transport: reuse SPEEDQ's quiet/paired-round measurement owner."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from peergates import read, sha, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos'))
import speedq_run as owner

ARMS = ['e34_butter_map', 'e34_candidate_map_retained', 'e34_candidate_map_before']
GEOMETRIES = ['64x64', '256x256', '1024x1024', '2048x2048']
AUDIT_KEYS = ['density_sha256', 'features_gradient_sha256', 'queries_sha256', 'refinement_sha256']


def refresh_activity(activity):
    """Keep the owner's check, publishing our shared lane heartbeat atomically."""
    from datetime import datetime, timezone
    name = os.environ.get('SPEEDQ_LOCK_OWNER', 'codex SPEEDQ')
    path = Path(owner.__file__).resolve().parents[2] / '.workongoing'
    if not path.exists() or f' {name}' not in path.read_text():
        raise RuntimeError('SPEEDQ workspace lock ownership changed')
    temporary = path.with_name(f'.workongoing.{os.getpid()}.new')
    temporary.write_text(datetime.now(timezone.utc).isoformat() + f' {name} ' + activity + '\n')
    os.replace(temporary, path)


owner.refresh_activity = refresh_activity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cycle', 'parity', 'timing', 'profile', 'report', 'native-cache-prepare', 'native-cache-run', 'native-cache-compare'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model')
    p.add_argument('--binary', type=Path)
    a = p.parse_args()
    root = a.root
    if a.mode == 'cycle':
        assert a.model
        command = [sys.executable, __file__]
        with owner.segment_lock(Path.home() / 'tmp/zensim-paper/rev4/heavy.lock'):
            subprocess.run(command + ['parity', '--root', str(root), '--model', a.model], check=True)
        # The timing owner acquires/releases this same lock per segment.
        subprocess.run(command + ['timing', '--root', str(root), '--model', a.model], check=True)
        return
    if a.mode.startswith('native-cache-'):
        native_cache(root, a.mode, a.model, a.binary)
        return
    job = read(root / 'JOB.json')
    owner.CPUSETS[1] = str(job['cpuset'])
    if 'benchmark_cache' in job:
        os.environ['XDG_CACHE_HOME'] = str(root / job['benchmark_cache'])
    assert owner.CPUSETS[1].isdigit()
    if a.mode == 'report':
        report(root, job, a.model)
        return
    assert a.model in job['models'], 'explicit registered model required'
    binding = job['models'][a.model]
    binaries = {key: root / value['path'] for key, value in job['binaries'].items()}
    for key, binary in binaries.items():
        assert sha(binary) == job['binaries'][key]['sha256'], key
    model = root / binding['path']
    assert sha(model) == binding['sha256']
    os.environ.update(
        SPEEDQ_LOCK_OWNER='codex-MAPPERF', ZENSIM_NEIGHBOUR_EXACT='1',
        ZEN_S2_E34_BAKE_CANDIDATE=str(model),
        ZEN_S2_E34_SHA_CANDIDATE=binding['sha256'],
        ZEN_S2_E34_HIDDEN_CANDIDATE=str(binding['hidden']),
    )
    os.environ.pop('ZENSIM_MAX_TIER', None)
    if a.mode == 'profile':
        dest = root / 'profiles' / a.model
        dest.mkdir(parents=True, exist_ok=False)
        with owner.segment_lock(Path.home() / 'tmp/zensim-paper/rev4/heavy.lock'):
            owner.refresh_activity('quiet gate before profile ' + a.model)
            gate = owner.quiet_gate(dest / 'quiet-waits.jsonl')
            env = owner.environment('1024x1024', 'v4x', 1)
            env.update(ZEN_S2_SPEEDQ_WORKER='e34_candidate_map_retained',
                       ZENSIM_FORMULA_REV='5', ZEN_S2_RSS_ONLY='1', ZEN_S2_PROFILE_CALLS='100')
            command = ['taskset', '-c', owner.CPUSETS[1], 'nice', '-n19', 'ionice', '-c3',
                       'perf', 'record', '-F', '199', '-g', '-o', str(dest / 'perf.data'),
                       '--', str(binaries['after'])]
            with (dest / 'record.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (dest / 'report.txt').open('x') as log:
                subprocess.run(['perf', 'report', '--stdio', '--no-children', '-i', str(dest / 'perf.data')],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            write(dest / 'PROVENANCE.json', dict(job_sha256=sha(root / 'JOB.json'), quiet_gate=gate,
                  command=command, calls=100, cpuset=job['cpuset'], model_sha256=binding['sha256'],
                  binary_sha256=job['binaries']['after']['sha256']))
        return
    dest = root / 'results' / a.model
    dest.mkdir(parents=True, exist_ok=True)
    parity = dest / 'PREFLIGHT_PASS.json'

    def load(path):
        data = read(path)
        assert data['status'] == 'PASS' and data['job_sha256'] == sha(root / 'JOB.json')
        records = {(r['geometry'], r['arm']): r['record'] for r in data['records']}
        assert set(records) == {(g, arm) for g in GEOMETRIES for arm in ARMS}
        return records

    def ready(rows, name, g, t, n, rec):
        old = rows[g, name]
        assert (rec['arm'], rec['tier'], int(rec['threads']), rec['width'], rec['height']) == (
            name, t, n, *map(int, g.split('x')))
        assert all(rec[k] == old[k] for k in ['input_sha256', 'score_bits', 'model'])

    if a.mode == 'parity':
        records = []
        audits = []
        golden_path = root / binding['golden']
        assert sha(golden_path) == binding['golden_sha256']
        golden = read(golden_path)['records']
        for geometry in GEOMETRIES:
            inputs = set()
            for arm in ARMS:
                binary = binaries['before'] if arm.endswith('_before') else binaries['after']
                rec = owner.worker_run(binary, arm, 5, geometry, 'v4x', 1, dest / f'{geometry}-{arm}.log')
                records.append(dict(geometry=geometry, arm=arm, record=rec))
                inputs.add(rec['input_sha256'])
            assert len(inputs) == 1
            audit = {}
            for build in ['before', 'after']:
                rec = owner.worker_run(binaries[build], 'e34_candidate_map_audit', 5, geometry, 'v4x', 1,
                                       dest / f'{geometry}-audit-{build}.log')
                audit[build] = rec
                assert rec['input_sha256'] in inputs
                scalar = owner.worker_run(binaries[build], 'e34_candidate', 5, geometry, 'v4x', 1,
                                         dest / f'{geometry}-scalar-{build}.log')
                assert scalar['score_bits'] == rec['score_bits']
            assert all(audit['before'][k] == audit['after'][k] for k in ['input_sha256', 'score_bits'])
            before_model, after_model = audit['before']['model'], audit['after']['model']
            omitted = binding.get('omitted_reference_only_ids', [])
            if omitted:
                # The owner approved the declared-subset correction: exactly
                # these unread reference-only positions become structural zero.
                assert binding['sha256'] == '7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275'
                assert omitted == [422,480,509,538,567,596,625,654,683,712]
                assert before_model.keys() == after_model.keys()
                for record in [before_model, after_model]:
                    raw = b''.join(int(v,16).to_bytes(8,'little') for k in ['feature_bits','gradient_bits'] for v in record[k])
                    import hashlib
                    assert hashlib.sha256(raw).hexdigest() == record['features_gradient_sha256']
                assert before_model['gradient_bits'] == after_model['gradient_bits']
                changed = [i for i,(x,y) in enumerate(zip(before_model['feature_bits'],after_model['feature_bits'],strict=True)) if x!=y]
                assert changed == omitted
                assert all(after_model['feature_bits'][i] == after_model['gradient_bits'][i] == '0000000000000000' for i in omitted)
                unchanged = set(before_model)-{'feature_bits','features_gradient_sha256'}
                assert all(before_model[k] == after_model[k] for k in unchanged)
            else:
                assert before_model == after_model
            assert all(before_model[k] == after_model[k] for k in AUDIT_KEYS if k!='features_gradient_sha256' or not omitted)
            if geometry in ['1024x1024', '2048x2048']:
                prior = next(r for r in golden if r['width'] == int(geometry.split('x')[0])
                             and r['tier'] == 'v4x' and r['arm'] == 'e34_candidate_map')
                assert all(prior[k] == audit['before'][k] for k in ['input_sha256', 'score_bits'])
            for row in records:
                if row['geometry'] == geometry and row['arm'] != 'e34_butter_map':
                    assert row['record']['score_bits'] == audit['before']['score_bits']
            audits.append(dict(geometry=geometry, before=audit['before'], after=audit['after']))
            print(f'bit-exact map/score/features/gradient/queries/refinement: {geometry}', flush=True)
        write(parity, dict(status='PASS', job_sha256=sha(root / 'JOB.json'), records=records, audits=audits))
        load(parity)
    else:
        geometry_order = job.get('timing_geometry_order', GEOMETRIES)
        assert len(geometry_order) == len(GEOMETRIES) and set(geometry_order) == set(GEOMETRIES)
        owner.timing(
            binaries['after'], dest / 'timing', 32, parity, binaries['analyzer'], arms=ARMS,
            cells=[(g, 'v4x', 1) for g in geometry_order], baseline='e34_butter_map',
            parity_loader=load, ready_check=lambda *args: ready(load(parity), *args),
            worker_binaries={'e34_candidate_map_before': binaries['before']},
            lock=Path.home() / 'tmp/zensim-paper/rev4/heavy.lock',
        )


def report(root, job, selected=None):
    from speed_matrix_report import least_squares
    from speedq_report_support import verify_speedq_batches

    panel = root / job['binaries']['panel']['path']
    assert sha(panel) == job['binaries']['panel']['sha256']
    result = dict(models={}, job_sha256=sha(root / 'JOB.json'), exploratory=True,
                  qualification_allowed=False, host=job['host'], cpuset=str(job['cpuset']), threads=1, tier='v4x',
                  complete_model_grid=selected is None)
    lines = ['# Complete score and map performance', '',
             f"Same deterministic RGB8 pairs, complete retained outputs, cached references, CPU{job['cpuset']}/v4x/one thread. "
             'First32cleanrounds of at most64; unchanged SPEEDQ quiet admission. '
             'Paired p95 intervals use the existing Rust owner: 10000round resamples, seed20261016, nominal95%, '
             'type7. Exploratory engineering evidence on one deterministic pair per geometry; no content-wide claim.', '',
             '| model | size | before p95 ms | after p95 ms | butter p95 ms | after−before95% CI ms | after−butter95% CI ms |',
             '|---|---|---:|---:|---:|---|---|']
    assert selected is None or selected in job['models']
    for model in ([selected] if selected else job['models']):
        base = root / 'results' / model
        parity = read(base / 'PREFLIGHT_PASS.json')
        assert parity['status'] == 'PASS' and parity['job_sha256'] == sha(root / 'JOB.json')
        rows = {}
        medians = {arm: [] for arm in ARMS}
        for g in GEOMETRIES:
            dest = base / 'timing' / f'v4x-t1-{g}'
            done = read(dest / 'COMPLETE.json')
            header = read(dest / 'header.json')
            inner = read(dest / 'zenbench.inner.json')
            values, selection = owner.select_clean_rounds(inner)
            assert all(done[k] == v for k, v in selection.items())
            assert header['quiet_gate']['admitted'] and header['arms'] == ARMS
            assert header['cpuset'] == str(job['cpuset'])
            assert header['worker_binary_sha256']['e34_candidate_map_before'] == job['binaries']['before']['sha256']
            assert header['binary_sha256'] == job['binaries']['after']['sha256']
            assert read(dest / 'interference.json')['admitted']
            batches = verify_speedq_batches(dest, inner)

            def comparison(baseline, candidates, tag):
                inp = dest / f'{tag}-input.json'
                out = dest / f'{tag}-result.json'
                packet = dict(schema='addrfix-map-cost-v1', baseline_ns=values[baseline],
                              candidates_ns={c: values[c] for c in candidates})
                if inp.exists():
                    assert read(inp) == packet
                else:
                    write(inp, packet)
                if not out.exists():
                    subprocess.run([str(panel), 'e34', 'map-cost', '--input', str(inp), '--output', str(out)], check=True)
                return read(out)

            peer = comparison(ARMS[0], ARMS[1:], 'butter')
            change = comparison(ARMS[2], [ARMS[1]], 'change')
            rows[g] = dict(peer=peer, change=change, selection=selection, validated_batches=batches)
            for arm in ARMS:
                medians[arm].append(statistics.median(values[arm]))
            before = peer['comparisons'][ARMS[2]]['p95_ns'] / 1e6
            after = peer['comparisons'][ARMS[1]]['p95_ns'] / 1e6
            butter = peer['baseline_p95_ns'] / 1e6
            def interval(iv):
                return f"[{iv['lo']/1e6:.6f}, {iv['hi']/1e6:.6f}]"
            lines.append(f'| {model} | {g} | {before:.6f} | {after:.6f} | {butter:.6f} | '
                         f"{interval(change['comparisons'][ARMS[1]]['delta_ns'])} | "
                         f"{interval(peer['comparisons'][ARMS[1]]['delta_ns'])} |")
        pixels = [int(g.split('x')[0]) * int(g.split('x')[1]) for g in GEOMETRIES]
        fits = {}
        for arm, times in medians.items():
            alpha, beta = least_squares(pixels, times)
            fits[arm] = dict(alpha_ns=alpha, beta_ns_per_pixel=beta,
                             residuals_ns=[y-alpha-beta*x for x, y in zip(pixels, times)])
        result['models'][model] = dict(cells=rows, median_fits=fits, parity_sha256=sha(base / 'PREFLIGHT_PASS.json'))
    lines += ['', 'Descriptive unconstrained OLS of median time_ns=alpha_ns+beta_ns_per_pixel*pixels '
              'through the existing speed_matrix_report.least_squares owner. Negative intercepts are fit artifacts.', '',
              '| model | arm | alpha ms | beta ns/pixel |', '|---|---|---:|---:|']
    for model, record in result['models'].items():
        for arm, fit in record['median_fits'].items():
            lines.append(f"| {model} | {arm} | {fit['alpha_ns']/1e6:.6f} | {fit['beta_ns_per_pixel']:.6f} |")
    suffix = f'-{selected}' if selected else ''
    write(root / f'RESULTS{suffix}.json', result)
    (root / f'REPORT{suffix}.md').write_text('\n'.join(lines) + '\n')


def native_cache(root, mode, selected, binary):
    old = Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part6')
    results = {}
    for name in ([selected] if mode == 'native-cache-run' else ['a', 'c']):
        assert name in ['a', 'c']
        dest = root / 'native-cache' / name
        dest.mkdir(parents=True, exist_ok=True)
        source_job = old / f'cached-gate-{name}' / 'job.json'
        previous = old / f'cached-gate-{name}' / 'result.json'
        packet = read(source_job)
        assert len(packet['cases']) == 135 and packet['engine'] is True
        assert packet['repair_feature_cache'] == str(old / 'native-repair-cache.json')
        packet['output'] = str(dest / 'result.json')
        if mode == 'native-cache-prepare':
            if (dest / 'job.json').exists():
                assert read(dest / 'job.json') == packet
            else:
                write(dest / 'job.json', packet)
                write(dest / 'PINS.json', dict(source_job_sha256=sha(source_job), previous_sha256=sha(previous),
                                             cache_sha256=sha(packet['repair_feature_cache'])))
            cache = read(packet['repair_feature_cache'])
            assert len(cache['rows']) == 135
            for start in range(0, 135, 9):
                chunk = dest / f'chunk-{start:03}'
                chunk.mkdir(exist_ok=True)
                cache_part = {**cache, 'rows': cache['rows'][start:start+9]}
                part = {**packet, 'cases': packet['cases'][start:start+9],
                        'repair_feature_cache': str(chunk / 'cache.json'), 'output': str(chunk / 'result.json')}
                write(chunk / 'cache.json', cache_part)
                write(chunk / 'job.json', part)
        else:
            pins = read(dest / 'PINS.json')
            assert sha(previous) == pins['previous_sha256']
            assert sha(source_job) == pins['source_job_sha256']
            assert sha(packet['repair_feature_cache']) == pins['cache_sha256']
            prior = read(previous)
            rows = []
            for start in range(0, 135, 9):
                chunk = dest / f'chunk-{start:03}'
                if mode == 'native-cache-run':
                    assert binary is not None and binary.is_file()
                    os.environ.update(SPEEDQ_LOCK_OWNER='codex-MAPPERF')
                    owner.refresh_activity(f'native cache {name}/{start}')
                    env = {**os.environ, 'ZENSIM_FORMULA_REV': '5', 'ZENSIM_NEIGHBOUR_EXACT': '1',
                           'RAYON_NUM_THREADS': '1', 'STEERFIX_PACKET': str(chunk / 'job.json')}
                    home = Path.home()
                    cmd = ['flock', str(home / 'tmp/zensim-paper/rev4/heavy.lock'),
                           'flock', str(home / 'tmp/zen-heavy.lock'),
                           'flock', str(home / 'tmp/devin/heavy.lock'),
                           str(home / 'work/claudehints/scripts/run-heavy'), '--mem', '16G', '--jobs', '1', '--',
                           str(binary), 'metric::bake::steerfix_packet::engineering_packet', '--exact', '--nocapture']
                    from shipgates_admission import run_with_memory_admission
                    run_with_memory_admission(cmd, chunk / 'run.log', chunk / 'result.json', env=env)
                    write(chunk / 'BUILD.json', dict(binary_sha256=sha(binary)))
                current = read(chunk / 'result.json')
                assert len(current['rows']) == 9
                for new, old_row in zip(current['rows'], prior['rows'][start:start+9], strict=True):
                    assert new['repair_feature_cache'] == str(chunk / 'cache.json')
                    assert old_row['repair_feature_cache'] == packet['repair_feature_cache']
                    assert {k: v for k, v in new.items() if k != 'repair_feature_cache'} == {
                        k: v for k, v in old_row.items() if k != 'repair_feature_cache'
                    }, f'{name}/{start}: cached numerical result changed'
                rows.extend(current['rows'])
                print(f'native cache exact: {name}/{start} (9cases)', flush=True)
            assert len(rows) == len(prior['rows']) == 135
            results[name] = dict(rows=135, every_numerical_output_field_equal=True,
                                 sole_provenance_change='explicit nine-case cache location',
                                 chunk_result_sha256s=[sha(dest / f'chunk-{start:03}' / 'result.json') for start in range(0, 135, 9)], **pins)
    if mode == 'native-cache-compare':
        write(root / 'NATIVE_CACHE_PARITY.json', dict(status='PASS', models=results))


if __name__ == '__main__':
    main()
