"""Validate COSTCMP scaling and byte-budget receipts with its statistics owner."""
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


def check_shared_load_policy(row):
    """Shared-load RSS needs the coordinator tag; the misattributed one only on the original 36 records."""
    if row['rss_policy'] == owner.RSS_UNDER_LOAD_POLICY_RECORDED:
        assert row['quiet_gate']['utc'] < owner.RSS_UNDER_LOAD_POLICY_RECORDED_UNTIL, 'misattributed tag on a new record'
    else:
        assert row['rss_policy'] == owner.RSS_UNDER_LOAD_POLICY, 'unknown shared-load RSS policy'


def rss_measurement(root, grid, tag, rows, arm, geometry, threads, expected_sha, require_threads=False):
    row = json.loads((root/grid/'rss'/(tag+'.json')).read_text())
    if require_threads:
        assert row['worker']['actual_rayon_threads'] == threads, 'RSS worker did not record its pool size'
    if row.get('quiet_required', True):
        assert row['quiet_gate']['admitted'] and row['quiet_gate']['load1'] < 2
    else:
        assert grid == 'budget'
        check_shared_load_policy(row)
    assert row['binary_sha256'] == expected_sha
    assert (row['arm'],row['geometry'],row['tier'],row['threads']) == (arm,geometry,'v4x',threads)
    cmp.validate_ready(rows, arm, geometry, 'v4x', threads, row['worker'])
    log = (root/grid/'rss'/(tag+'.log')).read_text()
    measured = next(int(line.rsplit(':',1)[1]) for line in log.splitlines()
                    if 'Maximum resident set size (kbytes)' in line)
    assert row['max_rss_kib'] == measured and measured > 0
    assert any(line.strip() == 'Exit status: 0' for line in log.splitlines())
    return measured


def rss_only_report(args):
    """Coordinator-instructed shared-load RSS result with no timing or speed verdict."""
    assert args.budget_grid
    root = args.raw_dir
    rows = cmp.receipt(root/'budget/parity/PREFLIGHT_PASS.json', budget=True)
    inventory = cmp.budget_inventory(root/'provenance/budget-binaries.json')
    from rev5perf_report import compare_bits
    parity = {f'cap{mib}':compare_bits(root/'uncapped',root/f'cap{mib}') for mib in [64,128,256]}
    source = json.loads((root/'provenance/budget-source.json').read_text())
    assert sha(root/'provenance/paired-rounds-analyzer') == source['paired_analyzer_sha256']
    assert sha(root/'provenance/production-f16.bin') == owner.SOURCE_SHA
    accounting = [json.loads(line.split('REV5_JOB_ACCOUNTING ',1)[1]) for line in
                  (root/'provenance/byte-accounting.log').read_text().splitlines()
                  if line.startswith('REV5_JOB_ACCOUNTING ')]
    assert len(accounting) == 3 and {r['width'] for r in accounting} == {1024,4096,8192}
    for row in accounting:
        assert row['planes'] == 11 and row['float_bytes'] == 4 and row['fixed_bytes'] > 0
        assert row['per_job_bytes'] == row['width']*(row['strip_rows']+2*row['halo'])*11*4+row['fixed_bytes']
        row['slots'] = {str(mib):{str(n):min(n,16,mib*1024*1024//row['per_job_bytes'])
                                 for n in cmp.BUDGET_THREADS} for mib in [64,128,256]}
    memories = {}
    for g,t,n in cmp.cells(budget=True):
        for arm in cmp.BUDGET_ARMS:
            tag = f'{t}-t{n}-{g}-{arm}'
            memories[tag] = rss_measurement(root,'budget',tag,rows,arm,g,n,sha(inventory[arm]))
    result = dict(status='RSS_ONLY', speed='speed not yet measured', provisional_default_mib=128,
                  strict_parity=parity, rss_observations=len(memories), rss_kib=memories,
                  byte_accounting=accounting, required_timing_configurations=9,
                  validated_timing_configurations=0, medians_ns={}, paired_analyses={},
                  timing_completion_markers=len(list((root/'budget/timing').glob('*/COMPLETE.json'))),
                  binary_sha256={arm:sha(path) for arm,path in inventory.items()},
                  source_provenance=source, raw_directory=str(root),
                  rss_scope='fresh-process time -v under coordinator-instructed shared load; no speed inference',
                  rss_policy_note='the 36 records carry the tag '+repr(owner.RSS_UNDER_LOAD_POLICY_RECORDED)+'; the shared-load RSS run was the coordinator\'s instruction, not the owner\'s')
    owner.write(args.out_json,result)
    lines = ['# Rev5 byte-budget RSS (2026-10-09)', '',
             '**Memory record, written before timing.** 128 MiB was then the provisional default. The measured decision (256 MiB, floor 3) and the timing are in [the design note](rev5perf5_budget_design_2026-10-09.md) and [the timing report](rev5perf5_scaling_2026-10-09.md). Fresh-process RSS ran under shared load on the coordinator\'s instruction (the raw records\' `rss_policy` tag says "owner-approved"; that attribution is wrong). Timing still requires the unchanged load <2/no foreign build or training gate and 32 common clean rounds from at most 64. No timing medians or confidence intervals are reported.', '',
             'All three frozen candidates pass 384/384 strict parity and 576/576 comparisons overall. The full 36-record measurement preflight agrees in score and all 420 consumed feature bits.', '',
             '| threads | size | uncapped KiB | 64 MiB KiB | 128 MiB KiB | 256 MiB KiB |',
             '|---:|---|---:|---:|---:|---:|']
    for g,t,n in cmp.cells(budget=True):
        values = [str(memories[f'{t}-t{n}-{g}-{arm}']) for arm in cmp.BUDGET_ARMS]
        lines.append(f'| {n} | {g} | '+' | '.join(values)+' |')
    lines += ['', 'Each observation is a separate `/usr/bin/time -v` process with identical model/input setup. These are single observations, not distributions. Queue budgets exclude allocator overhead, input pixels and producer scratch.', '',
              'Actual x86_64 Rust `size_of` accounting: `per_job = 11 × width × (strip_rows + 2 × halo) × 4 + fixed_bytes`. In the measured binaries live slots are `min(threads, 16, floor(budget/per_job))`, and zero slots use the original route. The source now also takes the original route when fewer than three jobs fit (`REV5_MIN_JOB_SLOTS`), so the 1- and 2-slot cells below show the queued route\'s memory in binaries that predate that floor.', '',
              '| width | strip rows | halo | fixed bytes | per-job bytes | 64 MiB slots (8/16/32T) | 128 MiB slots | 256 MiB slots |',
              '|---:|---:|---:|---:|---:|---|---|---|']
    for r in accounting:
        slots = ['/'.join(str(r['slots'][str(m)][str(n)]) for n in cmp.BUDGET_THREADS) for m in [64,128,256]]
        lines.append(f"| {r['width']} | {r['strip_rows']} | {r['halo']} | {r['fixed_bytes']} | {r['per_job_bytes']} | "+' | '.join(slots)+' |')
    lines += ['', '128 MiB retains more jobs than 64 MiB while measuring lower peak RSS than 256 MiB on both large geometries. The later timing chose 256 MiB for speed; see the design note.', '',
              f'Raw evidence and binary/source/model pins: `{root}`. Full time-v logs and exact byte accounting are replayed by the report recipe. Timing measurements remain pending.', '']
    args.out_md.write_text('\n'.join(lines))


def floor_decision(root, cells, medians, direct_pairs):
    """Per cell: 128 MiB queue slots and the paired queue/floor-3 versus original-route intervals."""
    accounting = [json.loads(line.split('REV5_JOB_ACCOUNTING ', 1)[1]) for line in
                  (root/'provenance/byte-accounting.log').read_text().splitlines()
                  if line.startswith('REV5_JOB_ACCOUNTING ')]
    shape = {(r['strip_rows'], r['halo'], r['fixed_bytes']) for r in accounting}
    assert len(shape) == 1
    strip_rows, halo, fixed = shape.pop()
    rows = []; by_slots = {}
    for g, t, n in cells:
        tag = f'{t}-t{n}-{g}'
        width = int(g.split('x')[0])
        per_job = width*(strip_rows+2*halo)*11*4+fixed
        slots = min(n, 16, 128*1024*1024//per_job)
        q = direct_pairs[tag][cmp.BUDGET_ORIGINAL_ARM+'->by_v2fy_r5_b128']
        f = direct_pairs[tag][cmp.BUDGET_ORIGINAL_ARM+'->'+cmp.BUDGET_FLOOR_ARM]
        verdict = ('queue_faster' if q['ci_upper'] < 0 and not q['resolution_limited'] else
                   'queue_slower' if q['ci_lower'] > 0 and not q['resolution_limited'] else 'inconclusive')
        by_slots.setdefault(slots, dict(queue_faster=0, queue_slower=0, inconclusive=0))[verdict] += 1
        rows.append(dict(geometry=g, threads=n, slots=slots, queue_verdict=verdict,
                         original_ms=medians[tag][cmp.BUDGET_ORIGINAL_ARM]/1e6, queue_ms=medians[tag]['by_v2fy_r5_b128']/1e6,
                         floor3_ms=medians[tag][cmp.BUDGET_FLOOR_ARM]/1e6,
                         queue_minus_original=q, floor3_minus_original=f))
    return dict(cells=rows, by_slots={k: by_slots[k] for k in sorted(by_slots)},
                rule='queue verdict from the paired 95% CI of frozen 128 MiB minus floor 17 (original route)')


def report(args):
    if getattr(args, 'rss_only', False):
        return rss_only_report(args)
    root = args.raw_dir
    budget = getattr(args, 'budget_grid', False)
    floor = getattr(args, 'floor_arm', False)
    grid = 'budget' if budget else 'scaling'
    tiers, threads, geometries, arms, _ = cmp.configuration(scaling=not budget, budget=budget, floor=floor)
    cells = cmp.cells(scaling=not budget, budget=budget, floor=floor)
    rows = cmp.receipt(root / grid / ('parity-v3' if floor else 'parity') / 'PREFLIGHT_PASS.json',
                       scaling=not budget, budget=budget, floor=floor)
    from rev5perf_report import compare_bits
    labels = (('uncapped', 'cap64', 'cap128', 'cap256') + (('floor3-grid', 'floor17-grid') if floor else ())
              if budget else ('before', 'verified'))
    parent = ('floor3-grid' if floor else 'cap128') if budget else 'verified'
    parity = ({label: compare_bits(root/'uncapped', root/label) for label in labels[1:]}
              if budget else compare_bits(root/'before', root/'verified'))
    builds = {label: json.loads((root / f'provenance/instrument-{label}.artifact.json').read_text())
              for label in labels}
    assert all(builds[label]['dependencies'] == builds[labels[0]]['dependencies'] for label in labels)
    for label, artifact in builds.items():
        assert sha(root / f'provenance/instrument-{label}') == artifact['binary_sha256']
    medians = {}; analyses = {}; selections = {}; packets = []; saved = []; batches = 0
    if budget:
        inventory = cmp.budget_inventory(root/'provenance'/('budget-binaries-v3.json' if floor else 'budget-binaries.json'), floor)
        expected_workers = {arm: sha(path) for arm,path in inventory.items()}
    else:
        expected_workers = {arm: builds['before' if arm == 'by_v2fy_r5_before' else 'verified']['binary_sha256'] for arm in arms}
    paired_values = {}
    for g, t, n in cells:
        tag = f'{t}-t{n}-{g}'
        dest = root / grid / 'timing' / tag
        done = json.loads((dest / 'COMPLETE.json').read_text())
        header = json.loads((dest / 'header.json').read_text())
        assert done['status'] == 'PASS' and done['rounds'] == 32
        assert done['paired_alignment_verified'] and done['zenbench_gate_clean']
        assert header['quiet_gate']['admitted'] and header['quiet_gate']['load1'] < 2
        assert not header['gate_trace'] and header['cpuset'] == owner.CPUSETS[n]
        assert header['binary_sha256'] == builds[parent]['binary_sha256']
        assert header['worker_binary_sha256'] == expected_workers
        assert header['arms'] == arms
        assert header['round_cap'] == 64 and header['round_rule'] == owner.CLEAN_ROUND_RULE
        assert header['model_source_sha256'] == owner.SOURCE_SHA
        interference = json.loads((dest / 'interference.json').read_text())
        assert interference['admitted'] and not interference['foreign']
        inner = json.loads((dest / 'zenbench.inner.json').read_text())
        values, selection = owner.select_clean_rounds(inner, 32)
        assert set(values) == set(arms)
        assert all(done[k] == v for k, v in selection.items())
        for name in arms:
            cmp.validate_ready(rows, name, g, t, n, inner['workers'][name])
        batches += verify_speedq_batches(dest, inner)
        analysis = json.loads((dest / 'paired_analysis.json').read_text())
        assert analysis['baseline_arm'] == arms[0]
        assert set(analysis['comparisons']) == set(arms[1:])
        for arm in arms[1:]:
            ci = analysis['comparisons'][arm]
            assert ci['n_samples'] == 32 and ci['ci_lower'] <= ci['ci_median'] <= ci['ci_upper']
            packets.append(dict(baseline=values[arms[0]], candidate=values[arm],
                                iterations=[1]*32, timer_resolution_ns=inner['timer_resolution_ns']))
            saved.append(ci)
        medians[tag] = {name: statistics.median(v) for name, v in values.items()}
        paired_values[tag] = (values, inner['timer_resolution_ns'])
        analyses[tag] = analysis
        selections[tag] = selection
    analyzer = root / 'provenance/paired-rounds-analyzer'
    source = json.loads((root / ('provenance/budget-source.json' if budget else 'provenance/verified-source.json')).read_text())
    assert sha(analyzer) == source['paired_analyzer_sha256']
    replay = json.loads(subprocess.check_output([str(analyzer)], input=json.dumps(packets), text=True))
    assert replay == saved, 'saved paired analyses differ from exact replay'
    memories = {}
    # RSS: the four measured budget arms on their three geometries, plus (floor
    # grid) the two floor arms on every floor-grid geometry.
    rss_arms = cmp.BUDGET_ARMS if budget else ('by_v2fy_r5', 'by_v2fy_r5_before')
    rss_cells = [(arm, g, n) for g in (cmp.BUDGET_GEOMETRIES if budget else geometries) for n in threads for arm in rss_arms]
    if floor:
        rss_cells += [(arm, g, n) for g in geometries for n in threads for arm in cmp.BUDGET_FLOOR_SLOTS]
    for arm, g, n in rss_cells:
        tag = f'v4x-t{n}-{g}-{arm}'
        memories[tag] = rss_measurement(root,grid,tag,rows,arm,g,n,expected_workers[arm],
                                        require_threads=arm in cmp.BUDGET_FLOOR_SLOTS)
    if budget:
        pairs = [('by_v2fy_r5_b64','by_v2fy_r5_b128'), ('by_v2fy_r5_b64','by_v2fy_r5_b256'), ('by_v2fy_r5_b128','by_v2fy_r5_b256')]
        if floor:
            pairs += [('by_v2fy_r5_b128', cmp.BUDGET_FLOOR_ARM), (cmp.BUDGET_ORIGINAL_ARM, 'by_v2fy_r5_b128'),
                      (cmp.BUDGET_ORIGINAL_ARM, cmp.BUDGET_FLOOR_ARM)]
        packets = []; coordinates = []
        for g,t,n in cells:
            tag = f'{t}-t{n}-{g}'
            values, resolution = paired_values[tag]
            for left,right in pairs:
                packets.append(dict(baseline=values[left],candidate=values[right],iterations=[1]*32,timer_resolution_ns=resolution))
                coordinates.append((tag,left+'->'+right))
        direct = json.loads(subprocess.check_output([str(analyzer)],input=json.dumps(packets),text=True))
        assert len(direct) == len(coordinates)
        direct_pairs = {}
        for (tag,pair),analysis in zip(coordinates,direct):
            direct_pairs.setdefault(tag,{})[pair] = analysis
        decision = floor_decision(root, cells, medians, direct_pairs) if floor else None
        result = dict(status='MEASURED',strict_parity=parity,floor_decision=decision,timing_configurations=len(cells),
                      arm_timings=len(cells)*len(arms),rss_observations=len(memories),medians_ns=medians,
                      paired_analyses=analyses,direct_budget_pairs=direct_pairs,round_selection=selections,
                      rss_kib=memories,validated_parent_batches=batches,paired_analysis_replays=len(replay),
                      axes=dict(tiers=tiers,threads=threads,geometries=geometries,arms=arms),builds=builds,
                      model_sha256=owner.SOURCE_SHA,raw_directory=str(root),
                      interval_scope='pointwise paired 95% CI, candidate minus baseline; not simultaneous grid-wide intervals')
        owner.write(args.out_json,result)
        titles = {'by_v2fy_r5_b64': '64 MiB', 'by_v2fy_r5_b128': '128 MiB', 'by_v2fy_r5_b256': '256 MiB',
                  cmp.BUDGET_FLOOR_ARM: '128 MiB + floor 3', cmp.BUDGET_ORIGINAL_ARM: 'original route'}
        lines = ['# Rev5 byte-budget measurements', '',
                 'Frozen 64/128/256 MiB candidates' + (', the 128 MiB source with floor 3 (shipped) and floor 17 (never queues: the original route),' if floor else '') + f' and an uncapped control share COSTCMP inputs, the exclusive lock, quiet gate and first 32 shared clean rounds from at most 64. All {len(cells)*(len(arms)-1)} saved comparisons replay through the frozen paired analyzer. Direct budget comparisons use that same analyzer and those same admitted rounds. Complete scoring calls include allocations; setup is untimed. Each build passes 384/384 strict parity against the uncapped control.', '',
                 '| threads | size | uncapped ms | ' + ' | '.join(titles[a]+' ms [95% CI]' for a in arms[1:]) + ' |',
                 '|---:|---|---:|' + '---:|'*(len(arms)-1)]
        for g,t,n in cells:
            tag=f'{t}-t{n}-{g}';entries=[f'{medians[tag][arms[0]]/1e6:.6f}']
            for arm in arms[1:]:
                ci=analyses[tag]['comparisons'][arm]
                entries.append(f"{medians[tag][arm]/1e6:.6f} [{ci['ci_lower']/1e6:.6f}, {ci['ci_upper']/1e6:.6f}]" + (' limited' if ci['resolution_limited'] else ''))
            lines.append(f'| {n} | {g} | '+' | '.join(entries)+' |')
        if floor:
            lines += ['', '## Floor decision: 128 MiB queue against the original route', '',
                      'Direct paired comparisons on the same admitted rounds, in ms: frozen 128 MiB (queue, no floor) minus the floor-17 build (original route), and the shipped floor-3 build minus the original route. Slots are the 128 MiB queue size at that width and thread count. Negative means the first build costs less.', '',
                      '| threads | size | slots | original ms | queue ms | queue − original [95% CI] | queue verdict | floor 3 − original [95% CI] |',
                      '|---:|---|---:|---:|---:|---|---|---|']
            for row in decision['cells']:
                q = row['queue_minus_original']; f = row['floor3_minus_original']
                lines.append(f"| {row['threads']} | {row['geometry']} | {row['slots']} | {row['original_ms']:.6f} | {row['queue_ms']:.6f} | "
                             f"[{q['ci_lower']/1e6:.6f}, {q['ci_upper']/1e6:.6f}]{' limited' if q['resolution_limited'] else ''} | {row['queue_verdict']} | "
                             f"[{f['ci_lower']/1e6:.6f}, {f['ci_upper']/1e6:.6f}]{' limited' if f['resolution_limited'] else ''} |")
            lines += ['', 'By slot count (all thread counts): ' + '; '.join(f"{k} slots: {v['queue_faster']} faster / {v['queue_slower']} slower / {v['inconclusive']} inconclusive" for k, v in decision['by_slots'].items()) + '.']
        lines += ['', 'Intervals are pointwise paired bootstrap differences, candidate minus uncapped, in ms. Negative intervals mean the candidate costs less. Warm medians use all 32 admitted rounds; analyzer means use its original IQR rule. These synthetic inputs measure scoring cost, not corpus-wide quality or performance.', '',
                  '## Fresh-process peak RSS', '',
                  'KiB measured by `/usr/bin/time -v`, including identical model/input setup. The byte budget bounds queue-owned allocations, not allocator overhead or total process RSS.', '',
                  '| threads | size | uncapped KiB | 64 MiB KiB | 128 MiB KiB | 256 MiB KiB |', '|---:|---|---:|---:|---:|---:|']
        rss_columns = list(rss_arms) + (list(cmp.BUDGET_FLOOR_SLOTS) if floor else [])
        if floor:
            lines[-2:] = ['| threads | size | ' + ' | '.join(('uncapped' if a.endswith('_before') else titles[a])+' KiB' for a in rss_columns) + ' |',
                          '|---:|---|' + '---:|'*len(rss_columns)]
        for g,t,n in cells:
            lines.append(f'| {n} | {g} | '+' | '.join(str(memories.get(f'{t}-t{n}-{g}-{arm}', '—')) for arm in rss_columns)+' |')
        lines += ['', f'Raw evidence: `{root}`. Model, binary/source pins, direct paired budget analyses, clean-round selections, full logs and strict receipts are preserved there.', '']
        args.out_md.write_text('\n'.join(lines))
        return 0
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
