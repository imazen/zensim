#!/usr/bin/env python3
"""Profile the frozen SPEEDQ serving action; these are not timing receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import speedq_run as speedq

# Worst percentage slowdown in each reported class, production SPEEDQ grid.
CELLS = [('scalar', 8, '1920x1080', 16),
         ('v4', 8, '1920x1080', 64),
         ('v4', 8, '4096x4096', 16),
         ('v4', 2, '64x64', 8192)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--mode', choices=['perf', 'callgrind', 'phases', 'heaptrack'], required=True)
    ap.add_argument('--only', help='comma-separated production grid tags; defaults to the original four cells')
    ap.add_argument('--calls', type=int, help='explicit profiler-only extra scoring calls')
    args = ap.parse_args()
    cells = CELLS
    if args.only:
        grid = {f'{tier}-t{threads}-{geometry}': (tier, threads, geometry,
                8192 if geometry == '64x64' else 2048)
                for tier in speedq.TIERS for threads in speedq.THREADS
                for geometry in speedq.GEOMETRIES}
        tags = args.only.split(',')
        if len(set(tags)) != len(tags) or any(tag not in grid for tag in tags):
            ap.error('--only requires unique production grid tags')
        cells = [grid[tag] for tag in tags]
    if args.calls is not None and args.calls < 0:
        ap.error('--calls must be nonnegative')
    args.dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(binary=str(args.binary.resolve()),
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    mode=args.mode, cells=cells, extra_calls=args.calls, commands=[])
    try:
        for tier, threads, geometry, calls in cells:
            if args.mode == 'callgrind' and tier != 'scalar':
                continue
            for rev in (4, 5):
                tag = f'{tier}-t{threads}-{geometry}-r{rev}'
                env = speedq.environment(geometry, tier, threads)
                env.update(ZEN_S2_SPEEDQ_WORKER='by_v2fy',
                           ZENSIM_FORMULA_REV=str(rev), ZEN_S2_RSS_ONLY='1',
                           ZEN_S2_PROFILE_CALLS=str(args.calls if args.calls is not None else calls if args.mode == 'perf' else 0))
                if args.mode == 'phases':
                    env['ZENSIM_FOLD_TIMING'] = '1'
                worker = ['taskset', '-c', speedq.CPUSETS[threads], str(args.binary.resolve())]
                if args.mode == 'perf':
                    commands = [
                        ['perf', 'stat', '-e', 'cycles,instructions,branches,branch-misses,cache-misses',
                         '-o', str(args.dest / (tag + '.stat'))] + worker,
                        ['perf', 'record', '-F', '997', '-g', '-o',
                         str(args.dest / (tag + '.data'))] + worker]
                elif args.mode == 'callgrind':
                    commands = [['valgrind', '--tool=callgrind', '--trace-children=yes', '--dump-instr=yes',
                                 '--callgrind-out-file=' + str(args.dest / (tag + '.callgrind'))] + worker]
                elif args.mode == 'heaptrack':
                    commands = [['taskset', '-c', speedq.CPUSETS[threads],
                                 'heaptrack', '--record-only', '-o',
                                 str(args.dest / (tag + '.heaptrack')), str(args.binary.resolve())]]
                else:
                    commands = [worker]
                for i, cmd in enumerate(commands):
                    print(tag, args.mode, i, flush=True)
                    with (args.dest / f'{tag}.{i}.log').open('x') as log:
                        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    manifest['commands'].append(dict(tag=tag, command=cmd, returncode=result.returncode))
                    if result.returncode:
                        raise RuntimeError(f'profiler failed: {tag} (see full log)')
                if args.mode == 'phases':
                    continue
                if args.mode == 'perf':
                    report = ['perf', 'report', '--stdio', '--no-children', '-i',
                              str(args.dest / (tag + '.data'))]
                elif args.mode == 'heaptrack':
                    traces = list(args.dest.glob(tag + '.heaptrack*'))
                    if len(traces) != 1:
                        raise RuntimeError('one heaptrack trace required')
                    report = ['heaptrack_print', '-f', str(traces[0]), '-n', '20']
                else:
                    report = ['callgrind_annotate', '--inclusive=no', '--threshold=99',
                              str(args.dest / (tag + '.callgrind'))]
                with (args.dest / (tag + '.report')).open('x') as log:
                    subprocess.run(report, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        (args.dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
