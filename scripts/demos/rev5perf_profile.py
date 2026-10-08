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
    ap.add_argument('--mode', choices=['perf', 'callgrind'], required=True)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(binary=str(args.binary.resolve()),
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    mode=args.mode, cells=CELLS, commands=[])
    try:
        for tier, threads, geometry, calls in CELLS:
            if args.mode == 'callgrind' and tier != 'scalar':
                continue
            for rev in (4, 5):
                tag = f'{tier}-t{threads}-{geometry}-r{rev}'
                env = speedq.environment(geometry, tier, threads)
                env.update(ZEN_S2_SPEEDQ_WORKER='by_v2fy',
                           ZENSIM_FORMULA_REV=str(rev), ZEN_S2_RSS_ONLY='1',
                           ZEN_S2_PROFILE_CALLS=str(calls if args.mode == 'perf' else 0))
                worker = ['taskset', '-c', speedq.CPUSETS[threads], str(args.binary.resolve())]
                if args.mode == 'perf':
                    commands = [
                        ['perf', 'stat', '-e', 'cycles,instructions,branches,branch-misses,cache-misses',
                         '-o', str(args.dest / (tag + '.stat'))] + worker,
                        ['perf', 'record', '-F', '997', '-g', '-o',
                         str(args.dest / (tag + '.data'))] + worker]
                else:
                    commands = [['valgrind', '--tool=callgrind', '--dump-instr=yes',
                                 '--callgrind-out-file=' + str(args.dest / (tag + '.callgrind'))] + worker]
                for i, cmd in enumerate(commands):
                    print(tag, args.mode, i, flush=True)
                    with (args.dest / f'{tag}.{i}.log').open('x') as log:
                        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    manifest['commands'].append(dict(tag=tag, command=cmd, returncode=result.returncode))
                    if result.returncode:
                        raise RuntimeError(f'profiler failed: {tag} (see full log)')
                if args.mode == 'perf':
                    report = ['perf', 'report', '--stdio', '--no-children', '-i',
                              str(args.dest / (tag + '.data'))]
                else:
                    report = ['callgrind_annotate', '--inclusive=no', '--threshold=99',
                              str(args.dest / (tag + '.callgrind'))]
                with (args.dest / (tag + '.report')).open('x') as log:
                    subprocess.run(report, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        (args.dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
