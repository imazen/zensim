#!/usr/bin/env python3
"""COSTSET3 before/after bit-identity gate. Runs the existing `diffmap_block_coherence` owner (prepared steering) with
the pre-change binary and the post-change binary over the owner-12 panel (single bake, block 32) for each arm x seed
x bake set, and compares every JSON field exactly. The broad-96 panel is run by `run_broad.py` (STEERCHECK_PREPARED=1,
STEERCHECK_REV) for both binaries; `compare_broad` compares those RESULT rows exactly.

usage: gate.py owner <old-bin> <new-bin> <bakes-dir> <rev> <out-dir>
       gate.py broad <old-out> <new-out>
"""
import json, os, subprocess, sys
from pathlib import Path

ARMS = ['v2basic', 'by_v2fy', 'b228_v2s123', 'b156_v2s123_y']
SEEDS = (5101, 5103, 5107)


def owner(old, new, bakes, rev, out):
    cases = json.load(open('/home/lilith/work/zensim-validation-2026-09-13/minimal-top/spatial-eval.json'))['cases']
    env = dict(os.environ, RAYON_NUM_THREADS='1', ZENSIM_FORMULA_REV=rev, ZENSIM_PREPARED_STEERING='1')
    out = Path(out)
    n = bad = 0
    for arm in ARMS:
        for seed in SEEDS:
            for c in cases:
                docs = []
                for tag, tool in (('old', old), ('new', new)):
                    p = out / tag / f'{arm}-s{seed}-{c["name"]}.json'
                    p.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(['taskset', '-c', '8-15', tool, c['reference'], c['distorted'], '--bake',
                                    f'{bakes}/human-{arm}-h128-full-s{seed}.bin', '--block', '32', '--json', str(p)],
                                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    docs.append(json.load(open(p)))
                n += 1
                if docs[0] != docs[1]:
                    bad += 1
                    print('DIFF', arm, seed, c['name'], flush=True)
    print(f'owner {bakes} rev{rev}: {n} cases, {bad} differing')
    return bad


def broad(old, new):
    a, b = (json.load(open(Path(p) / 'RESULT.json'))['rows'] for p in (old, new))
    assert len(a) == len(b)
    keys = ('model', 'index', 'block', 'score', 'm2', 'm3f', 'interventions', 'status', 'unsupported_ids')
    bad = [x['report'] for x, y in zip(a, b) if any(x[k] != y[k] for k in keys)]
    # full per-case reports too (every field)
    for x in a:
        if json.load(open(Path(old) / x['report'])) != json.load(open(Path(new) / x['report'])):
            bad.append(x['report'])
    print(f'broad {old} vs {new}: {len(a)} rows, {len(set(bad))} differing')
    return len(set(bad))


if __name__ == '__main__':
    sys.exit(1 if (owner(*sys.argv[2:7]) if sys.argv[1] == 'owner' else broad(*sys.argv[2:4])) else 0)
