#!/usr/bin/env python3
"""Re-run the 144 TRAIN pair × six-mode f0–f985 toggle identity gate."""

import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from datetime import datetime, timezone

GATE = Path('/var/tmp/featbank-impl/evidence/gate')
OUT = Path('/var/tmp/featbank-impl/identity_revision')
INPUTS = {
    'cid22': (GATE / 'cid22-64.tsv', 64),
    'safesyn': (GATE / 'safesyn-64.tsv', 64),
    'kadid': (GATE / 'kadid-train16-pairs.tsv', 16),
}
MODES = {
    't1': ('native', '1'),
    't8': ('native', '8'),
    'v3t1': ('v3', '1'),
    'v3t8': ('v3', '8'),
    'scalart1': ('scalar', '1'),
    'scalart8': ('scalar', '8'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def run(binary, corpus, input_path, mode, tier, threads, regime, binary_sha):
    stem = f'{corpus}-{mode}-{regime}'
    output = OUT / f'{stem}.csv'
    cmd = [str(binary), '--corpus', 'pairs-tsv', '--path', str(input_path),
           '--out', str(output), '--full-' + regime, '--force-tier', tier]
    env = dict(os.environ, ZENSIM_FORMULA_REV='3', ZENSIM_ROOT_FORM='sqrt',
               RAYON_NUM_THREADS=threads)
    start = now()
    with (OUT / f'{stem}.stdout').open('wb') as stdout, \
            (OUT / f'{stem}.stderr').open('wb') as stderr:
        rc = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr,
                            check=False).returncode
    end = now()
    manifest = Path(str(output) + '.manifest.json')
    info = {'start_utc': start, 'end_utc': end, 'cwd': str(Path.cwd()),
            'argv': cmd, 'env': {'ZENSIM_FORMULA_REV': '3',
                                 'ZENSIM_ROOT_FORM': 'sqrt',
                                 'RAYON_NUM_THREADS': threads},
            'exit_code': rc, 'output_sha256': sha(output) if output.exists() else None,
            'manifest_sha256': sha(manifest) if manifest.exists() else None,
            'stderr_sha256': sha(OUT / f'{stem}.stderr')}
    with (OUT / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(info, sort_keys=True) + '\n')
    if rc:
        raise RuntimeError(f'{stem}: extractor exit {rc}')
    metadata = json.loads(manifest.read_text())
    assert metadata['producer_binary_sha256'] == binary_sha, stem
    assert metadata['simd_tier_request'] == tier, stem
    assert metadata['rayon_num_threads'] == threads, stem
    return output


def compare(a_path, b_path, expected):
    with a_path.open() as af, b_path.open() as bf:
        ar, br = csv.reader(af), csv.reader(bf)
        ah, bh = next(ar), next(br)
        ai = [ah.index(f'f{i}') for i in range(986)]
        bi = [bh.index(f'f{i}') for i in range(986)]
        rows = diffs = 0
        for a, b in zip(ar, br, strict=True):
            assert a[0] == b[0], f'row {rows}: mismatched reference'
            rows += 1
            for ia, ib in zip(ai, bi):
                diffs += struct.pack('>d', float(a[ia])) != struct.pack('>d', float(b[ib]))
    assert rows == expected, f'got {rows} rows, expected {expected}'
    return rows, rows * 986, diffs


def main():
    binary = Path(sys.argv[1]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    binary_sha = sha(binary)
    summary = {'binary_sha256': binary_sha, 'comparisons': []}
    for corpus, (input_path, expected) in INPUTS.items():
        for mode, (tier, threads) in MODES.items():
            a = run(binary, corpus, input_path, mode, tier, threads, '986', binary_sha)
            b = run(binary, corpus, input_path, mode, tier, threads, 'rev4', binary_sha)
            rows, cells, diff = compare(a, b, expected)
            print(f'{corpus} {mode} rows={rows} cells={cells} diff={diff}', flush=True)
            assert diff == 0, f'{corpus} {mode} changed an old slot'
            summary['comparisons'].append({'corpus': corpus, 'mode': mode,
                                           'rows': rows, 'cells': cells, 'diff': diff})
    summary['cells'] = sum(x['cells'] for x in summary['comparisons'])
    summary['diffs'] = sum(x['diff'] for x in summary['comparisons'])
    assert len(summary['comparisons']) == 18 and summary['cells'] == 851904
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"TOTAL comparisons=18 cells={summary['cells']} diffs={summary['diffs']}")


if __name__ == '__main__':
    main()
