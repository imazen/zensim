#!/usr/bin/env python3
"""Prepare and seal exact-pixel GMSD peer scores for the Rev4 bank."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BANK = Path('/var/tmp/rev4-featbank/bank')
ROOT = Path('/var/tmp/gmsbank/peer_gmsd')
COLS = ('pair_key', 'ref_pixels_sha256', 'dist_pixels_sha256', 'width', 'height', 'ref_path', 'dist_path')


def file_sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(set_name):
    table = pq.read_table(BANK / set_name / 'keys.parquet', columns=list(COLS))
    return [dict(zip(COLS, row)) for row in zip(*(table[c].to_pylist() for c in COLS))]


def main():
    mode = sys.argv[1]
    ROOT.mkdir(parents=True, exist_ok=True)
    sets = sorted(p.name for p in BANK.iterdir() if p.is_dir() and (p / 'keys.parquet').is_file())
    if mode == 'prepare':
        counts = {}
        for set_name in sets:
            items = rows(set_name)
            with (ROOT / f'{set_name}.keys.tsv').open('w') as f:
                for r in items:
                    f.write('\t'.join(str(r[k]) for k in COLS) + '\n')
            counts[set_name] = len(items)
            print(set_name, len(items))
        (ROOT / '_INPUTS.json').write_text(json.dumps(dict(sets=counts,
            bank_manifest_sha256=file_sha(BANK / '_MANIFEST.json'),
            keys_sha256={s: file_sha(BANK / s / 'keys.parquet') for s in sets}), indent=2) + '\n')
        print('total', sum(counts.values()), 'sets', len(sets))
        return
    if mode != 'finish' or len(sys.argv) != 4:
        raise SystemExit('prepare | finish <build-commit> <binary-path>')
    build_commit, binary = sys.argv[2:]
    inputs = json.loads((ROOT / '_INPUTS.json').read_text())
    manifest = dict(schema='gmsbank-peer-gmsd-v1', build_commit=build_commit,
                    binary_sha256=file_sha(binary), bank_manifest_sha256=inputs['bank_manifest_sha256'],
                    key_check='all unique keys passed decoded RGB8 SHA256 and pair-key assertion; identical-pixel stimuli expanded by bank n_stimuli', sets={})
    for set_name in sets:
        expected = rows(set_name)
        scores = []
        with (ROOT / f'{set_name}.scores.tsv').open() as f:
            for r in csv.DictReader(f, delimiter='\t'):
                scores.append(r)
        if len(scores) != len(expected):
            raise ValueError(f'{set_name}: {len(scores)} != {len(expected)}')
        for i, (want, got) in enumerate(zip(expected, scores)):
            if want['pair_key'] != got['pair_key']:
                raise ValueError(f'{set_name}: key order mismatch at {i}')
        multiplicity = pq.read_table(BANK / set_name / 'keys.parquet', columns=['n_stimuli'])['n_stimuli'].to_pylist()
        expanded = [r for r, n in zip(scores, multiplicity) for _ in range(n)]
        table = pa.table(dict(pair_key=pa.array([r['pair_key'] for r in expanded], type=pa.string()),
                              gmsd=pa.array([float(r['gmsd']) for r in expanded], type=pa.float64()),
                              gmsm=pa.array([float(r['gmsm']) for r in expanded], type=pa.float64())))
        dst = ROOT / f'{set_name}.parquet'
        pq.write_table(table, dst, compression='zstd')
        manifest['sets'][set_name] = dict(rows=len(expanded), unique_pair_keys=len(set(r['pair_key'] for r in scores)),
                                         parquet_sha256=file_sha(dst), keys_sha256=inputs['keys_sha256'][set_name],
                                         scores_tsv_sha256=file_sha(ROOT / f'{set_name}.scores.tsv'),
                                         decoded_pair_key_checks=len(scores), key_check=True)
        print(set_name, len(expanded), manifest['sets'][set_name]['parquet_sha256'])
    manifest['total_rows'] = sum(v['rows'] for v in manifest['sets'].values())
    manifest['set_count'] = len(manifest['sets'])
    (ROOT / '_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('total', manifest['total_rows'], 'sets', manifest['set_count'])


if __name__ == '__main__':
    main()
