"""Actual trainer admission controls; synthetic keys/payloads only, syscall oracle."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
from e21_cheap_recipe import columns
from e29_consensus import TEACHER, SOURCE_TABLE, SOURCE_KEYS, SOURCE_MANIFEST
from v2_teacher import row_keys_sha


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary, table, keep, directory, label, target_column="human_score", target_scale="1"):
    trace = directory / f'{label}.syscalls.log'
    argv = [str(binary), '--group', f'hdr:{table}:1:0:rank', '--hdr-consensus-research',
            '--nonneg-distance', '--target-column', target_column, '--target-scale', target_scale, '--keep-features', str(keep), '--max-features', '1825',
            '--out', str(directory / f'{label}-never.bin'), '--no-auto-eval',
            '--epochs', '1', '--pairs-per-epoch', '1']
    p = subprocess.run(['strace', '-qq', '-f', '-e', 'trace=open,openat,openat2',
                        '-o', str(trace), *argv], capture_output=True, text=True)
    (directory / f'{label}.stdout.log').write_text(p.stdout)
    (directory / f'{label}.stderr.log').write_text(p.stderr)
    text = trace.read_text()
    count = lambda path: sum(f'"{path}"' in line for line in text.splitlines())
    return dict(case=label, rc=p.returncode, payload_opens=count(table),
                key_opens=count(table.with_suffix('.keys.parquet')), argv=argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--prior-binary', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.prior_binary) == '49b0b844454a7401636c9eda07118b77e4f5a9108399ff77d261879933b73d51'
    args.dest.mkdir(parents=True, exist_ok=False)
    table = args.dest / 'synthetic-hdr.parquet'
    table.write_bytes(b'ONLY SYNTHETIC INVALID PARQUET')
    key_path = table.with_suffix('.keys.parquet')
    sidecar = Path(str(table) + '.manifest.json')
    keep = args.dest / 'keep.txt'
    ids = columns('by_v2fy')
    keep.write_text('\n'.join(map(str, ids)) + '\n')
    def keys(role='train'):
        return pa.table({'row_id': [f'row{i}' for i in range(7390)], 'role': [role] * 7390,
                         'agree': [True] * 7390, 'ref_basename': [f'ref{i // 10}' for i in range(7390)]})
    def declaration(key_table):
        pq.write_table(key_table, key_path, compression='zstd')
        return dict(study='E29', role='train', rows=7390, population='agree-only', formula_revision=5,
                    teacher_sha256=TEACHER, requested_ids=ids, arm='hb4', build_commit='a' * 40,
                    source_table_sha256=SOURCE_TABLE, source_keys_sha256=SOURCE_KEYS,
                    source_manifest_sha256=SOURCE_MANIFEST, source_bank_feature_set_id=None,
                    target_transform='pooled-midrank-Borda-[0,1]', keys_sha256=sha(key_path),
                    row_keys_sha256=row_keys_sha(key_table), table_sha256=sha(table))
    report = []
    # The reviewer's TRAIN declaration +7390 VAL-key reproduction fails on old
    # native entry (payload opens) and refuses on the fixed binary (zero opens).
    d = declaration(keys('val')); sidecar.write_text(json.dumps(d))
    prior = run(args.prior_binary, table, keep, args.dest, 'reviewed-VAL-keys')
    assert prior['rc'] != 0 and prior['payload_opens'] > 0 and prior['key_opens'] == 0, prior
    report.append(prior)
    fixed = run(args.binary, table, keep, args.dest, 'fixed-VAL-keys')
    assert fixed['rc'] == 2 and fixed['payload_opens'] == 0 and fixed['key_opens'] > 0, fixed
    report.append(fixed)
    valid_keys = keys(); baseline = declaration(valid_keys)
    sidecar.write_text(json.dumps(baseline))
    for label, column, scale in [('wrong-target-column', 'different_target', '1'),
                                  ('wrong-target-scale', 'human_score', '-1')]:
        case = run(args.binary, table, keep, args.dest, label, column, scale)
        assert case['rc'] == 2 and case['payload_opens'] == 0 and case['key_opens'] == 0, case
        report.append(case)
    variants = [
        ('missing-source-table', 'source_table_sha256', None),
        ('wrong-source-keys', 'source_keys_sha256', '0' * 64),
        ('missing-source-manifest', 'source_manifest_sha256', None),
        ('wrong-teacher', 'teacher_sha256', '0' * 64),
        ('wrong-transform', 'target_transform', 'per-reference-normalized'),
        ('wrong-declared-ids', 'requested_ids', [0, *ids[1:]]),
        ('wrong-build', 'build_commit', ''),
        ('wrong-revision', 'formula_revision', 4),
        ('wrong-native-family', 'source_bank_feature_set_id', 'invented'),
        ('wrong-role', 'role', 'val'),
        ('missing-key-pin', 'keys_sha256', None),
        ('wrong-key-pin', 'keys_sha256', '0' * 64),
        ('wrong-row-key-pin', 'row_keys_sha256', '0' * 64),
    ]
    for label, field, value in variants:
        d = copy.deepcopy(baseline)
        if value is None:
            d.pop(field)
        else:
            d[field] = value
        sidecar.write_text(json.dumps(d))
        case = run(args.binary, table, keep, args.dest, label)
        assert case['rc'] == 2 and case['payload_opens'] == 0, case
        report.append(case)
    # Wrong IDs cannot become admitted by changing the declaration and argv together.
    d = copy.deepcopy(baseline); d['requested_ids'] = [0, *ids[1:]]
    keep.write_text('\n'.join(map(str, d['requested_ids'])) + '\n')
    sidecar.write_text(json.dumps(d))
    case = run(args.binary, table, keep, args.dest, 'wrong-ids-matched-argv')
    assert case['rc'] == 2 and case['payload_opens'] == 0 and case['key_opens'] == 0, case
    report.append(case)
    keep.write_text('\n'.join(map(str, ids)) + '\n')
    for label, key_table in [
        ('duplicate-row-id', valid_keys.set_column(0, 'row_id', pa.array(['duplicate'] * 7390))),
        ('nonagree-keys', valid_keys.set_column(2, 'agree', pa.array([False] * 7390))),
        ('missing-rows', valid_keys.slice(0, 7389)),
        ('label-bearing-key-schema', valid_keys.append_column('human_score', pa.array([1.] * 7390))),
    ]:
        d = declaration(key_table); sidecar.write_text(json.dumps(d))
        case = run(args.binary, table, keep, args.dest, label)
        assert case['rc'] == 2 and case['payload_opens'] == 0 and case['key_opens'] > 0, case
        report.append(case)
    # Valid label-free declaration/keys intentionally reach the malformed fixture
    # payload. This positive admission control is not training or qualification.
    d = declaration(valid_keys); sidecar.write_text(json.dumps(d))
    case = run(args.binary, table, keep, args.dest, 'valid-TRAIN-admission')
    assert case['rc'] != 0 and case['payload_opens'] > 0 and case['key_opens'] > 0, case
    assert 'parquet' in (args.dest / 'valid-TRAIN-admission.stderr.log').read_text().lower()
    report.append(case)
    assert not list(args.dest.glob('*-never.bin'))
    result = dict(schema='e29r2-native-hdr-admission-probes-v1', status='PASS',
                  binary_sha256=sha(args.binary), prior_binary_sha256=sha(args.prior_binary),
                  synthetic_only=True, cases=report)
    (args.dest / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
