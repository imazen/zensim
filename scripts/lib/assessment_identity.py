#!/usr/bin/env python3
"""Validate a features-only identity capsule; statistics stay with Rust owners."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from v2c_wide import safe_path as bank_safe_path
import pyarrow.parquet as pq
from rev5_bank import ASSESSMENT_KEY_COLUMNS
from v2_common import refuse_immutable_output


def safe_path(path):
    p = bank_safe_path(path)
    if any(v.startswith('labels__') for v in (*p.parts, *p.resolve().parts)):
        raise PermissionError('protected label path refused')
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(path, output, composition=()):
    models = [safe_path(p) for p in composition]
    if any(not p.is_file() for p in models):
        raise ValueError("composition artifact missing")
    path = safe_path(path)
    capsule = json.loads(path.read_text())
    if capsule.get('schema') != 'rev5-assessment-eval-identity-v1' or capsule.get('features_only') is not True or capsule.get('labels_read') is not False:
        raise ValueError('explicit features-only instrument capsule required')
    manifests = capsule['assessments']
    specs = [path, *[safe_path(v['path']) for v in manifests], safe_path(capsule['admission']['path'])]
    records = []
    for spec in manifests:
        p = safe_path(spec['path'])
        if sha(p) != spec['sha256']:
            raise ValueError('changed assessment manifest')
        record = json.loads(p.read_text())
        if record.get('schema') != 'rev5-assessment-tables-v1' or record.get('features_only') is not True or record.get('labels_read') is not False or len(record.get('feature_ids', [])) != 420:
            raise ValueError('unknown assessment contract')
        records.append(record)
        for table in record['tables']:
            specs.extend(safe_path(v['path']) for v in [table, table['declaration'], table['keys']])
    root = safe_path(capsule['features_root'])
    inputs = capsule['inputs']
    if set(inputs) != {'ext_cid22val.parquet', 'dial-grid', 'negtail-probe', 'identity-probe'}:
        raise ValueError('explicit label-free cid22 and gate instrument paths required')
    specs.extend(safe_path(v['path']) for v in inputs.values())
    roots = [root, *[p.parent for p in models], *[safe_path(r['bank_root']) for r in records]]
    for record in records:
        for table in record['tables']:
            # Declarations are metadata, never a corpus label payload.
            d = json.loads(safe_path(table['declaration']['path']).read_text())
            roots.extend(safe_path(v) for v in d.get('assessment', {}).get('immutable_roots', []))
    refuse_immutable_output(Path(output), roots + [p.parent for p in specs])
    allowed = {}
    for record in records:
        for table in record['tables']:
            for payload in [table, table['keys']]:
                columns = pq.ParquetFile(safe_path(payload['path'])).schema_arrow.names
                metadata = ASSESSMENT_KEY_COLUMNS | {'ref_basename', 'width', 'height', 'reference_file_sha256', 'distorted_file_sha256', 'reference_pixels_sha256', 'distorted_pixels_sha256'}
                if any(n not in metadata and not (n.startswith('f') and n[1:].isdigit()) for n in columns):
                    raise ValueError('label-bearing assessment payload forbidden')
            for spec in [table, table['declaration'], table['keys']]:
                p = safe_path(spec['path'])
                if sha(p) != spec['sha256']:
                    raise ValueError('changed assessment table/declaration/keys')
            allowed[safe_path(table['path']).resolve()] = (table['sha256'], table['declaration']['sha256'], table['keys']['sha256'])
    for name, spec in inputs.items():
        p = safe_path(spec['path'])
        if p.resolve() not in allowed or allowed[p.resolve()][0] != spec['sha256'] or sha(p) != spec['sha256']:
            raise ValueError('input must be an admitted assessment table')
        if name == 'ext_cid22val.parquet' and (root / name).resolve() != p.resolve():
            raise ValueError('explicit corpus slot mismatch')
    admission = capsule['admission']
    if sha(safe_path(admission['path'])) != admission['sha256']:
        raise ValueError('changed Rust admission proof')
    proof = json.loads(safe_path(admission['path']).read_text())
    if proof.get('qualified_provenance') is not True:
        raise ValueError('Rust admission must pass')
    admitted = {safe_path(v['path']).resolve(): (v['sha256'], v['declaration']['sha256'], v['keys']['sha256']) for v in proof['tables']}
    if admitted != allowed:
        raise ValueError('admission proof must bind every assessment table')
    return dict(capsule, capsule_sha256=sha(path))


if __name__ == '__main__':
    composition = [p for value in sys.argv[3:] for p in value.split(',') if p]
    print(json.dumps(validate(sys.argv[1], sys.argv[2], composition)))
