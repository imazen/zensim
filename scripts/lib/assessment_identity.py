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
    if any('_sealed' in v.lower() or v.lower().startswith('labels__') for v in (*p.parts, *p.resolve().parts)):
        raise PermissionError('protected label path refused')
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration_paths(table):
    p = Path(table)
    return [p.parent/'_MANIFEST.json', Path(str(p)+'.manifest.json'), Path(str(p)+'._MANIFEST.json')]


def checked_inventory(paths):
    # Check the COMPLETE candidate list, including absent files and symlinks,
    # before hashing or opening any discovered metadata payload.
    paths = sorted({safe_path(p) for p in paths}, key=str)
    return [{'path':str(p), 'resolved_path':str(p.resolve()),
             'sha256':sha(p) if p.is_file() else None} for p in paths]


def validate(path, output, composition=(), owner_inputs=None, owner_identity=None):
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
    metadata_paths = [root/'_MANIFEST.json']
    for table in [t['path'] for r in records for t in r['tables']] + [v['path'] for v in inputs.values()] + [root/'ext_cid22val.parquet']:
        metadata_paths.extend(declaration_paths(table))
    metadata_paths.extend(Path(str(p)+'.spec.json') for p in models)
    discovered = []
    if owner_inputs is not None:
        if owner_inputs.get('schema') != 'bake-verdict-input-paths-v1' or owner_inputs.get('complete') is not True or owner_inputs.get('metadata_read') is not False:
            raise ValueError('complete metadata-free Rust input discovery required')
        discovered = [safe_path(p) for p in owner_inputs['files']]
    # Guard all automatically discovered locations before reading declarations.
    candidates = [safe_path(p) for p in metadata_paths + discovered]
    roots = [root, *[p.parent for p in models], *[safe_path(r['bank_root']) for r in records]]
    for record in records:
        for table in record['tables']:
            # Declarations are metadata, never a corpus label payload.
            d = json.loads(safe_path(table['declaration']['path']).read_text())
            roots.extend(safe_path(v) for v in d.get('assessment', {}).get('immutable_roots', []))
    refuse_immutable_output(Path(output), roots + [p.parent for p in specs] + [p.parent for p in candidates if p.is_file()])
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
    inventory = checked_inventory(specs + models + candidates)
    if owner_identity is not None:
        if owner_inputs is None:
            raise ValueError('owner identity requires complete input discovery')
        checked = {v['path']:v['sha256'] for v in inventory}
        for v in owner_identity['files']:
            if v['path'] not in checked or v['sha256'] != checked[v['path']]:
                raise ValueError('verdict owner read an unchecked or changed input')
    boundary = {'complete_discovery_checked':owner_inputs is not None,
                'protected_paths_refused':True, 'table_schemas_label_free':True,
                'checked_input_count':len(inventory)}
    # No no-label assertion is issued for an incomplete discovery. Failure
    # raises before a receipt; successful complete checks establish the claim.
    labels_read = not (boundary['protected_paths_refused'] and boundary['table_schemas_label_free']) if boundary['complete_discovery_checked'] else None
    return dict(capsule, capsule_sha256=sha(path), checked_inputs=inventory,
                label_boundary=boundary, labels_read=labels_read)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('capsule')
    parser.add_argument('output')
    parser.add_argument('composition', nargs='*')
    parser.add_argument('--owner-inputs', action='store_true')
    parser.add_argument('--owner-identity', type=Path)
    args = parser.parse_args()
    composition = [p for value in args.composition for p in value.split(',') if p]
    owner_inputs = json.load(sys.stdin) if args.owner_inputs else None
    owner_identity = json.loads(safe_path(args.owner_identity).read_text()) if args.owner_identity else None
    print(json.dumps(validate(args.capsule, args.output, composition, owner_inputs, owner_identity)))
