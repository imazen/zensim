#!/usr/bin/env bash
# Final local identity audit; caller holds heavy.lock and uses run-heavy.
set -euo pipefail
python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

bundle, mirror, metrics = map(Path, sys.argv[1:])
source = Path.cwd()
sys.path.insert(0, str(source / 'scripts/tests'))
from v40_binary_metadata import validate

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def commit(root, revision):
    value = subprocess.check_output(['jj', 'log', '-r', revision, '--no-graph', '-T', 'commit_id'], cwd=root, text=True).strip()
    assert len(value) == 40
    return value

source_tip = commit(source, '@-')
metrics_tip = commit(metrics, '@-')
metadata = read(bundle / 'build-meta.json')
bindings = read(bundle / 'SOURCE_BINDINGS.json')
validate(bundle, metadata, bindings, metadata['files'])
producer = bindings['binary_producer_commit']
assert commit(source, f'{producer} & ancestors({source_tip})') == producer
for record in bindings['source_checks']:
    assert sha(source / record['path']) == record['producer_sha256']
for name, record in bindings['binaries'].items():
    assert sha(mirror / 'bin' / name) == record['sha256']
assessment = read(bundle / 'ASSESSMENT_SOURCE.json')
assert commit(source, f"{assessment['source_commit']} & ancestors({source_tip})") == assessment['source_commit']
for path, expected in assessment['files'].items():
    current = source / path
    if not current.exists():
        assert path == 'benchmarks/v40_fit_contract_2026-10-07.json'
        current = bundle / 'v40-fit-contract.json'
    assert sha(current) == expected, path
    assert sha(bundle / 'assessment-runtime' / path) == expected, path
assert not list(bundle.glob('LAUNCH_AUTHORIZATION-*.json'))
assert not list(bundle.glob('EXPOSURE_FREEZE-*.json'))
assert not (source / 'target').exists()
assert read(bundle / 'MIRROR_CHECK.json')['status'] == 'PASS'
assert read(bundle / 'BUNDLE_CHECK.json')['status'] == 'PASS'
assert read(bundle / 'AUTHORIZATION_GATE.json')['status'] == 'PASS'
for name in ('parity-kadid', 'parity-tid2013'):
    assert read(bundle / name / 'PARITY.json')['status'] == 'PASS'
late = {}
for name in ('FINALIZE_COMMAND.log', 'CARGO_TARGET_CLEANUP.json'):
    assert sha(bundle / name) == sha(mirror / name)
    late[name] = sha(bundle / name)
for jobset in read(bundle / 'PACKAGE_PINNED.json')['manifests']:
    identities = read(bundle / f'AUTHORIZATION_REQUIRED-{jobset}.json')['identities']
    assert commit(metrics, f"{identities['zenmetrics_commit']} & ancestors({metrics_tip})") == identities['zenmetrics_commit']
    for path, expected in identities['files'].items():
        assert sha(bundle / path) == expected, path
record = dict(status='PASS', source_tip=source_tip, zenmetrics_tip=metrics_tip,
              assessment_source=assessment['source_commit'], trainer_producer=producer,
              bound_source_files=len(bindings['source_checks']),
              assessment_members=len(assessment['files']),
              production_authorizations=0, production_exposure_freezes=0,
              cargo_target_exists=False, late_receipts=late)
for root in (bundle, mirror):
    with (root / 'FINAL_LOCAL_CHECK.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
