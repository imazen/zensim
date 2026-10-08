set working-directory := '..'
heavy := 'flock /home/lilith/tmp/zensim-paper/rev4/heavy.lock /home/lilith/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 --'
evidence := '/mnt/v/output/zensim/releasegate6-2026-10-08'
scratch := '/home/lilith/tmp/releasegate6'
panel := '/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel'

suite:
    {{heavy}} env TMPDIR={{scratch}} OPENBLAS_NUM_THREADS=1 ZEN_PANEL_BIN={{panel}} just releasegate-tests > {{evidence}}/tests.log 2>&1

lint:
    ruff check --select F scripts/rev4_featpot/_terminal_acceptance.py scripts/rev4_featpot/_terminal_bound_io.py scripts/rev4_featpot/_terminal_owner.py scripts/tests/test_kadid_terminal_read.py scripts/tests/test_kadid_terminal_hardening.py scripts/tests/test_kadid_terminal_ledger.py scripts/tests/kadid_terminal_fixture_devices.py scripts/tests/releasegate6_before.py > {{evidence}}/ruff.log 2>&1

before:
    {{heavy}} env TMPDIR={{scratch}} OPENBLAS_NUM_THREADS=1 ZEN_PANEL_BIN={{panel}} python3 scripts/tests/releasegate6_before.py {{scratch}}/base > {{evidence}}/before.log 2>&1

[positional-arguments]
archive commit:
    #!/usr/bin/env bash
    set -euo pipefail
    {{heavy}} env TMPDIR={{scratch}} OPENBLAS_NUM_THREADS=1 python3 - "$1" > {{scratch}}/archive.log 2>&1 <<'PY'
    import hashlib, json, random, shutil, subprocess, sys
    from datetime import datetime, timezone
    from pathlib import Path
    root = Path('{{evidence}}')
    mirror = Path('/mnt/tower/output/zensim-releasegate6-2026-10-08')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = subprocess.check_output(['jj', 'diff', '-r', f'42972b5a3281342fac6a747f31beacf6a7a15781..{sys.argv[1]}', '--name-only'], text=True).splitlines()
    source = root / 'source'
    for name in files:
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest)
    for name in ('kadid_terminal_read.py', 'v2c_labels.py'):
        dest = source / 'scripts/rev4_featpot' / name
        shutil.copyfile(Path('scripts/rev4_featpot') / name, dest)
    dest = source / 'scripts/lib/zen_stats.py'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile('scripts/lib/zen_stats.py', dest)
    shutil.copytree('{{scratch}}/base', root / 'base-control-source', ignore=shutil.ignore_patterns('__pycache__'))
    program = Path('{{panel}}')
    (root / 'bin').mkdir(exist_ok=True)
    shutil.copy2(program, root / 'bin/panel')
    inventory = subprocess.check_output([sys.executable, '-c', 'import sys,json; sys.path.insert(0,"scripts/rev4_featpot"); import kadid_terminal_read as o; print(json.dumps(o.CODE_SHA256,sort_keys=True))'], text=True)
    pins = json.loads(inventory)
    assert len(pins) == 6
    for name, digest in pins.items():
        folder = 'scripts/lib' if name == 'zen_stats.py' else 'scripts/rev4_featpot'
        assert sha(source / folder / name) == digest
    receipt = {'build_commit': sys.argv[1], 'runtime_commit': '9aef0c67457f8c47d82319643ad16bcd90ad00b7', 'base_commit': '42972b5a3281342fac6a747f31beacf6a7a15781', 'code_sha256': pins, 'panel_sha256': sha(program), 'synthetic_only': True, 'real_protected_payload_reads': [], 'mount_observation_utc': datetime.now(timezone.utc).isoformat()}
    receipt['mounts'] = subprocess.check_output(['findmnt', '-T', '/mnt/tower', '-o', 'TARGET,SOURCE,FSTYPE'], text=True)
    assert 'tower:' in receipt['mounts'] and 'nfs' in receipt['mounts']
    (root / 'VERIFY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    members = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('_MANIFEST.json', 'MIRROR_VERIFICATION.json')]
    (root / '_MANIFEST.json').write_text(json.dumps({'build_commit': sys.argv[1], 'files': members}, indent=2) + '\n')
    mirror.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--no-owner', '--no-group', str(root)+'/', str(mirror)+'/'], check=True)
    for item in members:
        assert sha(mirror / item['path']) == item['sha256']
    assert sha(root / '_MANIFEST.json') == sha(mirror / '_MANIFEST.json')
    result = {'files_verified': len(members), 'bytes_verified': sum(x['bytes'] for x in members), 'manifest_sha256': sha(root / '_MANIFEST.json'), 'three_random_files': random.Random(20261008).sample(members, 3), 'mirror': str(mirror), 'deletions': False}
    (root / 'MIRROR_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copyfile(root / 'MIRROR_VERIFICATION.json', mirror / 'MIRROR_VERIFICATION.json')
    assert sha(root / 'MIRROR_VERIFICATION.json') == sha(mirror / 'MIRROR_VERIFICATION.json')
    print(json.dumps(result, indent=2))
    PY
