"""Launch a prepared jobset only with matching live coordinator authorization.

The existing fleet queue and fillers own execution. Preparing these files grants
no launch permission; E28 must run first. Missing authorization has no side effects.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def gate(root, jobset):
    authorization = root / f"LAUNCH_AUTHORIZATION-{jobset}.json"
    if not authorization.is_file():
        raise PermissionError("NOT AUTHORIZED: matching coordinator authorization file required; E28 runs first")
    a = json.loads(authorization.read_text())
    required = json.loads((root / f"AUTHORIZATION_REQUIRED-{jobset}.json").read_text())
    if (a.get('schema') != 'shippath10-coordinator-launch-v1' or not a.get('coordinator_message')
            or any(a.get(k) is not True for k in ('source_landed', 'pins_pushed', 'reviewed', 'E28_completed'))
            or a.get('identities') != required['identities']):
        raise PermissionError("coordinator authorization does not match reviewed artifacts/E28 completion")
    ids = required['identities']
    for filename, pin in ids['files'].items():
        if sha(root / filename) != pin:
            raise ValueError(f"prepared launch input changed: {filename}")
    for route in ('e30', 'production'):
        smoke = json.loads((root / f'{route}-EXECUTOR_SMOKE.json').read_text())
        if (smoke['status'] != 'PASS' or smoke['qualified_provenance'] is not True
                or smoke['program_sha'] != ids['program_sha'] or smoke['data_sha'] != ids['data_sha']):
            raise ValueError("matching installed-program strict smoke required")
    peak = int((root / 'CONTAINER_MEMORY_PEAK.txt').read_text())
    events = dict(line.split() for line in (root / 'CONTAINER_MEMORY_EVENTS.txt').read_text().splitlines())
    if peak >= ids['memory_cap_bytes'] or int(events['oom']) or int(events['oom_kill']):
        raise ValueError("prepared image failed its memory envelope smoke")
    caps = json.loads(Path('/var/tmp/fitv2/jobset_caps.json').read_text())[jobset]
    if hashlib.sha256(json.dumps(caps, sort_keys=True).encode()).hexdigest() != ids['jobset_cap_sha256']:
        raise ValueError("jobset cap changed")
    return ids


def launch(root, jobset):
    ids = gate(root, jobset)  # No output or subprocess before this gate.
    actual = subprocess.check_output(['docker', 'image', 'inspect', '-f', '{{.Id}}', ids['image']], text=True).strip()
    if actual != ids['image_id']:
        raise ValueError("prepared image changed")
    queue = Path('/var/tmp/fitv2/fleet_queue')
    lines = queue.read_text().splitlines()
    if any(line.split() and line.split()[0] == jobset for line in lines):
        raise ValueError("jobset already queued")
    subprocess.run(['docker', 'push', ids['image']], check=True)
    manifest = root / f'fit-manifest-{jobset}.json'
    control = root / f'control-{jobset}.json'
    control.write_text(json.dumps({'paused':False, 'drain':False, 'note':'reviewed D1/E30; E28 complete'})+'\n')
    # Existing R2 credentials and transfer owner, loaded only after authorization.
    subprocess.run(['bash', '-c', '''set -euo pipefail
. ~/.config/zen/s3env.sh >/dev/null 2>&1
s5cmd --endpoint-url "$EP" cp "$1/fit-manifest-$2.json" "s3://zentrain/jobs/$2/manifest.json"
s5cmd --endpoint-url "$EP" cp "$1/control-$2.json" "s3://zentrain/jobs/$2/control.json"
s5cmd --endpoint-url "$EP" cp "$1/d1-fit-data.tar.gz" "s3://zentrain/jobs/$2/inputs/$3"
''', '--', str(root), jobset, ids['data_sha']], check=True)
    # Append behind existing work; never reorder E28 or another jobset.
    queue.with_name(f'fleet_queue.{jobset}.before').write_bytes(queue.read_bytes())
    temporary = queue.with_name(f'fleet_queue.{jobset}.new')
    temporary.write_text(queue.read_text().rstrip() + f'\n{jobset} {manifest} {ids["image"]}\n')
    os.replace(temporary, queue)
    print(f"Queued {jobset}; existing fillers own execution. Run the prepared harvest/assessment commands after completion.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--jobset', choices=('fitv2e30-20261007','fitv2d1-20261007'), required=True)
    a = p.parse_args()
    launch(a.root, a.jobset)


if __name__ == '__main__':
    main()
