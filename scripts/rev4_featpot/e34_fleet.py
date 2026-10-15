"""Read E34's current worker identities without changing workers or host keys.

Connection evidence stays in the operator's local report directory. Only the
six frozen host aliases and the eighteen released E34 jobsets are admitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path('/mnt/v/output/zensim/e34-launch-2026-10-15')
HOSTS = ['tower', 'r3500', 'r3800x', 'i270', 'i134', 'r5600g']
OPS = Path.home() / 'tmp/zensim-paper/rev4/e34-worker-ops'


def gone_hosts():
    path = OPS / 'GONE_HOSTS.json'
    return set(json.loads(path.read_text())['hosts']) if path.exists() else set()


def remote(host, argv):
    assert host in HOSTS
    assert host not in gone_hosts(), f'owner forbids connecting to reclaimed host {host}'
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           host, shlex.join(argv)], capture_output=True, text=True, timeout=30)


def inventory_host(host):
    if host in gone_hosts():
        return dict(host=host, state='GONE', error=None,
                    detail='owner confirmed reclaimed; no connection attempted')
    try:
        return _inventory_host(host)
    except subprocess.TimeoutExpired:
        return dict(host=host, state='UNVERIFIED', error='timeout', detail='SSH inventory timed out')


def _inventory_host(host):
    os_result = remote(host, ['uname', '-s'])
    if os_result.returncode:
        error = os_result.stderr
        kind = 'host-key-verification' if 'Host key verification failed' in error else 'connection-unavailable'
        return dict(host=host, state='UNVERIFIED', error=kind, detail=error)
    if os_result.stdout.strip() != 'Linux':
        return dict(host=host, state='UNVERIFIED', error='unexpected-os', detail=os_result.stdout.strip())
    boot = remote(host, ['cat', '/proc/sys/kernel/random/boot_id'])
    rows = remote(host, ['docker', 'ps', '--format', '{{json .}}'])
    if boot.returncode or rows.returncode:
        return dict(host=host, state='UNVERIFIED', error='inventory-unavailable',
                    detail=boot.stderr+rows.stderr)
    candidates = [r['ID'] for line in rows.stdout.splitlines() if (r := json.loads(line))['Names'].startswith('zen-score-')]
    inspect = remote(host, ['docker', 'inspect', *candidates]) if candidates else None
    if inspect is not None and inspect.returncode:
        return dict(host=host, state='UNVERIFIED', error='inspect-unavailable', detail=inspect.stderr)
    jobs = json.loads((ROOT / 'launch/HARVEST_STATUS.json').read_text())['jobsets']
    workers = []
    for r in json.loads(inspect.stdout) if inspect is not None else []:
        image = r['Config']['Image']
        if 'fit-e34-3c45506693f5' not in image:
            continue
        present = [js for js in jobs if js in json.dumps(r)]
        workers.append(dict(id=r['Id'], name=r['Name'], image=image,
                            jobsets=present, state=r['State']['Status']))
    return dict(host=host, state='VERIFIED', os='Linux', boot_id=boot.stdout.strip(), workers=workers)


def inventory():
    OPS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(inventory_host, HOSTS))
    now = datetime.now(timezone.utc).isoformat()
    path = OPS / ('inventory-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
    with path.open('x') as f:
        json.dump(dict(time=now, hosts=records), f, indent=2)
        f.write('\n')
    summary = {r['host']: dict(state=r['state'], workers=len(r['workers']) if r['state']=='VERIFIED' else None, error=r.get('error'))
               for r in records}
    print(json.dumps(dict(time=now, hosts=summary), indent=2))
    with (Path.home() / 'tmp/zensim-paper/rev4/E34_LAUNCH_STATUS.md').open('a') as f:
        f.write('\nEvaluation worker inventory '+now+' (read-only, no host-key changes or reboots):\n\n')
        for host, r in summary.items():
            f.write(f'- {host}: {r["state"]}; E34 workers {r["workers"]}; error {r["error"]}.\n')
    return all(r['state'] in ['VERIFIED', 'GONE'] for r in records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['inventory'])
    p.parse_args()
    raise SystemExit(0 if inventory() else 1)
