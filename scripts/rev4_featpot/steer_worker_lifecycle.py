"""Bounded STEER worker enrollment through the packet's canonical LAN launcher.

This owner never creates jobs, changes manifests, claims work or manufactures
ledger rows. The pinned worker performs all scientific work and reconciliation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import inspect as source_inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import e33_launch as owner


GIB = 1024**3


def read_json(path):
    return json.loads(path.read_text())


def transition_proof(bundle):
    proof = read_json(bundle.parent/'launch/OLD_CHUNKS_DRAINED.json')
    keys = ('all_old_chunks_complete', 'ledgers_published', 'native_verified',
            'NAS_verified', 'old_workers_stopped')
    if proof.get('packet_sha256') != owner.PACKET_SHA or not all(proof.get(k) is True for k in keys):
        raise ValueError('old scheduling transition is incomplete')
    return proof


def ssh(host, command):
    return subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, command], text=True)


def repaired_admission(bundle):
    proof = read_json(bundle.parent/'launch/REPAIRED_CELL_ADMITTED.json')
    required = ('verified', 'inside_pinned_image', 'worker_drained', 'NAS_verified')
    if (proof.get('packet_sha256') != owner.PACKET_SHA
            or not all(proof.get(k) is True for k in required)):
        raise ValueError('repaired first-cell admission is incomplete')
    dest = Path(proof['dest'])
    for name, key in [('result.json', 'result_sha256'), ('fleet_receipt.json', 'receipt_sha256')]:
        if owner.sha(dest/name) != proof[key]:
            raise ValueError('repaired first-cell bytes changed')
    return proof


def inspect(host, name):
    sudo = '' if host == 'tower' else 'sudo -n '
    return json.loads(ssh(host, sudo+'docker inspect '+shlex.quote(name)))[0]


def capacity(host):
    sudo = '' if host == 'tower' else 'sudo -n '
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in ssh(host, 'cat /proc/meminfo').splitlines()
           if line.startswith(('MemTotal:', 'MemAvailable:'))}
    raw = ssh(host, sudo+'docker ps -q | xargs -r '+sudo+'docker inspect').strip()
    workers = []
    for d in json.loads(raw) if raw else []:
        env = dict(v.split('=', 1) for v in d['Config']['Env'] if '=' in v)
        workers.append(dict(name=d['Name'], memory=d['HostConfig']['Memory'],
                            cpuset=d['HostConfig']['CpusetCpus'], run=env.get('ZEN_RUN', '')))
    disk = '/mnt/cache' if host == 'tower' else '/home'
    free = int(ssh(host, 'df -B1 --output=avail '+disk+' | sed -n "2p"').strip())
    docker_free = int(ssh(host, 'df -B1 --output=avail /var/lib/docker | sed -n "2p"').strip())
    return dict(memory=mem, workers=workers, disk=disk, free=free, docker_free=docker_free)


def admit(host, cap, own_name=None):
    workers = cap['workers']
    own = [w for w in workers if w['name'] == '/'+str(own_name)]
    if len(own) > 1 or any(w['run'].startswith('jobs/fitsteer-') and w not in own for w in workers):
        raise ValueError('STEER host slot occupied')
    new = 0 if own else 6*GIB
    reserve = (12 if host == 'tower' else 4)*GIB
    if host != 'tower' and any(w['memory'] <= 0 for w in workers):
        raise ValueError('uncapped active reservation')
    if sum(w['memory'] for w in workers)+new+reserve > cap['memory']['MemTotal']:
        raise ValueError('configured reservations exceed admission')
    if cap['memory']['MemAvailable'] < reserve+new:
        raise ValueError('available memory below admission')
    if min(cap['free'], cap['docker_free']) < 20*GIB:
        raise ValueError('disk below 20 GiB floor')
    if host == 'tower' and sum(w['memory'] for w in workers
                               if w['run'].startswith('jobs/fit'))+new > 12*GIB:
        raise ValueError('shared fit pool reservation exceeded')


def envelope(binding, host, container, d, deadline, kernel_swap=None):
    env = dict(v.split('=', 1) for v in d['Config']['Env'] if '=' in v)
    h = d['HostConfig']
    rename = {'ZEN_FORMULA_REV': 'ZENSIM_FORMULA_REV'}
    expected = {**binding['shared_environment'], 'ZEN_CHUNK_WALL_SEC': '1',
                'ZEN_MAX_MIN': '1'}
    checks = {k: env.get(rename.get(k, k)) == v for k, v in expected.items()
              if k not in ('ZEN_MEMORY', 'ZEN_CPU_SHARES')}
    checks.update(image=d['Config']['Image'] == binding['image'],
                  memory=h['Memory'] == 6*GIB,
                  swap=(kernel_swap == [6*GIB, 0, 0] if host == 'tower'
                        else h['MemorySwap'] == 6*GIB),
                  cpuset=h['CpusetCpus'] == binding['active'][host]['ZEN_CPUSET'],
                  shares=h['CpuShares'] == 256,
                  restart=h['RestartPolicy']['Name'] == 'no',
                  restarts=d['RestartCount'] == 0, running=d['State']['Running'],
                  deadline=env.get('ZEN_NO_NEW_AFTER_EPOCH') == str(deadline),
                  scratch=any(m['Source'] == binding['active'][host]['ZEN_TMPDIR_HOST_DIR']
                              and m['Destination'] == '/scratch' for m in d['Mounts']))
    return checks


def control(bundle, js, value=None):
    root = bundle.parent/'launch/lifecycle-controls'
    root.mkdir(exist_ok=True)
    path = root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+js+'.json')
    uri = 's3://zentrain/jobs/'+js+'/control.json'
    private = Path.home()/'tmp/zensim-paper/rev4/private_telemetry/steer-launch'
    private.mkdir(parents=True, exist_ok=True)
    if value is None:
        with (private/'lifecycle-control-transport.log').open('a') as log:
            subprocess.run(['aws', '--endpoint-url', os.environ['EP'], 's3api', 'get-object',
                            '--bucket', 'zentrain', '--key', 'jobs/'+js+'/control.json', str(path)],
                           check=True, stdout=subprocess.DEVNULL, stderr=log)
        return read_json(path)
    path.write_text(json.dumps(value)+'\n')
    # Capture transport errors on disk; never print endpoint-bearing output.
    with (private/'lifecycle-control-transport.log').open('a') as log:
        subprocess.run(['s5cmd', '--endpoint-url', os.environ['EP'], 'cp', str(path), uri],
                       stdout=log, stderr=log, check=True)


def enroll(bundle, doc, host, role, js):
    binding = read_json(bundle/'WORKER_BINDING.json')
    if host not in binding['active'] or host not in ('r3500', 'i270', 'tower'):
        raise ValueError('only frozen active hosts are enrolled by this owner')
    if not re.fullmatch('steer-'+host+r'-[0-9]+', role):
        raise ValueError('invalid unique STEER role')
    if binding['image'] != doc['image_ref'] or js not in owner.steer_plan(bundle, doc)['order']:
        raise ValueError('image or jobset outside frozen plan')
    owner.first_steer_admission(bundle)
    transition_proof(bundle)
    repaired_admission(bundle)
    launch = bundle.parent/'launch'
    if (launch/'PAUSE_WORKERS').exists():
        return dict(state='PAUSED', host=host)
    active = control(bundle, js)
    if active.get('paused') or active.get('drain'):
        return dict(state='PAUSED', host=host)
    ctr = 'zen-score-'+role
    before = capacity(host)
    admit(host, before)
    # Refuse accidental reuse, including a stopped container with preserved evidence.
    sudo = '' if host == 'tower' else 'sudo -n '
    if ctr in ssh(host, sudo+'docker ps -a --format "{{.Names}}"').splitlines():
        raise ValueError('role already exists; preserve container and use a new role')
    paused = dict(paused=True, drain=False, note='STEER envelope '+role)
    control(bundle, js, paused)
    deadline = int(ssh(host, 'date -u +%s').strip())+30
    # In the frozen entrypoint this budget is checked BETWEEN passes. A fit may
    # finish under its unchanged 7663s cap; the wrapper then exits instead of
    # polling with an already-expired native claim deadline for 150 minutes.
    env = {**os.environ, **binding['shared_environment'], **binding['active'][host],
           'ZEN_STORE': 'tower', 'ZEN_CHUNK_WALL_SEC': '1',
           'ZEN_MAX_MIN': '1',
           'ZEN_NO_NEW_AFTER_EPOCH': str(deadline)}
    helper = bundle.parent/'launch-source/zenmetrics/scripts/jobsys/lan_score_launch.sh'
    receipt = launch/('ENVELOPE-'+role+'.json')
    if receipt.exists():
        raise ValueError('enrollment receipt already exists')
    with (bundle.parent/'logs'/('launch-'+role+'.log')).open('x') as log:
        subprocess.run(['bash', str(helper), host, js, role, 'cpu', binding['image']],
                       env=env, stdout=log, stderr=log, check=True)
    setup = ('set -eu; '+sudo+'docker update --restart=no '+ctr+' >/dev/null; '
             'pid=$('+sudo+'docker inspect --format "{{.State.Pid}}" '+ctr+'); '
             +sudo+'renice -n 19 -p "$pid" >/dev/null; '+sudo+'ionice -c3 -p "$pid"; ')
    if host == 'tower':
        setup += ('cg=$(awk -F: \'$1==0 {print $3}\' /proc/$pid/cgroup); '
                  'printf 0 > "/sys/fs/cgroup$cg/memory.swap.max"; ')
    ssh(host, setup)
    d = inspect(host, ctr)
    kernel = None
    if host == 'tower':
        command = ('pid=$(docker inspect --format "{{.State.Pid}}" '+ctr+'); '
                   'cg=$(awk -F: \'$1==0 {print $3}\' /proc/$pid/cgroup); '
                   'cat "/sys/fs/cgroup$cg/memory.max" '
                   '"/sys/fs/cgroup$cg/memory.swap.max" "/sys/fs/cgroup$cg/memory.swap.current"')
        kernel = [int(v) for v in ssh(host, command).splitlines()]
    checks = envelope(binding, host, ctr, d, deadline, kernel)
    checks['holding'] = False
    for _ in range(20):
        logs = subprocess.run(['ssh', '-o', 'BatchMode=yes', host,
                               sudo+'docker logs --since 60s '+ctr], capture_output=True, text=True)
        if 'holding' in logs.stdout+logs.stderr:
            checks['holding'] = True
            break
        time.sleep(0.5)
    post = capacity(host)
    admit(host, post, ctr)
    checks['one_STEER'] = [w['name'] for w in post['workers']
                            if w['run'].startswith('jobs/fitsteer-')] == ['/'+ctr]
    checks['claim_window_remaining'] = int(ssh(host, 'date -u +%s').strip()) <= deadline-5
    checks['pause_owner_unchanged'] = control(bundle, js) == paused
    checks['not_operator_paused'] = not (launch/'PAUSE_WORKERS').exists()
    record = dict(time=datetime.now(timezone.utc).isoformat(), host=host, role=role,
                  jobset=js, container=ctr, checks=checks, before=before, after=post,
                  claim_deadline=deadline, kernel_swap=kernel,
                  packet_sha256=owner.PACKET_SHA, cell_wall_cap_sec=7663,
                  pass_timeout_sec=8000, packing_target_sec=1,
                  allocation_postpass_budget_minutes=1)
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    if not all(checks.values()):
        raise ValueError('worker envelope failed; control remains paused; see '+str(receipt))
    control(bundle, js, active)
    return dict(state='RELEASED', host=host, role=role, jobset=js, receipt=str(receipt))


def next_jobset(bundle, doc, status):
    """Use only complete canonical harvest metadata, never scientific values."""
    order = owner.steer_plan(bundle, doc)['order']
    if status.get('packet_sha256') != owner.PACKET_SHA:
        raise ValueError('harvest packet differs')
    if status.get('state') == 'DEGRADED' or status.get('errors'):
        raise ValueError('canonical harvest reports a defect')
    if status.get('state') == 'RUNNING':
        return 'WAIT'
    if set(status.get('jobsets', {})) != set(order):
        raise ValueError('canonical harvest inventory incomplete')
    for js in order:
        s = status['jobsets'][js]
        if s.get('state') != 'OK' or any(s.get('ledger', {}).get(k, 0)
                                        for k in ('failed', 'poison', 'poisoned')):
            raise ValueError('canonical harvest has failed or invalid evidence')
        installed = s.get('installed')
        if type(installed) is not int or not 0 <= installed <= doc['jobsets'][js]:
            raise ValueError('canonical installed count invalid')
    return next((js for js in order
                 if status['jobsets'][js]['installed'] < doc['jobsets'][js]), None)


def metadata_signature(status):
    if status.get('state') not in ('PROGRESS', 'COMPLETE') or status.get('errors'):
        return None
    return dict(packet_sha256=status['packet_sha256'], installed=status['installed'],
                jobsets={js: dict(installed=s['installed'], done=s['done'], ledger=s['ledger'])
                         for js, s in status['jobsets'].items()})


def attribute_native_rows(status, folded, manifest_ids):
    """Validate and count already-folded scheduler rows; never reduce ledger history."""
    from collections import Counter

    hosts = {h: dict(installed=0, failed=0, poison=0) for h in ('r3500', 'i270', 'tower')}
    if type(status['installed']) is not int or status['installed'] < 0:
        raise ValueError('canonical installed count is invalid')
    if set(status['jobsets']) != set(folded) or set(folded) != set(manifest_ids):
        raise ValueError('native metadata inventory differs from harvest inventory')
    for js, s in status['jobsets'].items():
        if s.get('state') != 'OK' or any(type(v) is not int or v < 0 for v in s['ledger'].values()):
            raise ValueError('canonical jobset state is invalid')
        if any(type(s[k]) is not int or s[k] < 0 for k in ('total', 'done', 'installed')):
            raise ValueError('canonical jobset count is invalid')
        if max(s['done'], s['installed']) > s['total']:
            raise ValueError('canonical jobset count exceeds its manifest')
        ids = manifest_ids[js]
        if len(ids) != len(set(ids)) or len(ids) != s['total']:
            raise ValueError('canonical manifest IDs are missing or duplicated')
        rows = folded[js]
        if len({r['job_id'] for r in rows}) != len(rows):
            raise ValueError('metadata must come from a native folded snapshot')
        if any(r['job_id'] not in ids for r in rows):
            raise ValueError('native snapshot contains a foreign job')
        if any(r['status'] not in ('done', 'failed', 'poison', 'poisoned') for r in rows):
            raise ValueError('native snapshot contains an unexpected scheduler state')
        counts = dict(Counter(r['status'] for r in rows))
        if (counts != s['ledger'] or counts.get('done', 0) != s['installed']
                or s['done'] != s['installed']):
            # A newly published cell may precede its canonical verified install.
            return None
        for r in rows:
            host = next((h for h in hosts if r['worker'].lower().startswith(h+'-')), None)
            if host is None:
                raise ValueError('native snapshot names a worker outside active placement')
            key = 'installed' if r['status'] == 'done' else 'failed' if r['status'] == 'failed' else 'poison'
            hosts[host][key] += 1
    if sum(v['installed'] for v in hosts.values()) != status['installed']:
        raise ValueError('native attribution does not cover canonical installed cells')
    return hosts


def scheduler_metadata(bundle, doc):
    """Read native compact snapshots without uploading, amnestying or inspecting fit data."""
    launch = bundle.parent/'launch'
    status = read_json(launch/'HARVEST_STATUS.json')
    signature = metadata_signature(status)
    if signature is None:
        return dict(state='WAIT_FOR_CANONICAL_HARVEST')
    if status['packet_sha256'] != owner.PACKET_SHA or set(status['jobsets']) != set(doc['jobsets']):
        raise ValueError('scheduler metadata packet or inventory differs')
    latest = launch/'SCHEDULER_METADATA.json'
    if latest.exists():
        previous = read_json(latest)
        if (previous.get('signature') == signature and previous.get('native_fold') is True
                and previous.get('upload') is False and previous.get('amnesty') is False):
            return previous
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = launch/'scheduler-metadata'/stamp
    dest.mkdir(parents=True)
    (dest/'HARVEST_STATUS.json').write_text(json.dumps(status)+'\n')
    binding = read_json(bundle/'QUALIFICATION_BUNDLE.json')['binaries']['zenfleet-ctl']
    ctl = Path(binding['path'])
    if owner.sha(ctl) != binding['sha256']:
        raise ValueError('frozen native metadata tool changed')
    private = Path.home()/'tmp/zensim-paper/rev4/private_telemetry/steer-launch'
    private.mkdir(parents=True, exist_ok=True)
    for js, s in status['jobsets'].items():
        ids = subprocess.check_output([str(ctl), 'ids', '--manifest',
                                       str(bundle/('fit-manifest-'+js+'.json'))], text=True)
        (dest/(js+'.ids.tsv')).write_text(ids)
        if s['ledger']:
            log = private/('native-metadata-'+stamp+'-'+js+'.log')
            with log.open('wb') as stream:
                try:
                    rc = subprocess.run([str(ctl), 'compact', '--run', js,
                                         '--endpoint', os.environ['EP'], '--out-dir', str(dest)],
                                        stdout=stream, stderr=stream, timeout=60).returncode
                except subprocess.TimeoutExpired:
                    raise ValueError('native metadata timed out; inspect restricted transport log') from None
            if rc or re.search(rb'(?im)^WARNING|[1-9][0-9]* unreadable', log.read_bytes()):
                raise ValueError('native metadata refused; inspect restricted transport log')
    code = ('import json\nfrom pathlib import Path\nimport pyarrow.parquet as pq\n'
            +source_inspect.getsource(attribute_native_rows)+'''
root=Path('/metadata');status=json.loads((root/'HARVEST_STATUS.json').read_text())
ids={};folded={}
for js,s in status['jobsets'].items():
 ids[js]=[line.split('\\t',2)[1] for line in (root/(js+'.ids.tsv')).read_text().splitlines()]
 folded[js]=(pq.read_table(root/('snap_'+js+'.parquet'),columns=['job_id','status','worker']).to_pylist()
             if s['ledger'] else [])
print(json.dumps(attribute_native_rows(status,folded,ids)))
''')
    command = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--memory', '512m', '--memory-swap', '512m', '--cpus', '1',
               '-e', 'TMPDIR=/metadata', '-v', str(dest)+':/metadata',
               '--entrypoint', '/usr/bin/nice', doc['image_ref'],
               '-n19', 'ionice', '-c3', 'python3', '-c', code]
    hosts = json.loads(subprocess.check_output(command, text=True))
    if hosts is None or metadata_signature(read_json(launch/'HARVEST_STATUS.json')) != signature:
        return dict(state='WAIT_FOR_CANONICAL_HARVEST', generation=str(dest))
    proof = dict(state='OK', time=datetime.now(timezone.utc).isoformat(), signature=signature,
                 packet_sha256=owner.PACKET_SHA, native_fold=True, upload=False, amnesty=False,
                 installed=status['installed'], per_host=hosts, generation=str(dest),
                 tool_sha256=binding['sha256'], model_or_target_values_read=False)
    (dest/'RESULT.json').write_text(json.dumps(proof, indent=2)+'\n')
    if latest.exists():
        latest.rename(latest.with_name(latest.name+'.bak-'+stamp))
    latest.write_text(json.dumps(proof, indent=2)+'\n')
    return proof


def supervise_metadata(bundle, status):
    path = bundle.parent/'launch/SCHEDULER_METADATA.json'
    if path.exists():
        previous = read_json(path)
        if (previous.get('state') == 'OK' and previous.get('native_fold') is True
                and previous.get('upload') is False and previous.get('amnesty') is False
                and previous.get('signature') == metadata_signature(status)):
            return previous
    command = ['flock', str(Path.home()/'tmp/zen-heavy.lock'),
               str(Path.home()/'work/claudehints/scripts/run-heavy'), '--mem', '2G', '--jobs', '1',
               '--', 'python3', str(Path(__file__).resolve()), '--packet', str(bundle),
               '--packet-sha', owner.PACKET_SHA, '--scheduler-metadata']
    value = json.loads(subprocess.check_output(command, text=True))
    return value if value.get('state') == 'OK' and value.get('signature') == metadata_signature(status) else None


def watch(bundle, doc):
    """Supervise bounded allocations; native worker remains the only job scheduler."""
    transition_proof(bundle)
    repaired_admission(bundle)
    launch = bundle.parent/'launch'
    log = launch/'LIFECYCLE.log'
    state_path = launch/'LIFECYCLE_STATE.json'
    state = read_json(state_path) if state_path.exists() else {}
    def note(value):
        value = dict(time=datetime.now(timezone.utc).isoformat(), **value)
        with log.open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
        if path := os.environ.get('STEER_STATUS_FILE'):
            with Path(path).open('a') as stream:
                stream.write('\n'+value['time']+' — LIFECYCLE '+json.dumps(value)+'\n')
    while True:
        status = read_json(launch/'HARVEST_STATUS.json')
        js = next_jobset(bundle, doc, status)
        metadata = None
        if js != 'WAIT':
            metadata = supervise_metadata(bundle, status)
            if metadata is None:
                js = 'WAIT'
        active = []
        for host in ['r3500', 'i270', 'tower']:
            cap = capacity(host)
            running = [w for w in cap['workers'] if w['run'].startswith('jobs/fitsteer-')]
            if min(cap['free'], cap['docker_free']) < 20*GIB:
                (launch/'PAUSE_WORKERS').touch(exist_ok=True)
                sudo = '' if host == 'tower' else 'sudo -n '
                for w in running:
                    ssh(host, sudo+'docker stop --time 30 '+shlex.quote(w['name']))
                raise ValueError('disk floor failed; own worker stopped on '+host)
            if running:
                if len(running) != 1:
                    raise ValueError('multiple STEER workers on '+host)
                active.append(host)
                continue
            previous = state.get(host)
            if previous:
                d = inspect(host, previous['container'])
                if d['State']['Running'] or d['State']['ExitCode'] != 0 or d['State']['OOMKilled'] or d['RestartCount']:
                    raise ValueError('previous bounded worker did not drain cleanly on '+host)
            if js == 'WAIT' or js is None or (launch/'PAUSE_WORKERS').exists():
                continue
            if previous:
                # Require a canonical verification cycle after that allocation finished.
                finished = datetime.fromisoformat(d['State']['FinishedAt'].replace('Z', '+00:00'))
                updated = datetime.fromisoformat(status['updated_at'].replace('Z', '+00:00'))
                if updated <= finished:
                    continue
            ctl = control(bundle, js)
            if ctl.get('paused') or ctl.get('drain'):
                continue
            role = 'steer-'+host+'-'+str(time.time_ns())
            command = ['flock', str(Path.home()/'tmp/zen-heavy.lock'),
                       str(Path.home()/'work/claudehints/scripts/run-heavy'), '--mem', '6G', '--jobs', '1',
                       '--', 'python3', str(Path(__file__).resolve()), '--packet', str(bundle),
                       '--packet-sha', owner.PACKET_SHA, '--host', host, '--role', role, '--jobset', js]
            note(dict(action='ENROLL', host=host, role=role, jobset=js))
            subprocess.run(command, check=True)
            # The enrollment may have honored a newly arrived pause without launching.
            receipt = launch/('ENVELOPE-'+role+'.json')
            if receipt.exists():
                state[host] = dict(container='zen-score-'+role, role=role, jobset=js)
                active.append(host)
                if state_path.exists():
                    state_path.rename(state_path.with_name(state_path.name+'.bak-'+str(time.time_ns())))
                state_path.write_text(json.dumps(state, indent=2)+'\n')
        note(dict(action='OBSERVE', installed=status.get('installed'),
                  harvest_state=status.get('state'), running_hosts=active,
                  installed_per_host=(None if metadata is None else
                                      {h: v['installed'] for h, v in metadata['per_host'].items()}),
                  operator_pause=(launch/'PAUSE_WORKERS').exists()))
        if js is None and not active:
            note(dict(action='COMPLETE', installed=172, workers_drained=True))
            return
        time.sleep(20)


def pause_workers(bundle, doc):
    """Close enrollment before publishing pause, serialized against allocation release."""
    (bundle.parent/'launch/PAUSE_WORKERS').touch(exist_ok=True)
    paused = []
    with (Path.home()/'tmp/zen-heavy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for js in owner.steer_plan(bundle, doc)['order']:
            previous = control(bundle, js)
            control(bundle, js, dict(previous, paused=True))
            paused.append(js)
    return dict(state='ENROLLMENT_AND_LATER_PASSES_PAUSED', jobsets=paused,
                current_fits='continue; native rolling claim deadlines still apply')


def preserve_coverage_scratch(scratch, declarations, generation):
    """Preserve exact manifest-bound temporary outputs; never reuse or delete them."""
    import hashlib
    import json
    from pathlib import Path
    import re
    import stat

    root = Path(scratch)
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
        raise ValueError('scratch must be a resolved disk path')
    tmp = root/'tmp'
    if tmp.is_symlink():
        raise ValueError('temporary directory must not be a symlink')
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}\.[0-9]{6}Z', generation):
        raise ValueError('invalid preservation generation')
    names = set()
    for mask, token in declarations:
        if not re.fullmatch('[0-9a-f]+', mask) or not re.fullmatch('[0-9a-f]{64}', token):
            raise ValueError('invalid manifest coverage identity')
        stem = 'coverage_cf'+mask+'_steer_'+token
        names.update(stem+suffix for suffix in
                     ('.parquet', '.keys.parquet', '.parquet.manifest.json'))
    files = []
    for name in sorted(names):
        path = tmp/name
        if not path.exists() and not path.is_symlink():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('temporary output is not a regular file')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        files.append(dict(name=name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    backup = root/'steer-interrupted-preserved'/generation
    if backup.exists() or backup.parent.is_symlink():
        raise ValueError('preservation destination already exists or is a symlink')
    if not files:
        return dict(files=[], generation=generation, scratch=str(root))
    backup.mkdir(parents=True)
    receipt = dict(files=files, generation=generation, scratch=str(root), backup=str(backup))
    (backup/'PLAN.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for item in files:
        (tmp/item['name']).rename(backup/item['name'])
        digest = hashlib.sha256()
        with (backup/item['name']).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('preserved temporary output changed')
    (backup/'PRESERVED.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def preserve_interrupted_scratch(bundle, doc, host):
    """Operator repair while enrollment and every STEER worker are quiescent."""
    launch = bundle.parent/'launch'
    if not (launch/'PAUSE_WORKERS').exists():
        raise ValueError('operator pause required before scratch preservation')
    binding = read_json(bundle/'WORKER_BINDING.json')
    if host not in ('r3500', 'i270', 'tower') or host not in binding['active']:
        raise ValueError('host outside frozen active placement')
    declarations = set()
    for js in owner.steer_plan(bundle, doc)['order']:
        ctl = control(bundle, js)
        if not (ctl.get('paused') or ctl.get('drain')):
            raise ValueError('all STEER controls must be paused')
        for job in read_json(bundle/('fit-manifest-'+js+'.json')):
            argv = job['kind']['argv']
            spec = argv[argv.index('--spec')+1]
            masks = [m[1] for part in spec.split(':')
                     if (m := re.fullmatch(r'cf([0-9a-f]+)', part))]
            if len(masks) != 1:
                raise ValueError('manifest lacks an exact coverage mask')
            declarations.add((masks[0], job['kind']['argv_sha']))
    cap = capacity(host)
    if any(w['run'].startswith('jobs/fitsteer-') for w in cap['workers']):
        raise ValueError('STEER worker still running; preserve only after it stops')
    if min(cap['free'], cap['docker_free']) < 20*GIB:
        raise ValueError('disk below 20 GiB floor')
    generation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    code = ('import json,sys\n'+source_inspect.getsource(preserve_coverage_scratch)
            +'\nprint(json.dumps(preserve_coverage_scratch(sys.argv[1],json.loads(sys.argv[2]),sys.argv[3])))')
    sudo = '' if host == 'tower' else 'sudo -n '
    scratch = binding['active'][host]['ZEN_TMPDIR_HOST_DIR']
    # The appliance host has no Python; keep tooling in the same pinned image.
    command = sudo+shlex.join([
        'docker', 'run', '--rm', '--network', 'none', '--memory', '256m',
        '--memory-swap', '256m', '--cpus', '1', '-v', scratch+':'+scratch,
        '-e', 'TMPDIR='+scratch+'/tmp', '--entrypoint', 'nice', binding['image'],
        '-n19', 'ionice', '-c3', 'python3', '-c', code, scratch,
        json.dumps(sorted(declarations)), generation])
    receipt = dict(host=host, packet_sha256=owner.PACKET_SHA,
                   scientific_changes=False, retry_limits_changed=False,
                   preservation=json.loads(ssh(host, command)))
    (launch/('SCRATCH_PRESERVED-'+host+'-'+generation+'.json')).write_text(
        json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--packet-sha', required=True)
    p.add_argument('--host')
    p.add_argument('--role')
    p.add_argument('--jobset')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--pause', action='store_true')
    p.add_argument('--preserve-interrupted-scratch', action='store_true')
    p.add_argument('--scheduler-metadata', action='store_true')
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', a.packet_sha):
        p.error('full packet SHA-256 required')
    owner.PACKET_SHA = a.packet_sha
    doc = owner.packet(a.packet)
    if a.scheduler_metadata:
        if a.pause or a.watch or a.preserve_interrupted_scratch or a.host:
            p.error('scheduler metadata is a separate read-only operation')
        print(json.dumps(scheduler_metadata(a.packet, doc)))
    elif a.preserve_interrupted_scratch:
        if a.pause or a.watch or not a.host:
            p.error('scratch preservation requires only --host')
        with (Path.home()/'tmp/zen-heavy.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(json.dumps(preserve_interrupted_scratch(a.packet, doc, a.host)))
    elif a.pause:
        if a.watch:
            p.error('--pause and --watch are mutually exclusive')
        print(json.dumps(pause_workers(a.packet, doc)))
    elif a.watch:
        with (a.packet.parent/'launch/lifecycle-supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            watch(a.packet, doc)
    else:
        if not all([a.host, a.role, a.jobset]):
            p.error('enrollment requires --host, --role and --jobset')
        if a.host not in ('r3500', 'i270', 'tower'):
            p.error('unsupported host')
        with (a.packet.parent/('launch/lifecycle-host-'+a.host+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(enroll(a.packet, doc, a.host, a.role, a.jobset)))


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError:
        print('STEER enrollment refused: transport command failed; inspect restricted transport log',
              file=sys.stderr)
        sys.exit(1)
