"""Exercise the actual pinned worker on invented jobs, without training or a store."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def entrypoint_probe(args, root):
    """Actual image wrapper, invented pass/store stubs, and no trainer or network."""
    original = args.original_entrypoint.read_bytes()
    guarded = args.guarded_entrypoint.read_bytes()
    guard = '''  if [ -n "${ZEN_NO_NEW_AFTER_EPOCH:-}" ] && [ "$(date +%s)" -ge "$ZEN_NO_NEW_AFTER_EPOCH" ]; then
    prog "$WORKER admission deadline reached; in-flight pass finished — exiting clean"
    break
  fi
'''.encode()
    if guarded.count(guard) != 1 or guarded.replace(guard, b'', 1) != original:
        raise ValueError('proposal must add only the exact four-line loop guard')
    (root/'bin').mkdir()
    store = root/'bin/s5cmd'
    store.write_text('''#!/bin/bash
case "$*" in
 *manifest.json.gz*) exit 1;;
 *manifest.json*) printf '[]\\n' > "${@: -1}";;
 *ledger_snapshot.parquet*) exit 1;;
 *pipe*) cat >/dev/null;;
 *) echo 'unexpected invented store call' >&2; exit 9;;
esac
''')
    worker = root/'bin/zenfleet-worker'
    worker.write_text('''#!/bin/bash
printf 'start %s\\n' "$(date +%s)" >> "$PROBE_DIR/events"
if [ "$PROBE_MODE" = work ]; then
 while [ "$(date +%s)" -le "${PROBE_END_EPOCH:-$ZEN_NO_NEW_AFTER_EPOCH}" ]; do sleep 0.1; done
 printf 'end %s\\n' "$(date +%s)" >> "$PROBE_DIR/events"
 echo 'zenfleet-worker: done=1 failed=0 poisoned=0 skipped=26 rows=1 already_poison=0'
else
 printf 'end %s\\n' "$(date +%s)" >> "$PROBE_DIR/events"
 echo 'zenfleet-worker: done=0 failed=0 poisoned=0 skipped=0 rows=0 already_poison=0'
fi
''')
    for p in [store, worker]:
        p.chmod(0o755)
    results = {}
    for case, script, mode, deadline_offset in [
            ('original_holds', args.original_entrypoint, 'work', 4),
            ('guarded_drains', args.guarded_entrypoint, 'work', 4),
            ('expired_no_pass', args.guarded_entrypoint, 'work', -1),
            ('unset_idle_unchanged', args.guarded_entrypoint, 'empty', None),
            ('original_budget_drains', args.original_entrypoint, 'work', 4)]:
        directory = root/case; directory.mkdir(); (directory/'tmp').mkdir()
        name = 'steer-entrypoint-probe-'+str(os.getpid())+'-'+case.replace('_', '-')
        deadline = int(time.time())+deadline_offset if deadline_offset is not None else None
        env = dict(TMPDIR='/probe/'+case+'/tmp', PROBE_DIR='/probe/'+case,
                   PROBE_MODE=mode, PATH='/probe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                   ZEN_R2_ENDPOINT='http://invented.invalid', ZEN_BUCKET='invented',
                   ZEN_RUN='jobs/invented', ZEN_MANIFEST_URI='s3://invented/manifest.json',
                   ZEN_WORKER='invented', ZEN_PROVIDER='engineering', ZEN_REQUIRE_SNAPSHOT='0',
                   ZEN_MAX_MIN='150', ZEN_PASS_TIMEOUT='15', ZEN_IDLE_PASSES='1', ZEN_LONG_LIVED='0')
        if deadline is not None:
            env['ZEN_NO_NEW_AFTER_EPOCH'] = str(deadline)
        if case == 'original_budget_drains':
            # Existing post-pass allocation budget must not truncate the in-flight pass.
            env['ZEN_MAX_MIN'] = '1'
            env['PROBE_END_EPOCH'] = str(int(time.time())+65)
            env['ZEN_PASS_TIMEOUT'] = '80'
        command = ['docker', 'run', '-d', '--pull', 'never', '--restart', 'no',
                   '--name', name, '--network', 'none', '--memory', '512m',
                   '--memory-swap', '512m', '--cpus', '1',
                   '-v', str(root)+':/probe',
                   '-v', str(script.resolve())+':/usr/local/bin/fleet-entrypoint.sh:ro',
                   '--entrypoint', '/usr/bin/nice']
        for key, value in env.items():
            command += ['-e', key+'='+value]
        command += [args.image, '-n19', 'ionice', '-c3', '/usr/local/bin/fleet-entrypoint.sh']
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        state = None
        try:
            until = time.monotonic()+(80 if case == 'original_budget_drains' else 15)
            while time.monotonic() < until:
                state = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
                events = (directory/'events').read_text().splitlines() if (directory/'events').exists() else []
                if case == 'original_holds':
                    assert state['State']['Running'], 'original exited before the holding observation'
                    if sum(e.startswith('end ') for e in events) >= 2:
                        break
                elif not state['State']['Running']:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError('entrypoint oracle timed out in '+case)
            assert state is not None and not state['State']['OOMKilled']
            if case != 'original_holds':
                assert not state['State']['Running'] and state['State']['ExitCode'] == 0
            if case in ('guarded_drains', 'original_budget_drains'):
                assert len(events) == 2 and events[0].startswith('start ') and events[1].startswith('end ')
                assert int(events[0].split()[1]) < deadline < int(events[1].split()[1])
                if case == 'original_budget_drains':
                    assert int(events[1].split()[1])-int(events[0].split()[1]) > 60
            elif case == 'expired_no_pass':
                assert events == [], 'expired allocation started a pass'
            elif case == 'unset_idle_unchanged':
                assert len(events) == 2
            results[case] = dict(events=events, state=state['State'], deadline=deadline)
        finally:
            log = subprocess.run(['docker', 'logs', name], capture_output=True)
            (directory/'entrypoint.log').write_bytes(log.stdout+log.stderr)
            subprocess.run(['docker', 'stop', '--time', '2', name], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(['docker', 'rm', name], check=True, stdout=subprocess.DEVNULL)
    receipt = dict(schema='steer-entrypoint-deadline-probe-v1', image=args.image,
                   scientific_fits=0, network='none', registered_results_touched=False,
                   original_sha256=hashlib.sha256(original).hexdigest(),
                   proposed_sha256=hashlib.sha256(guarded).hexdigest(), cases=results)
    (root/'PASS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--original-entrypoint', type=Path)
    parser.add_argument('--guarded-entrypoint', type=Path)
    args = parser.parse_args()
    if '@sha256:' not in args.image:
        parser.error('an immutable image digest is required')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    if args.original_entrypoint or args.guarded_entrypoint:
        if not (args.original_entrypoint and args.guarded_entrypoint):
            parser.error('both original and guarded entrypoint paths are required')
        entrypoint_probe(args, root)
        return
    jobs = []
    for i in range(40):
        argv = ['engineering-only-no-fit', str(i)]
        argv_sha = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
        jobs.append(dict(
            kind=dict(kind='fit_cell', program_sha='0'*64, data_sha='1'*64,
                      argv_sha=argv_sha, argv=argv),
            cell=dict(image_path=f'engineering-only-cell-{i}', codec='fit', q=0,
                      knob_tuple_json='{}'),
            inputs=['0'*64, '1'*64, argv_sha],
            hint=dict(peak_mem_bytes=2147483648, threads=4, vram_bytes=None),
            requires=['fit-cell-v1']))
    (root/'manifest.json').write_text(json.dumps(jobs)+'\n')
    fake = root/'fake-exec'
    fake.write_text('#!/bin/sh\n'
                    'if [ "${1:-}" = capabilities ]; then printf "fit-cell-v1\\n"; exit 0; fi\n'
                    'cat >/dev/null\nprintf "engineering-only-no-fit-bytes\\n"\n')
    fake.chmod(0o755)
    cpu = min(os.sched_getaffinity(0))
    base = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--memory', '1g',
            '--memory-swap', '1g', '--cpuset-cpus', str(cpu),
            '-e', 'TMPDIR=/probe/tmp', '-v', str(root)+':/probe']
    for target in [7663, 1]:
        (root/f'target-{target}').mkdir()
        command = [*base, '--entrypoint', '/usr/bin/nice',
                   '-e', f'ZEN_CHUNK_WALL_SEC={target}',
                   '-e', 'ZEN_CORE_OVERSUBSCRIBE=1', '-e', 'ZEN_CLAIM_ORDER=manifest',
                   args.image, '-n19', 'ionice', '-c3', 'zenfleet-worker',
                   '--manifest', '/probe/manifest.json',
                   '--ledger-out', f'/probe/target-{target}/out.parquet',
                   '--blobs', f'/probe/target-{target}/blobs',
                   '--exec', '/probe/fake-exec', '--worker', 'engineering-only',
                   '--provider', 'engineering']
        with (root/f'target-{target}/worker.log').open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=stream)
    # Decode only invented ledger metadata, using the same image's PyArrow.
    check = '''import json
from pathlib import Path
import pyarrow.parquet as pq
rows={}
for target in [7663,1]:
 files=sorted(Path('/probe/target-'+str(target)).glob('*.parquet'))
 tables=[pq.read_table(p,columns=['job_id','status']).to_pylist() for p in files]
 flat=[row for table in tables for row in table]
 assert len(flat)==len({r['job_id'] for r in flat})==40
 assert all(r['status']=='done' for r in flat)
 rows[str(target)]={'sidecars':len(files),'rows_per_sidecar':[len(t) for t in tables]}
assert rows['7663']=={'sidecars':10,'rows_per_sidecar':[4]*10}
assert rows['1']=={'sidecars':40,'rows_per_sidecar':[1]*40}
print(json.dumps(rows))
'''
    result = subprocess.check_output([*base, '--entrypoint', 'python3', args.image,
                                      '-c', check], text=True)
    receipt = dict(schema='steer-worker-scheduling-probe-v1', image=args.image,
                   scientific_fits=0, network='none', registered_results_touched=False,
                   argv_and_data='invented; no trainer or target file invoked',
                   measurements=json.loads(result))
    (root/'PASS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
