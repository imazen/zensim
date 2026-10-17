"""Continue the existing native E36 ledger after an expired control process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import signal
import shutil

from e36_store import sha
from e36_svt_matrix import configuration


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2)


def same_worker(pid,start_ticks,scope,worker):
    process=Path('/proc')/str(pid)
    try:
        group=Path('/sys/fs/cgroup'+(process/'cgroup').read_text().strip().split('::')[1])
        fields=(process/'stat').read_text().rsplit(')',1)[1].split()
        return group==scope and (process/'exe').resolve()==worker and fields[19]==start_ticks
    except (OSError,ProcessLookupError):return False


def worker_handles(scope,worker):
    result=[]
    for pid in (scope/'cgroup.procs').read_text().split():
        handle=None
        try:
            handle=os.pidfd_open(int(pid))
            fields=(Path('/proc')/pid/'stat').read_text().rsplit(')',1)[1].split()
            if not same_worker(int(pid),fields[19],scope,worker):
                os.close(handle);continue
            result.append((dict(pid=int(pid),start_ticks=fields[19],state=fields[0]),handle))
        except (OSError,ProcessLookupError):
            if handle is not None:os.close(handle)
    return result


def hold_admission(root,scope,worker,reason,memory=None):
    from e36_disk_policy import atomic,sticky_stop
    owner=root/'owner-svt-ssd-v1';memory={} if memory is None else memory
    # Preserve verified ownership of a worker already paused by this resource
    # owner. Never adopt a stopped process from its state alone.
    for path in [owner/'resource-failure-hold.json',owner/'disk-claim-state.json']:
        try:
            prior=json.loads(path.read_text())
            if prior['scope']==str(scope):
                for item in prior.get('owned_stops',[]):memory[(item['pid'],item['start_ticks'])]=item
        except (OSError,ValueError,KeyError):pass
    handles=worker_handles(scope,worker);selected=[];error=None
    try:
        for item,handle in handles:
            key=(item['pid'],item['start_ticks'])
            if item['state']=='T' and key not in memory:continue
            if not same_worker(item['pid'],item['start_ticks'],scope,worker):continue
            record=dict(pid=item['pid'],start_ticks=item['start_ticks'])
            memory[key]=record;selected.append((record,handle))
        stops=[item for item,_ in selected]
        hold=dict(scope=str(scope),expected_worker=str(worker),owned_stops=stops,reason=reason,utc=time.time(),fits=0)
        try:
            atomic(owner/'resource-failure-hold.json',hold)
            atomic(owner/'disk-claim-state.json',dict(claims_paused=True,hard_stop=sticky_stop(shutil.disk_usage(root).free),scope=str(scope),owned_stops=stops,native_workers=[v['pid'] for v in stops],utc=time.time(),fits=0))
        except OSError as failure:
            # A failed receipt write must not bypass the owned producer pause.
            # The live supervisor retains ownership in memory and retries.
            error=str(failure)
        finally:
            for item,handle in selected:
                if same_worker(item['pid'],item['start_ticks'],scope,worker):
                    try:signal.pidfd_send_signal(handle,signal.SIGSTOP)
                    except ProcessLookupError:pass
        return dict(owned_stops=stops,persistence_error=error)
    finally:
        for _,handle in handles:os.close(handle)


def supervise(native,roles,root,scope,worker,force_reason=None,forced_event=None,memory=None):
    import sys
    memory={} if memory is None else memory;reason=force_reason;reported=None
    while native.poll() is None:
        if reason is None and (any(child.poll() is not None for child in roles.values()) or forced_event is not None and forced_event.is_set()):
            reason='required resource child exited or controller cleanup was attempted during live production'
        if reason is not None:
            try:
                result=hold_admission(root,scope,worker,reason,memory)
                error=result['persistence_error']
            except Exception as failure:error=repr(failure)
            if error and error!=reported:
                print('E36 resource hold persistence/reconciliation failure; retaining supervision: '+error,file=sys.stderr,flush=True)
                reported=error
        time.sleep(.2)
    return native.returncode


def drain_before_cleanup(native,roles,root,scope,worker,memory):
    while native is not None and native.poll() is None:
        try:supervise(native,roles,root,scope,worker,force_reason='controller exceptional exit',memory=memory)
        except BaseException as failure:
            print('E36 retaining native supervision after controller exception: '+repr(failure),flush=True)


def wait_resources(roles,out,scope,timeout=30):
    until=time.monotonic()+timeout
    while True:
        if any(child.poll() is not None for child in roles.values()):raise RuntimeError('resource observer exited before producer')
        if all((out/(role+'-ready.json')).exists() for role in roles):break
        if time.monotonic()>=until:raise TimeoutError('required resource owner did not become ready')
        time.sleep(.1)
    for role,child in roles.items():
        ready=json.loads((out/(role+'-ready.json')).read_text())
        if ready['pid']!=child.pid or ready['scope']!=str(scope) or ready['role']!=role:
            raise ValueError('resource readiness identity differs')


def run(root,out,phase_name='fallback'):
    proof,clock,_=configuration(root)
    if time.time()>=clock['deadline_unix']:raise TimeoutError('registered17Z deadline passed')
    if (root/'bulk-choice.json').exists():raise ValueError('existing final choice must not be replaced')
    if phase_name not in ['fallback','additions']:raise ValueError('registered phase required')
    if phase_name=='additions':
        from e36_svt_matrix import publication
        publication(root,proof['phases']['fallback']['run'])
    old=root/'runs'/proof['phases'][phase_name]['old_run' if phase_name=='fallback' else 'run']
    original=json.loads((root/'scheduling-v7/amendment.json').read_text())
    worker=Path(original['worker']['path']);scheduled=Path(original['manifests'][phase_name]['path'])
    expected_scheduled=original['manifests'][phase_name]['sha256']
    if phase_name=='additions':
        scheduled=Path(proof['phases'][phase_name]['scheduled']['path'])
        expected_scheduled=proof['phases'][phase_name]['scheduled']['sha256']
    for path,digest in [(worker,original['worker']['sha256']),(scheduled,expected_scheduled)]:
        if sha(path)!=digest:raise ValueError('frozen original native scheduling input changed')
    if not (root/'owner-svt-ssd-v1/native-exit-selected-status-v1.json').exists():
        raise ValueError('retained prior native exit audit required')
    prior=json.loads((root/'owner-svt-ssd-v1/native-exit-selected-status-v1.json').read_text())
    if prior['bad']:raise ValueError('required prior native failures need explicit adjudication')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            if (process/'exe').resolve()==worker:
                raise ValueError('a native producer is still active')
            args=(process/'cmdline').read_bytes().split(b'\0')
            if len(args)>1 and args[1]==str(root/'runtime-v6/program/e36_data.py').encode():
                raise ValueError('an original numerical executor is still active')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    import pyarrow.parquet as pq
    phase=proof['phases'][phase_name]
    ids=lambda path:[line.split('\t')[1] for line in path.read_text().splitlines()]
    universe=dict(zip(ids(Path(phase['original_native_ids']['path'])),json.loads(Path(phase['original']['path']).read_text())))
    selected=set(ids(Path(phase['native_ids']['path'])))
    ledgers=sorted(old.glob('ledger*.parquet'));completed={}
    for path in ledgers:
        for batch in pq.ParquetFile(path).iter_batches(columns=['job_id','status','attempts','output_sha','kind_json']):
            for row in batch.to_pylist():
                key=row['job_id']
                if key not in universe or json.loads(row['kind_json'])!=universe[key]['kind']:
                    raise ValueError('prior native identity changed')
                if key in selected:
                    if row['status']!='done' or row['attempts']!=1 or not row['output_sha']:
                        raise ValueError('prior required native row failed')
                    if completed.setdefault(key,row['output_sha'])!=row['output_sha']:
                        raise ValueError('conflicting prior native output')
    out.mkdir()
    snapshot=out/'ledger-inputs';snapshot.mkdir()
    preserved=[]
    for path in ledgers:
        destination=snapshot/path.name;shutil.copy2(path,destination)
        if sha(destination)!=sha(path):raise ValueError('native ledger snapshot differs')
        preserved.append(destination)
    scope=Path('/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().strip().split('::')[1])
    if int((scope/'memory.max').read_text())!=38*1024**3:
        raise ValueError('exact38GiB run-heavy scope required')
    owner=root/'owner-svt-ssd-v1'
    for stop in ['mirror-stop.json','disk-monitor-stop.json','DISK20_STOP.json','resource-failure-hold.json']:
        if (owner/stop).exists():raise ValueError('uncleared resource stop: '+stop)
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    budget=min(30*1024**3,available-8*1024**3)
    if budget<4*1024**3:raise RuntimeError('insufficient live RAM beyond8GiB reserve')
    argv=['/usr/bin/timeout','--signal=TERM','--kill-after=120',str(clock['deadline_unix']-time.time()),
          'taskset','--cpu-list','0-23',str(worker),'--manifest',str(scheduled),
          '--ledger-out',str(old/'ledger.parquet'),'--blobs',str(old/'blobs'),
        '--exec',str(root/'runtime-v6/executor'),'--worker','e36-dev-owner-'+phase_name+'-v1',
          '--provider','local','--max-attempts','1']
    for ledger in preserved:argv.extend(['--ledger-in',str(ledger)])
    resources=owner/'resource-budget4-v2' if 'archive_budget4' in proof else owner/'closure-v6'
    if 'archive_transport' in proof:resources=owner/'resource-budget4-v3'
    env={**os.environ,'PYTHONPATH':str(resources),'ZEN_LOCAL_FAIR_SCHEDULER':'1',
         'ZEN_LOCAL_RESERVE_BYTES':str(8*1024**3),'ZEN_RAM_BYTES':str(budget*4//3),
         'ZEN_CHUNK_WALL_SEC':'300','ZEN_PERSISTENT_EXEC':'1','ZEN_PERSISTENT_KINDS':'bake',
         'RAYON_NUM_THREADS':'24','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'PYTHONDONTWRITEBYTECODE':'1','E36_GENERATION_DEADLINE':str(clock['deadline_unix'])}
    pin=lambda path:dict(path=str(path),sha256=sha(path))
    launch=dict(schema='e36-existing-native-ledger-continuation-v1',argv=argv,
        started_unix=time.time(),phase=phase_name,deadline_unix=clock['deadline_unix'],worker=pin(worker),
        scheduled=pin(scheduled),clock=proof['clock'],owner_matrix=pin(owner/'matrix-binding.json'),
        ledger_inputs=[pin(path) for path in preserved],original_ledger_paths=[str(path) for path in ledgers],prior_exit=pin(owner/'fallback-exit.json'),
        prior_audit=pin(owner/'native-exit-selected-status-v1.json'),source=pin(Path(__file__)),
        job_budget_bytes=budget,scope=str(scope),controller_pid=os.getpid(),prior_completed_required=len(completed),fits=0,scientific_change=False,
        note='Original native IDs/ledger/CAS; existing completed jobs retained, rav1e rejected by frozen owner overlay. No deadline extension.')
    children=[];native=None;roles={};ownership={}
    try:
        mirror=owner/'resource-budget4-v1/e36_artifact_mirror.py' if 'archive_budget4' in proof else owner/'mirror-v13/e36_artifact_mirror.py'
        if 'archive_transport' in proof:mirror=resources/'e36_artifact_mirror.py'
        for role,path in [('monitor',resources/'e36_disk_policy.py'),('mirror',mirror)]:
            log=(out/(role+'.log')).open('x')
            child=subprocess.Popen(['python3',str(path)],stdout=log,stderr=subprocess.STDOUT,
                                   env={**env,'E36_RESOURCE_READY_PATH':str(out/(role+'-ready.json'))})
            children.append(child)
            roles[role]=child
            suffix='resume' if phase_name=='fallback' else 'additions'
            write(owner/(role+'-'+suffix+'-v1-pid.json'),dict(pid=child.pid,scope=str(scope),utc=time.time(),fits=0))
        wait_resources(roles,out,scope)
        remaining=clock['deadline_unix']-time.time()
        if remaining<=0:raise TimeoutError('registered deadline passed during preparation')
        argv[3]=str(remaining);launch['started_unix']=time.time()
        write(out/'launch.json',launch)
        native=subprocess.Popen(argv,env=env)
        rc=supervise(native,roles,root,scope,worker,memory=ownership)
        write(out/'exit.json',dict(returncode=rc,closed_unix=time.time(),fits=0))
    finally:
        # Exceptions after spawn cannot shut down resource owners around a
        # live producer. Reconcile until the original native timeout drains.
        drain_before_cleanup(native,roles,root,scope,worker,ownership)
        # Cooperative resource shutdown; never terminate archival transfers.
        for name in ['mirror-stop.json','disk-monitor-stop.json']:
            if not (owner/name).exists():write(owner/name,dict(utc=time.time(),reason='native phase ended',fits=0))
        exits=[dict(pid=child.pid,returncode=child.wait()) for child in children]
        write(out/'helper-exits.json',dict(exits=exits,utc=time.time(),fits=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=['fallback','additions'],default='fallback')
    args=parser.parse_args();run(args.root,args.out,args.phase)
