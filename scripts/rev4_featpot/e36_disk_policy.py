"""Authorized local resource pause: native zenfleet retains all claim ownership."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import types
import uuid
from e36_artifact_location import sync_directory

ROOT=Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
POLICY=ROOT/'owner-svt-ssd-v1/urgent-disk-policy.json'
STATE=ROOT/'owner-svt-ssd-v1/disk-claim-state.json'


def sticky_stop(free):
    stop=ROOT/'owner-svt-ssd-v1/DISK20_STOP.json'
    if free<20*1024**3 and not stop.exists():
        try:
            with stop.open('x') as stream:
                json.dump(dict(utc=time.time(),free_bytes=free,reason='absolute20GiBstop; no process killed',fits=0),stream)
                stream.flush();os.fsync(stream.fileno())
            sync_directory(stop.parent)
        except FileExistsError:pass
    return stop.exists()


def disk20(root):
    if sticky_stop(shutil.disk_usage(root).free):
        raise OSError('coordinator20GiB absolute stop; incomplete staging retained')


def wait(deadline):
    paused=False
    while True:
        free=shutil.disk_usage(ROOT).free
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        stopped=sticky_stop(free)
        if free<30*1024**3:paused=True
        if paused and free>40*1024**3:paused=False
        if not stopped and not paused and not state.get('claims_paused',False) and not (ROOT/'owner-svt-ssd-v1/resource-failure-hold.json').exists():return
        if time.time()>=deadline:raise TimeoutError('registered deadline expired during disk admission wait')
        time.sleep(.2)


def install(runtime):
    if getattr(runtime,'_e36_disk_policy',False)==2:return
    # Remove the superseded45GiB pre-command wrappers. An already-running
    # physical cell may finish; admission now waits at the next cell boundary.
    call=runtime.call
    while hasattr(call,'__func__') and Path(call.__func__.__code__.co_filename).name in ['e36_disk_reserve.py','disk_wait.py']:
        names=call.__func__.__code__.co_freevars
        key='original' if 'original' in names else 'old'
        call=call.__func__.__closure__[names.index(key)].cell_contents
    runtime.call=call
    original=runtime.execute_cell
    def cell(owner,*args,**kwargs):
        if len(args)>1 and args[1]=='rav1e':return original(*args,**kwargs)
        deadline=kwargs['deadline'] if 'deadline' in kwargs else args[6]
        wait(deadline)
        return original(*args,**kwargs)
    runtime.execute_cell=types.MethodType(cell,runtime)
    import e34_encode
    e34_encode.e36_disk=disk20
    runtime._e36_disk_policy=2


def atomic(path,value):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('x') as stream:
        json.dump(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    sync_directory(path.parent)


def paused_state(free,previous,hard):
    if hard or free<30*1024**3:return True
    if previous and free<=40*1024**3:return True
    return False


def io_bytes(scope):
    if not (scope/'io.stat').exists():
        device=os.stat(ROOT).st_dev
        stat=Path('/sys/dev/block')/(str(os.major(device))+':'+str(os.minor(device)))/'stat'
        return int(stat.read_text().split()[6])*512,'host SSD sectors written; includes other lanes'
    total=0
    for line in (scope/'io.stat').read_text().splitlines():
        values=dict(v.split('=',1) for v in line.split()[1:])
        total+=int(values.get('wbytes',0))
    return total,'E36 run-heavy scope io.stat wbytes'


def monitor():
    policy=json.loads(POLICY.read_text())
    scope=Path('/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().strip().split('::')[1])
    expected=ROOT/'bin/zenfleet-worker-fair-v7'
    previous=json.loads(STATE.read_text()) if STATE.exists() else {}
    paused=previous.get('claims_paused',False)
    stopped={int(v['pid']):v for v in previous.get('owned_stops',[])}
    last=None;offset=0;mirrored=0;reclaimed=0;last_emit=0
    relocation=ROOT/'owner-svt-ssd-v1/artifact-relocations.jsonl'
    telemetry=ROOT/'owner-svt-ssd-v1/disk-rates.jsonl'
    while time.time()<1792256400 and not (ROOT/'owner-svt-ssd-v1/disk-monitor-stop.json').exists():
        now=time.time();free=shutil.disk_usage(ROOT).free
        hard=sticky_stop(free)
        hold_path=ROOT/'owner-svt-ssd-v1/resource-failure-hold.json'
        held=json.loads(hold_path.read_text()) if hold_path.exists() else None
        paused=paused_state(free,paused,hard or held is not None)
        workers={}
        for entry in (scope/'cgroup.procs').read_text().split():
            p=Path('/proc')/entry
            try:
                if p.joinpath('exe').resolve()==expected:
                    fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
                    workers[int(entry)]=dict(pid=int(entry),start_ticks=fields[19],state=fields[0])
            except (OSError,ProcessLookupError):continue
        if held is not None and held['scope']==str(scope) and held['expected_worker']==str(expected):
            for item in held['owned_stops']:
                if item['pid'] in workers and workers[item['pid']]['start_ticks']==item['start_ticks']:
                    stopped[item['pid']]=item
        if paused:
            stopped={pid:item for pid,item in stopped.items() if pid in workers and workers[pid]['start_ticks']==item['start_ticks']}
            for pid,item in workers.items():
                if pid in stopped:
                    if item['state']!='T':
                        try:os.kill(pid,signal.SIGSTOP)
                        except ProcessLookupError:pass
                    continue
                if item['state']=='T':continue
                # Commit owned intent before the signal. A restart replays it
                # only for the same exact executable/scope/process lifetime.
                stopped[pid]=dict(pid=pid,start_ticks=item['start_ticks'])
                atomic(STATE,dict(utc=now,claims_paused=True,hard_stop=hard,native_workers=sorted(workers),
                                 owned_stops=list(stopped.values()),scope=str(scope),fits=0))
                try:os.kill(pid,signal.SIGSTOP)
                except ProcessLookupError:pass
        else:
            for pid in list(stopped):
                if pid in workers and workers[pid]['start_ticks']==stopped[pid]['start_ticks']:
                    try:os.kill(pid,signal.SIGCONT)
                    except ProcessLookupError:pass
                del stopped[pid]
        current=dict(utc=now,free_bytes=free,claims_paused=paused,hard_stop=hard,native_workers=sorted(workers),
                     owned_stops=list(stopped.values()),scope=str(scope),fits=0)
        saved=json.loads(STATE.read_text()) if STATE.exists() else {}
        if not STATE.exists() or saved.get('claims_paused')!=paused or saved.get('owned_stops')!=current['owned_stops'] or saved.get('native_workers')!=current['native_workers']:
            atomic(STATE,current)
            print(json.dumps(dict(event='claim-state',**current)),flush=True)
        ready=os.environ.get('E36_RESOURCE_READY_PATH')
        if ready and not Path(ready).exists():
            atomic(Path(ready),dict(role='monitor',pid=os.getpid(),scope=str(scope),utc=now,fits=0))
        if now-last_emit>=30:
            try:
                if relocation.exists():
                    with relocation.open() as stream:
                        stream.seek(offset)
                        while True:
                            start=stream.tell();line=stream.readline()
                            if not line:break
                            if not line.endswith('\n'):stream.seek(start);break
                            record=json.loads(line);mirrored+=record['bytes'];reclaimed+=record.get('local_allocated_bytes',0)
                            offset=stream.tell()
                        offset=stream.tell()
                write_bytes,write_source=io_bytes(scope)
                counters=dict(utc=now,free_bytes=free,observed_write_bytes=write_bytes,write_counter_source=write_source,verified_mirror_bytes=mirrored,recorded_local_reclaimed_bytes=reclaimed)
                if last:
                    seconds=now-last['utc'];counters.update(window_seconds=seconds,
                        observed_ssd_write_bytes_per_second=(write_bytes-last['observed_write_bytes'])/seconds,
                        verified_mirror_bytes_per_second=(mirrored-last['verified_mirror_bytes'])/seconds,
                        host_net_free_consumption_bytes_per_second=(last['free_bytes']-free)/seconds)
                with telemetry.open('a') as stream:stream.write(json.dumps(counters)+'\n')
                print(json.dumps(counters),flush=True);last=counters
            except (OSError,ValueError,KeyError) as error:
                # Rate accounting cannot disable native admission control.
                try:print(json.dumps(dict(event='telemetry-error',error=str(error),utc=now)),flush=True)
                except OSError:pass
            last_emit=now
        time.sleep(1)


if __name__=='__main__':monitor()
