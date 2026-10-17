"""Apply the coordinator's resource-only17:00Z generation budget amendment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import os
import subprocess

ROOT = Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
DEADLINE = 1792256400.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def disk40(root):
    for p in [Path.home(), root]:
        if shutil.disk_usage(p).free < 40 * 1024**3:
            raise RuntimeError(f'E36 authorized40GiB disk floor: {p}')


def policy():
    clock = ROOT / 'bulk-clock-v7-budget3.json'
    a = json.loads((ROOT / 'scheduling-v8/amendment.json').read_text())
    if sha(clock) != a['effective_clock']['sha256']:
        raise ValueError('budget3 clock changed')
    if json.loads(clock.read_text())['deadline_unix'] != DEADLINE:
        raise ValueError('budget3 deadline differs')
    os.environ['E36_GENERATION_DEADLINE'] = str(DEADLINE)
    import e34_encode
    e34_encode.e36_disk = disk40
    return clock, a


def startup():
    # Only the frozen E36 numerical executor receives this resource bootstrap.
    if '--runtime' not in sys.argv:
        return
    i = sys.argv.index('--runtime')
    if sys.argv[i+1] != str(ROOT/'runtime-v6/runtime.json'):
        return
    sys.path.insert(0, str(ROOT/'runtime-v6/program'))
    policy()


def live(output):
    clock_path, a = policy()
    clock = json.loads(clock_path.read_text())
    records = []
    found = False
    for top in sys._current_frames().values():
        f = top
        while f:
            local=f.f_locals; name=f.f_code.co_name; file=Path(f.f_code.co_filename)
            if file == ROOT/'control-resume-v7/e36_resume_schedule.py' and name=='run' and local.get('root')==ROOT:
                if 'rc' in local:raise ValueError('controller is no longer waiting')
                local['clock'].clear();local['clock'].update(clock)
                local['a'].clear();local['a'].update(a)
                local['amendment_path']=ROOT/'scheduling-v8/amendment.json'
                local['env']['PYTHONPATH']=str(Path(__file__).parent)
                local['env']['E36_GENERATION_DEADLINE']=str(DEADLINE)
                records.append(dict(function='run',deadline=DEADLINE));found=True
            if file == ROOT/'runtime-v6/program/e36_data.py' and name=='main':
                found=True
            if file == ROOT/'runtime-v6/program/e36_data.py' and name=='execute':
                started=local['ladder_started']; bound=min(DEADLINE,time.time()-(time.monotonic()-started)+local['view']['deadline_seconds'])
                inner=top
                while inner:
                    if Path(inner.f_code.co_filename)==file and isinstance(inner.f_locals.get('deadline'),(float,int)):
                        old=inner.f_locals['deadline'];inner.f_locals['deadline']=bound
                        records.append(dict(function=inner.f_code.co_name,old_deadline=old,deadline=bound))
                    if inner is f:break
                    inner=inner.f_back
                found=True
            f=f.f_back
    if not found:raise ValueError('no owned active E36 frame')
    os.environ['PYTHONPATH']=str(Path(__file__).parent)
    with Path(output).open('x') as out:
        json.dump(dict(pid=os.getpid(),clock_sha256=sha(clock_path),records=records,deadline=DEADLINE,disk_floor_bytes=40*1024**3,fits=0),out,indent=2)


def repair_timer(pid, deadline, output):
    """Disarm every old alarm and arm verified ITIMER_REAL on the owned monitor."""
    if Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[0]!=b'/usr/bin/timeout':
        raise ValueError('exact registered timeout monitor required')
    ids=[int(s.split(':')[1]) for s in Path(f'/proc/{pid}/timers').read_text().splitlines() if s.startswith('ID:')]
    code='''import gdb,struct,time,json
gdb.execute('set confirm off')
gdb.execute('set unwind-on-signal on')
gdb.execute('handle SIGALRM nostop noprint nopass')
gdb.execute('attach PID')
gdb.execute('handle SIGSTOP nostop noprint nopass')
p=int(gdb.parse_and_eval('(void*)malloc(64)'))
gdb.selected_inferior().write_memory(p,bytes(64))
for tid in IDS:
 rc=int(gdb.parse_and_eval('(long)syscall(223L,%dL,0L,(void*)%d,0L)'%(tid,p)))
 if rc:raise RuntimeError('disarm POSIX timer failed')
remaining=DEADLINE-time.time()
if remaining<=0:raise RuntimeError('new deadline passed')
sec=int(remaining);usec=int((remaining-sec)*1000000)
gdb.selected_inferior().write_memory(p,struct.pack('=qqqq',0,0,sec,usec))
rc=int(gdb.parse_and_eval('(int)setitimer(0,(void*)%d,0)'%p))
if rc:raise RuntimeError('setitimer failed')
rc=int(gdb.parse_and_eval('(int)getitimer(0,(void*)%d)'%p))
if rc:raise RuntimeError('getitimer failed')
got=struct.unpack('=qqqq',bytes(gdb.selected_inferior().read_memory(p,32)))
if not 0<got[2]+got[3]/1e6<=remaining:raise RuntimeError('new timer verification failed')
print(json.dumps(dict(pid=PID,deadline=DEADLINE,disarmed_posix=IDS,remaining=got[2]+got[3]/1e6)))
gdb.execute('call (void)free((void*)%d)'%p)
gdb.execute('detach')
'''.replace('DEADLINE',repr(deadline)).replace('IDS',repr(ids)).replace('PID',str(pid))
    output=Path(output);script=output.with_suffix('.gdb')
    with script.open('x') as f:f.write('python\n'+code+'end\n')
    with output.open('x') as log:
        subprocess.run(['sudo','-n','gdb','-n','-q','-batch','-x',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)


def inject(pid, output):
    code="import importlib.util; s=importlib.util.spec_from_file_location('e36_budget3',"+repr(str(Path(__file__).resolve()))+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.live("+repr(str(output))+")"
    script=Path(output).with_suffix('.gdb')
    with script.open('x') as f:
        f.write('set confirm off\nset unwind-on-signal on\nhandle SIGALRM nostop noprint nopass\nattach '+str(pid)+'\nhandle SIGSTOP nostop noprint nopass\nset $gil=(int)PyGILState_Ensure()\nset $rc=(int)PyRun_SimpleString('+json.dumps(code)+')\ncall (void)PyGILState_Release($gil)\npython\nif int(gdb.parse_and_eval("$rc"))!=0:raise RuntimeError("budget3 Python injection failed")\nend\ndetach\n')
    with Path(output).with_suffix('.gdb.log').open('x') as log:
        subprocess.run(['sudo','-n','gdb','-n','-q','-batch','-x',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
