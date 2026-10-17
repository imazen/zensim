"""The authorized 22Z resource-only clock and split archive resolver."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
DEADLINE=1792274400.0


def sibling(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location(name+'_budget4',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def policy():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    _,clock,_=sibling('e36_pause_budget').resolve(ROOT/'bulk-clock-v8-budget4.json')
    if clock['deadline_unix']!=DEADLINE:raise ValueError('authorized22Z clock required')
    os.environ['E36_GENERATION_DEADLINE']=str(DEADLINE)
    # The numerical runtime and local put owner stay frozen. Only the location
    # resolver gains the independently pinned second archive namespace.
    import e36_owner_amendment
    prior=ROOT/'owner-svt-ssd-v1/resource-budget4-v2/e36_artifact_location.py'
    spec=importlib.util.spec_from_file_location('e36_location_budget4_fixed',prior)
    location=importlib.util.module_from_spec(spec);spec.loader.exec_module(location)
    e36_owner_amendment.resolve=location.resolve
    sys.modules['e36_archive_store']=sibling('e36_archive_store')
    os.environ['E36_ARCHIVE_ROOT']=str(ROOT)
    return clock


def startup():
    if '--runtime' in sys.argv and sys.argv[sys.argv.index('--runtime')+1]==str(ROOT/'runtime-v6/runtime.json'):
        policy()


def live(output):
    policy();records=[];found=False
    for top in sys._current_frames().values():
        frame=top
        while frame:
            local=frame.f_locals;file=Path(frame.f_code.co_filename)
            if file==ROOT/'runtime-v6/program/e36_data.py' and frame.f_code.co_name=='main':found=True
            if file==ROOT/'runtime-v6/program/e36_data.py' and frame.f_code.co_name=='execute':
                bound=min(DEADLINE,time.time()-(time.monotonic()-local['ladder_started'])+local['view']['deadline_seconds'])
                inner=top
                while inner:
                    if isinstance(inner.f_locals.get('deadline'),(int,float)):
                        old=inner.f_locals['deadline'];inner.f_locals['deadline']=bound
                        records.append(dict(function=inner.f_code.co_name,old_deadline=old,deadline=bound))
                    if inner is frame:break
                    inner=inner.f_back
                found=True
            frame=frame.f_back
    if not found:raise ValueError('owned active E36 executor required')
    with Path(output).open('x') as stream:json.dump(dict(pid=os.getpid(),deadline=DEADLINE,records=records,fits=0,scientific_change=False),stream,indent=2)


def inject_code(pid,code,output):
    output=Path(output);script=output.with_suffix('.gdb')
    script.write_text('set confirm off\nset debuginfod enabled off\nset unwind-on-signal on\nattach '+str(pid)+
        '\nhandle SIGSTOP nostop noprint nopass\nset $gil=(int)PyGILState_Ensure()\nset $rc=(int)PyRun_SimpleString('+json.dumps(code)+
        ')\ncall (void)PyGILState_Release($gil)\npython\nassert int(gdb.parse_and_eval("$rc"))==0,"E36 resource application failed"\nend\ndetach\n')
    with output.open('x') as stream:subprocess.run(['sudo','-n','gdb','--readnever','-n','-q','-batch','-x',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True)


def repair_timer(pid,deadline,output):
    """Rearm the EXISTING uutils monotonic POSIX timer with a relative interval.

    No second timer is introduced. The real same-binary fixture must pass
    before applying this function to generation processes.
    """
    path=Path('/proc')/str(pid)
    if (path/'cmdline').read_bytes().split(b'\0')[0]!=b'/usr/bin/timeout':raise ValueError('exact timeout owner required')
    lines=(path/'timers').read_text().splitlines()
    ids=[int(line.split(':')[1]) for line in lines if line.startswith('ID:')]
    if len(ids)!=1 or 'ClockID: 1' not in lines:raise ValueError('single monotonic POSIX timer required')
    code='''import gdb,struct,time,json
gdb.execute('set confirm off')
gdb.execute('set unwind-on-signal on')
gdb.execute('attach PID')
gdb.execute('handle SIGSTOP nostop noprint nopass')
p=int(gdb.parse_and_eval('(void*)malloc(32)'))
remaining=DEADLINE-time.time()
if remaining<=0:raise RuntimeError('deadline passed')
seconds=int(remaining);nanos=int((remaining-seconds)*1000000000)
gdb.selected_inferior().write_memory(p,struct.pack('=qqqq',0,0,seconds,nanos))
rc=int(gdb.parse_and_eval('(long)syscall(223L,TIMERL,0L,(void*)%d,0L)'%p))
if rc:raise RuntimeError('timer_settime failed')
rc=int(gdb.parse_and_eval('(long)syscall(224L,TIMERL,(void*)%d)'%p))
if rc:raise RuntimeError('timer_gettime failed')
got=struct.unpack('=qqqq',bytes(gdb.selected_inferior().read_memory(p,32)))
if not 0<got[2]+got[3]/1e9<=remaining:raise RuntimeError('timer readback differs')
print(json.dumps(dict(pid=PID,deadline=DEADLINE,timer_id=TIMER,clock='CLOCK_MONOTONIC',relative_seconds=got[2]+got[3]/1e9)))
gdb.execute('call (void)free((void*)%d)'%p)
gdb.execute('detach')
'''.replace('PID',str(pid)).replace('DEADLINE',repr(deadline)).replace('TIMER',str(ids[0]))
    output=Path(output);script=output.with_suffix('.gdb');script.write_text('python\n'+code+'end\n')
    with output.open('x') as stream:subprocess.run(['sudo','-n','gdb','--readnever','-n','-q','-batch','-x',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True)
