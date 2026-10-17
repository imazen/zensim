"""Validate an explicitly authorized frozen-time extension; preserve original clocks.

This is a budget adapter only. It neither schedules jobs nor changes any view,
numerical producer, source, quality, target, loss or selection rule.
"""
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unix_ns(value):
    return int(Decimal(str(value))*1_000_000_000)


def resolve(path):
    """Return the immutable base clock and the exact authorized effective clock."""
    path=Path(path);clock=json.loads(path.read_text())
    if 'budget_only_amendment' not in clock:return clock,clock,[]
    pin=clock['budget_only_amendment'];ap=Path(pin['path'])
    if sha(ap)!=pin['sha256']:raise ValueError('budget amendment hash differs')
    a=json.loads(ap.read_text());parent=a['original_clock'];pp=Path(parent['path'])
    if sha(pp)!=parent['sha256']:raise ValueError('budget original clock changed')
    base=json.loads(pp.read_text())
    if 'budget_only_amendment' in base:raise ValueError('stacked budget amendments require a new registration')
    evidence=a['freeze_evidence'];ep=Path(evidence['path'])
    if sha(ep)!=evidence['sha256']:raise ValueError('freeze evidence changed')
    events=json.loads(ep.read_text())
    if (a['schema']!='e36-authorized-frozen-budget-v1' or a['authorization']!='coordinator-2026-10-17-frozen-time-exclusion'
            or a['scientific_change'] is not False or a['fits']!=0
            or a['resume_unix_ns']<=a['freeze_unix_ns']
            or a['excluded_frozen_ns']!=a['resume_unix_ns']-a['freeze_unix_ns']
            or events['freeze_unix_ns']!=a['freeze_unix_ns'] or events['resume_unix_ns']!=a['resume_unix_ns']
            or a['original_deadline_unix_ns']!=unix_ns(base['deadline_unix'])
            or a['effective_deadline_unix_ns']!=a['original_deadline_unix_ns']+a['excluded_frozen_ns']
            or clock['deadline_unix']!=a['effective_deadline_unix_ns']/1_000_000_000
            or {k:v for k,v in clock.items() if k not in ['deadline_unix','budget_only_amendment']}!={k:v for k,v in base.items() if k!='deadline_unix'}):
        raise ValueError('exact authorized frozen-time budget chain differs')
    proofs=[dict(path=str(ap),sha256=sha(ap)),dict(path=str(ep),sha256=sha(ep)),parent]
    if any(k in a for k in ['launcher_before','launcher_after','live_budget_helper']):
        for key in ['launcher_before','launcher_after','live_budget_helper']:
            pin=a[key]
            if sha(pin['path'])!=pin['sha256']:raise ValueError('budget launcher/helper changed')
            proofs.append(pin)
        original=Path(a['launcher_before']['path']).read_bytes().splitlines(keepends=True)
        expected=b''.join(original[:-1])+('export E36_GENERATION_DEADLINE='+repr(str(clock['deadline_unix']))+'\n').encode()+original[-1]
        if len(original)!=3 or not original[-1].startswith(b'exec ') or Path(a['launcher_after']['path']).read_bytes()!=expected:
            raise ValueError('budget launcher changes more than the admitted deadline')
    return base,clock,proofs


def live_python(clock_path,output):
    """Called by an attached debugger on a stopped E36 Python main thread.

    CPython3.14 frame-local proxies write through. The caller must verify the
    exact PID/cgroup and retain its GDB log. No function or numerical code is
    replaced: only currently live deadline values and the controller's clock
    path change. Already-created native timeout timers are repaired separately.
    """
    import os
    import sys
    base,clock,_=resolve(clock_path)
    delta=clock['deadline_unix']-base['deadline_unix'];records=[];recognized=False
    for thread,top in sys._current_frames().items():
        f=top
        while f:
            name=f.f_code.co_name;file=Path(f.f_code.co_filename)
            if file.name=='e36_data.py' and name=='main' and 'runtime' in f.f_locals:recognized=True
            if file.name in ['e36_bulk.py','e36_data.py'] and name in ['bulk','run','execute','execute_cell','call','oracle']:
                recognized=True
                local=f.f_locals
                if isinstance(local.get('deadline'),(int,float)):
                    old=local['deadline'];local['deadline']=clock['deadline_unix'] if old==base['deadline_unix'] else old+delta
                    records.append(dict(thread=thread,function=name,file=str(file),old_deadline=old,new_deadline=local['deadline']))
                if name=='bulk':
                    local['clock']=Path(clock_path);local['times']=clock
            f=f.f_back
    if not recognized:raise ValueError('no live E36 controller/executor frame found')
    os.environ['E36_GENERATION_DEADLINE']=str(clock['deadline_unix'])
    p=Path(output)
    with p.open('x') as out:json.dump(dict(schema='e36-live-budget-frame-receipt-v1',pid=os.getpid(),clock_path=str(clock_path),clock_sha256=sha(clock_path),records=records,optimizer_steps=0),out,indent=2)


def repair_python(pid,clock_path,output):
    """Apply the tested frame operation through GDB, retaining command and receipt."""
    import subprocess
    output=Path(output)
    if output.exists():raise ValueError('live Python budget receipt already exists; never apply twice')
    module=Path(__file__).resolve()
    code=("import importlib.util; s=importlib.util.spec_from_file_location('e36_live_budget',"+repr(str(module))+
          "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.live_python("+repr(str(clock_path))+','+repr(str(output))+")")
    script=output.with_suffix('.gdb')
    with script.open('x') as f:f.write('''set confirm off
set unwind-on-signal on
handle SIGALRM nostop noprint nopass
attach '''+str(pid)+'''
handle SIGSTOP nostop noprint nopass
set $gil=(int)PyGILState_Ensure()
set $rc=(int)PyRun_SimpleString('''+json.dumps(code)+''')
call (void)PyGILState_Release($gil)
python
if int(gdb.parse_and_eval('$rc'))!=0:raise RuntimeError('live Python budget operation failed')
end
detach
''')
    with output.with_suffix('.gdb.log').open('x') as log:
        subprocess.run(['sudo','-n','gdb','-q','-n','-batch','-x',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
    if not output.exists():raise ValueError('live Python budget operation emitted no receipt')


def repair_timer(pid,deadline_unix_ns,output):
    """Reset only kernel POSIX timer0 of an identified, stopped timeout monitor.

    Linux x86_64 timer_settime/timer_gettime are read from installed headers by
    the operator before this tool is used. GDB suppresses an already-pending
    SIGALRM while setting the replacement timer. No signal kills the inferior.
    """
    import subprocess
    output=Path(output)
    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if command[0]!=b'/usr/bin/timeout':raise ValueError('timer repair accepts only exact timeout PIDs')
    script=output.with_suffix('.gdb')
    code='''import gdb,struct,time,json
gdb.execute('set confirm off')
gdb.execute('set unwind-on-signal on')
gdb.execute('handle SIGALRM nostop noprint nopass')
gdb.execute('attach PID')
gdb.execute('handle SIGSTOP nostop noprint nopass')
pointer=int(gdb.parse_and_eval('(void*)malloc(32)'))
remaining_ns=DEADLINE-time.time_ns()
if remaining_ns<=0:raise RuntimeError('effective native deadline already expired')
expiration=time.monotonic_ns()+remaining_ns
sec,nsec=divmod(expiration,1000000000)
gdb.selected_inferior().write_memory(pointer,struct.pack('=qqqq',0,0,sec,nsec))
rc=int(gdb.parse_and_eval('(long)syscall(223L,0L,1L,(void*)%d,0L)'%pointer))
if rc!=0:raise RuntimeError('timer_settime refused: '+str(rc))
rc=int(gdb.parse_and_eval('(long)syscall(224L,0L,(void*)%d)'%pointer))
if rc!=0:raise RuntimeError('timer_gettime refused')
got=struct.unpack('=qqqq',bytes(gdb.selected_inferior().read_memory(pointer,32)))
print(json.dumps(dict(pid=PID,deadline_unix_ns=DEADLINE,timer_remaining_seconds=got[2]+got[3]/1e9,repaired=True)))
gdb.execute('call (void)free((void*)%d)'%pointer)
gdb.execute('detach')
'''.replace('PID',str(pid)).replace('DEADLINE',str(deadline_unix_ns))
    with script.open('x') as f:f.write('python\n'+code+'end\n')
    with output.open('x') as f:subprocess.run(['sudo','-n','gdb','-q','-n','-batch','-x',str(script)],stdout=f,stderr=subprocess.STDOUT,check=True)


def resume(root,scope,supervisor,controller,auto_thaw_shell,registration,recovery=None):
    """Resume the identified owned scope after an authorized, audited timer repair."""
    import datetime
    import os
    import signal
    import subprocess
    import time
    root=Path(root);out=root/('budget-amendment-v2' if recovery else 'budget-amendment-v1');out.mkdir()
    supervisor_argv=[x.decode() for x in Path(f'/proc/{supervisor}/cmdline').read_bytes().split(b'\0') if x]
    if Path(supervisor_argv[0]).name!='flock' or str(root) not in supervisor_argv:raise ValueError('unexpected original supervisor')
    paper_lock=supervisor_argv[1]
    def owner():
        result=json.loads(subprocess.check_output(['lslocks','--json','-o','PID,PATH,MODE'],text=True))
        return any(r.get('pid')==supervisor and r.get('mode')=='WRITE' and r.get('path')==paper_lock for r in result['locks'])
    if not owner():raise ValueError('original E36 supervisor does not own the paper lock')
    if not recovery and subprocess.check_output(['systemctl','--user','show',scope,'-p','FreezerState','--value'],text=True).strip()!='frozen':
        raise ValueError('budget repair requires the original scope still frozen')
    cgline=Path(f'/proc/{controller}/cgroup').read_text().strip().split('::')[1]
    if not cgline.endswith('/'+scope):raise ValueError('controller scope differs')
    cg=Path('/sys/fs/cgroup'+cgline);records={}
    if recovery:
        recovery=Path(recovery)
        if list(recovery.glob('*.gdb')) or (recovery/'receipt.json').exists():raise ValueError('recovery requires failed preapplication admission, never partly applied frames')
        records={int(k):v for k,v in json.loads((recovery/'processes-before.json').read_text()).items()}
    else:
        for text in (cg/'cgroup.procs').read_text().split():
            pid=int(text);p=Path('/proc')/text
            cmd=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            status=(p/'status').read_text();parent=int(next(s for s in status.splitlines() if s.startswith('PPid:')).split()[1])
            records[pid]=dict(pid=pid,parent=parent,argv=cmd)
    python_pids=[p for p,r in records.items() if len(r['argv'])>1 and Path(r['argv'][1]).name in ['e36_bulk.py','e36_data.py']]
    timers=[p for p,r in records.items() if r['argv'][0]=='/usr/bin/timeout']
    workers=[p for p,r in records.items() if Path(r['argv'][0]).name=='zenfleet-worker']
    if controller not in python_pids or len(workers)!=1 or not timers:raise ValueError('unexpected live generation inventory')
    with (out/'processes-before.json').open('x') as f:json.dump(records,f,indent=2)
    for pid in python_pids+timers+workers:
        if recovery:
            if 'State:\tT' not in Path(f'/proc/{pid}/status').read_text():raise ValueError('preapplication process no longer stopped')
        else:os.kill(pid,signal.SIGSTOP)
    if not owner():raise ValueError('original generation lock lost before thaw')
    if recovery:
        previous=json.loads((recovery/'freeze-evidence.json').read_text());freeze=previous['freeze'];thaw=previous['thaw']
    else:
        subprocess.run(['systemctl','--user','thaw',scope],check=True)
        journal=[json.loads(s) for s in subprocess.check_output(['journalctl','--user','-u',scope,'--since','2026-10-17 07:30:00 UTC','--no-pager','-o','json'],text=True).splitlines()]
        freeze=[r for r in journal if r['MESSAGE']==scope+': Unit now frozen.'][-1]
        thaw=[r for r in journal if r['MESSAGE']==scope+': Unit now thawed.'][-1]
    freeze_ns=int(freeze.get('_SOURCE_REALTIME_TIMESTAMP',freeze['__REALTIME_TIMESTAMP']))*1000
    resume_ns=int(thaw.get('_SOURCE_REALTIME_TIMESTAMP',thaw['__REALTIME_TIMESTAMP']))*1000
    if resume_ns<=freeze_ns:raise ValueError('journal freeze/thaw ordering differs')
    evidence=out/'freeze-evidence.json'
    with evidence.open('x') as f:json.dump(dict(freeze_unix_ns=freeze_ns,resume_unix_ns=resume_ns,freeze=freeze,thaw=thaw),f,indent=2)
    original=root/'bulk-clock-v6.json';base=json.loads(original.read_text());old_deadline_ns=unix_ns(base['deadline_unix']);delta=resume_ns-freeze_ns
    clock_path=root/('bulk-clock-v6-budget2.json' if recovery else 'bulk-clock-v6-budget1.json');ap=out/'amendment.json'
    launcher=root/'runtime-v6/executor';old_launcher=launcher.with_name('executor.before-budget1')
    original_launcher=(old_launcher if recovery else launcher).read_bytes();old_launcher_sha=sha(old_launcher if recovery else launcher)
    newline=('export E36_GENERATION_DEADLINE='+repr(str((old_deadline_ns+delta)/1_000_000_000))+'\n').encode()
    lines=original_launcher.splitlines(keepends=True)
    if len(lines)!=3 or not lines[-1].startswith(b'exec '):raise ValueError('frozen executor launcher shape differs')
    new_launcher=b''.join(lines[:-1])+newline+lines[-1]
    launcher.rename(launcher.with_name('executor.failed-budget1') if recovery else old_launcher)
    with launcher.open('xb') as f:f.write(new_launcher)
    launcher.chmod(0o755)
    amendment=dict(schema='e36-authorized-frozen-budget-v1',authorization='coordinator-2026-10-17-frozen-time-exclusion',scientific_change=False,fits=0,
        original_clock=dict(path=str(original),sha256=sha(original)),freeze_evidence=dict(path=str(evidence),sha256=sha(evidence)),
        freeze_unix_ns=freeze_ns,resume_unix_ns=resume_ns,excluded_frozen_ns=delta,original_deadline_unix_ns=old_deadline_ns,effective_deadline_unix_ns=old_deadline_ns+delta,
        live_budget_helper=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        launcher_before=dict(path=str(old_launcher),sha256=old_launcher_sha),launcher_after=dict(path=str(launcher),sha256=sha(launcher)))
    with ap.open('x') as f:json.dump(amendment,f,indent=2)
    effective={**base,'deadline_unix':(old_deadline_ns+delta)/1_000_000_000,'budget_only_amendment':dict(path=str(ap),sha256=sha(ap))}
    with clock_path.open('x') as f:json.dump(effective,f,indent=2)
    resolve(clock_path)
    def utc(ns):return datetime.datetime.fromtimestamp(ns/1e9,datetime.timezone.utc).isoformat()
    with Path(registration).open('a') as f:f.write('\nExact budget-only amendment: journal freeze '+utc(freeze_ns)+', actual native-generation thaw '+utc(resume_ns)+
        '; excluded '+str(delta)+'ns ('+str(Decimal(delta)/Decimal(1_000_000_000))+'s). Original deadline '+utc(old_deadline_ns)+
        '; effective deadline '+utc(old_deadline_ns+delta)+'. Amendment SHA256 '+sha(ap)+'. No scientific change; no fit has run.\n')
    receipts={}
    for pid in python_pids:
        path=out/(str(pid)+'.json');repair_python(pid,clock_path,path);receipts[pid]=json.loads(path.read_text())
    for pid in timers:
        parent=records[pid]['parent'];visited=set()
        while parent not in receipts:
            if parent in visited or parent not in records:raise ValueError('native timer has no identified executor/controller ancestor')
            visited.add(parent);parent=records[parent]['parent']
        deadlines=[r['new_deadline'] for r in receipts[parent]['records'] if r['function'] in ['bulk','execute']]
        if len(deadlines)!=1:raise ValueError('native timer deadline ancestor is ambiguous')
        repair_timer(pid,unix_ns(deadlines[0]),out/(str(pid)+'.timer.log'))
        os.kill(pid,signal.SIGCONT)
    for pid in python_pids+workers:
        if Path(f'/proc/{pid}').exists():os.kill(pid,signal.SIGCONT)
    # Complete the retained old helper without killing it. Its stale legacy
    # log wording is explicitly superseded by the immutable amendment below.
    os.kill(auto_thaw_shell,signal.SIGCONT)
    for _ in range(50):
        if not Path(f'/proc/{auto_thaw_shell}').exists():break
        time.sleep(.02)
    result=dict(schema='e36-live-budget-resume-v1',generation_resumed=True,clock_path=str(clock_path),clock_sha256=sha(clock_path),amendment_sha256=sha(ap),
        python_receipts=len(receipts),native_timers_repaired=len(timers),processes_killed=False,fits=0,optimizer_steps=0)
    with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
    with (root/'LOCK_WINDOWS.jsonl').open('a') as f:f.write(json.dumps({**result,'event':'cooperative-window-budget-amended','freeze_utc':utc(freeze_ns),'resume_utc':utc(resume_ns),'effective_deadline_utc':utc(old_deadline_ns+delta)})+'\n')
    print(json.dumps({**result,'freeze_utc':utc(freeze_ns),'resume_utc':utc(resume_ns),'excluded_seconds':str(Decimal(delta)/Decimal(1e9)),'deadline_utc':utc(old_deadline_ns+delta)}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','registration']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scope',required=True);p.add_argument('--recover-preapplication',type=Path)
    for name in ['supervisor','controller','auto-thaw-shell']:p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args();resume(a.root,a.scope,a.supervisor,a.controller,a.auto_thaw_shell,a.registration,a.recover_preapplication)
