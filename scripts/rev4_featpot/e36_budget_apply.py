"""Owned E36 live resource amendment; retain every process and debugger receipt."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
CONTROL=ROOT/'control-budget3'
sys.path.insert(0,str(CONTROL))
from e36_budget_extend import inject,repair_timer,DEADLINE


def main():
    pid=2672159
    cg=Path('/sys/fs/cgroup'+Path(f'/proc/{pid}/cgroup').read_text().strip().split('::')[1])
    scope=cg.name
    if scope!='run-p2672154-i275014310.scope':raise ValueError('owned scope differs')
    out=ROOT/'budget-amendment-v3/live';out.mkdir()
    records={};stopped=[]
    subprocess.run(['systemctl','--user','freeze',scope],check=True)
    try:
        for s in (cg/'cgroup.procs').read_text().split():
            p=int(s);d=Path('/proc')/s
            argv=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x]
            parent=int(next(x for x in (d/'status').read_text().splitlines() if x.startswith('PPid:')).split()[1])
            records[p]=dict(argv=argv,parent=parent)
        (out/'processes.json').write_text(json.dumps(records,indent=2))
        for p in records:
            os.kill(p,signal.SIGSTOP);stopped.append(p)
        subprocess.run(['systemctl','--user','thaw',scope],check=True)
        py=[p for p,r in records.items() if len(r['argv'])>1 and Path(r['argv'][1]).name in ['e36_resume_schedule.py','e36_data.py']]
        workers=[p for p,r in records.items() if r['argv'][0]==str(ROOT/'bin/zenfleet-worker-fair-v7')]
        if pid not in py or len(workers)!=1:raise ValueError('owned Python/worker inventory differs')
        for p in py:
            inject(p,out/f'python-{p}.json')
            print('python patched',p,flush=True)
        # Future persistent executors inherit the resource bootstrap. Numerical
        # launcher bytes remain unchanged, including their earlier budget proof.
        worker=workers[0];code='set confirm off\nset unwind-on-signal on\nattach '+str(worker)+'\nhandle SIGSTOP nostop noprint nopass\nset $rc=(int)setenv("PYTHONPATH",'+json.dumps(str(CONTROL))+',1)\npython\nif int(gdb.parse_and_eval("$rc"))!=0:raise RuntimeError("setenv failed")\nend\ndetach\n'
        script=out/'worker-environment.gdb';script.write_text(code)
        with (out/'worker-environment.log').open('x') as log:
            subprocess.run(['sudo','-n','gdb','-n','-q','-batch','-x',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
        for p,r in records.items():
            if r['argv'][0]!='/usr/bin/timeout':continue
            ancestor=r['parent'];owner=None
            while ancestor in records:
                if ancestor in py:owner=ancestor;break
                ancestor=records[ancestor]['parent']
            if owner is None:raise ValueError('timeout has no owned Python ancestor')
            receipt=json.loads((out/f'python-{owner}.json').read_text())
            bounds=[v['deadline'] for v in receipt['records'] if 'deadline' in v]
            target=min(bounds) if bounds else DEADLINE
            repair_timer(p,target,out/f'timer-{p}.log')
            print('timer patched',p,target,flush=True)
        (out/'receipt.json').write_text(json.dumps(dict(deadline=DEADLINE,python_pids=py,worker=worker,closed_unix_ns=time.time_ns(),fits=0,scientific_change=False),indent=2))
    finally:
        subprocess.run(['systemctl','--user','thaw',scope],check=True)
        for p in stopped:
            try:os.kill(p,signal.SIGCONT)
            except ProcessLookupError:pass
        (out/'resumed.json').write_text(json.dumps(dict(unix_ns=time.time_ns(),stopped_pids=stopped,manual_kills=0)))


if __name__=='__main__':main()
