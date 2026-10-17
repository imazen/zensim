"""Apply the frozen authorized clock to the exact owned native process tree."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from e36_budget4 import DEADLINE,inject_code,repair_timer


def apply(root,controller,folder,out):
    root=Path(root);folder=Path(folder);out=Path(out);out.mkdir()
    scope=Path('/sys/fs/cgroup'+Path(f'/proc/{controller}/cgroup').read_text().strip().split('::')[1])
    if scope.name!='run-p4053120-i288944556.scope':raise ValueError('exact owned E36 scope required')
    fixture=Path('/home/lilith/tmp/zensim-paper/rev4/e36-uutils-timer-fixture-v4/receipt.json')
    if json.loads(fixture.read_text())['passed'] is not True:raise ValueError('actual same-binary fixture required')
    records={}
    for value in (scope/'cgroup.procs').read_text().split():
        path=Path('/proc')/value
        try:
            args=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            records[int(value)]=dict(argv=args,parent=int(fields[1]),start_ticks=fields[19],state=fields[0])
        except FileNotFoundError:continue
    executors=[pid for pid,v in records.items() if len(v['argv'])>1 and v['argv'][1]==str(root/'runtime-v6/program/e36_data.py')]
    workers=[pid for pid,v in records.items() if v['argv'][0]==str(root/'bin/zenfleet-worker-fair-v7')]
    if len(executors)!=24 or workers!=[4053352]:raise ValueError('exact24 executor/owned worker inventory required')
    (out/'processes.json').write_text(json.dumps(records,indent=2)+'\n')
    for pid in executors:
        result=out/('executor-'+str(pid)+'.json')
        code='import importlib.util; s=importlib.util.spec_from_file_location("e36_budget4_live",'+repr(str(folder/'e36_budget4.py'))+'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.live('+repr(str(result))+')'
        inject_code(pid,code,out/('executor-'+str(pid)+'.log'))
        print('deadline and resolver applied',pid,flush=True)
    worker=workers[0]
    # --readnever avoids loading unrelated native debug data. The exported C
    # environment is the existing native executor inheritance contract.
    script=out/'worker-environment.gdb'
    script.write_text('set confirm off\nset debuginfod enabled off\nattach '+str(worker)+
        '\nhandle SIGSTOP nostop noprint nopass\nset $a=(int)setenv("PYTHONPATH",'+json.dumps(str(folder))+',1)'+
        '\nset $b=(int)setenv("E36_GENERATION_DEADLINE","1792274400.0",1)\npython\nassert int(gdb.parse_and_eval("$a"))==0 and int(gdb.parse_and_eval("$b"))==0\nend\ndetach\n')
    try:
        with (out/'worker-environment.log').open('x') as stream:subprocess.run(['sudo','-n','gdb','--readnever','-n','-q','-batch','-x',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    finally:
        if records[worker]['state']=='T':os.kill(worker,signal.SIGSTOP)
    timers=[]
    for pid,v in records.items():
        if v['argv'][0]!='/usr/bin/timeout' or not Path(f'/proc/{pid}').exists():continue
        if pid==4053351:bound=DEADLINE
        else:
            ancestor=v['parent'];visited=set()
            while ancestor not in executors:
                if ancestor not in records or ancestor in visited:raise ValueError('timeout has no owned executor ancestor')
                visited.add(ancestor);ancestor=records[ancestor]['parent']
            receipt=json.loads((out/('executor-'+str(ancestor)+'.json')).read_text())
            values=[r['deadline'] for r in receipt['records'] if r['function']=='execute']
            if len(values)!=1:raise ValueError('exact owning per-view deadline required')
            bound=values[0]
        try:repair_timer(pid,bound,out/('timer-'+str(pid)+'.log'))
        except FileNotFoundError:
            if Path(f'/proc/{pid}').exists():raise
            continue
        timers.append(dict(pid=pid,deadline=bound))
    result=dict(schema='e36-live-budget4-applied-v1',scope=str(scope),controller=controller,executors=executors,
        worker=worker,timers=timers,deadline=DEADLINE,scientific_change=False,fits=0,manual_kills=0,utc=time.time())
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','folder','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--controller',type=int,required=True)
    a=parser.parse_args();print(json.dumps(apply(a.root,a.controller,a.folder,a.out)))
