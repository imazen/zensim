"""Cooperative archive/monitor replacement inside the owned generation scope."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


class Planned:
    def __init__(self,child,event,until):self.child=child;self.event=event;self.until=until
    @property
    def pid(self):return self.child.pid
    def poll(self):
        if time.monotonic()>=self.until:self.event.set()
        return None
    def wait(self):return self.child.wait()


def transition(root,folder,receipt,roles,children,env,scope,event,write,mask_seconds=120):
    old=dict(roles);until=time.monotonic()+mask_seconds
    proxies={role:Planned(child,event,until) for role,child in old.items()}
    for role,proxy in proxies.items():roles[role]=proxy;children[children.index(old[role])]=proxy
    history=[]
    try:
        for role in ['mirror','monitor']:
            stop=root/'owner-svt-ssd-v1'/('mirror-stop.json' if role=='mirror' else 'disk-monitor-stop.json')
            write(stop,dict(reason='cooperative budget4 resource migration',utc=time.time(),fits=0))
            while old[role].poll() is None:
                if time.monotonic()>=until:event.set()
                time.sleep(.1)
            code=old[role].wait()
            if code!=0:raise RuntimeError('old resource helper did not close normally')
            retained=folder/(role+'-previous-stop.json');stop.rename(retained)
            ready=folder/(role+'-ready.json')
            script=folder/('e36_artifact_mirror.py' if role=='mirror' else 'e36_disk_policy.py')
            with (folder/(role+'.log')).open('x') as log:
                child=subprocess.Popen(['python3',str(script)],stdout=log,stderr=subprocess.STDOUT,
                    env={**env,'PYTHONPATH':str(folder),'E36_RESOURCE_READY_PATH':str(ready),
                         'E36_GENERATION_DEADLINE':'1792274400.0','PYTHONDONTWRITEBYTECODE':'1'})
                # Own BEFORE fallible readiness or closing the parent log FD.
                children.append(child);roles[role]=child
            while not ready.exists():
                if child.poll() is not None or time.monotonic()>=until:raise RuntimeError('new resource helper failed readiness')
                time.sleep(.1)
            evidence=json.loads(ready.read_text())
            if evidence['pid']!=child.pid or evidence['role']!=role or evidence['scope']!=str(scope):raise ValueError('resource readiness owner differs')
            children.remove(proxies[role])
            record=dict(role=role,old_pid=old[role].pid,old_returncode=code,new_pid=child.pid,ready=str(ready),utc=time.time(),fits=0)
            write(folder/(role+'-pid.json'),record);history.append(record)
        write(Path(receipt),dict(schema='e36-cooperative-budget4-resources-v1',history=history,scope=str(scope),
            controller_pid=os.getpid(),deadline_unix=1792274400.,fits=0,scientific_change=False,manual_kills=0,utc=time.time()))
    except BaseException as error:
        # The existing non-daemon supervisor consumes this SAME event even
        # when every proxy has already been replaced or the old child lives.
        event.set()
        write(folder/'handoff-failed.json',dict(error=repr(error),history=history,utc=time.time(),fits=0))
        raise


def install(root,folder,receipt):
    root=Path(root);folder=Path(folder);matches=[]
    for top in sys._current_frames().values():
        frame=top
        while frame:
            if frame.f_code.co_name=='run' and Path(frame.f_code.co_filename)==root/'owner-svt-ssd-v1/continuation-control-v1/e36_generation_resume.py':matches.append(frame)
            frame=frame.f_back
    # The same frame can occur below several observer stacks; identity is key.
    matches={id(frame):frame for frame in matches}
    if len(matches)!=1:raise ValueError('unique owned native controller required')
    frame=next(iter(matches.values()));local=frame.f_locals
    roles=local['roles'];children=local['children'];native=local['native'];scope=local['scope']
    if native.poll() is not None:raise ValueError('native phase already closed')
    guarded=frame.f_globals['write']
    names=guarded.__code__.co_freevars
    if 'event' not in names:raise ValueError('existing shared non-daemon supervisor failure event required')
    event=guarded.__closure__[names.index('event')].cell_contents
    if event.is_set():raise ValueError('existing resource failure requires adjudication first')
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    def change():
        transition(root,folder,receipt,roles,children,local['env'],scope,event,write)
    thread=threading.Thread(target=change,name='e36-budget4-resource-handoff',daemon=False);thread.start()
