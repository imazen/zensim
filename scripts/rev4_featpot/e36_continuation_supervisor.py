"""Install resource supervision on the already running native continuation."""
import json
import os
from pathlib import Path
import sys
import threading
import time

import e36_generation_resume as repaired


def install(root,receipt):
    root=Path(root)
    if Path(repaired.__file__).resolve().parent!=Path(__file__).resolve().parent:
        raise ValueError('repaired supervisor import did not use its frozen sibling')
    if any(thread.name=='e36-continuation-resource-supervisor' for thread in threading.enumerate()):
        raise ValueError('resource supervisor already installed')
    matches=[]
    for top in sys._current_frames().values():
        frame=top
        while frame:
            if (frame.f_code.co_name=='run' and Path(frame.f_code.co_filename)==root/'owner-svt-ssd-v1/continuation-control-v1/e36_generation_resume.py'
                and frame.f_locals.get('native') is not None):matches.append(frame)
            frame=frame.f_back
    if len(matches)!=1:raise ValueError('unique owned live continuation required')
    frame=matches[0];local=frame.f_locals
    native=local['native'];roles=local['roles'];scope=local['scope'];worker=local['worker'];original=frame.f_globals['write']
    if native.poll() is not None:raise ValueError('native continuation already exited')
    event=threading.Event();ownership={}
    def guarded_write(path,value):
        if Path(path).name in ['mirror-stop.json','disk-monitor-stop.json'] and native.poll() is None:
            event.set()
            return
        return original(path,value)
    def shared_hold(root_arg,scope_arg,worker_arg,reason):
        return repaired.hold_admission(root_arg,scope_arg,worker_arg,reason,ownership)
    def watch():
        try:repaired.supervise(native,roles,root,scope,worker,forced_event=event,memory=ownership)
        except BaseException:
            event.set()
            repaired.drain_before_cleanup(native,roles,root,scope,worker,ownership)
        if event.is_set():
            for name in ['mirror-stop.json','disk-monitor-stop.json']:
                path=root/'owner-svt-ssd-v1'/name
                if not path.exists():original(path,dict(utc=time.time(),reason='native exited after exceptional controller cleanup',fits=0))
    frame.f_globals['write']=guarded_write
    frame.f_globals['hold_admission']=shared_hold
    thread=threading.Thread(target=watch,name='e36-continuation-resource-supervisor',daemon=False)
    thread.start()
    with Path(receipt).open('x') as stream:
        json.dump(dict(schema='e36-live-continuation-supervision-v2',pid=os.getpid(),native_timeout_pid=native.pid,
                       scope=str(scope),thread=thread.name,thread_alive=thread.is_alive(),fits=0,scientific_change=False,utc=time.time()),stream,indent=2)
