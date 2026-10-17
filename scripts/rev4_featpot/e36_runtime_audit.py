"""Read an owned live Runtime's numerical and resource bindings, without fitting."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def audit(root,out):
    root=Path(root);instances={}
    for top in sys._current_frames().values():
        frame=top
        while frame:
            for key in ['runtime','self']:
                value=frame.f_locals.get(key)
                if type(value).__name__=='Runtime' and getattr(value,'root',None)==root:instances[id(value)]=value
            frame=frame.f_back
    if len(instances)!=1:raise ValueError('unique owned numerical Runtime required')
    runtime=next(iter(instances.values()));folder=root/'owner-svt-ssd-v1/closure-v6'
    expected=hashlib.sha256((root/'runtime-v6/runtime.json').read_bytes()).hexdigest()
    methods={key:Path(getattr(runtime,key).__func__.__code__.co_filename) for key in ['execute','execute_cell','call']}
    put=Path(runtime.store.put.__func__.__code__.co_filename)
    import e34_encode
    disk=Path(e34_encode.e36_disk.__code__.co_filename)
    if (runtime.spec_sha!=expected or runtime._owner_svt_ssd!=5 or runtime._e36_disk_policy!=2
        or float(os.environ['E36_GENERATION_DEADLINE'])!=1792256400.
        or methods['execute']!=folder/'e36_owner_amendment.py'
        or methods['execute_cell']!=folder/'e36_disk_policy.py'
        or methods['call']!=root/'runtime-v6/program/e36_data.py'
        or put!=folder/'e36_owner_amendment.py' or disk!=folder/'e36_disk_policy.py'):
        raise ValueError('live numerical/resource/source binding differs')
    with Path(out).open('x') as stream:
        json.dump(dict(schema='e36-live-runtime-audit-v1',pid=os.getpid(),utc=time.time(),
            runtime_sha256=expected,deadline_unix=1792256400.,methods={k:str(v) for k,v in methods.items()},
            put_owner=str(put),disk_owner=str(disk),fits=0,labels_read=False,admitted=True),stream,indent=2)
