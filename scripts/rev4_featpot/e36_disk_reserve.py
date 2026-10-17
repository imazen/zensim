"""Wait for E36 archive headroom before another native command; no clock credit."""
import shutil
import time
import types


def install(runtime):
    if getattr(runtime,'_e36_disk_wait',False):return
    original=runtime.call
    def call(owner,argv,log,deadline,*args,**kwargs):
        while shutil.disk_usage(owner.root).free<45*1024**3:
            if time.time()>=deadline:
                raise TimeoutError('registered clock expired while preserving SSD reserve')
            time.sleep(1)
        return original(argv,log,deadline,*args,**kwargs)
    runtime.call=types.MethodType(call,runtime)
    runtime._e36_disk_wait=True
