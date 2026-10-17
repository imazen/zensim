"""Defer parent cancellation while an exact owned child drains; never kill it."""
import signal
import subprocess
import sys
from pathlib import Path


class DeferredCancellation:
    def __enter__(self):
        self.received=[];self.handlers={s:signal.getsignal(s) for s in [signal.SIGINT,signal.SIGTERM]}
        self.mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(self.handlers))
        for value in self.handlers:signal.signal(value,lambda signum,frame:self.received.append(signum))
        signal.pthread_sigmask(signal.SIG_SETMASK,self.mask)
        return self

    def spawn(self,command,**kwargs):
        signal.pthread_sigmask(signal.SIG_BLOCK,set(self.handlers))
        # Restore in an exec shim rather than Python preexec_fn after fork;
        # numerical/framework imports may already own background threads.
        shim=str(Path(__file__).with_name('e36_unmask_exec.py'))
        wrapped=[sys.executable,shim,','.join(str(int(v)) for v in self.mask),*command]
        return subprocess.Popen(wrapped,**kwargs)

    def installed(self):
        # Call only after the returned Popen handle belongs to outer cleanup.
        signal.pthread_sigmask(signal.SIG_SETMASK,self.mask)

    def __exit__(self,kind,value,traceback):
        # The caller must have joined its exact child before leaving this scope.
        signal.pthread_sigmask(signal.SIG_SETMASK,self.mask)
        for sig,handler in self.handlers.items():signal.signal(sig,handler)
        if kind is None and self.received:raise InterruptedError('E36 cancellation deferred until owned child terminal: '+str(self.received))
        return False
