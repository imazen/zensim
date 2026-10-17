"""Release the shared paper lock between bounded storage-only pack batches."""
import fcntl
from pathlib import Path
import time


class Windows:
    def __init__(self,root,dispatcher,path):
        self.dispatcher=dispatcher;self.file=Path(path).open('a');self.held=False;self.submitted=0;self.after=0
        # Initial native-DONE/CAS inventory is itself bounded heavy work.
        while not self.acquire():time.sleep(.2)

    @property
    def pending(self):return self.dispatcher.pending

    def acquire(self):
        if self.held:return True
        if time.time()>=1792274400.:raise TimeoutError('registered deadline expired waiting for archive lock')
        if time.monotonic()<self.after:return False
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        self.held=True;self.submitted=0;return True

    def release(self):
        if self.pending:raise ValueError('cannot release paper lock with live archive transfers')
        if self.held:fcntl.flock(self.file,fcntl.LOCK_UN);self.held=False;self.after=time.monotonic()+.2

    def cycle(self,queue,priority):
        if not self.acquire():return []
        before=set(self.pending)
        rows=self.dispatcher.cycle(queue,priority,submit=self.submitted<2)
        self.submitted+=len(set(self.pending)-before)
        if not self.pending:self.release()
        return rows

    def drain(self,queue):
        try:self.dispatcher.drain(queue)
        finally:
            # The underlying drain joins every owned future even after one
            # fails. A live transfer prevents the lock from being dropped.
            self.release()

    def close(self):
        self.release();self.file.close()
