"""Real shared-lock interleaving while owned storage futures drain."""
from concurrent.futures import Future
import fcntl
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_archive_windows import Windows
from e36_pack_archive import Dispatcher


class WindowsTests(unittest.TestCase):
    def test_two_submissions_then_release_only_when_both_finish(self):
        class Transfer:
            def __init__(self):self.pending={};self.calls=[]
            def cycle(self,q,p,submit=True):
                self.calls.append(submit)
                for f in list(self.pending):
                    if f.done():f.result();del self.pending[f]
                if submit:
                    f=Future();self.pending[f]={}
                return []
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d,patch('e36_archive_windows.time.time',return_value=0):
            path=Path(d)/'lock';transfer=Transfer();window=Windows(Path(d),transfer,path)
            with path.open('a') as contender:
                window.cycle({},[1]);first=next(iter(transfer.pending));window.cycle({},[1]);self.assertEqual(len(transfer.pending),2)
                first.set_result(None);window.cycle({},[1]);self.assertFalse(transfer.calls[-1]);self.assertEqual(len(transfer.pending),1)
                with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                next(iter(transfer.pending)).set_result(None);window.cycle({},[1]);self.assertFalse(window.held)
                fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(contender,fcntl.LOCK_UN)
            window.close()

    def test_failed_first_transfer_retains_lock_until_second_is_joined(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d,patch('e36_archive_windows.time.time',return_value=0):
            path=Path(d)/'lock';transfer=Dispatcher.__new__(Dispatcher);first=Future();second=Future();first.set_exception(ValueError('first failed'));transfer.pending={first:dict(items=[]),second:dict(items=[])};window=Windows(Path(d),transfer,path)
            errors=[];entered=threading.Event()
            def drain():
                entered.set()
                try:window.drain({})
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=drain);thread.start();entered.wait(timeout=2)
            with path.open('a') as contender:
                with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                second.set_result([]);thread.join(timeout=2);self.assertFalse(thread.is_alive());self.assertEqual(len(errors),1)
                fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(contender,fcntl.LOCK_UN)
            window.close()


if __name__=='__main__':unittest.main()

class InterruptedWait(unittest.TestCase):
    def test_interrupted_wait_keeps_real_future_and_lock_until_terminal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d,patch('e36_archive_windows.time.time',return_value=0):
            path=Path(d)/'lock';transfer=Dispatcher.__new__(Dispatcher);first=Future();second=Future();first.set_result([]);transfer.pending={first:dict(items=[]),second:dict(items=[])};window=Windows(Path(d),transfer,path);interrupted=threading.Event();original=second.result;calls=[]
            def result(*a,**kw):
                calls.append(1)
                if len(calls)==1:interrupted.set();raise KeyboardInterrupt('waiting thread interrupted')
                return original(*a,**kw)
            second.result=result;errors=[]
            def drain():
                try:window.drain({})
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=drain);thread.start();self.assertTrue(interrupted.wait(timeout=2))
            with path.open('a') as contender:
                with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertIn(second,transfer.pending);second.set_result([]);thread.join(timeout=2);self.assertFalse(thread.is_alive());self.assertIsInstance(errors[0],KeyboardInterrupt)
                fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(contender,fcntl.LOCK_UN)
            window.close()
