"""Explicit operator-only timeout repair oracle; never a training smoke."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_pause_budget import repair_timer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    a.out.mkdir()
    with (a.out/'child.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/timeout','3',sys.executable,'-c','import time;time.sleep(8)'],stdout=log,stderr=subprocess.STDOUT)
        time.sleep(.25);os.kill(child.pid,signal.SIGSTOP)
        (a.out/'pid').write_text(str(child.pid))
        time.sleep(3.25)
        assert 'State:\tT' in Path(f'/proc/{child.pid}/status').read_text()
        repair_timer(child.pid,time.time_ns()+20_000_000_000,a.out/'repair.log')
        os.kill(child.pid,signal.SIGCONT)
        rc=child.wait(timeout=30)
    if rc!=0:raise RuntimeError('timeout repair fixture child exit '+str(rc))
    (a.out/'receipt.json').write_text(json.dumps(dict(expired_timer_repaired=True,child_rc=rc,fits=0,optimizer_steps=0)))


if __name__=='__main__':main()
