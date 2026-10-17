"""Run one nontraining build check after host-local heavy admission.

The caller acquires its mutex and run-heavy scope for this command alone.
This helper never declares jobs, splits a sweep, claims work or starts a fit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cwd','justfile','target','receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['gradient','panel','clippy','normal','identity'],required=True)
    a=p.parse_args()
    own=Path('/proc/self/cgroup').read_text().strip().split('/')[-1]
    scopes=subprocess.check_output(['systemctl','--user','list-units','--type=scope','--state=running','--no-legend','--no-pager'],text=True).splitlines()
    other=[r for r in scopes if r.strip().startswith('run-') and r.split()[0]!=own]
    if other:raise RuntimeError('E36 check refuses concurrent heavy scopes: '+repr(other))
    for name in ['RUSTFLAGS','E34_OWNER_ORACLE']:os.environ.pop(name,None)
    recipes=dict(gradient='e36-tv-gradient-check',panel='e36-final-panel-test',clippy='e36-final-private-clippy',normal='e36-normal-build-cargo',identity='e36-identity-build-cargo')
    argv=['just','--justfile',str(a.justfile),'--working-directory',str(a.cwd),recipes[a.phase],str(a.target)]
    subprocess.run(argv,check=True)
    with a.receipt.open('x') as f:json.dump(dict(phase=a.phase,argv=argv,source=str(a.cwd),fits=0,optimizer_steps=0,passed=True),f,indent=2)


if __name__=='__main__':main()
