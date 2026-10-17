"""Verify the actual native program archive and run target-denied help paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from e36_store import sha

AUDIT_HELP = r'''
import os,sys,runpy
def deny_payload(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0]);parts=path.lower().split('/')
        if path.endswith('.parquet') or any(p.startswith(('_sealed','labels__')) or 'holdout' in p for p in parts):
            raise RuntimeError('E36 program-help admission refuses data payload: '+path)
sys.addaudithook(deny_payload)
script=sys.argv[1];sys.argv=[script,'--help']
runpy.run_path(script,run_name='__main__')
'''


def run(program,out):
    out.mkdir();extracted=out/'program';extracted.mkdir()
    with tarfile.open(program) as archive:
        names=[]
        for member in archive.getmembers():
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not member.isfile() or member.name in names:
                raise ValueError('unsafe/duplicate native program member')
            names.append(member.name)
        archive.extractall(extracted,filter='data')
    meta=json.loads((extracted/'build_meta.json').read_text())
    if meta['schema']!='fleet-fits-build-v1' or set(names)!=set(meta['files'])|{'build_meta.json'}:
        raise ValueError('native program inventory differs from archive')
    for name,digest in meta['files'].items():
        if sha(extracted/name)!=digest:raise ValueError('program member changed: '+name)
    required=['fit_cell_exec.py','fit_paths.py','qualified_fit_contract.py','e36_fit_contract.py',
              'harvest_fit_cells.py','bin/inspect_qualified_checkpoint','benchmarks/e36_fit_contract_2026-10-17.json']
    if not set(required)<=set(names):raise ValueError('native E36 runtime/qualification closure incomplete')
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(extracted/'scripts')+os.pathsep+str(extracted/'scripts/rev4_featpot')}
    for script in ['v2_lodo_mlp.py','v2_confirm_fit.py']:
        target=extracted/'scripts/rev4_featpot'/script
        with (out/(script+'.help.log')).open('x') as log:
            subprocess.run([sys.executable,'-c',AUDIT_HELP,str(target)],cwd=extracted,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(schema='e36-actual-program-admission-v1',program_sha256=sha(program),
        build_metadata_sha256=sha(extracted/'build_meta.json'),all_inventory_hashes_match=True,
        help_paths_with_payload_reads_denied=['v2_lodo_mlp.py','v2_confirm_fit.py'],
        labels_read=False,optimizer_steps=0,fits_released=False)
    with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['program','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.program,a.out)
