"""One manifest-bound, label-free R probe through fit-cell-exec; no fitting."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from v2_common import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--dest',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--model-sha256',required=True)
    p.add_argument('--family',required=True,choices=['a','c'])
    p.add_argument('--wall-cap-sec',required=True,type=int)
    a=p.parse_args()
    if not 0<a.wall_cap_sec<=900 or sha(a.model)!=a.model_sha256:
        raise ValueError('bounded probe/model binding invalid')
    if (a.dest/'result.json').exists():
        raise FileExistsError('fresh probe required')
    (a.dest/'refit').mkdir(parents=True,exist_ok=True)
    model=a.dest/'refit/control.bin'
    shutil.copyfile(a.model,model)
    out=a.dest/'exact-probe.json'
    binary=Path(__file__).resolve().parents[2]/'bin/steer_exact_smoke'
    env=dict(os.environ,ZENSIM_FORMULA_REV='5',ZENSIM_STEER_EXACT='all-blocks-v1')
    subprocess.run([str(binary),'--bake',str(model),'--model-sha256',a.model_sha256,'--out',str(out)],env=env,check=True)
    (a.dest/'result.json').write_text(json.dumps(dict(schema='steer-exact-smoke-cell-v1',execution_contract='local-smoke',
        family=a.family,selected_bake=str(model),selected_bake_sha256=sha(model),probe_sha256=sha(out),
        epochs=0,scope='LABEL_FREE_ENGINEERING_ONLY'))+'\n')

if __name__=='__main__':
    main()
