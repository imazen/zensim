"""Before/after real-table short fits for historical, strict and hd/hp/ha paths.

Compares complete ZNPR bytes after the existing strip owner removes only the
run-specific repro metadata (timestamps, binary/output paths). Keeps originals.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rev4_featpot'))
import v2_lodo_mlp as fit
import v2_common as common
from v2_common import sha, recipe_of
import e21_cheap_recipe as e21


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--new',type=Path,required=True)
    ap.add_argument('--fit-bin',type=Path,required=True);ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--inspector',type=Path,required=True)
    args=ap.parse_args();args.dest.mkdir(parents=True,exist_ok=False)
    root=Path('/var/tmp/e27/data-stage/rev4-featpot/v2e27/wide/main/real')
    common.V2=root.parents[2]
    keep=args.dest/'keep.txt';keep.write_text('\n'.join(map(str,e21.columns('by_v2fy')))+'\n')
    fit.EPOCHS,fit.PAIRS_PER_EPOCH,fit.LOG_EVERY=2,128,1
    recipe={'hidden':128}
    base=[('cid22',root/'cid22_fit.parquet',1.,0.,'withinref,both'),
          ('cid22_development',root/'cid22_dev.parquet',0.,2.,'withinref,both')]
    modes={'historical':base}
    hdr=json.loads((root/'receipt.json').read_text())['legs']['hdr']
    for token in ('hd','hp','ha'):
        group,_=fit.hdr_training_group(hdr,e21.columns('by_v2fy'),recipe_of(e21.spec('by_v2fy')+f':{token}4'))
        # Record's paths are root-relative: use the unchanged admitted table
        # in the isolated E27 stage, never a protected/VAL payload.
        group=(group[0],root/'hdr_fit.parquet',*group[2:])
        modes[token]=base+[group]
    strict_root=Path('/var/tmp/shippath2/recipe-complete/wide/main/real')
    modes['strict']=[('cid22',strict_root/'cid22_fit.parquet',1.,0.,'withinref,both'),
                     ('cid22_development',strict_root/'cid22_dev.parquet',0.,2.,'withinref,both')]
    report={}
    def stable_repro(path):
        inspected=json.loads(subprocess.check_output([str(args.inspector),'inspect',str(path)],text=True))
        obj=json.loads(next(m['value_text'] for m in inspected['metadata'] if m['key']=='zentrain.repro'))
        obj['timestamp_epoch']=0
        argv=obj['argv'];argv[0]='<trainer>';argv[argv.index('--out')+1]='<output>'
        return obj
    for mode,groups in modes.items():
        files=[];repros=[]
        for label,binary in [('before',args.baseline),('after',args.new)]:
            out=args.dest/f'{mode}-{label}.bin'
            cmd=fit.train_command(groups,1101,101,1853,keep,'N',out,recipe)
            if mode=='strict':
                # Direct strict Rust admission regression on previously declared
                # oracle TRAIN tables. This does not claim a newly qualified
                # Python recipe-admission view (the old view lacks ancestry).
                fit.strict_training_groups(groups)
                i=cmd.index('--historical-replay');del cmd[i:i+2]
            cmd[0]=str(binary)
            with (args.dest/f'{mode}-{label}.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            repros.append(stable_repro(out))
            stripped=args.dest/f'{mode}-{label}-without-repro.bin'
            with (args.dest/f'{mode}-{label}-strip.log').open('w') as log:
                subprocess.run([str(args.fit_bin),'strip','--in',str(out),'--out',str(stripped),'--key','zentrain.repro'],stdout=log,stderr=subprocess.STDOUT,check=True)
            files.append(stripped)
        if files[0].read_bytes()!=files[1].read_bytes():raise AssertionError(f'{mode} model bytes changed')
        if repros[0]!=repros[1]:raise AssertionError(f'{mode} nonvolatile reproduction metadata changed')
        report[mode]=dict(status='PASS',bytes=files[0].stat().st_size,sha256=sha(files[0]),epochs=2,pairs_per_epoch=128,
                         repro_nonvolatile_sha256=hashlib.sha256(json.dumps(repros[0],sort_keys=True).encode()).hexdigest())
        if mode in ('hd','hp','ha'):report[mode]['hdr_group']=dict(weight=groups[-1][2],mode=groups[-1][4])
        print(json.dumps({mode:report[mode]}),flush=True)
    (args.dest/'PARITY.json').write_text(json.dumps(dict(status='PASS',baseline_sha256=sha(args.baseline),new_sha256=sha(args.new),
          model_comparison='all non-repro bytes identical; repro identical after timestamp and trainer/output argv normalization',paths=report),indent=1)+'\n')


if __name__=='__main__':main()
