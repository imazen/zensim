"""Before/after real-table short fits for historical, strict and hd/hp/ha paths.

Compares complete ZNPR bytes after the existing strip owner removes only the
run-specific repro metadata (timestamps, binary/output paths). Keeps originals.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rev4_featpot'))
import v2_lodo_mlp as fit
from v2_common import sha
import e21_cheap_recipe as e21


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--new',type=Path,required=True)
    ap.add_argument('--fit-bin',type=Path,required=True);ap.add_argument('--dest',type=Path,required=True)
    args=ap.parse_args();args.dest.mkdir(parents=True,exist_ok=False)
    root=Path('/var/tmp/e27/data-stage/rev4-featpot/v2e27/wide/main/real')
    keep=args.dest/'keep.txt';keep.write_text('\n'.join(map(str,e21.columns('by_v2fy')))+'\n')
    fit.EPOCHS,fit.PAIRS_PER_EPOCH,fit.LOG_EVERY=2,128,1
    recipe={'hidden':128}
    base=[('cid22',root/'cid22_fit.parquet',1.,0.,'withinref,both'),
          ('cid22_development',root/'cid22_dev.parquet',0.,2.,'withinref,both')]
    modes={'historical':base,
           'hd':base+[('hdr',root/'hdr_fit.parquet',4.,0.,'withinref,rank')],
           'hp':base+[('hdr',root/'hdr_fit.parquet',4.,0.,'rank')],
           'ha':base+[('hdr',root/'hdr_fit.parquet',4.,0.,'withinref,both')]}
    strict_root=Path('/var/tmp/shippath2/recipe-complete/wide/main/real')
    modes['strict']=[('cid22',strict_root/'cid22_fit.parquet',1.,0.,'withinref,both'),
                     ('cid22_development',strict_root/'cid22_dev.parquet',0.,2.,'withinref,both')]
    report={}
    for mode,groups in modes.items():
        files=[]
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
            stripped=args.dest/f'{mode}-{label}-without-repro.bin'
            with (args.dest/f'{mode}-{label}-strip.log').open('w') as log:
                subprocess.run([str(args.fit_bin),'strip','--in',str(out),'--out',str(stripped),'--key','zentrain.repro'],stdout=log,stderr=subprocess.STDOUT,check=True)
            files.append(stripped)
        if files[0].read_bytes()!=files[1].read_bytes():raise AssertionError(f'{mode} model bytes changed')
        report[mode]=dict(status='PASS',bytes=files[0].stat().st_size,sha256=sha(files[0]),epochs=2,pairs_per_epoch=128)
        print(json.dumps({mode:report[mode]}),flush=True)
    (args.dest/'PARITY.json').write_text(json.dumps(dict(status='PASS',baseline_sha256=sha(args.baseline),new_sha256=sha(args.new),
          model_comparison='all bytes after canonical removal of timestamp/path-bearing zentrain.repro only',paths=report),indent=1)+'\n')


if __name__=='__main__':main()
