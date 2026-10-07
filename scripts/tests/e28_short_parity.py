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


def normalized_control_repro(repro):
    """Remove only run/build locations; preserve recipe, input SHA and sampler facts."""
    obj=json.loads(json.dumps(repro))
    for key in ('timestamp_epoch','cwd','hostname','trainer_head_at_train','trainer_source_dir'):
        obj[key]='<run-specific>'
    argv=obj['argv'];argv[0]='<trainer>'
    for flag in ('--out','--keep-features','--dump-checkpoints-dir'):
        if flag in argv:argv[argv.index(flag)+1]=f'<{flag}>'
    group=0
    for i,token in enumerate(argv):
        if token=='--group':
            fields=argv[i+1].split(':')
            fields[1]=f'<input-{group}>'
            argv[i+1]=':'.join(fields);group+=1
    for i,entry in enumerate(obj['inputs']):entry['path']=f'<input-{i}>'
    for i,entry in enumerate(obj['table_admission']['tables']):
        old=entry['path'];entry['path']=f'<input-{i}>'
        entry['source']=entry['source'].replace(old,f'<input-{i}>')
    return obj


def full_control(args):
    """Run one registered full-budget cell; compare its fixed final epoch to E30."""
    if os.environ.get('RAYON_NUM_THREADS')!='1' or os.environ.get('ZENSIM_MAX_TIER')!='v3':
        raise ValueError('full control parity requires one Rayon thread and v3 tier')
    control=args.e30_control
    result=json.loads((control/'result.json').read_text())
    freeze=args.control_freeze
    if sha(freeze)!='304f67a5d0b7bc78778b9deec731c213825b6a3811671c4fc3bdbf699978a07f':
        raise ValueError('E30 control freeze changed')
    name=f"{result['spec']}__N/without_{result['heldout']}_s{result['seed_index']}"
    pin=json.loads(freeze.read_text())['cells'][name]
    if (sha(control/'result.json')!=pin['result_sha256']
            or sha(control/'refit/last.bin')!=pin['selected_bake_sha256']
            or sha(control/'keep_features.txt')!=pin['feature_list_sha256']
            or result['head']!='N' or result['epochs']!=120 or result['pairs_per_epoch']!=50000
            or result['selection']['selected_epoch']!=119):
        raise ValueError('E30 baseline cell identity/budget changed')
    args.dest.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,REV4_V2_BIN_DIR=str(args.new.parent),OPENBLAS_NUM_THREADS='1')
    cell=args.dest/'cell'
    argv=[sys.executable,str(REPO/'scripts/rev4_featpot/v2_lodo_mlp.py'),
          '--root',str(args.prepared_root),'--spec',result['spec'],'--head','N',
          '--heldout',result['heldout'],'--seed-index',str(result['seed_index']),
          '--columns',','.join(map(str,e21.columns('by_v2fy'))),'--strict-admission','--train-only',
          '--data-role-decision',str(args.prepared_root/'human_role_decision.json'),'--dest',str(cell)]
    with (args.dest/'cell.log').open('w') as log:
        subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    new_result=json.loads((cell/'result.json').read_text())
    if new_result['selection']['selected_epoch']!=119:
        raise AssertionError('extended cell did not select fixed final epoch 119')
    files=[];repros=[]
    for label,source in [('e30',control/'refit/last.bin'),('extended',cell/'refit/last.bin')]:
        decoded=json.loads(subprocess.check_output([str(args.inspector),str(source)],text=True))
        repros.append(normalized_control_repro(decoded['repro']))
        stripped=args.dest/f'{label}-without-repro.bin'
        with (args.dest/f'{label}-strip.log').open('w') as log:
            subprocess.run([str(args.fit_bin),'strip','--in',str(source),'--out',str(stripped),
                            '--key','zentrain.repro'],stdout=log,stderr=subprocess.STDOUT,check=True)
        files.append(stripped)
    identical=files[0].read_bytes()==files[1].read_bytes()
    metadata_same=repros[0]==repros[1]
    report={'schema':'registered-control-full-budget-parity-v1','status':'PASS' if identical and metadata_same else 'MISMATCH',
            'cell':name,'epochs':120,'pairs_per_epoch':50000,'selected_epoch':119,'rayon_threads':1,'tier':'v3',
            'frozen_e30_control_sha256':sha(freeze),'e30_bake_sha256':sha(control/'refit/last.bin'),
            'extended_bake_sha256':sha(cell/'refit/last.bin'),'extended_trainer_sha256':sha(args.new),
            'all_non_repro_model_bytes_identical':identical,'nonvolatile_repro_identical':metadata_same,
            'e30_non_repro_sha256':sha(files[0]),'extended_non_repro_sha256':sha(files[1]),
            'model_comparison':'complete model bytes after canonical strip of zentrain.repro only',
            'repro_normalization':'timestamp/cwd/host/source build identity and argv/input locations only; all SHA, recipe, sampler and admission facts retained',
            'control_decision':'fresh v40 matched control remains the registered comparison even on mismatch'}
    for label,repro in zip(('e30','extended'),repros):
        (args.dest/f'{label}-normalized-repro.json').write_text(json.dumps(repro,indent=2)+'\n')
    (args.dest/'PARITY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not identical or not metadata_same:raise AssertionError('registered baseline parity mismatch; recorded, never substitute E30 for the fresh control')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path);ap.add_argument('--new',type=Path,required=True)
    ap.add_argument('--fit-bin',type=Path,required=True);ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--inspector',type=Path,required=True)
    ap.add_argument('--e30-control',type=Path);ap.add_argument('--prepared-root',type=Path)
    ap.add_argument('--control-freeze',type=Path)
    args=ap.parse_args()
    if args.e30_control:
        if not args.prepared_root or not args.control_freeze:ap.error('full control requires --prepared-root and --control-freeze')
        full_control(args);return
    if not args.baseline:ap.error('short before/after parity requires --baseline')
    args.dest.mkdir(parents=True,exist_ok=False)
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
