"""Registered TRAIN-only coverage x fine-chroma experiment.

This adapter declares the study and calls v2_confirm_fit's existing strict
trainer/packer. It contains no training, calibration or inference implementation.
Local cells run serially under the shared heavy lock; no fleet is orchestrated.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

import pyarrow.parquet as pq

from v2_common import derived_declaration, selection_id, sha, confirm_seeds

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
DATA = Path('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29')
PROGRAM = Path('/mnt/v/output/zensim/e33-impl-2026-10-09/packet/program.tar.gz')
LOCK = Path.home()/'tmp/zensim-paper/rev4/heavy.lock'
HEAVY = Path.home()/'work/claudehints/scripts/run-heavy'
ARMS = {'current':(0x98,'fx1'), 'coverage':(0x9e,'fx1'),
        'chroma':(0x98,'fx2'), 'both':(0x9e,'fx2')}


def write(path, value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def read(path):return json.loads(path.read_text())


def arm_record(name):
    mask,token=ARMS[name];path,pin,decl=derived_declaration(token)
    return dict(name=name,coverage_mask=mask,token=token,declaration_sha256=pin,
        declaration=str(path),columns=decl['direct'],input_count=len(decl['direct'])+len(decl['products']),
        spec=f'sel:{selection_id(decl["direct"])}@h32:H128:cv16:cf{mask:x}:{token}')


def plan(out):
    if out.exists():raise ValueError('refusing existing study root')
    out.mkdir(parents=True)
    from v2_human_role import preflight_recipe
    decision=preflight_recipe(DATA,DATA/'human_role_decision.json')
    records=[arm_record(n) for n in ARMS]
    bins=out/'bin';bins.mkdir()
    with tarfile.open(PROGRAM) as archive:
        for name in ['zensim_mlp_train','bake_dial_refit','panel','inspect_qualified_checkpoint']:
            member=archive.getmember('bin/'+name)
            with archive.extractfile(member) as src,(bins/name).open('xb') as dest:shutil.copyfileobj(src,dest)
            (bins/name).chmod(0o755)
    registration=dict(schema='color-noise-repair-registration-v1',state='frozen-before-fits',
        build_commit=subprocess.check_output(['jj','log','--no-graph','-r','@','-T','commit_id'],text=True).strip(),
        owner_sha256=sha(__file__),data_root=str(DATA),frozen_sha256=sha(DATA/'wide/frozen.json'),
        role_decision_sha256=sha(DATA/'human_role_decision.json'),role_sources=decision['sources'],
        program_path=str(PROGRAM),program_sha256=sha(PROGRAM),binaries={n:sha(bins/n) for n in [p.name for p in bins.iterdir()]},
        arms=records,seeds=[dict(index=i,init=confirm_seeds(i)[0],sample=confirm_seeds(i)[1]) for i in range(3)],
        epochs=120,pairs_per_epoch=50000,selected_epoch=119,hidden=128,head='N',tier='v3',
        selection=dict(population='existing TRAIN-development source-disjoint human splits only',seeds=[0,1,2],
            statistic='canonical signed SROCC',source_equal_mean_delta_min=-0.002,per_source_mean_delta_min=-0.005,
            noise='KADID types11,12,13,14 and TID types1,2,3,4,5,6,7; equal-source signed SROCC delta must be positive',
            choice='Eligible arm with largest equal-source noise delta, tie by human mean delta, then registration order; seed0 serves, seeds1/2 never chosen. No eligible repair retains current.',
            status='Exploratory three-seed screening only; original ten-seed E33 and product gates remain required before promotion'),
        loss='Existing within-source rank-only coverage ordinal ladders; no new loss or hyperparameter search. Other group weights, final epoch, packing and TRAIN calibration unchanged.',
        assessment='Freeze TRAIN-selected arm before any external scores. External CSIQ/PIPAL assessment; LIVE is exposed targeted regression only and never fit/selection data. No adaptive reruns.',
        exclusions=['protected P/HR/HC/HH/H0/H1','all sealed-directory data','KADID terminal/T0','all AIC','CIQA jpg2_pref','CLIC answers','all LIVE fit/selection'],
        scope='Exploratory repair of frozen C128. Reuses the existing finite sparse ordinal grid; no newly dense sweep or shipping qualification is claimed.',
        time_utc=datetime.now(timezone.utc).isoformat())
    write(out/'REGISTRATION.json',registration)
    print('Registered',len(records),'arms x3seeds; binaries from exact E33 program.',flush=True)


def registration(out):
    d=read(out/'REGISTRATION.json')
    if d['state']!='frozen-before-fits' or d['arms']!=[arm_record(n) for n in ARMS]:raise ValueError('registered arms changed')
    if sha(DATA/'wide/frozen.json')!=d['frozen_sha256'] or sha(DATA/'human_role_decision.json')!=d['role_decision_sha256']:raise ValueError('training root changed')
    for name,pin in d['binaries'].items():
        if sha(out/'bin'/name)!=pin:raise ValueError('native binary changed')
    return d


def admit(out):
    d=registration(out)
    from v2_common import load_frozen,refuse_nonfinite_kept
    from v2_human_role import preflight_recipe
    preflight_recipe(DATA,DATA/'human_role_decision.json')
    load_frozen(DATA,training_only=True)
    receipt=read(DATA/'wide/main/real/receipt.json');paths=[]
    for name in ['safesyn','cid22','human_all']:
        paths.extend(DATA/receipt['legs'][name][split]['rel'] for split in ['fit','dev'])
    paths.append(DATA/'e15/coverage_pool.parquet')
    decl=derived_declaration('fx2')[2];ids=sorted(set(decl['direct'])|{b for _,b in decl['products']})
    refuse_nonfinite_kept(paths,ids)
    pool_keys=pq.read_table(DATA/'e15/coverage_pool.keys.parquet')
    families=pool_keys['family'].to_pylist()
    counts={f:families.count(f) for f in sorted(set(families))}
    payload=dict(schema='color-noise-repair-admission-v1',registration_sha256=sha(out/'REGISTRATION.json'),
        complete=True,rows={str(p):pq.ParquetFile(p).metadata.num_rows for p in paths},
        feature_ids=ids,finite_feature_count=len(ids),pool_families=counts,
        inputs={str(p):dict(table_sha256=sha(p),manifest_sha256=sha(Path(str(p)+'.manifest.json'))) for p in paths})
    write(out/'ADMISSION.json',payload)
    print('Admission PASS;504 inputs finite on all TRAIN/development legs. Pool families:',counts,flush=True)


def command(out,arm,seed,smoke=False):
    a=arm_record(arm);dest=out/('smokes' if smoke else 'cells')/arm/f's{seed}'
    cmd=[sys.executable,str(REPO/'scripts/rev4_featpot/v2_confirm_fit.py'),
        '--root',str(DATA),'--spec',a['spec'],'--head','N','--seed-index',str(seed),
        '--columns',','.join(map(str,a['columns'])),'--strict-admission','--train-only',
        '--data-role-decision',str(DATA/'human_role_decision.json'),'--dest',str(dest)]
    cmd+=['--local-smoke-budget','1:49999'] if smoke else ['--pack-production','--e33-output-stage']
    return cmd,dest


def fit(out,arm,seed,smoke=False):
    registration(out)
    admission=read(out/'ADMISSION.json')
    if not admission['complete'] or admission['registration_sha256']!=sha(out/'REGISTRATION.json'):raise ValueError('admission missing or changed')
    if shutil.disk_usage('/home').free<45*1024**3:raise ValueError('large-write admission requires45GiB free on /home')
    cmd,dest=command(out,arm,seed,smoke)
    if dest.exists():raise ValueError('refusing existing fit output; inspect it before recovery')
    dest.parent.mkdir(parents=True,exist_ok=True)
    scratch=out/'scratch'/arm/f's{seed}'
    scratch.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'TMPDIR':str(scratch),'PYTHONPATH':str(REPO/'scripts')+os.pathsep+str(REPO/'scripts/rev4_featpot'),'REV4_V2_BIN_DIR':str(out/'bin'),
         'ZENSIM_MAX_TIER':'v3','RAYON_NUM_THREADS':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    print(datetime.now(timezone.utc).isoformat(),'START',arm,seed,'smoke' if smoke else 'registered fit',flush=True)
    with (dest.parent/(f's{seed}.smoke.run.log' if smoke else f's{seed}.run.log')).open('x') as log:
        subprocess.run(['flock',str(LOCK),str(HEAVY),'--mem','16G','--jobs','1','--',*cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(dest/'result.json')
    if result['selection']['selected_epoch']!=(0 if smoke else 119) or result['execution_contract']!=('local-smoke' if smoke else 'registered-fit'):raise ValueError('fit budget/selection differs')
    print(datetime.now(timezone.utc).isoformat(),'COMPLETE',arm,seed,flush=True)


def run(out,smoke=False):
    for seed in ([0] if smoke else range(3)):
        for arm in ARMS:fit(out,arm,seed,smoke)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','admit','smokes','fits'])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    {'plan':plan,'admit':admit,'smokes':lambda r:run(r,True),'fits':run}[a.mode](a.out)


if __name__=='__main__':main()
