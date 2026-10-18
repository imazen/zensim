"""Transport the registered TRAIN decision and frozen external comparison.

Training, prediction, rank/scatter statistics and HTML rendering delegate to
their existing native owners. This file only binds populations and decisions.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(REPO/'scripts/v_next'))
BASE = Path('/mnt/v/output/zensim/gauntlet-2026-10-18')
BANK = Path('/var/tmp/rev5-featbank')
SOURCES = ['kadid', 'tid2013', 'konfig', 'cid22_a25']
MEMBER_SOURCE = {'kadid_train':'kadid','kadid_select':'kadid','tid2013':'tid2013',
                 'konfig_train':'konfig','konfig_val':'konfig','cid22_a25':'cid22_a25'}


def now():return datetime.now(timezone.utc).isoformat()


def prepare(out):
    from color_noise_repair import DATA, registration, read, write, sha
    registration(out)
    receipt=read(DATA/'wide/main/real/receipt.json')
    record=receipt['legs']['human_all']['dev'];table=DATA/record['rel']
    keys=table.with_suffix('.keys.parquet');manifest=Path(str(table)+'.manifest.json')
    declaration=read(manifest)
    if sha(table)!=record['sha256'] or sha(manifest)!=record['manifest_sha256'] or sha(keys)!=declaration['keys_sha256']:
        raise ValueError('TRAIN-development identity changed')
    rows=pq.read_table(keys).to_pylist();codecs={};pins={}
    for member in MEMBER_SOURCE:
        root=BANK/member;m=read(root/'_MANIFEST.json')
        if sha(root/'_MANIFEST.json')!=declaration['bank_manifest_sha256'][member] or sha(root/'keys.parquet')!=m['keys_sha256']:
            raise ValueError('TRAIN member key identity changed')
        codecs[member]={r['pair_key']:r['codec'] for r in pq.read_table(root/'keys.parquet',columns=['pair_key','codec']).to_pylist()}
        pins[member]=dict(keys_sha256=sha(root/'keys.parquet'),manifest_sha256=sha(root/'_MANIFEST.json'))
    for r in rows:
        r['source']=MEMBER_SOURCE[r['member_set']]
        r['codec']=codecs[r['member_set']][r['pair_key']]
    populations={s:[i for i,r in enumerate(rows) if r['source']==s] for s in SOURCES}
    for s,types in [('kadid',range(11,15)),('tid2013',range(1,8))]:
        populations[s+'_noise']=[i for i,r in enumerate(rows) if r['source']==s and r['codec'] in {('kadid' if s=='kadid' else 'tid')+f'_{t:02d}' for t in types}]
    if sum(len(populations[s]) for s in SOURCES)!=len(rows) or any(len(v)<3 for v in populations.values()):
        raise ValueError('missing TRAIN-development source/noise population')
    write(out/'DEVELOPMENT_POPULATIONS.json',dict(schema='color-noise-development-populations-v1',
        registration_sha256=sha(out/'REGISTRATION.json'),table=str(table),table_sha256=sha(table),
        keys_sha256=sha(keys),manifest_sha256=sha(manifest),bank_keys=pins,rows=rows,
        populations=populations,time_utc=now()))
    print('TRAIN-development populations:',{k:len(v) for k,v in populations.items()},flush=True)


def choose(rows):
    """Apply the registered exploratory rule to canonical per-cell statistics."""
    from color_noise_repair import ARMS
    by={(r['arm'],r['seed'],r['population']):r['statistics']['srocc_signed'] for r in rows}
    decisions=[]
    for arm in list(ARMS)[1:]:
        per_source={s:float(np.mean([by[arm,k,s]-by['current',k,s] for k in range(3)])) for s in SOURCES}
        mean=float(np.mean(list(per_source.values())))
        noise=float(np.mean([by[arm,k,s+'_noise']-by['current',k,s+'_noise'] for k in range(3) for s in ['kadid','tid2013']]))
        decisions.append(dict(arm=arm,source_mean_delta=per_source,human_mean_delta=mean,
            noise_mean_delta=noise,eligible=mean>=-.002 and min(per_source.values())>=-.005 and noise>0))
    eligible=[r for r in decisions if r['eligible']]
    selected=max(eligible,key=lambda r:(r['noise_mean_delta'],r['human_mean_delta'],-list(ARMS).index(r['arm'])))['arm'] if eligible else 'current'
    return selected,decisions


def select(out):
    from color_noise_repair import ARMS, registration, read, write, sha
    registration(out)
    p=read(out/'DEVELOPMENT_POPULATIONS.json')
    if p['registration_sha256']!=sha(out/'REGISTRATION.json') or sha(p['table'])!=p['table_sha256']:
        raise ValueError('development inputs changed')
    from v2_lodo_mlp import predict
    from lib.zen_stats import panel_batch
    target=pq.read_table(p['table'],columns=['human_score'])['human_score'].to_numpy()
    jobs=[];records=[];models={}
    for seed in range(3):
        for arm in ARMS:
            cell=out/'cells'/arm/f's{seed}';r=read(cell/'result.json')
            model=Path(r['packed_model'])
            if (r['execution_contract']!='registered-fit' or r['selection']['selected_epoch']!=119
                    or r['epochs']!=120 or r['pairs_per_epoch']!=50000 or sha(model)!=r['packed_model_sha256']):
                raise ValueError('registered full fit/packing missing: '+str(cell))
            dest=out/'development'/arm/f's{seed}';dest.mkdir(parents=True,exist_ok=False)
            pred=predict(model,Path(p['table']),dest/'pred.tsv')
            if len(pred)!=len(target):raise ValueError('development prediction coverage differs')
            models[f'{arm}/s{seed}']=dict(path=str(model),sha256=sha(model),result_sha256=sha(cell/'result.json'))
            for name,indices in p['populations'].items():
                jobs.append((f'{arm}/s{seed}/{name}',pred[indices],target[indices]))
                records.append(dict(arm=arm,seed=seed,population=name,rows=len(indices)))
    stats=panel_batch(jobs)
    for r,s in zip(records,stats,strict=True):
        if s['n_dropped'] or not np.isfinite(s['srocc_signed']):raise ValueError('incomplete canonical statistics')
        r['statistics']=s
    selected,decisions=choose(records)
    write(out/'TRAIN_SELECTION.json',dict(schema='color-noise-train-choice-v1',state='frozen-before-external-read',
        registration_sha256=sha(out/'REGISTRATION.json'),population_sha256=sha(out/'DEVELOPMENT_POPULATIONS.json'),
        selected=selected,assessment_seed=0,decisions=decisions,models=models,statistics=records,
        selection_inputs='TRAIN-development only; no external/LIVE input',qualification=False,time_utc=now()))
    print('Frozen TRAIN decision:',selected,decisions,flush=True)


def external(out):
    from color_noise_repair import ARMS, registration, read, write, sha
    from gauntlet_frozen import PREDICT
    registration(out);selection=read(out/'TRAIN_SELECTION.json')
    if selection['state']!='frozen-before-external-read' or selection['registration_sha256']!=sha(out/'REGISTRATION.json'):
        raise ValueError('frozen TRAIN decision required before external read')
    source=BASE/'WITHHELD_INPUTS.json';original=read(source)
    doc=dict(schema='zensim-frozen-gauntlet-v1',models=copy.deepcopy(original['models']),
        datasets={s:copy.deepcopy(original['datasets'][s]) for s in ['csiq','pipal','live']},
        references=original['references'],notes=[
            'Exploratory fixed 2x2 color/noise repair. Three matched seeds selected on TRAIN-development only; assessment seed0 is fixed. Selected arm: '+selection['selected']+'. No product qualification.',
            'CSIQ/PIPAL and LIVE were previously exposed in the original gauntlet. LIVE motivated the diagnosis and is an exposed regression diagnostic, never fitting or selection data. This is not fresh independent holdout confirmation.',
            'The source-pixel overlap screen found no exact matches in these admitted populations. LIVE retains the existing10-family subset. Other crop/resize lineage is not comprehensively cleared. Native feature/binary identities remain pinned; external extractor source-commit closure remains unresolved.',
            'All statistics and scatter diagnostics use the canonical native owners. SROCC intervals:10000 source-cluster draws; small-source intervals descriptive. PLCC/KROCC intervals unmeasured. Every population point is retained in JSON.',
            'The separate TRAIN-development decision and all three seeds are recorded in TRAIN_SELECTION.json. Original frozen comparators and peers reuse their exact original vectors.'])
    exposures=dict(schema='frozen-public-test-exposure-v1',secret_holdouts_accessed=False,
        populations=['csiq','pipal','live'],prior_exposure=str(BASE),source_sha256=sha(source),
        choice_sha256=sha(out/'TRAIN_SELECTION.json'),models=selection['models'],time_utc=now(),
        use='Frozen assessment after TRAIN-only choice; no adaptive tuning or independent-holdout claim')
    write(out/'EXTERNAL_EXPOSURE.json',exposures)
    work=out/'external';work.mkdir()
    for arm in ARMS:
        name='repair-'+arm;r=selection['models'][arm+'/s0'];model=Path(r['path'])
        if sha(model)!=r['sha256']:raise ValueError('frozen model changed')
        doc['models'].append(dict(name=name,bake=str(model),bake_sha256=r['sha256'],regime='TRAIN-only 2x2; seed0',
            model=dict(kind='mlp',note=arm+'; '+('TRAIN-selected' if arm==selection['selected'] else 'fixed comparison arm'))))
        for corpus,data in doc['datasets'].items():
            bank=Path(data['feature_bank']);manifest=read(bank/'_MANIFEST.json')
            if sha(bank/'features.parquet')!=manifest['features_parquet_sha256'] or sha(bank/'keys.parquet')!=manifest['keys_sha256']:
                raise ValueError('external feature bank changed')
            wire=BASE/'scoring'/f'{corpus}.features-f64.bin'
            inventory=read(BASE/'ARCHIVE_FILES.json')
            expected=next(r['sha256'] for r in inventory['files'] if r['relative']==f'scoring/{corpus}.features-f64.bin')
            if sha(wire)!=expected:raise ValueError('original feature wire changed')
            dest=work/f'{corpus}-{arm}.pred.txt'
            with dest.open('x') as output,(work/f'{corpus}-{arm}.log').open('x') as log:
                subprocess.run([str(PREDICT),'--bake',str(model),'--features-file',str(wire),'--f64-wire','--production'],
                    check=True,stdout=output,stderr=log,env={**os.environ,'ZENSIM_FORMULA_REV':'5'})
            pred=[float(x) for x in dest.read_text().splitlines()]
            if len(pred)!=len(data['target']) or not np.isfinite(pred).all():raise ValueError('external prediction population differs')
            data['scores'][name]=dict(pred=pred,status='external-screened',
                note='Frozen TRAIN-only repair; exact same original gauntlet population. LIVE is exposed diagnostic only.',
                provenance=dict(model_sha256=sha(model),feature_manifest_sha256=sha(bank/'_MANIFEST.json'),
                    wire_sha256=sha(wire),predictor_sha256=sha(PREDICT),selection_sha256=sha(out/'TRAIN_SELECTION.json')))
            print(name,corpus,len(pred),flush=True)
    write(out/'EXTERNAL_INPUTS.json',doc)


def dashboard(out):
    from build_peer_fullevals import build_comparison
    build_comparison(out/'EXTERNAL_INPUTS.json',out)
    subprocess.run([sys.executable,str(REPO/'scripts/v_next/gauntlet.py'),'--fulleval-dir',str(out),
        '--comparison-only','--include-all-scatter','--out',str(out/'gauntlet.html')],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','select','external','dashboard'])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.environ['REV4_V2_BIN_DIR']=str(a.out/'bin')
    os.environ['ZEN_PANEL_BIN']=str(BASE/'bin/panel')
    {'prepare':prepare,'select':select,'external':external,'dashboard':dashboard}[a.mode](a.out)


if __name__=='__main__':main()
