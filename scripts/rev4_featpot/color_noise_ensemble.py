"""Prospective fixed-seed ensemble supplement; all scoring stays in Rust."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import color_noise_assessment as assessment
import numpy as np


def production_lines(values):
    """Match predict_features_with_bake --production's Rust {score:.17} wire."""
    return [format(float(v),'.17f') for v in values]


def plan(out):
    from color_noise_repair import ARMS,registration,write,sha
    registration(out)
    if (out/'EXTERNAL_EXPOSURE.json').exists():raise ValueError('supplement must precede external reads')
    root=out/'ensembles';root.mkdir()
    write(root/'REGISTRATION.json',dict(schema='color-noise-ensemble-registration-v1',
        original_registration_sha256=sha(out/'REGISTRATION.json'),
        state='frozen-before-ensemble-scoring-and-external-read',arms=list(ARMS),seeds=[0,1,2],
        composition='Equal-weight mean of all three packed serving models, after each member output stage; native --ensemble --score-units. No seed or weight search.',
        selection='TRAIN-development only. Paired against current ensemble: human equal-source delta >= -0.002, every source >= -0.005, noise equal-source delta >0. Among eligible repairs choose greatest human delta, then noise delta, then arm order; otherwise retain current ensemble.',
        assessment=assessment.EXTERNAL,qualification=False,
        scope='Separate prospective exploratory supplement for owner-requested overall SROCC; original single-model study/selection/gates unchanged. All four ensembles assessed frozen; no adaptive reruns.',
        time_utc=assessment.now()))


def predict(out,arm,table,dest,seeds=range(3)):
    from color_noise_repair import read,sha
    from v2_common import dense_bake,FITBIN
    if FITBIN.resolve()!=(out/'bin/bake_dial_refit').resolve():raise ValueError('ensemble densifier is not the pinned native owner')
    choice=read(out/'TRAIN_SELECTION.json')
    members=[]
    for seed in seeds:
        binding=choice['models'][f'{arm}/s{seed}'];model=Path(binding['path'])
        if sha(model)!=binding['sha256']:raise ValueError('ensemble member changed')
        members.append(dense_bake(model,dest.parent))
    with dest.with_suffix('.log').open('x') as log:
        subprocess.run([str(out/'bin/bake_dial_refit'),'predict','--ensemble',','.join(map(str,members)),
            '--corpus',str(table),'--score-units','--out',str(dest)],
            stdout=log,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'ZENSIM_FORMULA_REV':'5'})
    import pandas as pd
    frame=pd.read_csv(dest,sep='\t',float_precision='round_trip')
    if list(frame.columns)!=['row_idx','pred'] or not np.array_equal(frame.row_idx,np.arange(len(frame))):
        raise ValueError('ensemble row order differs')
    values=frame.pred.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('nonfinite ensemble prediction')
    return values


def choose(records):
    from color_noise_repair import ARMS
    by={(r['arm'],r['population']):r['statistics']['srocc_signed'] for r in records}
    decisions=[]
    for arm in list(ARMS)[1:]:
        per={s:by[arm,s]-by['current',s] for s in assessment.SOURCES}
        human=float(np.mean(list(per.values())))
        noise=float(np.mean([by[arm,s+'_noise']-by['current',s+'_noise'] for s in ['kadid','tid2013']]))
        decisions.append(dict(arm=arm,human_mean_delta=human,source_mean_delta=per,noise_mean_delta=noise,
            eligible=human>=-.002 and min(per.values())>=-.005 and noise>0))
    eligible=[r for r in decisions if r['eligible']]
    winner=max(eligible,key=lambda r:(r['human_mean_delta'],r['noise_mean_delta'],-list(ARMS).index(r['arm'])))['arm'] if eligible else 'current'
    return winner,decisions


def score(out):
    from color_noise_repair import ARMS,registration,read,write,sha
    import pandas as pd
    import pyarrow.parquet as pq
    from lib.zen_stats import panel_batch
    registration(out);root=out/'ensembles';reg=read(root/'REGISTRATION.json')
    if reg['original_registration_sha256']!=sha(out/'REGISTRATION.json') or reg['arms']!=list(ARMS) or reg['seeds']!=[0,1,2]:
        raise ValueError('ensemble registration differs')
    if (out/'EXTERNAL_EXPOSURE.json').exists():raise ValueError('ensemble choice must precede external reads')
    original=read(out/'TRAIN_SELECTION.json');population=read(out/'DEVELOPMENT_POPULATIONS.json')
    if sha(population['table'])!=population['table_sha256']:raise ValueError('development table changed')
    target=pq.read_table(population['table'],columns=['human_score'])['human_score'].to_numpy()
    records=[];models={};parity=[]
    for arm in ARMS:
        dest=root/arm;dest.mkdir()
        members=[original['models'][f'{arm}/s{s}'] for s in range(3)]
        composition=dest/'composition.json'
        write(composition,dict(schema='native-score-ensemble-v1',members=members,weights=None,
            native_surface='zensim::BakeScorer::ensemble(models, None)',formula_revision=5,qualification=False))
        models[arm]=dict(path=str(composition),sha256=sha(composition),members=members)
        pred=predict(out,arm,Path(population['table']),dest/'pred.tsv')
        individual=[pd.read_csv(out/'development'/arm/f's{s}/pred.tsv',sep='\t',float_precision='round_trip').pred.to_numpy() for s in range(3)]
        oracle=np.mean(individual,axis=0)
        if len(pred)!=len(target):raise ValueError('ensemble development population differs')
        difference=float(np.max(np.abs(pred-oracle)))
        if difference>8*np.spacing(max(1.,float(np.max(np.abs(oracle))))):
            raise ValueError('native ensemble differs from independent member-score arithmetic')
        parity.append(dict(arm=arm,max_abs_difference=difference,rows=len(pred)))
        jobs=[(name,pred[idx],target[idx]) for name,idx in population['populations'].items()]
        for (name,_,_),st in zip(jobs,panel_batch(jobs),strict=True):
            if st['n_dropped'] or not np.isfinite(st['srocc_signed']):raise ValueError('ensemble statistics incomplete')
            records.append(dict(arm=arm,population=name,statistics=st))
    selected,decisions=choose(records)
    write(root/'TRAIN_SELECTION.json',dict(schema='color-noise-ensemble-choice-v1',state='frozen-before-external-read',
        registration_sha256=sha(root/'REGISTRATION.json'),parent_choice_sha256=sha(out/'TRAIN_SELECTION.json'),
        selected=selected,decisions=decisions,models=models,statistics=records,independent_arithmetic_check=parity,
        qualification=False,time_utc=assessment.now()))
    print('Frozen ensemble TRAIN decision:',selected,decisions,flush=True)


def external(out):
    from color_noise_repair import ARMS,registration,read,write,sha
    registration(out);root=out/'ensembles';choice=read(root/'TRAIN_SELECTION.json')
    reg=read(root/'REGISTRATION.json')
    if (choice['state']!='frozen-before-external-read' or choice['registration_sha256']!=sha(root/'REGISTRATION.json')
            or choice['parent_choice_sha256']!=sha(out/'TRAIN_SELECTION.json') or reg['assessment']!=assessment.EXTERNAL):
        raise ValueError('frozen ensemble choice changed')
    data=copy.deepcopy(read(out/'EXTERNAL_INPUTS.json'))
    data['models']=[m for m in data['models'] if not m['name'].startswith('repair-')]
    data['notes']=['Fixed equal-weight three-seed ensemble supplement. All member models and composition selected on TRAIN-development before external read. No product qualification.',
        'All external populations were previously exposed. LIVE motivated the investigation and is diagnostic only. All original exclusions and overlap/source-commit limits remain.']
    write(root/'EXTERNAL_EXPOSURE.json',dict(schema='frozen-public-test-exposure-v1',secret_holdouts_accessed=False,
        populations=assessment.EXTERNAL,choice_sha256=sha(root/'TRAIN_SELECTION.json'),
        original_exposure_sha256=sha(out/'EXTERNAL_EXPOSURE.json'),time_utc=assessment.now()))
    work=root/'external';work.mkdir()
    # The retained Parquet prediction CLI requires a target column even though
    # inference never reads its values. Keep real external labels out of this
    # compatibility view; all statistical truth remains in the frozen input.
    import pyarrow as pa
    import pyarrow.parquet as pq
    views={};receipts=[]
    for corpus,d in data['datasets'].items():
        if shutil.disk_usage('/home').free<45*1024**3:raise ValueError('prediction view requires45GiB free')
        bank=Path(d['feature_bank']);manifest=read(bank/'_MANIFEST.json')
        source=bank/'features.parquet'
        if sha(source)!=manifest['features_parquet_sha256'] or sha(bank/'keys.parquet')!=manifest['keys_sha256']:
            raise ValueError('external ensemble bank changed')
        table=pq.read_table(source,columns=[f'f{i}' for i in range(720)])
        if table.num_rows!=len(d['target']):raise ValueError('external view row population differs')
        table=table.append_column('human_score',pa.array(np.zeros(table.num_rows),type=pa.float64()))
        metadata=dict(table.schema.metadata or {})
        metadata[b'prediction_view_role']=b'Inference only; human_score is an unused constant-zero loader placeholder. Not training or statistical truth.'
        table=table.replace_schema_metadata(metadata)
        view=work/f'{corpus}.features.parquet';pq.write_table(table,view,compression='zstd')
        views[corpus]=view
        control=predict(out,'current',view,work/f'{corpus}-view-control.tsv',seeds=[0])
        if production_lines(control)!=(out/'external'/f'{corpus}-current.pred.txt').read_text().splitlines():
            raise ValueError('native compatibility-view control differs from original frozen scores')
        receipts.append(dict(corpus=corpus,path=str(view),sha256=sha(view),source_sha256=sha(source),
            feature_manifest_sha256=sha(bank/'_MANIFEST.json'),rows=table.num_rows,feature_ids=list(range(720)),
            original_control_vector_exact=True,role=metadata[b'prediction_view_role'].decode()))
    write(root/'PREDICTION_VIEWS.json',dict(schema='ensemble-prediction-views-v1',views=receipts))
    for arm in ARMS:
        binding=choice['models'][arm]
        if sha(binding['path'])!=binding['sha256']:raise ValueError('composition changed')
        name='repair-ensemble-'+arm
        data['models'].append(dict(name=name,bake=binding['path'],bake_sha256=binding['sha256'],
            regime='Three fixed seeds; native equal-weight score ensemble',model=dict(kind='ensemble',note='TRAIN-selected' if arm==choice['selected'] else 'fixed comparison ensemble')))
        for corpus,d in data['datasets'].items():
            bank=Path(d['feature_bank'])
            pred=predict(out,arm,views[corpus],work/f'{corpus}-{arm}.tsv')
            if len(pred)!=len(d['target']):raise ValueError('external ensemble population differs')
            d['scores'][name]=dict(pred=[float(v) for v in production_lines(pred)],status='external-screened',note='Prospectively frozen native ensemble; exposed assessment only.',
                provenance=dict(composition_sha256=binding['sha256'],feature_manifest_sha256=sha(bank/'_MANIFEST.json'),
                    prediction_view_sha256=sha(views[corpus]),selection_sha256=sha(root/'TRAIN_SELECTION.json'),
                    serialization='Original production CLI fixed17 decimal places; native round-trip score TSV retained.'))
            print(name,corpus,len(pred),flush=True)
    write(root/'EXTERNAL_INPUTS.json',data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','score','external']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    os.environ['REV4_V2_BIN_DIR']=str(a.out/'bin')
    os.environ['ZEN_PANEL_BIN']=str(assessment.BASE/'bin/panel')
    from color_noise_repair import sha,read
    if sha(assessment.BASE/'bin/panel')!=read(assessment.BASE/'_MANIFEST.json')['panel_binary_sha256']:
        raise ValueError('canonical panel changed')
    {'plan':plan,'score':score,'external':external}[a.mode](a.out)


if __name__=='__main__':main()
