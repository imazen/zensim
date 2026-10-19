"""Transport the registered TRAIN decision and frozen external comparison.

Training, prediction, rank/scatter statistics and HTML rendering delegate to
their existing native owners. This file only binds populations and decisions.
"""
import argparse
import copy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# The shared statistical owners use Python temporary files. Bind their scratch
# before importing array/Parquet libraries that can cache tempfile's default.
SCRATCH = Path.home()/'tmp'
SCRATCH.mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
tempfile.tempdir = str(SCRATCH)

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(REPO/'scripts/v_next'))
BASE = Path('/mnt/v/output/zensim/gauntlet-2026-10-18')
BANK = Path('/var/tmp/rev5-featbank')
SOURCES = ['kadid', 'tid2013', 'konfig', 'cid22_a25']
EXTERNAL = ['csiq', 'pipal', 'cid22_b', 'konjnd_jpeg_select', 'live']
MEMBER_SOURCE = {'kadid_train':'kadid','kadid_select':'kadid','tid2013':'tid2013',
                 'konfig_train':'konfig','konfig_val':'konfig','cid22_a25':'cid22_a25'}


def now():return datetime.now(timezone.utc).isoformat()


def auxiliary_overlap(out):
    """Independent RGB source-identity audit of the entire auxiliary pool."""
    import hashlib
    from PIL import Image
    from color_noise_repair import DATA,registration,read,write,sha
    registration(out)
    keys=DATA/'e15/coverage_pool.keys.parquet'
    wanted=set(pq.read_table(keys,columns=['source_filename'])['source_filename'].to_pylist())
    selection=Path('/var/tmp/rev5-featpot/v2c/e15/selection.parquet')
    paths={}
    for row in pq.read_table(selection,columns=['source_filename','ref_path']).to_pylist():
        if row['source_filename'] not in wanted:continue
        previous=paths.setdefault(row['source_filename'],row['ref_path'])
        if previous!=row['ref_path']:raise ValueError('ambiguous auxiliary source path')
    if set(paths)!=wanted:raise ValueError('auxiliary selection coverage differs')
    references=[];hashes=set()
    for index,(name,path) in enumerate(sorted(paths.items()),1):
        with Image.open(path) as image:
            if image.mode!='RGB':raise ValueError('independent auxiliary hash needs explicit RGB source: '+path)
            digest=hashlib.sha256(image.tobytes()).hexdigest();width,height=image.size
        hashes.add(digest)
        references.append(dict(source_filename=name,path=path,file_sha256=sha(path),rgb_sha256=digest,width=width,height=height))
        if index%500==0:print('Auxiliary RGB source hashes',index,'/',len(paths),flush=True)
    original=read(out/'EXTERNAL_INPUTS.json');populations={}
    for corpus,d in original['datasets'].items():
        bank=Path(d['feature_bank']);meta=pq.read_table(bank/'keys.parquet').to_pylist()
        if len(meta)!=len(d['target']):raise ValueError('external auxiliary-audit population differs')
        overlap=[]
        for index,row in enumerate(meta):
            digest=row.get('ref_pixels_sha256',row.get('reference_pixels_sha256'))
            if digest is None:raise ValueError('external reference digest missing')
            if digest in hashes:overlap.append(index)
        populations[corpus]=dict(rows=len(meta),keys_sha256=sha(bank/'keys.parquet'),overlap_rows=overlap)
    result=dict(schema='auxiliary-source-overlap-v1',time_utc=now(),selection_sha256=sha(selection),
        pool_keys_sha256=sha(keys),source_count=len(paths),references=references,populations=populations,
        exact_matches=sum(len(v['overlap_rows']) for v in populations.values()),
        scope='Conservative entire auxiliary-pool source union, including unused families; current retained RGB reference PNGs hashed independently through Pillow. Exact RGB only; crop/resize lineage and historical raw-source immutability are not established by this check.')
    write(out/'AUXILIARY_OVERLAP.json',result)
    print('Auxiliary source audit:',len(paths),'references;',result['exact_matches'],'exact overlapping rows',flush=True)
    if result['exact_matches']:raise ValueError('auxiliary source overlap requires withheld-population correction')


def auxiliary_d1_overlap(out):
    from color_noise_repair import read,write,sha
    auxiliary=read(out/'AUXILIARY_OVERLAP.json')
    hashes={r['rgb_sha256'] for r in auxiliary['references']}
    populations=read(out/'DEVELOPMENT_POPULATIONS.json');records=[]
    for member,pin in populations['bank_keys'].items():
        keys=BANK/member/'keys.parquet'
        if sha(keys)!=pin:raise ValueError('D1 source keys changed')
        refs=pq.read_table(keys,columns=['ref_pixels_sha256'])['ref_pixels_sha256'].to_pylist()
        indices=[i for i,digest in enumerate(refs) if digest in hashes]
        records.append(dict(member=member,rows=len(refs),keys_sha256=pin,overlap_rows=indices))
    total=sum(len(r['overlap_rows']) for r in records)
    write(out/'AUXILIARY_D1_OVERLAP.json',dict(schema='auxiliary-d1-source-overlap-v1',
        auxiliary_sha256=sha(out/'AUXILIARY_OVERLAP.json'),populations=records,exact_matches=total,
        scope='Same conservative auxiliary union versus all six admitted D1 human bank members. Exact RGB only; crop/resize and historical raw-source limitations remain.',time_utc=now()))
    print('Auxiliary versus D1 banks:',len(records),'members;',total,'exact overlapping rows',flush=True)
    if total:raise ValueError('auxiliary overlap with D1 source populations requires correction')


def assessment_plan(out):
    from color_noise_repair import registration,write,sha
    registration(out)
    if (out/'TRAIN_SELECTION.json').exists():raise ValueError('assessment expansion must precede selection')
    write(out/'ASSESSMENT_PLAN.json',dict(schema='color-noise-assessment-plan-v1',
        state='frozen-before-selection',registration_sha256=sha(out/'REGISTRATION.json'),
        populations=EXTERNAL,source=str(BASE/'WITHHELD_INPUTS.json'),source_sha256=sha(BASE/'WITHHELD_INPUTS.json'),
        amendment='Prospectively include the original exposed CID22-B and JPEG KonJND SELECT panels alongside CSIQ/PIPAL; LIVE remains diagnostic. No new population, selection metric, fit or qualification gate.',
        models='All four fixed arms at seed0 plus all six original gauntlet rows',time_utc=now()))


def feature_compatibility(out):
    """Replay twelve TRAIN pixel pairs through the existing bank owner."""
    import pyarrow as pa
    from color_noise_repair import read,write,sha,derived_declaration
    from gauntlet_frozen import EXTRACT
    root=out/'feature-compatibility';request=root/'request';request.mkdir(parents=True)
    samples=[];keys=[];reference=[];pins={}
    ids=sorted(set(derived_declaration('fx2')[2]['direct'])|
        {b for _,b in derived_declaration('fx2')[2]['products']})
    for member in MEMBER_SOURCE:
        bank=BANK/member;m=read(bank/'_MANIFEST.json')
        if sha(bank/'keys.parquet')!=m['keys_sha256'] or sha(bank/'features.parquet')!=m['features_parquet_sha256']:
            raise ValueError('TRAIN feature bank identity changed')
        rows=pq.read_table(bank/'keys.parquet').to_pylist()
        subset=[r for r in rows if not r['pixels_identical']]
        if member.startswith('kadid'):subset=[r for r in subset if r['codec']=='kadid_12']
        elif member=='tid2013':subset=[r for r in subset if r['codec']=='tid_02']
        chosen=[subset[0],subset[-1]]
        features=pq.read_table(bank/'features.parquet',columns=['pair_key']+[f'f{i}' for i in ids])
        index={k:i for i,k in enumerate(features['pair_key'].to_pylist())}
        for row in chosen:
            i=len(keys);keys.append({**{k:row[k] for k in ['pair_key','ref_path','dist_path']},'row_id':i})
            reference.append([features[f'f{j}'][index[row['pair_key']]].as_py() for j in ids])
            samples.append(dict(member=member,bank_pair_key=row['pair_key'],codec=row['codec'],
                ref_pixels_sha256=row['ref_pixels_sha256'],dist_pixels_sha256=row['dist_pixels_sha256']))
        pins[member]=dict(manifest_sha256=sha(bank/'_MANIFEST.json'),features_sha256=sha(bank/'features.parquet'),keys_sha256=sha(bank/'keys.parquet'))
    keyfile=request/'keys.parquet';pq.write_table(pa.Table.from_pylist(keys),keyfile)
    reference=np.asarray(reference,dtype='<f8')
    pq.write_table(pa.table({f'f{j}':reference[:,k] for k,j in enumerate(ids)}),request/'reference.parquet')
    spec=dict(schema='rev5-assessment-keys-v1',labels_read=False,
        keys=dict(path=str(keyfile),sha256=sha(keyfile)),rows=len(keys),immutable_roots=[str(BANK)],
        extractor=dict(sha256=sha(EXTRACT)),instrument='TRAIN-feature-compatibility-only; no labels or model scoring',
        exposure_freeze=dict(path=str(out/'REGISTRATION.json'),sha256=sha(out/'REGISTRATION.json')))
    write(request/'request.json',spec)
    cmd=[sys.executable,str(REPO/'scripts/rev4_featpot/rev5_bank.py'),'extract','train_compatibility',
        '--bin',str(EXTRACT),'--build-commit','unresolved-retained-binary-sha256-'+sha(EXTRACT),
        '--revision','5','--era','rev5_localwin','--instrument-manifest',str(request/'request.json'),
        '--out',str(root/'fresh'),'--threads','1','--chunk','12']
    with (root/'extract.log').open('x') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    bank=root/'fresh/train_compatibility'
    fresh=pq.read_table(bank/'features.parquet',columns=[f'f{i}' for i in ids])
    actual=np.column_stack([fresh[f'f{j}'].to_numpy() for j in ids]).astype('<f8')
    pixels=pq.read_table(bank/'keys.parquet').to_pylist()
    for expected,observed in zip(samples,pixels,strict=True):
        if (expected['ref_pixels_sha256']!=observed['reference_pixels_sha256']
                or expected['dist_pixels_sha256']!=observed['distorted_pixels_sha256']):
            raise ValueError('TRAIN replay decoded pixels differ from feature producer')
    a,b=reference.astype('<f4'),actual.astype('<f4')
    changed=(a.view('<u4')!=b.view('<u4'))
    differences=[dict(row=i,feature=ids[j],stored=float(reference[i,j]),fresh=float(actual[i,j]))
        for i,j in np.argwhere(changed)]
    result=dict(schema='color-noise-train-feature-compatibility-v1',rows=len(keys),feature_ids=ids,
        source_banks=pins,request_sha256=sha(request/'request.json'),extractor_sha256=sha(EXTRACT),
        fresh_manifest_sha256=sha(bank/'_MANIFEST.json'),decoded_pixels_exact=True,
        f32_exact=not changed.any(),f32_differences=differences,
        max_abs_difference=float(np.max(np.abs(reference-actual))),
        f64_exact=bool(np.array_equal(reference.view('<u8'),actual.view('<u8'))),
        scope='Bounded TRAIN arithmetic replay, not complete cross-corpus or source-commit qualification.',time_utc=now())
    write(out/'FEATURE_COMPATIBILITY.json',result)
    print(json.dumps({k:result[k] for k in ['rows','decoded_pixels_exact','f32_exact','f64_exact','max_abs_difference']}),flush=True)


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


def smoke_score(out):
    from color_noise_repair import ARMS,read,write,sha
    from v2_lodo_mlp import predict
    population=read(out/'DEVELOPMENT_POPULATIONS.json');records=[]
    for arm in ARMS:
        r=read(out/'smokes'/arm/'s0/result.json');model=Path(r['selected_bake'])
        if r['execution_contract']!='local-smoke' or sha(model)!=r['selected_bake_sha256']:
            raise ValueError('expected original local smoke model')
        dest=out/'smoke-scoring'/arm;dest.mkdir(parents=True)
        pred=predict(model,Path(population['table']),dest/'pred.tsv')
        if len(pred)!=len(population['rows']):raise ValueError('smoke prediction population differs')
        records.append(dict(arm=arm,rows=len(pred),all_finite=True,model_sha256=sha(model),
            predictions_sha256=sha(dest/'pred.tsv'),densification='native BIT-IDENTICAL gate passed'))
    write(out/'SMOKE_SCORING.json',dict(schema='color-noise-native-smoke-scoring-v1',
        cells=records,not_selection_data=True,time_utc=now()))


def parity(out, seeds=range(3)):

    """Reuse the E33 parity owner's native metadata stripping operation."""
    from color_noise_repair import read,write,sha
    original=read(BASE/'models/diff-frag-820-training-manifest.json')
    oldroot=Path(original['selected_bake']).parent.parent.parent
    partial=list(seeds)!=[0,1,2]
    dest=out/('control-parity-seed0' if partial else 'control-parity');dest.mkdir()
    records=[]
    for seed in seeds:
        old=oldroot/f'full_s{seed}'/'result.json'
        new=out/'cells/current'/f's{seed}'/'result.json'
        a,b=read(old),read(new)
        same={k:a[k]==b[k] for k in ['init_seed','sample_seed','train_weights','dev_curve']}
        for key in ['selected_bake','packed_model']:
            stripped=[]
            for label,result in [('original',a),('matched',b)]:
                model=Path(result[key])
                if sha(model)!=result[key+'_sha256']:raise ValueError('control model pin changed')
                target=dest/f'{label}-s{seed}-{key}.bin'
                subprocess.run([str(out/'bin/bake_dial_refit'),'strip','--in',str(model),
                    '--out',str(target),'--key','zentrain.repro'],check=True,capture_output=True)
                stripped.append(sha(target))
            same[key+'_nonrepro_bytes']=stripped[0]==stripped[1]
        records.append(dict(seed=seed,fields_equal=same,pass_all=all(same.values())))
    write(out/('CONTROL_PARITY_S0.json' if partial else 'CONTROL_PARITY.json'),dict(schema='color-noise-control-replay-v1',
        records=records,pass_all=all(r['pass_all'] for r in records),time_utc=now()))
    if not all(r['pass_all'] for r in records):raise ValueError('original control replay mismatch; inspect before assessment')


def development_cell(out,arm,seed):
    from color_noise_repair import read,write,sha,arm_record,confirm_seeds
    from v2_lodo_mlp import predict
    from lib.zen_stats import panel_batch
    p=read(out/'DEVELOPMENT_POPULATIONS.json')
    if p['registration_sha256']!=sha(out/'REGISTRATION.json') or sha(p['table'])!=p['table_sha256']:
        raise ValueError('development inputs changed')
    cell=out/'cells'/arm/f's{seed}';r=read(cell/'result.json');model=Path(r['packed_model'])
    expected=arm_record(arm);reg=read(out/'REGISTRATION.json')
    if (r['execution_contract']!='registered-fit' or r['selection']['selected_epoch']!=119
            or r['epochs']!=120 or r['pairs_per_epoch']!=50000 or sha(model)!=r['packed_model_sha256']
            or r.get('spec')!=expected['spec'] or r.get('head')!='N' or r.get('seed_index')!=seed
            or (r.get('init_seed'),r.get('sample_seed'))!=confirm_seeds(seed)
            or r.get('kept_features')!=len(expected['columns'])
            or r.get('derived_inputs')!=dict(token=expected['token'],declaration_sha256=expected['declaration_sha256'])
            or r.get('frozen_sha256')!=reg['frozen_sha256']
            or r.get('data_role_decision_sha256')!=reg['role_decision_sha256']
            or r.get('output_stage')!='e33-identity-knot-tail2'):
        raise ValueError('registered full fit/packing missing: '+str(cell))
    binding=dict(path=str(model),sha256=sha(model),result_sha256=sha(cell/'result.json'))
    dest=out/'development'/arm/f's{seed}';receipt=dest/'CELL.json'
    if receipt.exists():
        cached=read(receipt)
        if (cached['model']!=binding or cached['population_sha256']!=sha(out/'DEVELOPMENT_POPULATIONS.json')
                or cached['predictions_sha256']!=sha(dest/'pred.tsv')):
            raise ValueError('cached TRAIN-development identity differs')
        return cached
    dest.mkdir(parents=True,exist_ok=False)
    target=pq.read_table(p['table'],columns=['human_score'])['human_score'].to_numpy()
    pred=predict(model,Path(p['table']),dest/'pred.tsv')
    if len(pred)!=len(target):raise ValueError('development prediction coverage differs')
    jobs=[(name,pred[indices],target[indices]) for name,indices in p['populations'].items()]
    stats=panel_batch(jobs);records=[]
    for (name,_,_),st in zip(jobs,stats,strict=True):
        if st['n_dropped'] or not np.isfinite(st['srocc_signed']):raise ValueError('incomplete canonical statistics')
        records.append(dict(arm=arm,seed=seed,population=name,rows=len(p['populations'][name]),statistics=st))
    # Report-only severity diagnostics. The registered pooled choice is unchanged.
    ladders={}
    for source in ['kadid','tid2013']:
        for i in p['populations'][source+'_noise']:
            row=p['rows'][i];key=(source,row['ref_basename'],row['codec'])
            ladders.setdefault(key,[]).append(i)
    local_jobs=[('/'.join(key),pred[indices],target[indices]) for key,indices in sorted(ladders.items())]
    local_stats=panel_batch(local_jobs,stats='srocc');local=[]
    for ((key,indices),st) in zip(sorted(ladders.items()),local_stats,strict=True):
        if st['n_dropped'] or not np.isfinite(st['srocc_signed']):raise ValueError('incomplete local-noise diagnostic')
        local.append(dict(arm=arm,seed=seed,source=key[0],reference=key[1],codec=key[2],
            rows=len(indices),indices=indices,pred=pred[indices].tolist(),target=target[indices].tolist(),statistics=st))
    result=dict(schema='color-noise-development-cell-v1',model=binding,
        population_sha256=sha(out/'DEVELOPMENT_POPULATIONS.json'),predictions_sha256=sha(dest/'pred.tsv'),
        records=records,local_noise=local,time_utc=now())
    write(receipt,result)
    return result


def development_completed(out):
    from color_noise_repair import ARMS,registration
    registration(out);complete=[];pending=[]
    for seed in range(3):
        for arm in ARMS:
            if not (out/'cells'/arm/f's{seed}'/'result.json').exists():
                pending.append(f'{arm}/s{seed}');continue
            complete.extend(development_cell(out,arm,seed)['records'])
    print(json.dumps(dict(state='partial TRAIN diagnostics; no selection',pending=pending,statistics=complete),allow_nan=False),flush=True)


def select(out):
    from color_noise_repair import ARMS, registration, read, write, sha
    registration(out)
    if read(out/'CONTROL_PARITY.json')['pass_all'] is not True:
        raise ValueError('original control replay must be resolved before selection')
    records=[];models={};local=[]
    for seed in range(3):
        for arm in ARMS:
            cell=development_cell(out,arm,seed)
            records.extend(cell['records']);models[f'{arm}/s{seed}']=cell['model']
            local.extend(cell['local_noise'])
    selected,decisions=choose(records)
    write(out/'TRAIN_SELECTION.json',dict(schema='color-noise-train-choice-v1',state='frozen-before-external-read',
        registration_sha256=sha(out/'REGISTRATION.json'),population_sha256=sha(out/'DEVELOPMENT_POPULATIONS.json'),
        assessment_plan_sha256=sha(out/'ASSESSMENT_PLAN.json'),
        selected=selected,assessment_seed=0,decisions=decisions,models=models,statistics=records,local_noise=local,
        selection_inputs='TRAIN-development only; no external/LIVE input',qualification=False,time_utc=now()))
    print('Frozen TRAIN decision:',selected,decisions,flush=True)


def external(out):
    from color_noise_repair import ARMS, registration, read, write, sha
    from gauntlet_frozen import PREDICT
    registration(out);selection=read(out/'TRAIN_SELECTION.json')
    if selection['state']!='frozen-before-external-read' or selection['registration_sha256']!=sha(out/'REGISTRATION.json'):
        raise ValueError('frozen TRAIN decision required before external read')
    plan=read(out/'ASSESSMENT_PLAN.json')
    if plan['populations']!=EXTERNAL or sha(out/'ASSESSMENT_PLAN.json')!=selection['assessment_plan_sha256']:
        raise ValueError('assessment population changed after selection')
    source=BASE/'WITHHELD_INPUTS.json';original=read(source)
    if sha(source)!=plan['source_sha256']:raise ValueError('original assessment input changed')
    doc=dict(schema='zensim-frozen-gauntlet-v1',models=copy.deepcopy(original['models']),
        datasets={s:copy.deepcopy(original['datasets'][s]) for s in EXTERNAL},
        references=original['references'],notes=[
            'Exploratory fixed 2x2 color/noise repair. Three matched seeds selected on TRAIN-development only; assessment seed0 is fixed. Selected arm: '+selection['selected']+'. No product qualification.',
            'All five external panels were previously exposed in the original gauntlet. LIVE motivated the diagnosis and is an exposed regression diagnostic, never fitting or selection data. KonJND is JPEG SELECT only; terminal sources excluded. This is not fresh independent holdout confirmation.',
            'The source-pixel overlap screen found no exact matches in these admitted populations. LIVE retains the existing10-family subset. Other crop/resize lineage is not comprehensively cleared. Native feature/binary identities remain pinned; external extractor source-commit closure remains unresolved.',
            'All statistics and scatter diagnostics use the canonical native owners. SROCC intervals:10000 source-cluster draws; small-source intervals descriptive. PLCC/KROCC intervals unmeasured. Every population point is retained in JSON.',
            'The separate TRAIN-development decision and all three seeds are recorded in TRAIN_SELECTION.json. Original frozen comparators and peers reuse their exact original vectors.'])
    exposures=dict(schema='frozen-public-test-exposure-v1',secret_holdouts_accessed=False,
        populations=EXTERNAL,prior_exposure=str(BASE),source_sha256=sha(source),
        choice_sha256=sha(out/'TRAIN_SELECTION.json'),models=selection['models'],time_utc=now(),
        use='Frozen assessment after TRAIN-only choice; no adaptive tuning or independent-holdout claim')
    write(out/'EXTERNAL_EXPOSURE.json',exposures)
    work=out/'external';work.mkdir()
    for arm in ARMS:
        name='repair-'+arm;r=selection['models'][arm+'/s0'];model=Path(r['path'])
        if sha(model)!=r['sha256']:raise ValueError('frozen model changed')
        doc['models'].append(dict(name=name,bake=str(model),bake_sha256=r['sha256'],regime='TRAIN-only 2x2; seed0',
            n_inputs=next(a['input_count'] for a in registration(out)['arms'] if a['name']==arm),
            model=dict(kind='mlp',note=arm+'; '+('TRAIN-selected' if arm==selection['selected'] else 'fixed comparison arm'))))
        for corpus,data in doc['datasets'].items():
            bank=Path(data['feature_bank']);manifest=read(bank/'_MANIFEST.json')
            if sha(bank/'features.parquet')!=manifest['features_parquet_sha256'] or sha(bank/'keys.parquet')!=manifest['keys_sha256']:
                raise ValueError('external feature bank changed')
            if sha(PREDICT)!=data['scores']['diff-frag-820']['provenance']['predictor_sha256']:
                raise ValueError('original native prediction owner changed')
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
            if arm=='current' and pred!=data['scores']['diff-frag-820']['pred']:
                raise ValueError('matched control serving differs from original frozen vector')
            data['scores'][name]=dict(pred=pred,status='external-screened',
                note='Frozen TRAIN-only repair; exact same original gauntlet population. LIVE is exposed diagnostic only.',
                provenance=dict(model_sha256=sha(model),feature_manifest_sha256=sha(bank/'_MANIFEST.json'),
                    wire_sha256=sha(wire),predictor_sha256=sha(PREDICT),selection_sha256=sha(out/'TRAIN_SELECTION.json')))
            print(name,corpus,len(pred),flush=True)
    write(out/'EXTERNAL_INPUTS.json',doc)


def dashboard(out):
    from build_peer_fullevals import build_comparison
    from color_noise_repair import read,write,sha
    doc=read(out/'EXTERNAL_INPUTS.json')
    fresh=copy.deepcopy(doc)
    fresh['models']=[m for m in fresh['models'] if m['name'].startswith('repair-')]
    write(out/'NEW_MODEL_INPUTS.json',fresh)
    build_comparison(out/'NEW_MODEL_INPUTS.json',out)
    for model in doc['models']:
        if model['name'].startswith('repair-'):continue
        source=BASE/(model['name']+'.fulleval.json');old=read(source)
        for field in ['rank','scatter','scatter_assessment','per_pair','training_overlap']:
            old[field]={k:v for k,v in old[field].items() if k in doc['datasets']}
        old['comparison_notes']=doc['notes']
        old['retained_statistics_provenance']=dict(path=str(source),sha256=sha(source),
            requested_manifest_sha256=sha(out/'EXTERNAL_INPUTS.json'),
            note='Original canonical statistics and complete vectors copied verbatim for unchanged populations.')
        for corpus,d in doc['datasets'].items():
            if old['per_pair'][corpus]['pred']!=d['scores'][model['name']]['pred'] or old['per_pair'][corpus][d['axis']]!=d['target']:
                raise ValueError('retained comparison population changed')
        write(out/(model['name']+'.fulleval.json'),old)
    subprocess.run([sys.executable,str(REPO/'scripts/v_next/gauntlet.py'),'--fulleval-dir',str(out),
        '--comparison-only','--include-all-scatter','--out',str(out/'gauntlet.html')],check=True)


def report(out):
    from color_noise_repair import ARMS, read, write, sha
    choice=read(out/'TRAIN_SELECTION.json');data=read(out/'EXTERNAL_INPUTS.json')
    names=['diff-frag-820','w-c','diff-410']+['repair-'+a for a in ARMS]+['peer_cvvdp_fhd','peer_ssim2','peer_butter3']
    labels=['Frozen C','Frozen W-C','Frozen A','Matched current','+coverage','+fine chroma','Both','CVVDP-FHD','SSIMULACRA2','Butteraugli 3-norm']
    ensemble_choice=None
    if (out/'ensembles/TRAIN_SELECTION.json').exists():
        ensemble_choice=read(out/'ensembles/TRAIN_SELECTION.json')
        names+=['repair-ensemble-'+a for a in ARMS]
        labels+=['Mean3 current','Mean3 coverage','Mean3 chroma','Mean3 both']
    docs={n:read(out/('ensembles' if n.startswith('repair-ensemble-') else '')/(n+'.fulleval.json')) for n in names}
    text=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Color-noise repair comparison</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px system-ui;background:#10141d;color:#e4eaf5;line-height:1.6}a{color:#8ecaff}h1{font-size:28px}h2{font-size:22px;margin-top:36px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #485166;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%}summary{cursor:pointer}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>',
        '<h1>Color-noise repair comparison</h1>',
        '<p>TRAIN-development selection: <strong>'+html.escape(choice['selected'])+'</strong>. Four fixed arms, three matched seeds, original 120-epoch budget. Assessment uses seed0. This is an exploratory repair experiment; original qualification gates remain required.</p>',
        '<p><a href="gauntlet.html">Open full scatter comparison</a> · <a href="TRAIN_SELECTION.json">All TRAIN-development results</a> · <a href="REGISTRATION.json">Frozen experiment</a></p>',
        '<p>LIVE motivated this investigation and is an exposed diagnostic. It did not choose the repair. All five assessment panels were previously exposed; this assessment cannot establish fresh independent holdout performance. KonJND uses JPEG SELECT only.</p>',
        '<h2>TRAIN-only decision</h2><p>Deltas are three-seed means against matched current, using canonical signed SROCC. Noise combines the registered KADID/TID subsets with equal source weights.</p>',
        '<div class="scroll"><table><tr><th>Arm</th><th>Human mean Δ</th><th>Noise Δ</th>'+''.join('<th>'+html.escape(s)+' Δ</th>' for s in SOURCES)+'<th>Eligible</th></tr>']
    for r in choice['decisions']:
        text.append('<tr><td>'+r['arm']+'</td>'+''.join(f'<td>{v:+.5f}</td>' for v in [r['human_mean_delta'],r['noise_mean_delta'],*[r['source_mean_delta'][s] for s in SOURCES]])+'<td>'+str(r['eligible'])+'</td></tr>')
    text.append('</table></div>')
    if ensemble_choice:
        text.append('<h2>Fixed three-seed ensemble supplement</h2><p>Separate prospective TRAIN choice: <strong>'+html.escape(ensemble_choice['selected'])+'</strong>. Equal-weight native serving scores, fixed seeds0/1/2; no seed/weight search. Human deltas are against the current ensemble. Original single-model selection is unchanged. <a href="ensembles/gauntlet.html">Ensemble scatter comparison</a> · <a href="ensembles/TRAIN_SELECTION.json">TRAIN results and exact composition manifests</a></p><table><tr><th>Arm</th><th>Human mean Δ</th><th>Noise Δ</th><th>Eligible</th></tr>')
        for r in ensemble_choice['decisions']:
            text.append(f'<tr><td>{r["arm"]}</td><td>{r["human_mean_delta"]:+.5f}</td><td>{r["noise_mean_delta"]:+.5f}</td><td>{r["eligible"]}</td></tr>')
        text.append('</table>')
    if choice.get('models'):
        text.append('<h2>Candidate bakes</h2><p>Fixed seed0 packed f16 artifacts, scored through the native serving surface. These are exploratory candidates; the named production profile has not changed.</p><div class="scroll"><table><tr><th>Arm</th><th>Download</th><th>SHA256</th></tr>')
        for arm in ARMS:
            binding=choice['models'][arm+'/s0']
            relative=Path(binding['path']).relative_to(out).as_posix()
            text.append('<tr><td>'+arm+(' (TRAIN-selected)' if arm==choice['selected'] else '')+'</td><td><a href="'+html.escape(relative,quote=True)+'">Packed bake</a></td><td><code>'+binding['sha256']+'</code></td></tr>')
        text.append('</table></div>')
    if choice.get('local_noise'):
        text.append('<h2>TRAIN noise severity within each reference</h2><p>Report-only checks of individual reference × distortion-type ladders. Each cell below is the mean / minimum of canonical signed SROCCs across those ladders. Individual predictions, targets and statistics are retained in TRAIN_SELECTION.json. These checks do not change the registered pooled selection rule.</p><table><tr><th>Arm / seed</th><th>KADID</th><th>TID2013</th></tr>')
        for arm in ARMS:
            for seed in range(3):
                text.append('<tr><td>'+arm+f' / {seed}</td>')
                for source in ['kadid','tid2013']:
                    values=[r['statistics']['srocc_signed'] for r in choice['local_noise'] if r['arm']==arm and r['seed']==seed and r['source']==source]
                    text.append(f'<td>{float(np.mean(values)):.4f} / {min(values):.4f} ({len(values)} ladders)</td>')
                text.append('</tr>')
        text.append('</table>')
    text.append('<h2>Frozen external assessment</h2><p>Each cell shows signed SROCC / fitted PLCC. Full canonical statistics, source-cluster SROCC intervals and every point are in the downloadable JSONs and scatter comparison.</p><div class="scroll"><table><tr><th>Corpus</th>'+''.join('<th>'+s+'</th>' for s in labels)+'</tr>')
    ranks=[]
    for corpus in EXTERNAL:
        text.append('<tr><td>'+corpus+(' (exposed diagnostic)' if corpus=='live' else '')+'</td>')
        for name in names:
            r=docs[name]['rank'][corpus]
            text.append(f'<td>{r["srocc_signed"]:.4f} / {r["plcc"]:.4f}</td>')
            ranks.append(dict(model=name,corpus=corpus,**r))
        text.append('</tr>')
    text.append('</table></div><h2>The ten severe LIVE white-noise examples</h2><p>The order preserves the original inspection, with manfishing first. Tables show native serving scores (higher means better quality), not MOS-normalized plot coordinates. Human quality uses the original gauntlet scale. Images retain their original PNG bytes.</p>')
    cases=read(BASE.parent/'gauntlet-live-zero-2026-10-18/CASES.json')
    live=data['datasets']['live'];target=live['target'];case_scores=[]
    for case in cases['cases']:
        if not case['primary']:continue
        i=case['index']
        if target[i]!=case['target']:raise ValueError('original LIVE case order changed')
        text.append('<h3>'+html.escape(case['reference']+' / '+case['filename'])+f' — human quality {target[i]:.4f}</h3><div class="pair">')
        for field,label in [('ref_image','Reference'),('dist_image','Severe white noise')]:
            text.append('<figure><figcaption>'+label+'</figcaption><img loading="lazy" alt="'+label+'" src="/zensim/gauntlet-live-zero-2026-10-18/'+case[field]+'"></figure>')
        text.append('</div><div class="scroll"><table><tr>'+''.join('<th>'+s+'</th>' for s in labels)+'</tr><tr>'+''.join(f'<td>{docs[n]["per_pair"]["live"]["pred"][i]:.4f}</td>' for n in names)+'</tr></table></div>')
        rows=[j for j,p in enumerate(live['pixels']) if live['reference'][j]==live['reference'][i] and Path(p['dist_path']).parent.name=='wn']
        text.append('<details><summary>All actual white-noise severities for this reference</summary><div class="scroll"><table><tr><th>Image</th><th>Human quality</th>'+''.join('<th>'+s+'</th>' for s in labels)+'</tr>')
        for j in sorted(rows,key=lambda j:target[j]):
            text.append('<tr><td>'+html.escape(Path(live['pixels'][j]['dist_path']).name)+f'</td><td>{target[j]:.4f}</td>'+''.join(f'<td>{docs[n]["per_pair"]["live"]["pred"][j]:.4f}</td>' for n in names)+'</tr>')
        text.append('</table></div></details>')
        case_scores.append(dict(index=i,reference=case['reference'],target=target[i],scores={n:docs[n]['per_pair']['live']['pred'][i] for n in names}))
    text.append('<p>Remaining limitations: reused sparse ordinal ladders; three-seed exploratory selection; external feature extractor source-commit closure unresolved; crop/resize source lineage not comprehensively cleared. No protected populations were accessed.</p></html>')
    with (out/'index.html').open('x') as f:f.write(''.join(text))
    write(out/'REPORT.json',dict(schema='color-noise-repair-report-v1',selected=choice['selected'],
        train_selection_sha256=sha(out/'TRAIN_SELECTION.json'),external_exposure_sha256=sha(out/'EXTERNAL_EXPOSURE.json'),
        ensemble_selected=ensemble_choice['selected'] if ensemble_choice else None,
        ensemble_selection_sha256=sha(out/'ensembles/TRAIN_SELECTION.json') if ensemble_choice else None,
        ranks=ranks,severe_cases=case_scores,qualified=False,time_utc=now()))


def finalize(out):
    """Bind the completed study for the existing full-tree archive owner."""
    from color_noise_repair import ARMS,registration,read,write,sha
    reg=registration(out);choice=read(out/'TRAIN_SELECTION.json')
    exposure=read(out/'EXTERNAL_EXPOSURE.json');data=read(out/'EXTERNAL_INPUTS.json')
    report_data=read(out/'REPORT.json')
    if not read(out/'CONTROL_PARITY.json')['pass_all']:
        raise ValueError('control replay failed')
    if (exposure['choice_sha256']!=sha(out/'TRAIN_SELECTION.json')
            or report_data['train_selection_sha256']!=sha(out/'TRAIN_SELECTION.json')
            or report_data['external_exposure_sha256']!=sha(out/'EXTERNAL_EXPOSURE.json')
            or report_data['selected']!=choice['selected']):
        raise ValueError('decision/exposure/report binding differs')
    if report_data['qualified'] or choice['qualification']:
        raise ValueError('exploratory study cannot claim qualification')
    if set(choice['models'])!={f'{a}/s{s}' for a in ARMS for s in range(3)}:
        raise ValueError('registered model population differs')
    compatibility=read(out/'FEATURE_COMPATIBILITY.json')
    if not all(compatibility[k] for k in ['decoded_pixels_exact','f32_exact','f64_exact']):
        raise ValueError('TRAIN feature replay differs')
    if read(out/'AUXILIARY_OVERLAP.json')['exact_matches']!=0:
        raise ValueError('auxiliary source overlap unresolved')
    if read(out/'AUXILIARY_D1_OVERLAP.json')['exact_matches']!=0:
        raise ValueError('auxiliary D1 source overlap unresolved')
    for binding in choice['models'].values():
        if sha(binding['path'])!=binding['sha256']:
            raise ValueError('frozen model changed')
    expected={'diff-frag-820','w-c','diff-410','peer_cvvdp_fhd','peer_ssim2','peer_butter3'}|{'repair-'+a for a in ARMS}
    if {m['name'] for m in data['models']}!=expected or len(data['models'])!=len(expected):
        raise ValueError('comparison model population differs')
    if set(data['datasets'])!=set(EXTERNAL):raise ValueError('external panel population differs')
    cells=0
    for model in data['models']:
        doc=read(out/(model['name']+'.fulleval.json'))
        if set(doc['rank'])!=set(EXTERNAL):raise ValueError('assessment panel population differs')
        for corpus,d in data['datasets'].items():
            points=doc['per_pair'][corpus]
            if points['pred']!=d['scores'][model['name']]['pred'] or points[d['axis']]!=d['target']:
                raise ValueError('assessment vectors differ')
            cells+=1
    execution=read(out/'EXECUTION_SOURCE.json')
    training=['color_noise_repair.py','v2_confirm_fit.py','v2_common.py','v2_lodo_mlp.py','v2_human_role.py']
    for name in training:
        relative='scripts/rev4_featpot/'+name
        if sha(REPO/relative)!=execution['sources'][relative]:
            raise ValueError('executed training owner changed')
    files=['REGISTRATION.json','ADMISSION.json','EXECUTION_SOURCE.json','ASSESSMENT_PLAN.json','AUXILIARY_OVERLAP.json','AUXILIARY_D1_OVERLAP.json',
           'DEVELOPMENT_POPULATIONS.json','FEATURE_COMPATIBILITY.json','CONTROL_PARITY.json',
           'TRAIN_SELECTION.json','EXTERNAL_EXPOSURE.json','EXTERNAL_INPUTS.json','REPORT.json',
           'gauntlet.html','index.html']
    ensemble_cells=0;ensemble_models=0
    if (out/'ensembles/TRAIN_SELECTION.json').exists():
        root=out/'ensembles';ec=read(root/'TRAIN_SELECTION.json');ed=read(root/'EXTERNAL_INPUTS.json')
        if ec['parent_choice_sha256']!=sha(out/'TRAIN_SELECTION.json') or ec['qualification']:
            raise ValueError('ensemble parent decision/qualification differs')
        if read(root/'EXTERNAL_EXPOSURE.json')['choice_sha256']!=sha(root/'TRAIN_SELECTION.json'):
            raise ValueError('ensemble exposure differs')
        if report_data['ensemble_selection_sha256']!=sha(root/'TRAIN_SELECTION.json') or report_data['ensemble_selected']!=ec['selected']:
            raise ValueError('ensemble report binding differs')
        ensemble_models=len(ed['models'])
        for binding in ec['models'].values():
            if sha(binding['path'])!=binding['sha256']:raise ValueError('ensemble composition changed')
        for model in ed['models']:
            doc=read(root/(model['name']+'.fulleval.json'))
            for corpus,d in ed['datasets'].items():
                if doc['per_pair'][corpus]['pred']!=d['scores'][model['name']]['pred'] or doc['per_pair'][corpus][d['axis']]!=d['target']:
                    raise ValueError('ensemble assessment vectors differ')
                ensemble_cells+=1
        files+=['ensembles/'+n for n in ['REGISTRATION.json','TRAIN_SELECTION.json','EXTERNAL_EXPOSURE.json','PREDICTION_VIEWS.json','EXTERNAL_INPUTS.json','gauntlet.html']]
    write(out/'_MANIFEST.json',dict(schema='color-noise-repair-complete-v1',
        build_commit=subprocess.check_output(['jj','log','--no-graph','-r','@-','-T','commit_id'],text=True,cwd=REPO).strip(),
        owner_sha256=sha(__file__),training_build_commit=execution['build_commit'],
        program_sha256=reg['program_sha256'],binaries=reg['binaries'],models=choice['models'],
        files={name:sha(out/name) for name in files},single_dashboard_model_count=len(data['models']),
        ensemble_dashboard_model_count=ensemble_models,report_model_count=len({r['model'] for r in report_data['ranks']}),report_rank_cells=len(report_data['ranks']),
        panel_count=len(EXTERNAL),statistical_cells=cells,ensemble_dashboard_cells=ensemble_cells,selected=choice['selected'],
        qualification_allowed=False,source_commit_limitation='Retained external extractor source commit unresolved; twelve TRAIN pixel pairs replay exactly at all504 consumed f32/f64 inputs.',
        scope='Completed exploratory three-seed 2x2 study. Original qualification gates remain required. External panels previously exposed; LIVE diagnostic only.',time_utc=now()))
    print('Completed evidence manifest:',len(data['models']),'models;',cells,'statistical cells',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['assessment-plan','prepare','feature-compatibility','development-completed','smoke-score','parity','parity-seed0','select','external','dashboard','report','finalize','aux-overlap','aux-d1-overlap'])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.environ['REV4_V2_BIN_DIR']=str(a.out/'bin')
    os.environ['ZEN_PANEL_BIN']=str(BASE/'bin/panel')
    from color_noise_repair import sha, read
    if sha(BASE/'bin/panel')!=read(BASE/'_MANIFEST.json')['panel_binary_sha256']:
        raise ValueError('canonical panel binary changed')
    {'assessment-plan':assessment_plan,'prepare':prepare,'feature-compatibility':feature_compatibility,'development-completed':development_completed,'smoke-score':smoke_score,'parity':parity,'parity-seed0':lambda r:parity(r,[0]),'select':select,'external':external,'dashboard':dashboard,'report':report,'finalize':finalize,'aux-overlap':auxiliary_overlap,'aux-d1-overlap':auxiliary_d1_overlap}[a.mode](a.out)


if __name__=='__main__':main()
