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


def select(out):
    from color_noise_repair import ARMS, registration, read, write, sha
    registration(out)
    if read(out/'CONTROL_PARITY.json')['pass_all'] is not True:
        raise ValueError('original control replay must be resolved before selection')
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


def report(out):
    from color_noise_repair import ARMS, read, write, sha
    choice=read(out/'TRAIN_SELECTION.json');data=read(out/'EXTERNAL_INPUTS.json')
    names=['diff-frag-820','w-c','diff-410']+['repair-'+a for a in ARMS]
    docs={n:read(out/(n+'.fulleval.json')) for n in names}
    labels=['Frozen C','Frozen W-C','Frozen A','Matched current','+coverage','+fine chroma','Both']
    text=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Color-noise repair comparison</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px system-ui;background:#10141d;color:#e4eaf5;line-height:1.6}a{color:#8ecaff}h1{font-size:28px}h2{font-size:22px;margin-top:36px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #485166;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%}summary{cursor:pointer}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>',
        '<h1>Color-noise repair comparison</h1>',
        '<p>TRAIN-development selection: <strong>'+html.escape(choice['selected'])+'</strong>. Four fixed arms, three matched seeds, original 120-epoch budget. Assessment uses seed0. This is an exploratory repair experiment; original qualification gates remain required.</p>',
        '<p><a href="gauntlet.html">Open full scatter comparison</a> · <a href="TRAIN_SELECTION.json">All TRAIN-development results</a> · <a href="REGISTRATION.json">Frozen experiment</a></p>',
        '<p>LIVE motivated this investigation and is an exposed diagnostic. It did not choose the repair. CSIQ/PIPAL were also previously exposed; this assessment cannot establish fresh independent holdout performance.</p>',
        '<h2>TRAIN-only decision</h2><p>Deltas are three-seed means against matched current, using canonical signed SROCC. Noise combines the registered KADID/TID subsets with equal source weights.</p>',
        '<div class="scroll"><table><tr><th>Arm</th><th>Human mean Δ</th><th>Noise Δ</th>'+''.join('<th>'+html.escape(s)+' Δ</th>' for s in SOURCES)+'<th>Eligible</th></tr>']
    for r in choice['decisions']:
        text.append('<tr><td>'+r['arm']+'</td>'+''.join(f'<td>{v:+.5f}</td>' for v in [r['human_mean_delta'],r['noise_mean_delta'],*[r['source_mean_delta'][s] for s in SOURCES]])+'<td>'+str(r['eligible'])+'</td></tr>')
    text.append('</table></div><h2>Frozen external assessment</h2><p>Each cell shows signed SROCC / fitted PLCC. Full canonical statistics, source-cluster SROCC intervals and every point are in the downloadable JSONs and scatter comparison.</p><div class="scroll"><table><tr><th>Corpus</th>'+''.join('<th>'+s+'</th>' for s in labels)+'</tr>')
    ranks=[]
    for corpus in ['csiq','pipal','live']:
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
        ranks=ranks,severe_cases=case_scores,qualified=False,time_utc=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','smoke-score','parity','parity-seed0','select','external','dashboard','report'])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.environ['REV4_V2_BIN_DIR']=str(a.out/'bin')
    os.environ['ZEN_PANEL_BIN']=str(BASE/'bin/panel')
    from color_noise_repair import sha, read
    if sha(BASE/'bin/panel')!=read(BASE/'_MANIFEST.json')['panel_binary_sha256']:
        raise ValueError('canonical panel binary changed')
    {'prepare':prepare,'smoke-score':smoke_score,'parity':parity,'parity-seed0':lambda r:parity(r,[0]),'select':select,'external':external,'dashboard':dashboard,'report':report}[a.mode](a.out)


if __name__=='__main__':main()
