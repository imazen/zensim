#!/usr/bin/env python3
"""Explicit frozen-model/cached-fold transport for build_peer_fullevals.

This adapter reads only named admitted inputs, never discovers label corpora.
Rank/scatter statistics remain owned by zen_stats and the gauntlet renderer.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import struct
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
E33 = Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/noise-input.json')
E34 = Path('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json')
RANK = Path('/mnt/v/output/zensim/addrfix-2026-10-16/ranking')
TABLES = Path('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/wide/main/real')
BANK = Path('/var/tmp/rev5-featbank')
RM = Path('/mnt/v/output/zensim/reports/refmetrics')
PREDICT = Path('/mnt/v/output/zensim/e34-launch-2026-10-15/bin/predict_features_with_bake')
CV = Path('/var/tmp/paper-cvvdp/bin/zenmetrics')
EXTRACT = Path('/mnt/v/output/zensim/shipgates-2026-10-16/bin/native-extractor')
MODELS = {'diff-frag-820':'89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e',
          'w-c':'f73059ff8f67ff177265d35dfd5ff5db61d8fc825a6b519cf6925dc8fc1d5e2a',
          'diff-410':'7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275'}
LIVE_CLEAN = {'building2','carnivaldolls','cemetry','churchandcapitol','coinsinfountain','dancers','flowersonih35','manfishing','monarch','studentsculpture'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,allow_nan=False)
        f.write('\n')


def tsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f,delimiter='\t'))


def key(ref, dist):
    # Exact named source plus distorted filename; reference ID avoids filename
    # collisions across sources. Key uniqueness and full coverage are checked.
    return Path(ref).stem.lower(), Path(dist).parent.name.lower(), Path(dist).name.lower()


def metric_cache(path, column):
    # Projection deliberately excludes all subjective columns (and never reads
    # terminal human labels from the historical all-KADID peer tables).
    table=pcsv.read_csv(path,parse_options=pcsv.ParseOptions(delimiter='\t'),
        convert_options=pcsv.ConvertOptions(include_columns=['ref_path','dist_path',column]))
    result={}
    for r in table.to_pylist():
        k=key(r['ref_path'],r['dist_path']);v=float(r[column])
        if k in result and result[k]!=v:raise ValueError('ambiguous peer cache key')
        result[k]=v
    return result


def prepare(out):
    assert sha(E33)=='45e5f49dd329eee4bfb7751fe43b5dd70e548094a89b0af5dfdfd31828dd2eda'
    assert sha(E34)=='a1511bab58833e202fe89d3c90ac663cd1e9dbbb88cc71057b5a7d99385caa9c'
    assert sha(CV)=='ca154a6a39f910c9cba21422eb9b238463661bfa2d4a8d8f10407904c83e4cd1'
    doc={'schema':'zensim-frozen-gauntlet-v1','models':[],'datasets':{},'references':{'peer_ssim2':'ssim2','peer_butter3':'butter','peer_cvvdp_fhd':'cvvdp'},
         'notes':['Exploratory frozen-model comparison; no product qualification or new training/selection. All displayed points are retained in the downloadable fulleval JSONs; the interactive plot embeds a deterministic sample of at most500 points per cell.',
                  'D1 training corpora use cached seed0 leave-one-dataset-out research fits (raw score units), NOT predictions from the full-data packed bake. The named SHA identifies that family’s frozen full-data deployment only. No in-sample full-data rank is shown. Fold fits and packed full-data models differ in weights, precision and output calibration; PLCC is after the canonical registered four-parameter logistic fit.',
                  '95% SROCC intervals resample whole source images (10000 draws, seed20261017, type7); fewer than30 sources: descriptive and potentially under-covering. PLCC/KROCC intervals are not computed by the available owner adapter.',
                  'Exclusions are listed in EXCLUSIONS.json beside this page. KonJND is JPEG SELECT only,404 sources; terminal100 excluded. KonFiG uses the exposed reconstructed design-grid quality scale, not raw observer MOS; SSIMULACRA2 was tuned on KonFiG, so that peer is descriptive. LIVE uses only the audit-fixed10 source families without known TID/SafeSyn overlap.']}
    doc['notes'].append('Excluded: '+', '.join(read(out/'EXCLUSIONS.json')['excluded']))
    for name,pin in MODELS.items():
        cmd=read(Path('/mnt/v/output/zensim/shipgates-2026-10-16/gaddr-attempt3')/name/'standard/command.json')
        model=Path(cmd[cmd.index('--bake')+1]);assert sha(model)==pin
        result=model.parent.parent/'result.json';r=read(result)
        assert r['human_sources']==['kadid','tid2013','konfig','cid22_a25']
        doc['models'].append(dict(name=name,bake=str(model),bake_sha256=pin,regime='Rev5; frozen family',
            model={'kind':'mlp','note':'Frozen full-data deployment; D1 rows explicitly use withheld family fits.'},
            training_manifest={'path':str(result),'sha256':sha(result),'admission':r['selection']['strict_table_admission']}))
    for name in doc['references']:
        doc['models'].append(dict(name=name,peer=True,regime='reference-metric',model={'kind':'reference-metric','note':name}))
    old={d['name']:d for d in read(E33)['datasets']};new={d['name']:d for d in read(E34)['datasets']}
    for fold in ['kadid','tid2013','konfig','cid22_a25']:
        a,b=old[fold],new[fold]
        assert a['target']==b['target'] and a['reference']==b['reference']
        pixels=read(RANK/f'{fold}-PIXEL_KEYS.json');assert len(pixels)==len(a['target'])
        data=dict(target=a['target'],reference=a['reference'],axis='jnd' if fold=='konfig' else 'mos',scores={},pixels=pixels,missing={})
        for name,source,arm in [('diff-frag-820',a,'c'),('diff-410',a,'a'),('w-c',b,'w-c')]:
            data['scores'][name]=dict(pred=source['predictions'][arm][0],status='withheld-fold',
                note=f'Full-data bake trained on {fold}; plot uses seed0 fit with entire {fold} withheld, from '+('E33' if source is a else 'E34')+'.',
                provenance={'capture':str(E33 if source is a else E34),'sha256':sha(E33 if source is a else E34),'arm':arm,'seed_index':0})
        cvpath=RANK/('kadid-indexed-scores.tsv' if fold=='kadid' else f'{fold}-scores.tsv')
        rows=tsv(cvpath);by_id={int(r['row_index']):r for r in rows};assert set(by_id)==set(range(len(pixels)))
        for i,p in enumerate(pixels):assert key(p['ref_path'],p['dist_path'])==key(by_id[i]['ref_path'],by_id[i]['dist_path'])
        data['scores']['peer_cvvdp_fhd']=dict(pred=[10*float(by_id[i]['cvvdp_cpu_imazen_v0_1_0_standard_fhd']) for i in range(len(pixels))],status='peer',note='Pinned ca154a6a; standard_fhd; 10*JOD.',provenance={'path':str(cvpath),'sha256':sha(cvpath)})
        doc['datasets'][fold]=data
    for name in ['csiq','konjnd_jpeg_select','cid22_b']:
        m=read(BANK/name/'_MANIFEST.json');features=BANK/name/'features.parquet';keys=BANK/name/'keys.parquet'
        assert sha(features)==m['features_parquet_sha256'] and sha(keys)==m['keys_sha256']
        pixels=pq.read_table(keys).to_pylist()
        # Exact source populations only: labels parsed numerically only after
        # membership in the admitted key list has been established.
        if name=='csiq':src=Path('/mnt/v/dataset/csiq/csiq_pairs.tsv');column='human_score';direction=1
        elif name=='cid22_b':src=Path('/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv');column='human_score';direction=1
        else:src=Path('/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv');column='human_score';direction=-1
        wanted={key(p['ref_path'],p['dist_path']) for p in pixels};labels={}
        for row in tsv(src):
            k=key(row['ref_path'],row['dist_path'])
            if k in wanted:
                value=direction*float(row[column])
                if k in labels:assert labels[k]==value
                labels[k]=value
        assert set(labels)==wanted,(name,len(labels),len(wanted))
        refs={n:i for i,n in enumerate(sorted({key(p['ref_path'],p['dist_path'])[0] for p in pixels}))}
        doc['datasets'][name]=dict(target=[labels[key(p['ref_path'],p['dist_path'])] for p in pixels],reference=[refs[key(p['ref_path'],p['dist_path'])[0]] for p in pixels],axis='jnd' if name.startswith('konjnd') else 'mos',pixels=pixels,scores={},missing={},feature_bank=str(BANK/name),label_provenance={'path':str(src),'sha256':sha(src),'column':column,'orientation':direction})
    # LIVE: the published audit supplies the fixed disjoint-source subset.
    rows=tsv('/mnt/v/datasets/LIVE/live_r2_pairs_png.tsv')
    rows=[r for r in rows if Path(r['ref_path']).stem.lower() in LIVE_CLEAN]
    assert len(rows)==267
    add_external(doc,out,'live',rows,'human_score')
    # PIPAL published ELO stimuli; the current three models have no PIPAL leg.
    rows=[];base=Path('/mnt/v/dataset/pipal');index={p.name:str(p) for d in ['Distortion_1','Distortion_2','Distortion_3','Distortion_4'] for p in (base/d).iterdir() if p.is_file()}
    for labels in sorted((base/'Train_Label').glob('*.txt')):
        for line in labels.read_text().splitlines():
            if not line.strip():continue
            filename,score=line.strip().split(',');filename=filename.strip()
            ref=base/'Train_Ref'/(filename.split('_')[0]+'.bmp')
            rows.append(dict(ref_path=str(ref),dist_path=index[filename],human_score=float(score)))
    assert len(rows)==23200
    add_external(doc,out,'pipal',rows,'human_score')
    # Match cached peer scores using metric-only column projection.
    stems={'kadid':'kadid','tid2013':'tid','cid22_a25':'cid22','cid22_b':'cid22','csiq':'csiq','live':'live','konjnd_jpeg_select':'konjnd'}
    for corpus,data in doc['datasets'].items():
        if corpus not in stems:continue
        stem=stems[corpus]
        for metric,col in [('ssim2','ssim2'),('butter3','butteraugli_pnorm3')]:
            filename=(f'{stem}_ssim2.tsv' if stem=='cid22' else f'{stem}_ssim2_heldout.tsv' if stem=='konjnd' else f'{stem}_ssim2_gpu.tsv') if metric=='ssim2' else (f'{stem}_butter.tsv' if stem=='cid22' else f'{stem}_butteraugli_heldout.tsv' if stem=='konjnd' else f'{stem}_butteraugli_gpu.tsv')
            path=RM/filename
            names=pcsv.open_csv(path,parse_options=pcsv.ParseOptions(delimiter='\t')).schema.names
            if col not in names:col=col+'_gpu'
            cache=metric_cache(path,col);ids=[key(p['ref_path'],p['dist_path']) for p in data['pixels']]
            if not all(k in cache for k in ids):
                data['missing']['peer_'+metric]='Cached pair paths do not cover this exact admitted population; fresh scoring required.';continue
            sign=-1 if metric=='butter3' else 1
            data['scores']['peer_'+metric]=dict(pred=[sign*cache[k] for k in ids],status='peer',note='Stored peer scores; '+('negative 3-norm distance' if sign<0 else 'higher quality'),provenance={'path':str(path),'sha256':sha(path),'column':col,'projection':'metric and pixel paths only; no human target column read'})
    write(out/'INPUTS.json',doc)
    print('prepared',[(k,len(d['target'])) for k,d in doc['datasets'].items()],flush=True)


def add_external(doc,out,name,rows,column):
    missing=[p for r in rows for p in [r['ref_path'],r['dist_path']] if not Path(p).is_file()]
    refs={v:i for i,v in enumerate(sorted({Path(r['ref_path']).stem.lower() for r in rows}))}
    pixels=[dict(ref_path=r['ref_path'],dist_path=r['dist_path'],pair_key=hashlib.sha256((r['ref_path']+'\0'+r['dist_path']).encode()).hexdigest(),row_id=i) for i,r in enumerate(rows)]
    data=dict(target=[float(r[column]) for r in rows],reference=[refs[Path(r['ref_path']).stem.lower()] for r in rows],axis='mos',pixels=pixels,scores={},missing={})
    if missing:
        data['unavailable_pixels']=sorted(set(missing))
        data['missing']={n:'Required original pixels unavailable: '+str(len(set(missing)))+' files; no partial-population substitution.' for n in [*MODELS,'peer_ssim2','peer_butter3','peer_cvvdp_fhd']}
    else:
        path=out/f'{name}-keys.parquet';pq.write_table(pa.Table.from_pylist(pixels),path,compression='zstd')
        spec=dict(schema='rev5-assessment-keys-v1',labels_read=False,keys={'path':str(path),'sha256':sha(path)},rows=len(rows),immutable_roots=[str(path.parent)],extractor={'sha256':sha(EXTRACT)})
        # Output bank is a sibling of immutable keys, required by owner admission.
        spec['immutable_roots']=[str(Path(rows[0]['ref_path']).parent)]
        write(out/f'{name}-feature-request.json',spec)
    if name=='live':data['label_provenance']={'path':'/mnt/v/datasets/LIVE/live_r2_pairs_png.tsv','sha256':sha('/mnt/v/datasets/LIVE/live_r2_pairs_png.tsv')}
    if name=='pipal':data['label_provenance']=[{'path':str(p),'sha256':sha(p)} for p in sorted(Path('/mnt/v/dataset/pipal/Train_Label').glob('*.txt'))]
    doc['datasets'][name]=data


def score(out):
    """Reuse native caches, extract only absent banks, and fill missing peers."""
    doc=read(out/'INPUTS.json')
    work=out/'scoring';work.mkdir(exist_ok=True)
    for corpus,data in doc['datasets'].items():
        print('scoring '+corpus,flush=True)
        if data.get('unavailable_pixels'):
            continue
        bank=Path(data['feature_bank']) if 'feature_bank' in data else None
        if corpus in ['live','pipal']:
            bank=work/'banks'/corpus
            if not (bank/'_MANIFEST.json').exists():
                request=out/'extraction-inputs'/corpus
                request.mkdir(parents=True,exist_ok=True)
                spec=read(out/f'{corpus}-feature-request.json')
                keys=request/'keys.parquet'
                if not keys.exists():keys.write_bytes(Path(spec['keys']['path']).read_bytes())
                spec['keys']['path']=str(keys)
                spec['instrument']='frozen-human-gauntlet-2026-10-18'
                spec['exposure_freeze']={'path':str(out/'INPUTS.json'),'sha256':sha(out/'INPUTS.json')}
                if not (request/'request.json').exists():write(request/'request.json',spec)
                cmd=[sys.executable,str(REPO/'scripts/rev4_featpot/rev5_bank.py'),'extract',corpus,
                     '--bin',str(EXTRACT),'--build-commit','unresolved-retained-binary-sha256-'+sha(EXTRACT),
                     '--revision','5','--era','rev5_localwin','--instrument-manifest',str(request/'request.json'),
                     '--out',str(work/'banks'),'--threads','4','--chunk','2000']
                with (work/f'{corpus}-extract.log').open('x') as log:
                    subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
            data['feature_bank']=str(bank)
        if bank is not None:
            manifest=read(bank/'_MANIFEST.json');features=bank/'features.parquet'
            assert sha(features)==manifest['features_parquet_sha256']
            names=[f'f{i}' for i in range(manifest['feature_width'])]
            table=pq.read_table(features,columns=names)
            assert len(table)==len(data['target'])
            # Exact feature values and canonical IDs; absent slots remain NaN,
            # never replaced by zeros. BakeScorer owns derived feature products.
            wire=work/f'{corpus}.features-f64.bin'
            if not wire.exists():
                values=np.column_stack([table[c].to_numpy() for c in names]).astype('<f8')
                with wire.open('xb') as f:
                    f.write(struct.pack('<II',len(names),len(table)));f.write(values.tobytes())
            for model in doc['models']:
                name=model['name']
                if model.get('peer') or name in data['scores']:continue
                dest=work/f'{corpus}-{name}.pred.txt'
                if not dest.exists():
                    cmd=[str(PREDICT),'--bake',model['bake'],'--features-file',str(wire),'--f64-wire','--production']
                    with dest.open('x') as output,(work/f'{corpus}-{name}.log').open('x') as log:
                        subprocess.run(cmd,check=True,stdout=output,stderr=log,env={**os.environ,'ZENSIM_FORMULA_REV':'5'})
                pred=[float(v) for v in dest.read_text().splitlines()]
                assert len(pred)==len(data['target']) and np.isfinite(pred).all()
                data['scores'][name]=dict(pred=pred,status='source-disjoint',note='Frozen packed full-data model; corpus is absent from the named human training sources. '+('LIVE excludes the19 known training-overlap source families. ' if corpus=='live' else '')+'No comprehensive crop-aware overlap clearance is claimed.',provenance={'model_sha256':model['bake_sha256'],'feature_manifest_sha256':sha(bank/'_MANIFEST.json'),'feature_sha256':sha(features),'predictor_sha256':sha(PREDICT)})
        # Existing zenmetrics batch owner: explicit pair TSV, no label column,
        # pinned CPU executable, no benchmark. Cached scores above stay reused.
        peer_pixels=data['pixels']
        if corpus=='pipal':
            receipts=[json.loads(line) for line in (out/'PEER_PIXELS.jsonl').read_text().splitlines()]
            delivery={r['input']:r for r in receipts};assert len(delivery)==23400
            native=pq.read_table(bank/'keys.parquet').to_pylist()
            for pixel in native:
                for side,native_side in [('ref','reference'),('dist','distorted')]:
                    delivered=delivery[pixel[side+'_path']]
                    assert delivered['source_sha256']==pixel[native_side+'_file_sha256']
                    assert delivered['rgb_sha256']==pixel[native_side+'_pixels_sha256']
            peer_pixels=[dict(ref_path=delivery[p['ref_path']]['output'],dist_path=delivery[p['dist_path']]['output']) for p in data['pixels']]
        pairs=work/f'{corpus}-png-pairs.tsv' if corpus=='pipal' else work/f'{corpus}-pairs.tsv'
        if not pairs.exists():
            with pairs.open('x') as f:
                writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['ref_path','dist_path','row_index'])
                writer.writerows((p['ref_path'],p['dist_path'],i) for i,p in enumerate(peer_pixels))
        for name,metric,field,sign in [('peer_cvvdp_fhd','cvvdp','cvvdp_cpu_imazen_v0_1_0_standard_fhd',10),('peer_ssim2','ssim2','ssim2',1),('peer_butter3','butteraugli','butteraugli_pnorm3',-1)]:
            if name in data['scores']:continue
            dest=work/f'{corpus}-{name}.tsv'
            if not dest.exists():
                cmd=[str(CV),'batch','--metric',metric,'--bench-on-start','no','--use-legacy-scheduler','--pairs',str(pairs),'--output',str(dest),'--jobs','4','--group-by-ref']
                if metric=='cvvdp':cmd+=['--display-model','standard_fhd']
                write(work/f'{corpus}-{name}-COMMAND.json',dict(argv=cmd,binary_sha256=sha(CV),pair_sha256=sha(pairs)))
                with (work/f'{corpus}-{name}.log').open('x') as log:
                    subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'RAYON_NUM_THREADS':'4'})
            rows=tsv(dest);lut={int(r['row_index']):r for r in rows}
            assert set(lut)==set(range(len(data['target']))) and len(rows)==len(lut)
            for i,pixel in enumerate(peer_pixels):assert key(pixel['ref_path'],pixel['dist_path'])==key(lut[i]['ref_path'],lut[i]['dist_path'])
            # Some implementations use a versioned suffix: select exactly one
            # native named output, never a max-norm fallback.
            if field not in rows[0]:
                candidates=[k for k in rows[0] if k.startswith(field+'_')]
                assert len(candidates)==1,(metric,list(rows[0]));field=candidates[0]
            pred=[sign*float(lut[i][field]) for i in range(len(lut))]
            assert np.isfinite(pred).all()
            data['scores'][name]=dict(pred=pred,status='peer',note='Pinned CPU executable ca154a6a; '+('standard_fhd;10*JOD' if metric=='cvvdp' else 'native '+metric+'; '+field),provenance={'path':str(dest),'sha256':sha(dest),'column':field,'binary_sha256':sha(CV)})
        if not (work/f'{corpus}-COMPLETE.json').exists():
            write(work/f'{corpus}-COMPLETE.json',dict(rows=len(data['target']),metrics=list(data['scores'])))
    write(out/'SCORED_INPUTS.json',doc)


def audit_overlap(out):
    """Mark every original item and remove exact known training source overlap."""
    doc=read(out/'SCORED_INPUTS.json')
    doc['datasets']['pipal']['label_provenance']=[{'path':str(p),'sha256':sha(p)} for p in sorted(Path('/mnt/v/dataset/pipal/Train_Label').glob('*.txt'))]
    doc['datasets']['live']['label_provenance']={'path':'/mnt/v/datasets/LIVE/live_r2_pairs_png.tsv','sha256':sha('/mnt/v/datasets/LIVE/live_r2_pairs_png.tsv')}
    train_hashes=set();training_keys={};pins=[]
    for member in ['safesyn','cid22_train','kadid_train','kadid_select','tid2013','konfig_train','konfig_val','cid22_a25']:
        p=BANK/member/'keys.parquet'
        table=pq.read_table(p,columns=['ref_pixels_sha256'])
        train_hashes.update(table['ref_pixels_sha256'].to_pylist())
        pins.append(dict(path=str(p),sha256=sha(p),columns=['ref_pixels_sha256']))
    for role in ['fit','dev']:
        p=TABLES/f'human_all_{role}.keys.parquet'
        for model in doc['models']:
            if model.get('peer'):continue
            name='human' if role=='fit' else 'human_development'
            admitted=next(r for r in model['training_manifest']['admission'] if r['name']==name)
            assert sha(p)==admitted['keys_sha256']
        rows=pq.read_table(p).to_pylist()
        training_keys.update({(r['member_set'],r['source_row_id']):role for r in rows})
        pins.append(dict(path=str(p),sha256=sha(p),columns=['member_set','source_row_id']))
    audit={'training_inputs':pins,'datasets':{},'limits':'Exact decoded-pixel source matching plus the prior LIVE crop-aware audit. Other cross-corpus crops/resizes are not comprehensively cleared.'}
    for corpus,data in doc['datasets'].items():
        pixels=data['pixels'];entries=[];keep=[]
        if corpus in ['kadid','tid2013','konfig','cid22_a25']:
            for i,p in enumerate(pixels):
                state=training_keys[(p['member_set'],p['source_row_id'])]
                entries.append(dict(row=i,key=p['pair_key'],full_data_use=state,plotted_prediction='withheld-fold'))
            keep=list(range(len(pixels)))
        else:
            bank=Path(data['feature_bank']);meta=pq.read_table(bank/'keys.parquet').to_pylist()
            assert len(meta)==len(pixels)
            for i,p in enumerate(meta):
                assert key(p['ref_path'],p['dist_path'])==key(pixels[i]['ref_path'],pixels[i]['dist_path'])
                digest=p.get('ref_pixels_sha256',p.get('reference_pixels_sha256'))
                assert digest is not None
                overlap=digest in train_hashes
                entries.append(dict(row=i,key=p['pair_key'],full_data_use='exact-training-source-match' if overlap else 'no-exact-match',reference_pixels_sha256=digest,plotted_prediction='excluded-source-overlap' if overlap else 'frozen-full-data'))
                if not overlap:keep.append(i)
            # All metrics use the identical pre-score, source-identity mask.
            for field in ['target','reference','pixels']:data[field]=[data[field][i] for i in keep]
            for scores in data['scores'].values():
                scores['pred']=[scores['pred'][i] for i in keep]
                if scores['status']=='source-disjoint':scores['status']='external-screened'
        audit['datasets'][corpus]=dict(original_rows=len(pixels),plotted_rows=len(keep),excluded_exact_overlap=len(pixels)-len(keep),items=entries)
        print(corpus, 'plotted',len(keep),'excluded exact overlaps',len(pixels)-len(keep),flush=True)
    doc['notes'].append('Peer implementation provenance is corpus-specific: metric-only cached GPU/native scores are reused; absent peer coverage is filled by the pinned CPU executable. No uniform-implementation claim is made. Seed0 fold intervals are conditional on that fit; they do not include training-seed variability.')
    doc['notes'].append('TRAINING_OVERLAP.json marks every item as full-data fit/development, source-match exclusion, or no exact source match. All metric rows use the same retained population per corpus. This exact-source screen does not certify absence of cross-corpus crops or resizes beyond the prior LIVE audit.')
    doc['notes'].append('External fresh feature extraction uses retained binary SHA256 '+sha(EXTRACT)+' with an unresolved source-commit closure; the manifest labels that uncertainty explicitly. Formula revision, feature-set identity, f32 audit and each original pixel hash are bound by the native bank owner. No fresh-confirmation claim.')
    write(out/'TRAINING_OVERLAP.json',audit)
    write(out/'WITHHELD_INPUTS.json',doc)


def peer_containers(out):
    """Recontainerize true BMPs; retain disguised PNG/JPEG bytes unchanged."""
    from PIL import Image
    records=[json.loads(line) for line in (out/'pipal-png/PIXELS.jsonl').read_text().splitlines()]
    by_path={r['input']:r for r in records}
    native=pq.read_table(out/'scoring/banks/pipal/keys.parquet').to_pylist()
    expected={}
    for r in native:
        for side,prefix in [('ref','reference'),('dist','distorted')]:
            expected[r[side+'_path']]=(r[prefix+'_file_sha256'],r[prefix+'_pixels_sha256'])
    assert len(expected)==23400
    aliases=out/'pipal-aliases';aliases.mkdir(exist_ok=False)
    for index,(path,(source,pixels)) in enumerate(expected.items()):
        assert sha(path)==source
        if path not in by_path:
            with Path(path).open('rb') as f:magic=f.read(8)
            # Transport routing only; compressed bytes remain untouched and
            # the pinned metric owner performs format detection and decoding.
            ext='.png' if magic==b'\x89PNG\r\n\x1a\n' else '.jpg' if magic[:2]==b'\xff\xd8' else None
            if ext is None:raise ValueError('unconverted non-BMP format')
            dest=aliases/(Path(path).stem+ext)
            os.link(path,dest)
            assert sha(dest)==source
            by_path[path]=dict(input=path,output=str(dest),source_sha256=source,rgb_sha256=pixels,unchanged_compressed_bytes=True)
        r=by_path[path];assert r['rgb_sha256']==pixels
        # Independent decoder is a read-only oracle. All written PNGs come
        # from zenpng; all existing compressed aliases retain original bytes.
        with Image.open(r['output']) as image:
            actual=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        if r.get('roundtrip_exact'):
            assert actual==pixels,(path,'independent BMP/PNG pixel disagreement')
        r['independent_decode_complete']=True
        if (index+1)%1000==0:print('independent pixel admission',index+1,'/',len(expected),flush=True)
    assert len(by_path)==23400
    with (out/'PEER_PIXELS.jsonl').open('x') as f:
        for r in by_path.values():f.write(json.dumps(r)+'\n')
    write(out/'PIPAL_CONTAINER_RECEIPT.json',dict(originals=23400,bmp_converted=len(records),compressed_aliases=23400-len(records),
        native_reference_file_and_rgb_hashes_verified=True,all_written_pngs_independent_pixel_hash_parity=True,
        unchanged_compressed_aliases_independently_decoded=True,receipt_sha256=sha(out/'PEER_PIXELS.jsonl')))


def summary(out):
    """Format recorded owner statistics; do not recalculate any statistic."""
    names=[*MODELS,'peer_cvvdp_fhd','peer_ssim2','peer_butter3']
    docs={name:read(out/(name+'.fulleval.json')) for name in names}
    corpora=[('kadid','KADID TRAIN+SELECT'),('tid','TID2013'),('konfig','KonFiG'),('cid22','CID22-A25'),('cid22_b','CID22-B24'),('csiq','CSIQ'),('konjnd_jpeg_select','KonJND JPEG SELECT'),('live','LIVE clean10'),('pipal','PIPAL')]
    lines=['# Frozen human gauntlet — 2026-10-18','',
        '6 metrics × 9 admitted panels (8 datasets); all54 cells complete. Exploratory; no qualification. Cells: **SROCC [95% source-cluster CI] / fitted PLCC**. KROCC and full-precision results are in each linked fulleval JSON. PLCC/KROCC intervals are unmeasured.','',
        '| Panel (rows) | diff+frag-820 | W-C | diff-410 | CVVDP-FHD | SSIM2 | butter3 |',
        '|---|---|---|---|---|---|---|']
    records=[]
    for corpus,label in corpora:
        ranks=[docs[name]['rank'][corpus] for name in names]
        n=ranks[0]['n'];assert all(r['n']==n for r in ranks)
        cells=[]
        for name,r in zip(names,ranks,strict=True):
            assert r['n_dropped']==0
            lo,hi=r['srocc_ci'];cells.append(f"{r['srocc_signed']:.3f} [{lo:.3f},{hi:.3f}] / {r['plcc']:.3f}")
            records.append(dict(metric=name,corpus=corpus,**r))
        lines.append('| '+label+f' ({n}) | '+' | '.join(cells)+' |')
    lines+=['',
        '**Training overlap.** All three full-data bakes used the four D1 sources (KADID,TID,KonFiG,CID22-A). Their rows above use seed0 source-withheld E33 A/C or E34 W-C fits in raw units, not the packed full-data model. External panels use the exact packed bakes. CID22-A/B remain separate. TRAINING_OVERLAP.json marks each fit/development item; the external exact-pixel screen found0 matches. LIVE omits19 known overlapping source families. Other crop/resize lineage is not comprehensively cleared.',
        '',
        '**Method/limits.** Canonical zen_stats.panel; PLCC after registered4-parameter logistic fit. SROCC bootstrap:10000 whole-source draws,seed20261017,type7; small-source intervals are descriptive and conditional on seed0. KonFiG is the exposed reconstructed quality/JND grid (SSIM2 tuned on it); KonJND is the existing JPEG SELECT ruler,404 sources, with quality-oriented minus-PJND. BPG504 is outside that ruler. Cached peer implementations vary by corpus; exact columns/binaries are pinned. Fresh external extractor6dad3b66 has an unresolved source-commit closure, disclosed in its manifests.',
        '',
        '**Excluded.** '+ '; '.join(read(out/'EXCLUSIONS.json')['excluded'])+'.',
        '',
        'Dashboard: http://dev.lan:3300/zensim/gauntlet-2026-10-18/gauntlet.html',
        'Artifacts: '+str(out)+'. All population points are retained in JSON; the offline HTML embeds at most500 per scatter cell, with full-population density/tail diagnostics.']
    text='\n'.join(lines)+'\n'
    (out/'GAUNTLET_SUMMARY.md').write_text(text)
    write(out/'STATISTICS.json',dict(schema='frozen-gauntlet-statistics-v1',cells=records))


def live_images(source, out):
    """Inspect existing zero-human-quality points without new scoring or fitting."""
    source = source.resolve()
    names = [*MODELS, 'peer_cvvdp_fhd', 'peer_ssim2', 'peer_butter3']
    data = read(source/'SCORED_INPUTS.json')['datasets']['live']
    docs = {name: read(source/(name+'.fulleval.json')) for name in names}
    points = {name: doc['per_pair']['live'] for name, doc in docs.items()}
    target = data['target']
    for name, cols in points.items():
        if cols['mos'] != target or cols['pred'] != data['scores'][name]['pred']:
            raise ValueError('LIVE stored score/target identity mismatch: '+name)
        if len(cols['normalized_pred']) != len(target):
            raise ValueError('missing full-population normalized coordinates')
    # Explicit visual-inspection query, never an admission or qualification gate.
    # Show ALL near-zero points; white-noise cases are the first section.
    near = [i for i, y in enumerate(target) if abs(y) <= .12]
    primary = [i for i in near if Path(data['pixels'][i]['dist_path']).parent.name == 'wn']
    primary.sort(key=lambda i: -sum(points[n]['normalized_pred'][i]-target[i] for n in MODELS))
    other = [i for i in near if i not in primary]
    if not primary:
        raise ValueError('no LIVE near-zero white-noise points to inspect')
    if out.exists():
        raise ValueError('refusing to overwrite image inspection output: '+str(out))
    out.mkdir(parents=True)
    images = out/'images'; images.mkdir()
    copied = {}; cases = []
    def image(path):
        path = Path(path); digest = sha(path)
        dest = images/(digest+path.suffix.lower())
        if not dest.exists():
            shutil.copyfile(path, dest)
        if sha(dest) != digest:
            raise ValueError('copied image differs from scored source')
        copied[str(path)] = {'file':str(dest.relative_to(out)), 'sha256':digest}
        return str(dest.relative_to(out))
    for i in primary+other:
        p = data['pixels'][i]
        cases.append(dict(index=i, pair_key=p['pair_key'], row_id=p['row_id'],
            reference=Path(p['ref_path']).stem, distortion=Path(p['dist_path']).parent.name,
            filename=Path(p['dist_path']).name, target=target[i],
            primary=i in primary, ref_image=image(p['ref_path']), dist_image=image(p['dist_path']),
            scores={n:dict(raw=points[n]['pred'][i], plotted=points[n]['normalized_pred'][i]) for n in names}))
    source_files = ['SCORED_INPUTS.json', *[n+'.fulleval.json' for n in names]]
    manifest = dict(schema='gauntlet-live-image-inspection-v1',
        build_commit=subprocess.check_output(['jj','log','--no-graph','-r','@', '-T','commit_id'],text=True).strip(),
        inspection_owner_sha256=sha(__file__),
        source=str(source), source_files={n:sha(source/n) for n in source_files},
        selection='All LIVE abs(human quality) <= 0.12; white-noise rows shown first. Post-hoc visual query only.',
        population=len(target), primary=len(primary), additional=len(other), images=copied, cases=cases,
        score_note='Existing packed frozen-model predictions and existing full-population rank-to-human-quantile coordinates. No new scores or fit.')
    write(out/'CASES.json', manifest)
    esc = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>LIVE near-zero human ratings · image inspection</title><style>',
        'body{font:16px system-ui;background:#10141d;color:#e5e9f0;margin:24px auto;padding:0 20px;max-width:1400px}a{color:#90c9ff}p{line-height:1.5}h1{font-size:28px}h2{font-size:23px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.plot,article{background:#1b2230;border:1px solid #3a4557;border-radius:10px;padding:16px;margin:14px 0}svg{width:100%;height:auto}svg text{fill:#d4dbe7;font:11px system-ui}article{scroll-margin-top:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;height:auto;display:block}figcaption{margin:7px 0}.native img{width:auto;max-width:none;image-rendering:pixelated}.native figure{overflow:auto}button{padding:8px 12px;cursor:pointer;margin:6px}table{border-collapse:collapse;margin:16px 0;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px;border-bottom:1px solid #3a4557}th:first-child,td:first-child{text-align:left}.small{font-size:13px;color:#b5bfd0}.tag{color:#ffb49b}@media(max-width:650px){.pair{grid-template-columns:1fr}table{font-size:12px}th,td{padding:4px}}</style>',
        '<h1>LIVE: the points along the zero-human-quality axis</h1>',
        f'<p><b>{len(primary)} white-noise cases</b> match the cluster: all three variants place them above the bottom of the quality scale. The six panels below retain all {len(target)} LIVE pairs; red points link to images. Blue points are other near-zero cases, retained for context.</p>',
        '<p>Human quality is the dashboard’s MOS axis: <b>1 − DMOS/100</b>; lower is worse, and negative values are retained. The x coordinate reuses each metric’s stored full-population rank mapped to human-score quantiles, exactly as the dashboard’s default shape-normalized view. It is <b>not a calibrated MOS prediction</b>. Native scores are listed separately. No images, scores, fits or labels were regenerated.</p>',
        '<p><a href="../gauntlet-2026-10-18/gauntlet.html">Original dashboard</a> · <a href="CASES.json">Cases, hashes and selection rule</a></p><div class="plots">']
    lo, hi = min(target), max(target); span = hi-lo
    def xy(x, y):return 44+280*(x-lo)/span, 294-260*(y-lo)/span
    for name in names:
        parts.append('<div class="plot"><b>'+esc(name)+'</b><svg viewBox="0 0 350 340" role="img" aria-label="'+esc(name)+' versus human quality">')
        for tick in [0,.25,.5,.75,1]:
            x,y = xy(tick,tick)
            parts.append(f'<path d="M44 {y:.2f}H324 M{x:.2f} 34V294" stroke="#384254"/><text x="{x:.2f}" y="310" text-anchor="middle">{tick:g}</text><text x="38" y="{y+4:.2f}" text-anchor="end">{tick:g}</text>')
        x0,y0=xy(lo,lo);x1,y1=xy(hi,hi)
        parts.append(f'<path d="M{x0} {y0}L{x1} {y1}" stroke="#c0c9d6" stroke-dasharray="5 5"/><text x="180" y="333" text-anchor="middle">Prediction rank → human quantile</text><text x="12" y="170" transform="rotate(-90 12 170)" text-anchor="middle">Human quality</text>')
        order = [i for i in range(len(target)) if i not in near]+other+primary
        for i in order:
            x,y = xy(points[name]['normalized_pred'][i],target[i]); p=data['pixels'][i]
            color='#ff866a' if i in primary else '#86c7ff' if i in other else '#8b95a8'
            title=f"{Path(p['ref_path']).stem} / {Path(p['dist_path']).parent.name}/{Path(p['dist_path']).name}; human={target[i]:.4f}; raw={points[name]['pred'][i]:.4f}"
            circle=f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{4 if i in near else 2}" fill="{color}" opacity="{1 if i in near else .38}"><title>{esc(title)}</title></circle>'
            parts.append(f'<a href="#pair-{i}">{circle}</a>' if i in near else circle)
        parts.append('</svg></div>')
    parts.append('</div><h2>White-noise cluster</h2>')
    for k, case in enumerate(cases):
        if k == len(primary):parts.append(f'<h2>Other near-zero human ratings ({len(other)})</h2><p>These cases are shown for context, regardless of model disagreement.</p>')
        i=case['index']; ref=esc(case['reference']); dist=esc(case['distortion']+'/'+case['filename'])
        parts.append(f'<article id="pair-{i}"><h2>{ref} · {dist}</h2><p class="tag">Human quality {case["target"]:.4f} · dashboard row {i} (zero-based)</p><button onclick="this.closest(\'article\').classList.toggle(\'native\')">Toggle native pixel size</button><div class="pair">')
        for kind,label in [('ref_image','Reference'),('dist_image','Distorted')]:
            url=case[kind];parts.append(f'<figure><a href="{url}" target="_blank"><img src="{url}" loading="lazy" alt="{label}: {ref} {dist}"></a><figcaption>{label} · <a href="{url}" target="_blank">open original PNG</a></figcaption></figure>')
        parts.append('</div><table><tr><th>Metric</th><th>Native score</th><th>Plotted x</th><th>Human y</th></tr>')
        for name in names:
            score=case['scores'][name]; native=-score['raw'] if name=='peer_butter3' else score['raw']
            label=name+(' (distance ↓)' if name=='peer_butter3' else ' ↑')
            parts.append(f'<tr><td>{esc(label)}</td><td>{native:.4f}</td><td>{score["plotted"]:.4f}</td><td>{case["target"]:.4f}</td></tr>')
        parts.append('<tr><td colspan="4" class="small">CVVDP-FHD native score is 10 × JOD. Butter3 is shown as its positive 3-norm distance; plots use the quality-oriented negative distance.</td></tr></table></article>')
    parts.append('</html>')
    (out/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(primary=len(primary),additional=len(other),images=len(copied),html=str(out/'index.html'))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','score','audit-overlap','peer-containers','summary','live-images'])
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--source',type=Path,help='Existing immutable gauntlet root for live-images')
    args=parser.parse_args()
    if args.mode == 'live-images':
        if args.source is None:parser.error('live-images requires --source')
        live_images(args.source,args.out)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'score':score,'audit-overlap':audit_overlap,'peer-containers':peer_containers,'summary':summary}[args.mode](args.out)


if __name__=='__main__':main()
