"""Streaming native-receipt to feature-only E36 pools; no labels or fitting."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from e36_data import Q, PREFIX, canonical
from e36_store import sha
from e36_matrix import admit as admit_matrix
from e36_semantic import required_strata,admit as semantic_admit
sys.path.insert(0,str(Path(__file__).parents[1]/'canonical_corpus'))
from build_ladder_tv_pairs import e36_pairs


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def outputs(root,name):
    run=root/'runs'/name
    census=json.loads((run/'census.json').read_text())
    if census.get('admitted') is not True:raise ValueError('incomplete predeclared matrix')
    expected={j['kind']['view'] for j in json.loads((run/'manifest.json').read_text())}
    seen=set()
    for item in census['output_receipts']:
        p=Path(item['path'])
        if sha(p)!=item['sha256']:raise ValueError('native worker output digest changed')
        d=json.loads(gzip.decompress(p.read_bytes()));key=PREFIX+canonical(d['view'])
        if not d['complete'] or key not in expected or key in seen:raise ValueError('native declared/completed view mismatch')
        if d['view']['role']!='bulk' or d['view']['q_grid']!=Q:raise ValueError('diagnostic or thinned output is not bulk')
        seen.add(key);yield item['sha256'],d
    if seen!=expected:raise ValueError('missing whole registered ladders')


def build(root,names,out,commit,archive_pending=False):
    matrix_admission=admit_matrix(root,names,archive_pending=True) if archive_pending else admit_matrix(root,names)
    out.mkdir();write(out/'matrix-admission.json',matrix_admission);db=sqlite3.connect(out/'physical-index.sqlite3')
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE features(id INTEGER PRIMARY KEY,ref TEXT,enc TEXT,split TEXT,vec BLOB,distpix TEXT,UNIQUE(ref,enc));
    CREATE TABLE pairs(a INTEGER,b INTEGER,class INTEGER,band INTEGER,codec TEXT,content TEXT,size INTEGER,split TEXT,mode TEXT,qlo INTEGER,qhi INTEGER,source TEXT,priority TEXT,PRIMARY KEY(a,b));
    CREATE UNIQUE INDEX unordered_pair ON pairs(min(a,b),max(a,b));
    CREATE TABLE aliases(a INTEGER,b INTEGER,class INTEGER,band INTEGER,codec TEXT,content TEXT,size INTEGER,split TEXT,mode TEXT,qlo INTEGER,qhi INTEGER,source TEXT,origin TEXT,family TEXT);
    ''')
    ids=json.loads((root/'FEATURE_LAYOUT_BINDING.json').read_text())['consumed_ids']
    declaration=json.loads((root/'runs'/names[0]/'declaration.json').read_text())
    runtime_path=Path(declaration['runtime'])
    if sha(runtime_path)!=declaration['runtime_sha256']:raise ValueError('runtime declaration changed')
    runtime_spec=json.loads(runtime_path.read_text())
    references_path=Path(runtime_spec['references'])
    budget_path=Path(runtime_spec.get('source_budget',root/'source-budget/plan.json'))
    if matrix_admission.get('effective_source_budget'):
        budget_path=Path(matrix_admission['effective_source_budget']['path'])
        if sha(budget_path)!=matrix_admission['effective_source_budget']['sha256']:raise ValueError('effective source-count budget changed')
    references=json.loads(references_path.read_text());refindex={(r['origin'],r['size']):r for r in references['rows']};refbykey={r['reference_key']:r for r in references['rows']}
    mode_keys={}
    for codec,digest in runtime_spec['mode_inventory_sha256'].items():
        path=Path(runtime_spec['mode_inventory'])/(codec+'.jsonl')
        if sha(path)!=digest:raise ValueError('native mode inventory changed')
        mode_keys[codec]={json.loads(line)['mode_key'] for line in path.read_text().splitlines()}
    # For this frozen universe a PNG hash is an exact surrogate for the
    # dimension-qualified RGB pixel identity. Refuse any ambiguous surrogate,
    # including a cross-split collision, before joining a single data row.
    pixel_to_file={};file_to_pixel={}
    for r in references['rows']:
        if r['split'] not in ['fit','selection']:continue
        key=(r['width'],r['height'],r['pixels_sha256'])
        prior=pixel_to_file.setdefault(key,(r['sha256'],r['split']))
        if prior!=(r['sha256'],r['split']):raise ValueError('reference PNG/pixel identity is not bijective within the admitted splits')
        prior=file_to_pixel.setdefault(r['sha256'],key)
        if prior!=key:raise ValueError('one reference PNG names incompatible pixels')
    mode_coverage=defaultdict(lambda:defaultdict(int));raw_counts=defaultdict(int);raw_strata=defaultdict(lambda:defaultdict(int));effective=defaultdict(int);effective_mass=defaultdict(int);producer_outputs=[];runtime=None
    for name in names:
        for receipt_sha,doc in outputs(root,name):
            if doc['view']['codec'] in matrix_admission.get('excluded_codecs',[]):
                raise ValueError('owner-excluded codec cannot enter the E36 pool')
            if runtime is None:runtime=doc['runtime_sha256']
            if doc['runtime_sha256']!=runtime:raise ValueError('mixed native runtimes')
            producer_outputs.append(receipt_sha);rows=doc['rows'];local=[];rowmap=[];declared_ref=refbykey[doc['view']['reference_key']]
            for r in rows:
                ref=refindex[(r['origin'],r['size'])]
                if ref!=declared_ref or r['codec']!=doc['view']['codec'] or r['panel']!=doc['view']['panel']:
                    raise ValueError('native row differs from its exact declared source/codec/panel')
                if r['reference_sha256']!=ref['sha256'] or r['reference_pixels_sha256']!=ref['pixels_sha256'] or (r['width'],r['height'])!=(ref['width'],ref['height']) or r['split']!=ref['split'] or r['split'] not in ['fit','selection'] or r['feature_ids']!=ids:
                    raise ValueError('feature/split binding changed')
                if r['origin']!=ref['origin'] or r['family']!=ref['family'] or r['size']!=ref['size']:
                    raise ValueError('native row provenance changed')
                values=np.asarray(r['features'],dtype='<f4')
                if values.size!=420 or not np.isfinite(values).all() or not np.array_equal(values.astype(np.float64),np.asarray(r['features'],dtype=np.float64)):
                    raise ValueError('nonfinite or nonnative-f32 consumed features')
                vector=values.tobytes();key=(r['reference_sha256'],r['encoded_sha256'])
                found=db.execute('SELECT id,split,vec,distpix FROM features WHERE ref=? AND enc=?',key).fetchone()
                if found:
                    idx,split,prior,prior_pixels=found
                    if split!=r['split'] or prior!=vector or prior_pixels!=r['distorted_pixels_sha256']:raise ValueError('same physical pixels have inconsistent features/split')
                else:
                    idx=db.execute('INSERT INTO features(ref,enc,split,vec,distpix) VALUES(?,?,?,?,?)',(*key,r['split'],vector,r['distorted_pixels_sha256'])).lastrowid
                rowmap.append(idx)
                x=dict(r);x['knobs']=json.loads(r['mode_key']);local.append(x)
            ladder_groups=defaultdict(list)
            for r in local:ladder_groups[r['mode_key']].append(r['q'])
            if any(sorted(qs)!=Q for qs in ladder_groups.values()):raise ValueError('incomplete or duplicate requested quality ladder')
            if doc['view']['panel']=='mode-panel':
                if set(ladder_groups)!=mode_keys[doc['view']['codec']]:raise ValueError('omitted/extra native mode group')
                for r in local:
                    key=canonical([r['codec'],r['mode_key'],r['size'],r['split']]);mode_coverage[key]['logical_cells']+=1;mode_coverage[key]['physical_alias_cells']+=bool(r['physical_alias'])
            elif len(ladder_groups)!=1:raise ValueError('baseline changed configuration across quality')
            pairs,counts=e36_pairs(local)
            for k,v in counts.items():
                if isinstance(v,int):raw_counts[k]+=v
            for key,values in counts['by_stratum'].items():
                for metric,count in values.items():raw_strata[key][metric]+=count
            for p in pairs:
                lo,hi=p['worse_row'],p['better_row'];a,b=rowmap[lo],rowmap[hi]
                if a==b:raise ValueError('nonzero native teacher margin on identical encoded pair')
                x,y=local[lo],local[hi]
                if x['reference_sha256']!=y['reference_sha256'] or x['codec']!=y['codec'] or x['mode_key']!=y['mode_key'] or x['size']!=y['size']:
                    raise ValueError('unrelated preference endpoints')
                lowq,highq=sorted([x['q'],y['q']]);lag=Q.index(highq)-Q.index(lowq)
                if lag not in [1,2,4,8,16]:raise ValueError('unregistered preference lag')
                weightclass=0 if p['weight']==2 else 1
                db.execute('INSERT INTO aliases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (a,b,weightclass,p['band'],x['codec'],x['content'],x['size'],x['split'],x['mode_key'],lowq,highq,receipt_sha,x['origin'],x['family']))
                # No repeated physical edge mass. Prefer the preregistered
                # stronger codec weight, then lexical metadata; no teacher
                # magnitude or model prediction chooses the representative.
                priority=canonical([weightclass,x['codec'],x['mode_key'],lowq,highq])
                key=(min(a,b),max(a,b));found=db.execute('SELECT a,b,priority FROM pairs WHERE min(a,b)=? AND max(a,b)=?',key).fetchone()
                if found:
                    if (found[0],found[1])!=(a,b):raise ValueError('contradictory preferred physical edge')
                    if found[2]<=priority:continue
                    db.execute('DELETE FROM pairs WHERE a=? AND b=?',(a,b))
                db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(a,b,weightclass,p['band'],x['codec'],x['content'],x['size'],x['split'],x['mode_key'],lowq,highq,receipt_sha,priority))
            db.commit()
        print(canonical(dict(completed_run=name,physical_features=db.execute('SELECT count(*) FROM features').fetchone()[0],unique_edges=db.execute('SELECT count(*) FROM pairs').fetchone()[0])),flush=True)
    # Pair lookup uses a canonical unordered index; retain the complete index as
    # evidence, including features not consumed by any admitted preference.
    expected=defaultdict(int)
    for ref,codec,content,size,split,band,count in db.execute('SELECT f.ref,p.codec,p.content,p.size,p.split,p.band,count(*) FROM pairs p JOIN features f ON f.id=p.a GROUP BY f.ref,p.codec,p.content,p.size,p.split,p.band'):
        key=canonical([codec,content,size,band,split]);effective[key]+=count
        effective_mass[key]+=count*(2 if codec in ['jpeg','rav1e'] else 1)
    matrix=matrix_admission['matrix']
    budget=json.loads(budget_path.read_text())
    expected=required_strata(budget,matrix,matrix_admission.get('excluded_codecs',()))
    missing=[k for k in expected if effective[k]==0]
    write(out/'whole-matrix.json',dict(schema='e36-whole-matrix-completion-v1',matrix=matrix,
        runtime_sha256=runtime,source_budget_sha256=sha(budget_path),outputs=producer_outputs,complete=True,build_commit=commit,
        admission=dict(path=str(out/'matrix-admission.json'),sha256=sha(out/'matrix-admission.json'))))
    feature_schema=pa.schema([('row_idx',pa.uint64()),('reference_sha256',pa.string()),('encoded_sha256',pa.string()),('features',pa.list_(pa.float32()))])
    pair_schema=pa.schema([('worse_idx',pa.uint64()),('better_idx',pa.uint64()),('pair_weight_class',pa.uint8()),('q_band',pa.uint8())])
    proof_schema=pa.schema([('worse_idx',pa.uint64()),('better_idx',pa.uint64()),('reference_sha256',pa.string()),('codec',pa.string()),('size',pa.uint32()),('mode_key',pa.string()),('q_low',pa.uint8()),('q_high',pa.uint8()),('worse_encoded_sha256',pa.string()),('better_encoded_sha256',pa.string()),('consensus_receipt_sha256',pa.string())])
    for split in ['fit','selection']:
        target=out/split;target.mkdir();db.execute('DROP TABLE IF EXISTS export_ids');db.execute('CREATE TEMP TABLE export_ids(old INTEGER PRIMARY KEY,new INTEGER)')
        used=db.execute('SELECT id,ref,enc,vec FROM features WHERE split=? AND id IN (SELECT a FROM pairs UNION SELECT b FROM pairs) ORDER BY ref,enc',(split,))
        batch=[];n=0
        with pq.ParquetWriter(target/'features.parquet',feature_schema,compression='zstd') as writer:
            for old,ref,enc,vec in used:
                db.execute('INSERT INTO export_ids VALUES(?,?)',(old,n));batch.append(dict(row_idx=n,reference_sha256=ref,encoded_sha256=enc,features=np.frombuffer(vec,dtype='<f4').tolist()));n+=1
                if len(batch)==256:writer.write_table(pa.Table.from_pylist(batch,schema=feature_schema));batch=[]
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=feature_schema))
        count=0;pb=[];eb=[]
        with pq.ParquetWriter(target/'pairs.parquet',pair_schema,compression='zstd') as pw,pq.ParquetWriter(target/'edge-proof.parquet',proof_schema,compression='zstd') as ew:
            query='SELECT p.*,ea.new,eb.new,fa.ref,fa.enc,fb.enc FROM pairs p JOIN export_ids ea ON p.a=ea.old JOIN export_ids eb ON p.b=eb.old JOIN features fa ON p.a=fa.id JOIN features fb ON p.b=fb.id WHERE p.split=? ORDER BY fa.ref,fa.enc,fb.enc'
            for row in db.execute(query,(split,)):
                a,b,cl,band,codec,content,size,role,mode,ql,qh,source,priority,newa,newb,ref,enca,encb=row
                pb.append(dict(worse_idx=newa,better_idx=newb,pair_weight_class=cl,q_band=band))
                eb.append(dict(worse_idx=newa,better_idx=newb,reference_sha256=ref,codec=codec,size=size,mode_key=mode,q_low=ql,q_high=qh,worse_encoded_sha256=enca,better_encoded_sha256=encb,consensus_receipt_sha256=source));count+=1
                if len(pb)==256:pw.write_table(pa.Table.from_pylist(pb,schema=pair_schema));ew.write_table(pa.Table.from_pylist(eb,schema=proof_schema));pb=[];eb=[]
            if pb:pw.write_table(pa.Table.from_pylist(pb,schema=pair_schema));ew.write_table(pa.Table.from_pylist(eb,schema=proof_schema))
        bind=lambda p:dict(path=str(p),sha256=sha(p))
        provenance=dict(references=bind(references_path),source_budget=bind(budget_path),cohort=bind(root/'cohort-selection.json'))
        provenance['runtime']=bind(runtime_path)
        derived=target/'fx1.json';derived.write_bytes(Path(__file__).with_name('e33_fx1_declaration.json').read_bytes())
        report=dict(schema='e36-required-support-v1',admitted=not missing,zero_support_required_strata=len(missing),missing=missing,
            features_sha256=sha(target/'features.parquet'),pairs_sha256=sha(target/'pairs.parquet'),edge_proof_sha256=sha(target/'edge-proof.parquet'),
            source_budget_sha256=provenance['source_budget']['sha256'],references_sha256=provenance['references']['sha256'],cohort_sha256=provenance['cohort']['sha256'],runtime_sha256=runtime,
            derived_sha256=sha(derived),feature_rows=n,pair_rows=count,teachers=['butteraugli_pnorm3','cvvdp@standard_fhd'],teacher_margins=[.01,.01],
            required_strata={k:dict(selected_references=v,unique_preferences=effective[k]) for k,v in expected.items()},
            whole_matrix=bind(out/'whole-matrix.json'),build_commit=commit)
        write(target/'support.json',report)
        manifest=dict(schema='e36-tv-feature-only-v1',protocol_build='e36_tv_v1',labels_read=False,fits=0,complete=True,split=split,zero_dropped_pairs=True,mandatory_matrix_complete=True,
            required_support=bind(target/'support.json'),features=bind(target/'features.parquet'),pairs=bind(target/'pairs.parquet'),edge_proof=bind(target/'edge-proof.parquet'),derived=bind(derived),
            feature_ids=ids,raw_width=1853,feature_rows=n,pair_rows=count,**provenance,source_budget_sha256=provenance['source_budget']['sha256'],generation_runtime_sha256=runtime)
        write(target/'manifest.json',manifest)
        semantic=semantic_admit(db,target,split,expected,effective,out/'whole-matrix.json') if not missing else None
        write(target/'contract.json',dict(schema='e36-tv-locked-contract-v1',manifest_sha256=sha(target/'manifest.json'),protocol_build='e36_tv_v1',fits_released=False,semantic_export=semantic,matrix_admission=dict(path=str(out/'matrix-admission.json'),sha256=sha(out/'matrix-admission.json'))))
    alias_schema=pa.schema([(n,pa.string()) for n in ['reference_sha256','worse_encoded_sha256','better_encoded_sha256','codec','content','split','mode_key','source_receipt_sha256','origin','family']]+[(n,pa.uint32()) for n in ['size','q_low','q_high','pair_weight_class','q_band']])
    alias_batch=[];alias_count=0
    with pq.ParquetWriter(out/'alias-memberships.parquet',alias_schema,compression='zstd') as writer:
        query='SELECT fa.ref,fa.enc,fb.enc,a.codec,a.content,a.split,a.mode,a.source,a.origin,a.family,a.size,a.qlo,a.qhi,a.class,a.band FROM aliases a JOIN features fa ON a.a=fa.id JOIN features fb ON a.b=fb.id ORDER BY fa.ref,fa.enc,fb.enc,a.codec,a.mode,a.qlo,a.qhi'
        for row in db.execute(query):
            alias_batch.append(dict(zip(alias_schema.names,row)));alias_count+=1
            if len(alias_batch)==256:writer.write_table(pa.Table.from_pylist(alias_batch,schema=alias_schema));alias_batch=[]
        if alias_batch:writer.write_table(pa.Table.from_pylist(alias_batch,schema=alias_schema))
    if alias_count!=raw_counts['admitted']:raise ValueError('lost physical preference alias membership')
    db.close()
    write(out/'_MANIFEST.json',dict(schema='e36-native-feature-pools-v1',build_commit=commit,labels_read=False,fits=0,matrix=matrix,required_support_admitted=not missing,
        mode_coverage={k:dict(v) for k,v in mode_coverage.items()},
        raw_preference_counts=dict(raw_counts),raw_proposals_by_stratum={k:dict(v) for k,v in raw_strata.items()},
        unique_effective_by_stratum={k:dict(preferences=v,effective_mass=effective_mass[k]) for k,v in effective.items()},
        zero_support_required_strata=missing,physical_dedup='highest codec weight then lexical codec/mode/q; never teacher magnitude',
        source_budget_sha256=sha(budget_path),runtime_sha256=runtime,whole_matrix_sha256=sha(out/'whole-matrix.json'),
        alias_memberships=alias_count,alias_memberships_sha256=sha(out/'alias-memberships.parquet'),
        reference_pixel_bijection_admitted=True))
    print(canonical(dict(matrix=matrix,missing_required=len(missing),fits=0)),flush=True)
    if missing:raise ValueError('zero post-dedup consensus support in required strata; no fit pool qualifies')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='append',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--prepare-while-archiving',action='store_true',help='nonlaunchable feature preparation; final artifact admission/rebinding still required');a=p.parse_args();build(a.root,a.run,a.out,a.commit,a.prepare_while_archiving)
