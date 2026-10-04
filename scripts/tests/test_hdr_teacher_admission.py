"""Behavioral integration fixtures for the actual HDR teacher table owner."""
import copy,hashlib,json,os,subprocess,tempfile
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq
repo=Path(__file__).resolve().parents[2];builder=repo/'scripts/hdr/build_hdr_train_parquets.py'
scratch=Path(os.environ.get('TMPDIR',str(Path.home()/'tmp')));scratch.mkdir(parents=True,exist_ok=True)
r=Path(tempfile.mkdtemp(prefix='hdr-teacher-check-',dir=scratch))
def pin(path):
 with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(path),sha256=h)
def js(name,value):
 p=r/name;p.write_text(json.dumps(value));return pin(p)
prereg={'agree_rule':{'tolerance_positions':1.0,'minimum_group_rows':2,'frozen_before_hdrvdp3_labels':True},'input_contract':'synthetic-fixture'};prov={'zenmetrics_commit':'synthetic-test-fixture','zenmetrics_binary_sha256':'fixture-bin'};hashes={};sets={};sc=[]
for role in ['train','val']:
 meta=[]
 for i in range(3):
  ref=f'/{role}/native-ref.png';dist=f'/{role}/native-{i}.jxl';hashes[ref]='test-ref';hashes[dist]=f'test-dist-{i}'
  x=dict(row_id=i,role=role,source_family=role,ref_path=ref,dist_path=dist,q=str(15+i),image_path=f'original/{role}/{i}',codec='zenjxl',knob_tuple_json='{}')
  if role=='val':x.update(cvvdp=[7.,8.,9.][i],target=[.3,.5,.7][i])
  meta.append(x);sc.append(dict(image_path=f'hdrteach://{role}/{i}',codec='zenjxl',q=15+i,knob_tuple_json='{}',runtime='unknown',score=([7.,7.,9.] if role=='train' else [9.,8.,7.])[i]))
 auth=r/f'{role}-authority';auth.write_bytes(b'synthetic fixture authority')
 spec=dict(metadata=js(f'{role}-metadata.json',meta),rows=3,authority=pin(auth),cvvdp_label_era='fixture')
 if role=='train':
  cv=[dict(row_id=x['row_id'],image_path=x['image_path'],codec=x['codec'],q=int(x['q']),knob_tuple_json='{}',jod=[8.,7.,9.][i]) for i,x in enumerate(meta)];p=r/'cv.parquet';pq.write_table(pa.Table.from_pylist(cv[::-1]),p);spec.update(cvvdp=pin(p),cvvdp_column='jod')
 sets[role]=spec
raw=r/'raw.parquet';pq.write_table(pa.Table.from_pylist(sc[::-1]),raw)
manifest=dict(study='HDRTEACH-2026-10-04',sets=sets,preregistration=js('prereg.json',prereg),provenance=js('prov.json',prov),input_hashes=js('hashes.json',hashes),score_column='score',shards=[{**pin(raw),'host':'fixture','binary_sha256':'fixture-bin'}])
def run(name,man,want_ok):
 m=r/(name+'.json');m.write_text(json.dumps(man));out=r/name
 p=subprocess.run(['python3',str(builder),'--teacher-manifest',str(m),'--teacher-output-dir',str(out)],capture_output=True,text=True)
 assert (p.returncode==0)==want_ok,(name,p.stdout,p.stderr)
 return out
out=run('valid',manifest,True);tr=pq.read_table(out/'hdrteach_train.parquet').to_pylist();va=pq.read_table(out/'hdrteach_val.parquet').to_pylist()
assert [x['hdrvdp3_q_jod'] for x in tr]==[7,7,9];assert [x['rank_residual_positions'] for x in tr]==[-.5,.5,0];assert all(x['agree'] for x in tr);assert [x['agree'] for x in va]==[False,True,False];assert 'historic_cvvdp_mix' in va[0] and 'target' not in va[0]
# The preregistered <=1 position rule includes its exact boundary.
boundary=[dict(x,score=[7.,8.,9.][int(x['image_path'].rsplit('/',1)[1])]) if '/train/' in x['image_path'] else x for x in sc]
p=r/'boundary.parquet';pq.write_table(pa.Table.from_pylist(boundary),p);m=copy.deepcopy(manifest);m['shards'][0].update(pin(p));b=run('boundary',m,True)
br=pq.read_table(b/'hdrteach_train.parquet').to_pylist();assert [x['rank_residual_positions'] for x in br]==[-1,1,0];assert all(x['agree'] for x in br)
# The metric's unbounded-below q_jod is retained, rather than clipped/dropped.
negative=[dict(x,score=-2.0) if x['image_path']=='hdrteach://val/0' else x for x in sc]
p=r/'negative.parquet';pq.write_table(pa.Table.from_pylist(negative),p);m=copy.deepcopy(manifest);m['shards'][0].update(pin(p));b=run('negative',m,True)
assert pq.read_table(b/'hdrteach_val.parquet')['hdrvdp3_q_jod'].to_pylist()[0]==-2.0
# Missing/duplicate/extra key, wrong q, nonfinite score, binary mixing and hash mutation fail closed.
for name,rows in [('missing',sc[:-1]),('duplicate',sc+[sc[0]]),('extra',sc+[dict(sc[0],image_path='hdrteach://train/999')]),('wrong-q',[dict(x,q=999) if i==0 else x for i,x in enumerate(sc)]),('nan',[dict(x,score=float('nan')) if i==0 else x for i,x in enumerate(sc)])]:
 p=r/(name+'.parquet');pq.write_table(pa.Table.from_pylist(rows),p);m=copy.deepcopy(manifest);m['shards'][0].update(pin(p));run(name,m,False)
m=copy.deepcopy(manifest);m['shards'][0]['binary_sha256']='different';run('mixed-binary',m,False)
m=copy.deepcopy(manifest);m['sets']['train']['metadata']['sha256']='wrong';run('hash-mismatch',m,False)
m=copy.deepcopy(manifest);m['sets']['test']=m['sets'].pop('val');run('forbidden-role',m,False)
# The end-to-end panel reads our real tables and the canonical signed-SROCC binary.
import os
p=subprocess.run(['python3',str(repo/'scripts/hdr/hdr_route_panel.py'),'--teacher-parquet',str(out/'hdrteach_train.parquet'),'--teacher-parquet',str(out/'hdrteach_val.parquet'),'--teacher-output-dir',str(r/'panel')],env=os.environ.copy(),capture_output=True,text=True)
assert p.returncode==0,(p.stdout,p.stderr);stats=json.loads((r/'panel/agreement.json').read_text());assert stats['val']['pooled_srocc_signed']==-1.0;assert stats['train']['pooled_srocc_signed']>0;assert stats['val']['agree_count']==1
print('PASS keyed shuffled join, tied ranks, exact tolerance boundary, negative target retention, original VAL target, 8 fail-closed controls, canonical signed panel',r)
(r/'JOIN_CHECK.json').write_text(json.dumps({'status':'PASS','fixture_dir':str(r),'negative_controls':8,'population_counts':{'train':3,'val':3}})+'\n')
