"""Finish the admitted E34 pixel-only build; all artifacts are exclusive-create."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from e34_augmentation import sha, write
from v2_common import fx1_declaration, WIDTH

ROOT=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2')

def development(root):
    out=root/'development';out.mkdir()
    rows=[]
    for s in json.loads((root/'encodes/development-sources.json').read_text()):
        for index,d in [(10,.5),(13,1.6168174743652344),(17,7.731237411499023)]:
            rows.append(dict(s,scalar_index=index,distance=d,encoded=str(out/f'{s["origin"]}-{index}.jxl'),distorted=str(out/f'{s["origin"]}-{index}.png')))
    job=dict(output=str(out/'bases.jsonl'),rows=rows)
    write(out/'job.json',job)
    subprocess.run(['tools/e34-jxl-scalar/target/debug/e34-jxl-scalar',str(out/'job.json')],check=True)
    bases=list(map(json.loads,(out/'bases.jsonl').read_text().splitlines()))
    cases=[dict(s,**b,block_size=size,case_id=f'{s["origin"]}-{s["scalar_index"]}-{size}') for s,b in zip(rows,bases) for size in [8,16,32,64]]
    assert len(cases)==96
    write(out/'panel.json',dict(schema='e34-development-pixels-v1',labels_read=False,cases=cases,selection_sha256=sha(root/'selection.json'),extractor_tier='v3',scalar_encoder_sha256=sha('tools/e34-jxl-scalar/target/debug/e34-jxl-scalar')))
    print(json.dumps(dict(development_cases=len(cases))),flush=True)

def dev_contract(root):
    panel=json.loads((root/'development/panel.json').read_text())
    refs={Path(r['destination']).stem:r for r in map(json.loads,(root/'references.jsonl').read_text().splitlines())}
    cases=[]
    for row in panel['cases']:
        cases.append(dict(key=row['case_id'],reference=row['reference'],reference_sha256=row['reference_sha256'],
            reference_pixels_sha256=refs[row['origin']]['pixels_sha256'],distorted=row['distorted'],
            distorted_sha256=row['distorted_sha256'],distorted_pixels_sha256=row['decoded_pixels_sha256'],
            block=row['block_size'],origin=row['origin'],scalar_index=row['scalar_index'],objective='served'))
    assert len(cases)==len({r['key'] for r in cases})==96
    write(root/'development/assessment-inputs.json',dict(schema='e34-development-assessment-inputs-v1',cases=cases,
        labels_read=False,tier='v3',formula_revision=5,panel_sha256=sha(root/'development/panel.json'),
        model_binding='deferred until authorized lambda pilot',sensitivity_method='pending owner decision D12'))

def repair_job(root):
    out=root/'repairs';out.mkdir()
    rows=json.loads((root/'encodes/augmentation/bases.json').read_text())
    rng=np.random.default_rng(20261011)
    for r in rows:
        blocks=[]
        for size in [8,16,32,64]:
            positions=[(x,y,size) for y in range(0,r['height'],size) for x in range(0,r['width'],size)]
            ids=sorted(rng.choice(len(positions),min(16,len(positions)),replace=False).tolist())
            blocks.extend(positions[i] for i in ids)
        r['blocks']=blocks
    model=json.loads(Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part4/full-job.json').read_text())['models'][1]
    decl=fx1_declaration()
    ids=sorted(set(decl['direct'])|{i for pair in decl['products'] for i in pair})
    job=dict(mode='repairs',output=str(out/'features.jsonl'),model=model['path'],model_sha256=model['sha256'],feature_ids=ids,width=WIDTH,rows=rows)
    write(out/'job.json',job)
    print(json.dumps(dict(bases=len(rows),sampled_repairs=sum(len(r['blocks']) for r in rows),feature_ids=len(ids))),flush=True)

def table(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out=root/'repairs';path=out/'augmentation.parquet';assert not path.exists()
    schema=pa.schema([('base_id',pa.uint32()),('block_size',pa.uint32()),('base_features',pa.list_(pa.float32())),('repair_features',pa.list_(pa.float32()))])
    job=json.loads((out/'job.json').read_text())
    expected={(r['base_id'],size,x,y) for r in job['rows'] for x,y,size in r['blocks']}
    seen=set();base_hashes={}
    counts={};dropped=0;batch=[];n=0
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        def flush():
            writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch.clear()
        with (out/'features.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                key=(r['base_id'],r['block_size'],*r['block'])
                assert key in expected and key not in seen,key
                seen.add(key)
                a=np.asarray(r['base_features'],dtype=np.float32);b=np.asarray(r['repair_features'],dtype=np.float32)
                assert len(a)==len(b)==WIDTH and np.isfinite(a).all() and np.isfinite(b).all()
                digest=hashlib.sha256(a.tobytes()).hexdigest()
                assert base_hashes.setdefault(r['base_id'],digest)==digest
                identical=np.array_equal(a.view(np.uint32),b.view(np.uint32))
                assert identical==r['drop_bit_identical']
                if identical:dropped+=1;continue
                batch.append({k:r[k] for k in schema.names});n+=1;counts[r['base_id']]=counts.get(r['base_id'],0)+1
                if len(batch)==256:flush()
        if batch:flush()
    assert seen==expected and len(base_hashes)==1536,(len(seen),len(expected),len(base_hashes))
    manifest=dict(schema='e34-steer-pairs-v1',formula_revision=5,tier='v3',labels_read=False,build_complete=len(counts)==1536,base_images=1536,origins=128,width=WIDTH,repair_rows=n,sampled_rows=len(seen),bit_identical_dropped=dropped,represented_bases=len(counts),eligibility_receipt_sha256=sha(root/'eligibility.json'),extractor_sha256=json.loads((out/'extraction-binding.json').read_text())['extractor_sha256'],build_commit=json.loads((out/'extraction-binding.json').read_text())['build_commit'],table_sha256=sha(path),job_sha256=sha(out/'job.json'),features_sha256=sha(out/'features.jsonl'))
    write(path.with_suffix('.parquet.manifest.json'),manifest)
    print(json.dumps(manifest),flush=True)
    assert len(counts)==1536,'some registered bases have no nonidentical repair rows'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['development','dev-contract','repair-job','table']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    {'development':development,'dev-contract':dev_contract,'repair-job':repair_job,'table':table}[a.mode](a.root)
if __name__=='__main__':main()
