"""One logged E33 noise capture; inference is delegated to the Rust panel owner.

Capture opens each of the 120 admitted stored prediction files once and only
the four E33 target columns. Recomputations use the immutable captured panel,
never the original predictions. A started marker makes this read non-repeatable.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow.parquet as pq
from v2_common import sha

E33=Path('/mnt/v/output/zensim/e33-impl-2026-10-09/assessment-e33')
ROOT=Path('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29')
FOLDS=['kadid','tid2013','konfig','cid22_a25']
COUNTS=[7869,3000,756,2192]
REFERENCES=[65,25,7,25]
PINS=['d1f37f29b9d05980b68e79272fc61c6a249a054973c44567286d0ee3a9a2f010',
      'cce93ad4dec5a77d308a2aaa0ff01c923da4e34950468001b54ce33504079c05',
      '94f5c1aa1ac562a497da19b1ba0f9cf2cc174de59189980b4c58feb21904954a',
      '32c7b77b3d75c4ed9c61914eb4f62201d7085057bcbb2880b1702d3ea30279d6']

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def capture(out):
    parity=json.loads((out/'seed-parity-check.json').read_text())
    assert parity['state']=='PASS'
    ledger=Path(__file__).resolve().parents[2]/'docs/DATA_SPLITS.md'
    assert 'E34 part 5: one-time E33 noise read' in ledger.read_text()
    summary=E33/'decision.json'
    assert sha(summary)=='e0e5e1a1689e59e9b380e003337aa44ef187cef985db9547136e8e8c816e7fef'
    decision=json.loads(summary.read_text())
    receipt=ROOT/'wide/main/real/receipt.json'
    r=json.loads(receipt.read_text())
    # Admit exact role-bound E33 tables and their label-free keys first.
    groups=[]
    for fold,n,g,pin in zip(FOLDS,COUNTS,REFERENCES,PINS,strict=True):
        rec=r['legs'][fold]['full'];p=ROOT/rec['rel']
        assert rec['rows']==n and rec['sha256']==pin
        assert p==ROOT/f'wide/main/real/{fold}.parquet'
        assert sha(Path(str(p)+'.manifest.json'))==rec['manifest_sha256']
        keys=p.with_suffix('.keys.parquet');kb=keys.read_bytes()
        assert hashlib.sha256(kb).hexdigest()==rec['keys_sha256']
        k=pq.read_table(io.BytesIO(kb),columns=['ref_basename','pair_key','member_set','source_row_id'])
        # E33 retains 7,869 KADID observations, including three repeated
        # image-pair keys. The frozen physical row is the observation identity.
        physical=list(zip(k['member_set'].to_pylist(),k['source_row_id'].to_pylist(),strict=True))
        assert len(k)==n and len(set(physical))==n
        names=k['ref_basename'].to_pylist()
        if fold=='konfig':names=[x.removesuffix('_parta').removesuffix('_partb') for x in names]
        ids={name:i for i,name in enumerate(sorted(set(names)))}
        assert len(ids)==g,(fold,len(ids),g)
        groups.append((fold,p,pin,k,[ids[x] for x in names]))
    reads=dict(schema='e34-e33-noise-read-v1',state='STARTED',
        ledger_sha256=sha(ledger),parity=parity,summary_sha256=sha(summary),
        source_receipt_sha256=sha(receipt),source_script_sha256=sha(Path(__file__)),
        scope='only stored E33 A/C/control E21 predictions and their four admitted E21 target columns; no fit or margin change')
    write(out/'noise-read-started.json',reads)
    datasets=[];payloads=[]
    for fold,path,pin,keys,reference in groups:
        blob=path.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==pin
        target=pq.read_table(io.BytesIO(blob),columns=['human_score'])['human_score'].to_pylist()
        del blob
        predictions={}
        for arm in ['a','c','control']:
            values=[]
            for seed in range(10):
                rel=f'{arm}/{fold}_s{seed}/pred.tsv'
                blob=(E33/rel).read_bytes()  # one original prediction payload open
                digest=hashlib.sha256(blob).hexdigest()
                assert digest==decision['artifacts'][rel],rel
                df=pd.read_csv(io.BytesIO(blob),sep='\t',float_precision='round_trip')
                assert df.row_idx.tolist()==list(range(len(target)))
                assert df.pred.notna().all()
                values.append(df.pred.tolist())
                payloads.append(dict(path=str(E33/rel),sha256=digest,rows=len(df)))
            predictions[arm]=values
        datasets.append(dict(name=fold,target=target,reference=reference,
            # Codec is unused by the noise-only pooled/reference statistic.
            codec=[0]*len(target),orientation=1.,predictions=predictions))
    data=dict(schema='e34-paired-seed-v1',datasets=datasets,
        tests=[dict(index=i+1,arm=arm,control='control',eligibility=True) for i,arm in enumerate(['a','c'])])
    write(out/'noise-input.json',data)
    write(out/'noise-capture.json',dict(**reads,payloads=payloads,
        capture_state='COMPLETE',input_sha256=sha(out/'noise-input.json'),
        observations=sum(COUNTS),references=sum(REFERENCES),seeds=10,
        codec_field='unused sentinel; no pair-class or W2 calculation in noise mode'))
    print(json.dumps(dict(capture='COMPLETE',prediction_payloads=len(payloads),observations=sum(COUNTS),references=sum(REFERENCES))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['capture','compute'])
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--panel',type=Path)
    a=p.parse_args()
    if a.mode=='capture':capture(a.out)
    else:
        receipt=json.loads((a.out/'noise-capture.json').read_text())
        assert receipt['capture_state']=='COMPLETE'
        assert sha(a.out/'noise-input.json')==receipt['input_sha256']
        assert a.panel is not None
        write(a.out/'noise-compute-started.json',dict(panel_sha256=sha(a.panel),input_sha256=receipt['input_sha256']))
        subprocess.run([str(a.panel),'e34','noise','--input',str(a.out/'noise-input.json'),
            '--output',str(a.out/'noise-result.json')],check=True)

if __name__=='__main__':main()
