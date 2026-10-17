"""Private export admission against the index built from native consensus receipts."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from e36_store import sha


def required_strata(budget,matrix,excluded_codecs=()):
    from collections import defaultdict
    from e36_data import canonical
    expected=defaultdict(int)
    for label,panel in budget['panels'].items():
        if ':' not in label:continue
        size,group=label.split(':')
        for key in panel[matrix]:
            r=budget['references'][key]
            for codec in (['rav1e'] if group=='rav1e' else ['jpeg','svt','webp','jxl']):
                if codec in excluded_codecs:continue
                for band in range(3):expected[canonical([codec,r['content'],int(size),band,r['split']])]+=1
    return dict(expected)


def exact_support(report,expected,effective):
    if set(report['required_strata'])!=set(expected):raise ValueError('omitted/extra required stratum')
    for key,count in expected.items():
        if report['required_strata'][key]!=dict(selected_references=count,unique_preferences=effective.get(key,0)):
            raise ValueError('required support differs from native index')
    if any(effective.get(k,0)==0 for k in expected):raise ValueError('zero required native consensus support')


def edge(db,proof,cl,band,split):
    row=db.execute('''SELECT p.class,p.band,p.codec,p.size,p.split,p.mode,p.qlo,p.qhi,p.source
        FROM pairs p JOIN features a ON a.id=p.a JOIN features b ON b.id=p.b
        WHERE a.ref=? AND b.ref=? AND a.enc=? AND b.enc=?''',
        (proof['reference_sha256'],proof['reference_sha256'],proof['worse_encoded_sha256'],proof['better_encoded_sha256'])).fetchone()
    expected=(cl,band,proof['codec'],proof['size'],split,proof['mode_key'],proof['q_low'],proof['q_high'],proof['consensus_receipt_sha256'])
    if row!=expected:raise ValueError('exported edge direction/mode/weight is absent from native consensus index')


def admit(db,target,split,expected,effective,whole_matrix):
    report=json.loads((target/'support.json').read_text());exact_support(report,expected,effective)
    features={}
    for batch in pq.ParquetFile(target/'features.parquet').iter_batches(batch_size=256):
        for r in batch.to_pylist():
            if r['row_idx']!=len(features):raise ValueError('semantic feature order')
            found=db.execute('SELECT vec,split FROM features WHERE ref=? AND enc=?',(r['reference_sha256'],r['encoded_sha256'])).fetchone()
            import numpy as np
            if found is None or found!=(np.asarray(r['features'],dtype='<f4').tobytes(),split):raise ValueError('semantic native feature mismatch')
            features[r['row_idx']]=(r['reference_sha256'],r['encoded_sha256'])
    proofs=iter(pq.ParquetFile(target/'edge-proof.parquet').iter_batches(batch_size=256));seen=set();count=0
    for batch in pq.ParquetFile(target/'pairs.parquet').iter_batches(batch_size=256):
        proof_batch=next(proofs,None)
        if proof_batch is None or proof_batch.num_rows!=batch.num_rows:raise ValueError('semantic proof row mismatch')
        for pair,proof in zip(batch.to_pylist(),proof_batch.to_pylist()):
            a,b=pair['worse_idx'],pair['better_idx'];key=tuple(sorted([a,b]))
            if key in seen or a==b or proof['worse_idx']!=a or proof['better_idx']!=b or features.get(a)!=(proof['reference_sha256'],proof['worse_encoded_sha256']) or features.get(b)!=(proof['reference_sha256'],proof['better_encoded_sha256']):raise ValueError('semantic endpoint/duplicate mismatch')
            seen.add(key);edge(db,proof,pair['pair_weight_class'],pair['q_band'],split);count+=1
    if next(proofs,None) is not None or count!=db.execute('SELECT count(*) FROM pairs WHERE split=?',(split,)).fetchone()[0]:raise ValueError('semantic edge omission/addition')
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    value=dict(schema='e36-semantic-export-admission-v1',admitted=True,labels_read=False,fits=0,
        split=split,features=bind(target/'features.parquet'),pairs=bind(target/'pairs.parquet'),edge_proof=bind(target/'edge-proof.parquet'),
        support=bind(target/'support.json'),whole_matrix=bind(whole_matrix),pair_rows=count,feature_rows=len(features),
        owner='hash-bound native outputs -> registered e36_pairs consensus -> unordered physical index -> exact export replay',
        semantic_source_sha256=sha(Path(__file__)),
        consensus_source_sha256=sha(Path(__file__).parents[1]/'canonical_corpus/build_ladder_tv_pairs.py'))
    with (target/'semantic-admission.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return bind(target/'semantic-admission.json')
