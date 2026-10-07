"""Audit only the explicitly prepared D1 bundle; no bank payload discovery."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from v2_human_role import PRODUCTION_SOURCES, MEMBER_SOURCE, preflight_recipe
from v2_common import EPOCHS, EPOCH_RULE, PAIRS_PER_EPOCH, sha

e29 = '--e29' in sys.argv[2:]
bundle=Path(sys.argv[1]); root=bundle/('v2e29' if e29 else 'v2d1')
assert (EPOCHS,EPOCH_RULE,PAIRS_PER_EPOCH)==(120,'last',50000)
for heldout in [None,*PRODUCTION_SOURCES]:
    preflight_recipe(root,root/'human_role_decision.json',heldout)
if e29:
    from e29_consensus import preflight
    for heldout in PRODUCTION_SOURCES:
        for arm in ('hb4','hc4'):
            preflight(root, heldout, arm)
if '--mirror-only' in sys.argv:
    import random
    mirror=Path(sys.argv[sys.argv.index('--mirror-only')+1])
    excluded={'MIRROR_RECEIPT.json','logs/mirror-verify.log'}
    files=sorted(p for p in bundle.rglob('*') if p.is_file() and str(p.relative_to(bundle)) not in excluded)
    entries={}
    for i,p in enumerate(files):
        rel=str(p.relative_to(bundle));target=mirror/rel
        assert target.is_file() and target.stat().st_size==p.stat().st_size,rel
        source_hash=sha(p);assert sha(target)==source_hash,rel
        entries[rel]={'sha256':source_hash,'bytes':p.stat().st_size}
        if i%250==0:
            print(f'mirror verified {i+1}/{len(files)} files',flush=True)
    selected=random.SystemRandom().sample(sorted(entries),3)
    report={'schema':'e29-tower-mirror-receipt-v1','build_commit':json.loads((bundle/'build-meta.json').read_text())['build_commit'],
            'status':'PASS','source':str(bundle),'mirror':str(mirror),'file_count':len(entries),
            'files':entries,'three_random_files':{n:entries[n] for n in selected},
            'excluded_self_and_active_log':sorted(excluded)}
    raw=json.dumps(report,indent=2)+'\n'
    (bundle/'MIRROR_RECEIPT.json').write_text(raw);(mirror/'MIRROR_RECEIPT.json').write_text(raw)
    print(f'PASS: all {len(entries)} file hashes; three random files: {selected}',flush=True)
    raise SystemExit(0)
with tarfile.open(bundle/('e29-fit-data.tar.gz' if e29 else 'd1-fit-data.tar.gz')) as archive:
    inventory=json.load(archive.extractfile('input_inventory.json'))
    assert inventory['schema']=='zenfleet-fit-data-v1'
    tables=0
    for name,pin in inventory['files'].items():
        path=Path(name)
        assert not any(p.lower() in ('aic3','aic4','sdr25') or 'holdout' in p.lower() or '_sealed' in p.lower()
                       or p.startswith('labels__') for p in path.parts)
        assert path.name not in ('aic3.parquet','aic4.parquet','sdr25.parquet')
        h=hashlib.sha256()
        with archive.extractfile(name) as f:
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        assert h.hexdigest()==pin,name
        if name.endswith('.parquet') and not name.endswith('.keys.parquet'):tables+=1
        if name.endswith('.parquet.manifest.json'):
            d=json.load(archive.extractfile(name))
            if e29 and d.get('study') == 'E29':
                assert d['formula_revision']==5
                assert d['role']=='train' and d['rows']==7390 and d['population']=='agree-only'
                assert d['source_bank_feature_set_id'] is None and len(d['build_commit'])==40
            else:
                assert d['formula_revision']==5 and d['decoder_era']
                assert d['feature_set_id']=='basic+peaks+v2@w1825/rev5_localwin#36c3f3af'
            if d.get('human_sources'):
                assert set(d['human_sources'])<=set(PRODUCTION_SOURCES)
    assert tables==(21 if e29 else 19)
with tarfile.open(bundle/'image-context/program.tar.gz') as archive:
    metadata=json.load(archive.extractfile('build_meta.json'))
    for name,pin in metadata['files'].items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==pin,name
    assert 'fit_paths.py' in metadata['files']
program,data=sha(bundle/'image-context/program.tar.gz'),sha(bundle/('e29-fit-data.tar.gz' if e29 else 'd1-fit-data.tar.gz'))
sets=[('arms',80),('control-proposal',40)] if e29 else [('fitv2e30-20261007',40),('fitv2d1-20261007',3)]
for jobset,count in sets:
    spec=json.loads((bundle/f'fit-spec-{jobset}.json').read_text())
    jobs=json.loads((bundle/f'fit-manifest-{jobset}.json').read_text())
    assert len(jobs)==len(spec['cells'])==count and len({j['cell']['image_path'] for j in jobs})==count
    assert spec['program_sha']==program and spec['data_sha']==data
    for cell,job in zip(spec['cells'],jobs):
        argv=cell['argv'];kind=job['kind']
        assert argv==kind['argv'] and cell['name']==job['cell']['image_path']
        assert '--strict-admission' in argv and '--train-only' in argv and '--dest' in argv
        assert kind['program_sha']==program and kind['data_sha']==data
        assert hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==kind['argv_sha']
        assert len(argv[argv.index('--columns')+1].split(','))==420
        if e29 or count==40:
            assert argv[argv.index('--heldout')+1] in PRODUCTION_SOURCES
        else:
            assert '--heldout' not in argv and '--pack-production' in argv
            assert argv[argv.index('--seed-index')+1] in ('0','1','2')
decision=json.loads((root/'human_role_decision.json').read_text())
assert sha(Path(decision['source_receipt']))==decision['source_receipt_sha256']
assert sha(Path(decision['source_frozen']))==decision['source_frozen_sha256']
report={'status':'PASS','data_files':len(inventory['files']),'table_payloads':tables,'program_files':len(metadata['files']),
        'proposed_cells':dict(sets),'build_commit':inventory.get('build_commit'),'epochs':120,'selected_epoch':119,'pairs_per_epoch':50000,
        'four_source_routes':True,'source_receipt_and_freeze_unchanged':True,'aic_table_payloads':0}
if e29:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from e29_consensus import SOURCE_TABLE, SOURCE_KEYS, SOURCE_MANIFEST, TEACHER
    from v2_teacher import key_path
    original=Path('/var/tmp/rev4-featpot/v2e26r2/wide/main/real/hdr_fit.parquet')
    d=json.loads(Path(str(original)+'.manifest.json').read_text())
    assert d['role']=='train' and d['population']=='agree-only' and d['rows']==7390
    assert d['teacher_sha256']==TEACHER and sha(Path(str(original)+'.manifest.json'))==SOURCE_MANIFEST
    identity=pq.read_table(key_path(original),columns=['row_id','role','agree','ref_path'])
    assert identity.num_rows==7390 and set(identity['role'].to_pylist())=={'train'}
    assert set(identity['agree'].to_pylist())=={True} and len(set(identity['row_id'].to_pylist()))==7390
    assert sha(original)==SOURCE_TABLE and sha(key_path(original))==SOURCE_KEYS
    source=pq.read_table(original)
    teachers=pq.read_table(key_path(original),columns=['hdrvdp3_q_jod','cvvdp_jod'])
    a,b=[teachers[c].to_numpy() for c in teachers.column_names]
    refs=np.array(source['ref_basename'].to_pylist())
    def independent_rank(v):
        _,inverse,counts=np.unique(v,return_inverse=True,return_counts=True)
        return (np.cumsum(counts)-counts+(counts+1)/2)[inverse]
    targets={'hb4':(independent_rank(a)+independent_rank(b)-2)/(2*(len(a)-1)), 'hc4':10*a}
    for arm in ('hb4','hc4'):
        actual=pq.read_table(root/f'wide/main/real/hdr_{arm}.parquet')
        assert actual.column_names==source.column_names
        for c in source.column_names:
            if c=='human_score':
                np.testing.assert_array_equal(actual[c].to_numpy(),targets[arm])
            else:
                if pa.types.is_floating(source[c].type):
                    # Exact IEEE bits include NaN padding; Arrow equals treats NaN as unequal.
                    x,y=actual[c].combine_chunks(),source[c].combine_chunks()
                    assert x.type==y.type and len(x)==len(y) and x.null_count==y.null_count==0
                    assert x.buffers()[1].to_pybytes()==y.buffers()[1].to_pybytes(),(arm,c)
                else:
                    assert actual[c].equals(source[c]), (arm,c)
    pairs=np.array(json.loads((root/'wide/main/real/hdr_hc4.pairs.json').read_text()),dtype=np.int64)
    assert pairs.shape==(16140412,2) and (pairs[:,0]<pairs[:,1]).all()
    assert (np.diff(pairs[:,0]*len(a)+pairs[:,1])>0).all()
    for chunk in np.array_split(pairs,200):
        i,j=chunk.T; da,db=a[i]-a[j],b[i]-b[j]
        assert (refs[i]!=refs[j]).all() and (np.abs(da)>=.05).all() and (np.abs(db)>=.05).all()
        assert (np.sign(da)==np.sign(db)).all()
    eligible=0
    for i in range(len(a)-1):
        da,db=a[i]-a[i+1:],b[i]-b[i+1:]
        eligible+=int(np.count_nonzero((refs[i]!=refs[i+1:]) & (np.abs(da)>=.05) & (np.abs(db)>=.05) & (da*db>0)))
    assert eligible==len(pairs)
    report.update(native_feature_bits_unchanged=True,independent_consensus_targets=True,
                  agreement_pairs=eligible,complete_cross_reference_pair_universe=True)
if not e29:
    report.update(E30_cells=40, production_cells=3)
(bundle/'BUNDLE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
