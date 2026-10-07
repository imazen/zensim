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

bundle=Path(sys.argv[1]); root=bundle/'v2d1'
assert (EPOCHS,EPOCH_RULE,PAIRS_PER_EPOCH)==(120,'last',50000)
for heldout in [None,*PRODUCTION_SOURCES]:
    preflight_recipe(root,root/'human_role_decision.json',heldout)
with tarfile.open(bundle/'d1-fit-data.tar.gz') as archive:
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
            assert d['formula_revision']==5 and d['decoder_era']
            assert d['feature_set_id']=='basic+peaks+v2@w1825/rev5_localwin#36c3f3af'
            if d.get('human_sources'):
                assert set(d['human_sources'])<=set(PRODUCTION_SOURCES)
    assert tables==19
with tarfile.open(bundle/'image-context/program.tar.gz') as archive:
    metadata=json.load(archive.extractfile('build_meta.json'))
    for name,pin in metadata['files'].items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==pin,name
    assert 'fit_paths.py' in metadata['files']
program,data=sha(bundle/'image-context/program.tar.gz'),sha(bundle/'d1-fit-data.tar.gz')
for jobset,count in [('fitv2e30-20261007',40),('fitv2d1-20261007',3)]:
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
        if count==40:
            assert argv[argv.index('--heldout')+1] in PRODUCTION_SOURCES
        else:
            assert '--heldout' not in argv and '--pack-production' in argv
            assert argv[argv.index('--seed-index')+1] in ('0','1','2')
decision=json.loads((root/'human_role_decision.json').read_text())
assert sha(Path(decision['source_receipt']))==decision['source_receipt_sha256']
assert sha(Path(decision['source_frozen']))==decision['source_frozen_sha256']
report={'status':'PASS','data_files':len(inventory['files']),'table_payloads':tables,'program_files':len(metadata['files']),
        'E30_cells':40,'production_cells':3,'epochs':120,'selected_epoch':119,'pairs_per_epoch':50000,
        'four_source_routes':True,'source_receipt_and_freeze_unchanged':True,'aic_table_payloads':0}
(bundle/'BUNDLE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
