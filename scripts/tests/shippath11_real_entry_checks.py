"""Verify real executor blobs; reject reviewer controls after recomputing receipts."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--zenmetrics',type=Path,required=True)
args=p.parse_args(); root=args.bundle
sys.path.insert(0,str(args.zenmetrics/'scripts/jobsys'))
import harvest_fit_cells as h
program=root/'image-context/program.tar.gz'
inspector=root/'bin/inspect_qualified_checkpoint'
report={'scope':'real image fit-cell-exec; explicit local smoke; no registered fit installation','smokes':{},'negative_controls':{}}


def modified_blob(blob, source_job, target_job, changes, output):
    oldprefix=h.blob_root(source_job['kind'])+'/'+source_job['cell']['image_path']+'/'
    newprefix=h.blob_root(target_job['kind'])+'/'+target_job['cell']['image_path']+'/'
    with tarfile.open(blob) as t:
        files={m.name.removeprefix(oldprefix):t.extractfile(m).read() for m in t.getmembers()}
    r=json.loads(files['result.json'])
    r.update(changes)
    selected=Path(r['selected_bake']).name
    r['selected_bake']=str(h.POT_ROOT/newprefix/'refit'/selected)
    if r.get('packed_model'):
        r['packed_model']=str(h.POT_ROOT/newprefix/'refit'/Path(r['packed_model']).name)
    files['result.json']=json.dumps(r).encode()
    receipt=json.loads(files.pop('fleet_receipt.json'))
    receipt.update(cell=target_job['cell']['image_path'],program_sha=target_job['kind']['program_sha'],
                   data_sha=target_job['kind']['data_sha'],argv_sha=target_job['kind']['argv_sha'],
                   files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
                   result_sha=hashlib.sha256(files['result.json']).hexdigest())
    files['fleet_receipt.json']=json.dumps(receipt).encode()
    with tarfile.open(output,'w:gz') as t:
        for n,b in files.items():
            m=tarfile.TarInfo(newprefix+n);m.size=len(b);t.addfile(m,io.BytesIO(b))


for route,jobset in [('e30','fitv2e30-20261007'),('production','fitv2d1-20261007')]:
    smoke=json.loads((root/f'local-smoke-manifest-{jobset}.json').read_text())[0]
    full=json.loads((root/f'fit-manifest-{jobset}.json').read_text())[0]
    blob=root/f'{route}-executor-blob.tar.gz'
    receipt=h.verify_blob(blob,root/f'{route}-harvest-smoke',smoke['cell']['image_path'],smoke['kind'],
                         program_archive=program,checkpoint_inspector=inspector,allow_local_smoke=True)
    assert receipt['execution_contract']=='local-smoke'
    report['smokes'][route]={'status':'PASS','epochs':2,'pairs_per_epoch':128,'selected_epoch':1,
                            'tier':receipt['tier_status'],'blob_sha256':h.digest(blob)}
    with tarfile.open(blob) as t:
        prefix=h.blob_root(smoke['kind'])+'/'+smoke['cell']['image_path']+'/'
        original=json.load(t.extractfile(prefix+'result.json'))
    selection=copy.deepcopy(original['selection']);selection['selected_epoch']=119
    claimed={'epochs':120,'pairs_per_epoch':50000,'execution_contract':'registered-fit','selection':selection}
    zeros={**claimed,'data_role_decision_sha256':'0'*64,'wide_receipt_sha256':'0'*64,'frozen_sha256':'0'*64,
           'selection':{**selection,'strict_table_admission':[{}]}}
    for case,changes in [('short-under-full',{}),('epoch119-with-epoch1',claimed),('zero-admission',zeros),('mode-bypass',{'training_only':False,'prediction':[],'test_rows':0})]:
        output=root/f'{route}-negative-{case}.tar.gz'
        modified_blob(blob,smoke,full,changes,output)
        try:
            h.verify_blob(output,root/f'{route}-negative-{case}',full['cell']['image_path'],full['kind'],
                          program_archive=program,checkpoint_inspector=inspector)
        except ValueError as exc:
            report['negative_controls'][route+'/'+case]={'status':'REFUSED','reason':str(exc)}
        else:
            raise AssertionError(f'negative control accepted: {route}/{case}')
    try:
        h.verify_blob(blob,root/f'{route}-smoke-install-refusal',smoke['cell']['image_path'],smoke['kind'],
                      program_archive=program,checkpoint_inspector=inspector)
    except ValueError as exc:
        assert 'local smoke cannot install' in str(exc)
    else:
        raise AssertionError('smoke accepted without verification-only flag')
report['status']='PASS'
(root/'HARVEST_BINDING_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
