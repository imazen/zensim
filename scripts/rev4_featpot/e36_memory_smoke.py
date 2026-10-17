"""Prepare one distinct, noninstallable native E36 memory-smoke cell."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from e36_store import sha


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')

def bind(path):return dict(path=str(path),sha256=sha(path))


def seed_zero(blueprint):
    def arg(argv,key):
        if argv.count(key)!=1:raise ValueError('unique diagnostic identity argument required: '+key)
        return argv[argv.index(key)+1]
    cells=blueprint['cells'];seeds=[]
    for cell in cells:
        argv=cell['argv']
        if argv[0]!='v2_confirm_fit.py' or arg(argv,'--e36-arm')!='tv10-c' or '--heldout' in argv:raise ValueError('full-data tv10 diagnostic parent required')
        seed=int(arg(argv,'--seed-index'));seeds.append(seed)
        if cell['name']!='e36-tv10-c__N/full_s'+str(seed):raise ValueError('diagnostic parent name/seed differs')
    if sorted(seeds)!=[0,1,2]:raise ValueError('exact three-seed full-data blueprint required')
    return copy.deepcopy(cells[seeds.index(0)])


def diagnostic_result(result,qualified):
    if (result['execution_contract']!='local-smoke' or qualified['execution_contract']!='local-smoke'
        or result['selection']['epoch_rule']!='last' or result['selection']['selected_epoch']!=1):
        raise ValueError('diagnostic cannot be scientific output')

def prepare(root,packet,out,owner,commit):
    root=Path(root);packet=Path(packet);out=Path(out);owner=Path(owner)
    original={name:sha(packet/name) for name in ['program.tar.gz','data.tar.gz','qualified-package.json','source-pins.json']}
    admission=json.loads((packet/'program-admission/receipt.json').read_text())
    if admission['program_sha256']!=original['program.tar.gz'] or not admission['all_inventory_hashes_match']:raise ValueError('admitted scientific program required')
    out.mkdir();source=out/'source';source.mkdir()
    pins=json.loads((packet/'source-pins.json').read_text());frozen=root/'program-source-v1'
    for name,digest in pins['files'].items():
        if sha(frozen/name)!=digest:raise ValueError('frozen final scientific source changed')
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(frozen/name,target)
    projection=Path(__file__).with_name('e36_training.py');target=source/'scripts/rev4_featpot/e36_training.py'
    before=sha(target);shutil.copyfile(projection,target);pins['files']['scripts/rev4_featpot/e36_training.py']=sha(target)
    owner_receipt=json.loads((root/(owner.name+'.json')).read_text())
    for name,digest in owner_receipt['files'].items():
        if sha(owner/name)!=digest:raise ValueError('frozen native smoke owner changed')
    sys.path.insert(0,str(owner))
    from e36_fit_contract import SMOKE
    package=json.loads((packet/'qualified-package.json').read_text())
    variants=[v for v in package['variants'] if v['e36']['arm']=='tv10-c' and 'production' in v['routes']]
    if len(variants)!=1:raise ValueError('unique full-data tv10 diagnostic parent required')
    variant=copy.deepcopy(variants[0]);variant['e36_memory_smoke']=SMOKE
    write(out/'qualified-package.json',dict(schema=package['schema'],variants=[variant]))
    contract='benchmarks/e36_fit_contract_2026-10-17.json';pins.update(build_commit=commit,data={contract:sha(out/'qualified-package.json')})
    write(out/'source-pins.json',pins)
    meta=json.loads((packet/'build-meta.json').read_text());meta.update(build_commit=commit,source_pins_sha256=sha(out/'source-pins.json'))
    meta['e36_memory_smoke']=dict(policy=SMOKE,projection_overlay=bind(projection),original_projection_sha256=before,final_scientific_program_sha256=original['program.tar.gz'],native_owner=bind(owner/'SOURCE.json'),final_scientific_program_unchanged=True)
    write(out/'build-meta.json',meta)
    subprocess.run([sys.executable,str(owner/'pack_fit_program.py'),'--profile','e36','--source',str(source),'--source-pins',str(out/'source-pins.json'),'--source-pins-sha256',sha(out/'source-pins.json'),'--executor',str(owner/'fit_cell_exec.py'),'--bin-dir',str(root/'bin/e36-fit-final'),'--build-meta',str(out/'build-meta.json'),'--data',contract+'='+str(out/'qualified-package.json'),'--out',str(out/'program.tar.gz')],check=True)
    plan=json.loads((packet/'E36_FIT_PLAN.json').read_text());spec=packet/'e36-tv10-c-full.spec.json'
    if sha(spec)!=plan['manifests']['e36-tv10-c-full']['spec_sha256'] or plan['program_sha256']!=original['program.tar.gz'] or plan['data_sha256']!=original['data.tar.gz']:raise ValueError('registered scientific blueprint/program/data changed')
    blueprint=json.loads(spec.read_text());cell=seed_zero(blueprint);cell['name']='e36-memory-smoke__N/full_s0'
    argv=cell['argv'];argv[argv.index('--dest')+1]='/var/tmp/rev4-featpot/e36-memory-smoke-results/confirm/cells/'+cell['name']
    for flag in ['--pack-production','--e33-output-stage']:
        if argv.count(flag)!=1:raise ValueError('exact original production flags required')
        argv.remove(flag)
    argv.extend(['--local-smoke-budget','2:128'])
    write(out/'smoke.spec.json',dict(program_sha=sha(out/'program.tar.gz'),data_sha=original['data.tar.gz'],cells=[cell]))
    subprocess.run([str(root/'bin/zenfleet-ctl'),'declare-fits','--spec',str(out/'smoke.spec.json'),'--out',str(out/'smoke.manifest.json')],check=True)
    with (out/'native.ids.tsv').open('x') as stream:subprocess.run([str(root/'bin/zenfleet-ctl'),'ids','--manifest',str(out/'smoke.manifest.json')],stdout=stream,check=True)
    jobs=json.loads((out/'smoke.manifest.json').read_text())
    if len(jobs)!=1:raise ValueError('exact one-cell native diagnostic declaration required')
    from qualified_fit_contract import trusted_contract
    expected=trusted_contract(jobs[0]['kind'],out/'program.tar.gz',root/'bin/e36-fit-final/inspect_qualified_checkpoint')
    if expected['execution_contract']!='local-smoke' or [expected[k] for k in ['epochs','pairs_per_epoch','selected_epoch']]!=[2,128,1] or expected['route']!='production' or expected['seed_index']!=0 or expected['e36']['arm']!='tv10-c':raise ValueError('actual native smoke qualification differs')
    if {name:sha(packet/name) for name in original}!=original:raise ValueError('diagnostic preparation changed scientific artifacts')
    receipt=dict(schema='e36-native-memory-smoke-preparation-v1',build_commit=commit,program=bind(out/'program.tar.gz'),data=bind(packet/'data.tar.gz'),spec=bind(out/'smoke.spec.json'),manifest=bind(out/'smoke.manifest.json'),native_ids=bind(out/'native.ids.tsv'),scientific_parent=bind(packet/'program.tar.gz'),policy=SMOKE,native_execution_contract=expected['execution_contract'],numerical_trainer_unchanged=True,initial_memory_cap_bytes=8*1024**3,rayon_threads=2,planned_host='i270',fits=0,labels_read=False)
    write(out/'PREPARED.json',receipt);return receipt


def verify(root,out,owner,ledger,blobs):
    root=Path(root);out=Path(out);owner=Path(owner);ledger=Path(ledger);blobs=Path(blobs)
    prepared=json.loads((out/'PREPARED.json').read_text())
    for key in ['program','data','spec','manifest','native_ids']:
        if sha(prepared[key]['path'])!=prepared[key]['sha256']:raise ValueError('prepared diagnostic identity changed')
    jobs=json.loads(Path(prepared['manifest']['path']).read_text())
    if len(jobs)!=1:raise ValueError('one diagnostic cell required')
    import pyarrow.parquet as pq
    rows=pq.read_table(ledger).to_pylist()
    if len(rows)!=1 or rows[0]['status']!='done' or rows[0]['attempts']!=1:raise ValueError('single-attempt native diagnostic DONE required')
    identities=Path(prepared['native_ids']['path']).read_text().splitlines()
    if len(identities)!=1 or rows[0]['job_id']!=identities[0].split('\t')[1] or json.loads(rows[0]['kind_json'])!=jobs[0]['kind']:raise ValueError('native diagnostic ID/kind differs')
    from e36_store import digest
    h=digest(rows[0]['output_sha']);blob=blobs/h
    if sha(blob)!=h:raise ValueError('native diagnostic output CAS changed')
    sys.path.insert(0,str(owner));import harvest_fit_cells as native
    stage=out/'verified-stage'
    if stage.exists():raise ValueError('preserve the previous diagnostic verification stage')
    job=jobs[0];name=job['cell']['image_path']
    receipt=native.verify_blob(blob,stage,name,job['kind'],program_archive=Path(prepared['program']['path']),checkpoint_inspector=root/'bin/e36-fit-final/inspect_qualified_checkpoint',allow_local_smoke=True)
    dest=native.cell_dir(stage,name,job['kind']);result=json.loads((dest/'result.json').read_text())
    qualified=json.loads((dest/'e36-qualified-checkpoint.json').read_text())
    diagnostic_result(result,qualified)
    value=dict(schema='e36-verified-memory-smoke-v1',admitted=True,execution_contract='local-smoke',diagnostic_fits=1,scientific_fits=0,program=prepared['program'],data=prepared['data'],manifest=prepared['manifest'],ledger=bind(ledger),blob=bind(blob),native_receipt=receipt,resource=bind(dest/'RESOURCE_USAGE.txt'),qualified=bind(dest/'e36-qualified-checkpoint.json'),labels_read=False)
    write(out/'VERIFIED.json',value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare')
    for name in ['root','packet','out','owner']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--commit',required=True);v=sub.add_parser('verify')
    for name in ['root','out','owner','ledger','blobs']:v.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.packet,a.out,a.owner,a.commit) if a.mode=='prepare' else verify(a.root,a.out,a.owner,a.ledger,a.blobs)))
