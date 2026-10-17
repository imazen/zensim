"""Prepare opaque data transport and future native FitCell specs; never execute fits."""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from e36_store import sha
from e36_matrix import admit as admit_matrix

BASE_SHA='9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008'
BASE_RECEIPT='501b209a7847a454a465675ac8b4c5fa53a5299111bf363c07699180841d5e77'


def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def packet(root,pools,base,trainer,out,owner,commit):
    choice=json.loads((root/'bulk-choice.json').read_text());admit_matrix(root,choice['selected_runs'])
    if sha(base)!=BASE_SHA:raise ValueError('original opaque base archive changed')
    # Only the first, target-free inventory member is decoded. The original
    # compressed payload is transported verbatim as a single opaque member.
    with tarfile.open(base) as archive:
        first=next(iter(archive))
        if first.name!='input_inventory.json':raise ValueError('original archive inventory must be first')
        inventory=json.load(archive.extractfile(first))
    if inventory['files']['rev4-featpot/v2e29/wide/main/real/receipt.json']!=BASE_RECEIPT:raise ValueError('original base root differs from metadata-only mask plan')
    mask=root/'matched-base-filter-plan-v2.json';d=json.loads(mask.read_text())
    if d['schema']!='e36-matched-base-filter-v2' or d['identity_normalization']!='member-prefix+rendition-v2' or not d['complete'] or d['root_receipt_sha256']!=BASE_RECEIPT or d['labels_read'] is not False:raise ValueError('strict version2 base filter required')
    out.mkdir();native=out/'native';native.mkdir();files={}
    def add(path):
        path=Path(path);relative=path.relative_to(root);dest=native/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(dest)!=sha(path):raise ValueError('conflicting native metadata member')
        else:shutil.copyfile(path,dest)
        files[str(relative)]=dest
    for split in ['fit','selection']:
        directory=pools/split
        for n in ['features.parquet','pairs.parquet','edge-proof.parquet','support.json','manifest.json','fx1.json','semantic-admission.json']:
            add(directory/n)
        m=json.loads((directory/'manifest.json').read_text())
        for key in ['references','source_budget','cohort','runtime']:add(m[key]['path'])
    for name in ['whole-matrix.json','matrix-admission.json']:add(pools/name)
    add(mask)
    choice_doc=json.loads((root/'bulk-choice.json').read_text())
    for path in [root/'bulk-choice.json',Path(choice_doc.get('plan_path',root/'bulk-plan.json')),Path(choice_doc.get('clock_path',root/'bulk-clock.json')),root/'bulk-plan.json',root/'bulk-clock.json',root/'resource-admission.json']:
        add(path)
    repaired_clock=json.loads(Path(choice_doc.get('clock_path',root/'bulk-clock.json')).read_text())
    if 'equivalence' in repaired_clock:
        add(repaired_clock['equivalence']['path']);eq=json.loads(Path(repaired_clock['equivalence']['path']).read_text())
        for key in ['original_clock','original_plan','repaired_plan','original_runtime','repaired_runtime']:add(eq[key]['path'])
        for pair in eq['manifests'].values():
            for key in ['original','repaired']:add(pair[key]['path'])
    matrix_doc=json.loads((pools/'matrix-admission.json').read_text())
    for value in matrix_doc['manifests'].values():add(value['path'])
    fit=pools/'fit';contract=copy.deepcopy(json.loads((fit/'contract.json').read_text()))
    if not contract['semantic_export'] or contract['fits_released'] is not False:raise ValueError('successful semantic export required before packet')
    contract.update(trainer_sha256=sha(trainer),base_filter_sha256=sha(mask),build_commit=commit)
    locked=pools/'fit/locked-contract.json';write(locked,contract);add(locked)
    overlay=out/'overlay.json';write(overlay,dict(schema='e36-fit-data-overlay-v1',base_archive_relative='base-data.tar.gz',
        base_data_sha256=BASE_SHA,base_root_receipt_sha256=BASE_RECEIPT,expected_native_root=str(root),
        native_member_sha256={name:sha(path) for name,path in files.items()},labels_read=False,fits_released=False,build_commit=commit))
    archive_files={'rev4-featpot/e36-native/overlay.json':overlay,'rev4-featpot/e36-native/base-data.tar.gz':base}
    archive_files.update({'rev4-featpot/e36-native/native/'+name:path for name,path in files.items()})
    module_spec=importlib.util.spec_from_file_location('native_pack_fit_data',owner/'pack_fit_data.py');native_owner=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(native_owner)
    payload=out/'data.tar.gz'
    data_inventory=dict(schema='zenfleet-fit-data-v1',sets=['E36-feature-only-overlay+opaque-E34-base'],files={name:sha(path) for name,path in sorted(archive_files.items())})
    raw_inventory=(json.dumps(data_inventory,sort_keys=True,separators=(',',':'))+'\n').encode()
    with payload.open('xb') as stream,gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0,compresslevel=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
        item=tarfile.TarInfo('input_inventory.json');item.size=len(raw_inventory);archive.addfile(item,io.BytesIO(raw_inventory))
        for name,path in sorted(archive_files.items()):native_owner.add_file(archive,path,name)
    write(out/'data-receipt.json',dict(schema='e36-native-fit-data-packet-v1',build_commit=commit,sha256=sha(payload),bytes=payload.stat().st_size,
        labels_decoded=False,original_archive_sha256=BASE_SHA,trainer_sha256=sha(trainer),overlay_sha256=sha(overlay),
        fit_manifest_sha256=sha(fit/'manifest.json'),locked_contract_sha256=sha(locked),base_filter_sha256=sha(mask),fit_release=False))


def specs(root,pools,packet_dir,original,program,ctl):
    receipt=json.loads((packet_dir/'data-receipt.json').read_text())
    if sha(packet_dir/'data.tar.gz')!=receipt['sha256']:raise ValueError('opaque packet changed')
    overlay_sha=sha(packet_dir/'overlay.json');locked=pools/'fit/locked-contract.json';mask=root/'matched-base-filter-plan-v2.json'
    records={};all_cells=[]
    for arm in ['tv0-c','tv1-c','tv10-c']:
        for kind in ['lodo','full']:
            old=original/('fit-spec-fite34-controlc-'+kind+'-20261015.json');d=json.loads(old.read_text());cells=[]
            if len(d['cells'])!=(40 if kind=='lodo' else 3):raise ValueError('original fixed C inventory changed')
            for oldcell in d['cells']:
                cell=copy.deepcopy(oldcell);cell['name']='e36-'+arm+'__N/'+oldcell['name'].split('/')[1];argv=cell['argv']
                original_dest=Path(argv[argv.index('--dest')+1]);suffix='confirm/cells' if kind=='full' else 'cells'
                argv[argv.index('--dest')+1]='/var/tmp/rev4-featpot/e36-'+arm+'-results/'+suffix+'/'+cell['name']
                argv.extend(['--e36-arm',arm,'--e36-tv-manifest',str(pools/'fit/manifest.json'),'--e36-tv-contract',str(locked),
                    '--e36-tv-contract-sha256',sha(locked),'--e36-base-filter-plan',str(mask),'--e36-base-filter-sha256',sha(mask),
                    '--e36-overlay','/var/tmp/rev4-featpot/e36-native/overlay.json','--e36-overlay-sha256',overlay_sha])
                cells.append(cell);all_cells.append(dict(arm=arm,kind=kind,name=cell['name'],argv_sha256=hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))
            name='e36-'+arm+'-'+kind;spec=packet_dir/(name+'.spec.json');manifest=packet_dir/(name+'.manifest.json')
            write(spec,dict(program_sha=sha(program),data_sha=receipt['sha256'],cells=cells))
            subprocess.run([str(ctl),'declare-fits','--spec',str(spec),'--out',str(manifest)],check=True)
            records[name]=dict(spec_path=str(spec),manifest_path=str(manifest),spec_sha256=sha(spec),manifest_sha256=sha(manifest),cells=len(cells),blueprint_sha256=sha(old))
    if len(all_cells)!=129:raise ValueError('fixed three-arm129-cell inventory differs')
    write(packet_dir/'E36_FIT_PLAN.json',dict(schema='e36-future-fit-plan-v1',program_sha256=sha(program),data_sha256=receipt['sha256'],
        fit_manifest_sha256=receipt['fit_manifest_sha256'],selection_manifest_sha256=sha(pools/'selection/manifest.json'),
        contract_sha256=receipt['locked_contract_sha256'],base_filter_sha256=receipt['base_filter_sha256'],trainer_sha256=receipt['trainer_sha256'],raw_binary_sha256=sha(root/'bin/e36-raw-score-final'),ctl_sha256=sha(ctl),
        manifests=records,cells=all_cells,fit_release=False,fit_smoke_permitted=False,policy='all three arms use the identical frozen pool/build/base-family mask; separate coordinator go required'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('data')
    for name in ['root','pools','base','trainer','out','owner']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--commit',required=True)
    a=sub.add_parser('specs')
    for name in ['root','pools','packet','original','program','ctl']:a.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.mode=='data':packet(args.root,args.pools,args.base,args.trainer,args.out,args.owner,args.commit)
    else:specs(args.root,args.pools,args.packet,args.original,args.program,args.ctl)
