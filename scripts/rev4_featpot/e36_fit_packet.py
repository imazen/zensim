"""Prepare opaque data transport and future native FitCell specs; never execute fits."""
import argparse
import ast
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
BLUEPRINTS={'lodo':'47d4dff82b977ff7cc3f2f7d04c44603b9b9bdb538168a720713f04fbf2297c1','full':'d4a47f7ec92a383a1eddac1ecc3e37465148bda8b818741d415294cdfb51478c'}
BASE_CONTRACT='55b8ca6a6cfe545c1c776e655fc9a0cc890581f46c6dc23a79651b3ff51c97f1'
BASE_RECEIPT='501b209a7847a454a465675ac8b4c5fa53a5299111bf363c07699180841d5e77'


def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def retain_copy(source,dest):
    if dest.exists():
        if sha(dest)!=sha(source):raise ValueError('retained E36 reference differs; preserve it before a new version')
    else:shutil.copyfile(source,dest)


def proof_relative(root,path):
    """Keep external source/diagnostic proofs without opening data tables."""
    root=Path(root);path=Path(path)
    if path.is_relative_to(root):return path.relative_to(root)
    if path.suffix not in ['.py','.json','.jsonl','.log','.gdb','.md','.txt','.sh']:
        raise ValueError('external proof must be explicit source or diagnostic text, not a data table')
    return Path('_external_proofs')/sha(path)/path.name


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
    out.mkdir();native=out/'native';native.mkdir();files={};external_proofs={}
    def add(path):
        path=Path(path);relative=proof_relative(root,path);dest=native/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not path.is_relative_to(root):external_proofs[str(path)]=dict(relative_path=str(relative),sha256=sha(path))
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
    from e36_current_panels import native_admitted
    audit=root/'current-panel-audit-final/receipt.json';current=json.loads(audit.read_text())
    native_audit=json.loads((audit.parent/'audit.hashes.json').read_text())
    cohort=json.loads((root/'cohort-selection.json').read_text())
    if (not native_admitted(native_audit,cohort,current['reference_records'])
            or current['native_binary_sha256']!=sha(root/'bin/e36-fit-final/check_holdout_overlap')
            or current['schema']!='e36-current-panel-exclusion-v1' or current['admitted'] is not True
            or current['cohort_sha256']!=sha(root/'cohort-selection.json') or current['labels_read'] is not False
            or current['strict_flags']!=0
            or current['panel_manifest_sha256']!=sha('/mnt/v/output/zensim/shipgates-2026-10-16/rd-inputs-corrected27/SOURCE_MANIFEST.json')
            or current['panel_manifest_sha256']!=sha(audit.parent/'SOURCE_MANIFEST.json')
            or current['native_hashes_sha256']!=sha(audit.parent/'audit.hashes.json')):
        raise ValueError('current ship panel native/family exclusion admission required')
    for path in [audit,audit.parent/'audit.hashes.json',audit.parent/'SOURCE_MANIFEST.json']:add(path)
    from e36_training import assessment_plan
    assessment_inputs=root/'assessment-inputs.json'
    assessment_plan(Path(d['root']),assessment_inputs);add(assessment_inputs)
    choice_doc=json.loads((root/'bulk-choice.json').read_text())
    for path in [root/'bulk-choice.json',Path(choice_doc.get('plan_path',root/'bulk-plan.json')),Path(choice_doc.get('clock_path',root/'bulk-clock.json')),root/'bulk-plan.json',root/'bulk-clock.json',root/'resource-admission.json']:
        add(path)
    repaired_clock=json.loads(Path(choice_doc.get('clock_path',root/'bulk-clock.json')).read_text())
    if 'budget_only_amendment' in repaired_clock or 'budget_only_extension' in repaired_clock:
        from e36_pause_budget import resolve
        _,_,budget_evidence=resolve(Path(choice_doc['clock_path']))
        for evidence in budget_evidence:add(evidence['path'])
    if 'equivalence' in repaired_clock:
        add(repaired_clock['equivalence']['path']);eq=json.loads(Path(repaired_clock['equivalence']['path']).read_text())
        for key in ['original_clock','original_plan','repaired_plan','original_runtime','repaired_runtime']:add(eq[key]['path'])
        for pair in eq['manifests'].values():
            for key in ['original','repaired']:add(pair[key]['path'])
    matrix_doc=json.loads((pools/'matrix-admission.json').read_text())
    for value in matrix_doc['manifests'].values():add(value['path'])
    for value in matrix_doc.get('remote_phase_evidence',[]):add(value['path'])
    for value in matrix_doc.get('scheduling_evidence',[]):add(value['path'])
    # Owner amendments carry nested native identity/ledger/resource and artifact
    # location proofs. Retain those exact bytes, checking each pin before copy.
    retained=set()
    def retain_proof(value):
        if isinstance(value,list):
            for item in value:retain_proof(item)
        elif isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                path=Path(value['path'])
                if sha(path)!=value['sha256']:raise ValueError('nested E36 proof changed')
                if path not in retained:
                    retained.add(path);add(path)
                    if path.suffix=='.json':retain_proof(json.loads(path.read_text()))
            for item in value.values():retain_proof(item)
    for key in ['owner_amendment','owner_matrix','native_publication','artifact_location','resource_closure','urgent_disk','resource_supervision','monitor_restart','storage_priority','native_continuation','archive_budget4','archive_transport','archive_pack','source_count_amendment','effective_source_budget']:
        if key in matrix_doc:retain_proof(matrix_doc[key])
    fit=pools/'fit';contract=copy.deepcopy(json.loads((fit/'contract.json').read_text()))
    if not contract['semantic_export'] or contract['fits_released'] is not False:raise ValueError('successful semantic export required before packet')
    contract.update(trainer_sha256=sha(trainer),base_filter_sha256=sha(mask),build_commit=commit)
    locked=pools/'fit/locked-contract.json';write(locked,contract);add(locked)
    overlay=out/'overlay.json';write(overlay,dict(schema='e36-fit-data-overlay-v1',base_archive_relative='base-data.tar.gz',
        base_data_sha256=BASE_SHA,base_root_receipt_sha256=BASE_RECEIPT,expected_native_root=str(root),
        native_member_sha256={name:sha(path) for name,path in files.items()},external_proof_sources=external_proofs,labels_read=False,fits_released=False,build_commit=commit))
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


def local_import_closure(source,members):
    """Include local Python owners reached by literal imports, even optional ones."""
    pending=list(members);done=set()
    while pending:
        name=pending.pop()
        if name in done:continue
        done.add(name);path=source/name
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text(),filename=name)):
            modules=[]
            if isinstance(node,ast.Import):modules=[r.name for r in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]
            elif (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='__import__'
                  and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)):
                modules=[node.args[0].value]
            for module in modules:
                relative=Path(*module.split('.')).with_suffix('.py')
                candidates=[path.parent/relative,source/relative,source/'scripts'/relative,source/'scripts/rev4_featpot'/relative]
                for candidate in candidates:
                    if candidate.is_file() and candidate.is_relative_to(source):
                        item=str(candidate.relative_to(source))
                        if item not in done:pending.append(item)
                        break
    return done


def admit_program_closure(files,preflight,preparation):
    frozen='62a43287c53d9e19824a66b556745c2a8fdb9cf7'
    if preparation['published_source']!=frozen or preflight['published_source']!=frozen or preparation['compiled_rust_source']!='534cb68f0204c02807b1a5c1612d1fcb6c805365':raise ValueError('authorized frozen program source differs')
    expected={name:value for name,value in preflight['files'].items() if Path(name).suffix in ['.py','.json']}
    # The two admitted Markdown registrations are evidence, not Python owners.
    omitted=set(preflight['files'])-set(expected)
    if omitted!={'benchmarks/e29_e31_e32_shared_control_decision_2026-10-07.md','benchmarks/e29_four_source_amendment_2026-10-07.md'} or files!=expected:raise ValueError('exact frozen program Python closure differs from preflight')


def program_packet(root,pools,packet_dir,original,source,binaries,owner,commit):
    """Use native program packaging; freeze qualification without fitting."""
    from e36_training import curated_projection,protocol_source_sha,SPEC,ARMS
    receipt=json.loads((packet_dir/'data-receipt.json').read_text())
    if sha(packet_dir/'data.tar.gz')!=receipt['sha256'] or sha(binaries/'zensim_mlp_train')!=receipt['trainer_sha256']:
        raise ValueError('E36 program/data/trainer mismatch')
    mask=root/'matched-base-filter-plan-v2.json'
    projections=curated_projection(mask,packet_dir/'curated-projection')
    preparation=root/'program-source-preparation-v1.json'
    source_proof=json.loads(preparation.read_text())
    if Path(source_proof['extracted_source'])!=source or source_proof['exact_rust_cargo_equivalence'] is not True or source_proof['fits']!=0:raise ValueError('frozen admitted trainer program source required')
    preflight=root/'program-closure-preflight-v1.json';source_closure=json.loads(preflight.read_text())
    protocol_sha=protocol_source_sha(source)
    if protocol_sha!=source_closure['protocol_source_sha256'] or source_proof['published_source']!=source_closure['published_source']:raise ValueError('frozen trainer protocol/source differs')
    admission=json.loads((root/'actual-pool-admission-final/receipt.json').read_text())
    if (admission['binary_sha256']!=receipt['trainer_sha256'] or admission['manifest_sha256']!=receipt['fit_manifest_sha256']
            or admission['native'].get('embedded_protocol_source_sha256')!=protocol_sha):
        raise ValueError('actual final trainer/source admission required before program packaging')
    manifest=json.loads((pools/'fit/manifest.json').read_text());n=0;mass=0
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(manifest['pairs']['path']).iter_batches(columns=['pair_weight_class'],batch_size=65536):
        classes=batch.column(0).to_pylist()
        if any(c not in [0,1] for c in classes):raise ValueError('bad physical pair weight class')
        n+=len(classes);mass+=sum(2 if c==0 else 1 for c in classes)
    if n!=manifest['pair_rows'] or n==0:raise ValueError('E36 mean pool weight requires the entire admitted fit pool')
    if sha(original/'e34-fit-contract.json')!=BASE_CONTRACT:raise ValueError('frozen original C qualification package changed')
    old=json.loads((original/'e34-fit-contract.json').read_text())
    controls=[v for v in old['variants'] if v['name'] in ['controlc','full-controlc']]
    if len(controls)!=2 or any(v['spec']!=SPEC or any(v['e34_flags'].values()) for v in controls):
        raise ValueError('exact unaugmented original C controls required')
    variants=[]
    for arm in ARMS:
        for oldvariant in controls:
            v=copy.deepcopy(oldvariant);v.update(schema='e36-research-fit-contract-v1',name=arm+'-'+oldvariant['name'],
                data_sha=receipt['sha256'],build_commit=commit,launchable=True)
            v['e36']=dict(arm=arm,trainer_sha256=receipt['trainer_sha256'],protocol_source_sha256=protocol_sha,
                manifest_sha256=receipt['fit_manifest_sha256'],contract_sha256=receipt['locked_contract_sha256'],
                base_filter_sha256=receipt['base_filter_sha256'],mean_pool_weight=mass/n,curated_tables=projections)
            variants.append(v)
    # Existing E33 exact-identity owner and inputs; no human ranks enter
    # TRAIN selection. The coordinator explicitly chose this branch.
    import struct
    identity_sources=root/'selection-identity-sources.json'
    retain_copy(Path('/mnt/v/output/zensim/e33-impl-2026-10-09/E3_SOURCES.json'),identity_sources)
    production=root/'bin/e36-selection-production.bin'
    production_source=Path('/mnt/v/output/zensim/adjudicate-2026-10-07/models/seed0.bin')
    if sha(production_source)!='f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400':raise ValueError('registered identity production owner input changed')
    retain_copy(production_source,production)
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    policy=packet_dir/'selection-guard-policy.json'
    write(policy,dict(schema='e36-train-selection-guard-policy-v1',selection_has_human_ranks=False,
        required_guards=['native_raw_forward_finite','existing_e33_exact_identity'],identity_expected_rows=620,
        identity_expected_sources=62,identity_pin_bits=struct.unpack('<Q',struct.pack('<d',100.))[0],
        admission_owner_sha256=sha(Path(__file__).with_name('e36_selection_guards.py')),
        binary=bind(binaries/'serve_custom_bake'),sources=bind(identity_sources),fx1=bind(source/'scripts/rev4_featpot/e33_fx1_declaration.json'),
        production=bind(production),identity_source_sha256=sha(source/'zensim/examples/serve_custom_bake/nearid.rs'),
        cross_reference_ranking='assessment: registered noninferiority, dataset point and W2 guards'))
    package=packet_dir/'qualified-package.json';write(package,dict(schema='e36-research-fit-package-v1',variants=variants))
    prior=json.loads((original/'source-pins.json').read_text());members=set(prior['files'])
    members.update(['scripts/rev4_featpot/e36_training.py','scripts/rev4_featpot/e36_store.py'])
    members=local_import_closure(source,members)
    files={name:sha(source/name) for name in sorted(members) if Path(name).suffix in ['.py','.json']}
    admit_program_closure(files,source_closure,source_proof)
    pins=packet_dir/'source-pins.json';contract_name='benchmarks/e36_fit_contract_2026-10-17.json'
    write(pins,dict(schema='e36-public-program-v1',build_commit=commit,files=files,data={contract_name:sha(package)}))
    import importlib.metadata
    import os
    import re
    import sys
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','pyarrow','scipy']}
    image=owner/'Dockerfile.fitcell'
    image_pins=dict(re.findall(r'([a-z][a-z0-9_-]*)==([0-9][a-zA-Z0-9.+_-]*)',image.read_text()))
    if any(versions[n]!=image_pins[n] for n in versions):raise ValueError('local program admission packages differ from frozen fit image pins')
    meta=packet_dir/'build-meta.json';write(meta,dict(schema='fleet-fits-build-v1',build_commit=commit,
        python_build_version=sys.version.split()[0],python_packages=versions,
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        cargo=subprocess.check_output(['cargo','--version'],text=True).strip(),
        cargo_lock_sha256=sha(source/'Cargo.lock'),image_recipe_sha256=sha(image),
        zensim_profile='release',source_pins_sha256=sha(pins),protocol_source_sha256=protocol_sha,
        rustflags=os.environ.get('RUSTFLAGS',''),cargo_encoded_rustflags=os.environ.get('CARGO_ENCODED_RUSTFLAGS',''),
        e36_source=dict(preparation_sha256=sha(preparation),preflight_sha256=sha(preflight),python_source_commit=source_proof['published_source'],compiled_rust_source_commit=source_proof['compiled_rust_source']),
        e36_fit=dict(data_sha256=receipt['sha256'],base_filter_sha256=receipt['base_filter_sha256'],fits_released=False)))
    output=packet_dir/'program.tar.gz'
    subprocess.run(['python3',str(owner/'pack_fit_program.py'),'--profile','e36','--source',str(source),
        '--source-pins',str(pins),'--source-pins-sha256',sha(pins),'--executor',str(owner/'fit_cell_exec.py'),
        '--bin-dir',str(binaries),'--build-meta',str(meta),'--data',contract_name+'='+str(package),'--out',str(output)],check=True)
    write(packet_dir/'program-receipt.json',dict(schema='e36-native-fit-program-v1',build_commit=commit,sha256=sha(output),
        protocol_source_sha256=protocol_sha,qualified_package_sha256=sha(package),source_pins_sha256=sha(pins),
        native_owner_sha256=sha(owner/'pack_fit_program.py'),binaries={p.name:sha(p) for p in binaries.iterdir() if p.is_file()},
        frozen_source_preparation=dict(path=str(preparation),sha256=sha(preparation)),frozen_source_preflight=dict(path=str(preflight),sha256=sha(preflight)),
        python_source_commit=source_proof['published_source'],compiled_rust_source_commit=source_proof['compiled_rust_source'],
        labels_read=False,fits_released=False,original_control_contract_sha256=sha(original/'e34-fit-contract.json')))


def specs(root,pools,packet_dir,original,program,ctl,owner):
    receipt=json.loads((packet_dir/'data-receipt.json').read_text())
    if sha(packet_dir/'data.tar.gz')!=receipt['sha256']:raise ValueError('opaque packet changed')
    overlay_sha=sha(packet_dir/'overlay.json');locked=pools/'fit/locked-contract.json';mask=root/'matched-base-filter-plan-v2.json'
    records={};all_cells=[]
    for arm in ['tv0-c','tv1-c','tv10-c']:
        for kind in ['lodo','full']:
            old=original/('fit-spec-fite34-controlc-'+kind+'-20261015.json')
            if sha(old)!=BLUEPRINTS[kind]:raise ValueError('frozen original C blueprint changed')
            d=json.loads(old.read_text());cells=[]
            if len(d['cells'])!=(40 if kind=='lodo' else 3):raise ValueError('original fixed C inventory changed')
            for oldcell in d['cells']:
                cell=copy.deepcopy(oldcell);cell['name']='e36-'+arm+'__N/'+oldcell['name'].split('/')[1];argv=cell['argv']
                suffix='confirm/cells' if kind=='full' else 'cells'
                argv[argv.index('--dest')+1]='/var/tmp/rev4-featpot/e36-'+arm+'-results/'+suffix+'/'+cell['name']
                argv.extend(['--e36-arm',arm,'--e36-tv-manifest',str(pools/'fit/manifest.json'),'--e36-tv-contract',str(locked),
                    '--e36-tv-contract-sha256',sha(locked),'--e36-base-filter-plan',str(mask),'--e36-base-filter-sha256',sha(mask),
                    '--e36-overlay','/var/tmp/rev4-featpot/e36-native/overlay.json','--e36-overlay-sha256',overlay_sha])
                cells.append(cell);all_cells.append(dict(arm=arm,kind=kind,name=cell['name'],seed_index=int(argv[argv.index('--seed-index')+1]),heldout=argv[argv.index('--heldout')+1] if kind=='lodo' else None,argv_sha256=hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))
            name='e36-'+arm+'-'+kind;spec=packet_dir/(name+'.spec.json');manifest=packet_dir/(name+'.manifest.json')
            write(spec,dict(program_sha=sha(program),data_sha=receipt['sha256'],cells=cells))
            subprocess.run([str(ctl),'declare-fits','--spec',str(spec),'--out',str(manifest)],check=True)
            records[name]=dict(spec_path=str(spec),manifest_path=str(manifest),spec_sha256=sha(spec),manifest_sha256=sha(manifest),cells=len(cells),blueprint_sha256=sha(old))
    from e36_inventory import admit_inventory
    admit_inventory(all_cells)
    with tarfile.open(program) as archive:program_meta=json.load(archive.extractfile('build_meta.json'))
    qualification_owners={k:program_meta['files'][v] for k,v in dict(inspector_sha256='bin/inspect_qualified_checkpoint',
        qualification_owner_sha256='qualified_fit_contract.py',e36_owner_sha256='e36_fit_contract.py').items()}

    # Reuse the actual executor path owner and qualified contract owner on
    # every declared slot. This only reads program/argv metadata, never fits.
    import sys
    sys.path.insert(0,str(owner))
    import fit_cell_exec as native_executor
    import qualified_fit_contract as native_qualification
    expected_by_name={c['name']:c for c in all_cells};native_seen=set()
    for phase in records.values():
        jobs=json.loads(Path(phase['manifest_path']).read_text())
        for job in jobs:
            name=job['cell']['image_path'];c=expected_by_name.get(name);kind=job['kind'];argv=kind['argv']
            if c is None or name in native_seen or kind['argv_sha']!=c['argv_sha256']:
                raise ValueError('native manifest differs from exact frozen inventory')
            native_seen.add(name)
            if Path(argv[argv.index('--dest')+1])!=native_executor.FIT_ROOT/native_executor.cell_root(argv)/name:
                raise ValueError('native executor results destination differs')
            expected=native_qualification.trusted_contract(kind,program,root/'bin/e36-fit-final/inspect_qualified_checkpoint')
            if expected['e36']['arm']!=c['arm'] or expected['seed_index']!=c['seed_index'] or expected['route']!=(c['heldout'] or 'production'):
                raise ValueError('native qualification route/seed/arm differs')
    if native_seen!=set(expected_by_name):raise ValueError('native declared inventory incomplete')
    write(packet_dir/'E36_FIT_PLAN.json',dict(schema='e36-future-fit-plan-v1',program_sha256=sha(program),data_sha256=receipt['sha256'],
        fit_manifest_sha256=receipt['fit_manifest_sha256'],selection_manifest_sha256=sha(pools/'selection/manifest.json'),
        contract_sha256=receipt['locked_contract_sha256'],base_filter_sha256=receipt['base_filter_sha256'],trainer_sha256=receipt['trainer_sha256'],raw_binary_sha256=sha(root/'bin/e36-raw-score-final'),ctl_sha256=sha(ctl),
        assessment_inputs=dict(path=str(root/'assessment-inputs.json'),sha256=sha(root/'assessment-inputs.json')),assessment_owner_sha256=program_meta['files']['bin/panel'],selection_guard_policy=dict(path=str(packet_dir/'selection-guard-policy.json'),sha256=sha(packet_dir/'selection-guard-policy.json')),selection_owner_sha256=sha(Path(__file__).with_name('e36_selection.py')),inventory_owner_sha256=sha(Path(__file__).with_name('e36_inventory.py')),qualification_owners=qualification_owners,manifests=records,cells=all_cells,fit_release=False,fit_smoke_permitted=False,policy='all three arms use the identical frozen pool/build/base-family mask; separate coordinator go required'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('data')
    for name in ['root','pools','base','trainer','out','owner']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--commit',required=True)
    a=sub.add_parser('program')
    for name in ['root','pools','packet','original','source','binaries','owner']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--commit',required=True)
    a=sub.add_parser('specs')
    for name in ['root','pools','packet','original','program','ctl','owner']:a.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.mode=='data':packet(args.root,args.pools,args.base,args.trainer,args.out,args.owner,args.commit)
    elif args.mode=='program':program_packet(args.root,args.pools,args.packet,args.original,args.source,args.binaries,args.owner,args.commit)
    else:specs(args.root,args.pools,args.packet,args.original,args.program,args.ctl,args.owner)
