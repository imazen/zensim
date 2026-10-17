"""Freeze the bounded ingress successor and prepare a nontraining native probe."""
import argparse
import copy
import difflib
import json
from pathlib import Path
import shutil
import tarfile
import time
from e36_store import sha
from e36_training import protocol_source_sha
from e36_pool_finalize import native_source_amendment

SOURCE='170dabe93463bca9a781a8cf65f2a3adf76b6fc9'
NATIVE='66ac610c528e1c89da6b00cf92d6b3ec5cee4ce4'
RUST={'zensim-validate/src/bin/e36_tv/mod.rs','zensim-validate/src/bin/zensim_mlp_train.rs'}
ASSESSMENT_SCRIPT='scripts/rev4_featpot/shipgates_direction.rs'

def pin(path):return dict(path=str(path),sha256=sha(path))
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')

def rust_files(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and (p.suffix=='.rs' or p.name in ['Cargo.toml','Cargo.lock'] or str(p.relative_to(root)).startswith('.cargo/'))}

def extract_native(archive,destination):
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if member.isdir():continue
            relative=Path(member.name).relative_to('scripts/jobsys')
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:raise ValueError('nonfile native owner archive entry')
            dest=destination/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as incoming,dest.open('xb') as output:shutil.copyfileobj(incoming,output)

def freeze(root,target):
    root=Path(root);target=Path(target);out=root/'ingress-integration-v1';out.mkdir()
    archive=root/'program-source-170dabe9.tar';receipt=json.loads((root/'program-source-170dabe9.json').read_text())
    if receipt['source_commit']!=SOURCE or receipt['archive_sha256']!=sha(archive):raise ValueError('published successor source archive changed')
    source=root/'program-source-v2';source.mkdir()
    with tarfile.open(archive) as bundle:bundle.extractall(source,filter='data')
    old=root/'program-source-v1';before=rust_files(old);after=rust_files(source)
    changed={name for name in before.keys()|after.keys() if before.get(name)!=after.get(name)}
    unrelated=changed-RUST
    if changed&RUST!=RUST or unrelated not in [set(),{ASSESSMENT_SCRIPT}]:raise ValueError('compiled Rust/Cargo changes exceed the bounded ingress repair: '+str(sorted(changed)))
    # This standalone assessment adapter is not a Cargo member or included by
    # any workspace Rust source. Preserve and report it; it is not numerical
    # training code and must not be misreported as an unchanged repository file.
    if unrelated:
        import tomllib
        members=tomllib.loads((source/'Cargo.toml').read_text())['workspace']['members']
        if any('shipgates_direction' in p.read_text() for member in members for p in (source/member).rglob('*.rs')):
            raise ValueError('assessment adapter unexpectedly enters compiled workspace Rust')
    if any(sha(Path.cwd()/name)!=value for name,value in after.items()):raise ValueError('compiled checkout Rust/Cargo differs from frozen successor')
    changes={}
    for name in sorted(changed):
        patch=out/(Path(name).name+'.patch')
        prior=(old/name).read_text().splitlines(True) if (old/name).exists() else []
        patch.write_text(''.join(difflib.unified_diff(prior,(source/name).read_text().splitlines(True),fromfile='original/'+name,tofile='successor/'+name)))
        changes[name]=dict(original_sha256=before.get(name),successor_sha256=after[name],patch=pin(patch),compiled_ingress=name in RUST)
    old_preflight=root/'program-closure-preflight-v1.json';previous=json.loads(old_preflight.read_text())
    files={name:sha(source/name) for name in previous['files']}
    python_changed=[name for name,value in files.items() if value!=previous['files'][name]]
    if python_changed!=['scripts/rev4_featpot/e36_training.py']:raise ValueError('frozen runtime closure changed beyond its adapter')
    native=root/'fit-native-owner-v3';native.mkdir()
    native_tar=root/'native-owner-66ac610c.tar'
    extract_native(native_tar,native)
    write(native/'SOURCE.json',dict(build_commit=NATIVE,source='imazen/zenmetrics',purpose='successor native full-fit and distinct archive-bound diagnostic execution; no fits'))
    owner_receipt=root/'fit-native-owner-v3.json'
    write(owner_receipt,dict(build_commit=NATIVE,files={str(p.relative_to(native)):sha(p) for p in native.rglob('*') if p.is_file()},archive=pin(native_tar),fits=0,utc=time.time()))
    binaries=root/'bin/e36-fit-integration-v2';binaries.mkdir()
    for name in ['zensim_mlp_train','panel','bake_dial_refit','check_holdout_overlap','inspect_qualified_checkpoint','serve_custom_bake']:
        input_path=target/'release'/('examples' if name in ['inspect_qualified_checkpoint','serve_custom_bake'] else '')/name
        shutil.copy2(input_path,binaries/name)
    proof=dict(schema='e36-ingress-integration-v1',source_commit=SOURCE,native_source_commit=NATIVE,
        predecessor_source=pin(root/'program-source-preparation-v1.json'),predecessor_closure=pin(old_preflight),
        source_archive=pin(archive),native_owner=pin(owner_receipt),rust_changes=changes,
        unchanged_other_rust_cargo_files=len(after)-len(changed),all_other_workspace_rust_cargo_bytes_unchanged=True,
        separately_reported_nonworkspace_assessment_changes=sorted(unrelated),
        runtime_python_changes=python_changed,numerical_optimizer_tv_feature_sources_unchanged=True,
        original_protocol_sha256=previous['protocol_source_sha256'],successor_protocol_sha256=protocol_source_sha(source),
        binaries={name:pin(path) for name,path in ((p.name,p) for p in binaries.iterdir())},
        scientific_change=False,labels_read=False,fits=0,utc=time.time())
    write(out/'receipt.json',proof)
    preparation=dict(schema='e36-published-program-source-preparation-v1',published_source=SOURCE,compiled_rust_source=SOURCE,
        exact_rust_cargo_equivalence=True,archive=pin(archive),extracted_source=str(source),ingress_integration=pin(out/'receipt.json'),fits=0,utc=time.time())
    write(root/'program-source-preparation-v2.json',preparation)
    preflight=copy.deepcopy(previous);preflight.update(published_source=SOURCE,source=str(source),files=files,
        protocol_source_sha256=proof['successor_protocol_sha256'],ingress_integration=pin(out/'receipt.json'),utc=time.time())
    write(root/'program-closure-preflight-v2.json',preflight)
    return proof

def probe(root,pool,out):
    root=Path(root);pool=Path(pool);out=Path(out)
    matrix=json.loads((pool/'matrix-admission.json').read_text())
    if matrix['matrix']!='full' or matrix['completed_views']!=12684:raise ValueError('entire approved full matrix required')
    amendment=native_source_amendment(matrix)
    if amendment is None:raise ValueError('exact source-budget amendment required')
    out.mkdir();manifest=json.loads((pool/'fit/manifest.json').read_text())
    manifest['source_budget_amendment']=amendment;write(out/'manifest.json',manifest)
    contract=json.loads((pool/'fit/contract.json').read_text())
    contract['manifest_sha256']=sha(out/'manifest.json');write(out/'contract.json',contract)
    value=dict(schema='e36-nontraining-ingress-probe-v1',prepared_pool=pin(pool/'matrix-admission.json'),manifest=pin(out/'manifest.json'),contract=pin(out/'contract.json'),fit_pool_admitted=False,optimizer_steps=0,labels_read=False,fits=0)
    write(out/'PROBE.json',value);return value

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','probe']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--target',type=Path);parser.add_argument('--pool',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    print(json.dumps(freeze(args.root,args.target) if args.action=='freeze' else probe(args.root,args.pool,args.out)))
