"""Private E36 recipe/filter adapter; label payloads stay inside fit-cell-exec."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from e36_store import sha

ARMS={'tv0-c':0.,'tv1-c':1.,'tv10-c':10.}
SPEC='sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1'
SOURCES=['kadid','tid2013','konfig','cid22_a25']


def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')


def base_id(value):
    name=Path(str(value)).name.rsplit(':',1)[-1]
    match=re.fullmatch(r'([0-9a-f]{16})(?:_(?:[0-9]+sq|[0-9]+x[0-9]+))?(?:\.[A-Za-z0-9]+)?',name)
    return match.group(1) if match else None


def key_mask(keys,excluded):
    columns=[n for n in ['ref_basename','source_filename','family','ladder'] if n in keys.column_names]
    if not columns:raise ValueError('E36 base keys lack reference identity')
    drop=np.zeros(keys.num_rows,dtype=bool)
    for n in columns:drop |= np.asarray([base_id(v) in excluded for v in keys[n].to_pylist()])
    return ~drop


def plan(root,exclusions,inventory,out):
    # This phase reads declarations and label-free key sidecars only.
    from v2_teacher import key_path,row_keys_sha,admit_ordered_keys
    from v2c_wide import safe_path
    root=safe_path(root);x=json.loads(exclusions.read_text());audit=json.loads(inventory.read_text())
    if x['schema']!='e36-train-selection-base-exclusion-v1' or x['labels_read'] is not False:raise ValueError('E36 exclusion schema')
    ids={base_id(m['path']) for r in x['rows'] for m in r['matches']}
    if None in ids or not ids:raise ValueError('unresolved relevant base family identity')
    family_files=[r for r in audit['training'] if base_id(r['path']) in ids]
    if not family_files or not all(any(f['path']==m['path'] and f['sha256']==m['sha256'] for f in family_files) for r in x['rows'] for m in r['matches']):raise ValueError('base family pixel audit binding')
    rp=root/'wide/main/real/receipt.json';receipt=json.loads(rp.read_text());legs=receipt['legs'];records=[];paths=set()
    required=['safesyn','cid22','human_all',*['human_without_'+s for s in SOURCES]]
    for name in required:
        for split in ['fit','dev']:
            rec=legs[name][split];p=safe_path(root/rec['rel'])
            if p in paths:continue
            paths.add(p);side=Path(str(p)+'.manifest.json');d=json.loads(side.read_text());kp=key_path(p)
            if sha(side)!=rec['manifest_sha256'] or sha(kp)!=d['keys_sha256']:raise ValueError('base declaration/key pin changed')
            keys=pq.read_table(kp);admit_ordered_keys(p,d,keys,rec['rows']);keep=key_mask(keys,ids)
            records.append(dict(name=name,split=split,path=str(p),table_sha256=d['table_sha256'],manifest_sha256=sha(side),
                keys_sha256=sha(kp),row_keys_sha256=row_keys_sha(keys),rows=keys.num_rows,removed=int((~keep).sum()),
                kept_indices_sha256=hashlib.sha256(np.flatnonzero(keep).astype('<u8').tobytes()).hexdigest()))
    # The coverage view is selected later by the unchanged E34 cf98 owner.
    p=root/'e15/coverage_pool.parquet';side=Path(str(p)+'.manifest.json');d=json.loads(side.read_text());kp=key_path(p)
    keys=pq.read_table(kp)
    # The immutable ordinal pool has no row-count field in its table sidecar.
    # Bind the independent Parquet footer count explicitly to the key owner.
    count=pq.ParquetFile(p).metadata.num_rows
    admit_ordered_keys(p,d,keys,count);keep=key_mask(keys,ids)
    if not keep.all():raise ValueError('E36 coverage needs a separately frozen composed-filter admission')
    records.append(dict(name='coverage',split='fit',path=str(p),table_sha256=d['table_sha256'],manifest_sha256=sha(side),keys_sha256=sha(kp),
        row_keys_sha256=row_keys_sha(keys),rows=keys.num_rows,removed=0))
    write(out,dict(schema='e36-matched-base-filter-v2',identity_normalization='member-prefix+rendition-v2',labels_read=False,complete=True,root=str(root),root_receipt_sha256=sha(rp),
        excluded_base_ids=sorted(ids),family_files=family_files,selection_sha256=x['selection_sha256'],
        exclusion_receipt_sha256=sha(exclusions),pixel_inventory_sha256=sha(inventory),groups=records,
        weights='preserve exact E34 numeric group weights; same source mask in all three arms',targets_changed=0))
    print(json.dumps(dict(groups=len(records),removed_by_group={r['name']+':'+r['split']:r['removed'] for r in records},labels_read=False)))


def curated_projection(mask_path,out):
    """Freeze child key/declaration projections without reading target columns."""
    from v2_teacher import key_path,row_keys_sha,selection_sha,admit_ordered_keys
    mask=json.loads(mask_path.read_text())
    if mask['schema']!='e36-matched-base-filter-v2' or mask['labels_read'] is not False:
        raise ValueError('E36 exact metadata-only mask required')
    out.mkdir();records={}
    for r in mask['groups']:
        if not r['removed']:continue
        source=Path(r['path']);side=Path(str(source)+'.manifest.json');d=json.loads(side.read_text())
        if sha(side)!=r['manifest_sha256'] or sha(key_path(source))!=r['keys_sha256']:raise ValueError('E36 parent projection changed')
        keys=pq.read_table(key_path(source));admit_ordered_keys(source,d,keys,r['rows'])
        keep=key_mask(keys,set(mask['excluded_base_ids']));indices=np.flatnonzero(keep)
        if int((~keep).sum())!=r['removed'] or hashlib.sha256(indices.astype('<u8').tobytes()).hexdigest()!=r['kept_indices_sha256']:
            raise ValueError('E36 projected selection differs')
        selected=keys.filter(pa.array(keep));kp=out/(r['table_sha256']+'.keys.parquet');pq.write_table(selected,kp,compression='zstd')
        child={**d,'source_table_sha256':d['table_sha256'],'source_manifest_sha256':sha(side),
            'source_row_selection_sha256':d.get('row_selection_sha256'),'source_row_keys_sha256':d.get('row_keys_sha256'),
            'rows_in':r['rows'],'rows_kept':len(indices),'row_selection_sha256':selection_sha(indices),
            'targets_changed':0,'keys_sha256':sha(kp),'row_keys_sha256':row_keys_sha(selected),
            'e36_base_filter_sha256':sha(mask_path)}
        child.pop('table_sha256')
        records[r['table_sha256']]=dict(removed=r['removed'],manifest=child)
    write(out/'projection.json',dict(schema='e36-curated-declaration-projection-v1',plan_sha256=sha(mask_path),tables=records,labels_read=False))
    return records


def assessment_plan(base_root,out):
    """Freeze original E34 row/reference/codec metadata without target reads."""
    from e34_evaluate import admit_tables
    from v2_teacher import key_path,row_keys_sha
    rp=base_root/'wide/main/real/receipt.json';receipt=json.loads(rp.read_text());tables={}
    for fold,path,reference,codec in admit_tables(base_root):
        r=receipt['legs'][fold]['full'];kp=key_path(path)
        tables[fold]=dict(path='/var/tmp/rev4-featpot/v2e29/'+r['rel'],table_sha256=r['sha256'],
            manifest_sha256=r['manifest_sha256'],keys_sha256=r['keys_sha256'],row_keys_sha256=row_keys_sha(pq.read_table(kp)),
            rows=r['rows'],reference=reference,codec=codec,orientation=1.)
    write(out,dict(schema='e36-frozen-assessment-inputs-v1',base_root_receipt_sha256=sha(rp),tables=tables,
        labels_read=False,metadata_owner_sha256=sha(Path(__file__).with_name('e34_evaluate.py'))))


def add_args(p):
    p.add_argument('--e36-arm',choices=list(ARMS))
    p.add_argument('--e36-tv-manifest',type=Path);p.add_argument('--e36-tv-contract',type=Path)
    p.add_argument('--e36-tv-contract-sha256');p.add_argument('--e36-base-filter-plan',type=Path)
    p.add_argument('--e36-base-filter-sha256')
    p.add_argument('--e36-overlay',type=Path);p.add_argument('--e36-overlay-sha256')


def options(args,recipe):
    if getattr(args,'e36_arm',None) is None:return recipe
    if (args.spec!=SPEC or args.head!='N' or not args.strict_admission or not args.train_only
        or recipe.get('hidden')!=128 or recipe.get('derived_inputs')!='fx1'
        or getattr(args,'steer_pairs',None) is not None or getattr(args,'steer_lambda',0)!=0
        or getattr(args,'local_smoke_budget',None) is not None or getattr(args,'monotone_cbc',False) or getattr(args,'heldout',None) not in [None,*SOURCES]):
        raise ValueError('E36 requires the exact strict C/plain-N H128 base recipe, train-only and four-source folds')
    for n in ['e36_tv_manifest','e36_tv_contract','e36_base_filter_plan']:
        if getattr(args,n,None) is None:raise ValueError('E36 missing '+n)
    if sha(args.e36_tv_contract)!=args.e36_tv_contract_sha256 or sha(args.e36_base_filter_plan)!=args.e36_base_filter_sha256:
        raise ValueError('E36 immutable contract/base-filter digest differs from fit argv')
    contract=json.loads(args.e36_tv_contract.read_text());mask=json.loads(args.e36_base_filter_plan.read_text())
    if contract['base_filter_sha256']!=args.e36_base_filter_sha256 or contract['manifest_sha256']!=sha(args.e36_tv_manifest) or mask['schema']!='e36-matched-base-filter-v2' or mask['identity_normalization']!='member-prefix+rendition-v2' or mask['labels_read'] is not False or mask['complete'] is not True:
        raise ValueError('E36 pool or matched base-filter incomplete')
    result=dict(recipe);result['e36']=dict(arm=args.e36_arm,weight=ARMS[args.e36_arm],
        manifest=str(args.e36_tv_manifest),contract=str(args.e36_tv_contract),contract_sha256=args.e36_tv_contract_sha256,
        base_filter=str(args.e36_base_filter_plan),base_filter_sha256=args.e36_base_filter_sha256)
    return result


def command_flags(recipe,trainer):
    e=recipe.get('e36')
    if e is None:return []
    contract=json.loads(Path(e['contract']).read_text())
    if sha(trainer)!=contract['trainer_sha256']:raise ValueError('E36 actual trainer differs from frozen three-arm build')
    return ['--e36-tv-manifest',e['manifest'],'--e36-tv-contract',e['contract'],'--e36-tv-contract-sha256',e['contract_sha256'],
        '--tv-weight',str(e['weight']),'--tv-apply-every','50','--tv-batch','32','--tv-margin','0.05']


def filter_groups(groups,recipe,scratch):
    e=recipe.get('e36')
    if e is None:return None
    # Enforced before any target column is read. The actual executor separately
    # binds this program SHA/data SHA/argv SHA to the immutable native FitCell.
    if not re.fullmatch('[0-9a-f]{64}',os.environ.get('ZEN_FIT_PROGRAM_SHA','')):
        raise ValueError('E36 base target transport is allowed only inside fit-cell-exec')
    from v2_teacher import key_path,write_curated,row_keys_sha,admit_ordered_keys,selection_sha
    p=Path(e['base_filter']);mask=json.loads(p.read_text())
    if mask['schema']!='e36-matched-base-filter-v2' or mask['identity_normalization']!='member-prefix+rendition-v2' or sha(p)!=e['base_filter_sha256']:raise ValueError('E36 base filter changed')
    excluded=set(mask['excluded_base_ids']);records=[];result=[];views={}
    cache=Path(scratch)/'e36-base-views';cache.mkdir(exist_ok=True)
    for name,source,tw,vw,mode in groups:
        source=Path(source);side=Path(str(source)+'.manifest.json');d=json.loads(side.read_text());keys=pq.read_table(key_path(source))
        admit_ordered_keys(source,d,keys);keep=key_mask(keys,excluded)
        parent=next((r for r in mask['groups'] if r['table_sha256']==d['table_sha256']),None)
        if parent is None:
            # Only the existing unchanged cf98 coverage selection is composed.
            if name!='coverage' or not keep.all() or not any(r['table_sha256']==d.get('source_table_sha256') and r['name']=='coverage' for r in mask['groups']):
                raise ValueError('E36 runtime base group is not a registered parent/coverage view')
        elif (parent['keys_sha256']!=sha(key_path(source)) or parent['row_keys_sha256']!=row_keys_sha(keys) or parent['removed']!=int((~keep).sum())
              or ('kept_indices_sha256' in parent and parent['kept_indices_sha256'] != hashlib.sha256(np.flatnonzero(keep).astype('<u8').tobytes()).hexdigest())):
            raise ValueError('E36 runtime key/source removal differs from frozen plan')
        if not keep.any():raise ValueError('E36 cannot silently empty a base group')
        dest=source;record=dict(name=name,source_table_sha256=d['table_sha256'],removed=int((~keep).sum()),targets_changed=0)
        if not keep.all():
            digest=hashlib.sha256((d['table_sha256']+e['base_filter_sha256']).encode()).hexdigest();dest=cache/(digest+'.parquet')
            with (cache/(digest+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                manifest=Path(str(dest)+'.manifest.json')
                if dest.exists():
                    previous=json.loads(manifest.read_text())
                    if previous.get('e36_base_filter_sha256')!=e['base_filter_sha256'] or previous['table_sha256']!=sha(dest) or previous['targets_changed']!=0:
                        raise ValueError('E36 cached base view changed or incomplete; preserve it and diagnose')
                else:
                    # Existing owner streams unchanged target values and binds
                    # ordered source rows, parent table/manifest and keys.
                    targets=pq.read_table(source,columns=['human_score'])['human_score'].to_numpy()
                    row=write_curated(source,dest,keep,targets)
                    if row['targets_changed']!=0:raise ValueError('E36 source filter changed a target')
                    previous=json.loads(manifest.read_text());previous['e36_base_filter_sha256']=e['base_filter_sha256']
                    manifest.rename(manifest.with_suffix(manifest.suffix+'.before-e36'))
                    write(manifest,previous)
            if (previous.get('source_table_sha256')!=d['table_sha256'] or previous.get('source_manifest_sha256')!=sha(side)
                or previous.get('source_row_selection_sha256')!=d.get('row_selection_sha256')
                or previous.get('source_row_keys_sha256')!=d.get('row_keys_sha256')
                or previous.get('row_selection_sha256')!=selection_sha(np.flatnonzero(keep))
                or previous.get('row_keys_sha256')!=row_keys_sha(keys.filter(pa.array(keep)))
                or previous.get('keys_sha256')!=sha(key_path(dest)) or previous.get('rows_in')!=len(keep)
                or previous.get('rows_kept')!=int(keep.sum())):
                raise ValueError('E36 cached/fresh child selection differs from exact registered parent mask')
            record.update(table_sha256=sha(dest),manifest_sha256=sha(manifest),stored_manifest=previous)
        result.append((name,dest,tw,vw,mode));records.append(record);views[str(source)]=dict(path=str(dest),sha256=sha(dest))
    groups[:]=result
    e['base_views']=views
    return dict(schema='e36-applied-base-filter-v1',plan_sha256=e['base_filter_sha256'],groups=records,targets_changed=0,
        exact_group_weights_preserved=True)

def model_binding(dest,recipe,args,init_seed,sample_seed,bake,packed=None,selection=None):
    e=recipe.get('e36')
    if e is None:return
    if selection is None or selection.get('epoch_rule')!='last' or selection.get('selected_epoch')!=119:raise ValueError('E36 projection requires actual final epoch119 selection')
    program=os.environ.get('ZEN_FIT_PROGRAM_SHA','')
    if not re.fullmatch('[0-9a-f]{64}',program):raise ValueError('E36 model projection requires fit-cell-exec')
    contract=json.loads(Path(e['contract']).read_text())
    if sha(e['contract'])!=e['contract_sha256'] or sha(e['base_filter'])!=e['base_filter_sha256']:raise ValueError('E36 projection input changed')
    packed=packed or {}
    write(Path(dest)/'e36-model-binding.json',dict(schema='e36-fit-model-binding-v1',arm=e['arm'],tv_weight=e['weight'],
        role='source-withheld' if getattr(args,'heldout',None) else 'full-data',heldout=getattr(args,'heldout',None),seed_index=args.seed_index,
        init_seed=init_seed,sample_seed=sample_seed,selected_epoch=119,spec=SPEC,head='N',
        program_sha256=program,trainer_sha256=contract['trainer_sha256'],manifest_sha256=sha(e['manifest']),contract_sha256=e['contract_sha256'],
        base_filter_sha256=e['base_filter_sha256'],raw_checkpoint_sha256=sha(bake),packed_model_sha256=packed.get('packed_model_sha256'),
        deployment_order=packed.get('deployment_order'),label_payload_read_by_projection=False,contains_labels=False,projection_only=True))


def calibration_anchor(path,recipe):
    e=recipe.get('e36')
    if e is None:return path
    if not re.fullmatch('[0-9a-f]{64}',os.environ.get('ZEN_FIT_PROGRAM_SHA','')):raise ValueError('E36 calibration transport requires fit-cell-exec')
    if sha(e['base_filter'])!=e['base_filter_sha256']:raise ValueError('E36 calibration mask changed')
    view=e.get('base_views',{}).get(str(path))
    if view is None or sha(view['path'])!=view['sha256']:raise ValueError('E36 calibration must use its exact matched filtered base view')
    return Path(view['path'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--exclusions',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan(a.root,a.exclusions,a.inventory,a.out)


def bootstrap(args):
    """Hydrate the unchanged opaque base archive only inside fit-cell-exec.

    Reuse its own checksum/atomic extraction/cache/link owners. No target
    column is decoded here; source filtering and trainer reads happen later.
    """
    if getattr(args,'e36_arm',None) is None:return
    if getattr(args,'local_smoke_budget',None) is not None:raise ValueError('E36 smoke fits are not part of the registered protocol')
    if not re.fullmatch('[0-9a-f]{64}',os.environ.get('ZEN_FIT_PROGRAM_SHA','')):
        raise ValueError('E36 training bootstrap requires fit-cell-exec; fits remain separately released')
    if args.e36_overlay is None or sha(args.e36_overlay)!=args.e36_overlay_sha256:
        raise ValueError('E36 immutable overlay metadata differs from fit argv')
    overlay=json.loads(args.e36_overlay.read_text())
    if overlay['schema']!='e36-fit-data-overlay-v1' or overlay['base_data_sha256']!='9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008':
        raise ValueError('E36 requires the frozen E34 base data archive')
    folder=args.e36_overlay.parent
    base=folder/overlay['base_archive_relative']
    if sha(base)!=overlay['base_data_sha256']:raise ValueError('opaque E34 base archive changed')
    import importlib.util
    program=Path(__file__).resolve().parents[2]
    owner=program/'fit_cell_exec.py'
    module_spec=importlib.util.spec_from_file_location('e36_native_fit_data_owner',owner)
    native=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(native)
    previous=os.environ.get('ZEN_FIT_DATA_LOCAL')
    try:
        os.environ['ZEN_FIT_DATA_LOCAL']=str(base)
        extracted=native.fetch_data(overlay['base_data_sha256'])
    finally:
        if previous is None:os.environ.pop('ZEN_FIT_DATA_LOCAL',None)
        else:os.environ['ZEN_FIT_DATA_LOCAL']=previous
    # The overlay's native root is already bound by the executor. Fill its
    # otherwise absent base children with verified immutable symlinks.
    data=folder.parent.parent
    for name in ['rev4-featpot','rev4-featbank','gmsbank']:
        source=extracted/name
        if not source.is_dir():continue
        target=data/name;target.mkdir(exist_ok=True)
        for child in source.iterdir():native.ensure_link(target/child.name,child,'E36 frozen base member')
    native.bind_expected_paths(data)
    expected=Path(overlay['expected_native_root']);expected.parent.mkdir(parents=True,exist_ok=True)
    native.ensure_link(expected,folder/'native','E36 native feature-only inputs')
    # The public-safe plan was made from label-free keys on dev. Its root
    # receipt must be byte-identical to the existing archive's actual root.
    base_receipt=data/'rev4-featpot/v2e29/wide/main/real/receipt.json'
    if sha(base_receipt)!=overlay['base_root_receipt_sha256']:
        raise ValueError('E36 metadata plan and frozen E34 base root differ')
