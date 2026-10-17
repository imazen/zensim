"""Finalize prepared feature pools after unchanged full artifact admission."""
import argparse
import json
import os
from pathlib import Path
from e36_matrix import admit
from e36_store import sha


def load(path):return json.loads(Path(path).read_text())

def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def finalize(root,prepared,out,commit):
    root=Path(root);prepared=Path(prepared);out=Path(out)
    old=load(prepared/'matrix-admission.json')
    if old.get('artifact_archive_pending') is not True:raise ValueError('explicit archive-pending preparation required')
    current=admit(root,old['selected_runs'])
    if current.get('artifact_archive_pending') is not False or not current.get('artifact_location'):raise ValueError('final artifact admission required')
    for key in ['matrix','selected_runs','manifests','completed_views','outputs','clock','plan']:
        if old[key]!=current[key]:raise ValueError('prepared scientific matrix changed: '+key)
    metadata=load(prepared/'_MANIFEST.json')
    if not metadata['required_support_admitted'] or metadata['zero_support_required_strata'] or metadata['fits']!=0 or metadata['labels_read'] is not False:raise ValueError('prepared support/label discipline differs')
    for split in ['fit','selection']:
        support=load(prepared/split/'support.json');semantic=load(prepared/split/'semantic-admission.json')
        if support['admitted'] is not True or support['zero_support_required_strata']!=0 or semantic['admitted'] is not True or semantic['fits']!=0:raise ValueError('prepared semantic/support admission required')
    out.mkdir();done={};active=set();verified={}
    def checked(path,expected):
        if path not in verified:verified[path]=sha(path)
        actual=verified[path]
        if actual!=expected:raise ValueError('prepared proof/data changed: '+str(path))
    def transform(value):
        if isinstance(value,list):return [transform(v) for v in value]
        if not isinstance(value,dict):return value
        answer={k:transform(v) for k,v in value.items()}
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            path=Path(value['path']);checked(path,value['sha256'])
            if path.is_relative_to(prepared):
                target=copy_file(path.relative_to(prepared));answer['path']=str(target);answer['sha256']=sha(target)
        return answer
    def copy_file(relative):
        if relative in done:return done[relative]
        if relative in active:raise ValueError('cyclic prepared proof graph')
        active.add(relative);source=prepared/relative;target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if relative==Path('matrix-admission.json'):write(target,current)
        elif source.suffix=='.json' and source.name!='fx1.json':
            value=transform(load(source))
            if relative==Path('_MANIFEST.json'):
                value['whole_matrix_sha256']=sha(copy_file(Path('whole-matrix.json')))
                value['artifact_archive_pending']=False
            write(target,value)
        else:os.link(source,target)
        active.remove(relative);done[relative]=target;return target
    for name in ['matrix-admission.json','whole-matrix.json','_MANIFEST.json','alias-memberships.parquet']:
        copy_file(Path(name))
    for split in ['fit','selection']:
        for name in ['features.parquet','pairs.parquet','edge-proof.parquet','support.json','manifest.json','semantic-admission.json','contract.json','fx1.json']:
            copy_file(Path(split)/name)
    receipt=dict(schema='e36-final-prepared-pool-admission-v1',build_commit=commit,prepared_matrix=dict(path=str(prepared/'matrix-admission.json'),sha256=sha(prepared/'matrix-admission.json')),final_matrix=dict(path=str(out/'matrix-admission.json'),sha256=sha(out/'matrix-admission.json')),scientific_matrix_unchanged=True,feature_and_pair_bytes_unchanged=True,artifact_archive_pending=False,fits=0,labels_read=False,files={str(k):sha(v) for k,v in sorted(done.items())})
    write(out/'FINAL_POOL_ADMISSION.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','prepared','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();print(json.dumps(finalize(a.root,a.prepared,a.out,a.commit)))
