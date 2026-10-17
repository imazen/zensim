"""Admit the applied storage-only pack source chain, never scientific data."""
import json
from pathlib import Path
from e36_store import sha


def checked(pin):
    path=Path(pin['path'])
    if sha(path)!=pin['sha256']:raise ValueError('applied pack proof changed')
    return path


def admit(root):
    root=Path(root);owner=root/'owner-svt-ssd-v1';path=owner/'archive-pack-binding-v2.json';doc=json.loads(path.read_text())
    if doc['schema']!='e36-archive-pack-applied-binding-v1' or doc['fits']!=0 or doc['scientific_change'] is not False or doc['manual_kills']!=0 or doc['deadline_unix']!=1792274400.:raise ValueError('pack resource identity differs')
    expected_previous={owner/'archive-budget4-binding.json',owner/'archive-transport-v3-binding.json'}
    if {checked(p) for p in doc['previous']}!=expected_previous:raise ValueError('pack predecessor chain differs')
    first=json.loads((owner/'archive-budget4-binding.json').read_text());handoff=json.loads(checked(first['handoff']).read_text())
    expected={r['new_pid'] for r in handoff['history']}
    for item in doc['resources']:
        sources=json.loads(checked(item['sources']).read_text())
        for p in sources['files'].values():checked(p)
        record=json.loads(checked(item['handoff']).read_text())
        if record['controller_pid']!=4053123 or record['scientific_change'] is not False or record['fits']!=0 or record['manual_kills']!=0 or record['deadline_unix']!=1792274400. or {r['old_pid'] for r in record['history']}!=expected or any(r['old_returncode']!=0 for r in record['history']):raise ValueError('pack cooperative helper source chain differs')
        expected={r['new_pid'] for r in record['history']}
    resolver=checked(doc['latest_resolver']);runtime_sha=sha(root/'runtime-v6/runtime.json');pids=set()
    for p in doc['live_runtime_audits']:
        record=json.loads(checked(p).read_text())
        if record['pid'] in pids or record['admitted'] is not True or record['fits']!=0 or record['labels_read'] is not False or record['runtime_sha256']!=runtime_sha or record['deadline_unix']!=1792274400. or record['resolver_owner']!=str(resolver):raise ValueError('latest24 runtime binding differs')
        pids.add(record['pid'])
    if len(pids)!=24:raise ValueError('pack requires all24 numerical audits')
    for p in doc['admission']+doc['worker_bootstrap']:checked(p)
    latest_sources=checked(doc['latest_sources'])
    if latest_sources!=Path(doc['resources'][-1]['sources']['path']):raise ValueError('future pack bootstrap source differs')
    from e36_artifact_location import require_pack_directory_proof
    require_pack_directory_proof(root)
    return dict(path=str(path),sha256=sha(path)),latest_sources.parent


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();proof,resources=admit(a.root);print(json.dumps(dict(admitted=True,proof=proof,resources=str(resources),fits=0,qualifies_data=False)))
