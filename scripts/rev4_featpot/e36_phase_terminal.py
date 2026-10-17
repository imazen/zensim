"""Bind terminal native optional-phase evidence without running a fit."""
import argparse
import json
import os
from pathlib import Path
import time
from e36_store import sha


def pin(path):return dict(path=str(path),sha256=sha(path))


def validate(root,path):
    root=Path(root);value=json.loads(Path(path).read_text())
    if value['schema']!='e36-optional-native-terminal-v1' or value['closed'] is not True or value['fits']!=0:raise ValueError('optional native terminal proof required')
    documents={}
    for name in ['launch','exit','helper_exits','log','sources','runtime_audit']:
        source=Path(value[name]['path'])
        if sha(source)!=value[name]['sha256']:raise ValueError('optional terminal proof changed')
        if name!='log':documents[name]=json.loads(source.read_text())
    launch=documents['launch'];exits=documents['helper_exits']['exits'];audit=documents['runtime_audit']
    if launch['phase']!='additions' or launch['deadline_unix']!=1792274400. or launch['fits']!=0 or len(exits)!=2 or any(row['returncode']!=0 for row in exits):raise ValueError('optional phase/clock/helper closure differs')
    if audit['admitted'] is not True or len(set(audit['executors']))!=24 or len(audit['executors'])!=24 or audit['fits']!=0 or audit['manual_timer_mutations']!=0 or audit['fresh_deadline_unix']!=1792274400. or audit['launch_sha256']!=value['launch']['sha256'] or audit['original_numerical_runtime_sha256']!=sha(root/'runtime-v6/runtime.json'):raise ValueError('optional actual runtime audit differs')
    if value['returncode']!=documents['exit']['returncode']:raise ValueError('optional native terminal returncode differs')
    for entry in documents['sources']['files'].values():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('optional controller source closure changed')
    for entry in value['runtime_receipts']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('optional executor audit changed')
        doc=json.loads(Path(entry['path']).read_text())
        if doc['admitted'] is not True or doc['runtime_sha256']!=audit['original_numerical_runtime_sha256'] or doc['deadline_unix']!=1792274400. or doc['fits']!=0 or doc['labels_read'] is not False:raise ValueError('optional executor audit identity differs')
    if {json.loads(Path(entry['path']).read_text())['pid'] for entry in value['runtime_receipts']}!=set(audit['executors']):raise ValueError('optional runtime receipts omit an executor')
    return value


def close(root,out,log,audit):
    root=Path(root);out=Path(out);launch=json.loads((out/'launch.json').read_text())
    p=Path('/proc')/str(launch['controller_pid'])
    if p.exists() and b'e36_generation_resume.py' in (p/'cmdline').read_bytes():raise ValueError('optional controller remains active')
    helpers=json.loads((out/'helper-exits.json').read_text());expected={json.loads((out/(role+'-ready.json')).read_text())['pid'] for role in ['mirror','monitor']}
    if {row['pid'] for row in helpers['exits']}!=expected:raise ValueError('optional helper ownership differs')
    if not any(line.startswith('run-heavy: done rc=') for line in Path(log).read_text().splitlines()):raise ValueError('optional run-heavy terminal line missing')
    source_manifest=Path(launch['source']['path']).parent/'sources.json';audit=Path(audit)
    value=dict(schema='e36-optional-native-terminal-v1',launch=pin(out/'launch.json'),exit=pin(out/'exit.json'),helper_exits=pin(out/'helper-exits.json'),log=pin(log),sources=pin(source_manifest),runtime_audit=pin(audit),runtime_receipts=[pin(p) for p in sorted(audit.parent.glob('executor-*.json'))],returncode=json.loads((out/'exit.json').read_text())['returncode'],closed=True,fits=0,utc=time.time())
    path=root/'owner-svt-ssd-v1/native-additions-final-v1.json'
    # Validate all evidence before publishing the terminal admission.
    temporary=out/'terminal-validation.json'
    with temporary.open('x') as stream:json.dump(value,stream,indent=2)
    validate(root,temporary)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return pin(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','out','log','audit']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(close(a.root,a.out,a.log,a.audit)))
