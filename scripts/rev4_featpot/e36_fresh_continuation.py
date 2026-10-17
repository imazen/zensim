"""Preserve a fresh-clock continuation without changing native job identities."""
import argparse
import json
from pathlib import Path
import time
from e36_store import sha


def pin(path):return dict(path=str(path),sha256=sha(path))
def checked(value):
    path=Path(value['path'])
    if sha(path)!=value['sha256']:raise ValueError('fresh continuation evidence changed')
    return path

def validate(root):
    root=Path(root);path=root/'owner-svt-ssd-v1/native-continuation-v3-binding.json';binding=json.loads(path.read_text())
    if binding['schema']!='e36-fresh-native-continuation-v1' or binding['fits']!=0 or binding['scientific_change'] is not False:raise ValueError('fresh continuation identity differs')
    launch=json.loads(checked(binding['launch']).read_text());old=json.loads(checked(binding['previous_terminal']).read_text());audit=json.loads(checked(binding['runtime_audit']).read_text())
    if old['closed'] is not True or old['returncode']!=124 or launch['deadline_unix']!=1792274400. or launch['clock']!=binding['clock'] or sha(root/'runtime-v6/runtime.json')!=binding['runtime']['sha256']:raise ValueError('fresh continuation clock/runtime/predecessor differs')
    if audit['admitted'] is not True or audit['fits']!=0 or len(set(audit['executors']))!=24 or audit['manual_timer_mutations']!=0 or audit['fresh_deadline_unix']!=1792274400. or audit['launch_sha256']!=binding['launch']['sha256'] or audit['original_numerical_runtime_sha256']!=binding['runtime']['sha256']:raise ValueError('actual fresh24-runtime audit differs')
    pids=set()
    for p in binding['runtime_receipts']:
        record=json.loads(checked(p).read_text())
        if record['admitted'] is not True or record['fits']!=0 or record['labels_read'] is not False or record['runtime_sha256']!=binding['runtime']['sha256'] or record['deadline_unix']!=1792274400. or record['resolver_owner']!=str(checked(binding['resolver'])):raise ValueError('fresh actual runtime source differs')
        pids.add(record['pid'])
    if pids!=set(audit['executors']):raise ValueError('fresh runtime receipts omit/duplicate an executor')
    for value in json.loads(checked(binding['sources']).read_text())['files'].values():checked(value)
    checked(binding['clock']);checked(binding['runtime']);return path,binding,launch


def bind(root):
    root=Path(root);owner=root/'owner-svt-ssd-v1';out=root/'native-continuation-v3';audits=owner/'continuation-v3-audit-v1'
    launch=json.loads((out/'launch.json').read_text())
    value=dict(schema='e36-fresh-native-continuation-v1',previous_terminal=pin(owner/'native-continuation-final-v1.json'),launch=pin(out/'launch.json'),
        sources=pin(owner/'continuation-control-v3/sources.json'),runtime_audit=pin(audits/'receipt.json'),runtime_receipts=[pin(p) for p in sorted(audits.glob('executor-*.json'))],
        resolver=pin(owner/'resource-budget4-v7/e36_artifact_location.py'),runtime=pin(root/'runtime-v6/runtime.json'),clock=launch['clock'],output_directory=str(out),log=str(root/'GENERATION-continuation-v3.log'),
        prior_completed_required=launch['prior_completed_required'],fits=0,scientific_change=False,utc=time.time(),note='Fresh original timeout from existing22Z; same native IDs/ledger/CAS, no attempt resets or new time credit.')
    path=owner/'native-continuation-v3-binding.json'
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    validate(root);return pin(path)


def close(root):
    root=Path(root);path,binding,launch=validate(root);out=Path(binding['output_directory']);controller=Path('/proc')/str(launch['controller_pid'])
    if controller.exists() and b'e36_generation_resume.py' in (controller/'cmdline').read_bytes():raise ValueError('fresh native controller still live')
    exit_record=json.loads((out/'exit.json').read_text());helpers=json.loads((out/'helper-exits.json').read_text());expected={json.loads((out/(role+'-ready.json')).read_text())['pid'] for role in ['mirror','monitor']}
    if len(helpers['exits'])!=2 or {r['pid'] for r in helpers['exits']}!=expected or any(r['returncode']!=0 for r in helpers['exits']):raise ValueError('fresh helpers did not close normally')
    log=Path(binding['log'])
    if not any(line.startswith('run-heavy: done rc=') for line in log.read_text().splitlines()):raise ValueError('fresh run-heavy terminal resource record missing')
    value=dict(schema='e36-fresh-native-continuation-final-v1',fresh_binding=pin(path),launch=binding['launch'],exit=pin(out/'exit.json'),helper_exits=pin(out/'helper-exits.json'),log=pin(log),
        returncode=exit_record['returncode'],runtime_audit=binding['runtime_audit'],closed=True,fits=0,scientific_change=False,utc=time.time())
    target=root/'owner-svt-ssd-v1/native-continuation-final-v3.json'
    with target.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return pin(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['bind','close']);parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();print(json.dumps(bind(a.root) if a.operation=='bind' else close(a.root)))
