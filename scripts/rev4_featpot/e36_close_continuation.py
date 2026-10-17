"""Bind actual completed native-control evidence; never qualify partial data."""
import argparse
import json
from pathlib import Path

from e36_store import sha


def close(root):
    owner=root/'owner-svt-ssd-v1';directory=root/'native-continuation-v1'
    launch=json.loads((directory/'launch.json').read_text())
    process=Path('/proc')/str(launch['controller_pid'])
    if process.exists() and b'e36_generation_resume.py' in (process/'cmdline').read_bytes():
        raise ValueError('native continuation controller is still live')
    exit_record=json.loads((directory/'exit.json').read_text())
    helpers=json.loads((directory/'helper-exits.json').read_text())
    expected={json.loads((directory/(role+'-ready.json')).read_text())['pid'] for role in ['monitor','mirror']}
    budget4=owner/'archive-budget4-binding.json'
    if budget4.exists():
        binding=json.loads(budget4.read_text());handoff_pin=binding['handoff']
        if sha(handoff_pin['path'])!=handoff_pin['sha256']:raise ValueError('resource handoff changed')
        handoff=json.loads(Path(handoff_pin['path']).read_text())
        if {item['old_pid'] for item in handoff['history']}!=expected or any(item['old_returncode']!=0 for item in handoff['history']):raise ValueError('old resource owners did not drain')
        expected={item['new_pid'] for item in handoff['history']}
    if len(helpers['exits'])!=2 or {item['pid'] for item in helpers['exits']}!=expected or any(item['returncode']!=0 for item in helpers['exits']):
        raise ValueError('required resource helpers did not close successfully')
    if (owner/'resource-failure-hold.json').exists():raise ValueError('unresolved native resource failure')
    log=root/'GENERATION-continuation-v1.log'
    if not any(line.startswith('run-heavy: done rc=') for line in log.read_text().splitlines()):
        raise ValueError('full run-heavy terminal resource record missing')
    pin=lambda path:dict(path=str(path),sha256=sha(path))
    result=dict(schema='e36-native-continuation-final-v1',continuation=pin(owner/'native-continuation-binding-v1.json'),
        supervision=pin(owner/'continuation-supervision-v2-binding.json'),runtime_audit=pin(owner/('runtime-audit-admission-v2.json' if budget4.exists() else 'runtime-audit-admission-v1.json')),
        log=pin(log),exit=pin(directory/'exit.json'),helper_exits=pin(directory/'helper-exits.json'),
        returncode=exit_record['returncode'],fits=0,scientific_change=False,closed=True,
        note='Terminal control evidence only; complete registered matrix admission remains separate.')
    if budget4.exists():result['archive_budget4']=pin(budget4)
    with (owner/'native-continuation-final-v1.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    close(parser.parse_args().root)
