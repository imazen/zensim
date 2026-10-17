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
    if len(helpers['exits'])!=2 or {item['pid'] for item in helpers['exits']}!=expected or any(item['returncode']!=0 for item in helpers['exits']):
        raise ValueError('required resource helpers did not close successfully')
    if (owner/'resource-failure-hold.json').exists():raise ValueError('unresolved native resource failure')
    log=root/'GENERATION-continuation-v1.log'
    if not any(line.startswith('run-heavy: done rc=') for line in log.read_text().splitlines()):
        raise ValueError('full run-heavy terminal resource record missing')
    pin=lambda path:dict(path=str(path),sha256=sha(path))
    result=dict(schema='e36-native-continuation-final-v1',continuation=pin(owner/'native-continuation-binding-v1.json'),
        supervision=pin(owner/'continuation-supervision-v2-binding.json'),runtime_audit=pin(owner/'runtime-audit-admission-v1.json'),
        log=pin(log),exit=pin(directory/'exit.json'),helper_exits=pin(directory/'helper-exits.json'),
        returncode=exit_record['returncode'],fits=0,scientific_change=False,closed=True,
        note='Terminal control evidence only; complete registered matrix admission remains separate.')
    with (owner/'native-continuation-final-v1.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    close(parser.parse_args().root)
