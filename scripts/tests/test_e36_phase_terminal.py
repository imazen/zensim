"""Optional phase proof refuses omitted executors and changed terminal evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_phase_terminal import validate,pin


class OptionalTerminal(unittest.TestCase):
    def fixture(self,root):
        def put(name,data):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return pin(path)
        runtime=put('runtime-v6/runtime.json',dict(fixture=True))
        launch=put('phase/launch.json',dict(phase='additions',deadline_unix=1792274400.,fits=0))
        audit=put('audit/receipt.json',dict(admitted=True,executors=list(range(24)),fits=0,manual_timer_mutations=0,fresh_deadline_unix=1792274400.,launch_sha256=launch['sha256'],original_numerical_runtime_sha256=runtime['sha256']))
        receipts=[put('audit/executor-'+str(n)+'.json',dict(pid=n,admitted=True,runtime_sha256=runtime['sha256'],deadline_unix=1792274400.,fits=0,labels_read=False)) for n in range(24)]
        value=dict(schema='e36-optional-native-terminal-v1',closed=True,fits=0,launch=launch,exit=put('phase/exit.json',dict(returncode=0)),helper_exits=put('phase/helpers.json',dict(exits=[dict(returncode=0),dict(returncode=0)])),log=put('phase/log.json',{}),sources=put('phase/sources.json',dict(files={})),runtime_audit=audit,runtime_receipts=receipts,returncode=0)
        return put('terminal.json',value),value

    def test_all24_audits_and_exact_terminal_hashes_required(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);path,value=self.fixture(root);validate(root,path['path'])
            value['runtime_receipts'].pop();Path(path['path']).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'omit an executor'):validate(root,path['path'])
            path,value=self.fixture(root);Path(value['exit']['path']).write_text('{"returncode":124}')
            with self.assertRaisesRegex(ValueError,'proof changed'):validate(root,path['path'])


if __name__=='__main__':unittest.main()
