"""Failure-event and live replacement ownership fixtures; no real processes."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_resource_handoff as owner


class Handoff(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory);(root/'owner-svt-ssd-v1').mkdir();folder=root/'new';folder.mkdir()
        children=[Mock(pid=101),Mock(pid=102)]
        for child in children:child.poll.return_value=None;child.wait.return_value=0
        return root,folder,dict(zip(['mirror','monitor'],children)),children,threading.Event()

    def test_stop_write_failure_while_old_helpers_live_sets_shared_event(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root,folder,roles,children,event=self.fixture(d)
            with self.assertRaises(OSError):owner.transition(root,folder,folder/'receipt.json',roles,children,{},root,event,Mock(side_effect=OSError('full')))
            self.assertTrue(event.is_set());self.assertTrue(all(child.poll() is None for child in children))

    def test_live_new_child_retained_when_readiness_times_out(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root,folder,roles,children,event=self.fixture(d)
            old=children[0];old.poll.return_value=0;new=Mock(pid=201);new.poll.return_value=None
            def write(path,value):path.write_text(json.dumps(value))
            with patch.object(owner.subprocess,'Popen',return_value=new),patch.object(owner.time,'monotonic',side_effect=[0,121]):
                with self.assertRaisesRegex(RuntimeError,'readiness'):owner.transition(root,folder,folder/'receipt.json',roles,children,{},root,event,write)
            self.assertTrue(event.is_set());self.assertIn(new,children);self.assertIs(roles['mirror'],new)

    def test_failure_after_both_replacements_sets_event(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root,folder,roles,children,event=self.fixture(d)
            for child in children:child.poll.return_value=0
            next_pid=iter([201,202])
            def spawn(argv,**kwargs):
                pid=next(next_pid);role='mirror' if 'artifact_mirror' in argv[1] else 'monitor'
                Path(kwargs['env']['E36_RESOURCE_READY_PATH']).write_text(json.dumps(dict(pid=pid,role=role,scope=str(root))))
                child=Mock(pid=pid);child.poll.return_value=None;return child
            def write(path,value):
                if path.name=='receipt.json':raise OSError('receipt failed')
                path.write_text(json.dumps(value))
            with patch.object(owner.subprocess,'Popen',side_effect=spawn):
                with self.assertRaisesRegex(OSError,'receipt'):owner.transition(root,folder,folder/'receipt.json',roles,children,{},root,event,write)
            self.assertTrue(event.is_set());self.assertEqual({p.pid for p in children},{201,202});self.assertEqual({p.pid for p in roles.values()},{201,202})

    def test_mask_expiry_engages_hold_without_inventing_process_exit(self):
        event=threading.Event();child=Mock();child.poll.return_value=None
        planned=owner.Planned(child,event,0)
        self.assertIsNone(planned.poll());self.assertTrue(event.is_set());child.terminate.assert_not_called();child.kill.assert_not_called()


if __name__=='__main__':unittest.main()
