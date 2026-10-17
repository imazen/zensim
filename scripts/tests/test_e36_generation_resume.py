import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_generation_resume as resume
import e36_disk_policy as policy


class ResourceContinuation(unittest.TestCase):
    def test_missing_failed_and_mismatched_resource_readiness_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-resume-ready-') as name:
            out=Path(name);scope=out/'scope'
            roles={role:SimpleNamespace(pid=i,poll=lambda:None) for i,role in enumerate(['monitor','mirror'],1)}
            with self.assertRaises(TimeoutError):resume.wait_resources(roles,out,scope,timeout=0)
            for role,child in roles.items():
                (out/(role+'-ready.json')).write_text(json.dumps(dict(role=role,pid=child.pid,scope=str(scope))))
            resume.wait_resources(roles,out,scope,timeout=0)
            (out/'mirror-ready.json').write_text(json.dumps(dict(role='mirror',pid=77,scope=str(scope))))
            with self.assertRaises(ValueError):resume.wait_resources(roles,out,scope,timeout=0)
            roles['mirror'].poll=lambda:1
            with self.assertRaises(RuntimeError):resume.wait_resources(roles,out,scope,timeout=0)

    def test_helper_failure_holds_only_exact_owned_native_and_wait_honors_it(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-resume-hold-') as name:
            root=Path(name);(root/'owner-svt-ssd-v1').mkdir();scope=root/'scope';scope.mkdir()
            (scope/'cgroup.procs').write_text(str(os.getpid()))
            expected=Path('/proc/self/exe').resolve()
            with patch.object(policy,'ROOT',root),patch.object(resume.os,'kill') as send:
                resume.hold_admission(root,scope,expected,'fixture resource child failure')
                send.assert_called_once_with(os.getpid(),resume.signal.SIGSTOP)
                hold=json.loads((root/'owner-svt-ssd-v1/resource-failure-hold.json').read_text())
                self.assertEqual(hold['owned_stops'][0]['pid'],os.getpid())
                state=root/'owner-svt-ssd-v1/disk-claim-state.json'
                self.assertTrue(json.loads(state.read_text())['claims_paused'])
                with patch.object(policy,'STATE',state):
                    with self.assertRaises(TimeoutError):policy.wait(0)
                send.reset_mock()
                resume.hold_admission(root,scope,root/'different-executable','fixture mismatch')
                send.assert_not_called()


if __name__=='__main__':unittest.main()
