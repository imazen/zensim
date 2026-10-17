import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

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
            ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
            def handles(scope,worker):
                return [(dict(pid=os.getpid(),start_ticks=ticks,state='R'),os.pidfd_open(os.getpid()))] if worker==expected else []
            with patch.object(policy,'ROOT',root),patch.object(resume.signal,'pidfd_send_signal') as send,patch.object(resume,'worker_handles',side_effect=handles),patch.object(resume,'same_worker',return_value=True):
                resume.hold_admission(root,scope,expected,'fixture resource child failure')
                self.assertEqual(send.call_count,1)
                self.assertEqual(send.call_args.args[1],resume.signal.SIGSTOP)
                hold=json.loads((root/'owner-svt-ssd-v1/resource-failure-hold.json').read_text())
                self.assertEqual(hold['owned_stops'][0]['pid'],os.getpid())
                state=root/'owner-svt-ssd-v1/disk-claim-state.json'
                self.assertTrue(json.loads(state.read_text())['claims_paused'])
                with patch.object(policy,'STATE',state):
                    with self.assertRaises(TimeoutError):policy.wait(0)
                send.reset_mock()
                resume.hold_admission(root,scope,root/'different-executable','fixture mismatch')
                send.assert_not_called()

    def test_hold_receipt_failure_still_signals_stable_owned_handle(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-resume-persist-') as name:
            root=Path(name);(root/'owner-svt-ssd-v1').mkdir();scope=root/'scope';scope.mkdir()
            item=dict(pid=os.getpid(),start_ticks='fixture',state='R');memory={}
            def handles(*args):return [(item,os.pidfd_open(os.getpid()))]
            with patch.object(resume,'worker_handles',side_effect=handles),patch.object(resume,'same_worker',return_value=True),patch.object(policy,'atomic',side_effect=OSError('receipt unavailable')),patch.object(resume.signal,'pidfd_send_signal') as send:
                result=resume.hold_admission(root,scope,root/'worker','fixture',memory)
                self.assertIn('receipt unavailable',result['persistence_error'])
                self.assertEqual(send.call_count,1)
                item['state']='T'
                resume.hold_admission(root,scope,root/'worker','fixture',memory)
                self.assertEqual(send.call_count,2)

    def test_delayed_worker_is_reconciled_after_first_empty_pause_snapshot(self):
        native=SimpleNamespace(poll=Mock(side_effect=[None,None,0]),returncode=0)
        child=SimpleNamespace(poll=lambda:1)
        with patch.object(resume,'hold_admission',side_effect=[dict(persistence_error=None,owned_stops=[]),dict(persistence_error=None,owned_stops=[dict(pid=7)])]) as hold,patch.object(resume.time,'sleep'):
            self.assertEqual(resume.supervise(native,dict(monitor=child),Path('.'),Path('.'),Path('worker')),0)
            self.assertEqual(hold.call_count,2)

    def test_post_spawn_exception_cannot_bypass_native_drain_before_cleanup(self):
        native=SimpleNamespace(poll=Mock(side_effect=[None,None,0]),returncode=0)
        with patch.object(resume,'supervise',side_effect=[OSError('hold write failed'),0]) as wait:
            resume.drain_before_cleanup(native,{},Path('.'),Path('.'),Path('worker'),{})
            self.assertEqual(wait.call_count,2)


if __name__=='__main__':unittest.main()
