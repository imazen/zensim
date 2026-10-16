"""E33 launch owner refusals: sequence, live-cap conflicts and empty host maps, before any side effect."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import e33_launch as owner


class Launch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "packet"
        self.bundle.mkdir()
        entry = {"memory": "6g", "hosts": {"i265": 3, "tower": 5}}
        self.mine = {owner.jobset(a): entry for a in owner.ORDER}
        (self.bundle / "jobset_caps.json").write_text(json.dumps(self.mine))
        (self.bundle / "PLACEMENT_REHEARSAL.json").write_text(json.dumps({"fleet_hosts": ["i265", "tower"]}))
        self.live = self.root / "jobset_caps.json"
        self.live.write_text(json.dumps({"fitv40-control-20261007": {"memory": "6g", "hosts": {"i265": 1}}}))
        self.queue = self.root / "fleet_queue"
        self.queue.write_text("fitv40-control-20261007 m img\n")
        doc = {"jobsets": {owner.jobset(a): 1 for a in owner.ORDER}, "image": "img", "image_id": "sha256:x"}
        self.patches = [mock.patch.object(owner, "packet", return_value=doc),
                        mock.patch.object(owner.subprocess, "check_output", return_value="sha256:x\n"),
                        mock.patch.object(owner.subprocess, "run", side_effect=AssertionError("side effect"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def steer_fixture(self):
        order = [f'fitsteer-{arm}-{route}-20261016' for route in ['lodo', 'full']
                 for arm in ['basea', 'basec', 'sa', 'sc']]
        doc = {'schema': 'steer-fleet-packet-v1', 'image': 'img', 'image_ref': 'img@sha256:x',
               'image_id': 'sha256:x', 'jobsets': {js: 40 if '-lodo-' in js else 3 for js in order}}
        plan = {'schema': 'steer-job-plan-v1', 'order': order, 'unconditional': order}
        (self.bundle/'STEER_JOB_PLAN.json').write_text(json.dumps(plan))
        entry = {'memory': '6g', 'hosts': {'tower': 1}}
        (self.bundle/'jobset_caps.json').write_text(json.dumps({js: entry for js in order}))
        self.live.write_text(json.dumps({js: entry for js in order}))
        return doc, plan

    def test_steer_inventory_refuses_missing_duplicate_and_wrong_total(self):
        doc, plan = self.steer_fixture()
        self.assertEqual(owner.steer_plan(self.bundle, doc), plan)
        for order in [plan['order'][:-1], [plan['order'][0]]*8]:
            (self.bundle/'STEER_JOB_PLAN.json').write_text(json.dumps(dict(plan, order=order)))
            with self.assertRaises(ValueError):
                owner.steer_plan(self.bundle, doc)
        (self.bundle/'STEER_JOB_PLAN.json').write_text(json.dumps(plan))
        doc['jobsets'][plan['order'][0]] -= 1
        with self.assertRaises(ValueError):
            owner.steer_plan(self.bundle, doc)

    def test_steer_first_cell_refuses_nonempty_queue_and_modified_subset(self):
        doc, plan = self.steer_fixture()
        with mock.patch.object(owner, 'packet', return_value=doc):
            with self.assertRaisesRegex(ValueError, 'empty dedicated queue'):
                owner.launch(self.bundle, plan['order'][0], self.queue, self.live, first_cell=True)
            self.queue.write_text('')
            (self.bundle/f"fit-manifest-{plan['order'][0]}.json").write_text('[{"cell":1},{"cell":2}]')
            (self.bundle/'FIRST_CELL_MANIFEST.json').write_text('[{"cell":2}]')
            with self.assertRaisesRegex(ValueError, 'exact first registered cell'):
                owner.launch(self.bundle, plan['order'][0], self.queue, self.live, first_cell=True)
            with self.assertRaises(FileNotFoundError):
                owner.launch(self.bundle, plan['order'][1], self.queue, self.live)

    def test_steer_first_admission_requires_drained_native_image_proof(self):
        (self.root/'launch').mkdir()
        dest=self.root/'cell';dest.mkdir()
        (dest/'result.json').write_text('{}');(dest/'fleet_receipt.json').write_text('{}')
        proof=dict(packet_sha256=owner.PACKET_SHA, verified=True, inside_pinned_image=True,
                   worker_drained=True, dest=str(dest), result_sha256=owner.sha(dest/'result.json'),
                   receipt_sha256=owner.sha(dest/'fleet_receipt.json'))
        path=self.root/'launch/FIRST_CELL_ADMITTED.json'
        path.write_text(json.dumps(proof))
        self.assertEqual(owner.first_steer_admission(self.bundle), proof)
        for key in ['verified', 'inside_pinned_image', 'worker_drained']:
            path.write_text(json.dumps(dict(proof, **{key:False})))
            with self.assertRaises(ValueError):owner.first_steer_admission(self.bundle)
        path.write_text(json.dumps(proof));(dest/'result.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):owner.first_steer_admission(self.bundle)

    def test_steer_complete_harvest_reader_uses_pinned_image(self):
        doc, _ = self.steer_fixture()
        args=['harvest-cycle','--packet',str(self.bundle),'--packet-sha',owner.PACKET_SHA]
        command=owner.steer_harvest_command(self.bundle, doc, args)
        self.assertIn('STEER_HARVEST_IMAGE='+doc['image_ref'], command)
        self.assertIn(doc['image_ref'], command)
        self.assertIn('AWS_SECRET_ACCESS_KEY', command)
        self.assertEqual(command[-len(args):], args)

    def test_caps_merge_keeps_other_entries_and_backs_up(self):
        out = owner.merge_caps(self.bundle, self.live)
        live = json.loads(self.live.read_text())
        self.assertIn("fitv40-control-20261007", live)
        self.assertEqual({k: live[k] for k in self.mine}, self.mine)
        self.assertTrue(Path(out["backup"]).is_file())

    def test_conflicting_live_entry_refuses(self):
        self.live.write_text(json.dumps({owner.jobset("a"): {"memory": "6g", "hosts": {"i265": 9}}}))
        with self.assertRaisesRegex(ValueError, "different"):
            owner.merge_caps(self.bundle, self.live)

    def test_empty_hosts_refuse(self):
        (self.bundle / "jobset_caps.json").write_text(json.dumps({owner.jobset("a"): {"memory": "6g", "hosts": {}}}))
        with self.assertRaises(ValueError):
            owner.merge_caps(self.bundle, self.live)

    def test_launch_refuses_out_of_sequence_and_unmerged_caps(self):
        with self.assertRaisesRegex(ValueError, "live jobset cap differs"):
            owner.launch(self.bundle, "control", self.queue, self.live)
        owner.merge_caps(self.bundle, self.live)
        with self.assertRaisesRegex(ValueError, "registered sequence"):
            owner.launch(self.bundle, "c", self.queue, self.live)
        self.queue.write_text(self.queue.read_text() + f"{owner.jobset('control')} m img\n")
        with self.assertRaisesRegex(ValueError, "already queued"):
            owner.launch(self.bundle, "control", self.queue, self.live)

    def test_advance_waits_for_drain_and_appends_in_order(self):
        self.queue.write_text(self.queue.read_text() + f"{owner.jobset('control')} m img\n")
        with mock.patch.object(owner, "open_cells", return_value=3), mock.patch.object(owner, "tower_workers", return_value=0):
            self.assertEqual(owner.advance(self.bundle, self.queue)["action"], "wait")
        with mock.patch.object(owner, "open_cells", return_value=0), mock.patch.object(owner, "tower_workers", return_value=2):
            self.assertEqual(owner.advance(self.bundle, self.queue)["action"], "wait")
        with mock.patch.object(owner, "open_cells", return_value=0), mock.patch.object(owner, "tower_workers", return_value=1), \
                mock.patch.object(owner, "launch", return_value={"queued": "t"}) as launch:
            out = owner.advance(self.bundle, self.queue)
        self.assertEqual((out["action"], out["next"]), ("launched", "a"))
        launch.assert_called_once()
        for arm in ("a", "c", "full"):
            self.queue.write_text(self.queue.read_text() + f"{owner.jobset(arm)} m img\n")
        self.assertEqual(owner.advance(self.bundle, self.queue)["action"], "none")

    def test_caps_amendments_only_add_hosts(self):
        (self.root / "launch").mkdir()
        js = owner.jobset("a")
        path = self.root / "launch" / "CAPS_AMENDMENTS.json"
        path.write_text(json.dumps({"amendments": [{"id": "loan", "jobsets": {js: {"added_hosts": {"r5600g": 2},
                                                                                     "reason": "loan"}}}]}))
        want = owner.expected_caps(self.bundle, js)
        self.assertEqual(want["hosts"], {"i265": 3, "tower": 5, "r5600g": 2})
        self.assertEqual((want["memory"], want["reason"]), ("6g", "loan"))
        self.assertEqual(owner.expected_caps(self.bundle, owner.jobset("c")), self.mine[owner.jobset("c")])
        for bad in ({"added_hosts": {"tower": 9}}, {"added_hosts": {}, "memory": "9g"}, {"added_hosts": {"x": 0}}):
            path.write_text(json.dumps({"amendments": [{"id": "bad", "jobsets": {js: bad}}]}))
            with self.assertRaises(ValueError):
                owner.expected_caps(self.bundle, js)


    def test_e34_unselected_branch_refuses_before_upload_or_queue(self):
        js='fite34-k1a-lodo-20261015'
        (self.bundle/'E34_JOB_PLAN.json').write_text(json.dumps({'unconditional':[]}))
        (self.root/'launch').mkdir()
        (self.root/'launch/LAMBDA_SELECTION.json').write_text(json.dumps({'selected_branches':
            {'a':'fite34-k01a-lodo-20261015','c':'fite34-k10c-lodo-20261015'}}))
        entry={'memory':'6g','hosts':{'tower':1}}
        (self.bundle/'jobset_caps.json').write_text(json.dumps({js:entry}))
        self.live.write_text(json.dumps({js:entry}))
        with mock.patch.object(owner,'packet',return_value={'schema':'e34-fleet-packet-v1',
                    'jobsets':{js:42},'image':'img','image_id':'sha256:x'}):
            with self.assertRaisesRegex(ValueError,'not selected'):
                owner.launch(self.bundle,js,self.queue,self.live)
        self.assertNotIn(js,self.queue.read_text())


    def test_explicit_partial_pilot_never_selects_with_pending_artifacts(self):
        import types
        instrument=self.root/'instrument';instrument.write_bytes(b'fixture tool')
        inputs=self.root/'dev96.json';inputs.write_text(json.dumps({'labels_read':False,
            'cases':[{'key':str(i),'origin':str(i//12)} for i in range(96)]}))
        (self.bundle/'DEVELOPMENT_BINDING.json').write_text(json.dumps({
            'assessment_inputs':str(inputs),'assessment_inputs_sha256':owner.sha(inputs),
            'instrument_sha256':owner.sha(instrument)}))
        for family in ['a','c']:
            for arm in ['k01','k1','k10']:
                (self.bundle/f'fit-manifest-fite34-{arm}{family}-pilot-20261015.json').write_text('[{}]')
        rt=self.root/'launch';rt.mkdir()
        with mock.patch.object(owner,'runtime',return_value=(rt,self.root,instrument)), \
             mock.patch.object(owner,'installed',return_value=False), \
             mock.patch.dict('sys.modules',{'harvest_fit_cells':types.SimpleNamespace()}):
            out=owner.lambda_pilot(self.bundle,instrument,partial=True)
            self.assertEqual((out['state'],len(out['pending_artifacts'])),('WAIT',6))
            self.assertEqual(out['completed_assessments'],[])
            self.assertFalse((rt/'LAMBDA_SELECTION.json').exists())
            with self.assertRaisesRegex(ValueError,'not been verified'):
                owner.lambda_pilot(self.bundle,instrument)


class HarvestRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sync_retries_transport_then_requires_success(self):
        import subprocess
        replies = [subprocess.CompletedProcess([], 2, "", "connection reset"),
                   subprocess.CompletedProcess([], 0, "", "")]
        with mock.patch.object(owner.subprocess, "run", side_effect=replies) as run, \
             mock.patch.object(owner.time, "sleep"):
            owner.sync_ledger("fixture", "pilot", self.root)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs["timeout"], 30)

    def test_missing_and_denied_ledgers_remain_failures_even_with_cached_files(self):
        import subprocess
        (self.root / "cached.parquet").write_bytes(b"cached evidence is not a successful sync")
        for error in ["no object found", "AccessDenied"]:
            with self.subTest(error=error), \
                 mock.patch.object(owner.subprocess, "run", return_value=
                                   subprocess.CompletedProcess([], 2, "", error)) as run, \
                 mock.patch.object(owner.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, error):
                    owner.sync_ledger("fixture", "pilot", self.root)
                self.assertEqual(run.call_count, 3)

    def test_sync_timeouts_are_bounded_and_visible(self):
        import subprocess
        with mock.patch.object(owner.subprocess, "run", side_effect=
                               subprocess.TimeoutExpired("s5cmd", 30, stderr=b"slow store")) as run, \
             mock.patch.object(owner.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "slow store"):
                owner.sync_ledger("fixture", "pilot", self.root)
            self.assertEqual(run.call_count, 3)

    def test_zero_exit_with_json_error_is_not_success(self):
        import subprocess
        for diagnostic in ['{"operation":"sync","error":"no object found"}',
                           '{"operation":"sync","error":"AccessDenied"}',
                           '{"success":false}', 'unstructured failure']:
            with self.subTest(diagnostic=diagnostic), \
                 mock.patch.object(owner.subprocess, "run", return_value=
                                   subprocess.CompletedProcess([], 0, "", diagnostic)) as run, \
                 mock.patch.object(owner.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    owner.sync_ledger("fixture", "pilot", self.root)
                self.assertEqual(run.call_count, 3)

    def cycle(self, failed=None, subset=False, count_failure=False):
        import types
        bundle = self.root / "packet"
        bundle.mkdir(exist_ok=True)
        runtime = self.root / "launch"
        runtime.mkdir(exist_ok=True)
        names = ["fite34-la-lodo-20261015", "fite34-k01a-pilot-20261015", "fite34-wa-lodo-20261015"]
        (bundle / "E34_JOB_PLAN.json").write_text(json.dumps({"unconditional": names,
                                                            "total_executed_cells": 3}))
        for name in names:
            (bundle / f"fit-manifest-{name}.json").write_text('[{}]')
        queue = self.root / "queue"
        queue.write_text("".join(f"{name} manifest image\n" for name in (names[:2] if subset else names)))
        seen = []

        def attempt(command, log, timeout):
            name = command[command.index("--arm") + 1]
            seen.append(name)
            if name == failed:
                log.write_text("ledger sync ERROR AccessDenied\n")
                return 1
            log.write_text(json.dumps(dict(jobset=name, total=1, done=1, installed=0, ledger={"done": 1})) + "\n")
            return 0

        with mock.patch.object(owner, "packet", return_value={"schema": "e34-fleet-packet-v1"}), \
             mock.patch.object(owner, "runtime", return_value=(runtime, self.root, self.root)), \
             mock.patch.object(owner, "harvest_attempt", side_effect=attempt), \
             mock.patch.object(owner, "installed", side_effect=
                               [RuntimeError("bad installed metadata"), True, True] if count_failure else None,
                               return_value=True), \
             mock.patch.dict("sys.modules", {"harvest_fit_cells": types.SimpleNamespace()}):
            result = owner.harvest_cycle(bundle, queue, self.root / "status.md")
        return result, seen

    def test_jobset_failure_continues_later_jobsets_and_is_not_completion(self):
        failed = "fite34-k01a-pilot-20261015"
        result, seen = self.cycle(failed)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[-1], "fite34-wa-lodo-20261015")
        self.assertEqual(result["installed"], 3)
        self.assertEqual((result["state"], result["errors"]), ("DEGRADED", [failed]))
        self.assertIn("AccessDenied", Path(result["jobsets"][failed]["attempt_log"]).read_text())
        self.assertIn("ERROR; installed 1/1", (self.root / "status.md").read_text())
        recovered, seen = self.cycle()
        self.assertEqual((recovered["state"], recovered["errors"]), ("COMPLETE", []))
        self.assertEqual(len(seen), 3)

    def test_incomplete_release_is_progress_not_complete(self):
        result, _ = self.cycle(subset=True)
        self.assertEqual((result["state"], result["installed"]), ("PROGRESS", 2))

    def test_count_failure_is_unknown_and_does_not_stop_other_jobsets(self):
        result, seen = self.cycle(count_failure=True)
        self.assertEqual(len(seen), 3)
        self.assertEqual(result["state"], "DEGRADED")
        self.assertEqual(result["installed_count_unknown"], [seen[0]])
        self.assertIsNone(result["jobsets"][seen[0]]["installed"])
        self.assertEqual(result["jobsets"][seen[-1]]["state"], "OK")

    def test_real_harvest_child_timeout_is_reaped(self):
        import subprocess
        import sys
        log = self.root / "timeout.log"
        with self.assertRaises(subprocess.TimeoutExpired):
            owner.harvest_attempt([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(10)"],
                                  log, 0.2)
        self.assertIn("started", log.read_text())

    def test_interruption_cleans_up_owned_verifier_group(self):
        import signal
        proc = mock.Mock(pid=12345)
        proc.wait.side_effect = [KeyboardInterrupt(), 0]
        with mock.patch.object(owner.subprocess, "Popen", return_value=proc), \
             mock.patch.object(owner.os, "killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                owner.harvest_attempt(["fixture"], self.root / "interrupted.log", 300)
        kill.assert_called_once_with(12345, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
