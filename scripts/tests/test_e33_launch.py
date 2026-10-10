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


if __name__ == "__main__":
    unittest.main()
