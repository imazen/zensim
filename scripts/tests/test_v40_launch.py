"""No fleet side effects when approval is absent, incomplete or mismatched."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import v40_launch as owner


class Launch(unittest.TestCase):
    def fixture(self, root):
        payload = root / "pin.bin"
        payload.write_bytes(b"reviewed synthetic identity")
        caps = {job: {"memory_gib": 6} for job in owner.JOBSETS}
        (root / "jobset_caps.json").write_text(json.dumps(caps))
        ids = dict(program_sha="a" * 64, files={"pin.bin": owner.sha(payload)})
        for job in owner.JOBSETS:
            (root / f"AUTHORIZATION_REQUIRED-{job}.json").write_text(
                json.dumps(dict(identities=ids)))
            (root / f"LAUNCH_AUTHORIZATION-{job}.json").write_text(json.dumps(dict(
                schema="v40-coordinator-launch-v1", coordinator_message="SYNTHETIC TEST ONLY",
                source_landed=True, pins_pushed=True, reviewed=True,
                E30_completed=True, control_choice_frozen=True, identities=ids)))
        return caps, ids

    def test_positive_approval_reaches_real_installed_cells_path(self):
        import e30_four_source
        import v40_score
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caps, ids = self.fixture(root)
            read_text = Path.read_text
            def read(path, *args, **kwargs):
                if str(path) == "/var/tmp/fitv2/jobset_caps.json":
                    return json.dumps(caps)
                return read_text(path, *args, **kwargs)
            def prerequisite(results, bundle):
                self.assertEqual(results, Path("/var/tmp/rev4-featpot/e30-results/cells"))
                self.assertEqual(bundle, Path("/mnt/v/output/zensim/shippath11-2026-10-07"))
                return dict(cell_count=40)
            with (
                patch.object(Path, "read_text", read),
                patch.object(e30_four_source, "completed_control_pins", side_effect=prerequisite) as e30,
                patch.object(v40_score, "complete", side_effect=ValueError("INCOMPLETE: missing control/kadid/s0")) as control,
                patch.object(owner.subprocess, "run", side_effect=AssertionError("fleet action")),
                patch.object(owner.subprocess, "check_output", side_effect=AssertionError("fleet read")),
            ):
                self.assertEqual(owner.gate(root, owner.JOBSETS[0]), ids)
                for job in owner.JOBSETS[1:]:
                    with self.assertRaisesRegex(ValueError, "missing control/kadid/s0"):
                        owner.gate(root, job)
                self.assertEqual(e30.call_count, 4)
                self.assertEqual(control.call_count, 3)
                # All forty fresh synthetic cells and pins permit each arm gate.
                cells, frozen = {}, {}
                for fold in ("kadid", "tid2013", "konfig", "cid22_a25"):
                    for seed in range(10):
                        cell = root / "cells" / f"{fold}_s{seed}"
                        (cell / "refit").mkdir(parents=True)
                        (cell / "result.json").write_text(json.dumps(dict(fold=fold, seed=seed)))
                        (cell / "refit/last.bin").write_bytes(f"synthetic {fold} {seed}".encode())
                        cells[fold, seed] = cell
                        frozen[f"{fold}_s{seed}"] = dict(result_sha256=owner.sha(cell / "result.json"), bake_sha256=owner.sha(cell / "refit/last.bin"))
                (root / "V40_CONTROL_PINS.json").write_text(json.dumps(dict(control_choice="fresh-matched-v40", program_sha=ids["program_sha"], cells=frozen)))
                control.side_effect = None
                control.return_value = dict(control=cells)
                for job in owner.JOBSETS[1:]:
                    self.assertEqual(owner.gate(root, job), ids)
                (root / "cells/kadid_s0/refit/last.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "fresh V40 control changed"):
                    owner.gate(root, owner.JOBSETS[1])
                (root / "pin.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "V40 input changed"):
                    owner.gate(root, owner.JOBSETS[0])

    def test_absent_approval_all_four_jobsets_open_no_fleet_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(
                    owner.subprocess, "run", side_effect=AssertionError("fleet action")
                ),
                patch.object(
                    owner.subprocess,
                    "check_output",
                    side_effect=AssertionError("fleet read"),
                ),
            ):
                for name in owner.JOBSETS + ("unregistered",):
                    with self.assertRaises(PermissionError):
                        owner.launch(root, name)
            self.assertEqual(list(root.iterdir()), [])

    def test_false_approval_and_changed_identity_refuse_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobset = owner.JOBSETS[0]
            required = dict(identities={"program_sha": "a" * 64})
            (root / f"AUTHORIZATION_REQUIRED-{jobset}.json").write_text(
                json.dumps(required)
            )
            approval = dict(
                schema="v40-coordinator-launch-v1",
                coordinator_message="synthetic",
                source_landed=True,
                pins_pushed=True,
                reviewed=False,
                E30_completed=True,
                control_choice_frozen=True,
                identities=required["identities"],
            )
            ap = root / f"LAUNCH_AUTHORIZATION-{jobset}.json"
            with (
                patch.object(
                    owner.subprocess, "run", side_effect=AssertionError("fleet action")
                ),
                patch.object(
                    owner.subprocess,
                    "check_output",
                    side_effect=AssertionError("fleet read"),
                ),
            ):
                ap.write_text(json.dumps(approval))
                with self.assertRaises(PermissionError):
                    owner.launch(root, jobset)
                approval.update(reviewed=True, identities={"program_sha": "b" * 64})
                ap.write_text(json.dumps(approval))
                with self.assertRaises(PermissionError):
                    owner.launch(root, jobset)
            self.assertEqual(len(list(root.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
