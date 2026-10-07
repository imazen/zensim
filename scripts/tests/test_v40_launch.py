"""No fleet side effects when approval is absent, incomplete or mismatched."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import v40_launch as owner


class Launch(unittest.TestCase):
    def test_absent_approval_and_blocked_e31_open_no_fleet_owner(self):
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
                for name in owner.JOBSETS + ("fitv40-e31-20261007",):
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
