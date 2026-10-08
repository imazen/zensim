"""Control discovery must refuse a foreign grid before reading result payloads."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import e30_four_source as e30


class CompletedControlAdmission(unittest.TestCase):
    def refused_grid(self, mutate):
        jobs = [
            {"cell": {"image_path": f"{e30.SPEC}__N/without_{source}_s{seed}"}}
            for source in e30.PRODUCTION_SOURCES
            for seed in range(10)
        ]
        mutate(jobs)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "fit-manifest-fitv2e30-20261007.json").write_text(
                json.dumps(jobs)
            )
            (bundle / "PROGRAM_INVENTORY.json").write_text("{}")
            (bundle / "ARTIFACT_PINS.json").write_text("{}")
            with (
                patch.object(
                    e30, "sha", side_effect=AssertionError("payload hash reached")
                ),
                patch.object(
                    e30.subprocess,
                    "run",
                    side_effect=AssertionError("model inspector reached"),
                ),
                patch.object(
                    Path,
                    "glob",
                    side_effect=AssertionError("foreign-cell discovery reached"),
                ),
                patch.object(
                    Path,
                    "iterdir",
                    side_effect=AssertionError("foreign-cell discovery reached"),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "exactly the four-source 40-cell manifest"
                ):
                    e30.completed_control_pins(Path(tmp) / "unread-results", bundle)

    def test_aic_fold_refused_before_payload(self):
        self.refused_grid(
            lambda jobs: jobs[0]["cell"].update(
                image_path=f"{e30.SPEC}__N/without_aic3_s0"
            )
        )

    def test_duplicate_is_not_a_complete_grid(self):
        self.refused_grid(lambda jobs: jobs.__setitem__(0, jobs[1]))

    def test_missing_cell_is_not_a_complete_grid(self):
        self.refused_grid(lambda jobs: jobs.pop())

    def test_unregistered_seed_refused_before_payload(self):
        self.refused_grid(
            lambda jobs: jobs[0]["cell"].update(
                image_path=f"{e30.SPEC}__N/without_kadid_s10"
            )
        )


if __name__ == "__main__":
    unittest.main()
