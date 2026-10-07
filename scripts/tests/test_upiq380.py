"""Production UPIQ metadata admission and zero-label-open refusal tripwires."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import upiq380 as u


class UpiqAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp", prefix="upiq-metadata-test-")
        self.root = Path(self.tmp.name)
        for ds, nc, nd, nl in [("narwaria", 10, 2, 7), ("korshunov", 20, 3, 4)]:
            for c in range(1, nc + 1):
                scene = self.root / ds / f"{c:02}"
                scene.mkdir(parents=True)
                (scene / f"i{c:02}.exr").touch()
                for d in range(1, nd + 1):
                    for l in range(1, nl + 1):
                        (scene / f"i{c:02}_{d:02}_{l}.exr").touch()
        self.a = dict(schema="upiq380-extraction-admission-v1", role="train", tier="T2", authority="D3-2026-10-07",
            formula_revision=5, input_contract="upiq-exr-bt709-nits-v1", requested_ids=u.columns("by_v2fy"),
            image_root=str(self.root), rows=u.metadata_rows(self.root), split_rule=u.RULE,
            target_transform=u.TRANSFORM, label_file_approved=str(u.LABEL), label_sha256=u.LABEL_SHA)

    def tearDown(self):
        self.tmp.cleanup()

    def refuse_without_opens(self, a):
        with patch.object(Path, "open", side_effect=AssertionError("payload opened before refusal")) as tripwire:
            # Requested IDs are the already admitted registered source metadata.
            with patch.object(u, "columns", return_value=self.a["requested_ids"]):
                with self.assertRaises(ValueError):
                    u.targets(a)
            self.assertEqual(tripwire.call_count, 0)

    def test_metadata_membership_is_380_and_only_30_HDR_references(self):
        with patch.object(Path, "open", side_effect=AssertionError("payload opened")) as tripwire:
            with patch.object(u, "columns", return_value=self.a["requested_ids"]):
                rows = u.admit_metadata(self.a)
            self.assertEqual(len(rows), 380)
            self.assertEqual(len({r["reference_rel"] for r in rows}), 30)
            self.assertEqual(sum(r["dataset"] == "narwaria" for r in rows), 140)
            self.assertEqual(sum(r["dataset"] == "korshunov" for r in rows), 240)
            self.assertEqual(tripwire.call_count, 0)

    def test_role_revision_source_and_label_inventory_refuse_before_open(self):
        for field, value in [("role", "val"), ("tier", "T0"), ("formula_revision", 4),
                ("authority", "other"), ("label_file_approved", "/mnt/v/datasets/upiq/upiq_subjective_scores.csv"),
                ("label_sha256", "0" * 64), ("target_transform", "fit-offset"), ("split_rule", "label-quantile")]:
            with self.subTest(field=field):
                a = copy.deepcopy(self.a)
                a[field] = value
                self.refuse_without_opens(a)

    def test_bad_last_member_and_duplicate_refuse_before_any_label_open(self):
        for field, value in [("dataset", "live"), ("distorted_rel", "../live/image.png"),
                ("condition_id", "l-i01-l-01-1"), ("reference_rel", "tid2013/01/i01.png")]:
            with self.subTest(field=field):
                a = copy.deepcopy(self.a)
                a["rows"][-1][field] = value
                self.refuse_without_opens(a)
        a = copy.deepcopy(self.a)
        a["rows"][-1] = a["rows"][0]
        self.refuse_without_opens(a)

    def test_unapproved_image_inventory_refuses_without_label_opens(self):
        (self.root / "narwaria/01/extra.exr").touch()
        self.refuse_without_opens(self.a)


if __name__ == "__main__":
    unittest.main()
