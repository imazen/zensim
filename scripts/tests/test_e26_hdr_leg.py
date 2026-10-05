"""Exercise the actual E26 admission/packer owners on synthetic receipts; never fit."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rev4_featpot"))
import e21_cheap_recipe as e21
import v2_common
import v2_teacher
import v2c_pack


class HdrLegAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/var/tmp"), prefix="e26-fixture-")
        self.root = Path(self.tmp.name)
        self.previous = v2_common.V2
        v2_common.V2 = self.root
        self.path = self.root / "wide/main/real/hdr_fit.parquet"
        self.path.parent.mkdir(parents=True)
        self.cols = e21.columns("by_v2fy")
        self.keys = [dict(row_id=i, role="train", agree=True, ref_path=f"fixture://train/ref{i % 495}",
                          source_family=str(i % 33), hdrvdp3_q_jod=-2.0 if i == 0 else 8.0+i/100000)
                     for i in range(7390)]
        self.manifest = dict(study="E26", role="train", population="agree-only", rows=7390,
                             teacher_sha256=v2_teacher.HDR_TRAIN_SHA,
                             formula_revision=5, target_transform=v2_teacher.HDR_TRANSFORM, requested_ids=self.cols)
        self.table = pa.table({"ref_basename": [k["ref_path"] for k in self.keys],
                               "human_score": [10*k["hdrvdp3_q_jod"] for k in self.keys]})
        self.write()

    def tearDown(self):
        v2_common.V2 = self.previous
        self.tmp.cleanup()

    def write(self):
        pq.write_table(self.table, self.path)
        pq.write_table(pa.Table.from_pylist(self.keys), self.path.with_suffix(".keys.parquet"))
        manifest_path = Path(f"{self.path}.manifest.json")
        manifest_path.write_text(json.dumps(self.manifest))
        keys_sha = v2_common.sha(self.path.with_suffix(".keys.parquet"))
        self.record = {"fit": dict(rel=str(self.path.relative_to(self.root)), sha256=v2_common.sha(self.path),
                                  manifest_sha256=v2_common.sha(manifest_path), keys_sha256=keys_sha),
                       "keys_sha256": keys_sha}

    def test_actual_loader_preserves_negative_and_requires_exact_join(self):
        path, rec = v2_teacher.hdr_leg(self.record, self.cols)
        self.assertEqual(path, self.path)
        self.assertEqual(pq.read_table(path)["human_score"][0].as_py(), -20.0)
        self.assertEqual(rec["target_transform"], v2_teacher.HDR_TRANSFORM)
        self.keys.reverse()
        self.write()
        with self.assertRaisesRegex(ValueError, "row order"):
            v2_teacher.hdr_leg(self.record, self.cols)

    def test_actual_loader_refuses_role_revision_membership_transform_and_pins(self):
        pristine = copy.deepcopy(self.manifest)
        for name, value in [("study", "other"), ("role", "val"), ("population", "all"),
                            ("formula_revision", 4), ("rows", 7391),
                            ("target_transform", "clip"), ("teacher_sha256", "0"*64),
                            ("requested_ids", self.cols[:-1])]:
            with self.subTest(name=name):
                self.manifest = dict(pristine, **{name: value})
                self.write()
                self.assert_no_payload_read(self.record)
        self.manifest = pristine
        for field, value in [("role", "val"), ("agree", False), ("row_id", 1)]:
            original = self.keys[0][field]
            with self.subTest(key=field):
                self.keys[0][field] = value
                self.write()
                with self.assertRaises(ValueError):
                    v2_teacher.hdr_leg(self.record, self.cols)
            self.keys[0][field] = original
        self.write()
        for part in ["sha256", "manifest_sha256"]:
            bad = copy.deepcopy(self.record)
            bad["fit"][part] = "0"*64
            with self.assertRaisesRegex(ValueError, "changed"):
                v2_teacher.hdr_leg(bad, self.cols)
        bad = copy.deepcopy(self.record)
        bad["dev"] = bad["fit"]
        self.assert_no_payload_read(bad)

    def assert_no_payload_read(self, record, loader=None):
        opened = []
        original = io.open
        forbidden = {self.path, self.path.with_suffix(".keys.parquet")}

        def tripwire(file, *args, **kwargs):
            if not isinstance(file, int) and Path(file) in forbidden:
                opened.append(str(file))
                raise AssertionError("forbidden payload opened before admission")
            return original(file, *args, **kwargs)

        with patch("io.open", side_effect=tripwire), self.assertRaises(ValueError):
            (loader or (lambda: v2_teacher.hdr_leg(record, self.cols)))()
        self.assertEqual(opened, [])

    def test_forbidden_val_fixture_has_zero_payload_opens_in_loader_and_packer(self):
        self.path = self.path.with_name("forbidden_val.parquet")
        self.manifest.update(role="val", rows=3900, population="all")
        self.write()
        self.assert_no_payload_read(self.record)
        (self.root / "wide/keep_lists.json").write_text('{}')
        receipt = dict(schema="rev4-featpot-v2c-wide-v1", family="main", variant="real", complete=True,
                       formula_revision=5, legs={"hdr": self.record})
        (self.path.parent / "receipt.json").write_text(json.dumps(receipt))
        self.assert_no_payload_read(self.record, lambda: v2c_pack.members_for(self.root, "lodo", [("main", "real")]))

    def test_actual_packer_refuses_confirmation_and_retains_fit_keys(self):
        wide = self.root / "wide"
        (wide / "keep_lists.json").write_text('{}')
        receipt = dict(schema="rev4-featpot-v2c-wide-v1", family="main", variant="real", complete=True,
                       formula_revision=5, legs={"hdr": self.record})
        (self.path.parent / "receipt.json").write_text(json.dumps(receipt))
        members = v2c_pack.members_for(self.root, "lodo", [("main", "real")])
        self.assertIn("wide/main/real/hdr_fit.keys.parquet", members)
        self.assertFalse(any("val" in p for p in members))
        (wide / "frozen.json").write_text("{}")  # Synthetic marker: reach the HDR packing guard.
        for kind in ["confirm", "all"]:
            with self.assertRaisesRegex(ValueError, "LODO only"):
                v2c_pack.members_for(self.root, kind, [("main", "real")])


if __name__ == "__main__":
    unittest.main()
