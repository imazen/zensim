"""E31 metadata tripwires and unchanged control argv."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import e31_training as e31
import v2_common as common
import v2_lodo_mlp as fit


class E31Training(unittest.TestCase):
    def test_control_argv_does_not_acquire_extension_flags(self):
        groups = [("teacher", Path("teacher.parquet"), 1.0, 0, "withinref,both")]
        recipe = common.recipe_of("sel:59f0bbc2f290@h32:H128:cv16:cf98")
        with patch.object(fit, "strict_training_groups"), patch.object(fit, "strict_group_input_roots", return_value=()):
            cmd = fit.train_command(groups, 1101, 101, 1853, Path("keep.txt"), "N", Path("model.bin"), recipe,
                                    strict_admission=True)
        self.assertNotIn("--upiq-label-disposition", cmd)
        self.assertNotIn("--historical-replay", cmd)
        self.assertEqual(cmd[1:], ["--group", "teacher:teacher.parquet:1.0:0:withinref,both",
            "--target-column", "human_score", "--target-scale", "1", "--hidden", "128",
            "--epochs", "120", "--pairs-per-epoch", "50000", "--init-seed", "1101",
            "--sample-seed", "101", "--pair-sampling", "uniform", "--max-features", "1853",
            "--keep-features", "keep.txt", "--mse-weight", "1", "--early-stop-patience", "0",
            "--val-policy", "mean", "--val-aggregate", "geomean3", "--out-dtype", "f32",
            "--log-every", "17", "--no-auto-eval", "--out", "model.bin", "--nonneg-distance"])

    def test_recipe_is_pooled_human_hdr_only(self):
        r = common.recipe_of("sel:59f0bbc2f290@h32:H128:cv16:cf98:uh4")
        self.assertEqual((r["hdr_weight"], r["hdr_mode"], r["upiq380"]), (4.0, "rank", True))
        with self.assertRaises(ValueError):
            common.recipe_of("sel:59f0bbc2f290@h32:H128:uh4:hp4")

    def test_foreign_roles_refused_before_keys_or_payloads(self):
        source = Path("/mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07/upiq380_fit.parquet.manifest.json")
        declaration = json.loads(source.read_text())
        for field, value in [("split", "development"), ("tier", "T0"), ("source", "AIC3"),
                             ("requested_ids", list(range(420))), ("formula_revision", 4)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                d = copy.deepcopy(declaration)
                d[field] = value
                path = Path(tmp) / "fit.parquet"
                Path(f"{path}.manifest.json").write_text(json.dumps(d))
                with patch.object(e31, "sha", side_effect=AssertionError("payload hash")), \
                     patch.object(e31.pq, "read_table", side_effect=AssertionError("keys opened")):
                    with self.assertRaisesRegex(ValueError, "declaration mismatch"):
                        e31.fit_metadata(path, Path(tmp) / "unread-decision.json", e31.columns("by_v2fy"))

    def test_d3_is_not_label_gap_disposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.json"
            path.write_text(json.dumps(dict(state="approved", authority="D3-2026-10-07")))
            with self.assertRaisesRegex(ValueError, "owner's bound"):
                e31.disposition(path)

    def test_real_fit_keys_only_derive_registered_weight(self):
        # A synthetic decision exists only inside this test. No feature payload
        # is opened and no actual owner disposition is created.
        with tempfile.TemporaryDirectory() as tmp:
            decision = Path(tmp) / "synthetic-decision.json"
            decision.write_text(json.dumps(dict(schema="e31-upiq-label-disposition-v1", state="approved",
                decision_id="E31-legacy-HDR-label-producer-gap", allowed_use="registered-E31-research-training",
                manifest_sha256=e31.MANIFEST_SHA, legacy_label_sha256=e31.LABEL_SHA,
                accept_unresolved_producer=True)))
            group, receipt = e31.fit_group(Path("/mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07/upiq380_fit.parquet"),
                                          decision, e31.columns("by_v2fy"))
            self.assertEqual(group[2:], (4.34410740924913, 0, "rank"))
            self.assertFalse(receipt["qualified_provenance"])
            self.assertIsNone(receipt["label_source"]["producer_commit"])


if __name__ == "__main__":
    os.environ.setdefault("TMPDIR", str(Path.home() / "tmp"))
    unittest.main()
