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
        with (
            patch.object(fit, "strict_training_groups"),
            patch.object(fit, "strict_group_input_roots", return_value=()),
        ):
            cmd = fit.train_command(
                groups,
                1101,
                101,
                1853,
                Path("keep.txt"),
                "N",
                Path("model.bin"),
                recipe,
                strict_admission=True,
            )
        self.assertNotIn("--upiq-label-disposition", cmd)
        self.assertNotIn("--historical-replay", cmd)
        self.assertEqual(
            cmd[1:],
            [
                "--group",
                "teacher:teacher.parquet:1.0:0:withinref,both",
                "--target-column",
                "human_score",
                "--target-scale",
                "1",
                "--hidden",
                "128",
                "--epochs",
                "120",
                "--pairs-per-epoch",
                "50000",
                "--init-seed",
                "1101",
                "--sample-seed",
                "101",
                "--pair-sampling",
                "uniform",
                "--max-features",
                "1853",
                "--keep-features",
                "keep.txt",
                "--mse-weight",
                "1",
                "--early-stop-patience",
                "0",
                "--val-policy",
                "mean",
                "--val-aggregate",
                "geomean3",
                "--out-dtype",
                "f32",
                "--log-every",
                "17",
                "--no-auto-eval",
                "--out",
                "model.bin",
                "--nonneg-distance",
            ],
        )

    def test_recipe_is_pooled_human_hdr_only(self):
        r = common.recipe_of("sel:59f0bbc2f290@h32:H128:cv16:cf98:uh4")
        self.assertEqual(
            (r["hdr_weight"], r["hdr_mode"], r["upiq380"]), (4.0, "rank", True)
        )
        with self.assertRaises(ValueError):
            common.recipe_of("sel:59f0bbc2f290@h32:H128:uh4:hp4")

    def test_foreign_roles_refused_before_keys_or_payloads(self):
        declaration = dict(
            schema="upiq380-v2-leg-v1",
            arm="uh4",
            source="UPIQ-380",
            split="fit",
            role="train",
            tier="T2",
            authority="D3-2026-10-07",
            rows=330,
            references=26,
            formula_revision=5,
            requested_ids=e31.columns("by_v2fy"),
            input_contract="upiq-exr-bt709-nits-v1",
            split_rule=e31.RULE,
            target_transform=e31.TRANSFORM,
            feature_dtype="float64",
            absent_slots="NaN",
            table_sha256=e31.TABLE_SHA,
            keys_sha256=e31.KEYS_SHA,
            qualified_provenance=False,
        )
        for field, value in [
            ("split", "development"),
            ("tier", "T0"),
            ("source", "AIC3"),
            ("requested_ids", list(range(420))),
            ("formula_revision", 4),
        ]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                d = copy.deepcopy(declaration)
                d[field] = value
                path = Path(tmp) / "fit.parquet"
                Path(f"{path}.manifest.json").write_text(json.dumps(d))
                with (
                    patch.object(
                        e31, "sha", side_effect=AssertionError("payload hash")
                    ),
                    patch.object(
                        e31.pq, "read_table", side_effect=AssertionError("keys opened")
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "declaration mismatch"):
                        e31.fit_metadata(
                            path,
                            Path(tmp) / "unread-decision.json",
                            e31.columns("by_v2fy"),
                        )

    def test_d3_is_not_label_gap_disposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.json"
            path.write_text(
                json.dumps(dict(state="approved", authority="D3-2026-10-07"))
            )
            with self.assertRaisesRegex(ValueError, "owner's bound"):
                e31.disposition(path)


def check_real_fit_keys(path):
    # A synthetic decision exists only inside this test. No feature payload
    # is opened and no actual owner disposition is created.
    with tempfile.TemporaryDirectory() as tmp:
        decision = Path(tmp) / "synthetic-decision.json"
        decision.write_text(
            json.dumps(
                dict(
                    schema="e31-upiq-label-disposition-v1",
                    state="approved",
                    decision_id="E31-legacy-HDR-label-producer-gap",
                    allowed_use="registered-E31-research-training",
                    manifest_sha256=e31.MANIFEST_SHA,
                    legacy_label_sha256=e31.LABEL_SHA,
                    accept_unresolved_producer=True,
                    decided_by="synthetic-unit-fixture",
                )
            )
        )
        group, receipt = e31.fit_group(path, decision, e31.columns("by_v2fy"))
        assert group[2:] == (4.34410740924913, 0, "rank")
        assert receipt["qualified_provenance"] is False
        assert receipt["label_source"]["producer_commit"] is None


if __name__ == "__main__":
    os.environ.setdefault("TMPDIR", str(Path.home() / "tmp"))
    if len(sys.argv) == 3 and sys.argv[1] == "--real-fit":
        check_real_fit_keys(Path(sys.argv[2]))
        print(
            "PASS: pinned fit keys yield 4.34410740924913; no feature/development payload opened"
        )
    else:
        unittest.main()
