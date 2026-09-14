#!/usr/bin/env python3
"""Split-boundary regression checks; synthetic manifests, no corpus reads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import feature_screen_ceiling as screen


class SplitBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="zensim-split-", dir=Path.home() / "tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recipe = {
            "schema": screen.RECIPE_SCHEMA, "split_policy": screen.SPLIT_POLICY,
            "tasks": ["human"], "epochs": 2, "pairs_per_epoch": 16,
            "input_segments": [self.segment("train"), self.segment("eval")],
        }

    def segment(self, role):
        source = dict(corpus="fixture", origin=role + "-source", source_family=role + "-family", split=role)
        admission = self.root / (role + "-admission.json")
        admission.write_text(json.dumps(dict(schema="zensim-source-admission-v1",
                                            authority="synthetic unit fixture", sources=[source])))
        row = dict(corpus=source["corpus"], origin=source["origin"], source_family=source["source_family"],
                   task="human", family="noise", target=80, reference=str(self.root / (role + "-ref.png")),
                   distorted=str(self.root / (role + "-dist.png")))
        path = self.root / (role + "-segment.json")
        path.write_text(json.dumps(dict(schema="zensim-feature-segment-v1", role=role, rows=[row])))
        return dict(role=role, path=str(path), sha256=screen.sha(path),
                    admission=dict(path=str(admission), sha256=screen.sha(admission)))

    def test_valid_admission_reads_no_pixels(self):
        rows, paths = screen.inputs(self.recipe)
        self.assertEqual([r["role"] for r in rows], ["train", "eval"])
        self.assertEqual(len(paths), 4)
        self.assertFalse(Path(rows[0]["reference"]).exists())

    def test_legacy_recipe_refused_before_any_data_open(self):
        for schema in ("zensim-feature-screen-recipe-v1", "zensim-feature-ceiling-recipe-v1"):
            recipe = dict(self.recipe, schema=schema)
            with patch.object(Path, "open", side_effect=AssertionError("data opened")):
                with self.assertRaisesRegex(ValueError, "retired"):
                    screen.inputs(recipe)

    def test_test_role_refused_before_any_data_open(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["input_segments"][1]["role"] = "test"
        with patch.object(Path, "open", side_effect=AssertionError("data opened")):
            with self.assertRaisesRegex(ValueError, "only train and eval"):
                screen.inputs(recipe)

    def test_direct_segment_refuses_test_before_admission(self):
        segment = dict(self.recipe["input_segments"][0], role="test")
        with patch.object(Path, "open", side_effect=AssertionError("data opened")):
            with self.assertRaisesRegex(ValueError, "before opening"):
                screen.admitted_segment(segment)

    def test_protected_path_and_symlink_refused(self):
        protected = self.root / "dataset_test.parquet"
        link = self.root / "apparently_train.parquet"
        link.symlink_to(protected)
        for path in (protected, link, self.root / "terminal" / "rows.json"):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                screen.allowed_path(path)

    def test_mixed_cache_refused_without_opening_it(self):
        recipe = dict(self.recipe, reuse_prepared={"path": "/does/not/exist"})
        with patch.object(Path, "open", side_effect=AssertionError("data opened")):
            with self.assertRaisesRegex(ValueError, "reuse is retired"):
                screen.inputs(recipe)

    def test_admission_catches_test_source_before_payload_read(self):
        segment = self.recipe["input_segments"][1]
        admission = Path(segment["admission"]["path"])
        data = json.loads(admission.read_text())
        data["sources"][0]["split"] = "test"
        admission.write_text(json.dumps(data))
        segment["admission"]["sha256"] = screen.sha(admission)
        Path(segment["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "test/terminal"):
            screen.admitted_segment(segment)

    def test_source_family_cannot_cross_tasks_or_roles(self):
        rows, _ = screen.inputs(self.recipe)
        rows[1]["source_family"] = rows[0]["source_family"]
        rows[1]["task"] = "codec"
        with self.assertRaisesRegex(ValueError, "crosses"):
            screen.validate_rows(rows)

    def test_retagged_row_refused(self):
        rows, _ = screen.inputs(self.recipe)
        rows[1]["source_split"] = "test"
        rows[1]["role"] = "train"
        with self.assertRaisesRegex(ValueError, "canonically admitted"):
            screen.validate_rows(rows)

    def test_input_bytes_are_pinned(self):
        segment = self.recipe["input_segments"][0]
        Path(segment["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            screen.admitted_segment(segment)

    def test_payload_source_split_is_not_silently_renamed(self):
        segment = self.recipe["input_segments"][0]
        payload = Path(segment["path"])
        data = json.loads(payload.read_text())
        data["rows"][0]["source_split"] = "test"
        payload.write_text(json.dumps(data))
        segment["sha256"] = screen.sha(payload)
        with self.assertRaisesRegex(ValueError, "cannot be relabeled"):
            screen.admitted_segment(segment)

    def test_renaming_family_does_not_hide_origin_overlap(self):
        rows, _ = screen.inputs(self.recipe)
        rows[1]["origin"] = rows[0]["origin"]
        with self.assertRaisesRegex(ValueError, "source/reference crosses"):
            screen.validate_rows(rows)

    def test_trainer_gets_only_training_groups(self):
        for task in ("human", "codec", "corruption"):
            for fraction in ("full", "half"):
                command = screen.training_command("trainer", self.root, self.recipe, task,
                                                  32, fraction, 5101, [0, 1], self.root / "model.bin")
                groups = [command[i+1] for i, x in enumerate(command) if x == "--group"]
                self.assertTrue(groups)
                self.assertTrue(all(":1:0:" in group for group in groups))
                self.assertTrue(all("eval" not in group and "test" not in group and "dev" not in group for group in groups))
                self.assertIn("--no-auto-eval", command)
                self.assertEqual(command[command.index("--early-stop-patience")+1], "0")

    def test_targeted_capacity_controls_do_not_expand_other_layouts(self):
        recipe = dict(arms={"cheap": [0], "wide": [0, 1]}, hidden=128,
                      capacity_arms={"cheap": [32], "wide": [256]},
                      control_arms=[], half_data_controls=False)
        self.assertEqual(screen.fit_specs(recipe), [
            ("cheap", 128, "full"), ("wide", 128, "full"),
            ("cheap", 32, "full"), ("wide", 256, "full")])
        for capacity in ({"cheap": [128]}, {"absent": [32]}, {"wide": [0]}):
            with self.assertRaises(ValueError):
                screen.fit_specs(dict(recipe, capacity_arms=capacity))


if __name__ == "__main__":
    unittest.main()
