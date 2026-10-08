"""Frozen panel pairing and zero-open assessment refusal regressions."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import v40_panels as owner


class Panels(unittest.TestCase):
    def test_external_pairing_uses_ten_four_fold_seed_units(self):
        panels = {a: {} for a in ("control", "palette")}
        for s in range(10):
            for k, f in enumerate(owner.PRODUCTION_SOURCES):
                panels["control"][f"{f}_s{s}"] = {"all": 0.5 + k / 100}
                panels["palette"][f"{f}_s{s}"] = {"all": 0.51 + k / 100 + s / 1000}
        result = owner.external_summary(panels)["all"]
        delta = np.arange(10) / 1000 + 0.01
        self.assertAlmostEqual(result["delta"], delta.mean())
        self.assertAlmostEqual(result["se"], delta.std(ddof=1) / np.sqrt(10))
        self.assertEqual(result["seed_units"], 10)
        self.assertTrue(result["report_only"])
        panels["control"].pop("kadid_s0")
        with self.assertRaisesRegex(ValueError, "forty-cell"):
            owner.external_summary(panels)

    def test_mciqa_dimensions_and_negative_rank_are_preserved(self):
        n = 20
        keys = pd.DataFrame(
            dict(
                human_score=np.arange(n),
                model=["a"] * n,
                gn_z=np.arange(n),
                cs_z=np.arange(n)[::-1],
                scm_z=np.roll(np.arange(n), 3),
            )
        )
        result = owner.external_metrics(np.arange(n), keys, "mciqa")
        self.assertAlmostEqual(result["dim:gn_z"], 1)
        self.assertAlmostEqual(result["dim:cs_z"], -1)
        self.assertEqual(
            set(result), {"all", "model:a", "dim:gn_z", "dim:cs_z", "dim:scm_z"}
        )

    def test_no_exposure_or_payload_read_with_incomplete_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(
                mode="external",
                bundle=root,
                results=root,
                control=root,
                tools=root,
                control_pins=root / "pins",
                exposure=root / "sentinel",
                out=root / "output",
            )
            with (
                patch.object(owner, "complete", side_effect=ValueError("INCOMPLETE")),
                patch.object(
                    owner,
                    "bound_bytes",
                    side_effect=AssertionError("label/exposure open reached"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
                    owner.run(args)
            self.assertFalse(args.out.exists())

    def test_every_ancestor_and_leaf_is_no_follow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "good"
            good.mkdir()
            (good / "sentinel").write_bytes(b"safe")
            alias = root / "alias"
            alias.symlink_to(good, target_is_directory=True)
            leaf = root / "leaf"
            leaf.symlink_to(good / "sentinel")
            self.assertEqual(owner.bound_bytes(good / "sentinel"), b"safe")
            for path in (alias / "sentinel", leaf):
                with self.assertRaises(OSError):
                    owner.bound_bytes(path)

    def test_false_exposure_refuses_before_payload_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = root / "freeze.json"
            freeze.write_text(
                json.dumps(
                    dict(
                        schema="v40-assessment-exposure-freeze-v1",
                        label_read_authorized=False,
                    )
                )
            )
            with patch.object(owner, "sha", side_effect=AssertionError("hash reached")):
                with self.assertRaises(PermissionError):
                    owner.exposure(freeze, root, "external", root / "pins")


if __name__ == "__main__":
    unittest.main()
