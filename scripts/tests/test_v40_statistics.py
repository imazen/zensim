"""Independent seed-composite and signed lower-tail oracles."""

import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from v40_score import sdr_decision, _e29_signed_w2
import v40_score


class Statistics(unittest.TestCase):
    def test_changed_manifest_refuses_before_cell_or_checkpoint_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "program.tar.gz").write_bytes(b"program identity")
            (root / "bin/inspect_qualified_checkpoint").write_bytes(
                b"inspector identity"
            )
            manifest = root / "fit-manifest-fitv40-control-20261007.json"
            manifest.write_text('[{"payload": "unopened sentinel"}]')
            def digest(p):
                return hashlib.sha256(p.read_bytes()).hexdigest()
            (root / "PACKAGE_PINNED.json").write_text(
                json.dumps(
                    dict(
                        program_sha=digest(root / "program.tar.gz"),
                        inspector_sha=digest(root / "bin/inspect_qualified_checkpoint"),
                        manifests={"fitv40-control-20261007": "0" * 64},
                    )
                )
            )

            def tripwire(*args, **kwargs):
                raise AssertionError("checkpoint/payload verifier reached")

            with patch.dict(
                "sys.modules",
                {
                    "qualified_fit_contract": types.SimpleNamespace(
                        trusted_contract=tripwire, verify_training=tripwire
                    )
                },
            ):
                with self.assertRaisesRegex(
                    ValueError, "frozen registered manifest changed"
                ):
                    v40_score.complete(root, "e29", root, root, root, only_control=True)

    def test_seed_composites_and_one_sided_df9(self):
        delta = np.arange(40, dtype=float).reshape(10, 4) / 10000 + 0.003
        w2 = np.arange(20, dtype=float).reshape(10, 2) / 10000 + 0.001
        result = sdr_decision(delta, w2, "e32")
        seeds = [sum(row) / 4 for row in delta]
        mean = sum(seeds) / 10
        se = (sum((v - mean) ** 2 for v in seeds) / 9) ** 0.5 / 10**0.5
        self.assertAlmostEqual(result["signed"]["delta"], mean)
        self.assertAlmostEqual(result["signed"]["se"], se)
        self.assertAlmostEqual(result["p_one_sided"], t.sf(mean / se, 9))
        self.assertEqual(result["signed"]["seed_deltas"], seeds)
        self.assertTrue(result["adopt"])
        self.assertNotIn("adopt", sdr_decision(delta, w2, "e31"))

    def test_negative_signed_tail_is_not_flipped(self):
        result = _e29_signed_w2([-0.9, -0.8, -0.7, 0.1, 0.5])
        self.assertAlmostEqual(result["w2_type_worst3"], -0.8)
        self.assertEqual(result["w2_type_min"], -0.9)

    def test_undefined_missing_and_zero_se_refuse(self):
        for delta in (
            np.zeros((10, 4)),
            np.zeros((9, 4)),
            np.full((10, 4), float("nan")),
        ):
            with self.assertRaises(ValueError):
                sdr_decision(delta, np.arange(20).reshape(10, 2), "e32")
        with self.assertRaises(ValueError):
            sdr_decision(np.arange(40).reshape(10, 4), np.zeros((10, 2)), "e32")
        with self.assertRaises(ValueError):
            _e29_signed_w2([0.1, float("nan"), 0.4, 0.8])

    def test_constant_seed_composites_refuse_despite_std_rounding(self):
        from e24_rev5 import e29_seed_stat
        for constant in (0.001, 0.002, 0.0021, -0.004, 0.003, 0.0007):
            for row in ([constant] * 4, [constant - .01, constant + .01, constant, constant]):
                delta = np.tile(row, (10, 1))
                with self.assertRaisesRegex(ValueError, "zero seed SE"):
                    e29_seed_stat(delta)
                with self.assertRaisesRegex(ValueError, "zero seed SE"):
                    sdr_decision(delta, np.arange(20).reshape(10, 2), "e32")
        # A small but real variance remains defined; no tolerance changes the rule.
        varying = np.full((10, 4), .0021)
        varying[-1, :] += 1e-12
        self.assertGreater(e29_seed_stat(varying)["se"], 0)
        with self.assertRaisesRegex(ValueError, "zero seed SE"):
            sdr_decision(np.arange(40).reshape(10, 4), np.full((10, 2), .0021), "e32")

    def test_each_guard_remains_binding(self):
        delta = np.arange(40, dtype=float).reshape(10, 4) / 100000 + 0.004
        delta[:, 2] -= 0.011
        w2 = np.arange(20, dtype=float).reshape(10, 2) / 100000 + 0.001
        result = sdr_decision(delta, w2, "e32")
        self.assertFalse(result["guards"]["each_source"])
        self.assertFalse(result["adopt"])


if __name__ == "__main__":
    unittest.main()
