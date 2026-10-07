"""Independent seed-composite and signed lower-tail oracles."""

import unittest

import numpy as np
from scipy.stats import t

from v40_score import sdr_decision, _e29_signed_w2


class Statistics(unittest.TestCase):
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

    def test_each_guard_remains_binding(self):
        delta = np.arange(40, dtype=float).reshape(10, 4) / 100000 + 0.004
        delta[:, 2] -= 0.011
        w2 = np.arange(20, dtype=float).reshape(10, 2) / 100000 + 0.001
        result = sdr_decision(delta, w2, "e32")
        self.assertFalse(result["guards"]["each_source"])
        self.assertFalse(result["adopt"])


if __name__ == "__main__":
    unittest.main()
