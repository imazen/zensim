"""Registered E28 gates use paired uncertainty and retain the SafeSyn preference."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
from e24_rev5 import e28_arm_decisions
from e13_teacher import paired


class E28Decision(unittest.TestCase):
    def test_both_correlations_and_as_good_are_required(self):
        sdr = {a: dict(as_good=True, signed_mean=0.003) for a in ("s2o", "s2m")}
        baseline = [0.6] * 10
        significant = paired([0.63 + i * 0.0001 for i in range(10)], baseline)
        uncertain = paired([0.6 + d for d in [-0.02, 0.03] * 5], baseline)
        self.assertGreater(uncertain["delta"], 0)
        self.assertLess(uncertain["delta"], 2 * uncertain["se"])
        pooled = {a: dict(krocc=significant, plcc_raw=significant) for a in sdr}
        decisions, adopted = e28_arm_decisions(sdr, pooled)
        self.assertTrue(all(v["passes"] for v in decisions.values()))
        self.assertEqual(adopted, "s2o")
        pooled["s2o"]["plcc_raw"] = uncertain
        decisions, adopted = e28_arm_decisions(sdr, pooled)
        self.assertFalse(decisions["s2o"]["passes"])
        self.assertEqual(adopted, "s2m")
        sdr["s2m"]["as_good"] = False
        self.assertIsNone(e28_arm_decisions(sdr, pooled)[1])

    def test_exact_two_se_is_not_a_recipe_signal(self):
        sdr = {a: dict(as_good=True) for a in ("s2o", "s2m")}
        pooled = {a: {m: dict(delta=0.02, se=0.01) for m in ("krocc", "plcc_raw")} for a in sdr}
        decisions, adopted = e28_arm_decisions(sdr, pooled)
        self.assertFalse(any(v["passes"] for v in decisions.values()))
        self.assertIsNone(adopted)


if __name__ == "__main__":
    unittest.main()
