#!/usr/bin/env python3
"""promote_fulleval --repair-rank-orientation: the equality gate compares numbers by
VALUE, and only by value.

A stored `0` and a fresh `0.0` are the same number spelled two ways (43 board blocks
spell `or` as `0` after a Python re-serialization pass), so the repair must accept
them. Any value that differs, however slightly (`0.0` vs `1e-12`), is not an
orientation-only correction and must still be refused, with the board file left
untouched (board_orientation_fix_2026-09-22 §1.3).
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import promote_fulleval as pf


def _block(or_value, per_ref_mean):
    return {"srocc": 0.91, "srocc_signed": -0.91, "plcc": 0.88, "krocc": 0.75,
            "or": or_value, "n": 300, "per_ref_n": 5,
            "per_ref_mean": per_ref_mean, "frac_negative": 1.0 if per_ref_mean < 0 else 0.0}


class NumCanon(unittest.TestCase):
    def same(self, a, b):
        return pf._jc(pf._num_canon(a)) == pf._jc(pf._num_canon(b))

    def test_int_and_equal_float_compare_equal(self):
        self.assertTrue(self.same(0, 0.0))
        self.assertTrue(self.same(1, 1.0))
        self.assertTrue(self.same({"or": 0, "v": [2, 3]}, {"or": 0.0, "v": [2.0, 3.0]}))

    def test_any_differing_value_is_refused(self):
        self.assertFalse(self.same(0.0, 1e-12))
        self.assertFalse(self.same(0, 1e-12))
        self.assertFalse(self.same({"or": 0.0}, {"or": 1e-12}))
        self.assertFalse(self.same(0, -0.0))

    def test_bool_is_not_a_number(self):
        self.assertFalse(self.same(True, 1.0))
        self.assertFalse(self.same(False, 0))

    def test_int_that_does_not_round_trip_stays_exact(self):
        big = 2**53 + 1
        self.assertEqual(pf._num_canon(big), big)
        self.assertIsInstance(pf._num_canon(big), int)
        self.assertFalse(self.same(big, float(big)))


class RepairGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, stored_or, fresh_or):
        board = {"name": "cell", "bake_sha256": "ab" * 32,
                 "rank": {"aic4": _block(stored_or, -0.9119), "cid22": _block(0.0, 0.93)}}
        verdict = {"name": "cell", "bake_sha256": "ab" * 32,
                   "rank": {"aic4": _block(fresh_or, 0.9119)}}
        bp, vp = self.dir / "cell.fulleval.json", self.dir / "cell.verdict.json"
        bp.write_text(json.dumps(board))
        vp.write_text(json.dumps(verdict))
        return bp, vp, copy.deepcopy(board)

    def test_int_zero_against_float_zero_is_repaired(self):
        bp, vp, before = self.write(0, 0.0)
        self.assertTrue(pf.repair_rank_orientation(bp, vp, "aic4",
                                                   tag="declared-orientation-2026-09-22"))
        after = json.loads(bp.read_text())
        self.assertEqual(after["rank"]["aic4"]["per_ref_mean"], 0.9119)
        self.assertEqual(after["rank"]["cid22"], before["rank"]["cid22"])
        src = after["rank_graft_sources"]["aic4"]
        self.assertEqual(src["repair"], "declared-orientation-2026-09-22")
        self.assertEqual(src["superseded_per_ref_mean"], -0.9119)

    def test_float_zero_against_1e_12_is_refused_and_board_untouched(self):
        bp, vp, _ = self.write(0.0, 1e-12)
        raw = bp.read_bytes()
        with self.assertRaises(SystemExit) as cm:
            pf.repair_rank_orientation(bp, vp, "aic4")
        self.assertIn("rank.aic4.or differs", str(cm.exception))
        self.assertEqual(bp.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
