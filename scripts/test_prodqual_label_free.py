#!/usr/bin/env python3
"""Near-identity report regression controls; synthetic scores are not measurements."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from prodqual_label_free import nearid_candidate, nearid_summary


def fixture(root):
    for model in ("seed0", "B", "A"):
        rows = []
        for ref in range(24):
            rungs = [("identity", "0")] + [(f"one_pixel_{s}", "1") for s in (1, -1)]
            rungs += [("fraction", str(p)) for p in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1)]
            rungs += [("noise", str(k)) for k in (1, 2, 3, 4, 6, 8)]
            rungs += [("zenjpeg444", str(q)) for q in (100, 99, 98, 97, 95, 92, 90)]
            rungs += [("gaussian", str(s)) for s in (.05, .1, .2, .3, .5)]
            for index, (ladder, rung) in enumerate(rungs):
                identical = ladder in ("identity", "gaussian")
                score = 100 if identical else 97.0
                if ladder == "fraction":
                    score = {"1e-05": 99.5, "0.0001": 98.5, "0.001": 99.2}.get(rung, 89.0)
                rows.append(dict(model=model, reference=str(ref),
                                 **{"class": ("photo", "screen", "line_art")[ref // 8]},
                                 ladder=ladder, rung=rung, rung_index=index, identical=identical,
                                 changed_pixels=0 if identical else 1, served_score=score,
                                 raw_model_score=94, precalibration_score=93,
                                 neutral_fragility_score=99.8))
        (root / f"{model}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")


class NearIdentityReport(unittest.TestCase):
    def setUp(self):
        self.scratch = Path.home() / "tmp"
        self.scratch.mkdir(exist_ok=True)

    def test_identical_excluded_and_all_recrossings_retained(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                nearid_summary(root)
            result = json.loads((root / "SUMMARY.json").read_text())
            self.assertEqual(result["aggregate"]["seed0"]["highest_nonidentical_score"], 99.5)
            ladder = result["references"][0]["ladders"]["fraction"]
            self.assertFalse(ladder["nonincreasing"])
            self.assertEqual(len(ladder["reversals"]), 1)
            crossing = ladder["crossings"]["99"]
            self.assertEqual(crossing["first_below_rung"], "0.0001")
            self.assertEqual([t["direction"] for t in crossing["all_transitions"]],
                             ["below", "above", "below"])

    def test_missing_rung_refuses_before_summary_creation(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "A.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]))
            with self.assertRaises(AssertionError):
                nearid_summary(root)
            self.assertFalse((root / "SUMMARY.json").exists())

    def test_candidate_gates_apply_the_registered_bars(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "candidate-c.jsonl").write_text((root / "seed0.jsonl").read_text())
            result = nearid_candidate(root, "c")
            # Fixture one-pixel rungs score 97.0 (< 99): N1 fails on all 24; highest 99.5 passes N2;
            # the fraction ladder reverses on every reference, so 120 of 144 ladders are nonincreasing.
            self.assertEqual((result["N1"]["pass"], len(result["N1"]["failing"])), (False, 24))
            self.assertTrue(result["N2"]["pass"])
            self.assertEqual((result["N3"]["monotone_ladders"], result["N3"]["total_ladders"]), (120, 144))
            self.assertFalse(result["N3"]["pass"])


if __name__ == "__main__":
    unittest.main()
