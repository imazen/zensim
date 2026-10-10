"""E33 scorer report pieces on synthetic data: C-vs-A improvement test and the high-quality slice."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import e33_score as owner
from v2_human_role import PRODUCTION_SOURCES


def panels(delta_c):
    out = {label: {} for label in owner.LABELS}
    for s in range(10):
        for i, f in enumerate(PRODUCTION_SOURCES):
            base = 0.8 + 0.01 * i
            out["control"][f"{f}_s{s}"] = {"signed": base}
            out["a"][f"{f}_s{s}"] = {"signed": base + 0.001}
            out["c"][f"{f}_s{s}"] = {"signed": base + 0.001 + delta_c + 0.0005 * ((s + i) % 3 - 1)}
    return {"panels": out}


class Score(unittest.TestCase):
    def test_improvement_needs_mean_above_bar_and_significance(self):
        self.assertTrue(owner.improvement(panels(0.01))["c_beats_a"])
        self.assertFalse(owner.improvement(panels(0.001))["c_beats_a"])  # mean below +0.002
        flat = panels(0.0)
        for v in flat["panels"]["c"].values():
            v["signed"] = 0.5
        for v in flat["panels"]["a"].values():
            v["signed"] = 0.5
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            owner.improvement(flat)

    def test_high_quality_slice_reads_saved_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root, out = Path(tmp) / "root", Path(tmp) / "out"
            real = root / "wide/main/real"
            real.mkdir(parents=True)
            legs, rng = {}, np.random.default_rng(1)
            for f in PRODUCTION_SOURCES:
                y = rng.normal(size=50)
                pq.write_table(pa.table({"human_score": y}), real / f"{f}.parquet")
                legs[f] = {"full": {"rel": f"wide/main/real/{f}.parquet"}}
                for label, noise in (("control", 1.0), ("a", 1.0), ("c", 0.1)):
                    for s in range(10):
                        d = out / label / f"{f}_s{s}"
                        d.mkdir(parents=True)
                        pred = y + noise * rng.normal(size=50)
                        d.joinpath("pred.tsv").write_text("row_idx\tpred\n" + "".join(
                            f"{i}\t{p}\n" for i, p in enumerate(pred)))
            (real / "receipt.json").write_text(json.dumps({"legs": legs}))
            with mock.patch.object(owner, "ROOT", root):
                result = owner.high_quality_slice(panels(0.0), out)
            for f in PRODUCTION_SOURCES:
                self.assertEqual((result[f]["slice_rows"], result[f]["rows"]), (10, 50))
                self.assertTrue(result[f]["quality_oriented"])
                self.assertGreater(result[f]["c_minus_control"], result[f]["a_minus_control"])


if __name__ == "__main__":
    unittest.main()
