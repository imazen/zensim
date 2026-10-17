"""Native census exact sums, aliases and complete-view refusal; invented rows."""

import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "rev4_featpot"))
from e36_census_stream import census
from e36_data import PREFIX, canonical


class Census(unittest.TestCase):
    def fixture(self, root):
        directory = root / "runs/invented"
        (directory / "blobs").mkdir(parents=True)
        row = dict(
            size=32,
            codec="jpeg",
            physical_alias=False,
            bundle=dict(bytes=7),
            encode_execution=dict(wall_seconds=1.0, peak_rss_kib=10),
            score_execution=dict(wall_seconds=2.0, peak_rss_kib=20),
            feature_execution=dict(wall_seconds=3.0, peak_rss_kib=30),
            oracle_execution=dict(peak_rss_kib=40),
            cell_total_seconds=10.0,
            artifact_persistence_seconds=2.0,
            maps=[
                dict(execution=dict(wall_seconds=1e16, peak_rss_kib=50)),
                dict(execution=dict(wall_seconds=1.0, peak_rss_kib=60)),
            ],
            features=[42.0] * 420,
        )
        second = copy.deepcopy(row)
        second["cell_total_seconds"] = 20.0
        second["maps"][0]["execution"]["wall_seconds"] = -1e16
        alias = copy.deepcopy(row)
        alias["physical_alias"] = True
        alias["bundle"]["bytes"] = 999
        views = [dict(invented=0), dict(invented=1)]
        for i, rows in enumerate([[row], [second, alias]]):
            doc = dict(
                schema="e36-native-ladder-receipt-v1",
                complete=True,
                view=views[i],
                rows=rows,
            )
            (directory / f"blobs/{i}").write_bytes(
                gzip.compress(json.dumps(doc).encode())
            )
        (directory / "manifest.json").write_text(
            json.dumps([dict(kind=dict(view=PREFIX + canonical(v))) for v in views])
        )
        return directory

    def test_exact_global_sum_median_alias_and_complete_view_admission(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-census-"
        ) as d:
            root = Path(d)
            directory = self.fixture(root)
            census(root, "invented")
            result = json.loads((directory / "census.json").read_text())
            g = result["groups"]["32:jpeg"]
            self.assertTrue(result["admitted"])
            self.assertEqual(result["logical_rows"], 3)
            self.assertEqual(
                g,
                dict(
                    logical=3,
                    physical=2,
                    stage_median_seconds=dict(
                        encode_execution=1.0, score_execution=2.0, feature_execution=3.0
                    ),
                    peak_native_rss_kib=60,
                    bundle_bytes=14,
                    total_median_seconds=15.0,
                    total_sum_seconds=30.0,
                    persistence_sum_seconds=4.0,
                    map_sum_seconds=2.0,
                ),
            )

    def test_duplicate_native_view_remains_an_admission_failure(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-census-"
        ) as d:
            root = Path(d)
            directory = self.fixture(root)
            (directory / "blobs/duplicate").write_bytes(
                (directory / "blobs/0").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "complete the declared matrix"):
                census(root, "invented")
            self.assertFalse(
                json.loads((directory / "census.json").read_text())["admitted"]
            )


if __name__ == "__main__":
    unittest.main()
