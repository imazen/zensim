"""Feature-only admission and selection checks for E34 preparation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import e34_prepare
from canonical_corpus import check_table_integrity


class CensusTests(unittest.TestCase):
    def test_contract_refuses_before_any_parquet_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = Path(tmp) / "contract.json"
            contract.write_text(json.dumps({"variants": []}))
            with patch.object(pq, "read_table", side_effect=AssertionError("payload read")):
                with self.assertRaisesRegex(ValueError, "frozen E33"):
                    e34_prepare.census(Path(tmp), contract)

    def test_masked_census_reads_only_requested_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "table.parquet"
            pq.write_table(pa.table({"f13": [0., -9., 2.], "f14": [3., 8., -1.],
                                     "human_score": [111., 222., 333.]}), table)
            real_read = pq.read_table
            calls = []

            def features_only(path, *, columns):
                self.assertEqual(columns, ["f13", "f14"])
                calls.append(columns)
                return real_read(path, columns=columns)

            with patch.object(pq, "read_table", side_effect=features_only):
                got = check_table_integrity.column_stats(
                    str(table), [(13, "f13"), (14, "f14")], row_mask=np.array([True, False, True]))
            self.assertEqual(len(calls), 1)
            self.assertEqual((got[13]["min"], got[13]["max"], got[13]["n"]), (0., 2., 2))
            self.assertEqual((got[14]["min"], got[14]["max"]), (-1., 3.))
            # The default retains all rows, preserving the existing owner's behavior.
            full = check_table_integrity.column_stats(str(table), [(13, "f13")])
            self.assertEqual((full[13]["min"], full[13]["n"]), (-9., 3))


if __name__ == "__main__":
    unittest.main()
