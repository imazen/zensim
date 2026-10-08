"""Observation joins preserve repeated rows, inherited bits and auxiliary IDs."""

import unittest

import numpy as np
import pyarrow as pa

from e32_palette import PALETTE_IDS, append_named_projection


class Projection(unittest.TestCase):
    def bank(self):
        return pa.table(
            {
                "pair_key": ["b", "a"],
                "row_id": [0, 1],
                **{f"f{i}": [float(i), float(-i)] for i in PALETTE_IDS},
            }
        )

    def test_repeated_observations_keep_order_and_ieee_bits(self):
        values = np.array([0x7FC01234, 0x80000000, 0x3F800000], dtype=np.uint32).view(
            np.float32
        )
        original = pa.table({"f1825": values, "human_score": [4.0, 8.0, 4.0]})
        keys = pa.table(
            {
                "member_set": ["safesyn"] * 3,
                "pair_key": ["a", "b", "a"],
                "source_row_id": [70, 50, 70],
            }
        )
        result, digest = append_named_projection(
            original, keys, {"safesyn": self.bank()}
        )
        self.assertEqual(result["f1825"].to_numpy().tobytes(), values.tobytes())
        self.assertEqual(result["human_score"].to_pylist(), [4.0, 8.0, 4.0])
        self.assertEqual(
            result["palette_f1825"].to_pylist(), [-1825.0, 1825.0, -1825.0]
        )
        self.assertEqual(len(digest), 64)
        changed = keys.set_column(2, "source_row_id", pa.array([71, 50, 70]))
        self.assertNotEqual(
            digest,
            append_named_projection(original, changed, {"safesyn": self.bank()})[1],
        )

    def test_missing_pair_and_bank_row_order_refuse(self):
        original = pa.table({"human_score": [1.0]})
        keys = pa.table(
            {"member_set": ["safesyn"], "pair_key": ["missing"], "row_id": [0]}
        )
        with self.assertRaisesRegex(ValueError, "observation"):
            append_named_projection(original, keys, {"safesyn": self.bank()})
        with self.assertRaisesRegex(ValueError, "bank pair/row"):
            append_named_projection(
                original, keys, {"safesyn": self.bank().take(pa.array([1, 0]))}
            )

    def test_ordinal_original_index_gaps_are_preserved(self):
        original = pa.table({"human_score": [1.0, 2.0]})
        keys = pa.table({"__index_level_0__": [1, 9]})
        result, digest = append_named_projection(
            original, keys, {"coverage_pool": self.bank()}
        )
        self.assertEqual(result["palette_f1825"].to_pylist(), [1825.0, -1825.0])
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
