"""Independent E29 numerical cases and population-open tripwires."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import e29_consensus as e29
import e21_cheap_recipe as e21
import v2_common as common
import v2_lodo_mlp as fit
import e24_rev5 as e24


class ConsensusTests(unittest.TestCase):
    def test_borda_ties_and_order_against_manual_ranks(self):
        # A ranks 1.5,1.5,4,3; B ranks 4,3,1.5,1.5.
        actual = e29.consensus([1, 1, 3, 2], [4, 3, 1, 1])
        np.testing.assert_array_equal(actual, np.array([3.5, 2.5, 3.5, 2.5]) / 6)
        np.testing.assert_array_equal(e29.consensus([3, 2, 1], [3, 2, 1]), [1, .5, 0])
        with self.assertRaises(ValueError):
            e29.consensus([1, float("nan")], [1, 2])

    def test_agreement_cross_reference_and_raw_threshold(self):
        a, b, refs = [0, .05, .2, .3], [0, .05, -.2, .3], ["a", "b", "c", "a"]
        self.assertEqual(list(e29.agreement_pairs(a, b, refs)), [[0, 1], [1, 3], [2, 3]])
        self.assertEqual(list(e29.agreement_pairs([0, .049], [0, .2], ["a", "b"])), [])
        with self.assertRaises(ValueError):
            list(e29.agreement_pairs([0, float("inf")], [0, 1], refs[:2]))

    def test_tokens_require_single_exact_arm(self):
        base = "sel:59f0bbc2f290@h32:H128:cv16:cf98:"
        for arm in ("hb4", "hc4"):
            r = common.recipe_of(base + arm)
            self.assertEqual((r["hdr_weight"], r["hdr_mode"], r["hdr_consensus"]), (4, "rank", arm))
        for arm in ("hb16", "hc0", "hb4:hc4", "hp4:hb4"):
            with self.assertRaises(ValueError):
                common.recipe_of(base + arm)

    def test_ten_correlated_seed_units_and_both_teacher_gate(self):
        delta = np.repeat((np.arange(10) * .001 + .01)[:, None], 4, axis=1)
        stat = e24.e29_seed_stat(delta)
        self.assertEqual(stat["se"], float(delta[:, 0].std(ddof=1) / np.sqrt(10)))
        with self.assertRaises(ValueError):
            e24.e29_seed_stat(np.ones((10, 4)))
        cells = {a: [] for a in ("control", "hb4", "hc4")}
        for source in e29.PRODUCTION_SOURCES:
            for seed in range(10):
                for arm in cells:
                    gain = (seed * .001 + .01) * (0 if arm == "control" else 1)
                    cvgain = gain if arm != "hc4" else -gain
                    cells[arm].append(dict(source=source, seed=seed,
                        pooled=dict(hdrvdp3=.8+gain, cvvdp=.8+cvgain),
                        within_reference=dict(hdrvdp3=.9+gain, cvvdp=.9+gain)))
        decisions, winner = e24.hdr_arm_decisions(cells, {a:dict(as_good=True) for a in ("hb4", "hc4")}, "e29")
        self.assertEqual(winner, "hb4")
        self.assertFalse(decisions["hc4"]["hdr_pass"])
        cells["hb4"].pop()
        with self.assertRaises(ValueError):
            e24.hdr_arm_decisions(cells, {}, "e29")

    def metadata(self):
        return dict(study="E29", role="train", rows=7390, population="agree-only", formula_revision=5,
                    teacher_sha256=e29.TEACHER, requested_ids=e21.columns("by_v2fy"), arm="hb4",
                    build_commit="a" * 40, source_table_sha256=e29.SOURCE_TABLE,
                    source_keys_sha256=e29.SOURCE_KEYS, source_manifest_sha256=e29.SOURCE_MANIFEST,
                    source_bank_feature_set_id=None, target_transform="pooled-midrank-Borda-[0,1]")

    def test_wrong_population_refuses_before_any_key_or_payload_open(self):
        for key, value in (("role", "val"), ("rows", 3900), ("population", "all"),
                           ("teacher_sha256", "0" * 64), ("study", "E31"),
                           ("arm", "other"), ("source_table_sha256", "0" * 64)):
            d = self.metadata()
            d[key] = value
            with self.subTest(key=key), patch.object(e29.pq, "read_table", side_effect=AssertionError("payload open")) as read:
                with self.assertRaises(ValueError):
                    e29.admit_metadata(Path("/nonexistent/hdr.parquet"), d)
                read.assert_not_called()

    def test_lower_strict_owner_checks_all_populations_before_hash(self):
        d = self.metadata()
        d["role"] = "val"
        with patch("v2c_wide.safe_path", side_effect=lambda p: p), \
                patch.object(Path, "read_text", return_value=__import__("json").dumps(d)), \
                patch.object(fit, "sha", side_effect=AssertionError("payload hash")) as hashed, \
                patch.object(fit.pq, "read_table", side_effect=AssertionError("payload open")) as opened:
            with self.assertRaises(ValueError):
                fit.strict_training_groups([("hdr", Path("/nonexistent/hdr.parquet"), 4, 0, "rank")])
            hashed.assert_not_called()
            opened.assert_not_called()

    def test_wrong_key_roles_refuse_before_feature_or_pair_payload_hash(self):
        keys = pa.table({'row_id': list(range(7390)), 'role': ['val'] * 7390,
                         'agree': [True] * 7390, 'ref_basename': ['ref'] * 7390})
        with patch.object(e29.pq, 'read_table', return_value=keys) as read, \
                patch.object(e29, 'sha', side_effect=AssertionError('payload hash')) as hashed:
            with self.assertRaises(ValueError):
                e29.admit_metadata(Path('/nonexistent/hdr.parquet'), self.metadata())
            self.assertEqual(read.call_count, 1)  # permitted label-free keys only
            hashed.assert_not_called()

    def test_unfrozen_baseline_blocks_scorer_before_labels(self):
        args = Namespace(root='/nonexistent', preflight_only=False, results='/nonexistent',
                         control_root='/nonexistent', control_pins=None)
        with patch.object(e24.pq, 'read_table', side_effect=AssertionError('label read')) as read:
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
                e24.cmd_e29_score(args)
            read.assert_not_called()

    def test_unfrozen_baseline_blocks_hdr_panel_before_val(self):
        import importlib.util
        module_path = Path(__file__).resolve().parents[1] / 'hdr/hdr_route_panel.py'
        spec = importlib.util.spec_from_file_location('e29_hdr_import_tripwire', module_path)
        hdr = importlib.util.module_from_spec(spec)
        with patch.object(e29.pq, 'read_table', side_effect=AssertionError('import payload read')) as imported_read, \
                patch('subprocess.run', side_effect=AssertionError('import subprocess')) as process:
            spec.loader.exec_module(hdr)
            imported_read.assert_not_called()
            process.assert_not_called()
        with patch.object(hdr.pq, 'read_table', side_effect=AssertionError('HDR VAL read')) as read, \
                patch.object(Path, 'mkdir', side_effect=AssertionError('assessment output')) as mkdir:
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
                hdr._e26_panel('/nonexistent', '/nonexistent', '/nonexistent',
                               '/nonexistent', '/nonexistent', study='e29')
            read.assert_not_called()
            mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
