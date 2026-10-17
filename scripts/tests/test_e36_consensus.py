"""Teacher direction, not nominal q, determines feature-only TV preferences."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('ladder_owner',
    Path(__file__).resolve().parents[1] / 'canonical_corpus/build_ladder_tv_pairs.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class Consensus(unittest.TestCase):
    def row(self, q, butter, cv):
        return dict(q=q, butteraugli_pnorm3=butter, cvvdp_standard_fhd=cv,
            reference_sha256='reference', codec='jpeg', size=32, split='fit',
            knobs={}, content='photo')

    def test_reversed_q_is_supervised_in_teacher_direction(self):
        pairs, counts = owner.e36_pairs([self.row(0, 1, 9), self.row(1, 2, 8)])
        self.assertEqual((pairs[0]['worse_row'], pairs[0]['better_row']), (1, 0))
        self.assertEqual(pairs[0]['weight'], 2)
        self.assertEqual(counts['reverse_nominal_q'], 1)

    def test_disagreement_tie_and_margin_are_excluded(self):
        for b in [self.row(1, .5, 8), self.row(1, 1, 9), self.row(1, .999, 9.001)]:
            pairs, counts = owner.e36_pairs([self.row(0, 1, 9), b])
            self.assertEqual(pairs, [])
            self.assertEqual(counts['excluded_disagreement_or_margin'], 1)

    def test_nonfinite_duplicate_and_unregistered_q_refuse(self):
        with self.assertRaises(ValueError):
            owner.e36_pairs([self.row(0, 1, 9), self.row(1, float('nan'), 10)])
        with self.assertRaises(ValueError):
            owner.e36_pairs([self.row(0, 1, 9), self.row(0, .5, 10)])
        with self.assertRaises(ValueError):
            owner.e36_pairs([self.row(21, 1, 9)])

    def test_no_cross_family_mode_or_split_pairs(self):
        for key, value in [('reference_sha256', 'other'), ('knobs', {'speed': 8}), ('split', 'selection')]:
            b = self.row(1, .5, 10)
            b[key] = value
            self.assertEqual(owner.e36_pairs([self.row(0, 1, 9), b])[0], [])

if __name__ == '__main__':
    unittest.main()
