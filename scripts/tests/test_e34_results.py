"""Undefined registered reports must not be hidden by a complete decision."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from e34_results import validate_statistical_reports


class E34ReportCompletenessTest(unittest.TestCase):
    def fixture(self):
        interval = dict(estimate=0., lo=0., hi=0., se=0., n_finite=10000,
                        student_interval=[0., 0.])
        seed = dict(delta=0., se=.01, n=10)
        return dict(tests=[dict(test=1, state='COMPLETE', theta=interval,
            reference_only=interval, seed_unit=seed,
            w2=dict(interval=interval, seed_unit=seed),
            dataset_guards=[dict(interval=interval)],
            pair_classes=[dict(interval=interval, seed_unit_se=0.)])])

    def test_degenerate_but_defined_report_interval_is_admitted(self):
        validate_statistical_reports(self.fixture())

    def test_complete_hypothesis_cannot_hide_incomplete_seed_report(self):
        for field in ['seed_unit', 'w2']:
            with self.subTest(field=field):
                stats = self.fixture()
                incomplete = dict(state='INCOMPLETE', reason='undefined seed SE')
                if field == 'seed_unit':
                    stats['tests'][0][field] = incomplete
                else:
                    stats['tests'][0][field]['seed_unit'] = incomplete
                with self.assertRaisesRegex(AssertionError, 'report undefined'):
                    validate_statistical_reports(stats)

    def test_undefined_reference_or_pair_class_report_refuses(self):
        stats = copy.deepcopy(self.fixture())
        stats['tests'][0]['reference_only'] = dict(stats['tests'][0]['reference_only'], se=None)
        with self.assertRaisesRegex(AssertionError, 'interval undefined'):
            validate_statistical_reports(stats)
        stats = self.fixture()
        stats['tests'][0]['pair_classes'][0]['seed_unit_se'] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'pair-class seed SE undefined'):
            validate_statistical_reports(stats)


if __name__ == '__main__':
    unittest.main()
