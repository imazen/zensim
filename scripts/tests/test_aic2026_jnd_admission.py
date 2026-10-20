"""Admission tests; all statistics and pixel scoring stay with native owners."""
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.prepare_aic2026_evaluation import prepare_jnd


class Admission(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'extracted/AIC2026-dataset-cropped/PTC-crops-all'
        self.images.mkdir(parents=True)
        self.header = (b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR'
                       + struct.pack('>II', 840, 944) + bytes([8, 2, 0, 0, 0]))
        for name in ('PTC_S01_Ref_00', 'PTC_S01_AVIF_01', 'PTC_S01_AVIF_02'):
            (self.images / f'{name}.png').write_bytes(self.header)
        self.labels = self.root / 'labels.csv'
        self.bake = self.root / 'frozen.bin'
        self.bake.write_bytes(b'identity-only-admission-fixture')
        self.rows = [dict(source='PTC_S01_Ref_00', distorted=f'PTC_S01_AVIF_0{i}',
                          bpp='1.5', JND_curve=str(i / 10),
                          JND_pointwise='' if i == 1 else '0.25') for i in (1, 2)]

    def run_admission(self):
        with self.labels.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return prepare_jnd(self.root, self.labels, self.root / 'out', self.bake, 'fixed-source')

    def test_population_polarity_and_no_score_before_single_run(self):
        result = self.run_admission()
        self.assertEqual(result['populations'], {'JND_curve': 2, 'JND_pointwise': 1})
        self.assertEqual(result['rows'], 2)
        self.assertEqual(len(result['images']), 3)
        suite = json.loads((self.root / 'out/suite.pending.json').read_text())
        self.assertTrue(all(d['target_direction'] == 'lower' for d in suite['datasets']))
        self.assertEqual(suite['metrics'][0]['source']['sha256'], '')
        self.assertFalse((self.root / 'out/scores.parquet').exists())
        with self.assertRaises(FileExistsError):
            self.run_admission()

    def test_missing_pair_refuses(self):
        (self.images / 'PTC_S01_AVIF_02.png').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_admission()
        self.assertFalse((self.root / 'out').exists())

    def test_duplicate_and_nonfinite_labels_refuse(self):
        self.rows[1] = dict(self.rows[0])
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.run_admission()
        self.rows[1]['distorted'] = 'PTC_S01_AVIF_02'
        self.rows[1]['JND_curve'] = 'NaN'
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.run_admission()

    def test_native_depth_and_dimensions_are_not_silently_changed(self):
        path = self.images / 'PTC_S01_AVIF_02.png'
        path.write_bytes(self.header[:24] + bytes([16]) + self.header[25:])
        with self.assertRaisesRegex(ValueError, 'RGB8'):
            self.run_admission()
        path.write_bytes(self.header[:16] + struct.pack('>II', 841, 944) + self.header[24:])
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.run_admission()


if __name__ == '__main__':
    unittest.main()
