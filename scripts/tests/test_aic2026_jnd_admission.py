"""Admission tests; all statistics and pixel scoring stay with native owners."""
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.prepare_aic2026_evaluation import bind_jnd_peers, bind_jnd_scores, prepare_jnd


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

    def test_sparse_target_joins_exactly_once_without_rescoring(self):
        self.run_admission()
        scores = self.root / 'native.tsv'
        scores.write_text('id\tscore\terror\nPTC_S01_AVIF_01\t89.25\t\nPTC_S01_AVIF_02\t80.5\t\n')
        suite = bind_jnd_scores(self.root / 'out', scores)
        self.assertEqual(suite['metrics'][0]['source'], {'kind': 'column', 'column': 'score_b'})
        with Path(suite['datasets'][1]['path']).open() as f:
            sparse = list(csv.DictReader(f))
        self.assertEqual([(r['distorted'], r['score_b']) for r in sparse], [('PTC_S01_AVIF_02', '80.5')])

    def test_incomplete_score_population_refuses(self):
        self.run_admission()
        scores = self.root / 'native.tsv'
        scores.write_text('id\tscore\terror\nPTC_S01_AVIF_01\t89.25\t\n')
        with self.assertRaisesRegex(ValueError, 'exactly'):
            bind_jnd_scores(self.root / 'out', scores)

    def peer_fixture(self):
        self.run_admission()
        scores = self.root / 'native.tsv'
        scores.write_text('id\tscore\terror\nPTC_S01_AVIF_01\t89.25\t\nPTC_S01_AVIF_02\t80.5\t\n')
        bind_jnd_scores(self.root / 'out', scores)
        peers = self.root / 'peers.csv'
        columns = ['distorted', 'source', 'SSIMULACRA2', 'proposal-Butteraugli',
                   'proposal-DVIFM', 'proposal-DVIFM-0.2',
                   'proposal-DVIFM-0.2-use_chroma', 'GMSD']
        with peers.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in (2, 1):
                writer.writerow([f'PTC_S01_AVIF_0{i}.png', 'PTC_S01_Ref_00.png'] + [str(i)] * 6)
        return peers

    def test_peers_join_by_id_preserving_sparse_labels_and_saved_b(self):
        peers = self.peer_fixture()
        suite = bind_jnd_peers(self.root / 'out', peers, self.images, self.root / 'comparison')
        self.assertEqual(len(suite['metrics']), 7)
        with Path(suite['datasets'][1]['path']).open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r['distorted'], r['score_b'], r['dvifm']) for r in rows],
                         [('PTC_S01_AVIF_02', '80.5', '2')])

    def test_peer_identity_mismatch_refuses_before_writing(self):
        peers = self.peer_fixture()
        (self.images / 'PTC_S01_AVIF_01.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            bind_jnd_peers(self.root / 'out', peers, self.images, self.root / 'comparison')
        self.assertFalse((self.root / 'comparison').exists())

    def test_explicit_reproduced_tables_bind_columns_paths_and_directions(self):
        self.peer_fixture()
        scores = self.root / 'reproduced.tsv'
        with scores.open('w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ref_path', 'dist_path', 'raw_jnd'])
            for i in (2, 1):
                writer.writerow([self.images / 'PTC_S01_Ref_00.png',
                                 self.images / f'PTC_S01_AVIF_0{i}.png', str(i / 10)])
        specs = [dict(id='dvifm-v03-rs', path=str(scores), column='raw_jnd',
                      direction='lower', implementation='explicit synthetic retained fixture')]
        suite = bind_jnd_peers(self.root / 'out', None, self.images,
                               self.root / 'comparison', specs)
        self.assertEqual(suite['metrics'][1]['direction'], 'lower')
        with Path(suite['datasets'][1]['path']).open() as f:
            sparse = list(csv.DictReader(f))
        self.assertEqual(sparse[0]['dvifm-v03-rs'], '0.2')
        self.assertEqual(sparse[0]['score_b'], '80.5')

    def test_explicit_missing_column_and_invalid_metric_id_refuse(self):
        peers = self.peer_fixture()
        specs = [dict(id='dvifm-0.3', path=str(peers), column='absent',
                      direction='lower', implementation='synthetic fixture')]
        with self.assertRaisesRegex(ValueError, 'metric ID'):
            bind_jnd_peers(self.root / 'out', None, self.images,
                           self.root / 'comparison', specs)
        specs[0].update(id='dvifm-v03', delimiter=',')
        with self.assertRaises(KeyError):
            bind_jnd_peers(self.root / 'out', None, self.images,
                           self.root / 'comparison', specs)
        self.assertFalse((self.root / 'comparison').exists())


if __name__ == '__main__':
    unittest.main()
