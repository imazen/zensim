"""Synthetic v4 admission boundaries; no scientific datasets."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v_next'))
import train_corruption_head as trainer


def admission_fixture(root):
    """Portable synthetic record used by format/ownership tests only."""
    def pin(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    ids = [13, 401, 719]
    table = root / 'train.csv'
    table.write_text(','.join(f'f{i}' for i in range(720)) + '\n' + ','.join(['0']*720) + '\n')
    producer = root / 'synthetic-decoder.bin'
    producer.write_bytes(b'synthetic fixture executable identity; never a real decoder')
    decoder = 'legacy-rgb8/executable-sha256:' + pin(producer)['sha256']
    identity = 'basic+peaks+v2@w1825/rev5_localwin#36c3f3af'
    declaration = Path(str(table) + '.manifest.json')
    declaration.write_text(json.dumps(dict(feature_set_id=identity, formula_revision=5, decoder_era=decoder)))
    selection = root / 'selection.json'
    selection.write_text(json.dumps(dict(schema='zcth-train-row-selection-v1', table_sha256=pin(table)['sha256'],
                                         roles=dict(fit=['synthetic-fit'], calibrate=['synthetic-calibrate']))))
    bindings = []
    for kind, value in [
        ('extraction-manifest', dict(feature_set_id=identity, formula_revision='5', producer_binary_sha256=pin(producer)['sha256'])),
        ('content-admission', dict(schema='canonical-corruption-content-admission-v1', complete=True, unresolved=[], scope='synthetic only')),
        ('recipe', dict(synthetic=True)), ('registration', dict(synthetic=True)),
        ('preparation', dict(synthetic=True)), ('rows', dict(synthetic=True))]:
        p = root / (kind + '.json'); p.write_text(json.dumps(value))
        bindings.append(pin(p) | dict(kind=kind))
    bindings.append(pin(producer) | dict(kind='decoder-producer'))
    return dict(schema='zcth-training-admission-v1', feature_set_id=identity, formula_revision=5,
                decoder_era=decoder, head_feature_ids=ids, scope='synthetic format fixture only',
                training_tables=[pin(table) | dict(role='TRAIN', usage=['fit', 'calibrate'],
                                                   declaration=pin(declaration), selection=pin(selection))],
                bindings=bindings)


class ZcthAdmission(unittest.TestCase):
    def test_synthetic_record_and_pinned_file_changes(self):
        with tempfile.TemporaryDirectory(prefix='zcth-admission-') as tmp:
            record = admission_fixture(Path(tmp))
            self.assertIs(trainer.verify_training_admission(record, 5, [13, 401, 719]), record)
            specs = [record['training_tables'][0], record['training_tables'][0]['declaration'],
                     record['training_tables'][0]['selection'], *record['bindings']]
            for spec in specs:
                p = Path(spec['path']); original = p.read_bytes(); p.write_bytes(original + b'changed')
                with self.subTest(path=p.name), self.assertRaisesRegex(ValueError, 'changed input'):
                    trainer.verify_training_admission(record, 5, record['head_feature_ids'])
                p.write_bytes(original)

    def test_forbidden_roles_and_paths_refuse_before_reads(self):
        with tempfile.TemporaryDirectory(prefix='zcth-roles-') as tmp:
            original = admission_fixture(Path(tmp))
            for role in ['EVAL', 'TEST', 'train', '', None]:
                record = copy.deepcopy(original); record['training_tables'][0]['role'] = role
                with self.subTest(role=role), patch.object(trainer, '_sha256', side_effect=AssertionError('payload read')), self.assertRaisesRegex(ValueError, 'TRAIN'):
                    trainer.verify_training_admission(record, 5, record['head_feature_ids'])
            for path in ['/var/tmp/_sealed/never-open.parquet', '/var/tmp/holdout/never-open.parquet', 'relative.csv']:
                record = copy.deepcopy(original); record['training_tables'][0]['path'] = path
                record['training_tables'][0]['declaration']['path'] = path + '.manifest.json'
                with self.subTest(path=path), patch.object(trainer, '_sha256', side_effect=AssertionError('payload read')), self.assertRaisesRegex(ValueError, 'protected/nonabsolute'):
                    trainer.verify_training_admission(record, 5, record['head_feature_ids'])
            record = copy.deepcopy(original)
            record['training_tables'][0]['source_table'] = '/var/tmp/_sealed/never-open.parquet'
            with patch.object(trainer, '_sha256', side_effect=AssertionError('payload read')), self.assertRaisesRegex(ValueError, 'protected/nonabsolute'):
                trainer.verify_training_admission(record, 5, record['head_feature_ids'])
            for field, value in [('formula_revision', 4), ('head_feature_ids', [13]), ('decoder_era', '')]:
                record = copy.deepcopy(original); record[field] = value
                with self.subTest(field=field), patch.object(trainer, '_sha256', side_effect=AssertionError('payload read')), self.assertRaises(ValueError):
                    trainer.verify_training_admission(record, 5, [13, 401, 719])

    def test_rehashed_declarations_and_selection_still_must_match(self):
        with tempfile.TemporaryDirectory(prefix='zcth-declarations-') as tmp:
            original = admission_fixture(Path(tmp))
            for field, value in [('feature_set_id', 'unknown'), ('formula_revision', 4), ('decoder_era', 'different')]:
                record = copy.deepcopy(original); spec = record['training_tables'][0]['declaration']; p = Path(spec['path'])
                previous = p.read_bytes(); data = json.loads(previous); data[field] = value; p.write_text(json.dumps(data))
                spec['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'declaration mismatch'):
                    trainer.verify_training_admission(record, 5, record['head_feature_ids'])
                p.write_bytes(previous)
            record = copy.deepcopy(original); spec = record['training_tables'][0]['selection']; p = Path(spec['path'])
            data = json.loads(p.read_text()); data['roles']['evaluate'] = ['forbidden']; p.write_text(json.dumps(data))
            spec['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'row selection/role mismatch'):
                trainer.verify_training_admission(record, 5, record['head_feature_ids'])


if __name__ == '__main__':
    unittest.main()
