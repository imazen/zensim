#!/usr/bin/env python3
"""Palette identity admission negative controls; no pixels or labels required."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

OWNER = Path(__file__).resolve().parents[1] / 'rev4_featpot/rev5_bank.py'
spec = importlib.util.spec_from_file_location('palette_bank', OWNER)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class PaletteAdmission(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', str(Path.home() / 'tmp')))
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name) / 'instrument.json'
        self.build = 'a' * 40
        self.bank = 'b' * 64
        self.manifest = dict(schema='palette-instrument-views-v1', build_commit=self.build,
            bank_manifest_sha256=self.bank, labels_read=False, serving_allowed=False,
            feature_set_id='palette@w1867/palette_v2#30b09cd1', feature_ids=owner.PALETTE_IDS,
            column_map={str(i): f'palette_f{i}' for i in owner.PALETTE_IDS},
            views={name: {} for name in owner.PALETTE_TABLES})

    def write(self, manifest):
        self.path.write_text(json.dumps(manifest, indent=2) + '\n')
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def admit(self, pin):
        return owner.validate_palette_instrument_manifest(self.path, self.build, self.bank, pin)

    def test_current_identity(self):
        pin = self.write(self.manifest)
        self.assertEqual(self.admit(pin), self.manifest)

    def test_palette_v1_identity_refused_even_with_matching_byte_pin(self):
        altered = copy.deepcopy(self.manifest)
        altered['feature_set_id'] = 'palette@w1867/palette_v1#30b09cd1'
        pin = self.write(altered)
        with self.assertRaises(ValueError): self.admit(pin)

    def test_swapped_mapping_refused_even_with_matching_byte_pin(self):
        altered = copy.deepcopy(self.manifest)
        altered['column_map']['1825'], altered['column_map']['1826'] = altered['column_map']['1826'], altered['column_map']['1825']
        pin = self.write(altered)
        with self.assertRaises(ValueError): self.admit(pin)

    def test_other_semantic_changes_refused(self):
        for field, value in [('feature_ids', list(reversed(owner.PALETTE_IDS))),
                ('build_commit', 'c' * 40), ('serving_allowed', True),
                ('serving_allowed', 0), ('labels_read', 0), ('schema', 'unknown'),
                ('feature_set_id', 'palette@w1867/unknown#30b09cd1')]:
            with self.subTest(field=field, value=value):
                altered = copy.deepcopy(self.manifest); altered[field] = value
                pin = self.write(altered)
                with self.assertRaises(ValueError): self.admit(pin)

    def test_missing_pin_refused(self):
        self.write(self.manifest)
        with self.assertRaises(ValueError): self.admit(None)

    def test_byte_pin_checked_not_only_parsed_identity(self):
        pin = self.write(self.manifest)
        with self.path.open('a') as stream: stream.write(' ')
        with self.assertRaises(ValueError): self.admit(pin)

    def test_wrong_key_membership_and_bank_refused(self):
        for field, value in [('views', {'aic3': {}}), ('bank_manifest_sha256', 'c' * 64), ('labels_read', True)]:
            with self.subTest(field=field):
                altered = copy.deepcopy(self.manifest); altered[field] = value
                pin = self.write(altered)
                with self.assertRaises(ValueError): self.admit(pin)

if __name__ == '__main__': unittest.main()
