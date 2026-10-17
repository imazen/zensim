"""Portable source/diagnostic retention without expanding opaque data."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_fit_packet import proof_relative
from e36_store import sha


class Proofs(unittest.TestCase):
    def test_unfinished_archive_preparation_cannot_make_fit_packet(self):
        import json
        from e36_fit_packet import packet
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);pools=root/'pools';pools.mkdir()
            (pools/'matrix-admission.json').write_text(json.dumps(dict(artifact_archive_pending=True)))
            with self.assertRaisesRegex(ValueError,'not final archive/pool admission'):
                packet(root,pools,root/'opaque-base',root/'trainer',root/'out',root/'owner','invented')
            self.assertFalse((root/'out').exists())

    def test_external_diagnostic_is_content_addressed_and_tables_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)/'native';root.mkdir();proof=Path(d)/'receipt.json';proof.write_text('{"fits":0}')
            self.assertEqual(proof_relative(root,proof),Path('_external_proofs')/sha(proof)/proof.name)
            inside=root/'fixed.json';self.assertEqual(proof_relative(root,inside),Path('fixed.json'))
            forbidden=Path(d)/'opaque.parquet'
            with self.assertRaisesRegex(ValueError,'not a data table'):proof_relative(root,forbidden)
            self.assertFalse(forbidden.exists())


if __name__=='__main__':unittest.main()

class ExactClosure(unittest.TestCase):
    def test_missing_or_changed_frozen_owner_refuses(self):
        from e36_fit_packet import admit_program_closure
        frozen='62a43287c53d9e19824a66b556745c2a8fdb9cf7';preparation=dict(published_source=frozen,compiled_rust_source='534cb68f0204c02807b1a5c1612d1fcb6c805365')
        files={'scripts/rev4_featpot/e36_training.py':'a','scripts/rev4_featpot/e36_store.py':'b'};preflight=dict(published_source=frozen,files={**files,'benchmarks/e29_e31_e32_shared_control_decision_2026-10-07.md':'m','benchmarks/e29_four_source_amendment_2026-10-07.md':'n'})
        admit_program_closure(files,preflight,preparation)
        with self.assertRaisesRegex(ValueError,'exact'):admit_program_closure({'scripts/rev4_featpot/e36_training.py':'a'},preflight,preparation)
        with self.assertRaisesRegex(ValueError,'authorized'):admit_program_closure(files,preflight,{**preparation,'published_source':'changed'})
