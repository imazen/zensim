"""Portable source/diagnostic retention without expanding opaque data."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_fit_packet import proof_relative
from e36_store import sha


class Proofs(unittest.TestCase):
    def test_external_diagnostic_is_content_addressed_and_tables_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)/'native';root.mkdir();proof=Path(d)/'receipt.json';proof.write_text('{"fits":0}')
            self.assertEqual(proof_relative(root,proof),Path('_external_proofs')/sha(proof)/proof.name)
            inside=root/'fixed.json';self.assertEqual(proof_relative(root,inside),Path('fixed.json'))
            forbidden=Path(d)/'opaque.parquet'
            with self.assertRaisesRegex(ValueError,'not a data table'):proof_relative(root,forbidden)
            self.assertFalse(forbidden.exists())


if __name__=='__main__':unittest.main()
