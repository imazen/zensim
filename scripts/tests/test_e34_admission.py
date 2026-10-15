"""E34 inventory path admission distinguishes populations from image subjects."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

source=Path(__file__).resolve().parents[1]/'rev4_featpot/e34_augmentation.py'
spec=importlib.util.spec_from_file_location('e34_inventory',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Admission(unittest.TestCase):
    def test_protected_population_paths_and_ordinary_terminal_subject(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch)
            image=root/'train'/'8146_terminal-python-repl.sdr.png'
            image.parent.mkdir();image.write_bytes(b'fixture')
            self.assertEqual(module.admitted(image),image.resolve())
            for folder in ['_sealed','kadid_terminal','TERMINAL','t0','pool-t0']:
                p=root/folder/'image.png';p.parent.mkdir();p.write_bytes(b'fixture')
                with self.assertRaises(ValueError):module.admitted(p)

if __name__=='__main__':unittest.main()
