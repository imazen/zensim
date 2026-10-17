"""Tracked native-source extraction, including git archive's parent directory."""
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_ingress_integration import extract_native

class SourceArchive(unittest.TestCase):
    def test_git_parent_directories_and_foreign_files(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);archive=root/'source.tar'
            with tarfile.open(archive,'w') as bundle:
                for name in ['scripts','scripts/jobsys']:
                    info=tarfile.TarInfo(name);info.type=tarfile.DIRTYPE;bundle.addfile(info)
                info=tarfile.TarInfo('scripts/jobsys/owner.txt');info.size=6;bundle.addfile(info,io.BytesIO(b'opaque'))
            out=root/'out';out.mkdir();extract_native(archive,out)
            self.assertEqual((out/'owner.txt').read_bytes(),b'opaque')
            with tarfile.open(archive,'w') as bundle:
                info=tarfile.TarInfo('unrelated.txt');info.size=1;bundle.addfile(info,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):extract_native(archive,out)

if __name__=='__main__':unittest.main()
