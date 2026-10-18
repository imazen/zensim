"""Explicit comparison admission and actual canonical-owner integration."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v_next'))
import build_peer_fullevals as build
import gauntlet
from gauntlet_frozen import key
from gauntlet_frozen import move_intermediates, sha


class FrozenComparison(unittest.TestCase):
    def fixture(self):
        return dict(schema='zensim-frozen-gauntlet-v1', notes=['Independent fixture'],
            references={}, models=[dict(name='unknown-model',regime='fixture',model={})],
            datasets={'new-corpus':dict(target=[1.,4.,2.,6.,5.,3.],reference=[0,0,1,1,2,2],axis='mos',
                scores={'unknown-model':dict(pred=[2.,3.,1.,5.,6.,4.],status='withheld-fold',note='entire source withheld')})})

    def test_pixel_keys_distinguish_codec_and_distortion_folders(self):
        self.assertNotEqual(key('/r/001.png','/d/001/a/q30.jpg'),key('/r/001.png','/d/001/b/q30.jpg'))
        self.assertNotEqual(key('/r/001.png','/d/wn/img1.bmp'),key('/r/001.png','/d/jpeg/img1.bmp'))

    def test_training_and_nonfinite_rows_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'manifest.json'
            for status,pred in [('in-sample',[1.]*6),('withheld-fold',[math.inf]*6)]:
                doc=self.fixture();doc['datasets']['new-corpus']['scores']['unknown-model'].update(status=status,pred=pred)
                source.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):build.build_comparison(source,root/'out')


class IntermediateRelocation(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';source.mkdir();records=[]
        for group in ['pipal-png','pipal-aliases','pipal-png.attempt1-disguised-png-refusal','scoring']:
            path=source/group/'nested'/'payload.dat';path.parent.mkdir(parents=True)
            path.write_bytes(group.encode())
            records.append(dict(relative=str(path.relative_to(source)),sha256=sha(path),bytes=path.stat().st_size))
        (source/'ARCHIVE_FILES.json').write_text(json.dumps(dict(files=records)))
        (source/'gauntlet.html').write_text('final dashboard stays local')
        return source,root/'destination',records

    def test_verified_move_preserves_old_paths_and_final_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination,records=self.fixture(Path(tmp))
            move_intermediates(source,destination)
            for r in records:
                self.assertEqual(sha(source/r['relative']),r['sha256'])
                self.assertEqual(sha(destination/r['relative']),r['sha256'])
                self.assertTrue((source/r['relative'].split('/')[0]).is_symlink())
            self.assertEqual((source/'gauntlet.html').read_text(),'final dashboard stays local')
            self.assertEqual(len(json.loads((source/'INTERMEDIATES_RELOCATED.json').read_text())['groups']),4)

    def test_changed_original_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination,records=self.fixture(Path(tmp))
            path=source/records[0]['relative'];path.write_bytes(b'new unarchived work')
            with self.assertRaisesRegex(ValueError,'original intermediate changed'):
                move_intermediates(source,destination)
            self.assertEqual(path.read_bytes(),b'new unarchived work')
            self.assertFalse((source/'pipal-png').is_symlink())

    def test_bad_NAS_readback_never_unlinks_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination,records=self.fixture(Path(tmp))
            def bad_copy(src,dest):Path(dest).write_bytes(b'bad transport')
            with patch('gauntlet_frozen.shutil.copyfile',side_effect=bad_copy):
                with self.assertRaisesRegex(ValueError,'NAS readback mismatch'):
                    move_intermediates(source,destination)
            self.assertEqual(sha(source/records[0]['relative']),records[0]['sha256'])
            self.assertFalse((source/'pipal-png').is_symlink())



if __name__=='__main__':unittest.main()
