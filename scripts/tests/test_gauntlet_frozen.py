"""Explicit comparison admission and actual canonical-owner integration."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v_next'))
import build_peer_fullevals as build
import gauntlet
from gauntlet_frozen import key


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



if __name__=='__main__':unittest.main()
