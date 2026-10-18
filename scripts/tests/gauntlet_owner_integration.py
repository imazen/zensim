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

    def test_unknown_rows_keep_scatter_and_source_interval(self):
        if not os.environ.get('ZEN_PANEL_BIN'):
            self.fail('caller must bind the canonical panel through ZEN_PANEL_BIN')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'manifest.json';source.write_text(json.dumps(self.fixture()))
            build.build_comparison(source,root/'out')
            record=json.loads((root/'out/unknown-model.fulleval.json').read_text())
            self.assertEqual(record['rank']['new-corpus']['srocc_ci_method']['units'],3)
            self.assertEqual(record['rank']['new-corpus']['n'],6)
            rows=gauntlet.load_fulleval(root/'out',include_all_scatter=True,comparison_only=True)
            self.assertEqual(len(rows[0]['scatter']['new-corpus']['mos']['pts']),6)
            self.assertEqual(rows[0]['trainingOverlap']['new-corpus'],'entire source withheld')
            gauntlet.build_html(rows,root/'fixture.html',comparison_only=True)
            page=(root/'fixture.html').read_text()
            self.assertIn('Exact statistics, intervals, training overlap and downloads',page)
            self.assertIn('new-corpus',page)


if __name__=='__main__':unittest.main()
