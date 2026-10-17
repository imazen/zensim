"""Admission refusals, using invented maps/scalars and no training calls."""
import copy
import gzip
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_data import validate_scores,validate_map,read_exact

class AdmissionTests(unittest.TestCase):
    def scores(self):
        return [dict(metric='butteraugli',scores={'butteraugli_pnorm3':2.25},score=2.25,
            reference_pixels_sha256='r',distorted_pixels_sha256='d'),
            dict(metric='cvvdp@standard_fhd',scores={'jod':7.125},score=7.125,
            reference_pixels_sha256='r',distorted_pixels_sha256='d')]
    def test_exact_teachers_and_pixel_bindings(self):
        self.assertEqual(validate_scores(self.scores(),'r','d'),(2.25,7.125))
        for mutate in [lambda x:x.pop(),lambda x:x[0].update(error='native failure'),
            lambda x:x[1].update(distorted_pixels_sha256='other'),
            lambda x:x[0]['scores'].update(other=float('nan')),
            lambda x:x[0]['scores'].update(another_pnorm3=1)]:
            rows=self.scores();mutate(rows)
            with self.assertRaises(ValueError):validate_scores(rows,'r','d')
    def test_maps_refuse_bad_shape_and_nonfinite_pixels(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-map-test-') as d:
            p=Path(d)/'map.gz'
            for values,valid in [([1.,2.,3.,4.],True),([1.,2.,3.],False),([1.,2.,3.,float('inf')],False)]:
                p.write_bytes(gzip.compress(b'Pf\n2 2\n-1.0\n'+struct.pack('<'+'f'*len(values),*values)))
                if valid:validate_map(p,2,2)
                else:
                    with self.assertRaises(ValueError):validate_map(p,2,2)
            with self.assertRaises(ValueError):validate_map(p,1,4)
    def test_warm_frame_is_not_silently_truncated(self):
        self.assertEqual(read_exact(io.BytesIO(b'abc'),3),b'abc')
        with self.assertRaises(EOFError):read_exact(io.BytesIO(b'ab'),3)

if __name__=='__main__':unittest.main()
