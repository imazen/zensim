"""Resource caps respect the recorded margin, available RAM and CPU reserves."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_fit_resources import cap,slots,GIB

class Resources(unittest.TestCase):
    def test_measured_margin_and_host_reserves(self):
        value,margin=cap(1024*1024)
        self.assertEqual(value,3*GIB//2);self.assertEqual(margin,GIB//2)
        for invalid in [0,-1,None,1.0]:
            with self.assertRaises(ValueError):cap(invalid)
        tower=slots('tower',dict(threads=32,available_bytes=20*GIB),value)
        self.assertEqual(tower['slots'],5);self.assertLessEqual(tower['aggregate_fit_caps_bytes']+tower['reserve_bytes'],20*GIB)
        small=slots('r3500',dict(threads=6,available_bytes=14*GIB),value)
        self.assertEqual(small['slots'],1);self.assertEqual(small['allowed_cpus'],2)
        self.assertEqual(slots('i265',dict(threads=20,available_bytes=7*GIB),value)['slots'],0)

if __name__=='__main__':unittest.main()
