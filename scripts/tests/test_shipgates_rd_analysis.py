"""Numerical identity and strong-baseline regressions for RD interpolation."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "v_next/rd_probe_analyze_2026-07-18.py"
spec = importlib.util.spec_from_file_location("rd_owner", PATH)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class FrontierTests(unittest.TestCase):
    def test_measured_endpoint_bytes_are_exact(self):
        points=[(2.0,40354),(8.0,90531)]
        self.assertEqual(owner.interp_bytes(points,2.),40354)
        self.assertEqual(owner.interp_bytes(points,8.),90531)
        self.assertIsNone(owner.interp_bytes(points,1.9))
        self.assertIsNone(owner.interp_bytes(points,8.1))

    def test_dominated_rungs_cannot_inflate_the_scalar_baseline(self):
        points=[(1.,1200),(2.,900),(2.,950),(3.,1500),(4.,1400)]
        self.assertEqual(owner.rd_frontier(points),[(2.,900),(4.,1400)])
        self.assertEqual(owner.interp_bytes(owner.rd_frontier(points),2.),900)

if __name__ == "__main__":
    unittest.main()
