"""Peer units preserve the registered error and undershoot decisions."""
import importlib.util
from pathlib import Path
import sys
import unittest

OWNER=Path(__file__).resolve().parents[1]/"rev4_featpot"
sys.path.insert(0,str(OWNER))
spec=importlib.util.spec_from_file_location("peer_target",OWNER/"shipgates_peer_target.py")
peer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(peer)


class PeerUnits(unittest.TestCase):
    def test_ssim_span_and_thresholds(self):
        alpha,beta=peer.unit_mapping({5:20,95:80},{5:-40,95:95})
        self.assertEqual(alpha,2.25)
        self.assertEqual((95-beta)/alpha,80)
        for error in (-8,-3,-1,0,.5,1,2,3,8):
            raw=alpha*(70+error)+beta
            self.assertAlmostEqual((raw-beta)/alpha-70,error)

    def test_butter_distance_undershoot_orientation(self):
        alpha,beta=peer.unit_mapping({5:20,95:80},{5:-16,95:-1})
        target=-(alpha*70+beta)
        too_poor=target+2*alpha
        normalized=(-too_poor-beta)/alpha
        self.assertAlmostEqual(normalized-70,-2)

    def test_degenerate_spans_refused(self):
        for z,p in [({5:1,95:1},{5:1,95:2}),({5:1,95:2},{5:1,95:1}),({5:1,95:2},{5:float("nan"),95:2})]:
            with self.assertRaises(ValueError):
                peer.unit_mapping(z,p)


if __name__=="__main__":
    unittest.main()
