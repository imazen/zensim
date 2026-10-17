"""Direction is diagnostic; mentor signs and the registered clamp stay explicit."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/"rev4_featpot"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("monotonicity",SCRIPTS/"shipgates_monotonicity.py")
owner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class DirectionTests(unittest.TestCase):
    def test_following_a_negative_mentor_edge_is_not_a_strict_increase(self):
        rows=[dict(image_id="source",codec="test",q=i) for i in range(4)]
        mentor={owner.key(r):-float(r["q"]) for r in rows}
        scores={owner.key(r):-.1*float(r["q"]) for r in rows}
        row=owner.diagnostics(rows,scores,mentor)[0]
        self.assertFalse(row["strict_represented"])
        self.assertTrue(row["direction_represented"])
        self.assertEqual(sum(e["agreement"] for e in row["edges"]),3)

    def test_materiality_ties_do_not_create_direction_failures(self):
        self.assertIsNone(owner.edge_agrees(-10.,.49,.5))
        self.assertFalse(owner.edge_agrees(0.,.5,.5))
        self.assertTrue(owner.edge_agrees(-.01,-.5,.5))
        self.assertFalse(owner.edge_agrees(.01,-.5,.5))

    def test_shared_instrument_floor_still_blocks_the_direction_variant(self):
        rows=[dict(image_id=s,codec="test",q=i) for s in ["a","b"] for i in range(4)]
        mentor={owner.key(r):float(r["q"]) for r in rows}
        result=owner.diagnostics(rows,mentor,mentor)
        self.assertTrue(all(r["strict_ordered"] for r in result))
        self.assertTrue(all(r["clamped"] for r in result))
        self.assertTrue(all(not r["direction_represented"] for r in result))


if __name__=="__main__":unittest.main()
