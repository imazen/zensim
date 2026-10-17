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
    def test_frozen_tier_keeps_partial_windows_and_uses_only_mentor(self):
        mentor={("a","jpeg",i):v for i,v in enumerate([0.,.1,.5,.4,-.1])}
        mentor.update({("b","jpeg",i):v for i,v in enumerate([0.,.1,.2])})
        edges,excluded=owner.direction_edges(mentor)
        self.assertEqual([(e["from_q"],e["to_q"]) for e in edges],[(0.,2.),(2.,4.)])
        self.assertEqual([e["image_id"] for e in excluded],["b"])
        self.assertTrue(all(owner.direction_flag(e,mentor) for e in edges))
        for values in [{}, {("a","jpeg",0.):0.,("a","jpeg",2.):float("inf")},
                       {("a","jpeg",0.):0.,("a","jpeg",2.):0.}]:
            self.assertEqual(owner.direction_flag(edges[0],values),0)

    def test_frozen_tier_requires_both_guards_and_minimum_ladders(self):
        self.assertEqual(owner.direction_verdict(9,0.,0.),"INSUFFICIENT")
        self.assertEqual(owner.direction_verdict(10,0.,-.05),"PASS")
        self.assertEqual(owner.direction_verdict(10,-.001,0.),"FAIL")
        self.assertEqual(owner.direction_verdict(10,.01,-.051),"FAIL")

    def test_legacy_avif_gate_driver_is_refused_before_any_encode(self):
        import shipgates_codec_rd
        with self.assertRaisesRegex(ValueError,"SVT map/target adapter"):
            shipgates_codec_rd.require_supported_gate_adapter("avif")
        shipgates_codec_rd.require_supported_gate_adapter("jpeg")
        shipgates_codec_rd.require_supported_gate_adapter("webp")

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
