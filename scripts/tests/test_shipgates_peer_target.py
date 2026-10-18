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


class ApprovedAmendments(unittest.TestCase):
    def comparison(self):
        def summary(dial):
            panels=[]
            for codec in ("jpeg","webp","avif"):
                for model in ("diff-frag-820","w-c"):
                    panel=dict(codec=codec,model=model,model_sha256=model)
                    for policy in ("train_curve","midpoint"):
                        panel[policy]=[dict(budget=k,n={"zensim":10,"ssim2":100,"butter":10}[dial],
                            median_abs_error={"zensim":2.,"ssim2":1.,"butter":3.}[dial],
                            p95_abs_error=4.,worst_abs_error={"zensim":7.,"ssim2":6.,"butter":7.}[dial],
                            undershoot_counts={"8":2,"3":2,"1":2}) for k in (1,2,3)]
                    panels.append(panel)
            return dict(panels=panels)
        return dict(candidate=summary("zensim"),peers={d:summary(d) for d in ("ssim2","butter")})

    def test_worse_peer_equality_and_only_registered_statistics(self):
        cells=peer.relative_cells(self.comparison())
        self.assertEqual(len(cells),120)
        self.assertTrue(all(r["peer_verdict"]=="PASS" for r in cells))
        self.assertTrue(all(r["budget"]==3 for r in cells if r["statistic"]=="max"))
        # Equal counts are unequal fractions; compare the registered fraction.
        under=next(r for r in cells if r["statistic"]=="undershoot_fraction")
        self.assertEqual(under["values"],{"zensim":.2,"ssim2":.02,"butter":.2})

    def test_candidate_above_worse_peer_fails_and_missing_panel_refuses(self):
        comparison=self.comparison()
        comparison["candidate"]["panels"][0]["train_curve"][0]["median_abs_error"]=4.
        cells=peer.relative_cells(comparison)
        self.assertEqual(sum(r["peer_verdict"]=="FAIL" for r in cells),1)
        comparison["peers"]["butter"]["panels"].pop()
        with self.assertRaises(AssertionError):peer.relative_cells(comparison)

    def test_base_only_uses_base_anchor_strict_order_and_pixel_dedup(self):
        import json
        anchor=dict(index=0,role="validate",origin="source",kind="honest_anchor",label=0,distorted="anchor-q20.png")
        positive=dict(index=1,role="validate",origin="source",kind="corruption",label=1,
                      distorted="dist.png",source_table="generator",source_row_id="source:1",family="real_bug",content_class="photo")
        duplicate=dict(positive,index=2)
        def audit(score,pixels):return dict(base_score=score,reference_pixels_sha256="ref",distorted_pixels_sha256=pixels,pixel_composed_score=0.)
        audits={0:audit(20.,"anchor"),1:audit(20.,"dist"),2:audit(20.,"dist")}
        metadata={("generator","source:1"):dict(params_json=json.dumps(dict(expected_below_lq=True,params=dict(family=dict(family="real_bug",ac=None),region="whole",severity="opaque"))))}
        rows=peer.base_only_rows([anchor,positive,duplicate],audits,metadata)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["q20_base_score"],20.)
        self.assertFalse(rows[0]["below_q20"])
        audits[1]["base_score"]=19.;audits[2]["base_score"]=19.
        self.assertTrue(peer.base_only_rows([anchor,positive,duplicate],audits,metadata)[0]["below_q20"])
        audits.pop(2)
        with self.assertRaises(AssertionError):peer.base_only_rows([anchor,positive,duplicate],audits,metadata)


if __name__=="__main__":
    unittest.main()
