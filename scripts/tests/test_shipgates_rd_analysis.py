"""Numerical identity and strong-baseline regressions for RD interpolation."""
import importlib.util
import hashlib
import json
import tempfile
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

    def test_matched_scalar_blocks_apparent_global_grid_gain(self):
        # A target controller can land below an interpolated global grid even
        # when steering emits exactly its scalar controller's bytes.
        grid=owner.rd_frontier([(0.,1000),(50.,2000),(100.,4000)])
        scalar=owner.rd_frontier([(50.,1800),(70.,2500)])
        active=scalar.copy()
        apparent=1-owner.interp_bytes(active,50.)/owner.interp_bytes(grid,50.)
        matched=1-owner.interp_bytes(active,50.)/owner.interp_bytes(scalar,50.)
        self.assertGreater(apparent,.01)
        self.assertEqual(matched,0.)
        summaries=[]
        for metric in ["ssim2","butter","zensim"]:
            for baseline,saving in [("none",apparent),("candidate-scalar",matched)]:
                for cohort in ["all","photo","screen"]:
                    summaries.append(dict(arm="candidate",baseline=baseline,metric=metric,
                        content_class=cohort,saving=saving,sources=24 if cohort=="all" else 12))
        native=owner.shipgates_numeric_gates(["candidate"],summaries,["photo","screen"],True,True)
        self.assertEqual(native["candidate"]["baseline"],"candidate-scalar")
        self.assertEqual(native["candidate"]["verdict"],"FAIL")
        self.assertFalse(native["candidate"]["at_least_one_judge_saves_one_percent"])
        global_grid=owner.shipgates_numeric_gates(["candidate"],summaries,["photo","screen"],True,False)
        self.assertEqual(global_grid["candidate"]["baseline"],"none")
        self.assertEqual(global_grid["candidate"]["verdict"],"PASS")

class JudgeBindingTests(unittest.TestCase):
    def test_rejects_aggregate_changes_and_wrong_fixed_profile(self):
        scratch = Path.home()/"tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="shipgates-judge-binding-",dir=scratch) as directory:
            root = Path(directory)
            def write(name, data):
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data if isinstance(data,bytes) else json.dumps(data).encode())
                return path
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            binary=write("binary",b"test-only judge identity")
            pin=write("PIN.json",dict(binary=str(binary),binary_sha256=digest(binary)))
            design=write("DESIGN.json",dict(schema="test-only"))
            oracle_source=write("map_oracle.py",b"test-only oracle identity")
            write("ORACLE_PIN.json",dict(sha256=digest(oracle_source)))
            write("INPUTS.json",dict(judge_binary_sha256=digest(binary),design_sha256=digest(design),
                                    sources=1,ssim2_map_policy="scalar-only"))
            packet=write("packet/judges.jsonl",b'{"test_only":true}\n')
            metadata=dict(driver_sha256=digest(binary),fixed_zensim_profile="B",
                          formula_revision_environment="1",requested_metrics=["ssim2","butter","zensim"],
                          ssim2_map_policy="scalar-only")
            write("packet/INPUTS.json",metadata)
            write("source-00-VERIFIED.json",dict(path=str(packet.parent),sha256=digest(packet),pairs=1))
            oracle=dict(oracle_source_sha256=digest(oracle_source),judges_jsonl_sha256=digest(packet),judge_rows=1)
            for key in ["all_physical_hashes_verified","all_logical_hashes_verified","all_float_values_finite",
                        "all_padding_zero","all_input_pixel_hashes_verified","all_map_dedup_keys_verified"]:
                oracle[key]=True
            write("source-00-MAP_ORACLE.json",oracle)
            write("judges.jsonl",packet.read_bytes())
            owner.validate_shipgates_judge_binding(root,design,pin,1)
            write("judges.jsonl",packet.read_bytes()+b"\n")
            with self.assertRaisesRegex(AssertionError,"aggregate judge rows differ"):
                owner.validate_shipgates_judge_binding(root,design,pin,1)
            write("judges.jsonl",packet.read_bytes())
            metadata["fixed_zensim_profile"]="other scorer"
            write("packet/INPUTS.json",metadata)
            with self.assertRaises(AssertionError):
                owner.validate_shipgates_judge_binding(root,design,pin,1)


def _r6_summaries(model="cand", overall=None, classes=None, butter_self=.02):
    """Invented summary rows: per (arm,baseline,metric,cohort) saving + sources."""
    rows = []
    metrics = ["ssim2", "butter", "zensim", "cvvdp"]
    for metric in metrics:
        rows.append(dict(arm=model, baseline="none", metric=metric,
                         content_class="all",
                         saving=(overall or {}).get(metric, .02), sources=27))
        for cls in ["photo", "screen"]:
            rows.append(dict(arm=model, baseline="none", metric=metric,
                             content_class=cls,
                             saving=(classes or {}).get((metric, cls), .02),
                             sources=12))
    rows.append(dict(arm="butteraugli", baseline="none", metric="butter",
                     content_class="all", saving=butter_self, sources=27))
    return rows


def _r6_deltas(model="cand", lo_overall=0., lo_class=0., metrics=("ssim2", "cvvdp"),
               classes=("photo", "screen")):
    out = {}
    for metric in metrics:
        out[(metric, model, "all")] = dict(estimate=lo_overall, lo=lo_overall,
                                           hi=lo_overall, sources=27, seed=1)
        for cls in classes:
            out[(metric, model, cls)] = dict(estimate=lo_class, lo=lo_class,
                                             hi=lo_class, sources=12, seed=1)
    return out


class R6PrimeGateTests(unittest.TestCase):
    METRICS = {"ssim2", "cvvdp"}
    CLASSES = ["photo", "screen"]

    def gate(self, **kw):
        summaries = kw.pop("summaries", _r6_summaries())
        deltas = kw.pop("deltas", _r6_deltas())
        return owner.shipgates_r6_prime_gates(
            ["cand"], summaries, deltas, self.METRICS, self.CLASSES, True)["cand"]

    def test_pass_baseline(self):
        g = self.gate()
        self.assertEqual(g["verdict"], "PASS")
        self.assertTrue(g["all_judges_nonnegative"])
        self.assertTrue(g["at_least_one_judge_saves_one_percent"])
        self.assertTrue(g["no_neutral_judge_class_regression"])
        self.assertTrue(g["self_judge_symmetry"]["holds"])

    def test_negative_saving_on_any_judge_fails(self):
        # savings >= 0 is required on every analyzed judge, not only neutral ones.
        g = self.gate(summaries=_r6_summaries(overall={"butter": -.001}))
        self.assertEqual(g["verdict"], "FAIL")
        self.assertFalse(g["all_judges_nonnegative"])

    def test_one_percent_must_hold_on_at_least_one_judge(self):
        g = self.gate(summaries=_r6_summaries(
            overall={"ssim2": .005, "butter": .005, "zensim": .005, "cvvdp": .005}))
        self.assertEqual(g["verdict"], "FAIL")
        self.assertFalse(g["at_least_one_judge_saves_one_percent"])

    def test_class_rule_applies_only_to_neutral_judges(self):
        # A negative class saving on butteraugli/zensim does not violate R6'(i);
        # the same value on ssim2 or cvvdp does.
        ok = self.gate(summaries=_r6_summaries(
            classes={("butter", "photo"): -.05, ("zensim", "screen"): -.05}))
        self.assertEqual(ok["verdict"], "PASS")
        bad = self.gate(summaries=_r6_summaries(classes={("cvvdp", "photo"): -.001}))
        self.assertEqual(bad["verdict"], "FAIL")
        self.assertFalse(bad["no_neutral_judge_class_regression"])
        bad2 = self.gate(summaries=_r6_summaries(classes={("ssim2", "screen"): -.001}))
        self.assertEqual(bad2["verdict"], "FAIL")

    def test_noninferiority_overall_bound(self):
        ok = self.gate(deltas=_r6_deltas(lo_overall=-.0099))
        self.assertEqual(ok["verdict"], "PASS")
        bad = self.gate(deltas=_r6_deltas(lo_overall=-.0101))
        self.assertEqual(bad["verdict"], "FAIL")

    def test_noninferiority_per_class_bound(self):
        ok = self.gate(deltas=_r6_deltas(lo_class=-.0199))
        self.assertEqual(ok["verdict"], "PASS")
        bad = self.gate(deltas=_r6_deltas(lo_class=-.0201))
        self.assertEqual(bad["verdict"], "FAIL")

    def test_self_judge_symmetry(self):
        ok = self.gate(summaries=_r6_summaries(
            overall={"zensim": .03}, butter_self=.02))
        self.assertEqual(ok["verdict"], "PASS")
        bad = self.gate(summaries=_r6_summaries(
            overall={"zensim": .01}, butter_self=.02))
        self.assertEqual(bad["verdict"], "FAIL")
        self.assertFalse(bad["self_judge_symmetry"]["holds"])

    def test_missing_delta_interval_is_incomplete(self):
        deltas = _r6_deltas()
        del deltas[("ssim2", "cand", "photo")]
        g = self.gate(deltas=deltas)
        self.assertEqual(g["verdict"], "INCOMPLETE")

    def test_thin_class_is_incomplete(self):
        rows = _r6_summaries()
        for r in rows:
            if r["content_class"] == "photo":
                r["sources"] = 11
        g = self.gate(summaries=rows)
        self.assertEqual(g["verdict"], "INCOMPLETE")
        self.assertFalse(g["minimum12_source_families_per_class"])

    def test_unbound_cvvdp_is_incomplete_not_silent(self):
        g = owner.shipgates_r6_prime_gates(
            ["cand"], _r6_summaries(), _r6_deltas(metrics=("ssim2",)),
            {"ssim2"}, self.CLASSES, False)["cand"]
        self.assertEqual(g["verdict"], "INCOMPLETE")
        self.assertFalse(g["cvvdp_bound"])


if __name__ == "__main__":
    unittest.main()
