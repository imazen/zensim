"""The target summary must refuse missing cells and unwitnessed requests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

OWNER = Path(__file__).parents[1] / "v_next/rd_probe_analyze_2026-07-18.py"
SPEC = importlib.util.spec_from_file_location("rd_probe_analyze", OWNER)
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


class ReachableTargetAnalysis(unittest.TestCase):
    def fixture(self, root):
        inp = {"schema":"reachable-target-v1", "fit_calibration":False,
               "source_manifest":{"sources":[{}]},"codecs":["jpeg"],"models":["B"],
               "budgets":[1],"policies":["midpoint","train_curve"],
               "bound_steps":3,"tolerance":1.}
        bound = {"id":"b0","source":0,"codec":"jpeg","model":"B",
                 "attained_min":-10.,"attained_max":30.,"steering_targets":[-10.,30.],
                 "probes":[{"score":s} for s in (-10.,10.,30.)],"requests":[]}
        rows = []
        for target in bound["steering_targets"]:
            for policy in inp["policies"]:
                rows.append({"bound_id":"b0","source":0,"codec":"jpeg","model":"B",
                             "origin":"1003","family":"1003","content_class":"photo",
                             "target":target,"target_status":"witnessed","achieved":target,
                             "error":0.,"pass_budget":1,"passes":1,"policy":policy,
                             "probes":[{}],"loop_seconds":.1,"ssim2":80.,
                             "butteraugli_pnorm3":1.,"bytes":123})
        (root/"INPUTS.json").write_text(json.dumps(inp))
        (root/"COMPLETE").write_text(json.dumps({"measurements":len(rows)}))
        (root/"bounds.jsonl").write_text(json.dumps(bound)+"\n")
        (root/"measurements.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
        return inp,bound,rows

    def test_complete_negative_target_matrix_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); inp,_,rows=self.fixture(root)
            owner.reachable_target_summary(root,inp,rows)
            result=json.loads((root/"analysis_summary.json").read_text())
            self.assertEqual(result["cells"],4)
            self.assertTrue(result["qualification"].startswith("unqualified"))

    def test_missing_cell_cannot_be_a_complete_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp,_,rows=self.fixture(root)
            with self.assertRaisesRegex(SystemExit,"steering cells"):
                owner.reachable_target_summary(root,inp,rows[:-1])

    def test_duplicate_cannot_replace_a_missing_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp,_,rows=self.fixture(root)
            rows[-1]=rows[0]
            with self.assertRaisesRegex(SystemExit,"steering cells"):
                owner.reachable_target_summary(root,inp,rows)

    def test_target_inside_envelope_still_needs_a_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp,bound,rows=self.fixture(root)
            bound["steering_targets"][0]=0.
            for r in rows:
                if r["target"] == -10.: r["target"]=r["achieved"]=0.
            (root/"bounds.jsonl").write_text(json.dumps(bound)+"\n")
            with self.assertRaisesRegex(SystemExit,"unwitnessed"):
                owner.reachable_target_summary(root,inp,rows)


class NativeScalarTargetAnalysis(unittest.TestCase):
    """New scalar scope retains the native coverage, byte and budget guards."""
    def fixture(self, root):
        import hashlib
        from PIL import Image
        digest = lambda b: hashlib.sha256(b).hexdigest()
        pixels = bytes([32,64,96])
        rgb_sha = digest(pixels)
        inp = {"schema":"native-codec-target-scalar-v1", "arms":["scalar"],
               "sources":{"sources":[{"origin":"1","family":"family1","content_class":"photo","path":"reference.png"}]},
               "codec":"jpg","quality_knots":[0.,100.],"bound_encodes":{"scalar":1},
               "tolerance":1.,"fixed_requests":[],"policies":["midpoint","train_curve"],"budgets":[1,2,3]}
        def work(knob, encoded):
            return dict(knob=knob,internal_reconstructions=0,native_pixel_comparisons=0,
                        map_evaluations=0,consumed_maps=0,encoded_sha256=digest(encoded),decoded_sha256=rgb_sha)
        bounds=[]
        for i,(q,score) in enumerate([(0.,10.),(100.,90.)]):
            encoded=bytes([i])
            (root/f"1-scalar-{i}.jpg").write_bytes(encoded)
            Image.frombytes("RGB",(1,1),pixels).save(root/f"1-scalar-{i}.png")
            bounds.append(dict(bound=dict(origin="1",arm="scalar",knob=q,score=score,seed_score=score,
                               bytes=1,encoded_sha256=digest(encoded),decoded_sha256=rgb_sha),
                               work=[work(q,encoded)],bound_probe_scores=[score]))
        rows=[]
        for q,score in [(0.,10.),(100.,90.)]:
            for policy in inp["policies"]:
                for budget in inp["budgets"]:
                    index=len(rows);encoded=bytes([index]);file=f"case-{index}.jpg";decoded=f"case-{index}.png"
                    (root/file).write_bytes(encoded)
                    Image.frombytes("RGB",(1,1),pixels).save(root/decoded)
                    rows.append(dict(origin="1",family="family1",**{"class":"photo"},arm="scalar",policy=policy,budget=budget,
                         target=score,achieved=score,signed_error=0.,bytes=1,full_encodes=1,
                         search_pixel_comparisons=1,terminal_decodes=1,terminal_pixel_comparisons=1,total_seconds=.1,
                         process_peak_rss_kib=1,probes=[dict(knob=q,score=score,bytes=1)],native_work=[work(q,encoded)],
                         bitstream=file,decoded=decoded,encoded_sha256=digest(encoded)))
        (root/"INPUTS.json").write_text(json.dumps(inp))
        (root/"COMPLETE").write_text(json.dumps(dict(cases=len(rows),bounds=len(bounds))))
        (root/"bounds.jsonl").write_text("\n".join(map(json.dumps,bounds))+"\n")
        (root/"measurements.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
        (root/"coverage.json").write_text(json.dumps([dict(origin="1",target=t,scalar_witnessed=True) for t in (10.,90.)]))
        return inp,rows

    def test_complete_scalar_scope_keeps_numeric_gates_and_disclaims_maps(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);inp,rows=self.fixture(root)
            owner.native_target_summary(root,inp,rows,True)
            result=json.loads((root/"analysis_summary.json").read_text())
            self.assertEqual(len(result["summary"]),6)
            self.assertTrue(all(s["registered_gate"]["verdict"]=="PASS" for s in result["summary"]))
            self.assertIn("active/neutral-map targeting UNMEASURED",result["qualification"])
            self.assertEqual(result["paired"],[])

    def test_missing_scalar_cell_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp,rows=self.fixture(root)
            with self.assertRaisesRegex(SystemExit,"steering cells"):
                owner.native_target_summary(root,inp,rows[:-1],True)

    def test_spatial_arm_cannot_masquerade_as_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp,rows=self.fixture(root);inp["arms"]=["active"]
            with self.assertRaisesRegex(SystemExit,"scalar-only scope"):
                owner.native_target_summary(root,inp,rows,True)

    def test_changed_encoded_bytes_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp,rows=self.fixture(root);(root/"case-0.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(SystemExit,"emitted bytes"):
                owner.native_target_summary(root,inp,rows,True)

    def test_third_shot_undershoot_is_still_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);inp,rows=self.fixture(root)
            r=next(r for r in rows if r["budget"]==3 and r["policy"]=="train_curve")
            r["achieved"]-=1.5;r["signed_error"]=-1.5;r["probes"][0]["score"]=r["achieved"]
            owner.native_target_summary(root,inp,rows,True)
            result=json.loads((root/"analysis_summary.json").read_text())
            row=next(s for s in result["summary"] if s["budget"]==3 and s["policy"]=="train_curve")
            self.assertEqual(row["registered_gate"]["verdict"],"FAIL")
            self.assertFalse(row["registered_gate"]["checks"]["undershoot"])


if __name__ == "__main__":
    unittest.main()
