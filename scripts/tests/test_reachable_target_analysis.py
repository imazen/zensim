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


if __name__ == "__main__":
    unittest.main()
