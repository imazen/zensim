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

class JudgeBindingTests(unittest.TestCase):
    def test_rejects_aggregate_changes_and_wrong_fixed_profile(self):
        with tempfile.TemporaryDirectory(prefix="shipgates-judge-binding-") as directory:
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


if __name__ == "__main__":
    unittest.main()
