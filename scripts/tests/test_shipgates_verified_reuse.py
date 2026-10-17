"""Reject altered score packets and maps when reusing successful certificates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from shipgates_verified_reuse import restore, sha, snapshot


class VerifiedReuseTests(unittest.TestCase):
    def test_reuse_refuses_changed_packet_or_physical_map(self):
        scratch=Path.home()/"tmp"
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix="shipgates-certified-") as name:
            root=Path(name)
            packet_root=root/"packet"
            packet_root.mkdir()
            def write(path,value):
                path.write_text(json.dumps(value))
            physical=root/"map.bin"
            physical.write_bytes(b"test-only unchanged map bytes")
            packet=packet_root/"judges.jsonl"
            packet.write_text(json.dumps(dict(maps=[dict(path=str(physical),map_sha256=sha(physical))]))+"\n")
            write(root/"source-00-VERIFIED.json",dict(path=str(packet_root),sha256=sha(packet),pairs=1))
            write(root/"INPUTS.json",dict(judge_binary_sha256="frozen-test-judge",ssim2_map_policy="scalar-only"))
            oracle=dict(oracle_source_sha256="frozen-test-oracle",judges_jsonl_sha256=sha(packet),judge_rows=1)
            for field in ["all_physical_hashes_verified","all_logical_hashes_verified","all_float_values_finite",
                          "all_padding_zero","all_input_pixel_hashes_verified","all_map_dedup_keys_verified"]:
                oracle[field]=True
            write(root/"source-00-MAP_ORACLE.json",oracle)
            snapshot([root],1,root/"SNAPSHOT.json")
            entry=json.loads((root/"SNAPSHOT.json").read_text())["sources"][0]
            restore(entry,packet_root,"frozen-test-judge","frozen-test-oracle",root/"copied.json",1)
            self.assertEqual((root/"copied.json").read_bytes(),(root/"source-00-MAP_ORACLE.json").read_bytes())
            physical.write_bytes(b"changed test-only bytes")
            with self.assertRaisesRegex(AssertionError,"changed certified map"):
                restore(entry,packet_root,"frozen-test-judge","frozen-test-oracle",root/"bad-map.json",1)
            packet.write_text(packet.read_text()+"\n")
            with self.assertRaises(AssertionError):
                restore(entry,packet_root,"frozen-test-judge","frozen-test-oracle",root/"bad-packet.json",1)
            self.assertFalse((root/"bad-map.json").exists())
            self.assertFalse((root/"bad-packet.json").exists())


if __name__=="__main__":
    unittest.main()
