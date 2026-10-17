"""Refuse measurement changes while applying an approved class annotation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
from shipgates_source_revision import reviewed_record, sha


class RevisionTests(unittest.TestCase):
    def test_class_annotation_retains_original_measurements_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp", prefix="shipgates-class-") as name:
            root = Path(name)
            manifest = root / "SOURCE_MANIFEST.json"
            manifest.write_text('{"sources":[]}\n')
            source = dict(name="gb82-mc1-lossless", sha256="frozen-test-source", content_class="screen")
            record = dict(source=source["name"], source_sha256=source["sha256"], content_class="photo",
                          case=str(root), bytes=731, encoded_sha256="unchanged-encoded-test-hash",
                          map_records=[dict(sha256="unchanged-map-test-hash")])
            (root/"CELL.json").write_text(json.dumps(record))
            review = dict(schema="shipgates-source-class-revision-v1",
                          coordinator_approval="Approve correction and add 3 photos",
                          revised_manifest_sha256=sha(manifest), changes=[dict(name=source["name"],
                          sha256=source["sha256"], **{"from":"photo","to":"screen"})])
            review_path = root/"CONTENT_CLASS_REVISION.json"
            review_path.write_text(json.dumps(review))
            revised = reviewed_record(record, source, root)
            self.assertEqual(revised["content_class"], "screen")
            self.assertEqual({k:v for k,v in revised.items() if k not in ["content_class","class_derivation"]},
                             {k:v for k,v in record.items() if k != "content_class"})
            self.assertEqual(json.loads((root/"CELL.json").read_text()), record)
            with self.assertRaisesRegex(AssertionError, "original cell metadata differs"):
                reviewed_record(dict(record, bytes=730), source, root)
            review["changes"][0]["sha256"]="a-different-source"
            review_path.write_text(json.dumps(review))
            with self.assertRaises(AssertionError):
                reviewed_record(record, source, root)


if __name__ == "__main__":
    unittest.main()
