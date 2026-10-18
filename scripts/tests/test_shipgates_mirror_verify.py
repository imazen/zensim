"""A completed evidence archive refuses missing, changed and extra files."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("mirror", Path(__file__).parents[1]/"rev4_featpot/shipgates_mirror_verify.py")
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


class CompleteArchive(unittest.TestCase):
    def test_model_readback_is_explicit_and_preserves_encoded_default(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as tmp:
            root=Path(tmp)
            for name in ['image.jpg','map.data.png','bake.bin','notes.json']:
                (root/name).write_bytes(b'payload')
            self.assertEqual({p.name for p in owner.payload_candidates(root)}, {'image.jpg','map.data.png'})
            self.assertEqual({p.name for p in owner.payload_candidates(root,'model')}, {'bake.bin'})
            with self.assertRaises(ValueError):owner.payload_candidates(root,'unknown')

    def fixture(self, root):
        source=root/"source";destination=root/"destination"
        source.mkdir();(source/"COMPLETE.json").write_text("{}\n")
        (source/"encoded.jpg").write_bytes(bytes(range(255)))
        shutil.copytree(source,destination)
        return source,destination

    def test_complete_tree_and_receipt_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination=self.fixture(Path(tmp))
            result=owner.verify_all(source,destination,Path("COMPLETE.json"))
            self.assertEqual(result["file_count"],2)
            self.assertEqual(result["total_bytes"],258)

    def test_self_archive_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,_=self.fixture(Path(tmp))
            with self.assertRaisesRegex(AssertionError,"distinct and disjoint"):
                owner.verify_all(source,source,Path("COMPLETE.json"))

    def test_changed_payload_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination=self.fixture(Path(tmp));(destination/"encoded.jpg").write_bytes(b"changed")
            with self.assertRaises(AssertionError):
                owner.verify_all(source,destination,Path("COMPLETE.json"))

    def test_missing_completion_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination=self.fixture(Path(tmp));(source/"COMPLETE.json").unlink()
            with self.assertRaisesRegex(AssertionError,"completion marker missing"):
                owner.verify_all(source,destination,Path("COMPLETE.json"))

    def test_extra_destination_file_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination=self.fixture(Path(tmp));(destination/"extra").write_bytes(b"unreviewed")
            with self.assertRaisesRegex(AssertionError,"population differs"):
                owner.verify_all(source,destination,Path("COMPLETE.json"))


if __name__ == "__main__":
    unittest.main()
