"""Byte preservation and refusal-before-output for the bounded Rev5 admission view."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import v2c_wide as owner


class TeacherAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rev5-teacher-test-")
        self.root = Path(self.tmp.name)
        self.bank, self.source, self.out = (self.root / p for p in ("bank", "source", "out"))
        self.wide = self.source / "wide/main/real"
        self.wide.mkdir(parents=True)
        self.original_profile = owner.PROFILE
        owner.PROFILE = owner.rev5_profile("rev5_localwin", "basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
                                           "a" * 64, "b" * 40, 1853)
        p = owner.PROFILE
        self.receipt = {"schema": owner.SCHEMA, "family": "main", "variant": "real", "formula_revision": 5,
                        "feature_set_id": p.feature_set_id, "era": p.era, "extras": [], "bank": {}, "legs": {}}
        for teacher, (name, _, _) in owner.TEACHERS.items():
            directory = self.bank / name
            directory.mkdir(parents=True)
            for filename in ("features.parquet", "keys.parquet"):
                (directory / filename).write_bytes(b"immutable fixture bytes")
            manifest = {"set": name, "schema": p.schema, "feature_width": owner.WIDTH,
                        "formula_revision": "Rev5", "era_label": p.era, "feature_set_id": p.feature_set_id,
                        "binary_sha256": p.binary, "build_commit": p.build, "dtype": "float64",
                        "requested_slot_ranges": [list(r) for r in p.requested], "input_contract": "legacy-rgb8",
                        "chunks": [{"extractor_manifest": {"producer_binary_sha256": p.binary,
                                    "feature_set_id": p.feature_set_id, "formula_revision": "5",
                                    "populated_feature_ids": [i for lo, hi in p.requested for i in range(lo, hi)]}}]}
            self.write_json(directory / "_MANIFEST.json", manifest)
            self.receipt["bank"][name] = {"manifest_sha256": owner.sha(directory / "_MANIFEST.json"),
                                          "features_sha256": owner.sha(directory / "features.parquet"),
                                          "keys_sha256": owner.sha(directory / "keys.parquet")}
            self.receipt["legs"][teacher] = {}
            for split in ("fit", "dev"):
                path = self.wide / f"{teacher}_{split}.parquet"
                path.write_bytes(f"fixture rows {teacher} {split}".encode())
                sidecar = Path(f"{path}.manifest.json")
                self.write_json(sidecar, {"source_bank_feature_set_id": p.feature_set_id, "formula_revision": 5})
                self.receipt["legs"][teacher][split] = {"rel": str(path.relative_to(self.source)),
                    "sha256": owner.sha(path), "manifest_sha256": owner.sha(sidecar), "rows": 3, "references": 1}
        self.freeze()

    def tearDown(self):
        owner.PROFILE = self.original_profile
        self.tmp.cleanup()

    @staticmethod
    def write_json(path, record):
        path.write_text(json.dumps(record))

    def freeze(self):
        path = self.wide / "receipt.json"
        self.write_json(path, self.receipt)
        self.write_json(self.source / "wide/frozen.json", {"wide_receipts": {"main/real": owner.sha(path)}})

    def admit(self):
        owner.admit_teachers(self.bank, self.source, self.out)

    def refused(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.admit()
        self.assertFalse(self.out.exists())

    def test_copy_keeps_exact_bytes_and_declares_producing_executable(self):
        self.admit()
        for path in self.wide.glob("*.parquet"):
            dest = self.out / path.name
            self.assertEqual(dest.read_bytes(), path.read_bytes())
            declaration = json.loads(Path(f"{dest}.manifest.json").read_text())
            self.assertEqual(declaration["feature_set_id"], owner.PROFILE.feature_set_id)
            self.assertEqual(declaration["decoder_era"], "legacy-rgb8/extract_features_372col@sha256:" + "a" * 64)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.admit()

    def test_changed_table_refused_before_output(self):
        (self.wide / "safesyn_fit.parquet").write_bytes(b"changed")
        self.refused("table/sidecar")

    def test_changed_bank_refused_before_output(self):
        (self.bank / "safesyn/features.parquet").write_bytes(b"changed")
        self.refused("bank receipt")

    def test_changed_receipt_refused_before_output(self):
        self.receipt["variant"] = "p1"
        self.write_json(self.wide / "receipt.json", self.receipt)
        self.refused("frozen real")

    def test_receipt_cannot_admit_a_human_table(self):
        self.receipt["legs"]["safesyn"]["fit"]["rel"] = "wide/main/real/human_all_fit.parquet"
        self.freeze()
        self.refused("permitted real TRAIN")

    def test_chunk_cannot_impersonate_pinned_decoder(self):
        path = self.bank / "safesyn/_MANIFEST.json"
        manifest = json.loads(path.read_text())
        manifest["chunks"][0]["extractor_manifest"]["producer_binary_sha256"] = "c" * 64
        self.write_json(path, manifest)
        self.receipt["bank"]["safesyn"]["manifest_sha256"] = owner.sha(path)
        self.freeze()
        self.refused("chunk producer")

    def test_sealed_path_refused_before_output(self):
        with self.assertRaises(PermissionError):
            owner.admit_teachers(self.bank, self.root / "_sealed", self.out)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
