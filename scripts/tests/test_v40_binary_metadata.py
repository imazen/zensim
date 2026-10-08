"""Regression for stale R2 binary_mix copied into a new packet."""

import copy
import tempfile
import unittest
from pathlib import Path

from v40_binary_metadata import digest, validate


class BinaryMetadata(unittest.TestCase):
    def test_current_producers_and_all_hashes_agree(self):
        with tempfile.TemporaryDirectory() as root:
            b = Path(root)
            (b / "bin").mkdir()
            (b / "BUILD_LOG.txt").write_text("actual build record")
            names = (
                "zensim_mlp_train",
                "bake_dial_refit",
                "panel",
                "predict_features_with_bake",
                "inspect_qualified_checkpoint",
            )
            for name in names:
                (b / "bin" / name).write_text(name)
            producer = "a" * 40
            bindings = dict(
                binary_producer_commit=producer,
                binaries={
                    name: dict(
                        producer_commit=producer, sha256=digest(b / "bin" / name)
                    )
                    for name in names
                },
            )
            metadata = dict(
                build_commit=producer,
                trainer_producer_commit=producer,
                zensim_source_commit=producer,
                binary_mix={
                    name: dict(
                        build_commit=producer,
                        sha256=digest(b / "bin" / name),
                        producer_record="BUILD_LOG.txt",
                        producer_record_sha256=digest(b / "BUILD_LOG.txt"),
                    )
                    for name in names
                },
            )
            archive = {
                "bin/" + n: digest(b / "bin" / n)
                for n in names
                if n != "predict_features_with_bake"
            }
            validate(b, metadata, bindings, archive)
            for name in names:
                for field, value in (
                    ("build_commit", "b" * 40),
                    ("sha256", "0" * 64),
                    ("producer_record", "old/build.log"),
                    ("producer_record_sha256", "0" * 64),
                ):
                    with self.subTest(name=name, field=field):
                        bad = copy.deepcopy(metadata)
                        bad["binary_mix"][name][field] = value
                        with self.assertRaises(AssertionError):
                            validate(b, bad, bindings, archive)
            bad = copy.deepcopy(metadata)
            bad["build_commit"] = "b" * 40
            with self.assertRaises(AssertionError):
                validate(b, bad, bindings, archive)
            changed_archive = dict(archive)
            changed_archive["bin/zensim_mlp_train"] = "0" * 64
            with self.assertRaises(AssertionError):
                validate(b, metadata, bindings, changed_archive)
            (b / "BUILD_LOG.txt").write_text("retargeted build record")
            with self.assertRaises(AssertionError):
                validate(b, metadata, bindings, archive)
