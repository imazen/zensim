"""Invented native evidence/bit-parity fixtures; no models, workers or labels."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "rev4_featpot"))
from e36_phase_handoff import admit_installed, projection, restore_inventory, sha


class NativeHandoff(unittest.TestCase):
    def test_installed_proof_replays_original_output_bytes_and_timestamps(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-installed-"
        ) as d:
            root = Path(d)

            def put(name, value):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value))
                return dict(path=str(p), sha256=sha(p))

            clock = put("clock.json", {})
            phase = dict(run="additions-v6", manifest_sha256="manifest")
            files = put(
                "files.json",
                dict(
                    root=str(root),
                    files=[
                        dict(
                            destination=str(root / "bin/zenfleet-worker"),
                            sha256="worker",
                        )
                    ],
                ),
            )
            binding = put(
                "container.json",
                dict(
                    phase="additions",
                    manifest=dict(sha256="manifest"),
                    effective_clock=clock,
                    files_manifest_path=files["path"],
                    files_manifest_sha256=files["sha256"],
                ),
            )
            first = put("first.json", dict(files_manifest_sha256=files["sha256"]))
            parity = put(
                "parity.json", dict(admitted=True, views=5, native_execution=first)
            )
            guard = put("guard.json", {})
            adapter = put("adapter.py", {})
            handoff = put(
                "TOWER_PHASE_HANDOFF.json",
                dict(
                    schema="e36-whole-native-phase-owner-v1",
                    run=phase["run"],
                    container=binding,
                    parity=parity,
                    live_guard=guard,
                    adapter=adapter,
                ),
            )
            blob = root / "runs/additions-v6/blobs/output"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"native invented output")
            stamp = 1792247416000000123
            os.utime(blob, ns=(stamp, stamp))
            execution = put(
                "runs/additions-v6/phase-execution.json",
                dict(
                    files_manifest_sha256=files["sha256"],
                    effective_clock_sha256=clock["sha256"],
                    manifest_sha256="manifest",
                    returncode=0,
                    fits=0,
                    optimizer_steps=0,
                    native_worker_sha256="worker",
                    native_files=[
                        dict(
                            relative="blobs/output",
                            sha256=sha(blob),
                            bytes=blob.stat().st_size,
                            mtime_ns=stamp,
                        )
                    ],
                ),
            )
            harvest = put("tower-additions-harvest-v1/harvest.json", {})
            put(
                "tower-additions-installed.json",
                dict(
                    handoff_sha256=handoff["sha256"],
                    native_execution_sha256=execution["sha256"],
                    harvest_sha256=harvest["sha256"],
                    output_source_timestamps_preserved=True,
                ),
            )
            evidence = admit_installed(
                root, Path(clock["path"]), dict(phases=dict(additions=phase))
            )
            self.assertTrue(any(e["path"] == execution["path"] for e in evidence))
            os.utime(blob, ns=(stamp + 1, stamp + 1))
            with self.assertRaisesRegex(ValueError, "source timestamps changed"):
                admit_installed(
                    root, Path(clock["path"]), dict(phases=dict(additions=phase))
                )

    def test_native_source_timestamps_restore_exactly_and_tamper_refuses(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-native-harvest-"
        ) as d:
            root = Path(d)
            blob = root / "blobs/output"
            blob.parent.mkdir()
            blob.write_bytes(b"invented native evidence")
            (root / "phase-execution.json").write_text("{}")
            stamp = 1792247416000000123
            receipt = dict(
                native_files=[
                    dict(
                        relative="blobs/output",
                        sha256=sha(blob),
                        bytes=blob.stat().st_size,
                        mtime_ns=stamp,
                    )
                ]
            )
            restore_inventory(root, receipt)
            self.assertEqual(blob.stat().st_mtime_ns, stamp)
            blob.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                restore_inventory(root, receipt)
            nested = root / "nested/phase-execution.json"
            nested.parent.mkdir()
            nested.write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                restore_inventory(root, receipt)

    def row(self):
        row = dict(
            codec="jpeg",
            q=0,
            reference_sha256="a" * 64,
            reference_pixels_sha256="b" * 64,
            distorted_pixels_sha256="c" * 64,
            encoded_sha256="d" * 64,
            encoded_bytes=1,
            feature_ids=[13],
            feature_layout_sha256="e" * 64,
            feature_tool_sha256="f" * 64,
            tool_sha256="1" * 64,
            knobs={},
            width=32,
            height=24,
            features=[-0.0],
            maps=[
                dict(
                    metric="butteraugli",
                    sha256="2" * 64,
                    bytes=1,
                    pixel_pair_metric_key="3" * 64,
                )
            ],
            native_scores=[
                dict(
                    metric="butteraugli",
                    score=0.0,
                    scores=dict(pnorm3=-0.0),
                    reference_pixels_sha256="b" * 64,
                    distorted_pixels_sha256="c" * 64,
                    elapsed_ms=123.0,
                )
            ],
        )
        return row

    def test_feature_and_native_scalar_signed_zero_bits_must_match(self):
        row = self.row()
        same = copy.deepcopy(row)
        same["native_scores"][0]["elapsed_ms"] = 456.0
        self.assertEqual(projection(row), projection(same))
        feature = copy.deepcopy(row)
        feature["features"] = [0.0]
        self.assertNotEqual(projection(row), projection(feature))
        scalar = copy.deepcopy(row)
        scalar["native_scores"][0]["scores"]["pnorm3"] = 0.0
        self.assertNotEqual(projection(row), projection(scalar))
        changed_map = copy.deepcopy(row)
        changed_map["maps"][0]["sha256"] = "4" * 64
        self.assertNotEqual(projection(row), projection(changed_map))


if __name__ == "__main__":
    unittest.main()
