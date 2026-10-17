"""Invented native-worker boundary fixtures; no actual worker or optimizer."""

import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "rev4_featpot"))
import e36_phase_worker as owner
from e36_tower_container import container_argv


def write(p, x):
    p.write_text(json.dumps(x))


class PhaseWorker(unittest.TestCase):
    def fixture(self, root):
        directory = root / "runs/additions-v6"
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        write(
            manifest,
            [
                dict(
                    kind=dict(
                        kind="bake", view='e36-native-ladder-v1:{"invented":true}'
                    )
                )
            ],
        )
        clock = root / "clock.json"
        write(clock, dict(deadline_unix=time.time() + 60))
        binary = root / "bin/zenfleet-worker"
        binary.parent.mkdir()
        binary.write_bytes(b"invented worker never executed")
        payload = root / "input.bin"
        payload.write_bytes(b"invented immutable input")
        files = root / "files.json"
        spec = dict(
            schema="e36-whole-native-phase-files-v1",
            root=str(root),
            scientific_change=False,
            fits=0,
            files=[
                dict(destination=str(p), sha256=owner.sha(p))
                for p in [manifest, clock, binary, payload]
            ],
            clock=dict(path=str(clock), sha256=owner.sha(clock)),
            runtime=dict(path="invented-runtime"),
            worker_slots=8,
            packing_ram_bytes=8 * 1024**3,
            manifests=dict(
                additions=dict(path=str(manifest), sha256=owner.sha(manifest), views=1)
            ),
        )
        write(files, spec)
        return files, directory, payload

    def test_whole_manifest_native_arguments_and_source_nanosecond_inventory(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-native-phase-"
        ) as d:
            root = Path(d)
            files, directory, _ = self.fixture(root)
            stamp = time.time_ns() - 12345

            def native(argv, env):
                self.assertEqual(
                    argv[argv.index("--manifest") + 1], str(directory / "manifest.json")
                )
                self.assertEqual(argv[argv.index("--max-attempts") + 1], "1")
                self.assertEqual(env["ZEN_RAM_BYTES"], str(8 * 1024**3))
                self.assertEqual(env["ZEN_PERSISTENT_EXEC"], "1")
                blob = directory / "blobs/hash"
                blob.parent.mkdir()
                blob.write_bytes(b"invented native output")
                import os

                os.utime(blob, ns=(stamp, stamp))
                return types.SimpleNamespace(returncode=0)

            with patch.object(owner.subprocess, "run", side_effect=native):
                self.assertEqual(owner.run(files, "additions"), 0)
            r = json.loads((directory / "phase-execution.json").read_text())
            blob = next(v for v in r["native_files"] if v["relative"] == "blobs/hash")
            self.assertEqual(blob["mtime_ns"], stamp)
            self.assertEqual(r["optimizer_steps"], 0)
            with self.assertRaisesRegex(ValueError, "prior native execution"):
                owner.run(files, "additions")

    def test_fit_job_and_changed_source_are_refused_without_success_receipt(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-native-phase-"
        ) as d:
            root = Path(d)
            files, directory, payload = self.fixture(root)

            def mutate(argv, env):
                payload.write_bytes(b"changed")
                return types.SimpleNamespace(returncode=0)

            with patch.object(owner.subprocess, "run", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "during execution"):
                    owner.run(files, "additions")
            self.assertFalse((directory / "phase-execution.json").exists())
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-native-phase-"
        ) as d:
            root = Path(d)
            files, directory, _ = self.fixture(root)
            manifest = directory / "manifest.json"
            write(
                manifest,
                [
                    dict(
                        kind=dict(kind="fit_cell", view="e36-native-ladder-v1:invented")
                    )
                ],
            )
            s = json.loads(files.read_text())
            s["files"][0]["sha256"] = owner.sha(manifest)
            s["manifests"]["additions"]["sha256"] = owner.sha(manifest)
            write(files, s)
            with patch.object(owner.subprocess, "run") as native:
                with self.assertRaisesRegex(ValueError, "including fits"):
                    owner.run(files, "additions")
                native.assert_not_called()

    def test_container_memory_and_entrypoint_follow_the_declared_phase(self):
        spec = dict(
            root="/dataset/e36",
            cpu_shares=256,
            container_memory_bytes=12 * 1024**3,
            cpuset="0-7,16-23",
            python_source_root="/host/python",
            files=[],
            image="image@sha256:invented",
            worker_slots=8,
            runtime=dict(path="/dataset/e36/runtime.json"),
        )
        stage = dict(resolved_physical_root="/native/phase", manifest_name="files.json")
        args = container_argv(spec, stage, "invented", "additions")
        self.assertEqual(args[args.index("--memory") + 1], str(12 * 1024**3))
        self.assertEqual(args[args.index("--mem") + 1], "12G")
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertEqual(args[args.index("--entrypoint") + 1], "/opt/e36/run-heavy")


if __name__ == "__main__":
    unittest.main()
