"""Invented resource and exclusive-start boundaries; no native work runs."""

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "rev4_featpot"))
import e36_phase_start as owner


class PhaseStart(unittest.TestCase):
    def test_a_fresh_receipt_without_a_live_parent_debugger_is_refused(self):
        with self.assertRaisesRegex(ValueError, "parent debugger still holding"):
            owner.stopped_owner(
                dict(controller_pid=os.getpid(), debugger_pid=os.getppid())
            )

    def test_busy_host_or_inadequate_available_memory_refuses(self):
        binding = dict(
            host="invented",
            host_config=dict(
                Memory=16 * 1024**3,
                CpuShares=256,
                CpusetCpus="0-7,16-23",
                NetworkMode="none",
            ),
        )

        def reply(args, **kwargs):
            raise AssertionError("unexpected remote invocation")

        def remote(host, args, **kwargs):
            if args[:2] == ["docker", "ps"]:
                return types.SimpleNamespace(stdout="zen-score-steer-invented\n")
            return reply(args, **kwargs)

        with patch.object(
            owner, "inspect_bound", return_value=dict(State=dict(Status="created"))
        ):
            with patch.object(owner, "remote", side_effect=remote):
                with self.assertRaisesRegex(
                    ValueError, "heavy allocation still active"
                ):
                    owner.resource_admission(binding)
            with patch.object(
                owner,
                "remote",
                side_effect=[
                    types.SimpleNamespace(stdout=""),
                    types.SimpleNamespace(stdout="MemAvailable: 20971520 kB\n"),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "at least12GiB"):
                    owner.resource_admission(binding)

    def test_exclusive_ownership_is_sealed_before_start_even_when_start_fails(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-start-"
        ) as d:
            root = Path(d)

            def put(name, value):
                p = root / name
                p.write_text(json.dumps(value))
                return dict(path=str(p), sha256=owner.sha(p))

            execution = put(
                "first.json",
                dict(
                    returncode=0,
                    phase="first-producer",
                    files_manifest_sha256="files",
                    effective_clock_sha256="clock",
                ),
            )
            parity = put(
                "parity.json",
                dict(
                    admitted=True,
                    views=5,
                    bounded_first_producer_only=True,
                    native_execution=execution,
                ),
            )
            guard = put(
                "guard.json",
                dict(
                    admitted=True, root=str(root), checked_unix_ns=owner.time.time_ns()
                ),
            )
            put(
                "bulk-plan-v6.json",
                dict(
                    phases=dict(
                        additions=dict(run="additions-v6", manifest_sha256="manifest")
                    )
                ),
            )
            binding = put(
                "container.json",
                dict(
                    phase="additions",
                    host="invented",
                    container_id="invented",
                    files_manifest_sha256="files",
                    effective_clock=dict(sha256="clock"),
                    manifest=dict(sha256="manifest"),
                ),
            )
            adapter = root / "adapter.py"
            adapter.write_text("# invented\n")

            def start(*args, **kwargs):
                self.assertTrue((root / "TOWER_PHASE_HANDOFF.json").exists())
                raise RuntimeError("invented Docker start failure")

            with (
                patch.object(
                    owner,
                    "resource_admission",
                    return_value=dict(cap_bytes=16 * 1024**3),
                ),
                patch.object(owner, "remote", side_effect=start),
                patch.object(owner, "stopped_owner"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "invented Docker start failure"
                ):
                    owner.seal_start(
                        root,
                        binding["path"],
                        adapter,
                        parity["path"],
                        guard["path"],
                        123,
                        root / "lock",
                        root / "out",
                    )
            self.assertTrue((root / "TOWER_PHASE_HANDOFF.json").exists())
            with (
                patch.object(owner, "resource_admission", return_value={}),
                patch.object(owner, "remote") as native,
                patch.object(owner, "stopped_owner"),
            ):
                with self.assertRaises(FileExistsError):
                    owner.seal_start(
                        root,
                        binding["path"],
                        adapter,
                        parity["path"],
                        guard["path"],
                        123,
                        root / "lock",
                        root / "retry",
                    )
                native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
