"""Synthetic late-retarget probes at real I/O boundaries, including ancestors."""

import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_kadid_terminal_read as fixtures

owner = fixtures.owner


class MetadataRace(unittest.TestCase):
    def setUp(self):
        self.t = fixtures.TerminalRead()
        self.t.setUp()
        self.sentinel = self.t.root / "kadid_terminal/dmos.csv"
        self.sentinel.parent.mkdir()
        self.sentinel.write_bytes(self.t.labels.read_bytes())
        self.sentinel_identity = (
            self.sentinel.stat().st_dev,
            self.sentinel.stat().st_ino,
        )
        self.seen = []

    def tearDown(self):
        self.t.tearDown()

    def opened_guard(self, mutate=lambda *a, **k: None):
        real = os.open

        def opened(path, *args, **kwargs):
            mutate(path, *args, **kwargs)
            fd = real(path, *args, **kwargs)
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) == self.sentinel_identity:
                self.seen.append(str(path))
                os.close(fd)
                raise AssertionError("protected sentinel opened")
            return fd

        return opened

    def unspent(self):
        self.assertEqual(self.seen, [])
        self.assertFalse(self.t.journal.exists())
        self.assertFalse(self.t.output.exists())
        self.assertNotIn("KADID-TERMINAL-SPENT:", self.t.ledger.read_text())

    def preflight(self, receipt=None, authorization=None):
        return owner.preflight(
            receipt or self.t.receipt,
            authorization or self.t.authorization,
            self.t.ledger,
            self.t.journal,
            self.t.output,
        )

    def test_reviewer_io_open_late_alias_retarget(self):
        # Same I/O-boundary mutation as alias_race.py. The old tip opens the
        # sentinel once; a stable control and both fixed calls open it zero times.
        for mutate in (False, True):
            with self.subTest(mutate=mutate):
                alias = self.t.root / f"metadata-alias-{mutate}.json"
                alias.symlink_to(self.t.receipt)
                real = io.open
                swapped = []

                def boundary(path, *args, **kwargs):
                    if not isinstance(path, int):
                        if Path(path) == alias and mutate and not swapped:
                            alias.unlink()
                            alias.symlink_to(self.sentinel)
                            swapped.append(True)
                        if Path(path).resolve() == self.sentinel:
                            self.seen.append(str(path))
                    return real(path, *args, **kwargs)

                with patch("io.open", boundary), patch("os.open", self.opened_guard()):
                    with self.assertRaises(FileNotFoundError):
                        self.preflight(
                            alias, self.t.root / "missing-authorization.json"
                        )
                self.unspent()

    def test_stable_prepared_alias_and_after_admission_retarget(self):
        alias = self.t.root / "metadata-alias.json"
        alias.symlink_to(self.t.receipt)
        result = self.preflight(alias)
        self.assertEqual(result[1], self.t.auth["receipt_sha256"])
        admit = owner.metadata_path
        swapped = []

        def retarget(path):
            resolved = admit(path)
            if Path(path) == alias and not swapped:
                alias.unlink()
                alias.symlink_to(self.sentinel)
                swapped.append(True)
            return resolved

        with (
            patch.object(owner, "metadata_path", retarget),
            patch("os.open", self.opened_guard()),
        ):
            result = self.preflight(alias)
        self.assertTrue(swapped)
        self.assertEqual(result[1], self.t.auth["receipt_sha256"])
        self.unspent()

    def test_leaf_retarget_at_os_open_refuses_before_reservation(self):
        # Exercise hashing and JSON/TSV consumers, not only initial receipt I/O.
        for path in (
            self.t.receipt,
            self.t.authorization,
            self.t.registration,
            self.t.qualification,
            self.t.e30,
            self.t.prediction_receipt,
            self.t.population,
            self.t.predictions,
            self.t.inspector,
            self.t.b,
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                swapped = []

                def retarget(name, *args, **kwargs):
                    directory = kwargs.get("dir_fd")
                    if (
                        directory is not None
                        and Path(os.readlink(f"/proc/self/fd/{directory}")) / name
                        == path
                        and not swapped
                    ):
                        path.unlink()
                        path.symlink_to(self.sentinel)
                        swapped.append(True)

                try:
                    with patch("os.open", self.opened_guard(retarget)):
                        with self.assertRaises((OSError, ValueError)):
                            self.preflight()
                    self.assertTrue(swapped)
                    self.unspent()
                finally:
                    if swapped:
                        path.unlink()
                        path.write_bytes(original)
                        if path == self.t.inspector:
                            path.chmod(0o755)

    def test_mutable_ancestor_before_and_after_directory_handle(self):
        for after_handle in (False, True):
            with self.subTest(after_handle=after_handle):
                parent = self.t.root / f"prepared-{after_handle}"
                parent.mkdir()
                receipt = parent / "dmos.csv"
                receipt.write_bytes(self.t.receipt.read_bytes())
                saved = self.t.root / f"prepared-{after_handle}.bak"
                swapped = []

                def retarget(name, *args, **kwargs):
                    directory = kwargs.get("dir_fd")
                    if directory is None:
                        return
                    bound_parent = Path(os.readlink(f"/proc/self/fd/{directory}"))
                    target = bound_parent / name
                    boundary = receipt if after_handle else parent
                    if target == boundary and not swapped:
                        parent.rename(saved)
                        parent.symlink_to(
                            self.sentinel.parent, target_is_directory=True
                        )
                        swapped.append(True)

                with patch("os.open", self.opened_guard(retarget)):
                    if after_handle:
                        with self.assertRaises(FileNotFoundError):
                            self.preflight(
                                receipt, self.t.root / "missing-authorization.json"
                            )
                    else:
                        with self.assertRaises(OSError):
                            self.preflight(
                                receipt, self.t.root / "missing-authorization.json"
                            )
                self.assertTrue(swapped)
                self.unspent()

    def test_checked_bytes_parse_retained_object_after_open(self):
        for slot in (
            "qualification",
            "prediction_receipt",
            "population",
            "predictions",
        ):
            with self.subTest(slot=slot):
                path = Path(self.t.record[slot]["path"])
                original = path.read_bytes()
                real = os.fdopen
                swapped = []

                def retarget(fd, *args, **kwargs):
                    if Path(os.readlink(f"/proc/self/fd/{fd}")) == path and not swapped:
                        path.unlink()
                        path.symlink_to(self.sentinel)
                        swapped.append(True)
                    return real(fd, *args, **kwargs)

                try:
                    with (
                        patch("os.fdopen", retarget),
                        patch("os.open", self.opened_guard()),
                    ):
                        # Hash and parse this object without reopening by name.
                        if slot in ("population", "predictions"):
                            data = owner.rows(self.t.record[slot])
                            self.assertEqual(len(data), 2000)
                        else:
                            data = json.loads(owner.checked_bytes(self.t.record[slot]))
                            self.assertIsInstance(data, dict)
                    self.assertTrue(swapped)
                    self.unspent()
                finally:
                    path.unlink()
                    path.write_bytes(original)

    def test_reservation_leaf_retarget_cannot_open_sentinel_or_spend(self):
        for slot in ("ledger", "journal"):
            with self.subTest(slot=slot):
                path = getattr(self.t, slot)
                original = path.read_bytes() if path.exists() else None
                swapped = []

                def retarget(name, *args, **kwargs):
                    directory = kwargs.get("dir_fd")
                    if (
                        directory is not None
                        and Path(os.readlink(f"/proc/self/fd/{directory}")) / name
                        == path
                        and not swapped
                    ):
                        if path.exists():
                            path.unlink()
                        path.symlink_to(self.sentinel)
                        swapped.append(True)

                try:
                    with patch("os.open", self.opened_guard(retarget)):
                        with self.assertRaises(OSError):
                            owner.reserve(self.t.ledger, self.t.journal, "a" * 64)
                    self.assertTrue(swapped)
                    self.assertEqual(self.seen, [])
                finally:
                    if swapped:
                        path.unlink()
                        if original is not None:
                            path.write_bytes(original)
                self.unspent()

    def test_inspector_consumes_hash_bound_model_handle(self):
        # Mutate both pathnames at the subprocess boundary. A model inspector
        # still reads the inherited admitted model fd and executes the admitted
        # inspector fd, without opening either retargeted sentinel alias.
        model = self.t.b
        original = model.read_bytes()
        inspector = self.t.inspector
        old_script = inspector.read_bytes()
        inspector.write_text(
            "#!/usr/bin/env python3\nimport sys,json\n"
            "from pathlib import Path\n"
            + "assert Path(sys.argv[1]).read_bytes() == "
            + repr(original)
            + "\n"
            + old_script.decode().split("import json\n", 1)[1]
        )
        self.t.record["inspector"] = self.t.spec(inspector)
        self.t.record["scorer"] = self.t.spec(inspector)
        self.t.pin()
        real = owner.subprocess.run
        swapped = []

        def retarget(argv, *args, **kwargs):
            if argv and str(argv[0]).startswith("/proc/self/fd/") and not swapped:
                model.unlink()
                model.symlink_to(self.sentinel)
                inspector.unlink()
                inspector.symlink_to(self.sentinel)
                swapped.append(True)
            return real(argv, *args, **kwargs)

        try:
            with (
                patch.object(owner.subprocess, "run", retarget),
                patch("os.open", self.opened_guard()),
            ):
                # Call the real external owner path. Later full preflight model
                # reads are absent; population and predictions remain admitted.
                result = self.preflight()
            self.assertTrue(swapped)
            self.assertEqual(result[1], self.t.auth["receipt_sha256"])
            self.unspent()
        finally:
            if swapped:
                model.unlink()
                model.write_bytes(original)
                inspector.unlink()
                inspector.write_bytes(old_script)
                inspector.chmod(0o755)
