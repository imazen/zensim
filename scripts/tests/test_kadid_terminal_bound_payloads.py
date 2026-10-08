"""Round-three reviewer probes as synthetic-only bound-payload regressions."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_kadid_terminal_read as fixtures

owner = fixtures.owner
labels = owner.v2c_labels
stats = owner.zen_stats
_PANEL = os.environ["ZEN_PANEL_BIN"]


class _Watch:
    """Kernel inode events, independent of Python hooks and alias spelling."""

    def __init__(self, path):
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        if libc.inotify_add_watch(self.fd, os.fsencode(path), 0x21) < 0:
            os.close(self.fd)
            raise OSError(ctypes.get_errno(), "inotify_add_watch")

    def __enter__(self):
        return self

    def counts(self):
        opened = accessed = 0
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            offset = 0
            while offset < len(data):
                _, mask, _, length = struct.unpack_from("iIII", data, offset)
                opened += bool(mask & 0x20)
                accessed += bool(mask & 0x01)
                offset += 16 + length
        return {"sentinel_opens": opened, "sentinel_reads": accessed}

    def __exit__(self, *exc):
        os.close(self.fd)


def _zero_events(test, watch, case):
    events = watch.counts()
    print(json.dumps({"probe": case, **events}), flush=True)
    test.assertEqual(events, {"sentinel_opens": 0, "sentinel_reads": 0})


class LabelBytes(unittest.TestCase):
    def test_label_retarget_or_replace_after_hash_keeps_approved_bytes(self):
        # The real parser boundary occurs after hashing. No admission/hash or
        # decoding logic is replaced; this hook only changes the filesystem.
        for fmt in ("tsv", "csv", "json"):
            for operation in ("alias", "replace"):
                with (
                    self.subTest(format=fmt, operation=operation),
                    tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as scratch,
                ):
                    root = Path(scratch)
                    original = root / "approved-labels"
                    sentinel = root / "cid22_validation_set" / "labels"
                    sentinel.parent.mkdir()
                    if fmt == "json":
                        original.write_text(
                            json.dumps(
                                {
                                    "rows": [
                                        {"r": "r1", "d": "d1", "q": 10},
                                        {"r": "r2", "d": "d2", "q": 20},
                                    ]
                                }
                            )
                        )
                        sentinel.write_text(
                            json.dumps(
                                {
                                    "rows": [
                                        {"r": "r1", "d": "d1", "q": 20},
                                        {"r": "r2", "d": "d2", "q": 10},
                                    ]
                                }
                            )
                        )
                    else:
                        sep = "\t" if fmt == "tsv" else ","
                        original.write_text(
                            sep.join(("r", "d", "q"))
                            + "\n"
                            + sep.join(("r1", "d1", "10"))
                            + "\n"
                            + sep.join(("r2", "d2", "20"))
                            + "\n"
                        )
                        sentinel.write_text(
                            original.read_text()
                            .replace("10", "swap")
                            .replace("20", "10")
                            .replace("swap", "20")
                        )
                    path = root / "label-alias" if operation == "alias" else original
                    if operation == "alias":
                        path.symlink_to(original)
                    spec = {
                        "path": str(path),
                        "sha256": owner.sha(path),
                        "format": fmt,
                        "rows_key": "rows",
                        "ref_col": "r",
                        "dist_col": "d",
                        "label_col": "q",
                    }
                    parser = labels._read
                    mutations = []

                    def mutate_then_parse(*args, **kwargs):
                        if operation == "alias":
                            path.unlink()
                            path.symlink_to(sentinel)
                        else:
                            os.replace(sentinel, path)
                        mutations.append(operation)
                        return parser(*args, **kwargs)

                    with (
                        _Watch(sentinel) as watch,
                        patch.object(labels, "_read", mutate_then_parse),
                    ):
                        result = labels.load_label_rows(spec)
                        _zero_events(self, watch, f"label-{fmt}-{operation}-after-hash")
                        self.assertEqual(result.label.tolist(), [10.0, 20.0])
                    self.assertEqual(mutations, [operation])

    def test_via_pairs_retarget_after_hash_keeps_approved_pair_bytes(self):
        for operation in ("alias", "replace"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as scratch,
            ):
                root = Path(scratch)
                source, pairs = root / "labels.csv", root / "approved-pairs.tsv"
                source.write_text("stimulus,quality\n1,10\n2,20\n")
                pairs.write_text("id\tref_path\tdist_path\n1\tr1\td1\n2\tr2\td2\n")
                sentinel = root / "cid22_validation_set" / "pairs.tsv"
                sentinel.parent.mkdir()
                sentinel.write_text(
                    "id\tref_path\tdist_path\n1\tOTHER1\td1\n2\tOTHER2\td2\n"
                )
                path = root / "pair-alias" if operation == "alias" else pairs
                if operation == "alias":
                    path.symlink_to(pairs)
                spec = {
                    "path": str(source),
                    "sha256": owner.sha(source),
                    "format": "csv",
                    "label_col": "quality",
                    "via_pairs": {
                        "path": str(path),
                        "sha256": owner.sha(path),
                        "on": [["stimulus", "id"]],
                    },
                }
                parser = labels._read
                calls = []

                def mutate_then_parse(*args, **kwargs):
                    calls.append(True)
                    if len(calls) == 2:
                        if operation == "alias":
                            path.unlink()
                            path.symlink_to(sentinel)
                        else:
                            os.replace(sentinel, path)
                    return parser(*args, **kwargs)

                with (
                    _Watch(sentinel) as watch,
                    patch.object(labels, "_read", mutate_then_parse),
                ):
                    result = labels.load_label_rows(spec)
                    _zero_events(self, watch, f"via-pairs-{operation}-after-hash")
                    self.assertEqual(result.ref_path.tolist(), ["r1", "r2"])
                    self.assertEqual(result.label.tolist(), [10.0, 20.0])
                self.assertEqual(len(calls), 2)


class PanelAndInodes(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"ZEN_PANEL_BIN": _PANEL})
        environment.start()
        self.addCleanup(environment.stop)
        self.t = fixtures.TerminalRead()
        self.t.setUp()
        self.addCleanup(self.t.tearDown)
        self.panel = Path(self.t.record["panel"]["path"])

    def assert_unspent(self):
        self.assertFalse(self.t.journal.exists())
        self.assertFalse(self.t.output.exists())
        self.assertNotIn("KADID-TERMINAL-SPENT:", self.t.ledger.read_text())

    def label_substitution(self, operation):
        t = self.t
        # The reviewer's discriminating fixture: approved labels favor the
        # production model; substituted labels favor both comparators.
        other = t.root / "cid22_validation_set/labels.tsv"
        other.parent.mkdir()
        with t.predictions.open("w") as predicted, other.open("w") as sentinel:
            predicted.write("source_row_id\t" + "\t".join(owner.MODELS) + "\n")
            sentinel.write("ref_path\tdist_path\thuman_score\n")
            for i in range(2000):
                y = (i % 100) + (i // 100) / 100
                z = ((i * 7919) % 2000) / 10.0
                predicted.write(f"{i}\t{y!r}\t{z!r}\t{z!r}\n")
                sentinel.write(f"r{i // 100}\td{i}\t{z!r}\n")
        path = t.root / "labels-alias" if operation == "alias" else t.labels
        if operation == "alias":
            path.symlink_to(t.labels)
        t.record["labels"]["path"] = str(path)
        t.pin()
        parser = labels._read
        mutations = []

        def mutate_then_parse(*args, **kwargs):
            if operation == "alias":
                path.unlink()
                path.symlink_to(other)
            else:
                os.replace(other, path)
            mutations.append(operation)
            return parser(*args, **kwargs)

        with _Watch(other) as watch, patch.object(labels, "_read", mutate_then_parse):
            result = t.run_read()
            _zero_events(self, watch, f"full-authorized-label-{operation}-after-hash")
        self.assertEqual(mutations, [operation])
        self.assertEqual(result["confirmation"], "PASS")
        self.assertEqual(result["metrics"]["production"]["srocc_signed"], 1.0)
        self.assertLess(result["metrics"]["profile_b"]["srocc_signed"], 0.01)
        self.assertEqual(result["receipt_sha256"], t.auth["receipt_sha256"])
        self.assertTrue(t.journal.exists())

    def test_reviewer_label_alias_after_hash_cannot_change_authorized_verdict(self):
        self.label_substitution("alias")

    def test_reviewer_label_replace_after_hash_cannot_change_authorized_verdict(self):
        self.label_substitution("replace")

    def panel_substitution(self, operation):
        t = self.t
        spy = t.root / "spy-panel"
        capture = t.root / "spy-called"
        spy.write_text(
            f"#!/bin/sh\necho called >> '{capture}'\nexec '{self.panel}' \"$@\"\n"
        )
        spy.chmod(0o755)
        path = t.root / "selected-panel"
        if operation == "replace":
            shutil.copyfile(self.panel, path)
            path.chmod(0o755)
        else:
            path.symlink_to(self.panel)
        t.record["panel"] = t.spec(path)
        t.pin()
        reserve = owner.reserve
        calls = []
        processes = []
        run = subprocess.run

        def reserve_then_mutate(*args, **kwargs):
            result = reserve(*args, **kwargs)
            if operation == "replace":
                os.replace(spy, path)
            else:
                path.unlink()
                if operation != "remove":
                    path.symlink_to(spy)
            calls.append(operation)
            return result

        def trace_process(argv, *args, **kwargs):
            if "--batch" in argv or "--input" in argv:
                if not processes and str(argv[0]).startswith("/proc/self/fd/"):
                    fd = int(str(argv[0]).rsplit("/", 1)[1])
                    digest, offset = hashlib.sha256(), 0
                    while block := os.pread(fd, 1024 * 1024, offset):
                        digest.update(block)
                        offset += len(block)
                    self.assertEqual(digest.hexdigest(), t.record["panel"]["sha256"])
                processes.append((str(argv[0]), kwargs.get("pass_fds", ())))
            return run(argv, *args, **kwargs)

        # The fallback is fixture-owned; never touch a real Cargo target tree.
        fallback = t.root / "target/release/panel"
        fallback.parent.mkdir(parents=True)
        fallback.symlink_to(spy)
        with (
            _Watch(spy) as watch,
            patch.object(owner, "reserve", reserve_then_mutate),
            patch.object(stats, "_REPO_ROOT", str(t.root)),
            patch.object(subprocess, "run", trace_process),
        ):
            result = error = None
            try:
                result = t.run_read()
            except owner.TerminalReadError as refused:
                error = refused.category
            _zero_events(self, watch, f"panel-{operation}-after-reserve")
        self.assertIsNone(error)
        self.assertEqual(result["confirmation"], "PASS")
        self.assertEqual(result["receipt_sha256"], t.auth["receipt_sha256"])
        self.assertEqual(calls, [operation])
        self.assertFalse(capture.exists(), "unapproved panel received label inputs")
        self.assertEqual(len(processes), 113)
        for program, inherited in processes:
            self.assertTrue(program.startswith("/proc/self/fd/"))
            self.assertEqual(inherited, (int(program.rsplit("/", 1)[1]),))
        self.assertTrue(t.journal.exists())

    def test_panel_alias_retarget_after_reservation_executes_only_pinned_inode(self):
        self.panel_substitution("alias")

    def test_panel_rename_replace_after_reservation_executes_only_pinned_inode(self):
        self.panel_substitution("replace")

    def test_removed_panel_after_reservation_cannot_use_fallback(self):
        self.panel_substitution("remove")

    def test_explicit_missing_panel_has_no_silent_fallback(self):
        fallback = self.t.root / "target/release/panel"
        fallback.parent.mkdir(parents=True)
        fallback.write_text("synthetic fallback sentinel")
        with (
            _Watch(fallback) as watch,
            patch.dict(
                os.environ, {"ZEN_PANEL_BIN": str(self.t.root / "missing-panel")}
            ),
            patch.object(stats, "_REPO_ROOT", str(self.t.root)),
        ):
            with self.assertRaises(FileNotFoundError):
                stats.panel([1, 2, 3], [1, 2, 3])
            _zero_events(self, watch, "explicit-missing-panel-with-present-fallback")
        self.assert_unspent()

    def test_panel_pin_changed_before_reservation_refuses_without_label_open(self):
        path = self.t.root / "panel-alias"
        path.symlink_to(self.panel)
        self.t.record["panel"] = self.t.spec(path)
        self.t.pin()
        sentinel = self.t.root / "different-panel"
        sentinel.write_text("synthetic unapproved executable bytes")
        preflight = owner.preflight
        swapped = []

        def preflight_then_swap(*args, **kwargs):
            result = preflight(*args, **kwargs)
            path.unlink()
            path.symlink_to(sentinel)
            swapped.append(True)
            return result

        with (
            _Watch(self.t.labels) as watch,
            patch.object(owner, "preflight", preflight_then_swap),
        ):
            with self.assertRaisesRegex(owner.TerminalReadError, "preflight-refused"):
                self.t.run_read()
            _zero_events(self, watch, "changed-panel-before-reservation-label-sentinel")
        self.assertEqual(swapped, [True])
        self.assert_unspent()

    def test_reviewer_h1_hardlinked_receipt_opens_no_terminal_sentinel(self):
        sentinel = self.t.root / "kadid_terminal/dmos.csv"
        sentinel.parent.mkdir()
        sentinel.write_bytes(self.t.labels.read_bytes())
        alias = self.t.root / "prepared-receipt.json"
        os.link(sentinel, alias)
        self.assertEqual(
            (alias.stat().st_dev, alias.stat().st_ino),
            (sentinel.stat().st_dev, sentinel.stat().st_ino),
        )
        with _Watch(sentinel) as watch:
            with self.assertRaises((ValueError, FileNotFoundError)):
                owner.preflight(
                    alias,
                    self.t.root / "missing-auth",
                    self.t.ledger,
                    self.t.journal,
                    self.t.output,
                )
            _zero_events(self, watch, "H1-hardlinked-receipt-before-authorization")
        self.assert_unspent()

    def test_reviewer_h2_hardlinked_binding_opens_no_approved_label_inode(self):
        alias = self.t.root / "binding.tsv"
        os.link(self.t.labels, alias)
        self.t.record["bindings"] = [self.t.spec(alias)]
        self.t.pin()
        # Preserve the reviewer's later refusing gate. On the reviewed tip the
        # label inode was opened during binding checks before this refusal.
        q = json.loads(self.t.qualification.read_text())
        q["gates"]["G-RD"]["state"] = "blocked"
        self.t.qualification.write_text(json.dumps(q))
        self.t.record["qualification"] = self.t.spec(self.t.qualification)
        self.t.receipt.write_text(json.dumps(self.t.record))
        subprocess.run(["git", "add", "receipt.json"], cwd=self.t.root, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "test: synthetic later gate refusal"],
            cwd=self.t.root,
            check=True,
        )
        self.t.auth.update(
            receipt_sha256=owner.sha(self.t.receipt),
            pre_read_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.t.root, text=True
            ).strip(),
        )
        self.t.authorization.write_text(json.dumps(self.t.auth))
        with _Watch(self.t.labels) as watch:
            with self.assertRaisesRegex(owner.TerminalReadError, "preflight-refused"):
                self.t.run_read()
            _zero_events(self, watch, "H2-hardlinked-binding-before-later-gates")
        self.assert_unspent()

    def test_hardlink_inserted_after_leaf_stat_never_gets_a_data_open(self):
        sentinel = self.t.root / "kadid_terminal/dmos.csv"
        sentinel.parent.mkdir()
        sentinel.write_bytes(self.t.labels.read_bytes())
        path = self.t.receipt
        backup = self.t.root / "receipt.backup"
        real = os.open
        swapped = []

        def late_hardlink(name, *args, **kwargs):
            parent = kwargs.get("dir_fd")
            if (
                parent is not None
                and Path(os.readlink(f"/proc/self/fd/{parent}")) / name == path
                and not swapped
            ):
                path.rename(backup)
                os.link(sentinel, path)
                swapped.append(True)
            return real(name, *args, **kwargs)

        with _Watch(sentinel) as watch, patch.object(os, "open", late_hardlink):
            with self.assertRaises(ValueError):
                owner.preflight(
                    path,
                    self.t.authorization,
                    self.t.ledger,
                    self.t.journal,
                    self.t.output,
                )
            _zero_events(self, watch, "hardlink-inserted-after-leaf-stat")
        self.assertEqual(swapped, [True])
        self.assert_unspent()

    def test_hardlink_inserted_after_bound_identity_cannot_redirect_data_open(self):
        sentinel = self.t.root / "kadid_terminal/dmos.csv"
        sentinel.parent.mkdir()
        sentinel.write_bytes(self.t.labels.read_bytes())
        path, backup = self.t.receipt, self.t.root / "receipt.backup"
        real = os.open
        swapped = []

        def late_hardlink(name, *args, **kwargs):
            if str(name).startswith("/proc/self/fd/") and not swapped:
                path.rename(backup)
                os.link(sentinel, path)
                swapped.append(True)
            return real(name, *args, **kwargs)

        with _Watch(sentinel) as watch, patch.object(os, "open", late_hardlink):
            with self.assertRaises(FileNotFoundError):
                owner.preflight(
                    path,
                    self.t.root / "missing-auth",
                    self.t.ledger,
                    self.t.journal,
                    self.t.output,
                )
            _zero_events(self, watch, "hardlink-inserted-after-bound-identity")
        self.assertEqual(swapped, [True])
        self.assert_unspent()


if __name__ == "__main__":
    unittest.main()
