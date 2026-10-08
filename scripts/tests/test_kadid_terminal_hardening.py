"""Round-five synthetic regressions: filesystem, destinations, loaded inputs."""

import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import test_kadid_terminal_read as fixtures
from test_kadid_terminal_bound_payloads import _Watch, _zero_events

owner = fixtures.owner


def prove_filesystem_separation(preparation, corpus):
    """Caller-selected, fresh synthetic directories; no real corpus payloads."""
    preparation, corpus = Path(preparation), Path(corpus)
    preparation.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)
    if preparation.stat().st_dev == corpus.stat().st_dev:
        raise ValueError("proof requires genuinely distinct filesystems")
    approved, sentinel = preparation / "approved.json", corpus / "sentinel.json"
    if approved.exists() or sentinel.exists():
        raise FileExistsError("preserve existing proof artifacts")
    approved.write_bytes(b'{"synthetic": "approved"}\n')
    sentinel.write_bytes(b'{"synthetic": "protected"}\n')
    with _Watch(sentinel) as watch:
        with owner._terminal_bound_io._open_admitted(
            approved, protected=(corpus,)
        ) as f:
            if f.read() != b'{"synthetic": "approved"}\n':
                raise AssertionError("wrong preparation bytes")
        try:
            os.link(sentinel, preparation / "forbidden-hardlink")
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        else:
            raise AssertionError("cross-filesystem hard link unexpectedly succeeded")
        events = watch.counts()
        if events != {"sentinel_opens": 0, "sentinel_reads": 0}:
            raise AssertionError(events)
    print(
        json.dumps(
            {
                "proof": "actual-filesystem-separation",
                "preparation_device": preparation.stat().st_dev,
                "corpus_device": corpus.stat().st_dev,
                "hardlink_errno": errno.EXDEV,
                **events,
            }
        ),
        flush=True,
    )


class Hardening(unittest.TestCase):
    def setUp(self):
        self.t = fixtures.TerminalRead()
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def preflight(self, receipt=None, authorization=None):
        t = self.t
        return owner.preflight(
            receipt or t.receipt,
            authorization or t.authorization,
            t.ledger,
            t.journal,
            t.output,
        )

    def unspent(self):
        t = self.t
        self.assertFalse(t.journal.exists())
        self.assertFalse(t.output.exists())
        self.assertNotIn("KADID-TERMINAL-SPENT:", t.ledger.read_text())

    def sentinel(self):
        p = self.t.root / "kadid_terminal/dmos.json"
        p.parent.mkdir()
        p.write_text(json.dumps({"human_score": 99, "stimulus": "synthetic"}))
        return p

    def test_same_filesystem_refuses_before_receipt_payload_open(self):
        sentinel = self.sentinel()
        # Even a legitimate receipt on the corpus filesystem must refuse.
        with (
            _Watch(self.t.receipt) as receipt_watch,
            _Watch(sentinel) as watch,
            patch.object(owner, "CORPUS_ROOTS", (sentinel.parent,)),
        ):
            with self.assertRaisesRegex(ValueError, "separate filesystems"):
                self.preflight()
            _zero_events(self, receipt_watch, "same-device-approved-receipt-refusal")
            _zero_events(self, watch, "same-device-protected-sentinel-refusal")
        self.unspent()

    def test_transient_single_link_report_cannot_admit_protected_inode(self):
        sentinel = self.sentinel()
        alias = self.t.root / "trusted-receipt.json"
        os.link(sentinel, alias)
        identity = (sentinel.stat().st_dev, sentinel.stat().st_ino)
        real_stat, real_fstat = os.stat, os.fstat

        def one_link(info):
            if (info.st_dev, info.st_ino) == identity:
                values = list(info)
                values[3] = 1
                return os.stat_result(values)
            return info

        # Model precisely the reviewer's measured transient kernel report;
        # real opens/read calls and device/inode identities are untouched.
        with (
            _Watch(sentinel) as watch,
            patch.object(owner, "CORPUS_ROOTS", (sentinel.parent,)),
            patch("os.stat", lambda *a, **k: one_link(real_stat(*a, **k))),
            patch("os.fstat", lambda *a, **k: one_link(real_fstat(*a, **k))),
        ):
            error = None
            try:
                self.preflight(alias, self.t.root / "missing-auth.json")
            except (OSError, ValueError) as exc:
                error = exc
            _zero_events(self, watch, "transient-nlink-one-protected-inode")
            self.assertIsInstance(error, ValueError)
        self.unspent()

    def test_reviewer_s6_rename_stress_20000_zero_data_opens(self):
        sentinel = self.sentinel()
        trusted = self.t.root / "flipped-receipt.json"
        trusted.write_bytes(self.t.receipt.read_bytes())
        stop, started = threading.Event(), threading.Event()
        counts, errors = [], []

        def flip():
            tmp = self.t.root / "receipt.flip"
            good = self.t.receipt.read_bytes()
            n = 0
            try:
                while not stop.is_set():
                    if tmp.exists():
                        tmp.unlink()
                    if n % 2:
                        os.link(sentinel, tmp)
                    else:
                        tmp.write_bytes(good)
                    os.replace(tmp, trusted)
                    n += 1
                    if n >= 2:
                        started.set()
            except BaseException as exc:
                errors.append(exc)
                started.set()
            finally:
                counts.append(n)

        real_open = os.open
        sid = (sentinel.stat().st_dev, sentinel.stat().st_ino)
        data_opens = []

        def observed_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            info = os.fstat(fd)
            if not args[1] & os.O_PATH and (info.st_dev, info.st_ino) == sid:
                data_opens.append(str(args[0]))
            return fd

        thread = threading.Thread(target=flip)
        with _Watch(sentinel) as watch:
            thread.start()
            try:
                self.assertTrue(started.wait(10), "flipper did not start")
                with (
                    patch.object(owner, "CORPUS_ROOTS", (sentinel.parent,)),
                    patch("os.open", observed_open),
                ):
                    for _ in range(20000):
                        with self.assertRaises((OSError, ValueError)):
                            self.preflight(trusted, self.t.root / "missing-auth.json")
            finally:
                stop.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertGreaterEqual(counts[0], 2)
            print(
                json.dumps(
                    {
                        "probe": "S6-rename-stress",
                        "iterations": 20000,
                        "flips": counts[0],
                        "harness_data_opens": len(data_opens),
                    }
                ),
                flush=True,
            )
            self.assertEqual(data_opens, [])
            _zero_events(self, watch, "S6-rename-stress-events")
        self.unspent()

    def ledger_substitution(self, operation, when):
        t = self.t
        approved = t.ledger
        other = t.root / "unauthorized-ledger.md"
        other.write_text("Unapproved synthetic ledger\n")
        alias = t.root / "ledger-alias.md"
        alias.symlink_to(approved)
        t.ledger = alias
        t.pin()
        mutated = []

        def mutate():
            if operation == "alias":
                alias.unlink()
                alias.symlink_to(other)
            else:
                saved = t.root / "authorized-ledger.retained.md"
                approved.rename(saved)
                other.rename(approved)
                mutated.append(saved)
            mutated.append(True)

        real = owner.preflight if when == "preflight" else owner.reserve

        def boundary(*args, **kwargs):
            result = real(*args, **kwargs)
            mutate()
            return result

        with (
            _Watch(other) as watch,
            patch.object(
                owner, "preflight" if when == "preflight" else "reserve", boundary
            ),
        ):
            result = t.run_read()
            _zero_events(self, watch, f"ledger-{operation}-after-{when}")
        self.assertTrue(mutated)
        self.assertEqual(result["confirmation"], "PASS")
        bound_name = mutated[0] if operation == "replace" else approved
        text = bound_name.read_text()
        self.assertIn("KADID-TERMINAL-SPENT:", text)
        self.assertIn("D2 result", text)
        other_name = approved if operation == "replace" else other
        self.assertEqual(other_name.read_text(), "Unapproved synthetic ledger\n")

    def test_ledger_alias_after_preflight_keeps_authorized_handle(self):
        self.ledger_substitution("alias", "preflight")

    def test_ledger_replace_after_preflight_keeps_authorized_inode(self):
        self.ledger_substitution("replace", "preflight")

    def test_ledger_alias_after_reservation_cannot_redirect_result_append(self):
        self.ledger_substitution("alias", "reserve")

    def test_contract_change_refuses_before_labels_or_spending(self):
        contract = self.t.root / "fit-contract.json"
        contents = json.loads(owner.CONTRACT.read_text())
        contents["unapproved_acceptance_input"] = "synthetic mutation"
        contract.write_text(json.dumps(contents))
        with patch.object(owner, "CONTRACT", contract), _Watch(self.t.labels) as watch:
            error = None
            try:
                self.t.run_read()
            except owner.TerminalReadError as exc:
                error = exc
            _zero_events(self, watch, "changed-contract-label-sentinel")
            self.assertIsInstance(error, owner.TerminalReadError)
        self.unspent()

    def test_missing_contract_pin_refuses_before_labels_or_spending(self):
        self.t.record.pop("contract_sha256")
        self.t.pin()
        with _Watch(self.t.labels) as watch:
            error = None
            try:
                self.t.run_read()
            except owner.TerminalReadError as exc:
                error = exc
            _zero_events(self, watch, "missing-contract-pin-label-sentinel")
            self.assertIsInstance(error, owner.TerminalReadError)
        self.unspent()

    def test_disk_hash_cannot_replace_loaded_source_identity(self):
        # Substitute the advertised disk pathname after its module was loaded.
        # A matching disk receipt cannot authorize different executed bytes.
        code = dict(owner.CODE)
        name = "v2c_labels.py"
        disk = self.t.root / name
        disk.write_bytes(
            code[name].read_bytes() + b"\n# changed after module execution\n"
        )
        code[name] = disk
        self.t.record["code_sha256"] = {k: owner.sha(p) for k, p in code.items()}
        self.t.pin()
        with patch.object(owner, "CODE", code), _Watch(self.t.labels) as watch:
            error = None
            try:
                self.t.run_read()
            except owner.TerminalReadError as exc:
                error = exc
            _zero_events(self, watch, "post-import-source-pin-label-sentinel")
            self.assertIsInstance(error, owner.TerminalReadError)
        self.unspent()

    def test_source_loader_compiles_the_same_bytes_it_hashes(self):
        path = self.t.root / "module.py"
        original = b"VALUE = 'approved'\n"
        path.write_bytes(original)
        real = Path.read_bytes

        def read_then_replace(p):
            data = real(p)
            if p == path:
                p.write_bytes(b"VALUE = 'unapproved'\n")
            return data

        try:
            with patch.object(Path, "read_bytes", read_then_replace):
                module = owner.acceptance.load(path, "_synthetic_terminal_source")
            self.assertEqual(module.VALUE, "approved")
            self.assertEqual(
                owner.acceptance.SOURCES[path], hashlib.sha256(original).hexdigest()
            )
        finally:
            owner.acceptance.SOURCES.pop(path, None)
            sys.modules.pop("_synthetic_terminal_source", None)

    def test_bootstrap_cached_code_cannot_match_changed_source(self):
        path = self.t.root / "bootstrap.py"
        executed = compile("VALUE = 'approved'\n", str(path), "exec")
        path.write_text("VALUE = 'unapproved'\n")
        with self.assertRaisesRegex(ValueError, "executing code"):
            owner.acceptance.capture(path, executed)
        self.assertNotIn(path, owner.acceptance.SOURCES)

    def test_terminal_import_has_no_unpinned_project_side_effects(self):
        root = self.t.root / "source-copy"
        for name, path in owner.CODE.items():
            dst = (
                root
                / "scripts"
                / ("lib" if name == "zen_stats.py" else "rev4_featpot")
                / name
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
        marker = self.t.root / "unapproved-import-marker"
        marker.write_text("synthetic sentinel")
        spy = f"from pathlib import Path\nPath({str(marker)!r}).write_text('unapproved')\n"
        (root / "scripts/rev4_featpot/v2_common.py").write_text(spy)
        (root / "scripts/lib/assessment_identity.py").write_text(spy)
        code = "import kadid_terminal_read as o; import json,sys; print(json.dumps(sorted(o.CODE))); assert 'v2_common' not in sys.modules; assert 'lib.assessment_identity' not in sys.modules"
        with _Watch(marker) as watch:
            result = subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONPATH": str(root / "scripts/rev4_featpot")},
                check=True,
                capture_output=True,
                text=True,
            )
            _zero_events(self, watch, "unpinned-project-import-sentinel")
        self.assertEqual(json.loads(result.stdout), sorted(owner.CODE))
