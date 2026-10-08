"""Round-six atomic-save, short-lock and label-device regressions; synthetic."""
import fcntl
from contextlib import ExitStack
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_kadid_terminal_read as fixtures
from test_kadid_terminal_bound_payloads import _Watch, _zero_events

owner = fixtures.owner


class LedgerTransactions(unittest.TestCase):
    def setUp(self):
        self.t = fixtures.TerminalRead()
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def replace(self):
        f = owner.acceptance._DESTINATIONS.get()['ledger']
        f.seek(0)
        text = f.read()
        replacement = self.t.root / 'atomic-save.md'
        replacement.write_text(text)
        os.replace(replacement, self.t.ledger)
        self.assertEqual(os.fstat(f.fileno()).st_nlink, 0)
        return f, text

    def test_rename_over_after_preflight_refuses_before_journal_or_labels(self):
        original = owner.preflight
        captured = []

        def boundary(*args):
            result = original(*args)
            captured.append(self.replace()[1])
            return result

        with patch.object(owner, 'preflight', boundary), _Watch(self.t.labels) as watch:
            with self.assertRaisesRegex(owner.TerminalReadError, 'exposure-refused'):
                self.t.run_read()
            _zero_events(self, watch, 'atomic-save-before-reservation-labels')
        self.assertEqual(self.t.ledger.read_text(), captured[0])
        self.assertFalse(self.t.journal.exists())
        self.assertFalse(self.t.output.exists())

    def test_rename_over_after_reservation_refuses_result_append(self):
        original = owner.reserve
        captured = []

        def boundary(*args):
            original(*args)
            captured.append(self.replace()[1])

        with patch.object(owner, 'reserve', boundary):
            with self.assertRaisesRegex(owner.TerminalReadError, 'exposure-refused'):
                self.t.run_read()
        self.assertIn('KADID-TERMINAL-SPENT:', captured[0])
        self.assertNotIn('D2 result', self.t.ledger.read_text())
        self.assertEqual(self.t.ledger.read_text(), captured[0])
        self.assertTrue(self.t.journal.exists())
        # A result file alone is not a completed execute; the append refused.
        self.assertTrue(self.t.output.exists())
        with self.assertRaisesRegex(owner.TerminalReadError, 'preflight-refused'):
            self.t.run_read()

    def test_rename_over_after_journal_before_write_keeps_design_spent(self):
        original = owner.acceptance.append_ledger
        captured = []

        def boundary(*args):
            captured.append(self.replace()[1])
            original(*args)

        with patch.object(owner.acceptance, 'append_ledger', boundary), _Watch(self.t.labels) as watch:
            with self.assertRaisesRegex(owner.TerminalReadError, 'exposure-refused'):
                self.t.run_read()
            _zero_events(self, watch, 'atomic-save-before-ledger-write-labels')
        self.assertTrue(self.t.journal.exists())
        self.assertFalse(self.t.output.exists())
        self.assertEqual(self.t.ledger.read_text(), captured[0])

    def test_rename_over_during_durable_write_is_detected_after_write(self):
        original = os.fsync
        captured = []

        def boundary(fd):
            state = owner.acceptance._DESTINATIONS.get()
            if state is not None and fd == state['ledger'].fileno():
                captured.append(self.replace()[1])
            original(fd)

        with patch('os.fsync', boundary), _Watch(self.t.labels) as watch:
            with self.assertRaisesRegex(owner.TerminalReadError, 'exposure-refused'):
                self.t.run_read()
            _zero_events(self, watch, 'atomic-save-during-reservation-labels')
        self.assertEqual(len(captured), 1)
        self.assertTrue(self.t.journal.exists())
        self.assertFalse(self.t.output.exists())

    def lock_probe(self):
        with self.t.ledger.open('r+') as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(other, fcntl.LOCK_UN)

    def test_other_writer_can_lock_during_assessment_and_after_append(self):
        original = owner.assess
        observed = []

        def boundary(*args):
            self.lock_probe()
            observed.append('assessment')
            return original(*args)

        with patch.object(owner, 'assess', boundary):
            result = self.t.run_read()
        self.assertEqual(result['confirmation'], 'PASS')
        self.assertEqual(observed, ['assessment'])
        self.lock_probe()
        self.assertIn('D2 result', self.t.ledger.read_text())

    def test_exception_releases_lock_before_descriptor_close(self):
        with owner.acceptance.ledger_file(self.t.ledger, Path.open):
            with self.assertRaises(BlockingIOError):
                self.lock_probe()
        # Exercise failure with a retained descriptor, so close cannot be
        # mistaken for the explicit LOCK_UN that the assessment needs.
        with ExitStack() as stack, owner.acceptance.destinations(stack):
            with self.t.ledger.open('r+') as retained:
                state = owner.acceptance._DESTINATIONS.get()
                state.update(ledger=retained, ledger_path=self.t.ledger)
                with self.assertRaisesRegex(RuntimeError, 'synthetic abort'):
                    with owner.acceptance.ledger_file(self.t.ledger, Path.open):
                        raise RuntimeError('synthetic abort')
                self.lock_probe()

    def test_labels_device_itself_refuses_before_label_open(self):
        self.t.label_device_patch.stop()
        # Corpus stand-in is distinct; only the actual label file's device
        # now makes the synthetic preparation layout invalid.
        self.assertNotEqual(self.t.root.stat().st_dev, Path('/proc').stat().st_dev)
        with _Watch(self.t.labels) as watch:
            with self.assertRaisesRegex(owner.TerminalReadError, 'preflight-refused'):
                self.t.run_read()
            _zero_events(self, watch, 'actual-label-device-preflight-refusal')
        self.assertFalse(self.t.journal.exists())
        self.assertFalse(self.t.output.exists())

    def test_device_scope_does_not_leak_into_next_read_only_preflight(self):
        self.t.label_device_patch.stop()
        with owner.acceptance.device_scope():
            owner.acceptance.protect_device(self.t.root.stat().st_dev)
            self.assertIn(self.t.root.stat().st_dev, owner.acceptance.protected_devices())
        self.assertEqual(tuple(owner.acceptance.protected_devices()), ())
