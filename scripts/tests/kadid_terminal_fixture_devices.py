"""Distinct-device stand-ins for synthetic payload/statistics fixtures only."""
from pathlib import Path
from unittest.mock import patch


def configure(test, owner):
    # Fixtures live together on one disk and read no real corpus. /proc
    # models a distinct protected device for the payload/statistic tests.
    # Dedicated hardening tests use real same-device roots, and the label
    # device regression removes this stand-in before exercising refusal.
    test.corpus_patch = patch.object(owner, "CORPUS_ROOTS", (Path('/proc'),))
    protect = owner.acceptance.protect_device
    test.label_device_patch = patch.object(
        owner.acceptance, "protect_device",
        side_effect=lambda _: protect(Path('/proc').stat().st_dev),
    )
    for guard in (test.corpus_patch, test.label_device_patch):
        guard.start()
        test.addCleanup(guard.stop)
