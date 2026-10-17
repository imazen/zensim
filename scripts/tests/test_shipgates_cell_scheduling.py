"""A slow cell must not leave the other bounded workers idle at a quality barrier."""
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
from shipgates import bounded_cell_records


class SchedulingTests(unittest.TestCase):
    def test_next_quality_can_release_a_slow_cell_without_changing_journal_order(self):
        next_quality_started = threading.Event()
        first_cell_started = threading.Event()
        guard = threading.Lock()
        active = 0
        peak = 0
        def encode(cell):
            nonlocal active, peak
            quality, _distance, arm = cell
            with guard:
                active += 1
                peak = max(peak, active)
            try:
                if quality == 0 and arm == "slow":
                    first_cell_started.set()
                    self.assertTrue(next_quality_started.wait(5), "per-quality barrier left a worker idle")
                if quality == 0 and arm == "fast":
                    self.assertTrue(first_cell_started.wait(5), "first worker did not start")
                if quality == 1:
                    next_quality_started.set()
                return quality, arm
            finally:
                with guard:
                    active -= 1
        rows = list(bounded_cell_records([(0, 0.1), (1, 0.2)], ["slow", "fast"], 2, True, encode))
        self.assertEqual(rows, [(0, "slow"), (0, "fast"), (1, "slow"), (1, "fast")])
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)


if __name__ == "__main__":
    unittest.main()
