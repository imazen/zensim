#!/usr/bin/env python3
"""Invented completion markers; no model, pixels or real timing required."""
import io
import json
from pathlib import Path
import tempfile
import unittest

from mapperf_lock_window import completed, hold


class LockWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = ['invented-a', 'invented-c']
        self.geometries = ['64x64', '1024x1024']
        (self.root / 'JOB.json').write_text(json.dumps(dict(
            models=dict.fromkeys(self.models, {}), geometries=self.geometries)))
        self.now = 0

    def marker(self, model, geometry, text='{"status":"PASS"}', suffix=''):
        path = self.root / 'results' / model / 'timing' / f'v4x-t1-{geometry}{suffix}' / 'COMPLETE.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def advance(self, seconds):
        self.now += seconds

    def test_partial_and_archived_markers_do_not_release_window(self):
        self.marker(self.models[0], self.geometries[0], text='{"status":')
        self.marker(self.models[1], self.geometries[1], suffix='.noise-invented.bak')
        self.assertEqual(completed(self.root, self.models, self.geometries), set())

    def test_expiry_does_not_create_measurements(self):
        log = io.StringIO()
        self.assertEqual(hold(self.root, 2, log, lambda: self.now, self.advance), 'window_expired')
        self.assertEqual(self.now, 2)
        self.assertFalse((self.root / 'results').exists())
        self.assertEqual(json.loads(log.getvalue().splitlines()[-1])['completed_cells'], [])

    def test_release_on_new_cell_preserves_existing_receipt(self):
        original = self.marker(self.models[0], self.geometries[0])
        before = original.read_bytes()

        def advance_and_complete(seconds):
            self.advance(seconds)
            self.marker(self.models[1], self.geometries[1])

        log = io.StringIO()
        self.assertEqual(hold(self.root, 2, log, lambda: self.now, advance_and_complete), 'one_cell_completed')
        self.assertEqual(self.now, 1)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(len(json.loads(log.getvalue().splitlines()[-1])['completed_cells']), 2)

    def test_already_complete_grid_releases_immediately(self):
        for model in self.models:
            for geometry in self.geometries:
                self.marker(model, geometry)

        def forbidden_sleep(_):
            self.fail('complete grid must not hold the resource window')

        self.assertEqual(hold(self.root, 2, io.StringIO(), lambda: self.now, forbidden_sleep), 'complete')


if __name__ == '__main__':
    unittest.main()
