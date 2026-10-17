"""Invented controller binding and stale-census refusal, without process control."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "rev4_featpot"))
sys.path.insert(0, str(Path(__file__).parent))
import e36_census_install as installer
import test_e36_census_stream as fixtures


class Install(unittest.TestCase):
    def test_live_owner_is_rebound_and_existing_census_is_rechecked(self):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-census-live-"
        ) as d:
            root = Path(d)
            directory = fixtures.Census().fixture(root)
            control = root / "control-resume-v7"
            control.mkdir()
            source = control / "e36_bulk.py"
            source.write_text(
                'def census(*args):\n raise RuntimeError("unrebound")\ndef qualify(*args):\n return True\n'
            )
            spec = importlib.util.spec_from_file_location(
                "invented_original_census", source
            )
            owner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(owner)
            clock = root / "bulk-clock-v6-budget2.json"
            clock.write_text("{}")
            path = control / "e36_resume_schedule.py"
            text = 'def run(root,owner):\n install(root,receipt,expected,123,root/"paper.lock")\n'
            path.write_text(text)
            scope = dict(
                install=installer.install_current,
                receipt=root / "binding.json",
                expected=installer.sha(source),
            )
            exec(compile(text, str(path), "exec"), scope)
            with patch.object(installer, "supervisor_lock") as locks:
                scope["run"](root, owner)
                self.assertTrue(owner.qualify(root, "invented", 0.0, 1.0))
                self.assertEqual(
                    [c.args[2] for c in locks.call_args_list], [8, 2, 8, 2]
                )
                manifest = directory / "manifest.json"
                manifest.write_text("[]")
                self.assertFalse(owner.qualify(root, "invented", 0.0, 1.0))
            self.assertTrue(list(directory.glob("census.retained-*.json")))
            self.assertEqual(clock.read_text(), "{}")
            proof = json.loads((root / "binding.json").read_text())
            self.assertFalse(proof["clock_changed"])


if __name__ == "__main__":
    unittest.main()
