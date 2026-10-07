"""Legacy evaluation-panel CLIs must refuse import before reading any data (2026-10-07 E29 incident class)."""
import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
MODULES = ["scripts/hdrgrid372_val_read.py", "scripts/hdrgrid944_val_read.py", "scripts/hdrp1_val_read.py",
           "scripts/hdr/upiq_panel.py", "scripts/v_next/hidden_terminal_read.py", "scripts/hdr/hdr_route_panel.py"]


class ImportGuards(unittest.TestCase):
    def test_import_refuses_before_any_data_read(self):
        for rel in MODULES:
            with self.subTest(module=rel):
                reads = []
                real_open = builtins.open

                def tripwire_open(path, *a, **k):
                    if not str(path).endswith(".py"):
                        reads.append(str(path))
                    return real_open(path, *a, **k)

                def tripwire_read_table(path, *a, **k):
                    reads.append(str(path))
                    raise AssertionError("panel read during import")

                spec = importlib.util.spec_from_file_location("guarded_cli_under_test", REPO / rel)
                module = importlib.util.module_from_spec(spec)
                with patch.object(builtins, "open", side_effect=tripwire_open), \
                        patch.object(pq, "read_table", side_effect=tripwire_read_table), \
                        patch("sys.argv", ["test_runner", "fake.bin"]):
                    try:
                        spec.loader.exec_module(module)
                    except ImportError:
                        pass  # the guard (or hdr_route_panel's own main guard path)
                self.assertEqual(reads, [], f"{rel} read data at import")


if __name__ == "__main__":
    unittest.main()
