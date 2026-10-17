"""Exercise exact run-heavy admission refusals without allocating heavy memory."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/"rev4_featpot/shipgates_admission.py"
spec=importlib.util.spec_from_file_location("memory_admission",SOURCE)
owner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class MemoryAdmissionTests(unittest.TestCase):
    def scratch(self):
        parent=Path.home()/"tmp"
        parent.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="shipgates-admission-",dir=parent)

    def test_unstarted_refusal_waits_and_preserves_both_attempt_logs(self):
        with self.scratch() as directory:
            root=Path(directory)
            command=[sys.executable,"-c",'from pathlib import Path; import sys; marker=Path(sys.argv[1]); output=Path(sys.argv[2]); first=not marker.exists(); marker.touch(); print("run-heavy: only 3133 MiB available — refusing to launch (free memory first)" if first else "executed"); output.touch() if not first else None; sys.exit(3 if first else 0)',str(root/"attempted"),str(root/"output")]
            with patch.object(owner.time,"sleep") as sleep:
                owner.run_with_memory_admission(command,root/"phase.log",root/"output")
                sleep.assert_called_once_with(30)
            self.assertTrue((root/"output").exists())
            self.assertIn("only 3133 MiB",(root/"phase.log").read_text())
            self.assertEqual((root/"phase.memory-admission-001.log").read_text(),"executed\n")

    def test_encoder_failure_is_fatal_even_with_same_exit_status(self):
        with self.scratch() as directory:
            root=Path(directory)
            command=[sys.executable,"-c",'print("codec failed after launch"); raise SystemExit(3)']
            with patch.object(owner.time,"sleep") as sleep:
                with self.assertRaises(subprocess.CalledProcessError):
                    owner.run_with_memory_admission(command,root/"phase.log",root/"output")
                sleep.assert_not_called()

    def test_populated_output_prevents_automatic_reexecution(self):
        with self.scratch() as directory:
            root=Path(directory)
            (root/"output").touch()
            command=[sys.executable,"-c",'print("run-heavy: only 3133 MiB available — refusing to launch (free memory first)"); raise SystemExit(3)']
            with patch.object(owner.time,"sleep") as sleep:
                with self.assertRaisesRegex(AssertionError,"populated output"):
                    owner.run_with_memory_admission(command,root/"phase.log",root/"output")
                sleep.assert_not_called()


if __name__=="__main__":
    unittest.main()
