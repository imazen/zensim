"""No authorization means no subprocess or output, even without other artifacts."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shippath10_launch as launch


class LaunchGate(unittest.TestCase):
    def test_missing_authorization_has_zero_effects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(launch.subprocess,'run',side_effect=AssertionError('subprocess before authorization')), \
                 patch.object(launch.subprocess,'check_output',side_effect=AssertionError('subprocess before authorization')):
                for jobset in ('fitv2e30-20261007','fitv2d1-20261007'):
                    with self.assertRaisesRegex(PermissionError,'NOT AUTHORIZED'):
                        launch.launch(root,jobset)
                    self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':
    unittest.main()
