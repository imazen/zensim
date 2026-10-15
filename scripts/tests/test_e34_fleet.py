"""The owner-confirmed reclaimed hosts must never reach SSH."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import e34_fleet as fleet


class ReclaimedHosts(unittest.TestCase):
    def test_inventory_and_direct_remote_refuse_connections(self):
        with patch.object(fleet, 'gone_hosts', return_value={'i134', 'r5600g'}), \
             patch.object(fleet.subprocess, 'run') as ssh:
            for host in ['i134', 'r5600g']:
                self.assertEqual(fleet.inventory_host(host)['state'], 'GONE')
                with self.assertRaisesRegex(AssertionError, 'owner forbids connecting'):
                    fleet.remote(host, ['uname', '-s'])
            ssh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
