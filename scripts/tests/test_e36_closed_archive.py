"""Closed inventory covers old CAS at high free space and rejects unreadable inputs."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_closed_archive as closed
import e36_artifact_mirror as mirror
from e36_store import sha


class ClosedInventory(unittest.TestCase):
    def fixture(self,d):
        root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir();terminal=owner/'native-continuation-final-v1.json';terminal.write_text('{}');local=root/'artifacts-local-v1';raw=b'native encoded bundle';h=hashlib.sha256(raw).hexdigest();path=local/'blobs'/h[:2]/(h+'.tar');path.parent.mkdir(parents=True);path.write_bytes(raw)
        doc=dict(schema='e36-native-ladder-receipt-v1',complete=True,rows=[dict(bundle=dict(host='dev',root=str(local),sha256=h,bytes=len(raw),relative_path=str(path.relative_to(local))))]);data=gzip.compress(json.dumps(doc).encode());v=hashlib.sha256(data).hexdigest();blobs=root/'runs/fallback-v6/blobs';blobs.mkdir(parents=True);cas=blobs/v;cas.write_bytes(data);os.utime(cas,(1,1));ledger=blobs.parent/'ledger.parquet';pq.write_table(pa.Table.from_pylist([dict(status='done',attempts=1,output_sha=v)]),ledger)
        return root,owner,terminal,ledger,h,path

    def test_old_eligible_cas_is_not_cut_off(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root,owner,terminal,ledger,h,path=self.fixture(d)
            with patch.object(closed,'closed_generation',return_value=(terminal,[ledger])):queue,proof=closed.inventory(root)
            self.assertIn(h,queue);self.assertEqual(proof['published_views'],1);self.assertEqual(proof['local_eligible_bytes'],path.stat().st_size)

    def test_unreadable_terminal_ledger_refuses_instead_of_skipping(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root,owner,terminal,ledger,h,path=self.fixture(d);ledger.write_bytes(b'interrupted parquet')
            with patch.object(closed,'closed_generation',return_value=(terminal,[ledger])):
                with self.assertRaises(pa.ArrowInvalid):closed.inventory(root)
            self.assertTrue(path.exists())

    def test_high_free_no_urgent_flag_still_inventories_before_completion(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root,owner,terminal,ledger,h,path=self.fixture(d);(root/'storage-runtime.json').write_text(json.dumps(dict(root='/legacy',host='tower',program_sha256='a'*64,command=['unused'])))
            seen=[]
            class Transfer:
                pending={};recovery=[]
                def __init__(self,*a):pass
                def cycle(self,queue,priority):seen.append(set(queue));queue.clear();priority.clear();return []
                def drain(self,queue):pass
            with patch.object(mirror,'ROOT',root),patch.object(mirror,'LOCAL',root/'artifacts-local-v1'),patch.dict(os.environ,{'E36_ARTIFACTS_ONLY':'1'},clear=False),patch('e36_mirror_dispatch.Dispatcher',Transfer),patch.object(closed,'closed_generation',return_value=(terminal,[ledger])),patch.object(mirror.shutil,'disk_usage',return_value=type('Space',(),dict(free=100*1024**3))()),patch.object(mirror.time,'time',return_value=0):mirror.run(None)
            self.assertEqual(seen,[{h}]);stop=json.loads((owner/'mirror-stop.json').read_text());self.assertIn('inventory',stop);self.assertTrue(Path(stop['inventory']['path']).exists())


if __name__=='__main__':unittest.main()
