"""Cache floor, cumulative bound and independent transport-batch fixtures."""
import heapq
import json
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import patch,Mock

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_cache_archive import CachePolicy
from e36_mirror_dispatch import Dispatcher


class Cache(unittest.TestCase):
    def storage(self):return dict(authorization='coordinator-2026-10-17-cache250GB',cache_budget_bytes=250000000000,cache_reserve_bytes=180*1024**3,root='/cache')
    def fixture(self,d):
        root=Path(d);(root/'owner-svt-ssd-v1').mkdir();return root

    def test_below_floor_sticks_and_no_cache_claim_without_check(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=self.fixture(d);policy=CachePolicy(root,self.storage())
            with self.assertRaisesRegex(ValueError,'observation'):policy.claim('a'*64,100)
            with patch('e36_cache_archive.subprocess.check_output',side_effect=['human df\n','Avail\n'+str(179*1024**3)+'\n']) as remote:
                self.assertFalse(policy.before_batch());self.assertFalse(policy.before_batch());self.assertEqual(remote.call_count,2)
            self.assertTrue((root/'owner-svt-ssd-v1/CACHE180_STOP.json').exists())

    def test_reservations_survive_restart_and_cannot_exceed250GB(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=self.fixture(d);policy=CachePolicy(root,self.storage())
            with patch('e36_cache_archive.subprocess.check_output',side_effect=['human\n','Avail\n'+str(430*1024**3)+'\n']):self.assertTrue(policy.before_batch())
            self.assertTrue(policy.claim('a'*64,1024**2));bound=policy.total
            self.assertGreaterEqual(bound,1024**2)
            restarted=CachePolicy(root,self.storage());self.assertEqual(restarted.total,bound)
            with self.assertRaisesRegex(ValueError,'observation'):restarted.claim('a'*64,1024**2)
            restarted.epoch=dict(free_bytes=430*1024**3,batch_reserved_bytes=0,id='fresh')
            self.assertTrue(restarted.claim('a'*64,1024**2));self.assertEqual(restarted.total,bound)
            restarted.total=249999999999;restarted.epoch=policy.epoch
            self.assertFalse(restarted.claim('b'*64,1024**2))

    def test_restarted_uploads_charge_batch_headroom_without_double_quota(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=self.fixture(d);policy=CachePolicy(root,self.storage())
            policy.epoch=dict(free_bytes=430*1024**3,batch_reserved_bytes=0,id='old')
            self.assertTrue(policy.claim('a'*64,1024**2));self.assertTrue(policy.claim('b'*64,1024**2));total=policy.total
            restarted=CachePolicy(root,self.storage());bound=restarted.reserved['a'*64]
            restarted.epoch=dict(free_bytes=180*1024**3+bound+1,batch_reserved_bytes=0,id='fresh')
            self.assertTrue(restarted.claim('a'*64,1024**2));self.assertFalse(restarted.claim('b'*64,1024**2))
            self.assertEqual(restarted.total,total);self.assertEqual(restarted.epoch['batch_reserved_bytes'],bound)

    def test_cache_refills_while_array_batch_is_pending(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=self.fixture(d);(root/'storage-cache-v1.json').write_text(json.dumps(dict(root='/cache',host='tower',program_sha256='b'*64)))
            policy=Mock();policy.before_batch.return_value=True;policy.claim.return_value=True;policy.locations={};policy.excluded=set();policy.reserved={}
            movers=[]
            def mover(*args):
                value=Mock();value.journal_lock=object();value.storage=args[1];movers.append(value);return value
            submitted=[]
            class Pool:
                def submit(self,fn,*args):
                    future=Future();submitted.append((fn,args,future));return future
            with patch('e36_mirror_dispatch.Mover',side_effect=mover),patch('e36_mirror_dispatch.CachePolicy',return_value=policy):dispatch=Dispatcher(root,Pool(),dict(root='/array',host='tower',program_sha256='a'*64))
            queue={};priority=[]
            for i in range(16):
                digest=f'{i:064x}';path=root/(digest+'.tar');path.write_bytes(b'x');queue[digest]=(path,'view');heapq.heappush(priority,(-1,digest))
            dispatch.cycle(queue,priority);self.assertEqual(len(dispatch.pending),8)
            for fn,args,future in submitted[:4]:future.set_result(dict(sha256=args[0],bytes=1,canonical_root='/cache'))
            dispatch.cycle(queue,priority)
            self.assertEqual(len(submitted),12);self.assertEqual(len(dispatch.pending),8)
            self.assertTrue(all(not future.done() for _,_,future in submitted[4:8]))
            self.assertTrue(all(fn==movers[1].move for fn,_,_ in submitted[8:]))


if __name__=='__main__':unittest.main()
