"""Invented resource/admission and transport failures; no encodes or fits."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_disk_policy as policy
import e36_artifact_move as movement
from e36_store import sha


class DiskPolicy(unittest.TestCase):
    def test_restart_hysteresis_and_each_low_space_observer_latches_stop(self):
        self.assertTrue(policy.paused_state(35*1024**3,True,False))
        self.assertTrue(policy.paused_state(40*1024**3,True,False))
        self.assertFalse(policy.paused_state(41*1024**3,True,False))
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-hard-stop-') as d:
            root=Path(d);(root/'owner-svt-ssd-v1').mkdir()
            with patch.object(policy,'ROOT',root),patch.object(policy.shutil,'disk_usage',return_value=types.SimpleNamespace(free=19*1024**3)):
                with self.assertRaises(OSError):policy.disk20(root)
            marker=root/'owner-svt-ssd-v1/DISK20_STOP.json';original=marker.read_bytes()
            with patch.object(policy,'ROOT',root),patch.object(policy.shutil,'disk_usage',return_value=types.SimpleNamespace(free=50*1024**3)):
                with self.assertRaises(OSError):policy.disk20(root)
            self.assertEqual(marker.read_bytes(),original)

    def test_pause30_waits_for_above40_and_sticky20_requires_explicit_release(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-disk-policy-') as d:
            root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir();state=owner/'disk-claim-state.json';state.write_text('{"claims_paused":false}')
            free=iter([29,35,40,41])
            with patch.object(policy,'ROOT',root),patch.object(policy,'STATE',state),patch.object(policy.shutil,'disk_usage',side_effect=lambda p:types.SimpleNamespace(free=next(free)*1024**3)),patch.object(policy.time,'sleep') as slept:
                policy.wait(1e20);self.assertEqual(slept.call_count,3)
            (owner/'DISK20_STOP.json').write_text('{}')
            with patch.object(policy,'ROOT',root),patch.object(policy,'STATE',state),patch.object(policy.shutil,'disk_usage',return_value=types.SimpleNamespace(free=50*1024**3)),patch.object(policy.time,'time',return_value=10):
                with self.assertRaisesRegex(TimeoutError,'disk admission wait'):policy.wait(9)

    def test_parallel_transport_keeps_unverified_local_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-mirror-transport-') as d:
            root=Path(d);(root/'owner-svt-ssd-v1').mkdir();storage=dict(command=['never-executed'],program_sha256='program',host='tower',root='canonical')
            paths={}
            for data in [b'first immutable bundle',b'second immutable bundle']:
                temporary=root/str(len(paths));temporary.write_bytes(data);digest=sha(temporary)
                path=root/'artifacts-local-v1/blobs'/digest[:2]/(digest+'.tar');path.parent.mkdir(parents=True,exist_ok=True);temporary.rename(path);paths[digest]=path
            remote={}
            class Client:
                def __init__(self,*args):pass
                def put(self,p):h=sha(p);remote[h]=p.read_bytes();return dict(bytes=len(remote[h]))
                def verify(self,h):return dict(bytes=len(remote[h]),relative_path='blobs/'+h[:2]+'/'+h+'.tar',program_sha256='program')
            with patch.object(movement,'Client',Client):
                mover=movement.Mover(root,storage)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    result=list(pool.map(lambda item:mover.move(item[0],item[1],'native-view'),paths.items()))
            self.assertEqual(len(result),2);self.assertTrue(all(not p.exists() for p in paths.values()))
            rows=[json.loads(l) for l in (root/'owner-svt-ssd-v1/artifact-relocations.jsonl').read_text().splitlines()];self.assertEqual({r['sha256'] for r in rows},set(paths))
            h=next(iter(paths));paths[h].write_bytes(remote[h])
            class Failed(Client):
                def verify(self,h):return dict(bytes=-1)
            with patch.object(movement,'Client',Failed):
                with self.assertRaisesRegex(ValueError,'verification failed'):movement.Mover(root,storage).move(h,paths[h],'native-view')
            self.assertTrue(paths[h].exists())


if __name__=='__main__':unittest.main()
