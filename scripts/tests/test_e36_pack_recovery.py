"""Exercise interrupted archival ownership and atomic commit boundaries."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from e36_archive_assignments import Assignments
from e36_artifact_location import relocation_records
from e36_pack_archive import Dispatcher,publish_segment,atomic_json


class Recovery(unittest.TestCase):
    def test_atomic_segment_ignores_only_unpublished_partials(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            owner=Path(d)/'owner-svt-ssd-v1';owner.mkdir();rows=[dict(sha256='a'*64,bytes=10)]
            path=publish_segment(owner,rows);(path.parent/'interrupted.partial').write_text('{"incomplete":')
            self.assertEqual(list(relocation_records(Path(d))),rows)
            path.write_text(path.read_text()+'corruption')
            with self.assertRaisesRegex(ValueError,'changed'):list(relocation_records(Path(d)))

    def test_atomic_assignment_replays_and_refuses_second_owner(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'owner-svt-ssd-v1').mkdir();index=Assignments(root)
            row=dict(sha256='a'*64,canonical_host='tower',canonical_root='/array',program_sha256='b'*64,pack_id='c'*64)
            index.seed_pack([row]);self.assertEqual(Assignments(root).get(row['sha256']),row)
            with self.assertRaisesRegex(ValueError,'destination'):Assignments(root).seed_pack([{**row,'canonical_root':'/cache'}])
            with self.assertRaisesRegex(ValueError,'different pack'):Assignments(root).seed_pack([{**row,'pack_id':'d'*64}])

    def test_recovery_assigns_before_submit_and_refuses_cache_without_headroom(self):
        dispatch=Dispatcher.__new__(Dispatcher);dispatch.owner=Path('/unused');dispatch.pending={};dispatch.recovery=[dict(storage=dict(root='/cache'),reserved_bytes=100,items=[])];dispatch.cache=dict(root='/cache');dispatch.validate_intent=Mock();dispatch.assigned_batch=Mock();dispatch.exists=Mock(return_value=False);dispatch.fresh_cache=Mock(return_value=False);dispatch.pool=Mock()
        with patch('e36_pack_archive.Path.exists',return_value=False):self.assertEqual(dispatch.cycle({},[]),[])
        dispatch.assigned_batch.assert_called_once();dispatch.fresh_cache.assert_called_once_with(100,recovery=True);dispatch.pool.submit.assert_not_called();self.assertEqual(len(dispatch.recovery),1)

    def test_published_array_recovery_needs_no_local_bundle_files(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir();intents=owner/'pack-intents-v1';intents.mkdir()
            dispatch=Dispatcher.__new__(Dispatcher);dispatch.root=root;dispatch.owner=owner;dispatch.cache=dict(root='/cache');dispatch.intents=intents;dispatch.journal_lock=threading.Lock();dispatch.verifier='v'*64;dispatch.exists=Mock(return_value=True)
            item=dict(sha256='a'*64,bytes=123,view_sha256='b'*64,local_path=str(root/'already-evicted.tar'))
            storage=dict(root='/array',host='tower',program_sha256='c'*64,container_id='fixture');doc=dict(pack_id='d'*64,items=[item],storage=storage,reserved_bytes=10240)
            actual=dict(members={'a'*64:123},stored_sha256='e'*64,stored_bytes=10240,verifier_sha256=dispatch.verifier)
            with patch('e36_pack_archive.subprocess.check_output',return_value=json.dumps(actual).encode()),patch('e36_pack_archive.subprocess.Popen',side_effect=AssertionError('must not upload')):
                rows=dispatch.pack(doc)
            self.assertEqual(rows[0]['sha256'],item['sha256']);self.assertEqual(list(relocation_records(root)),rows);self.assertTrue((intents/('d'*64+'.complete')).exists())

    def test_below_floor_is_sticky_for_pack_restart(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            dispatch=Dispatcher.__new__(Dispatcher);dispatch.owner=Path(d);dispatch.pending={};dispatch.used=0;dispatch.cache=dict(root='/cache');dispatch.cache_active=True
            with patch('e36_pack_archive.subprocess.check_output',side_effect=['human\n','Avail\n'+str(179*1024**3)+'\n']):self.assertFalse(dispatch.fresh_cache(1))
            self.assertTrue((Path(d)/'CACHE180_STOP.json').exists())


if __name__=='__main__':unittest.main()

class ExceptionalDrain(unittest.TestCase):
    def test_hold_failure_keeps_same_array_child_until_join(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir();intents=owner/'pack-intents-v1';intents.mkdir()
            dispatch=Dispatcher.__new__(Dispatcher);dispatch.root=root;dispatch.owner=owner;dispatch.cache=dict(root='/cache');dispatch.intents=intents;dispatch.journal_lock=threading.Lock();dispatch.verifier='v'*64;dispatch.exists=Mock(return_value=False)
            item=dict(sha256='a'*64,bytes=123,view_sha256='b'*64,local_path=str(root/'missing'))
            doc=dict(pack_id='d'*64,items=[item],storage=dict(root='/array',host='tower',program_sha256='c'*64,container_id='fixture'),reserved_bytes=10240)
            child=Mock();child.communicate.side_effect=[subprocess.TimeoutExpired('owned',30),(b'',None)]
            with patch('e36_pack_archive.subprocess.Popen',return_value=child) as launch,patch('e36_pack_archive.build',side_effect=ValueError('source failure')),patch('e36_generation_resume.hold_admission',side_effect=OSError('hold unavailable')):
                with self.assertRaisesRegex(ValueError,'source failure'):dispatch.pack(doc)
            self.assertEqual(launch.call_count,1);self.assertEqual(child.communicate.call_count,2);child.terminate.assert_not_called();child.kill.assert_not_called()

    def test_new_segment_directory_is_durable_before_unlink_boundary(self):
        from e36_artifact_location import durable_mkdir
        import e36_pack_archive
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            owner=Path(d)/'owner';owner.mkdir();calls=[]
            def sync(path):calls.append(Path(path))
            with patch('e36_artifact_location.sync_directory',side_effect=sync):path=publish_segment(owner,[dict(sha256='a'*64)])
            self.assertIn(owner,calls);self.assertIn(path.parent,calls);self.assertTrue(path.exists())

class DirectoryAdmission(unittest.TestCase):
    def test_missing_false_or_wrong_owner_refuses(self):
        from e36_artifact_location import require_pack_directory_proof
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir()
            with self.assertRaises(FileNotFoundError):require_pack_directory_proof(root)
            proof=owner/'pack-directory-durability-v1.json';proof.write_text(json.dumps(dict(admitted=False,stores=[])))
            with self.assertRaises(ValueError):require_pack_directory_proof(root)
            rows=[]
            for name in ['storage-cache-v1.json','storage-archive-v2.json']:
                (root/name).write_text(json.dumps(dict(root='/'+name,container_id=name)))
                rows.append(dict(storage=name,root='/'+name,container_id=name,actual=dict(fsynced=True,directories=['/data','/data/packs','/data/.pack-incoming'])))
            proof.write_text(json.dumps(dict(admitted=True,stores=rows)));self.assertEqual(require_pack_directory_proof(root),proof)
            rows[0]['container_id']='other';proof.write_text(json.dumps(dict(admitted=True,stores=rows)))
            with self.assertRaises(ValueError):require_pack_directory_proof(root)
