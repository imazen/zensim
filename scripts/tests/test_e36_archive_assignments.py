"""Crash-window destination recovery and actual finalizer cache-path coverage."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_archive_assignments import Assignments
from e36_artifact_move import Mover
from e36_artifact_location import freeze
from e36_store import sha


class Assignment(unittest.TestCase):
    def test_array_publication_survives_failed_journal_before_cache_restart(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir();artifact=root/'raw';artifact.write_bytes(b'raw native bundle fixture');digest=sha(artifact);remote=root/'remote'
            storage=dict(root='/array',host='tower',program_sha256='a'*64,command=['fixture'])
            class Remote:
                def put(self,path):remote.write_bytes(path.read_bytes());return self.verify(digest)
                def verify(self,value):
                    if sha(remote)!=value:raise ValueError('wrong remote bytes')
                    return dict(ok=True,sha256=value,bytes=remote.stat().st_size,relative_path='blobs/'+value[:2]+'/'+value+'.tar',program_sha256='a'*64)
            mover=Mover(root,storage);mover.assignments=Assignments(root)
            original=Path.open
            def opening(path,*args,**kwargs):
                if path==owner/'artifact-relocations.jsonl' and args and args[0]=='a':raise OSError('journal failed after publication')
                return original(path,*args,**kwargs)
            with patch('e36_artifact_move.Client',return_value=Remote()),patch.object(Path,'open',opening):
                with self.assertRaisesRegex(OSError,'journal failed'):mover.move(digest,artifact,'native-view')
            self.assertTrue(artifact.exists());self.assertEqual(remote.read_bytes(),artifact.read_bytes())
            cache=Mover(root,{**storage,'root':'/cache'});cache.assignments=Assignments(root)
            with patch('e36_artifact_move.Client') as client:
                with self.assertRaisesRegex(ValueError,'second archive destination'):cache.move(digest,artifact,'native-view')
                client.assert_not_called()

    def test_finalizer_cache_branch_with_real_native_receipt_and_mock_remote(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);owner=root/'owner-svt-ssd-v1';owner.mkdir();(owner/'mirror-stop.json').write_text('{}')
            run=root/'runs/fallback-v6';(run/'blobs').mkdir(parents=True)
            digest='c'*64;relative='blobs/cc/'+digest+'.tar.zst';legacy=dict(host='tower',root='/legacy',program_sha256='a'*64,command=['legacy'])
            cache=dict(host='tower',root='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache',program_sha256='b'*64,command=['cache'],encoding='zstd',authorization='coordinator-2026-10-17-cache250GB',cache_budget_bytes=250000000000,cache_reserve_bytes=180*1024**3)
            (root/'storage-runtime.json').write_text(json.dumps(legacy));(root/'storage-cache-v1.json').write_text(json.dumps(cache))
            bundle=dict(sha256=digest,bytes=123,host='dev',root=str(root/'artifacts-local-v1'))
            data=gzip.compress(json.dumps(dict(schema="e36-native-ladder-receipt-v1",complete=True,rows=[dict(bundle=bundle)])).encode(),mtime=0);path=run/'blobs/temporary';path.write_bytes(data);view=sha(path);path.rename(path.with_name(view))
            pq.write_table(pa.Table.from_pylist([dict(status='done',attempts=1,output_sha=view)]),run/'ledger.parquet')
            row=dict(sha256=digest,bytes=123,canonical_host='tower',canonical_root=cache['root'],relative_path=relative,program_sha256='b'*64,encoding='zstd',stored_sha256='d'*64,stored_bytes=111,sha_verified=True,completed_view_sha256=view)
            (owner/'artifact-relocations.jsonl').write_text(json.dumps(row)+'\n');(owner/'cache-reservations-v1.jsonl').write_text(json.dumps(dict(sha256=digest,reserved_bytes=1000))+'\n')
            class Remote:
                process=None
                def __init__(self,*args):pass
                def verify(self,value):return dict(ok=True,sha256=digest,bytes=123,relative_path=relative,program_sha256='b'*64,encoding='zstd',stored_sha256='d'*64,stored_bytes=111)
                def close(self):pass
            with patch('e36_store.Client',Remote),patch('e36_archive_store.ArchiveClient',Remote):result=freeze(root,root/'admission.json')
            self.assertEqual(result['cache_reserved_bytes'],1000);self.assertEqual(len(result['actual_random_readbacks']),1)
            self.assertEqual(result['cache_reservations']['path'],str(owner/'cache-reservations-v1.jsonl'))


if __name__=='__main__':unittest.main()
