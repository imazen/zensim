"""Invented artifacts and native rows; no labels, optimizer or codec execution."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_artifact_location as location
import e36_owner_amendment as overlay
import e36_svt_matrix as matrix
from e36_store import sha


class Storage(unittest.TestCase):
    def test_local_hardlink_ack_excludes_rav1e_and_keeps_legacy_owner(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-owner-') as d:
            root=Path(d);local=root/'artifacts-local-v1'
            storage=dict(host='tower',root='original-store',program_sha256='server')
            (root/'storage-runtime.json').write_text(json.dumps(storage))
            calls=[]
            def remote(h):calls.append(h);return dict(ok=True,sha256=h,bytes=3)
            runtime=types.SimpleNamespace(root=root,spec={'storage':storage.copy()},
                execute=lambda j:b'original',execute_cell=lambda *a,**k:'original',call=lambda *a:None,
                store=types.SimpleNamespace(verify=remote))
            with patch.object(overlay,'ROOT',root),patch.object(overlay,'LOCAL',local):
                overlay.install(runtime)
                self.assertEqual(runtime.spec['storage']['root'],'original-store')
                with self.assertRaisesRegex(ValueError,'EXCLUDED_RAV1E'):
                    runtime.execute({'kind':{'view':'e36-native-ladder-v1:{"codec":"rav1e"}'}})
                with self.assertRaisesRegex(ValueError,'EXCLUDED_RAV1E'):
                    runtime.execute_cell(None,'rav1e')
                self.assertEqual(runtime.execute({'kind':{'view':'e36-native-ladder-v1:{"codec":"svt"}'}}),b'original')
                artifact=root/'bundle.tar';artifact.write_bytes(b'abc')
                ack=runtime.store.put(artifact);artifact.unlink()
                self.assertEqual(runtime.store.verify(ack['sha256'])['bytes'],3)
                self.assertEqual(runtime.spec['storage']['host'],'dev')
                # A matching local hash must not override the old receipt owner.
                result=dict(bundle=dict(host='tower',root='original-store',sha256=ack['sha256'],bytes=3,relative_path=ack['relative_path']))
                runtime.store.verify(result['bundle']['sha256'])
                self.assertEqual(calls,[ack['sha256']])

    def test_missing_local_requires_verified_relocation_and_actual_destination(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-resolve-') as d:
            root=Path(d);(root/'owner-svt-ssd-v1').mkdir();h='1'*64
            bundle=dict(host='dev',root=str(root/'artifacts-local-v1'),sha256=h,bytes=7,relative_path='blobs/11/'+h+'.tar')
            storage=dict(host='tower',root='canonical',program_sha256='program')
            journal=root/'owner-svt-ssd-v1/artifact-relocations.jsonl'
            record=dict(sha256=h,bytes=7,canonical_host='tower',canonical_root='canonical',relative_path=bundle['relative_path'],program_sha256='program',sha_verified=True)
            journal.write_text(json.dumps(record)+'\n')
            pin=matrix.bind(journal);calls=[]
            def verify(value):calls.append(value);return dict(ok=True,sha256=value,bytes=7)
            actual=location.resolve(bundle,root,verify,storage,pin)
            self.assertEqual(actual['host'],'tower');self.assertEqual(calls,[h])
            with self.assertRaisesRegex(ValueError,'byte count'):
                location.resolve(bundle,root,lambda h:dict(bytes=8),storage,pin)
            journal.write_text(json.dumps(dict(record,canonical_root='elsewhere'))+'\n')
            with self.assertRaisesRegex(ValueError,'journal changed'):
                location.resolve(bundle,root,verify,storage,pin)


class NativePublication(unittest.TestCase):
    def fixture(self,root):
        old=root/'runs/old';new=root/'runs/svt-fallback-v1';old.mkdir(parents=True);new.mkdir()
        jobs=[dict(kind=dict(kind='bake',view='e36-native-ladder-v1:'+json.dumps(dict(codec=c)))) for c in ['jpeg','rav1e']]
        original=old/'manifest.json';original.write_text(json.dumps(jobs));manifest=new/'manifest.json';manifest.write_text(json.dumps(jobs[:1]))
        ids=old/'native-ids.tsv';ids.write_text('0\t'+'1'*64+'\n1\t'+'2'*64+'\n')
        filtered=new/'native.ids.tsv';filtered.write_text('0\t'+'1'*64+'\n')
        p=dict(run='svt-fallback-v1',old_run='old',original=matrix.bind(original),manifest=matrix.bind(manifest),native_ids=matrix.bind(filtered),original_native_ids=matrix.bind(ids))
        proof=dict(phases=dict(fallback=p))
        (root/'GENERATION-v7.log').write_text('')
        rows=[dict(job_id='1'*64,status='done',attempts=1,output_sha='3'*64,kind_json=json.dumps(jobs[0]['kind'])),
              dict(job_id='2'*64,status='poison',attempts=1,output_sha=None,kind_json=json.dumps(jobs[1]['kind']))]
        ledger=old/'ledger.parquet';pq.write_table(pa.Table.from_pylist(rows),ledger)
        return proof,ledger,rows

    def test_only_original_rav1e_completion_requirement_is_removed(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-native-filter-') as d:
            root=Path(d);proof,ledger,rows=self.fixture(root)
            with patch.object(matrix,'configuration',return_value=(proof,{},[])):
                self.assertEqual(len(matrix.publication(root,'svt-fallback-v1')[0]),1)
                for mutation in [dict(status='poison'),dict(attempts=2),dict(output_sha=None)]:
                    changed=[dict(rows[0],**mutation),rows[1]];pq.write_table(pa.Table.from_pylist(changed),ledger)
                    with self.assertRaisesRegex(ValueError,'publication failed'):
                        matrix.publication(root,'svt-fallback-v1')


if __name__=='__main__':unittest.main()
