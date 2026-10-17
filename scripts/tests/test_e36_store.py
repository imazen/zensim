"""Loss/refusal checks for the storage-only stream, using invented bytes."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('e36_store',Path(__file__).parents[1]/'rev4_featpot/e36_store.py')
store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)


class StorageTests(unittest.TestCase):
    def temporary(self):
        root=Path.home()/'tmp';root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='e36-store-test-',dir=root)

    def test_transport_reconnect_resends_same_immutable_hash(self):
        from unittest.mock import patch
        with self.temporary() as d:
            p=Path(d)/'bundle';p.write_bytes(b'invented transport bytes')
            expected=store.sha(p);client=store.Client(['unused'],'a'*64)
            attempts=[]
            class Stream:
                def __init__(self,broken):self.broken=broken
                def write(self,value):
                    attempts.append(value)
                    if self.broken:raise BrokenPipeError('invented dropped SSH')
                def flush(self):pass
            class Process:
                def __init__(self,broken):self.stdin=Stream(broken)
            with patch.object(client,'connect',side_effect=[Process(True),Process(False)]),patch.object(client,'disconnect') as disconnect,patch.object(client,'reply',return_value=dict(ok=True,sha256=expected,bytes=p.stat().st_size)),patch.object(store.time,'sleep'):
                result=client.put(p)
            self.assertEqual(result['sha256'],expected)
            self.assertEqual(attempts[0],attempts[1])
            self.assertEqual(p.read_bytes(),b'invented transport bytes')
            disconnect.assert_called_once()

    def test_verified_put_duplicate_and_fresh_readback(self):
        data=b'invented encoded bytes, pixels and maps\x00\xff'*71
        sha=hashlib.sha256(data).hexdigest()
        frame=json.dumps(dict(operation='put',sha256=sha,bytes=len(data))).encode()+b'\n'+data
        check=json.dumps(dict(operation='verify',sha256=sha)).encode()+b'\n'
        with self.temporary() as root:
            reply=io.BytesIO();store.serve(root,io.BytesIO(frame+frame+check),reply)
            rows=[json.loads(x) for x in reply.getvalue().splitlines()]
            self.assertEqual([r.get('existed') for r in rows[:2]],[False,True])
            self.assertEqual((Path(root)/rows[0]['relative_path']).read_bytes(),data)
            self.assertEqual(rows[2]['sha256'],sha)
            self.assertEqual(list((Path(root)/'.incoming').iterdir()),[])

    def test_corruption_is_retained_and_never_canonicalized(self):
        data=b'wrong bytes';expected=hashlib.sha256(b'right bytes').hexdigest()
        frame=json.dumps(dict(operation='put',sha256=expected,bytes=len(data))).encode()+b'\n'+data
        with self.temporary() as root:
            with self.assertRaisesRegex(ValueError,'corrupt upload'):
                store.serve(root,io.BytesIO(frame),io.BytesIO())
            self.assertEqual(len(list((Path(root)/'.incoming').glob('*.failed-sha256'))),1)
            self.assertFalse((Path(root)/'blobs').exists())

    def test_truncated_upload_remains_partial_and_paths_are_refused(self):
        data=b'prefix';expected=hashlib.sha256(data).hexdigest()
        frame=json.dumps(dict(operation='put',sha256=expected,bytes=len(data)+1)).encode()+b'\n'+data
        with self.temporary() as root:
            with self.assertRaises(EOFError):store.serve(root,io.BytesIO(frame),io.BytesIO())
            self.assertEqual(len(list((Path(root)/'.incoming').glob('*.partial'))),1)
        for value in ['../escape','A'*64,'f'*63]:
            with self.assertRaises(ValueError):store.digest(value)


if __name__=='__main__':unittest.main()
