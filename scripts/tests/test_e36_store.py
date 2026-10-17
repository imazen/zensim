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
