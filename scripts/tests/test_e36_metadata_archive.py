"""Opaque metadata streams retain exact bytes, including zero-length files."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_metadata_archive as client
import e36_metadata_store as store


class MetadataArchive(unittest.TestCase):
    def job(self,root):
        items=[]
        for n,data in enumerate([b'',bytes(range(256))*8193]):
            path=root/str(n);path.write_bytes(data);items.append(dict(local_path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        size=sum(512+((r['bytes']+511)//512)*512 for r in items);size=((size+1024+10239)//10240)*10240
        return dict(id='a'*64,items=items,tar_bytes=size)

    def test_actual_pipe_roundtrip_remote_member_hashes_and_corruption_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);job=self.job(root);destination=root/'remote';program=Path(store.__file__).resolve()
            # The production receiver's128GiB floor applies to actual storage;
            # this fixture directly uses a tiny deterministic local receiver.
            bootstrap=root/'receiver.py';bootstrap.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(program.parent))+')\nimport e36_metadata_store as s\nfrom unittest.mock import patch\nimport json\nwith patch.object(s.shutil,"disk_usage",return_value=type("Space",(),dict(free=1024**4))()):\n print(json.dumps(s.receive(sys.argv[2],sys.argv[4],int(sys.argv[6]))),flush=True)\n')
            result=client.transmit(job,[sys.executable,str(bootstrap)],store.sha(program),destination)
            self.assertEqual(result['members'],{r['sha256']:r['bytes'] for r in job['items']})
            archive=destination/(job['id']+'.tar');self.assertEqual(store.verify(archive)['sha256'],result['sha256'])
            content=bytearray(archive.read_bytes());content[1536]^=1;archive.write_bytes(content)
            with self.assertRaisesRegex(ValueError,'member hash differs'):store.verify(archive)
            for row in job['items']:self.assertTrue(Path(row['local_path']).exists())

    def test_changed_source_is_refused_and_receiver_drains(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);job=self.job(root);source=Path(job['items'][1]['local_path']);data=bytearray(source.read_bytes());data[0]^=1;source.write_bytes(data)
            bootstrap=root/'receiver.py';program=Path(store.__file__).resolve()
            bootstrap.write_text('import sys,json\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(program.parent))+')\nimport e36_metadata_store as s\nfrom unittest.mock import patch\nwith patch.object(s.shutil,"disk_usage",return_value=type("Space",(),dict(free=1024**4))()):\n print(json.dumps(s.receive(sys.argv[2],sys.argv[4],int(sys.argv[6]))),flush=True)\n')
            with self.assertRaisesRegex(ValueError,'opaque metadata changed'):
                client.transmit(job,[sys.executable,str(bootstrap)],store.sha(program),root/'remote')
            self.assertFalse(list((root/'remote').glob('*.tar')))


if __name__=='__main__':unittest.main()
