import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import e36_pack_store as pack


class PackTests(unittest.TestCase):
    def test_multiple_raw_bundles_survive_exactly(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);items=[]
            for i,data in enumerate([b'encoded pixels and maps\x00'*100001,b'other native bundle']):
                path=root/str(i);path.write_bytes(data);items.append(dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),local_path=str(path)))
            path=root/'pack.tar'
            with path.open('xb') as target:built=pack.build(target,items)
            checked=pack.verify(path)
            self.assertEqual(checked['members'],{item['sha256']:item['bytes'] for item in items})
            self.assertEqual({k:checked[k] for k in built},built)
            with tarfile.open(path) as archive:
                for item in items:self.assertEqual(archive.extractfile(item['sha256']+'.tar').read(),Path(item['local_path']).read_bytes())
            data=bytearray(path.read_bytes());data[600]^=1;path.write_bytes(data)
            with self.assertRaises(ValueError):pack.verify(path)

    def test_source_mutation_refuses_pack(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path=Path(directory)/'source';path.write_bytes(b'changed')
            with self.assertRaises(ValueError):pack.build(io.BytesIO(),[dict(sha256='a'*64,bytes=7,local_path=str(path))])

    def test_duplicate_and_truncated_members_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path=Path(directory)/'pack';data=b'x'*1000;h=hashlib.sha256(data).hexdigest()
            with tarfile.open(path,'w') as archive:
                for _ in range(2):
                    info=tarfile.TarInfo(h+'.tar');info.size=len(data);archive.addfile(info,io.BytesIO(data))
            with self.assertRaises(ValueError):pack.verify(path)
            path.write_bytes(path.read_bytes()[:700])
            with self.assertRaises((ValueError,tarfile.ReadError)):pack.verify(path)

    def test_real_receive_stream_frame_and_exact_readback(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);source=root/'raw';source.write_bytes(bytes(range(256))*1000)
            items=[dict(sha256=pack.sha(source),bytes=source.stat().st_size,local_path=str(source))];stream=io.BytesIO();built=pack.build(stream,items)
            size=sum(512+((i['bytes']+511)//512)*512 for i in items);declared=((size+1024+10239)//10240)*10240
            self.assertEqual(declared,len(stream.getvalue()))
            command=[sys.executable,str(Path(pack.__file__)),'receive',str(root),'--pack-id','a'*64,'--size',str(declared),'--sha256','-']
            import json
            actual=json.loads(subprocess.check_output(command,input=stream.getvalue()))
            self.assertEqual(actual['members'],{items[0]['sha256']:items[0]['bytes']});self.assertEqual(actual['stored_sha256'],built['stored_sha256'])


if __name__=='__main__':unittest.main()
