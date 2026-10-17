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


class MetadataOwnership(unittest.TestCase):
    def test_interrupted_close_and_failed_stdout_keep_real_lock_until_child_exits(self):
        import fcntl
        import subprocess
        import os
        for failure in ['close','read']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);lock=root/'paper.lock';proof=root/'child.json';script=root/'child.py'
                script.write_text('import sys,time,fcntl,json\nfrom pathlib import Path\nsys.stdin.buffer.read()\ntime.sleep(.05)\nwith Path(sys.argv[1]).open("a") as guard:\n try:\n  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);held=False\n except BlockingIOError:held=True\nPath(sys.argv[2]).write_text(json.dumps(dict(held=held)))\nprint("done",flush=True)\n')
                original=subprocess.Popen;children=[]
                class InterruptedClose:
                    def __init__(self,wrapped):self.wrapped=wrapped;self.interrupted=False
                    def __getattr__(self,key):return getattr(self.wrapped,key)
                    def close(self):
                        if not self.interrupted:self.interrupted=True;raise KeyboardInterrupt('fixture interrupted close')
                        return self.wrapped.close()
                def spawn(*args,**kwargs):
                    child=original(*args,**kwargs);children.append(child)
                    if failure=='close':child.stdin=InterruptedClose(child.stdin)
                    return child
                read=os.read;first=[True]
                def failed_read(*args):
                    if children and args[0]==children[0].stdout.fileno() and first[0]:first[0]=False;raise OSError('fixture stdout read failed')
                    return read(*args)
                from contextlib import nullcontext
                with lock.open('a') as guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    with patch.object(client.subprocess,'Popen',side_effect=spawn),patch.object(client.os,'read',side_effect=failed_read) if failure=='read' else nullcontext():
                        with self.assertRaises((KeyboardInterrupt,OSError)):
                            client.owned_exchange([sys.executable,str(script),str(lock),str(proof)],lambda writer:writer.write(b'opaque'))
                    self.assertTrue(json.loads(proof.read_text())['held']);self.assertIsNotNone(children[0].poll())
                    if failure=='close':
                        try:children[0].stdin.wrapped.close()
                        except OSError:pass

    def test_final_index_carries_path_map_and_remote_locator(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);source=root/'file.json';source.write_text('{"opaque":true}');out=root/'index';out.mkdir()
            job=client.final_index_job(out,[source]);program=Path(store.__file__).resolve();bootstrap=root/'receiver.py'
            bootstrap.write_text('import sys,json\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(program.parent))+')\nimport e36_metadata_store as s\nfrom unittest.mock import patch\nwith patch.object(s.shutil,"disk_usage",return_value=type("Space",(),dict(free=1024**4))()):\n print(json.dumps(s.receive(sys.argv[2],sys.argv[4],int(sys.argv[6]))),flush=True)\n')
            destination=root/'remote';answer=client.transmit(job,[sys.executable,str(bootstrap)],store.sha(program),destination)
            with tarfile.open(destination/(job['id']+'.tar')) as archive:
                descriptor=json.load(archive.extractfile('INDEX.json'));self.assertEqual(descriptor['files'][0]['path'],str(source))
                content=archive.extractfile(store.sha(source)+'.blob').read();self.assertEqual(content,source.read_bytes())
            locator=dict(index_id=job['id'],index_sha256=answer['sha256'],descriptor_sha256=answer['index_sha256']);body=json.dumps(locator).encode();h=hashlib.sha256(body).hexdigest()
            result=subprocess.run([sys.executable,str(program),'locate',str(destination),'--id',h,'--size',str(len(body))],input=body,stdout=subprocess.PIPE,check=True)
            self.assertEqual(json.loads(result.stdout)['sha256'],h);self.assertEqual((destination/('LOCATOR-'+h+'.json')).read_bytes(),body)


class MetadataClosure(unittest.TestCase):
    def terminal(self,root,reason='all native-published local bundles drained; metadata retained for separate canonical archive'):
        import e36_archive_terminal as terminal
        owner=root/'owner-svt-ssd-v1';owner.mkdir()
        def put(name,value):
            path=owner/name;path.write_text(json.dumps(value));return terminal.pin(path)
        inventory=put('inventory.json',dict(schema='e36-closed-native-archive-inventory-v1',fits=0,qualifies_data=False))
        stop=put('mirror-stop.json',dict(reason=reason,inventory=inventory,fits=0));source=put('source.json',{})
        launch=put('launch.json',dict(controller_pid=0,mirror_pid=0,source=source,mirror=source));exit_record=put('exit.json',dict(returncode=0));sources=put('sources.json',dict(files=dict(source=source)))
        log=owner/'drain.log';log.write_text('run-heavy: done rc=0 1s | peak-RSS 0.1GiB\n')
        put('archive-drain-final-v1.json',dict(schema='e36-raw-drain-terminal-v1',closed=True,fits=0,returncode=0,launch=launch,exit=exit_record,stop=stop,log=terminal.pin(log),sources=sources))
        return terminal,owner

    def test_shutdown_only_marker_and_live_owned_child_refuse(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);terminal,owner=self.terminal(root,reason='native phase ended')
            with self.assertRaisesRegex(ValueError,'completion semantics'):terminal.validate(root)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);terminal,owner=self.terminal(root);terminal.validate(root)
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)','e36_artifact_mirror.py'])
            try:
                launch=owner/'launch.json';value=json.loads(launch.read_text());value['mirror_pid']=child.pid;launch.write_text(json.dumps(value))
                path=owner/'archive-drain-final-v1.json';value=json.loads(path.read_text());value['launch']=terminal.pin(launch);path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'still active'):terminal.validate(root)
            finally:child.wait()

    def test_seal_prevents_restart_and_plan_union_omission_refuses(self):
        from e36_metadata_seal import freeze,lifecycle
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);terminal,owner=self.terminal(root);drain,_=terminal.validate(root);seal=freeze(root,drain)
            with self.assertRaisesRegex(ValueError,'restart refused'):
                with lifecycle(root):pass
            plan=dict(root=str(root),raw_drain=client.pin(drain),seal=client.pin(seal),files=[dict(sha256='a'*64,bytes=1)],jobs=[])
            with self.assertRaisesRegex(ValueError,'union differs'):client.validate_closed_plan(plan)


if __name__=='__main__':unittest.main()
