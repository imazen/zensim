"""Real streaming archive parity and corrupt/truncated-byte refusal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_archive_store import ArchiveClient,raw_identity
import e36_archive_store as store
from e36_artifact_location import record_owner
from e36_store import sha


class Archive(unittest.TestCase):
    def test_transport_drains_before_retry_without_termination(self):
        client=ArchiveClient(['unused'],'a'*64);process=Mock()
        process.wait.side_effect=[subprocess.TimeoutExpired('ssh',30),0];client.process=process
        def holding(reason):self.assertIs(client.process,process)
        with patch.object(client,'hold',side_effect=holding) as hold:
            self.assertEqual(client.disconnect(),0)
            hold.assert_called_once()
        process.stdin.close.assert_called_once();process.terminate.assert_not_called();process.kill.assert_not_called()
        self.assertIsNone(client.process)

    def test_actual_streaming_raw_parity_and_truncation_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-zstd-test-') as directory:
            root=Path(directory);raw=root/'bundle.tar';raw.write_bytes(bytes(range(256))*8193+os.urandom(131077))
            destination=root/'remote';destination.mkdir()
            command=[sys.executable,str(Path(store.__file__).resolve()),'--root',str(destination),'--min-free-gib','1']
            client=ArchiveClient(command,sha(store.__file__))
            with patch.dict(os.environ,{'E36_ARCHIVE_STAGE':str(root/'transient')}):
                try:
                    ack=client.put(raw);answer=client.verify(sha(raw));self.assertEqual(ack,answer)
                    archived=destination/ack['relative_path'];self.assertEqual(raw_identity(archived),(sha(raw),raw.stat().st_size))
                    truncated=root/'truncated.zst';truncated.write_bytes(archived.read_bytes()[:-1])
                    with self.assertRaisesRegex(ValueError,'truncated'):raw_identity(truncated)
                    corrupt=root/'corrupt.zst';corrupt.write_bytes(b'bad'+archived.read_bytes()[3:])
                    with self.assertRaisesRegex(ValueError,'invalid'):raw_identity(corrupt)
                finally:client.close()

    def test_split_owner_rejects_wrong_namespace_and_program(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-owner-test-') as directory:
            root=Path(directory);legacy=dict(host='tower',root='/legacy',program_sha256='a'*64)
            archive=dict(host='tower',root='/mnt/disk1/coefficient/output/zensim/e36-bulk-2026-10-17-disk1',program_sha256='b'*64,encoding='zstd')
            (root/'storage-archive-v2.json').write_text(json.dumps(archive))
            row=dict(canonical_host='tower',canonical_root=archive['root'],encoding='zstd',sha256='c'*64,stored_bytes=123,stored_sha256='d'*64,program_sha256='b'*64,relative_path='blobs/cc/'+('c'*64)+'.tar.zst')
            self.assertEqual(record_owner(root,row,legacy),archive)
            for key,value in [('canonical_root','/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-disk1'),('program_sha256','a'*64),('relative_path','../wrong'),('stored_sha256','bad')]:
                with self.assertRaises(ValueError):record_owner(root,{**row,key:value},legacy)


if __name__=='__main__':unittest.main()
