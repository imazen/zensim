"""Native single-cell launch controls; invented metadata only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_smoke_launch as owner


class Launch(unittest.TestCase):
    def test_exact_diagnostic_and_native_id_before_staging(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);out=root/'smoke';out.mkdir()
            def artifact(name,value):
                path=out/name;path.write_text(value);return dict(path=str(path),sha256=owner.sha(path))
            program=artifact('program.tar.gz','invented-program');data=artifact('data.tar.gz','opaque-fixture')
            job=dict(kind=dict(kind='fit_cell',argv=['v2_confirm_fit.py','--local-smoke-budget','2:128']))
            manifest=artifact('smoke.manifest.json',json.dumps([job]));ids=artifact('native.ids.tsv','invented-id\n')
            prepared=dict(planned_host='i270',policy=dict(wall_cap_sec=900),initial_memory_cap_bytes=8*1024**3,rayon_threads=2,program=program,data=data,manifest=manifest,native_ids=ids)
            (out/'PREPARED.json').write_text(json.dumps(prepared));admission=out/'program-admission';admission.mkdir()
            (admission/'receipt.json').write_text(json.dumps(dict(program_sha256=program['sha256'],all_inventory_hashes_match=True)))
            with patch.object(owner.subprocess,'check_output',return_value='invented-id\n'):
                owner.prepare(out)
            shell=(out/'run-i270.sh').read_text()
            for required in ['--memory 8g --memory-swap 8g','--cpuset-cpus 0-1','ZEN_FIT_RAYON_THREADS=2','ZENSIM_MAX_TIER=v3','--max-attempts 1','--mem 10G','--network none','--exec /usr/local/bin/fit-cell-exec']:
                self.assertIn(required,shell)
            self.assertNotIn('pip install',shell)
            self.assertEqual(json.loads((out/'LAUNCH_PLAN.json').read_text())['diagnostic_cells'],1)
            # An invalid host refuses before preparing any launch files.
            prepared['planned_host']='tower';(out/'PREPARED.json').write_text(json.dumps(prepared))
            with self.assertRaisesRegex(ValueError,'authorized bounded'):owner.prepare(out)


if __name__=='__main__':unittest.main()
