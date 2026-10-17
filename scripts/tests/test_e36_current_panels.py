"""Current-panel metadata/refusal fixtures; native hash frames are invented.

No real PNG, native hash, label, model or optimizer runs in these tests.
"""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_current_panels as owner


def write(p,x):p.write_text(json.dumps(x))


class CurrentPanels(unittest.TestCase):
    def fixture(self,root):
        family=root/'families.tsv';family.write_text('id\tfamily\n8206\tnot-selected\n')
        rows=[]
        for i in range(200):
            p=root/f'source-{i}';p.write_bytes(f'invented non-PNG source {i}'.encode())
            rows.append(dict(source=str(p),source_sha256=owner.sha(p),family=f'origin:{i}'))
        cohort=dict(labels_read=False,rows=rows,current_family_v2=dict(path=str(family),sha256=owner.sha(family)),inherited_lineage_limit='invented fixture')
        write(root/'cohort-selection.json',cohort)
        panel=root/'panel.json';write(panel,dict(sources=[dict(name='gb82-fixture',source_family='gb82-fixture',sha256='a'*64)]))
        prep=root/'prepared';prep.mkdir();png=prep/'source-00.png';png.write_bytes(b'invented non-PNG native-result fixture')
        write(prep/'INPUTS.json',dict(manifest_sha256=owner.sha(panel),sources=1))
        write(prep/'source-00-recipe.json',dict(input_sha256='a'*64,output_sha256=owner.sha(png)))
        binary=root/'native-owner';binary.write_bytes(b'invented nonexecutable native-owner identity')
        return cohort,panel,prep,binary

    def frame(self,cohort,refs):
        return dict(schema='native-png-dhash-audit-v1',hash_era='bt709-encoded-luma-zenresize-lanczos-gray8-v1',threshold=10,
            training=[dict(path=r['source'],sha256=r['source_sha256'],dhash=i) for i,r in enumerate(cohort['rows'])],
            holdout=[dict(path=str(p),sha256=owner.sha(p),dhash=999) for p in refs],strict_flags=0,close_pairs=[],admission_approved=False)

    def test_native_flags_and_coverage_cannot_be_replaced_by_pass_claims(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-panel-invented-') as d:
            root=Path(d);cohort,panel,prep,binary=self.fixture(root)
            refs=[dict(path=str(prep/'source-00.png'),sha256=owner.sha(prep/'source-00.png'))]
            frame=self.frame(cohort,[prep/'source-00.png'])
            self.assertTrue(owner.native_admitted(frame,cohort,refs))
            bad=copy.deepcopy(frame);bad['strict_flags']=1;bad['close_pairs']=[dict(strict_flag=True)]
            self.assertFalse(owner.native_admitted(bad,cohort,refs))
            bad=copy.deepcopy(frame);bad['training'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'coverage'):owner.native_admitted(bad,cohort,refs)
            bad=copy.deepcopy(frame);bad['holdout']=[]
            with self.assertRaisesRegex(ValueError,'coverage'):owner.native_admitted(bad,cohort,refs)

    def test_manifest_mutation_during_audit_retains_old_evidence_and_refuses(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-panel-race-') as d:
            root=Path(d);cohort,panel,prep,binary=self.fixture(root);before=panel.read_bytes();out=root/'audit'
            def invented_native(argv,**kwargs):
                prefix=Path(argv[argv.index('--out-tsv')+1]);refs=list((out/'references').iterdir())
                write(prefix.with_suffix('.hashes.json'),self.frame(cohort,refs))
                changed=json.loads(panel.read_text());changed['new_revision']='metadata changed during native replay';write(panel,changed)
            with patch.object(owner.subprocess,'run',side_effect=invented_native):
                with self.assertRaisesRegex(ValueError,'metadata changed'):owner.run(root,panel,prep,binary,out)
            self.assertEqual((out/'SOURCE_MANIFEST.json').read_bytes(),before)
            self.assertTrue((out/'audit.hashes.json').is_file())
            self.assertFalse((out/'receipt.json').exists())


if __name__=='__main__':unittest.main()
