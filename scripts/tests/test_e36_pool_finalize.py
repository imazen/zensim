"""Proof rebinding and unchanged opaque feature bytes, without a fit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_pool_finalize as owner
from e36_store import sha

class Finalize(unittest.TestCase):
    def test_full_archive_required_and_internal_hash_graph_rebound(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);prepared=root/'prepared';prepared.mkdir()
            def write(p,v):p.write_text(json.dumps(v))
            def pin(p):return dict(path=str(p),sha256=sha(p))
            old=dict(matrix='fallback',selected_runs=['mandatory'],manifests={},completed_views=1,outputs=['native'],clock={},plan={},artifact_archive_pending=True,artifact_location=None)
            write(prepared/'matrix-admission.json',old)
            write(prepared/'whole-matrix.json',dict(admission=pin(prepared/'matrix-admission.json')))
            (prepared/'alias-memberships.parquet').write_bytes(b'invented opaque aliases')
            write(prepared/'_MANIFEST.json',dict(required_support_admitted=True,zero_support_required_strata=[],fits=0,labels_read=False,whole_matrix_sha256=sha(prepared/'whole-matrix.json'),alias_memberships_sha256=sha(prepared/'alias-memberships.parquet')))
            for split in ['fit','selection']:
                p=prepared/split;p.mkdir()
                for n in ['features.parquet','pairs.parquet','edge-proof.parquet']:(p/n).write_bytes(('invented '+n).encode())
                write(p/'fx1.json',{'invented':'frozen declaration'})
                write(p/'support.json',dict(admitted=True,zero_support_required_strata=0,whole_matrix=pin(prepared/'whole-matrix.json'),features_sha256=sha(p/'features.parquet'),pairs_sha256=sha(p/'pairs.parquet'),edge_proof_sha256=sha(p/'edge-proof.parquet'),derived_sha256=sha(p/'fx1.json')))
                write(p/'manifest.json',dict(features=pin(p/'features.parquet'),pairs=pin(p/'pairs.parquet'),edge_proof=pin(p/'edge-proof.parquet'),required_support=pin(p/'support.json'),derived=pin(p/'fx1.json')))
                write(p/'semantic-admission.json',dict(admitted=True,fits=0,manifest=pin(p/'manifest.json'),whole_matrix=pin(prepared/'whole-matrix.json')))
                write(p/'contract.json',dict(manifest_sha256=sha(p/'manifest.json'),semantic_export=pin(p/'semantic-admission.json'),matrix_admission=pin(prepared/'matrix-admission.json')))
            final=old|dict(artifact_archive_pending=False,artifact_location={'invented':'verified archive'})
            with patch.object(owner,'admit',return_value=final|dict(outputs=['different'])):
                with self.assertRaisesRegex(ValueError,'scientific matrix changed'):owner.finalize(root,prepared,root/'bad','test')
            self.assertFalse((root/'bad').exists())
            out=root/'final'
            with patch.object(owner,'admit',return_value=final):receipt=owner.finalize(root,prepared,out,'test')
            self.assertTrue(receipt['feature_and_pair_bytes_unchanged'])
            self.assertEqual((out/'fit/features.parquet').read_bytes(),(prepared/'fit/features.parquet').read_bytes())
            self.assertTrue(json.loads((prepared/'matrix-admission.json').read_text())['artifact_archive_pending'])
            contract=json.loads((out/'fit/contract.json').read_text());self.assertEqual(contract['matrix_admission'],pin(out/'matrix-admission.json'))
            self.assertEqual(contract['semantic_export'],pin(out/'fit/semantic-admission.json'))
            self.assertEqual(contract['manifest_sha256'],sha(out/'fit/manifest.json'))
            self.assertNotEqual(contract['manifest_sha256'],sha(prepared/'fit/manifest.json'))
            cases=[(prepared/'alias-memberships.parquet',None),(prepared/'_MANIFEST.json','whole_matrix_sha256'),(prepared/'_MANIFEST.json','alias_memberships_sha256'),(prepared/'fit/contract.json','manifest_sha256')]
            for i,(path,key) in enumerate(cases):
                prior=path.read_bytes()
                if key is None:path.write_bytes(b'corrupt alias bytes')
                else:
                    value=json.loads(prior);value[key]='0'*64;write(path,value)
                with patch.object(owner,'admit',return_value=final):
                    with self.assertRaisesRegex(ValueError,'prepared proof/data changed'):owner.finalize(root,prepared,root/('scalar-bad-'+str(i)),'test')
                path.write_bytes(prior)
            (prepared/'fit/features.parquet').write_bytes(b'changed opaque features')
            with patch.object(owner,'admit',return_value=final):
                with self.assertRaisesRegex(ValueError,'prepared proof/data changed'):owner.finalize(root,prepared,root/'corrupt','test')

if __name__=='__main__':unittest.main()
