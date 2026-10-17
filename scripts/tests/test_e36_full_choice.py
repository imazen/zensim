"""Whole-phase membership and truthful amendment provenance; no fit or labels."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_full_choice as owner


class FullChoice(unittest.TestCase):
    def test_entire_matrix_retains_prior_and_refuses_support_independence(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);folder=root/'owner-svt-ssd-v1';folder.mkdir()
            def file(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return owner.pin(path)
            phases={}
            for phase,count in [('fallback',12300),('additions',384),('full',12684)]:
                run='run-'+phase;manifest=file(root/'runs'/run/'manifest.json',{'fixture':phase})
                ids=file(root/'runs'/run/'native.ids.tsv',{'fixture':phase})
                phases[phase]=dict(run=run,jobs=count,manifest=manifest,native_ids=ids)
                publication=file(root/'runs'/run/'publication.json',{})
                file(root/'runs'/run/'census.json',dict(admitted=True,output_receipts=[{}]*count,native_publication=publication))
            prior_amendment=file(folder/'mandatory-choice-amendment-v1.json',{})
            prior=file(root/'bulk-choice.json',dict(qualified='fallback',fits=0,teacher_support_not_used_for_choice=True,mandatory_choice_amendment=prior_amendment))
            for path in ['zero-support-optional-preview-v2.json','pools-final-v1/selection/support.json','owner-svt-ssd-v1/native-additions-final-v1.json']:file(root/path,{})
            file(root/'pools-final-v1/_MANIFEST.json',dict(required_support_admitted=False))
            proof=dict(phases=phases)
            with patch('e36_svt_matrix.configuration',return_value=(proof,{},[])),patch('e36_phase_terminal.validate'):
                path=owner.amend(root)
                self.assertEqual(owner.pin(root/'bulk-choice.json'),prior)
                self.assertEqual(owner.active(root,proof),path)
                choice=json.loads(path.read_text());self.assertEqual(choice['selected_runs'],['run-fallback','run-additions'])
                for changes in [dict(teacher_support_not_used_for_choice=True),dict(selected_runs=['run-fallback']),dict(teacher_support_triggered=False)]:
                    path.write_text(json.dumps(dict(choice,**changes)))
                    with self.assertRaises(ValueError):owner.active(root,proof)
                path.write_text(json.dumps(choice))
                amendment=Path(choice['full_choice_amendment']['path']);value=json.loads(amendment.read_text())
                value['sources_hand_picked']=True;amendment.write_text(json.dumps(value));choice['full_choice_amendment']=owner.pin(amendment);path.write_text(json.dumps(choice))
                with self.assertRaisesRegex(ValueError,'exact approved'):owner.active(root,proof)


if __name__=='__main__':unittest.main()
