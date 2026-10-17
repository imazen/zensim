"""Explicit pre-fit whole-matrix amendment; retain the failed prior choice."""
import copy
import json
import time
from pathlib import Path

from e36_store import sha


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def checked(value):
    path=Path(value['path'])
    if sha(path)!=value['sha256']:raise ValueError('full-choice evidence changed: '+str(path))
    return path


def active(root,proof=None):
    root=Path(root);path=root/'pool-choice-full-v2.json'
    if not path.exists():return root/'bulk-choice.json'
    if proof is None:
        from e36_svt_matrix import configuration
        proof,_,_=configuration(root)
    choice=json.loads(path.read_text())
    amendment=checked(choice['full_choice_amendment'])
    if amendment!=root/'owner-svt-ssd-v1/full-pool-choice-amendment-v2.json':raise ValueError('wrong full-choice owner')
    value=json.loads(amendment.read_text())
    expected=[proof['phases'][p]['run'] for p in ['fallback','additions']]
    required=dict(schema='e36-full-pool-choice-amendment-v2',authorization='coordinator-option1-entire-completed-preregistered-12684',
        selected_runs=expected,selected_views=12684,fits=0,labels_seen=False,loss_seen=False,outcome_seen=False,
        sources_hand_picked=False,teacher_support_triggered=True,numerical_change=False,
        reason='required-stratum-zero-support-data-availability',teachers_margins_axes_guards_seeds_unchanged=True)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('exact approved full-pool amendment required')
    for key in ['prior_choice','prior_amendment','diagnosis','full_manifest','full_native_ids','failed_pool_manifest','failed_selection_support']:
        checked(value[key])
    if checked(value['prior_choice'])!=root/'bulk-choice.json' or value['full_manifest']!=proof['phases']['full']['manifest'] or value['full_native_ids']!=proof['phases']['full']['native_ids'] or proof['phases']['full']['jobs']!=12684:raise ValueError('whole preregistered full matrix required')
    prior=json.loads(checked(value['prior_choice']).read_text())
    if prior['qualified']!='fallback' or prior['fits']!=0 or prior.get('mandatory_choice_amendment')!=value['prior_amendment']:raise ValueError('failed prior choice history differs')
    failed=json.loads(checked(value['failed_pool_manifest']).read_text())
    if failed['required_support_admitted'] is not False:raise ValueError('recorded failed support basis differs')
    for phase in ['fallback','additions']:
        item=value['phases'][phase]
        if item['manifest']!=proof['phases'][phase]['manifest']:raise ValueError('phase membership differs')
        checked(item['manifest']);census=json.loads(checked(item['census']).read_text())
        if checked(item['census'])!=root/'runs'/proof['phases'][phase]['run']/'census.json' or census['admitted'] is not True or len(census['output_receipts'])!=proof['phases'][phase]['jobs']:raise ValueError('complete phase census required')
        checked(census['native_publication'])
    from e36_phase_terminal import validate
    validate(root,checked(value['optional_terminal']))
    if (choice['qualified']!='full' or choice['selected_runs']!=expected or choice['fits']!=0
        or choice['teacher_support_not_used_for_choice'] is not False or choice.get('teacher_support_triggered') is not True
        or choice.get('mandatory_choice_amendment') is not None):raise ValueError('truthful support-triggered full choice required')
    return path


def amend(root):
    from e36_svt_matrix import configuration,write
    root=Path(root);proof,_,_=configuration(root)
    prior=root/'bulk-choice.json';choice=copy.deepcopy(json.loads(prior.read_text()))
    amendment=root/'owner-svt-ssd-v1/full-pool-choice-amendment-v2.json'
    value=dict(schema='e36-full-pool-choice-amendment-v2',authorization='coordinator-option1-entire-completed-preregistered-12684',
        recorded_unix=time.time(),selected_runs=[proof['phases'][p]['run'] for p in ['fallback','additions']],selected_views=12684,
        fits=0,labels_seen=False,loss_seen=False,outcome_seen=False,sources_hand_picked=False,teacher_support_triggered=True,
        numerical_change=False,reason='required-stratum-zero-support-data-availability',teachers_margins_axes_guards_seeds_unchanged=True,
        prior_choice=pin(prior),prior_amendment=choice['mandatory_choice_amendment'],
        diagnosis=pin(root/'zero-support-optional-preview-v2.json'),full_manifest=proof['phases']['full']['manifest'],
        full_native_ids=proof['phases']['full']['native_ids'],failed_pool_manifest=pin(root/'pools-final-v1/_MANIFEST.json'),
        failed_selection_support=pin(root/'pools-final-v1/selection/support.json'),
        optional_terminal=pin(root/'owner-svt-ssd-v1/native-additions-final-v1.json'),
        phases={p:dict(manifest=proof['phases'][p]['manifest'],census=pin(root/'runs'/proof['phases'][p]['run']/'census.json')) for p in ['fallback','additions']})
    write(amendment,value)
    choice.pop('mandatory_choice_amendment')
    choice.update(qualified='full',selected_runs=value['selected_runs'],teacher_support_not_used_for_choice=False,
        teacher_support_triggered=True,closed_unix=time.time(),full_choice_amendment=pin(amendment))
    write(root/'pool-choice-full-v2.json',choice)
    return active(root,proof)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['amend','runs']);parser.add_argument('root',type=Path)
    args=parser.parse_args()
    if args.action=='amend':print(amend(args.root))
    else:print('\n'.join(json.loads(active(args.root).read_text())['selected_runs']))
