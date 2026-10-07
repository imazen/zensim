"""Reviewer synthetic population substitutions refuse before any label payload open."""
import builtins
import copy
from contextlib import ExitStack, contextmanager
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
import e28_recipe as e
import e28_simplex as nm
import v2_common as c
import v2_lodo_mlp as mlp
from v2_teacher import row_keys_sha


class Admission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp', prefix='e28-admission-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pin = e.read_pin()
        self.policy = dict(schema='e28-approved-inputs-v1', teacher_pin_sha256=c.sha(e.PIN),
                           source_files={}, prepared_files={}, source_key_identities={})
        self.legs = {'cid22': copy.deepcopy(self.pin['teachers']['cid22'])}
        self.payloads = set()
        self.opens = []
        for source, members in self.pin['arms']['s2m']['human_members'].items():
            source_rel = f'wide/main/real/{source}.parquet'
            for suffix in ['', '.manifest.json']:
                self.policy['source_files'][source_rel+suffix] = ('a' if not suffix else 'b')*64
            self.policy['source_files'][source_rel.replace('.parquet','.keys.parquet')] = 'c'*64
            self.policy['source_key_identities'][source] = 'd'*64
            recs = {}
            for split in ['fit','dev']:
                ref = next(f'{source}_{i}' for i in range(100) if c.human_dev(f'{source}_{i}') == (split=='dev'))
                rel = f'wide/main/real/e28_s2m_{source}_{split}.parquet'
                path = self.root / rel; path.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.table(dict(ref_basename=[ref,ref], human_score=[10.,90.], f13=[1.,2.], f14=[3.,4.])),path)
                keys = pa.table(dict(pair_key=[ref+'_0',ref+'_1'], source_row_id=[0,1], ref_basename=[ref,ref], member_set=[members[0]]*2))
                kp = path.with_suffix('.keys.parquet');pq.write_table(keys,kp)
                d = dict(study='E28',arm='s2m',source=source,split=split,role='TRAIN' if split=='fit' else 'DEV',
                         registration_commit='82c9af81',teacher_pin_sha256=c.sha(e.PIN),member_sets=members,
                         split_rule='sha256(ref_basename) mod 5 == 0 is dev',source_table_sha256='a'*64,
                         source_manifest_sha256='b'*64,source_keys_sha256='c'*64,source_label_free_row_keys_sha256='d'*64,
                         keys_sha256=c.sha(kp),row_keys_sha256=row_keys_sha(keys),row_selection_sha256='e'*64,
                         build_commit='f'*40,rows_kept=2,formula_revision=5,table_sha256=c.sha(path))
                mp=Path(str(path)+'.manifest.json');mp.write_text(json.dumps(d))
                recs[split]=dict(rel=rel,sha256=c.sha(path),manifest_sha256=c.sha(mp),keys_sha256=c.sha(kp),rows=2,references=1)
                for p in [path,mp,kp]:self.policy['prepared_files'][str(p.relative_to(self.root))]=c.sha(p)
                self.payloads.add(path.resolve())
            self.legs['e28_s2m_'+source]=recs

    @contextmanager
    def guard(self):
        # SHA uses Python opens; Arrow has a native open, so guard its read owner too.
        original_io, original_builtin, original_arrow = io.open, builtins.open, pq.read_table
        def check(path):
            if isinstance(path,(str,Path)) and Path(path).resolve() in self.payloads:
                self.opens.append(str(path));raise AssertionError('label-bearing payload opened before refusal')
        def io_open(path,*a,**k):check(path);return original_io(path,*a,**k)
        def builtin_open(path,*a,**k):check(path);return original_builtin(path,*a,**k)
        def arrow(path,*a,**k):check(path);return original_arrow(path,*a,**k)
        with ExitStack() as stack:
            stack.enter_context(patch.object(c,'V2',self.root));stack.enter_context(patch.object(mlp,'V2',self.root))
            stack.enter_context(patch.object(nm,'V2',self.root));stack.enter_context(patch.object(e,'admission_pin',return_value=self.policy))
            stack.enter_context(patch.object(io,'open',side_effect=io_open));stack.enter_context(patch.object(builtins,'open',side_effect=builtin_open))
            stack.enter_context(patch.object(pq,'read_table',side_effect=arrow));yield

    def mutate(self,field,value):
        rec=self.legs['e28_s2m_konfig']['fit'];p=self.root/(rec['rel']+'.manifest.json')
        d=json.loads(p.read_text());d[field]=value;p.write_text(json.dumps(d))
        rec['manifest_sha256']=c.sha(p)
        # Even an independently approved hash may not override role/population semantics.
        self.policy['prepared_files'][rec['rel']+'.manifest.json']=rec['manifest_sha256']

    def test_reviewer_val_dev_wrong_pin_refuse_for_mlp_and_nm(self):
        changes={'role':'VAL','split':'dev','member_sets':['konfig_val'],'teacher_pin_sha256':'wrong',
                 'keys_sha256':'wrong','arm':'s2o','source':'tid2013','source_table_sha256':'wrong',
                 'source_manifest_sha256':'wrong','source_keys_sha256':'wrong','source_label_free_row_keys_sha256':'wrong'}
        for field,value in changes.items():
            rec=self.legs['e28_s2m_konfig']['fit'];mp=self.root/(rec['rel']+'.manifest.json');original=mp.read_bytes()
            with self.subTest(field=field):
                self.mutate(field,value)
                with self.guard():
                    with self.assertRaises(ValueError):e.training_groups('s2m','kadid',self.legs,[],32)
                    with self.assertRaises(ValueError):nm.load_table(rec,[13,14],source='konfig',arm='s2m')
                self.assertEqual(self.opens,[])
            mp.write_bytes(original);rec['manifest_sha256']=c.sha(mp);self.policy['prepared_files'][rec['rel']+'.manifest.json']=rec['manifest_sha256']

    def test_approved_paths_cannot_redirect_to_another_payload(self):
        rec=self.legs['e28_s2m_konfig']['fit'];path=self.root/rec['rel']
        protected=self.root/'synthetic-protected.parquet';protected.write_bytes(path.read_bytes())
        self.payloads.add(protected.resolve())
        original=path.read_bytes();path.unlink();path.symlink_to(protected)
        with self.guard():
            with self.assertRaisesRegex(ValueError,'symlink'):e.training_groups('s2m','kadid',self.legs,[],32)
            with self.assertRaisesRegex(ValueError,'symlink'):nm.load_table(rec,[13,14],source='konfig',arm='s2m')
        self.assertEqual(self.opens,[])
        path.unlink();path.write_bytes(original)

    def test_unapproved_record_and_reviewer_generic_nm_call_refuse_without_payload(self):
        rec=self.legs['e28_s2m_konfig']['fit'];rec['sha256']='0'*64
        with self.guard():
            with self.assertRaises(ValueError):e.training_groups('s2m','kadid',self.legs,[],32)
            with self.assertRaises(ValueError):nm.load_table(rec,[13,14])
        self.assertEqual(self.opens,[])

    def test_all_manifests_precede_any_mlp_cli_or_nm_fit_payload(self):
        self.mutate('role','VAL')
        vdir=self.root/'wide/main/real';(vdir/'receipt.json').write_text(json.dumps(dict(schema='rev4-featpot-v2c-wide-v1',family='main',variant='real',width=1853,formula_revision=5,legs=self.legs,e28_teacher_pin_sha256=c.sha(e.PIN))))
        self.policy['prepared_files']['wide/main/real/receipt.json']=c.sha(vdir/'receipt.json')
        (self.root/'wide/keep_lists.json').write_text(json.dumps(dict(schema='rev4-featpot-v2-keeplists-v2')))
        from e21_cheap_recipe import columns
        argv=['v2_lodo_mlp.py','--spec',e.spec('s2m'),'--head','N','--heldout','kadid','--seed-index','0','--columns',','.join(map(str,columns('by_v2fy')))]
        with self.guard(),patch.object(sys,'argv',argv):
            with self.assertRaises(ValueError):mlp.main()
        with self.guard(),patch.object(__import__('subprocess'),'check_output',return_value=e.GROUPING.read_bytes()):
            with self.assertRaises(ValueError):nm.fit('kadid',self.root/'nm-out')
        self.assertEqual(self.opens,[])

    def test_label_free_keys_admitted_without_opening_targets(self):
        with self.guard():
            declarations,keys=e.admit_humans('s2m','kadid',self.legs)
        self.assertEqual(set(declarations),{('tid2013','fit'),('tid2013','dev'),('konfig','fit'),('konfig','dev')})
        self.assertTrue(all(k.column_names==e.KEY_COLUMNS for k in keys.values()))
        self.assertEqual(self.opens,[])

    def test_key_population_split_and_identity_refuse_before_target_open(self):
        rec=self.legs['e28_s2m_konfig']['fit'];path=self.root/rec['rel'];kp=path.with_suffix('.keys.parquet')
        original=kp.read_bytes();mp=Path(str(path)+'.manifest.json');original_manifest=mp.read_bytes()
        for mutation in ['population','split','identity','target_column']:
            with self.subTest(mutation=mutation):
                keys=pq.read_table(kp)
                if mutation=='population':keys=keys.set_column(3,'member_set',pa.array(['konfig_val']*2))
                if mutation=='split':
                    ref=next(f'x{i}' for i in range(100) if c.human_dev(f'x{i}'));keys=keys.set_column(2,'ref_basename',pa.array([ref]*2))
                if mutation=='identity':keys=keys.set_column(0,'pair_key',pa.array(['wrong']*2))
                if mutation=='target_column':keys=keys.append_column('target',pa.array([.1,.9]))
                pq.write_table(keys,kp);d=json.loads(mp.read_text());d['keys_sha256']=c.sha(kp)
                if mutation in ['population','split','target_column']:d['row_keys_sha256']=row_keys_sha(keys)
                mp.write_text(json.dumps(d));rec['keys_sha256']=c.sha(kp);rec['manifest_sha256']=c.sha(mp)
                self.policy['prepared_files'][str(kp.relative_to(self.root))]=c.sha(kp);self.policy['prepared_files'][rec['rel']+'.manifest.json']=c.sha(mp)
                with self.guard():
                    with self.assertRaises(ValueError):e.training_groups('s2m','kadid',self.legs,[],32)
                self.assertEqual(self.opens,[])
                kp.write_bytes(original);mp.write_bytes(original_manifest);rec['keys_sha256']=c.sha(kp);rec['manifest_sha256']=c.sha(mp)
                self.policy['prepared_files'][str(kp.relative_to(self.root))]=c.sha(kp);self.policy['prepared_files'][rec['rel']+'.manifest.json']=c.sha(mp)

    def test_confirm_hdr_directories_refuse_before_copy_or_sentinel_open(self):
        for name in ['confirm','HDR','hdr']:
            source=self.root/('source-'+name);sentinel=source/'wide'/name/'synthetic-protected-labels.txt';sentinel.parent.mkdir(parents=True);sentinel.write_text('synthetic labels')
            self.payloads.add(sentinel.resolve())
            out=self.root/('out-'+name)
            with self.guard(),patch.object(e.shutil,'copy2',side_effect=AssertionError('copied before refusal')):
                with self.assertRaisesRegex(ValueError,'confirmation/HDR'):e.prepare(source,out)
            self.assertFalse(out.exists());self.assertEqual(self.opens,[])

    def test_prepare_copies_only_explicitly_approved_members(self):
        source=self.root/'approved-source';vdir=source/'wide/main/real';vdir.mkdir(parents=True)
        ref_fit=next(f'fit{i}' for i in range(100) if not c.human_dev(f'fit{i}'))
        ref_dev=next(f'dev{i}' for i in range(100) if c.human_dev(f'dev{i}'))
        table=vdir/'konfig.parquet';refs=[ref_fit,ref_fit,ref_dev,ref_dev]
        pq.write_table(pa.table(dict(ref_basename=refs,human_score=[10.,90.,20.,80.],f13=[1.,2.,3.,4.])),table)
        kp=table.with_suffix('.keys.parquet')
        pq.write_table(pa.table(dict(pair_key=['p0','p1','p2','p3'],source_row_id=[0,1,2,3],ref_basename=refs,member_set=['konfig_train']*4,target=[.1,.9,.2,.8])),kp)
        mp=Path(str(table)+'.manifest.json');mp.write_text(json.dumps(dict(formula_revision=5,table_sha256=c.sha(table))))
        receipt=vdir/'receipt.json';receipt.write_text(json.dumps(dict(formula_revision=5,legs={'konfig':dict(full=dict(rel='wide/main/real/konfig.parquet',sha256=c.sha(table),manifest_sha256=c.sha(mp)),keys_sha256=c.sha(kp))})))
        keep=source/'wide/keep_lists.json';keep.write_text('{}')
        policy=copy.deepcopy(self.policy);policy['source_files']={str(p.relative_to(source)):c.sha(p) for p in [table,kp,mp,receipt,keep]}
        pin=copy.deepcopy(self.pin);pin['source_receipt_sha256']=c.sha(receipt);pin['arms']={'s2m':{'human_members':{'konfig':['konfig_train']}}}
        for name in ['hdr_unapproved.parquet','unapproved-labels.parquet']:
            extra=vdir/name;extra.write_bytes(b'synthetic unreadable protected labels');self.payloads.add(extra.resolve())
        out=self.root/'approved-out';copies=[];copy2=e.shutil.copy2
        def copied(a,b):copies.append(str(Path(a).relative_to(source)));return copy2(a,b)
        def jj(argv,**kwargs):return 'f'*40 if argv[1]=='log' else Path(e.__file__).read_bytes()
        with self.guard(),patch.object(e,'read_pin',return_value=pin),patch.object(e,'admission_pin',return_value=policy),patch.object(e.subprocess,'check_output',side_effect=jj),patch.object(e.shutil,'copy2',side_effect=copied):
            e.prepare(source,out)
        self.assertEqual(set(copies),set(policy['source_files']))
        self.assertEqual(len(copies),5)
        self.assertFalse((out/'wide/main/real/hdr_unapproved.parquet').exists())
        self.assertFalse((out/'wide/main/real/unapproved-labels.parquet').exists())
        self.assertEqual(self.opens,[])
        self.assertEqual(pq.read_table(out/'wide/main/real/e28_s2m_konfig_fit.keys.parquet').column_names,e.KEY_COLUMNS)


if __name__=='__main__':unittest.main()
