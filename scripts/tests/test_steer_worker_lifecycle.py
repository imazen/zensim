"""Admission refusals using invented scheduler metadata; no fits or network."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import steer_worker_lifecycle as life


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home()/'tmp')
        self.root = Path(self.tmp.name)
        self.bundle = self.root/'packet'; self.bundle.mkdir()
        (self.root/'launch').mkdir(); (self.root/'logs').mkdir()
        self.binding = dict(image='image@sha256:fixture',
                            shared_environment={'ZEN_MEMORY': '6g', 'ZEN_CPU_SHARES': '256',
                                                'ZEN_PASS_TIMEOUT': '8000', 'ZEN_CHUNK_WALL_SEC': '7663'},
                            active={h: dict(ZEN_CPUSET='0', ZEN_TMPDIR_HOST_DIR='/disk/scratch')
                                    for h in ['r3500', 'i270', 'tower']})
        (self.bundle/'WORKER_BINDING.json').write_text(json.dumps(self.binding))
        self.cap = dict(memory=dict(MemTotal=32*life.GIB, MemAvailable=24*life.GIB),
                        workers=[], free=40*life.GIB, docker_free=40*life.GIB)
        self.docker = dict(Config=dict(Image=self.binding['image'],
                                      Env=['ZEN_PASS_TIMEOUT=8000', 'ZEN_CHUNK_WALL_SEC=1',
                                           'ZEN_NO_NEW_AFTER_EPOCH=130']),
                           HostConfig=dict(Memory=6*life.GIB, MemorySwap=6*life.GIB,
                                           CpusetCpus='0', CpuShares=256,
                                           RestartPolicy=dict(Name='no')),
                           RestartCount=0, State=dict(Running=True),
                           Mounts=[dict(Source='/disk/scratch', Destination='/scratch')])

    def tearDown(self):
        self.tmp.cleanup()

    def test_preservation_moves_only_declared_scratch_and_keeps_bytes(self):
        scratch=self.root/'scratch';tmp=scratch/'tmp';tmp.mkdir(parents=True)
        stem='coverage_cf98_steer_'+'a'*64
        for suffix in ['.parquet','.keys.parquet','.parquet.manifest.json']:
            (tmp/(stem+suffix)).write_bytes(b'preserved interrupted bytes')
        unrelated=tmp/('coverage_cf98_steer_'+'b'*64+'.parquet')
        unrelated.write_bytes(b'unrelated')
        result=life.preserve_coverage_scratch(str(scratch),[('98','a'*64)],'20261017T040000.000000Z')
        self.assertEqual(len(result['files']),3)
        for item in result['files']:
            self.assertFalse((tmp/item['name']).exists())
            self.assertEqual((Path(result['backup'])/item['name']).read_bytes(),b'preserved interrupted bytes')
        self.assertEqual(unrelated.read_bytes(),b'unrelated')
        with self.assertRaisesRegex(ValueError,'already exists'):
            life.preserve_coverage_scratch(str(scratch),[('98','a'*64)],'20261017T040000.000000Z')

    def test_preservation_refuses_symlinks_and_foreign_identity_before_mutation(self):
        scratch=self.root/'scratch';tmp=scratch/'tmp';tmp.mkdir(parents=True)
        good=tmp/('coverage_cf98_steer_'+'a'*64+'.parquet');good.write_bytes(b'partial')
        foreign=self.root/'input';foreign.write_bytes(b'immutable input')
        link=tmp/('coverage_cf98_steer_'+'b'*64+'.parquet');link.symlink_to(foreign)
        with self.assertRaisesRegex(ValueError,'regular file'):
            life.preserve_coverage_scratch(str(scratch),[('98','a'*64),('98','b'*64)],'20261017T040000.000000Z')
        self.assertEqual(good.read_bytes(),b'partial')
        self.assertEqual(foreign.read_bytes(),b'immutable input')
        for declarations in [[('98','../input')],[('../98','a'*64)]]:
            with self.assertRaisesRegex(ValueError,'coverage identity'):
                life.preserve_coverage_scratch(str(scratch),declarations,'20261017T040000.000000Z')
        alias=self.root/'alias';alias.symlink_to(scratch)
        with self.assertRaisesRegex(ValueError,'resolved disk path'):
            life.preserve_coverage_scratch(str(alias),[],'20261017T040000.000000Z')

    def test_preservation_requires_paused_controls_and_no_live_worker(self):
        manifest=[dict(kind=dict(argv=['--spec','fixture:cf98'],argv_sha='a'*64))]
        (self.bundle/'fit-manifest-job.json').write_text(json.dumps(manifest))
        with mock.patch.object(life,'ssh',side_effect=AssertionError('remote mutation')):
            with self.assertRaisesRegex(ValueError,'operator pause'):
                life.preserve_interrupted_scratch(self.bundle,{},'r3500')
        (self.root/'launch/PAUSE_WORKERS').touch()
        cap=copy.deepcopy(self.cap)
        cap['workers']=[dict(run='jobs/fitsteer-job')]
        with mock.patch.object(life.owner,'steer_plan',return_value={'order':['job']}), \
             mock.patch.object(life,'control',return_value=dict(paused=False)), \
             mock.patch.object(life,'ssh',side_effect=AssertionError('remote mutation')):
            with self.assertRaisesRegex(ValueError,'controls must be paused'):
                life.preserve_interrupted_scratch(self.bundle,{},'r3500')
        with mock.patch.object(life.owner,'steer_plan',return_value={'order':['job']}), \
             mock.patch.object(life,'control',return_value=dict(paused=True)), \
             mock.patch.object(life,'capacity',return_value=cap), \
             mock.patch.object(life,'ssh',side_effect=AssertionError('remote mutation')):
            with self.assertRaisesRegex(ValueError,'still running'):
                life.preserve_interrupted_scratch(self.bundle,{},'r3500')

    def test_preservation_admits_all_four_registered_spec_shapes(self):
        (self.root/'launch/PAUSE_WORKERS').touch()
        variants=['sel:abc@h32:H128:cv16:cf98', 'sel:abc@h32:H128:cv16:cf98:fx1',
                  'sel:abc@h32:H128:cv16:cf98:sh1', 'sel:abc@h32:H128:cv16:cf98:fx1:sh1']
        jobs=[dict(kind=dict(argv=['--spec',spec],argv_sha=token*64))
              for spec,token in zip(variants,'abcd')]
        manifest=self.bundle/'fit-manifest-job.json';manifest.write_text(json.dumps(jobs))
        with mock.patch.object(life.owner,'steer_plan',return_value={'order':['job']}), \
             mock.patch.object(life,'control',return_value=dict(paused=True)), \
             mock.patch.object(life,'capacity',return_value=self.cap), \
             mock.patch.object(life,'ssh',return_value='{"files": []}') as ssh:
            out=life.preserve_interrupted_scratch(self.bundle,{},'r3500')
            self.assertFalse(out['retry_limits_changed'])
            import shlex
            args=shlex.split(ssh.call_args.args[1])
            self.assertEqual(json.loads(args[-2]),[['98',c*64] for c in 'abcd'])
            jobs[0]['kind']['argv']=['--spec','sel:abc:cf98:cf99']
            manifest.write_text(json.dumps(jobs))
            with self.assertRaisesRegex(ValueError,'exact coverage mask'):
                life.preserve_interrupted_scratch(self.bundle,{},'r3500')
            self.assertEqual(ssh.call_count,1)

    def test_capacity_refuses_occupied_slot_reservations_memory_and_disk(self):
        life.admit('r3500', self.cap)
        cases=[]
        c=copy.deepcopy(self.cap);c['workers']=[dict(name='/other',memory=6*life.GIB,run='jobs/fitsteer-x')];cases.append(c)
        c=copy.deepcopy(self.cap);c['workers']=[dict(name='/other',memory=0,run='other')];cases.append(c)
        c=copy.deepcopy(self.cap);c['memory']['MemTotal']=9*life.GIB;cases.append(c)
        c=copy.deepcopy(self.cap);c['memory']['MemAvailable']=9*life.GIB;cases.append(c)
        for key in ['free','docker_free']:
            c=copy.deepcopy(self.cap);c[key]=20*life.GIB-1;cases.append(c)
        for c in cases:
            with self.assertRaises(ValueError):life.admit('r3500',c)

    def test_tower_fit_pool_and_kernel_swap(self):
        life.admit('tower',self.cap)
        c=copy.deepcopy(self.cap);c['workers']=[dict(name='/fit',memory=7*life.GIB,run='jobs/fit-other')]
        with self.assertRaisesRegex(ValueError,'fit pool'):life.admit('tower',c)
        d=copy.deepcopy(self.docker);d['HostConfig']['MemorySwap']=-1
        self.assertTrue(all(life.envelope(self.binding,'tower','ctr',d,130,[6*life.GIB,0,0]).values()))
        for kernel in [[6*life.GIB,1,0],[6*life.GIB,0,1],[5*life.GIB,0,0],None]:
            self.assertFalse(life.envelope(self.binding,'tower','ctr',d,130,kernel)['swap'])

    def test_envelope_refuses_binding_violations(self):
        self.assertTrue(all(life.envelope(self.binding,'r3500','ctr',self.docker,130).values()))
        for key,value in [('Memory',7*life.GIB),('MemorySwap',7*life.GIB),('CpusetCpus','0-4'),('CpuShares',1024)]:
            d=copy.deepcopy(self.docker);d['HostConfig'][key]=value
            self.assertFalse(all(life.envelope(self.binding,'r3500','ctr',d,130).values()))
        d=copy.deepcopy(self.docker);d['Config']['Env']=['ZEN_PASS_TIMEOUT=8000','ZEN_CHUNK_WALL_SEC=1']
        self.assertFalse(life.envelope(self.binding,'r3500','ctr',d,130)['deadline'])

    def test_transition_requires_all_evidence(self):
        proof=dict(packet_sha256=life.owner.PACKET_SHA,all_old_chunks_complete=True,
                   ledgers_published=True,native_verified=True,NAS_verified=True,old_workers_stopped=True)
        p=self.root/'launch/OLD_CHUNKS_DRAINED.json';p.write_text(json.dumps(proof))
        self.assertEqual(life.transition_proof(self.bundle),proof)
        for key in proof:
            changed=dict(proof);changed[key]=False;p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):life.transition_proof(self.bundle)

    def test_repaired_admission_refuses_missing_false_and_changed_evidence(self):
        with self.assertRaises(FileNotFoundError):life.repaired_admission(self.bundle)
        dest=self.root/'cell';dest.mkdir()
        for name in ['result.json','fleet_receipt.json']:(dest/name).write_text('{}')
        proof=dict(packet_sha256=life.owner.PACKET_SHA,verified=True,inside_pinned_image=True,
                   worker_drained=True,NAS_verified=True,dest=str(dest),
                   result_sha256=life.owner.sha(dest/'result.json'),
                   receipt_sha256=life.owner.sha(dest/'fleet_receipt.json'))
        p=self.root/'launch/REPAIRED_CELL_ADMITTED.json';p.write_text(json.dumps(proof))
        life.repaired_admission(self.bundle)
        for key in ['verified','inside_pinned_image','worker_drained','NAS_verified']:
            p.write_text(json.dumps(dict(proof,**{key:False})))
            with self.assertRaises(ValueError):life.repaired_admission(self.bundle)
        p.write_text(json.dumps(proof));(dest/'result.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):life.repaired_admission(self.bundle)

    def test_no_remote_action_before_repaired_admission(self):
        with mock.patch.object(life.owner,'steer_plan',return_value={'order':['job']}), \
             mock.patch.object(life.owner,'first_steer_admission'), \
             mock.patch.object(life,'transition_proof'), \
             mock.patch.object(life,'ssh',side_effect=AssertionError('remote action')), \
             mock.patch.object(life,'control',side_effect=AssertionError('store action')):
            with self.assertRaises(FileNotFoundError):
                life.enroll(self.bundle,{'image_ref':self.binding['image']},'r3500','steer-r3500-003','job')

    def test_release_order_and_operator_pause_race(self):
        import types
        for stop in [False,True]:
            role='steer-r3500-'+('004' if stop else '003')
            post=copy.deepcopy(self.cap)
            post['workers']=[dict(name='/zen-score-'+role,memory=6*life.GIB,run='jobs/fitsteer-job')]
            paused=dict(paused=True,drain=False,note='STEER envelope '+role)
            controls=[]
            def control(bundle,js,value=None):
                if value is not None:controls.append(value);return None
                if not controls:return dict(paused=False,drain=False)
                if stop:(self.root/'launch/PAUSE_WORKERS').touch()
                return paused
            def ssh(host,cmd):
                if cmd=='date -u +%s':return '100\n'
                return ''
            with mock.patch.object(life.owner,'steer_plan',return_value={'order':['job']}), \
                 mock.patch.object(life.owner,'first_steer_admission'), \
                 mock.patch.object(life,'transition_proof'),mock.patch.object(life,'repaired_admission'), \
                 mock.patch.object(life,'capacity',side_effect=[self.cap,post]), \
                 mock.patch.object(life,'inspect',return_value=self.docker), \
                 mock.patch.object(life,'control',side_effect=control), \
                 mock.patch.object(life,'ssh',side_effect=ssh), \
                 mock.patch.object(life.subprocess,'run',return_value=types.SimpleNamespace(stdout='holding',stderr='')) as run:
                if stop:
                    with self.assertRaisesRegex(ValueError,'envelope failed'):
                        life.enroll(self.bundle,{'image_ref':self.binding['image']},'r3500',role,'job')
                    self.assertEqual(controls,[paused])
                else:
                    out=life.enroll(self.bundle,{'image_ref':self.binding['image']},'r3500',role,'job')
                    self.assertEqual(out['state'],'RELEASED')
                    self.assertEqual(controls,[paused,dict(paused=False,drain=False)])
                    launch=run.call_args_list[0]
                    self.assertEqual(launch.args[0],['bash',str(self.root/'launch-source/zenmetrics/scripts/jobsys/lan_score_launch.sh'),'r3500','job',role,'cpu',self.binding['image']])
                    env=launch.kwargs['env']
                    self.assertEqual(env['ZEN_PASS_TIMEOUT'],'8000')
                    self.assertEqual(env['ZEN_CHUNK_WALL_SEC'],'1')
                    self.assertEqual(env['ZEN_NO_NEW_AFTER_EPOCH'],'130')

    def test_metadata_selection_refuses_partial_invalid_and_failed_evidence(self):
        jobs={str(i):40 if i<4 else 3 for i in range(8)}
        status=dict(packet_sha256=life.owner.PACKET_SHA,state='PROGRESS',errors=[],
                    jobsets={js:dict(state='OK',installed=0,ledger={}) for js in jobs})
        with mock.patch.object(life.owner,'steer_plan',return_value=dict(order=list(jobs))):
            self.assertEqual(life.next_jobset(self.bundle,dict(jobsets=jobs),status),'0')
            for mode in ['wrong_pin','bad_count','boolean_count','missing','failed','error']:
                s=copy.deepcopy(status)
                if mode=='wrong_pin':s['packet_sha256']='wrong'
                elif mode=='bad_count':s['jobsets']['0']['installed']=41
                elif mode=='boolean_count':s['jobsets']['0']['installed']=True
                elif mode=='missing':del s['jobsets']['7']
                elif mode=='failed':s['jobsets']['0']['ledger']['failed']=1
                else:s['errors']=['store failed']
                with self.assertRaises(ValueError):life.next_jobset(self.bundle,dict(jobsets=jobs),s)
            self.assertEqual(life.next_jobset(self.bundle,dict(jobsets=jobs),dict(status,state='RUNNING')),'WAIT')
            for js,n in jobs.items():status['jobsets'][js]['installed']=n
            self.assertIsNone(life.next_jobset(self.bundle,dict(jobsets=jobs),status))

    def test_pause_closes_enrollment_before_waiting_for_release_lock(self):
        events=[]
        def lock(fd,mode):
            self.assertTrue((self.root/'launch/PAUSE_WORKERS').is_file())
            events.append('locked')
        def control(bundle,js,value=None):
            self.assertEqual(events[0],'locked')
            if value is None:return dict(paused=False,drain=False,worker_weights={'native':1})
            self.assertEqual(value,dict(paused=True,drain=False,worker_weights={'native':1}))
            events.append(js)
        with mock.patch.object(life.owner,'steer_plan',return_value={'order':['a','b']}), \
             mock.patch.object(life.fcntl,'flock',side_effect=lock), \
             mock.patch.object(life,'control',side_effect=control):
            out=life.pause_workers(self.bundle,{})
        self.assertEqual(out['jobsets'],['a','b'])
        self.assertEqual(events,['locked','a','b'])

    def test_watch_completion_pause_and_disk_stop(self):
        (self.root/'launch/HARVEST_STATUS.json').write_text(json.dumps(dict(installed=172,state='COMPLETE')))
        with mock.patch.object(life,'transition_proof'),mock.patch.object(life,'repaired_admission'), \
             mock.patch.object(life,'next_jobset',return_value=None), \
             mock.patch.object(life,'capacity',return_value=self.cap), \
             mock.patch.object(life.subprocess,'run',side_effect=AssertionError('unexpected launch')):
            life.watch(self.bundle,{})
        (self.root/'launch/PAUSE_WORKERS').touch()
        with mock.patch.object(life,'transition_proof'),mock.patch.object(life,'repaired_admission'), \
             mock.patch.object(life,'next_jobset',return_value='job'), \
             mock.patch.object(life,'capacity',return_value=self.cap), \
             mock.patch.object(life.time,'sleep',side_effect=RuntimeError('iteration finished')), \
             mock.patch.object(life.subprocess,'run',side_effect=AssertionError('paused launch')):
            with self.assertRaisesRegex(RuntimeError,'iteration finished'):life.watch(self.bundle,{})
        c=copy.deepcopy(self.cap);c['free']=20*life.GIB-1
        c['workers']=[dict(name='/zen-score-steer-r3500-003',memory=6*life.GIB,run='jobs/fitsteer-job')]
        with mock.patch.object(life,'transition_proof'),mock.patch.object(life,'repaired_admission'), \
             mock.patch.object(life,'next_jobset',return_value='job'), \
             mock.patch.object(life,'capacity',return_value=c),mock.patch.object(life,'ssh') as ssh:
            with self.assertRaisesRegex(ValueError,'disk floor failed'):life.watch(self.bundle,{})
            ssh.assert_called_once_with('r3500','sudo -n docker stop --time 30 /zen-score-steer-r3500-003')


if __name__ == '__main__':
    unittest.main()
