#!/usr/bin/env python3
"""Explicit instrumented test probe; not a matched latency acceptance run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos'))
import speedq_run as owner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--cpu', type=int, default=2)
    args = parser.parse_args()
    assert hashlib.sha256(args.binary.read_bytes()).hexdigest() == args.sha256
    owner.CPUSETS[1] = str(args.cpu)
    args.output.mkdir(exist_ok=False)
    if args.runtime:
        runtime(args)
        return
    results = dict(kind='instrumented stage diagnostic, opt-level=2, archmage/magetypes=0.9.28; not matched release p95',
                   binary_sha256=args.sha256, geometry='1024x1024', cpuset=str(args.cpu), threads=1, warmups=2, models={})
    for model in ['diff-410', 'diff-frag-820', 'w-c']:
        quiet = owner.quiet_gate(args.output / f'{model}-quiet.jsonl')
        env = {**os.environ, 'TMPDIR':str(Path.home()/'tmp'), 'ZENSIM_FORMULA_REV':'5',
               'RAYON_NUM_THREADS':'1', 'MAPPERF_AUDIT_ROOT':str(args.root),
               'MAPPERF_AUDIT_MODEL':model, 'MAPPERF_STAGE_TIMING':'1', 'ZENSIM_FOLD_TIMING':'1'}
        log = args.output / f'{model}.log'
        with log.open('x') as out:
            subprocess.run(['/usr/bin/time', '-v', 'taskset', '-c', str(args.cpu), str(args.binary),
                            'metric::bake::mapperf_audit::declared_plan_and_stages', '--exact', '--nocapture'],
                           env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
        stages = {}
        totals = []
        scalar = []
        repeat = -1
        for line in log.read_text().splitlines():
            if 'MAPPERF_PLAN ' in line:
                plan = json.loads(line.split('MAPPERF_PLAN ',1)[1])
            elif line.startswith('MAPPERF_REPEAT '):
                repeat = int(line.split()[1])
            elif line.startswith('MAPPERF_STAGE ') and repeat >= 2:
                _, name, ns = line.split()
                stages.setdefault(name, []).append(int(ns))
            elif line.startswith('MAPPERF_TOTAL ') and repeat >= 2:
                totals.append(int(line.split()[1]))
            elif line.startswith('MAPPERF_SCALAR '):
                _, index, ns = line.split()
                if int(index) >= 2:
                    scalar.append(int(ns))
        assert len(totals) == len(scalar) == 10
        assert set(stages) == {'extract','forward','gradient','density','snapshot'}
        assert all(len(v) == 10 for v in stages.values())
        results['models'][model] = dict(plan=plan, quiet=quiet, stage_ns=stages, total_ns=totals,
                                       scalar_ns=scalar, stage_median_ms={k:statistics.median(v)/1e6 for k,v in stages.items()},
                                       total_median_ms=statistics.median(totals)/1e6,
                                       scalar_median_ms=statistics.median(scalar)/1e6,
                                       log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        (args.output / 'RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
        print(model, results['models'][model]['stage_median_ms'], flush=True)


def cpu2_preflight(log):
    # A separate assessment lane uses CPU2 in successive named processes.
    # Wait for a scheduling window before invoking the unchanged quiet owner.
    with log.open('x') as out:
        while True:
            probe=subprocess.run(['pgrep','-x','iqa-compete'],capture_output=True,text=True)
            assert probe.returncode in (0,1)
            pids=probe.stdout.split()
            out.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),iqa_compete=pids))+'\n')
            out.flush()
            if not pids:
                return
            owner.refresh_activity('waiting for CPU2 assessment lane')
            print('waiting for CPU2 assessment lane',pids,flush=True)
            time.sleep(10)


def runtime(args):
    results = dict(kind='instrumented release runtime stage diagnostic, same locked addrfix-runtime dependency closure; not paired p95',
                   binary_sha256=args.sha256, geometry='1024x1024', cpuset=str(args.cpu), tier='v4x', threads=1,
                   calls=13, warmups=3, models={})
    job = json.loads((args.root/'JOB.json').read_text())
    for model in ['diff-410','diff-frag-820','w-c']:
        binding=job['models'][model]
        os.environ.update(ZEN_S2_E34_BAKE_CANDIDATE=str(args.root/binding['path']),
                          ZEN_S2_E34_SHA_CANDIDATE=binding['sha256'],
                          ZEN_S2_E34_HIDDEN_CANDIDATE=str(binding['hidden']),
                          MAPPERF_STAGE_TIMING='1', ZEN_S2_PROFILE_CALLS='12', ZENSIM_FOLD_TIMING='1')
        records={}
        for arm, kind in [('e34_candidate_map_retained','map'),('e34_candidate','scalar')]:
            if args.cpu == 2:
                cpu2_preflight(args.output/f'{model}-{kind}-cpu2.jsonl')
            quiet=owner.quiet_gate(args.output/f'{model}-{kind}-quiet.jsonl')
            log=args.output/f'{model}-{kind}.log'
            ready=owner.worker_run(args.binary,arm,5,'1024x1024','v4x',1,log)
            if args.cpu == 2:
                assert subprocess.run(['pgrep','-x','iqa-compete'],capture_output=True).returncode==1, 'CPU2 assessment lane resumed during stage probe'
            assert ready['tier']=='v4x' and ready['threads']=='1' and ready['width']==ready['height']==1024
            stages={}
            totals=[]
            for line in log.read_text().splitlines():
                if line.startswith('MAPPERF_STAGE '):
                    _,name,ns=line.split()
                    stages.setdefault(name,[]).append(int(ns))
                elif line.startswith('MAPPERF_RUNTIME_TOTAL '):
                    _,name,ns=line.split()
                    assert name==kind
                    totals.append(int(ns))
            assert len(totals)==13 and all(len(v)==13 for v in stages.values())
            stages={k:v[3:] for k,v in stages.items()}
            totals=totals[3:]
            records[kind]=dict(quiet=quiet,ready=ready,stage_ns=stages,total_ns=totals,
                stage_median_ms={k:statistics.median(v)/1e6 for k,v in stages.items()},
                total_median_ms=statistics.median(totals)/1e6,
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        assert records['map']['ready']['score_bits']==records['scalar']['ready']['score_bits']
        results['models'][model]=records
        (args.output/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
        print(model,{k:v['stage_median_ms'] for k,v in records.items()},flush=True)


if __name__ == '__main__':
    main()
