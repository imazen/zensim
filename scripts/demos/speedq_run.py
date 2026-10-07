#!/usr/bin/env python3
"""SPEEDQ parity, quiet-gated paired rounds and fresh-process peak RSS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import os
from pathlib import Path
import subprocess
import time
import tempfile
import shutil
from datetime import datetime, timezone

GEOMETRIES = ['64x64','128x128','256x256','512x512','1024x1024','2048x2048','4096x4096','1920x1080']
TIERS = ['v4x','v4','v3','scalar']
THREADS = [1,2,4,8,16,32]
ARMS = ['by_v2fy_r3','by_v2fy_r4','by_v2fy_r5','zensim_B','fast_ssim2','butteraugli','ssimulacra2_rs']
BAKE = '/home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s0.bin'
TRAIN_NAMES = {'zensim_mlp_train','train_hybrid.py','v2_lodo_mlp.py','zen-train','zen_train','e28_recipe.py','e28_simplex.py','bake_dial_refit'}
CPUSETS = {1:'2',2:'2,3',4:'0-3',8:'0-7',16:'0-15',32:'0-31'}


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def freeze_binary(build_log, dest):
    """Select Cargo's artifact receipt, never an old filename in target/."""
    messages=[json.loads(line) for line in Path(build_log).read_text().splitlines() if line.startswith('{')]
    assert any(m.get('reason')=='build-finished' and m.get('success') is True for m in messages), 'successful Cargo build receipt required'
    paths={m['executable'] for m in messages if m.get('reason')=='compiler-artifact' and m.get('executable') and m.get('target',{}).get('name')=='ssim2_speed_bar' and m['target'].get('kind')==['bench']}
    assert len(paths)==1, 'one unambiguous speed-matrix artifact required'
    source=Path(paths.pop());dest=Path(dest)
    with dest.open('xb') as out,source.open('rb') as inp: shutil.copyfileobj(inp,out)
    dest.chmod(source.stat().st_mode & 0o777)
    dependencies={m['target']['name']:dict(package_id=m['package_id'],features=m['features']) for m in messages if m.get('reason')=='compiler-artifact' and m.get('target',{}).get('name') in ['fast_ssim2','butteraugli','ssimulacra2','archmage','rayon'] and m['target'].get('kind')==['lib']}
    write(dest.with_name(dest.name+'.artifact.json'),dict(cargo_artifact=str(source),build_log=str(build_log),binary_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),dependencies=dependencies))
    print(dest,flush=True)


def training_processes(proc=Path('/proc')):
    found = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():
            continue
        try:
            parts = (p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        names = [Path(arg).name for arg in parts[:4] if arg]
        matches = TRAIN_NAMES.intersection(names)
        if matches:
            found.append({'pid':int(p.name),'program':sorted(matches)})
    return found


def quiet_state():
    load = float(Path('/proc/loadavg').read_text().split()[0])
    foreign = {}
    for name in ['cargo','rustc']:
        run = subprocess.run(['pgrep','-x',name],capture_output=True,text=True)
        if run.returncode not in (0,1):
            raise RuntimeError(f'pgrep {name} failed')
        foreign[name] = [int(p) for p in run.stdout.split()]
    training = training_processes()
    return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'load1':load,'foreign':foreign,'training':training,
            'admitted':load<2.0 and not any(foreign.values()) and not training}


def quiet_gate(log):
    start = time.monotonic()
    with Path(log).open('a') as f:
        while True:
            state = quiet_state()
            state['wait_seconds'] = time.monotonic()-start
            f.write(json.dumps(state)+'\n'); f.flush()
            if state['admitted']:
                return state
            print(f"quiet wait: load1={state['load1']} cargo/rustc={state['foreign']} training={state['training']}",flush=True)
            refresh_activity("quiet wait")
            time.sleep(10)


def environment(geometry, tier, threads):
    return {**os.environ, 'TMPDIR':str(Path.home()/'tmp'),
            'ZEN_S2_SPEEDQ':'1','ZEN_S2_GEOMETRY':geometry,'ZEN_S2_TIER':tier,
            'ZEN_S2_SPEEDQ_BAKE':BAKE,'RAYON_NUM_THREADS':str(threads),
            'ZENBENCH_NO_SAVE':'1','ZENBENCH_NO_CALIBRATE':'1'}


def worker_run(binary, arm, revision, geometry, tier, threads, log):
    env = environment(geometry,tier,threads)
    env.update(ZEN_S2_PARITY_FEATURES="1",ZEN_S2_SPEEDQ_WORKER=arm,ZENSIM_FORMULA_REV=str(revision),ZEN_S2_RSS_ONLY='1')
    run = subprocess.run(['taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(binary)],
                         env=env,capture_output=True,text=True)
    Path(log).write_text(run.stdout+'\nSTDERR:\n'+run.stderr)
    if run.returncode:
        raise RuntimeError(f'worker failed: {log}')
    return json.loads(run.stdout)


class LegacyToleranceFailure(RuntimeError):
    def __init__(self, difference):
        super().__init__('Rev3 feature difference exceeds documented golden tolerance')
        self.difference = difference


def compare_parity(old, rec, revision):
    if rec['input_sha256'] != old['input_sha256']:
        raise RuntimeError('input identity changed')
    if revision in (4, 5):
        if rec['score_bits'] != old['score_bits']:
            raise RuntimeError('strict Rev4/Rev5 score parity bug')
        left,right=old['feature_values'],rec['feature_values']
        assert len(left)==len(right)==420
        if any(not math.isfinite(v) for v in left+right) or any(struct.pack('>d',a)!=struct.pack('>d',b) for a,b in zip(left,right)):
            raise RuntimeError('strict Rev4/Rev5 feature-bit parity bug')
        return None
    left, right = old['feature_values'], rec['feature_values']
    assert len(left) == len(right) == 420, 'Rev3 must check all consumed features'
    assert all(math.isfinite(v) for v in left+right), 'nonfinite features cannot pass tolerance'
    differences = [abs(a-b) for a,b in zip(left,right)]
    normalized = [d/max(1e-6,1e-5*max(abs(a),abs(b))) for a,b,d in zip(left,right,differences)]
    difference = {'score_difference':rec['score']-old['score'], 'max_abs_feature_difference':max(differences),
            'max_tolerance_fraction':max(normalized), 'differing_features':sum(v!=0 for v in differences),
            'tolerance_violations':sum(v>1 for v in normalized),
            'geometry':f"{rec['width']}x{rec['height']}", 'tier':rec['tier'], 'threads':rec['threads']}
    if difference['tolerance_violations']:
        raise LegacyToleranceFailure(difference)
    return difference


def preflight(binary, dest, collect_legacy_failures=False):
    """Run the amended correctness gate before any timing segment."""
    root=Path(dest); root.mkdir(parents=True,exist_ok=False)
    records=[]; baseline={}; legacy=[]
    for geometry in GEOMETRIES:
        for threads in THREADS:
            for tier in TIERS:
                for revision in [5,4,3]:
                    tag=f'{geometry}-{tier}-t{threads}-r{revision}'
                    rec=worker_run(binary,'by_v2fy',revision,geometry,tier,threads,root/f'{tag}.log')
                    rec['revision']=revision
                    records.append(rec)
                    key=(geometry,revision)
                    old=baseline.setdefault(key,rec)
                    try:
                        difference=compare_parity(old,rec,revision)
                    except LegacyToleranceFailure as exc:
                        if not collect_legacy_failures:
                            write(root/'PARITY_FAILED.json',{'status':'STOP_REV3_FEATURE_TOLERANCE','baseline':old,'different':rec,'checked':records,'timing_started':False,'reason':str(exc)})
                            raise
                        difference = exc.difference
                    except RuntimeError as exc:
                        report={'status':'STOP_SCORE_PARITY_BUG','baseline':old,'different':rec,
                                'checked':records,'timing_started':False,'reason':str(exc)}
                        write(root/'PARITY_FAILED.json',report)
                        raise RuntimeError(f'parity failure at {tag}: {exc}') from exc
                    if difference is not None:
                        legacy.append(difference)
        print(f'parity passed {geometry}',flush=True)
    legacy_ok = not any(r['tolerance_violations'] for r in legacy)
    status = 'PASS' if legacy_ok else 'STRICT_PASS_LEGACY_TOLERANCE_FAIL'
    write(root/('PARITY_PASS.json' if legacy_ok else 'PARITY_STRICT_PASS.json'),
          {'status':status,'records':records,'strict_revisions':[4,5],
           'rev3_feature_tolerance':'max(1e-6 abs, 1e-5*scale)',
           'rev3_all_within_tolerance':legacy_ok,'rev3_differences':legacy})




def refresh_activity(activity):
    path=Path(__file__).resolve().parents[2]/'.workongoing'
    if not path.exists() or ' codex SPEEDQ' not in path.read_text():
        raise RuntimeError('SPEEDQ workspace lock ownership changed')
    path.write_text(datetime.now(timezone.utc).isoformat()+' codex SPEEDQ '+activity+'\n')


def parity_receipt(path):
    receipt=json.loads(Path(path).read_text())
    assert len(receipt['records']) == 576 and receipt['strict_revisions'] == [4,5]
    assert receipt['status'] in ('PASS','STRICT_PASS_LEGACY_TOLERANCE_FAIL')
    rows={(f"{r['width']}x{r['height']}",r['tier'],int(r['threads']),r['revision']):r for r in receipt['records']}
    expected={(g,t,n,r) for g in GEOMETRIES for t in TIERS for n in THREADS for r in [3,4,5]}
    assert set(rows)==expected, 'exact parity grid must cover every required cell'
    for g in GEOMETRIES:
        for revision in (4,5):
            records=[rows[(g,t,n,revision)] for t in TIERS for n in THREADS]
            assert len({r['score_bits'] for r in records})==1, 'strict parity receipt differs'
            assert len({r['input_sha256'] for r in records})==1, 'strict input identity differs'
            for rec in records:
                assert len(rec['feature_values'])==420 and all(math.isfinite(v) for v in rec['feature_values'])
            assert len({b''.join(struct.pack('>d',v) for v in r['feature_values']) for r in records})==1, 'strict feature-bit receipt differs'
    return rows


class TimingNoise(RuntimeError):
    """Preserve a contaminated attempt, then admit a fresh quiet segment."""


def run_segment(binary, root, geometry, tier, threads, rounds, parity, analyzer, arms=ARMS):
    tag=f'{tier}-t{threads}-{geometry}'
    dest=Path(root)/tag
    if (dest/'COMPLETE.json').exists(): return
    dest.mkdir(parents=True,exist_ok=False)
    refresh_activity('quiet gate before '+tag)
    quiet_gate(dest/'quiet-waits.jsonl')
    owners=[]; sockets={}; worker_info={}; streams=[]
    env=environment(geometry,tier,threads)
    with tempfile.TemporaryDirectory(prefix='speedq-',dir=Path.home()/'tmp') as ipc:
        try:
            for name in arms:
                arm='by_v2fy' if name.startswith('by_v2fy_r') else name
                revision=int(name[-1]) if arm=='by_v2fy' else 1
                socket=str(Path(ipc)/name)
                sockets[name]=socket
                log=(dest/f'{name}.worker.log').open('x');streams.append(log)
                worker_env={**env,'ZEN_S2_SPEEDQ_WORKER':arm,'ZENSIM_FORMULA_REV':str(revision),'ZEN_S2_SOCKET':socket}
                proc=subprocess.Popen(['taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(binary)],env=worker_env,stdout=subprocess.PIPE,stderr=log,text=True)
                owners.append(proc)
                line=proc.stdout.readline()
                if not line: raise RuntimeError(f'{name} failed before READY: {dest}')
                rec=json.loads(line);worker_info[name]=rec
                if arm=='by_v2fy':
                    old=parity[(geometry,tier,threads,revision)]
                    assert rec['score_bits']==old['score_bits'] and rec['input_sha256']==old['input_sha256'], 'STOP: worker differs from parity receipt'
            # Warmup belongs to setup. Check the quiet gate again immediately
            # before measured rounds, when every declared owner is idle.
            gate=quiet_gate(dest/'quiet-waits.jsonl')
            refresh_activity('paired timing '+tag)
            header={'quiet_gate':gate,'uptime':subprocess.check_output(['uptime'],text=True).strip(),
                    'nproc':int(subprocess.check_output(['nproc'],text=True)),
                    'governors':sorted({p.read_text().strip() for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor')}),
                    'tier':tier,'threads':threads,'cpuset':CPUSETS[threads],'geometry':geometry,
                    'rounds':rounds,'arms':arms,'worker_pids':[p.pid for p in owners],
                    'binary_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest()}
            write(dest/'header.json',header)
            raw=dest/'zenbench.json'
            bench_env={**env,'ZEN_S2_ARMS':','.join(arms),'ZEN_S2_ROUNDS':str(rounds),
                       'ZENBENCH_RESULT_PATH':str(raw),'ZEN_S2_WORKER_SOCKETS':json.dumps(sockets),
                       'ZENBENCH_LAUNCHER_PIDS':','.join(str(p.pid) for p in owners)+','+str(os.getpid())}
            print('paired timing '+tag,flush=True)
            foreign=[]
            with (dest/'zenbench.log').open('x') as log:
                proc=subprocess.Popen(['taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(binary)],env=bench_env,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    state=quiet_state()
                    if any(state['foreign'].values()) or state['training']: foreign.append(state)
                    refresh_activity('paired timing '+tag)
                    time.sleep(1)
                if proc.returncode: raise RuntimeError(f'coordinator failed: {dest}')
            inner=json.loads(raw.with_suffix('.inner.json').read_text())
            clean=not inner['zenbench_unreliable'] and all(v is True for v in inner['gate_clean']) and not foreign
            write(dest/'interference.json',{'foreign':foreign,'admitted':clean})
            if not clean: raise TimingNoise(f'gate flagged timing segment: preserve and retry {dest}')
            a=inner['paired_rounds']['by_v2fy_r4'];b=inner['paired_rounds']['by_v2fy_r5']
            assert len(a)==len(b)==rounds
            packet={'baseline':a,'candidate':b,'iterations':[1]*rounds,'timer_resolution_ns':inner['timer_resolution_ns']}
            result=subprocess.check_output([str(analyzer)],input=json.dumps([packet]),text=True)
            write(dest/'paired_analysis.json',json.loads(result)[0])
            write(dest/'COMPLETE.json',{'status':'PASS','rounds':rounds,'paired_alignment_verified':True,'zenbench_gate_clean':True})
        finally:
            for proc in owners:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=30)
            for log in streams: log.close()


def timing(binary, root, rounds, parity_path, analyzer, only=None):
    parity=parity_receipt(parity_path)
    for tier in TIERS:
        for threads in THREADS:
            for geometry in GEOMETRIES:
                tag=f'{tier}-t{threads}-{geometry}'
                if only and tag not in only: continue
                while True:
                    try:
                        run_segment(binary,root,geometry,tier,threads,rounds,parity,analyzer)
                        break
                    except TimingNoise as exc:
                        dest=Path(root)/tag
                        archived=dest.with_name(tag+f'.noise-{time.time_ns()}.bak')
                        dest.rename(archived)
                        print(f'{exc}; excluded attempt preserved at {archived}',flush=True)
                        refresh_activity('retry after contaminated '+tag)
                        time.sleep(10)


def rss(binary, root, parity_path):
    parity=parity_receipt(parity_path)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for geometry in GEOMETRIES:
        for threads in [1,32]:
            for name in ARMS:
                arm='by_v2fy' if name.startswith('by_v2fy_r') else name
                revision=int(name[-1]) if arm=='by_v2fy' else 1
                tag=f'v4x-t{threads}-{geometry}-{name}'
                if (root/f'{tag}.json').exists():continue
                refresh_activity('quiet gate before RSS '+tag)
                gate=quiet_gate(root/'quiet-waits.jsonl')
                env=environment(geometry,'v4x',threads)
                env.update(ZEN_S2_SPEEDQ_WORKER=arm,ZENSIM_FORMULA_REV=str(revision),ZEN_S2_RSS_ONLY='1')
                with (root/f'{tag}.log').open('x') as log:
                    out=subprocess.check_output(['/usr/bin/time','-v','taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(binary)],env=env,stderr=log,text=True)
                rec=json.loads(out)
                if arm=='by_v2fy':
                    old=parity[(geometry,'v4x',threads,revision)]
                    assert rec['score_bits']==old['score_bits'], 'STOP: RSS score differs from parity'
                maxrss=next(int(l.rsplit(':',1)[1]) for l in (root/f'{tag}.log').read_text().splitlines() if 'Maximum resident set size (kbytes)' in l)
                write(root/f'{tag}.json',{'geometry':geometry,'arm':name,'tier':'v4x','threads':threads,'max_rss_kib':maxrss,'quiet_gate':gate,'worker':rec})
                print('RSS '+tag+' '+str(maxrss)+' KiB',flush=True)


def collection_status(root):
    """Inspect collection markers and the most recent logged quiet check."""
    root=Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    waits=list(root.glob('timing/*/quiet-waits.jsonl'))+list(root.glob('rss/quiet-waits.jsonl'))
    latest=None
    for path in sorted(waits,key=lambda p:p.stat().st_mtime,reverse=True):
        lines=path.read_text().splitlines()
        if lines:
            latest={'path':str(path),'check':json.loads(lines[-1])}
            break
    print(json.dumps({'raw_dir':str(root),
                      'timing_completion_markers':len(list(root.glob('timing/*/COMPLETE.json'))),
                      'expected_timing_segments':len(TIERS)*len(THREADS)*len(GEOMETRIES),
                      'rss_records':len(list(root.glob('rss/v4x-*.json'))),
                      'expected_rss_records':len(GEOMETRIES)*2*len(ARMS),
                      'excluded_attempts':len(list(root.glob('timing/*.bak'))),
                      'latest_quiet_check':latest},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['parity','timing','rss','freeze','status'])
    ap.add_argument('--binary'); ap.add_argument('--dest',required=True)
    ap.add_argument('--build-log')
    ap.add_argument('--rounds',type=int,default=32)
    ap.add_argument('--parity'); ap.add_argument('--analyzer'); ap.add_argument('--only',help='comma-separated segment tags')
    ap.add_argument('--collect-legacy-failures',action='store_true',help='complete the strict grid while recording Rev3 tolerance failures; does not admit timings')
    args=ap.parse_args()
    if args.mode=='status':
        collection_status(args.dest);return
    if args.mode=='freeze':
        if not args.build_log: ap.error('--build-log required for freeze')
        freeze_binary(args.build_log,args.dest);return
    if not args.binary: ap.error('--binary required')
    if args.mode in ('timing','rss') and not args.parity: ap.error('--parity required')
    if args.mode=='timing' and not args.analyzer: ap.error('--analyzer required')
    if args.mode=='parity': preflight(args.binary,args.dest,args.collect_legacy_failures)
    elif args.mode=='timing': timing(args.binary,args.dest,args.rounds,args.parity,args.analyzer,args.only.split(',') if args.only else None)
    else: rss(args.binary,args.dest,args.parity)



if __name__=='__main__': main()
