"""Local sequential source phases for already-frozen engineering encodes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from shipgates_admission import run_with_memory_admission


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['encodes', 'binary', 'heavy-lock', 'run-heavy', 'output', 'reuse-first', 'map-oracle']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reuse-source-root',type=Path)
    parser.add_argument('--reuse-verifications',type=Path)
    parser.add_argument('--encoder-pid', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((args.encodes / 'DESIGN.json').read_text())
    write(args.output / 'INPUTS.json', dict(design_sha256=sha(args.encodes/'DESIGN.json'),
        judge_binary_sha256=sha(args.binary), runner_sha256=sha(__file__),
        ssim2_map_policy='scalar-only', sources=design['sources']))
    frozen_oracle=args.output/'map_oracle.py'
    frozen_oracle.write_bytes(args.map_oracle.read_bytes())
    write(args.output/'ORACLE_PIN.json',dict(sha256=sha(frozen_oracle)))
    admission_adapter=args.output/"shipgates_admission.py"
    admission_adapter.write_bytes(Path(__file__).with_name("shipgates_admission.py").read_bytes())
    write(args.output/"ADMISSION_POLICY.json",dict(source_sha256=sha(admission_adapter),
        rule="wait30s only for exact run-heavy low-memory refusal before output creation; all execution failures fatal"))
    reused = {}
    if args.reuse_verifications:
        reuse = json.loads(args.reuse_verifications.read_text())
        assert reuse['schema']=='shipgates-verified-judge-reuse-v1'
        assert reuse['source_count']==len(reuse['sources'])
        reused={r['source_id']:r for r in reuse['sources']}
        assert set(reused)==set(range(reuse['source_count'])) and reuse['source_count']<=design['sources']
        write(args.output/'VERIFICATION_REUSE_PIN.json',dict(path=str(args.reuse_verifications),sha256=sha(args.reuse_verifications)))
        reuse_adapter=args.output/'shipgates_verified_reuse.py'
        reuse_adapter.write_bytes(Path(__file__).with_name('shipgates_verified_reuse.py').read_bytes())
        write(args.output/'VERIFICATION_REUSE_OWNER.json',dict(sha256=sha(reuse_adapter)))
    packets = []
    for source_id in range(design['sources']):
        source = args.encodes / f'source-{source_id:02d}'
        while not (source/'COMPLETE.json').is_file():
            os.kill(args.encoder_pid, 0)  # Missing producer is a failure, never a skip.
            print(f'waiting for complete encoded source{source_id:02d}', flush=True)
            time.sleep(30)
        cases = [json.loads(line) for line in (source/'cells.jsonl').read_text().splitlines()]
        assert len(cases) == json.loads((source/'COMPLETE.json').read_text())['cells'] == 168
        expected = {r['case']:(r['reference'],r['decoded']) for r in cases}
        assert len(expected) == len(cases)
        dest = args.output / f'source-{source_id:02d}'
        prior=args.reuse_source_root/f'source-{source_id:02d}' if args.reuse_source_root else None
        if source_id in reused:
            dest=Path(json.loads(Path(reused[source_id]['receipt']).read_text())['path'])
        elif source_id == 0:
            dest = args.reuse_first
        elif prior and (prior/'COMPLETE.json').is_file():
            dest=prior
        else:
            free = os.statvfs(str(Path.home()))
            assert free.f_bavail*free.f_frsize >= 20*1024**3, 'disk below20GiB'
            pairs = args.output / f'source-{source_id:02d}.tsv'
            with pairs.open('x') as stream:
                writer = csv.writer(stream, delimiter='\t')
                writer.writerow(['ref_path','dist_path','case_id'])
                writer.writerows((r['reference'],r['decoded'],r['case']) for r in cases)
            env = {k:v for k,v in os.environ.items() if not k.startswith(('ZENSIM_','JXL_','ZENJPEG_ZQ_','ZENWEBP_ZQ_','AVIF_ZENSIM_'))}
            env.update(ZENSIM_FORMULA_REV='1',RAYON_NUM_THREADS='1')
            command = ['flock',str(args.heavy_lock),str(args.run_heavy),'--mem','16G','--jobs','1','--',str(args.binary),
                '--pairs',str(pairs),'--out-dir',str(dest),'--ssim2-map-policy','scalar-only']
            write(args.output/f'source-{source_id:02d}-command.json',command)
            print(f'judge source{source_id:02d}: queued',flush=True)
            run_with_memory_admission(command,args.output/f'source-{source_id:02d}.log',dest,env=env)
        pins=json.loads((dest/'INPUTS.json').read_text())
        assert pins['driver_sha256']==sha(args.binary) and set(pins['requested_metrics'])=={'ssim2','butter','zensim'}
        assert pins['formula_revision_environment']=='1' and pins['ssim2_map_policy']=='scalar-only'
        assert json.loads((dest/'COMPLETE.json').read_text())['pairs'] == len(cases)
        judged = [json.loads(line) for line in (dest/'judges.jsonl').read_text().splitlines()]
        assert len(judged) == len(expected)
        assert {r['input']['case_id']:(r['ref_path'],r['dist_path']) for r in judged} == expected
        oracle_output=args.output/f'source-{source_id:02d}-MAP_ORACLE.json'
        if source_id in reused:
            from shipgates_verified_reuse import restore
            restore(reused[source_id],dest,sha(args.binary),sha(frozen_oracle),oracle_output,len(judged))
            print(f'judge source{source_id:02d}: certified packet/maps rehashed and reused',flush=True)
        else:
            oracle_command=['flock',str(args.heavy_lock),str(args.run_heavy),'--mem','16G','--jobs','1','--',sys.executable,str(frozen_oracle),
                '--judges',str(dest),'--output',str(oracle_output),'--lossless-png-expansion']
            print(f'judge source{source_id:02d}: independent map oracle queued',flush=True)
            run_with_memory_admission(oracle_command,args.output/f'source-{source_id:02d}-map-oracle.log',oracle_output)
        packets.append(dest/'judges.jsonl')
        write(args.output/f'source-{source_id:02d}-VERIFIED.json',dict(path=str(dest),pairs=len(judged),sha256=sha(dest/'judges.jsonl')))
        print(f'judge source{source_id:02d}: {len(judged)} verified rows',flush=True)
    with (args.output/'judges.jsonl').open('x') as stream:
        for packet in packets:
            stream.write(packet.read_text())
    write(args.output/'COMPLETE.json',dict(pairs=design['sources']*168,sources=design['sources']))


if __name__ == '__main__':
    main()
