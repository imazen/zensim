"""Exercise the actual pinned worker on invented jobs, without training or a store."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if '@sha256:' not in args.image:
        parser.error('an immutable image digest is required')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    jobs = []
    for i in range(40):
        argv = ['engineering-only-no-fit', str(i)]
        argv_sha = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
        jobs.append(dict(
            kind=dict(kind='fit_cell', program_sha='0'*64, data_sha='1'*64,
                      argv_sha=argv_sha, argv=argv),
            cell=dict(image_path=f'engineering-only-cell-{i}', codec='fit', q=0,
                      knob_tuple_json='{}'),
            inputs=['0'*64, '1'*64, argv_sha],
            hint=dict(peak_mem_bytes=2147483648, threads=4, vram_bytes=None),
            requires=['fit-cell-v1']))
    (root/'manifest.json').write_text(json.dumps(jobs)+'\n')
    fake = root/'fake-exec'
    fake.write_text('#!/bin/sh\n'
                    'if [ "${1:-}" = capabilities ]; then printf "fit-cell-v1\\n"; exit 0; fi\n'
                    'cat >/dev/null\nprintf "engineering-only-no-fit-bytes\\n"\n')
    fake.chmod(0o755)
    cpu = min(os.sched_getaffinity(0))
    base = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--memory', '1g',
            '--memory-swap', '1g', '--cpuset-cpus', str(cpu),
            '-e', 'TMPDIR=/probe/tmp', '-v', str(root)+':/probe']
    for target in [7663, 1]:
        (root/f'target-{target}').mkdir()
        command = [*base, '--entrypoint', '/usr/bin/nice',
                   '-e', f'ZEN_CHUNK_WALL_SEC={target}',
                   '-e', 'ZEN_CORE_OVERSUBSCRIBE=1', '-e', 'ZEN_CLAIM_ORDER=manifest',
                   args.image, '-n19', 'ionice', '-c3', 'zenfleet-worker',
                   '--manifest', '/probe/manifest.json',
                   '--ledger-out', f'/probe/target-{target}/out.parquet',
                   '--blobs', f'/probe/target-{target}/blobs',
                   '--exec', '/probe/fake-exec', '--worker', 'engineering-only',
                   '--provider', 'engineering']
        with (root/f'target-{target}/worker.log').open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=stream)
    # Decode only invented ledger metadata, using the same image's PyArrow.
    check = '''import json
from pathlib import Path
import pyarrow.parquet as pq
rows={}
for target in [7663,1]:
 files=sorted(Path('/probe/target-'+str(target)).glob('*.parquet'))
 tables=[pq.read_table(p,columns=['job_id','status']).to_pylist() for p in files]
 flat=[row for table in tables for row in table]
 assert len(flat)==len({r['job_id'] for r in flat})==40
 assert all(r['status']=='done' for r in flat)
 rows[str(target)]={'sidecars':len(files),'rows_per_sidecar':[len(t) for t in tables]}
assert rows['7663']=={'sidecars':10,'rows_per_sidecar':[4]*10}
assert rows['1']=={'sidecars':40,'rows_per_sidecar':[1]*40}
print(json.dumps(rows))
'''
    result = subprocess.check_output([*base, '--entrypoint', 'python3', args.image,
                                      '-c', check], text=True)
    receipt = dict(schema='steer-worker-scheduling-probe-v1', image=args.image,
                   scientific_fits=0, network='none', registered_results_touched=False,
                   argv_and_data='invented; no trainer or target file invoked',
                   measurements=json.loads(result))
    (root/'PASS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
