"""Stage and invoke one bounded cell through the native worker on i270."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from e36_store import sha

HOST='i270'
BASE='ghcr.io/imazen/zenfleet-worker@sha256:1cc5ab665d53025d320e5d15cd6a833c53afb395775583334f7c8e07c122efd0'


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def remote(command):
    return subprocess.run(['ssh','-o','BatchMode=yes',HOST,command],check=True,text=True,capture_output=True).stdout


def prepare(out):
    out=Path(out);prepared=json.loads((out/'PREPARED.json').read_text())
    if prepared['planned_host']!=HOST or prepared['policy']['wall_cap_sec']!=900 or prepared['initial_memory_cap_bytes']!=8*1024**3 or prepared['rayon_threads']!=2:raise ValueError('exact authorized bounded smoke required')
    for key in ['program','data','manifest','native_ids']:
        if sha(prepared[key]['path'])!=prepared[key]['sha256']:raise ValueError('prepared smoke changed')
    admitted=json.loads((out/'program-admission/receipt.json').read_text())
    if admitted['program_sha256']!=prepared['program']['sha256'] or admitted['all_inventory_hashes_match'] is not True:raise ValueError('actual smoke program admission required')
    jobs=json.loads(Path(prepared['manifest']['path']).read_text())
    if len(jobs)!=1 or jobs[0]['kind']['kind']!='fit_cell' or '--local-smoke-budget' not in jobs[0]['kind']['argv'] or jobs[0]['kind']['argv'][-2:]!=['--local-smoke-budget','2:128']:raise ValueError('one native diagnostic only')
    jobs[0]['hint']=dict(peak_mem_bytes=8*1024**3,threads=2,vram_bytes=None)
    scheduled=out/'scheduled.manifest.json';write(scheduled,jobs)
    root=Path(prepared['manifest']['path']).parents[1]
    ids=subprocess.check_output([str(root/'bin/zenfleet-ctl'),'ids','--manifest',str(scheduled)],text=True)
    if ids!=Path(prepared['native_ids']['path']).read_text():raise ValueError('resource hint changed native ID')
    digest=prepared['program']['sha256'];name='e36-memory-smoke-'+digest[:12];directory='/home/lilith/tmp/'+name;image='e36-memory-smoke:'+digest[:12]
    script=out/'run-i270.sh'
    q=shlex.quote
    lines=['#!/usr/bin/env bash','set -euo pipefail','cd '+q(directory),
        'test "$(awk \'/MemAvailable:/{print $2}\' /proc/meminfo)" -ge 16777216',
        'df -h /home','docker ps --format "{{.Names}} {{.Status}}"',
        'test "$(df --output=avail -B1 /home | sed -n \'2p\')" -ge 42949672960',
        'sha256sum -c INPUT_SHA256SUMS',
        'mkdir -p work scratch',
        '/home/lilith/work/claudehints/scripts/run-heavy --mem 8G --jobs 2 -- docker build --pull=false --build-arg FIT_PROGRAM_SHA='+digest+' -t '+q(image)+' .',
        'docker image inspect '+q(image)+' > image-inspect.json',
        '/home/lilith/work/claudehints/scripts/run-heavy --mem 10G --jobs 2 -- docker run --name '+q(name)+' --network none --memory 8g --memory-swap 8g --cpuset-cpus 0-1 --cpu-shares 256 --pids-limit 512 --security-opt no-new-privileges --mount type=bind,src='+directory+'/work,dst=/work --mount type=bind,src='+directory+'/scratch,dst=/scratch --mount type=bind,src='+directory+',dst=/input,readonly -e TMPDIR=/scratch/tmp -e ZEN_FIT_DATA_LOCAL=/input/data.tar.gz -e ZEN_FIT_RAYON_THREADS=2 -e RAYON_NUM_THREADS=2 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e ZENSIM_MAX_TIER=v3 -e ZEN_RAM_BYTES=8589934592 -e ZEN_CHUNK_REF_CONCURRENCY=1 -e ZEN_DEBUG_ADMIT=1 --entrypoint /usr/bin/nice '+q(image)+' -n19 /usr/bin/ionice -c3 /usr/local/bin/zenfleet-worker --manifest /input/scheduled.manifest.json --ledger-out /work/ledger.parquet --blobs /work/blobs --exec /usr/local/bin/fit-cell-exec --worker e36-i270-memory-smoke --provider local --max-attempts 1',
        'docker inspect '+q(name)+' > container-inspect.json']
    # run-heavy may wait for another host job: check only after that wait.
    invocation=lines[-2]
    prefix='/home/lilith/work/claudehints/scripts/run-heavy --mem 10G --jobs 2 -- '
    if not invocation.startswith(prefix):raise ValueError('diagnostic wrapper boundary changed')
    check='test "$(awk \'/MemAvailable:/{print $2}\' /proc/meminfo)" -ge 16777216'
    lines[-2]=prefix+'bash -c '+q(check+' && exec '+invocation[len(prefix):])
    script.write_text('\n'.join(lines)+'\n')
    docker=Path(__file__).parents[2]/'docker/Dockerfile.e36fit'
    files={'program.tar.gz':Path(prepared['program']['path']),'data.tar.gz':Path(prepared['data']['path']),'scheduled.manifest.json':scheduled,'Dockerfile':docker,'run-i270.sh':script}
    sums=out/'INPUT_SHA256SUMS';sums.write_text(''.join(sha(path)+'  '+name+'\n' for name,path in files.items()));files['INPUT_SHA256SUMS']=sums
    write(out/'LAUNCH_PLAN.json',dict(schema='e36-one-native-diagnostic-launch-v1',host=HOST,directory=directory,image=image,container=name,base=BASE,prepared_sha256=sha(out/'PREPARED.json'),program_admission_sha256=sha(out/'program-admission/receipt.json'),files={name:dict(path=str(path),sha256=sha(path)) for name,path in files.items()},native_ids_unchanged=True,scientific_fits=0,diagnostic_cells=1,container_memory_bytes=8*1024**3,threads=2,wall_cap_sec=900))


def stage(out):
    out=Path(out);plan=json.loads((out/'LAUNCH_PLAN.json').read_text());directory=plan['directory']
    remote('mkdir '+shlex.quote(directory))
    for name,pin in plan['files'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('launch input changed')
        subprocess.run(['scp','-q',pin['path'],HOST+':'+directory+'/'+name],check=True)
    observed=remote('cd '+shlex.quote(directory)+' && sha256sum -c INPUT_SHA256SUMS')
    write(out/'STAGED.json',dict(host=HOST,directory=directory,sha_checks=observed,utc=time.time(),labels_read=False,scientific_fits=0))


def run(out):
    out=Path(out);plan=json.loads((out/'LAUNCH_PLAN.json').read_text())
    staged=json.loads((out/'STAGED.json').read_text())
    if plan['host']!=HOST or staged['host']!=HOST or staged['directory']!=plan['directory']:raise ValueError('fixed admitted smoke receiver required')
    for pin in plan['files'].values():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('staged launch input changed')
    if (out/'RUN_EXIT.json').exists():raise ValueError('retain prior diagnostic execution; no automatic retry')
    start=time.time()
    result=subprocess.run(['ssh','-o','BatchMode=yes',HOST,'bash '+shlex.quote(plan['directory']+'/run-i270.sh')])
    write(out/'RUN_EXIT.json',dict(host=HOST,returncode=result.returncode,started_unix=start,closed_unix=time.time(),scientific_fits=0))
    destination=out/'native-run';destination.mkdir()
    subprocess.run(['scp','-qr',HOST+':'+plan['directory']+'/work',str(destination)],check=True)
    inspection=remote('docker inspect '+shlex.quote(plan['container']))
    (destination/'container-inspect.json').write_text(inspection)
    if result.returncode:raise RuntimeError('bounded native smoke did not succeed; retained exact ledger and container')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','stage','run']);parser.add_argument('out',type=Path);args=parser.parse_args()
    dict(prepare=prepare,stage=stage,run=run)[args.action](args.out)
