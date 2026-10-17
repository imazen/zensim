"""Planning model for E36 storage evacuation; never admits or truncates data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def project(root,compute,out):
    now=time.time()
    projection=json.loads(compute.read_text())
    if projection['qualifies_data'] is not False or projection['fits']!=0:
        raise ValueError('diagnostic projection required')
    names=['cost-v4','max-area-v4','modes-v4']
    paths={name:root/'runs'/name/'census.json' for name in names}
    censuses={name:json.loads(path.read_text()) for name,path in paths.items()}
    strata={}
    for key,remaining in projection['strata'].items():
        codec,size,panel=json.loads(key)
        name='modes-v4' if panel!='baseline' else 'max-area-v4' if size==4096 else 'cost-v4'
        measured=censuses[name]['groups'][str(size)+':'+codec]
        if measured['physical']<=0:
            raise ValueError('same-codec/same-size measured bytes required')
        mean=measured['bundle_bytes']/measured['physical']
        strata[key]=dict(measured_run=name,measured_physical_cells=measured['physical'],
            measured_bundle_bytes=measured['bundle_bytes'],mean_bundle_bytes=mean,
            remaining_physical_requests=remaining['physical_cells_remaining'],
            modeled_new_bundle_bytes=mean*remaining['physical_cells_remaining'])
    telemetry=root/'owner-svt-ssd-v1/disk-rates.jsonl'
    # Snapshot only complete lines; active monitor appends to the live journal.
    capture=telemetry.read_bytes()
    lines=capture.splitlines(keepends=True)
    rows=[json.loads(line) for line in lines if line.endswith(b'\n')]
    latest=rows[-1]['utc']
    windows=[row for row in rows if 'window_seconds' in row and row['utc']>=latest-600]
    seconds=sum(row['window_seconds'] for row in windows)
    if not seconds or now-latest>90:
        raise ValueError('fresh measured transfer windows required')
    rate=sum(row['verified_mirror_bytes_per_second']*row['window_seconds'] for row in windows)/seconds
    if rate<=0:raise ValueError('positive measured transfer rate required')
    total=sum(item['modeled_new_bundle_bytes'] for item in strata.values())
    free=shutil.disk_usage(root).free
    evacuation=max(0,total-max(0,free-30*1024**3))
    finish=now+max(projection['optimistic_modeled_seconds_remaining'],evacuation/rate)
    snapshot=out.with_suffix('.rates.json')
    with snapshot.open('x') as stream:json.dump(windows,stream,indent=2)
    result=dict(schema='e36-disk-planning-projection-v1',observed_unix=now,
        observed_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),compute=pin(compute),
        measured_censuses={name:pin(path) for name,path in paths.items()},rate_snapshot=pin(snapshot),
        measured_rate_window_seconds=seconds,verified_mirror_bytes_per_second=rate,
        free_bytes=free,modeled_new_bundle_bytes=total,modeled_new_bytes_requiring_eviction=evacuation,
        modeled_finish_utc=datetime.fromtimestamp(finish,timezone.utc).isoformat(),
        misses_registered_deadline_in_this_model=finish>projection['deadline_unix'],
        deadline_unix=projection['deadline_unix'],strata=strata,fits=0,qualifies_data=False,
        limitations=['Same-codec/same-size prebulk physical-bundle mean; no scaling between sizes.',
            'Not measured future bytes, a statistical bound, or a guaranteed finish.',
            'Excludes existing eligible backlog, staging/features, per-view overhead, and optional additions.',
            'Content/quality variation, future hash dedup, and pipeline overlap are not fully modeled.',
            'Recent measured mirror rate can change with source size and shared NAS load.'])
    with out.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key not in ['strata','limitations']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','compute','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    project(args.root,args.compute,args.out)
