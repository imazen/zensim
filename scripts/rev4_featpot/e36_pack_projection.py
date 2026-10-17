"""Measured grouped-transfer planning; never changes matrix or admits data."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
from e36_disk_projection import pin


def project(root,compute,out,optional_compute=None):
    root=Path(root);owner=root/'owner-svt-ssd-v1';now=time.time();cpu=json.loads(Path(compute).read_text())
    if cpu['fits'] or cpu['qualifies_data'] is not False:raise ValueError('diagnostic compute projection required')
    rates=[json.loads(s) for s in (owner/'pack-rates-v1.jsonl').read_text().splitlines()]
    release=owner/'pack-review-hold-release-v1.json'
    since=datetime.fromisoformat(json.loads(release.read_text())['utc']).timestamp() if release.exists() else 0
    candidates=[r for r in rates if r['backend']=='cache' and r['utc']-r['elapsed_seconds']>=since]
    latest=max(r['utc'] for r in candidates) if candidates else 0
    cache_rows=[r for r in candidates if r['utc']-r['elapsed_seconds']>=latest-300]
    if len(cache_rows)<2:raise ValueError('actual completed cache packs required')
    start=min(r['utc']-r['elapsed_seconds'] for r in cache_rows);end=max(r['utc'] for r in cache_rows)
    cache_rate=sum(r['raw_bytes'] for r in cache_rows)/(end-start)
    array_candidates=[r for r in rates if r['backend']=='disk1'];array_rate=27000000;array_basis='Coordinator 2026-10-17: measured 1GB dd array write27MB/s; not an end-to-end packed archival measurement.'
    if len(array_candidates)>=2:
        latest_array=max(r['utc'] for r in array_candidates)
        observed=[r for r in array_candidates if r['utc']-r['elapsed_seconds']>=latest_array-300]
        if len(observed)>=2:
            span=max(r['utc'] for r in observed)-min(r['utc']-r['elapsed_seconds'] for r in observed)
            array_rate=sum(r['raw_bytes'] for r in observed)/span;array_basis='Actual completed disk1 packs, including whole/member readback and local eviction; latest bounded300s window.'
    paths={n:root/'runs'/n/'census.json' for n in ['cost-v4','max-area-v4','modes-v4']}
    censuses={n:json.loads(p.read_text()) for n,p in paths.items()};strata={}
    def measured(codec,size,panel):
        run='modes-v4' if panel!='baseline' else 'max-area-v4' if size==4096 else 'cost-v4'
        group=censuses[run]['groups'][str(size)+':'+codec]
        return group['bundle_bytes']/group['physical'],run,group
    for key,remaining in cpu['strata'].items():
        codec,size,panel=json.loads(key);mean,run,group=measured(codec,size,panel)
        strata[key]=dict(measured_run=run,measured_cells=group['physical'],measured_bundle_bytes=group['bundle_bytes'],mean_bytes=mean,remaining_cells=remaining['physical_cells_remaining'],modeled_bytes=mean*remaining['physical_cells_remaining'])
    # Exact current local raw-file census. Includes partial/unpublished bundles
    # conservatively; it does not authorize their archival or deletion.
    local_bytes=0;local_objects=0
    for prefix in os.scandir(root/'artifacts-local-v1/blobs'):
        if not prefix.is_dir():continue
        for item in os.scandir(prefix.path):
            if item.name.endswith('.tar') and item.is_file():local_bytes+=item.stat().st_size;local_objects+=1
    claims={}
    for line in (owner/'cache-reservations-v1.jsonl').read_text().splitlines():
        row=json.loads(line);claims[row['sha256']]=row['reserved_bytes']
    used=sum(claims.values())+json.loads((owner/'pack-storage-admission-v1.json').read_text())['probe_reserved_bytes']
    for p in (owner/'pack-intents-v1').glob('*.json'):
        doc=json.loads(p.read_text())
        if doc['storage']['root']=='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache':used+=doc['reserved_bytes']
    remaining_allowance=max(0,250000000000-used)
    observations=[json.loads(line) for line in (owner/'pack-cache-space-v1.jsonl').read_text().splitlines()]
    observed_free=observations[-1]['free_bytes']
    pending_cache=sum(json.loads(p.read_text())['reserved_bytes'] for p in (owner/'pack-intents-v1').glob('*.json') if not p.with_suffix('.complete').exists() and json.loads(p.read_text())['storage']['root']=='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache')
    remaining_cache=min(remaining_allowance,max(0,observed_free-180*1024**3-pending_cache))
    if (owner/'CACHE180_STOP.json').exists():remaining_cache=0
    future=sum(s['modeled_bytes'] for s in strata.values())
    runtime=json.loads((root/'runtime-v6/runtime.json').read_text());refs={r['reference_key']:r for r in json.loads(Path(runtime['references']).read_text())['rows']}
    cut=root/'source-count-amendment-v1/binding.json'
    optional_path=Path(json.loads(cut.read_text())['phases']['additions']['manifest']['path']) if cut.exists() else owner/'additions-manifest.json'
    optional=json.loads(optional_path.read_text());optional_bytes=0;optional_by_size={}
    for job in optional:
        view=json.loads(job['kind']['view'].split(':',1)[1]);size=refs[view['reference_key']]['size'];mean,_,_=measured(view['codec'],size,view['panel'])
        if view['panel']!='baseline':raise ValueError('optional mode count needs its explicit unique-cell inventory')
        n=len(view['q_grid']);optional_bytes+=mean*n;optional_by_size[str(size)]=optional_by_size.get(str(size),0)+mean*n
    optional_cpu=None
    if optional_compute is not None:
        optional_cpu=json.loads(Path(optional_compute).read_text())
        if optional_cpu['phase']!='additions' or optional_cpu['fits']!=0 or optional_cpu['required_views']!=len(optional):raise ValueError('matched optional compute snapshot required')
        optional_bytes=0;optional_by_size={}
        for key,remaining in optional_cpu['strata'].items():
            codec,size,panel=json.loads(key);mean,_,_=measured(codec,size,panel);value=mean*remaining['physical_cells_remaining'];optional_bytes+=value;optional_by_size[str(size)]=optional_by_size.get(str(size),0)+value
    def model(extra):
        total=local_bytes+future+extra;cache=min(total,remaining_cache);array=max(0,total-cache)
        # The two routes are deliberately serialized cache FIRST. No invented
        # array pack measurement: use the coordinator's explicit 1GB dd result.
        seconds=cache/cache_rate+array/array_rate
        cpu_seconds=cpu['optimistic_modeled_seconds_remaining']+(optional_cpu['optimistic_modeled_seconds_remaining'] if extra and optional_cpu else 0)
        finish=now+max(cpu_seconds,seconds)
        return dict(modeled_remaining_bytes=total,cache_bytes=cache,array_bytes=array,storage_seconds=seconds,modeled_finish_utc=datetime.fromtimestamp(finish,timezone.utc).isoformat(),misses22Z=finish>1792274400.)
    result=dict(schema='e36-grouped-storage-planning-v1',observed_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),compute=pin(Path(compute)),
        actual_cache_pack_sample=cache_rows,measured_cache_active_window_seconds=end-start,measured_cache_raw_bytes_per_second=cache_rate,
        array_bytes_per_second=array_rate,array_rate_basis=array_basis,
        exact_local_raw_objects=local_objects,exact_local_raw_bytes=local_bytes,cache_reservations_bytes=used,remaining_cache_allowance_bytes=remaining_allowance,remaining_cache_capacity_after_floor_bytes=remaining_cache,cache_observed_free_bytes=observed_free,pending_cache_reservations_bytes=pending_cache,
        modeled_future_mandatory_bytes=future,modeled_optional_bytes=optional_bytes,optional_bytes_by_size=optional_by_size,
        registered_mandatory_fallback=model(0),registered_full_including_optional=model(optional_bytes),same_size_codec_measured_strata=strata,
        measured_censuses={n:pin(p) for n,p in paths.items()},optional_compute=pin(Path(optional_compute)) if optional_compute is not None else None,deadline_unix=1792274400.,fits=0,qualifies_data=False,source_count_change=cut.exists(),optional_manifest=pin(optional_path),
        limitations=['A planning projection, not observed future bytes or a guaranteed finish.','Cache rate is an actual short active-pack window; initialization and review hold excluded. Longer sustained rate may differ.','Before any actual disk1 pack window, array27MB/s is the coordinator dd proxy; verification/SSH/shared-load overhead may lower it.','Local raw census includes protected in-flight/unpublished bundles, conservatively charged but not eligible for eviction.','Future bytes use same-codec/same-size prebulk means; no scaling between sizes, statistical bound or future hash dedup assumption.','Optional future encodes modeled at all61 requestedq; no source/quality score selection.','Excludes future metadata/staging growth and retained features; local30/40/20GiB floors remain independently enforced.'])
    with Path(out).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['actual_cache_pack_sample','same_size_codec_measured_strata','limitations','measured_censuses']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ['root','compute','out']:parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--optional-compute',type=Path)
    a=parser.parse_args();project(a.root,a.compute,a.out,a.optional_compute)
