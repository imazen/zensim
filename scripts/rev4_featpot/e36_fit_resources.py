"""Measured diagnostic RSS -> explicit fit caps and native scheduling hints."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
from e36_store import sha

GIB=1024**3
HOSTS=('tower','i265','i270','r3500')
OBSERVE="awk '/^(MemTotal|MemAvailable):/ {printf \"%s %.0f\\n\",$1,$2*1024}' /proc/meminfo; nproc; date -u +%s"

def observe(host):
    command=['ssh','-o','ConnectTimeout=10',host,OBSERVE]
    lines=subprocess.check_output(command,text=True).splitlines()
    values=dict(line.split() for line in lines[:2])
    return command,dict(utc=int(lines[3]),threads=int(lines[2]),total_bytes=int(values['MemTotal:']),available_bytes=int(values['MemAvailable:']))

def cap(peak_kib):
    if type(peak_kib) is not int or peak_kib<=0:raise ValueError('positive measured maximum RSS required')
    peak=peak_kib*1024;margin=max((peak+1)//2,512*1024**2);quantum=256*1024**2
    return ((peak+margin+quantum-1)//quantum)*quantum,margin

def slots(host,observed,fit_cap):
    reserve=(12 if host=='tower' else 8)*GIB
    memory=max(0,observed['available_bytes']-reserve)
    cpus=min(24,max(0,observed['threads']-8)) if host=='tower' else max(0,observed['threads']-4)
    count=min(memory//fit_cap,cpus//2)
    return dict(reserve_bytes=reserve,available_beyond_reserve_bytes=memory,allowed_cpus=cpus,threads_per_fit=2,slots=count,aggregate_fit_caps_bytes=count*fit_cap,requires_fresh_preclaim_admission=True)

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def prepare(root,packet,resource,out):
    root=Path(root);packet=Path(packet);resource=Path(resource);out=Path(out)
    measured=re.findall(r'^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$',resource.read_text(),flags=re.M)
    if len(measured)!=1:raise ValueError('one actual native /usr/bin/time maximum RSS required')
    peak=int(measured[0]);fit_cap,margin=cap(peak);out.mkdir();observations={};packing={}
    for host in HOSTS:
        command,observed=observe(host);observations[host]=observed;packing[host]=slots(host,observed,fit_cap)
        write(out/(host+'.json'),dict(command=command,observed=observed,packing=packing[host]))
    plan=json.loads((packet/'E36_FIT_PLAN.json').read_text());jobs=[];manifests=[]
    for phase in plan['manifests'].values():
        path=Path(phase['manifest_path'])
        if sha(path)!=phase['manifest_sha256']:raise ValueError('scientific declaration changed')
        manifests.append(dict(path=str(path),sha256=sha(path)));jobs.extend(json.loads(path.read_text()))
    if len(jobs)!=129 or plan['fit_release'] is not False:raise ValueError('exact unreleased129 scientific cells required')
    original=out/'original.manifest.json';scheduled=out/'scheduled.manifest.json';write(original,jobs)
    changed=copy.deepcopy(jobs)
    for job in changed:job['hint']=dict(peak_mem_bytes=fit_cap,threads=2,vram_bytes=None)
    write(scheduled,changed);ids=[]
    for path in [original,scheduled]:
        target=path.with_suffix('.ids.tsv')
        with target.open('x') as stream:subprocess.run([str(root/'bin/zenfleet-ctl'),'ids','--manifest',str(path)],stdout=stream,check=True)
        ids.append(target)
    if ids[0].read_bytes()!=ids[1].read_bytes():raise ValueError('resource hints changed scientific native identity')
    result=dict(schema='e36-measured-fit-resource-plan-v1',measurement=dict(path=str(resource),sha256=sha(resource),peak_rss_kib=peak,scope='native smoke command maximum RSS; not a full-run or aggregate-process memory bound'),margin_bytes=margin,margin_policy='max(50percent measured peak,512MiB), cap rounded up to256MiB',per_fit_memory_cap_bytes=fit_cap,rayon_threads=2,hosts=packing,observations=observations,manifests=manifests,scheduled_manifest=dict(path=str(scheduled),sha256=sha(scheduled)),native_ids_unchanged=True,fit_release=False,requires_coordinator_steer_drain_and_go=True)
    write(out/'RESOURCE_PLAN.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','packet','resource','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.packet,a.resource,a.out)))
