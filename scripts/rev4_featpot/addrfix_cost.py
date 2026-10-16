#!/usr/bin/env python3
"""Bind matched complete-map workers to SPEEDQ's existing measurement owner."""
import argparse
import json
import os
from pathlib import Path
import sys
from peergates import read, sha, write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demos'))
import speedq_run as owner

ARMS=['e34_butter_map','e34_candidate_map_retained','e34_butter_scalar','e34_candidate']
RSS_ARMS=ARMS+['e34_butter_baseline','e34_candidate_baseline']
GRID=[(g,'v4x',1) for g in ['1024x1024','2048x2048']]
ANALYZER_SHA='798d938626e31c54e524da56687c468f7178d1629b4ee4f268692a290ccbba82'


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parity','timing','rss'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True)
    a=p.parse_args();root=a.root;job=read(root/'JOB.json');binding=job['models'][a.model]
    binary=root/'bin/speedq';analyzer=root/'bin/paired-rounds-analyzer'
    assert sha(binary)==job['binary_sha256'] and sha(analyzer)==ANALYZER_SHA
    model=root/binding['path'];assert sha(model)==binding['sha256']
    for label in ['CANDIDATE','CONTROL']:
        os.environ[f'ZEN_S2_E34_BAKE_{label}']=str(model)
        os.environ[f'ZEN_S2_E34_SHA_{label}']=binding['sha256']
        os.environ[f'ZEN_S2_E34_HIDDEN_{label}']=str(binding['hidden'])
    os.environ.update(SPEEDQ_LOCK_OWNER='codex-ADDRFIX',ZENSIM_NEIGHBOUR_EXACT='1')
    os.environ.pop('ZENSIM_MAX_TIER',None)
    dest=root/'results'/a.model;dest.mkdir(parents=True,exist_ok=True)
    parity=dest/'PREFLIGHT_PASS.json'
    def load(path):
        data=read(path);assert data['status']=='PASS' and data['binary_sha256']==sha(binary)
        records={(f"{r['width']}x{r['height']}",r['arm']):r for r in data['records']}
        assert len(records)==len(GRID)*len(RSS_ARMS)
        return records
    def ready(rows,name,g,t,n,rec):
        old=rows[g,name]
        assert (rec['arm'],rec['tier'],int(rec['threads']),rec['width'],rec['height'])==(name,t,n,*map(int,g.split('x')))
        assert all(rec[k]==old[k] for k in ['input_sha256','score_bits','model'])
        if name.startswith('e34_candidate'):assert rec['model']['source_sha256']==binding['sha256']
    if a.mode=='parity':
        assert sha(root/binding['golden'])==binding['golden_sha256']
        records=[];oracles=[];old=read(root/binding['golden'])['records']
        for g,t,n in GRID:
            for arm in RSS_ARMS:
                rec=owner.worker_run(binary,arm,5,g,t,n,dest/f'{g}-{arm}.log');records.append(rec)
            rows={r['arm']:r for r in records if r['width']==int(g.split('x')[0])}
            assert len({r['input_sha256'] for r in rows.values()})==1
            assert rows['e34_candidate']['score_bits']==rows['e34_candidate_map_retained']['score_bits']
            prior=next(r for r in old if r['width']==int(g.split('x')[0]) and r['tier']==t and r['arm']=='e34_candidate_map')
            assert all(rows['e34_candidate_map_retained'][k]==prior[k] for k in ['input_sha256','score_bits'])
            oracle=owner.worker_run(binary,'e34_butter_parity',5,g,t,n,dest/f'{g}-butter-parity.log')
            assert oracle['input_sha256']==rows['e34_butter_map']['input_sha256']
            assert oracle['score_bits']==rows['e34_butter_map']['score_bits']
            oracles.append(oracle)
        write(parity,dict(status='PASS',binary_sha256=sha(binary),records=records,cache_parity=oracles))
        load(parity)
    elif a.mode=='timing':
        owner.timing(binary,dest/'timing',32,parity,analyzer,arms=ARMS,
            cells=GRID,baseline='e34_butter_map',parity_loader=load,
            ready_check=lambda *args:ready(load(parity),*args),
            lock=Path.home()/'tmp/zensim-paper/rev4/heavy.lock')
    else:
        for repeat in range(3):
            with owner.segment_lock(Path.home()/'tmp/zensim-paper/rev4/heavy.lock'):
                owner.rss(binary,dest/f'rss-{repeat}',parity,arms=RSS_ARMS,parity_loader=load,
                    ready_check=lambda *args:ready(load(parity),*args),
                    geometries=[g for g,_,_ in GRID],thread_counts=[1])

if __name__=='__main__':main()
