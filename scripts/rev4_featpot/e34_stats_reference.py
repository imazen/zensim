"""Independent E33 seed-unit parity, using only the pinned stored summary.

This calls the existing Python reference; it never opens prediction or label
payloads. New E34 inference belongs to panel e34 and zenmetrics-stats.
"""
import argparse
import json
from pathlib import Path
from scipy.stats import t
from e24_rev5 import e29_seed_stat
from v40_score import sdr_decision
from v2_common import sha

SUMMARY=Path('/mnt/v/output/zensim/e33-impl-2026-10-09/assessment-e33/e33_e21.json')
PIN='4a3c58b53d6da2ff46c3ef504e78067bff55a8c7a8fb54a1d55c7aa045bf6bf2'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    assert sha(SUMMARY)==PIN
    old=json.loads(SUMMARY.read_text())
    out={}
    for arm in ('a','c'):
        row=old['decisions'][arm]
        got=sdr_decision(row['source_seed_deltas'],row['w2_source_seed_deltas'],'e33')
        assert got==row,(arm,got,row)
        out[arm]={k:got[k] for k in ('signed','w2')}
    stat=e29_seed_stat(old['c_vs_a']['source_seed_deltas'])
    statistic=stat['delta']/stat['se']
    cvsa=dict(signed=stat,t=statistic,df=9,p_one_sided=float(t.sf(statistic,df=9)))
    for k,v in cvsa.items():assert v==old['c_vs_a'][k],k
    out['c_vs_a']=cvsa
    out['input_sha256']=PIN
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
