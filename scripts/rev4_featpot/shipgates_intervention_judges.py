"""Complete all-output decode oracles, fixed judges and native control validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root','oracle','judge','analyzer','map-oracle']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.root/'measurements.jsonl').read_text().splitlines()]
    assert len(rows) == json.loads((args.root/'COMPLETE.json').read_text())['full_encodes']
    oracle_rows = []
    for row in rows:
        path = Path(row['cell']) / (row['work']['name']+'.jxl')
        assert sha(path) == row['work']['encoded_sha256']
        result = subprocess.run([str(args.oracle),str(path),'--disable_output','--num_threads=1'],capture_output=True,check=True)
        oracle_rows.append(dict(path=str(path),sha256=sha(path),status='success',stderr=result.stderr.decode()))
    with (args.root/'ALL_OUTPUTS_LIBJXL.json').open('x') as stream:
        json.dump(dict(oracle_sha256=sha(args.oracle),outputs=oracle_rows),stream,indent=2)
        stream.write('\n')
    judges = args.root/'fixed-judges'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('ZENSIM_','JXL_','ZENJPEG_ZQ_','ZENWEBP_ZQ_','AVIF_ZENSIM_'))}
    env.update(ZENSIM_FORMULA_REV='1',RAYON_NUM_THREADS='1')
    subprocess.run([str(args.judge),'--pairs',str(args.root/'judge_pairs.tsv'),'--out-dir',str(judges),'--ssim2-map-policy','scalar-only'],env=env,check=True)
    subprocess.run([sys.executable,str(args.map_oracle),'--judges',str(judges),'--output',str(judges/'INDEPENDENT_MAP_PIXEL_ORACLE.json'),'--lossless-png-expansion'],env=env,check=True)
    subprocess.run([sys.executable,str(args.analyzer),'--interventions',str(args.root),'--fixed-judges',str(judges)],env=env,check=True)
    print('All native control outputs independently decoded and judged:',len(rows),flush=True)


if __name__ == '__main__':
    main()
