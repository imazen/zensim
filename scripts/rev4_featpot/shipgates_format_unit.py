"""One tiny identity fixture validates lossless optimized PNG float-map storage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['binary','source','oracle','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.output/'fixture.png'
    shutil.copy2(args.source,source)
    with Image.open(source) as image:
        assert image.size==(64,64) and image.mode=='RGB'
    pairs=args.output/'pairs.tsv'
    with pairs.open('x') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['ref_path','dist_path','case_id'])
        writer.writerow([source,source,'tiny-identity-map-format-unit'])
    env={k:v for k,v in os.environ.items() if not k.startswith(('ZENSIM_','JXL_'))}
    env.update(ZENSIM_FORMULA_REV='1',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    judges=args.output/'judges'
    subprocess.run([str(args.binary),'--pairs',str(pairs),'--out-dir',str(judges),'--ssim2-map-policy','scalar-only'],env=env,check=True)
    row=json.loads((judges/'judges.jsonl').read_text())
    assert row['reference_pixels_sha256']==row['distorted_pixels_sha256']
    assert row['fixed_zensim']==100 and row['butteraugli_pnorm3']==0
    modes=[]
    for artifact in row['maps']:
        with Image.open(artifact['path']) as image:modes.append(image.mode)
    assert any(mode!='RGB' for mode in modes), 'fixture must exercise optimized PNG storage'
    strict=subprocess.run([sys.executable,str(args.oracle),'--judges',str(judges),'--output',str(args.output/'strict-not-admitted.json')],capture_output=True,env=env)
    assert strict.returncode!=0 and not (args.output/'strict-not-admitted.json').exists()
    (args.output/'strict-policy.log').write_bytes(strict.stdout+strict.stderr)
    subprocess.run([sys.executable,str(args.oracle),'--judges',str(judges),'--output',str(args.output/'LOSSLESS_ORACLE.json'),'--lossless-png-expansion'],env=env,check=True)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(scope='tiny identity artifact-format unit; not candidate evaluation',
        fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),stored_png_modes=modes,
        legacy_strict_rgb_policy_refuses=True,lossless_rgb_expansion_preserves_every_float_bit=True),indent=2)+'\n')


if __name__=='__main__':
    main()
