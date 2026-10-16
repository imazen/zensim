"""Native metadata/opaque-RGB admission with independent full-pixel readback."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_admission(args):
    manifest=json.loads(args.sources.read_text())
    pins=json.loads((args.output/'INPUTS.json').read_text())
    done=json.loads((args.output/'COMPLETE.json').read_text())
    assert pins['manifest_sha256']==sha(args.sources) and pins['native_binary_sha256']==sha(args.binary)
    assert done['sources']==len(manifest['sources'])==len(done['receipts'])==24
    assert done['all_original_rgb_bytes_preserved'] and done['no_experiment_inputs_replaced']
    for index,(source,record) in enumerate(zip(manifest['sources'],done['receipts'])):
        assert record['source']==source and sha(Path(source['path']))==source['sha256']
        recipe=args.output/f'source-{index:02d}-recipe.json'
        oracle=args.output/f'source-{index:02d}-oracle.json'
        assert sha(recipe)==record['recipe_sha256'] and sha(oracle)==record['oracle_sha256']
        preparation=json.loads(recipe.read_text());check=json.loads(oracle.read_text())
        assert preparation['source_png_bit_depth']==preparation['output_png_bit_depth']==8
        assert check['changed_channels']==0 and check['all_rgb_bytes_identical'] and check['opaque'] and check['dimensions_match']
    admission=dict(source_manifest_sha256=sha(args.sources),png8_binary_sha256=sha(args.binary),sources=24,
        receipt=str(args.output/'COMPLETE.json'),receipt_sha256=sha(args.output/'COMPLETE.json'))
    marker=args.sources.parent/'ADMISSION.json'
    with marker.open('x') as stream:
        json.dump(admission,stream,indent=2);stream.write('\n')
    print('All24source pixel/metadata receipts bound to immutable manifest',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sources','binary','oracle','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument("--bind-only",action="store_true")
    args = parser.parse_args()
    if args.bind_only:
        bind_admission(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    manifest = json.loads(args.sources.read_text())
    pins=dict(manifest_sha256=sha(args.sources),native_binary_sha256=sha(args.binary),
        independent_oracle_sha256=sha(args.oracle),sources=len(manifest['sources']))
    (args.output/'INPUTS.json').write_text(json.dumps(pins,indent=2)+'\n')
    receipts=[]
    for index,source in enumerate(manifest['sources']):
        original=Path(source['path'])
        assert sha(original)==source['sha256']
        prepared=args.output/f'source-{index:02d}.png'
        recipe=args.output/f'source-{index:02d}-recipe.json'
        oracle=args.output/f'source-{index:02d}-oracle.json'
        with recipe.open('x') as stream:
            subprocess.run([str(args.binary),'--expand-png8',str(original),str(prepared)],stdout=stream,check=True)
        result=json.loads(recipe.read_text())
        assert result['source_png_bit_depth']==8 and result['output_png_bit_depth']==8
        with Image.open(original) as image:
            assert result['crop']==dict(x=0,y=0,width=image.width,height=image.height), 'unexpected source alpha/crop'
        subprocess.run([sys.executable,str(args.oracle),'--receipt',str(recipe),'--output',str(oracle)],check=True)
        assert result['input_sha256']==source['sha256'] and result['output_sha256']==sha(prepared)
        receipts.append(dict(source=source,recipe_sha256=sha(recipe),oracle_sha256=sha(oracle),
            prepared_png_sha256=sha(prepared),source_rgb_sha256=result['output_rgb_sha256']))
        print(f"source admission{index:02d} {source['name']}: native contract and every RGB byte verified",flush=True)
    assert len(receipts)==len(manifest['sources'])==24
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(sources=24,all_original_rgb_bytes_preserved=True,
        no_experiment_inputs_replaced=True,receipts=receipts),indent=2)+'\n')
    bind_admission(args)


if __name__ == '__main__':
    main()
