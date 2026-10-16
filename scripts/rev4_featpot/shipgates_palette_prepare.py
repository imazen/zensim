"""Lossless indexed-PNG expansion through the private first-party PNG8 owner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['sources','binary','tool-pin','pixel-oracle','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-id',type=int,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.sources.read_text())
    pin=json.loads(args.tool_pin.read_text())
    assert sha(args.binary)==pin['binary_sha256']
    source=manifest['sources'][args.source_id]
    assert sha(source['path'])==source['sha256']
    with Image.open(source['path']) as image:
        assert image.mode=='P', 'this preparation is only for indexed PNG'
        width,height=image.size
    prepared=args.output/'prepared.png'
    receipt=args.output/'RECIPE.json'
    command=[str(args.binary),'--expand-png8',source['path'],str(prepared)]
    (args.output/'COMMAND.json').write_text(json.dumps(dict(command=command,
        source_manifest_sha256=sha(args.sources),tool_pin_sha256=sha(args.tool_pin)),indent=2)+'\n')
    with receipt.open('x') as stream:
        subprocess.run(command,stdout=stream,check=True)
    recipe=json.loads(receipt.read_text())
    assert recipe['crop']==dict(x=0,y=0,width=width,height=height)
    assert recipe['output_png_bit_depth']==8 and recipe['output_png_color_type'] in [0,2]
    subprocess.run([sys.executable,str(args.pixel_oracle),'--receipt',str(receipt),
        '--output',str(args.output/'INDEPENDENT_PIXEL_ORACLE.json')],check=True)
    source.update(raw_source_path=source['path'],raw_source_sha256=source['sha256'],
        path=str(prepared),sha256=sha(prepared),name=source['name']+'-rgb8',
        preparation_receipt=str(receipt),preparation_sha256=sha(receipt))
    manifest.update(parent_manifest_sha256=sha(args.sources),build_commit=pin['source_build_commit'],
        source_preparation_pin_sha256=sha(args.tool_pin),
        admission='GUI opaque crop and Windows95 lossless PNG8 palette expansion; all24 family identities preserved')
    (args.output/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(source=source,
        pixel_oracle_sha256=sha(args.output/'INDEPENDENT_PIXEL_ORACLE.json'),
        no_model_encoder_or_judge_arithmetic_changed=True),indent=2)+'\n')


if __name__=='__main__':
    main()
