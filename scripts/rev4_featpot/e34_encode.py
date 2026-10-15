"""Local, label-free E34 encode orchestration over the admitted fixed selection."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
from e34_augmentation import sha, write

SWEEP=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/bin/zenmetrics-existing')
MOZ=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/encoder-build/mozjpeg/cjpeg-static')


def command(argv,log):
    started=time.monotonic()
    with log.open('x') as f:
        result=subprocess.run(list(map(str,argv)),env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
    record=dict(argv=list(map(str,argv)),returncode=result.returncode,wall_seconds=time.monotonic()-started,log=str(log))
    write(log.with_suffix('.command.json'),record)
    if result.returncode:raise RuntimeError(f'command failed: {log}')
    return record


def prepare(root):
    selection=json.loads((root/'selection.json').read_text())
    normalized={Path(r['destination']).stem:r for r in map(json.loads,(root/'references.jsonl').read_text().splitlines())}
    out=root/'encodes';out.mkdir()
    for name,group in [('augmentation','augmentation'),('development','development')]:
        src=out/(name+'-sources');src.mkdir()
        rows=[]
        for origin in selection[group]['origins']:
            r=normalized[origin]
            assert sha(r['destination'])==r['sha256']
            (src/(origin+'.png')).symlink_to(r['destination'])
            rows.append(dict(origin=origin,reference=r['destination'],reference_sha256=r['sha256'],width=r['width'],height=r['height']))
        write(out/(name+'-sources.json'),rows)
    write(out/'tools.json',dict(zenmetrics=dict(path=str(SWEEP),sha256=sha(SWEEP),source_commit=None,
        provenance='retained existing local executable; source closure not recovered'),
        moz=dict(path=str(MOZ),sha256=sha(MOZ),source_commit='eacbf05d3fefaf2520d61aaef6f78b548a60bb1c'),
        selection_sha256=sha(root/'selection.json')))


def native(codec,sources,out,qgrid='20,50,80',knobs=None):
    out.mkdir()
    args=[SWEEP,'--use-legacy-scheduler','--bench-on-start','no','sweep','--codec','zen'+codec,
        '--sources',sources,'--q-grid',qgrid,'--metric','ssim2','--gpu-runtime','cpu','--jobs','1',
        '--output',out/'scores.tsv','--distorted-out-dir',out/'decoded','--encoded-out-dir',out/'encoded',
        '--pairs-tsv',out/'pairs.tsv']
    if knobs:args += ['--knob-grid',json.dumps(knobs,separators=(',',':'))]
    result=command(args,out/'run.log')
    rows=list(csv.DictReader((out/'pairs.tsv').open(),delimiter='\t'))
    assert len(rows)==len(list(Path(sources).glob('*.png')))*len(qgrid.split(','))*math.prod(len(v) for v in (knobs or {}).values()),(codec,len(rows))
    assert all(Path(r['dist_path']).is_file() for r in rows)
    write(out/'complete.json',dict(codec=codec,rows=len(rows),execution=result))
    return rows


def probe(root):
    out=root/'encodes/probe';out.mkdir()
    src=out/'source';src.mkdir()
    source=json.loads((root/'encodes/augmentation-sources.json').read_text())[0]
    (src/(source['origin']+'.png')).symlink_to(source['reference'])
    moz=out/'moz';moz.mkdir()
    for q in [20,50,80]:
        command([MOZ,'-quality',q,'-optimize','-outfile',moz/f'q{q}.jpg',source['reference']],moz/f'q{q}.log')
    for codec,knobs in [('jxl',{'effort':[8],'butteraugli_iters':[0]}),('webp',{'method':[4]}),('avif',{'speed':[8]})]:
        rows=native(codec,src,out/codec,knobs=knobs)
        print(json.dumps(dict(probe=codec,rows=len(rows))),flush=True)
    write(out/'complete.json',dict(codecs=['moz','jxl','webp','avif'],qualities=[20,50,80],reference=source))


def broad_bridge(root):
    out=root/'encodes/broad-bridge';out.mkdir()
    source=out/'source';source.mkdir()
    packet=json.loads(Path('/home/lilith/work/zensim-validation-2026-09-08/max-attribution/PIXEL_REGISTER.json').read_text())
    known=[r for r in packet['inputs'] if r['origin']=='1054']
    known.sort(key=lambda r:r['distance'])
    assert len(known)==3 and len({r['reference_sha256'] for r in known})==1
    assert sha(known[0]['reference'])==known[0]['reference_sha256']
    (source/'1054.png').symlink_to(known[0]['reference'])
    rows=native('jxl',source,out/'jxl',qgrid='50',knobs={'effort':[8],
        'distance':[r['distance'] for r in known],'butteraugli_iters':[0],'zensim_iters':[0]})
    encoded=list((out/'jxl/encoded').iterdir())
    actual={sha(p) for p in encoded}
    expected={r['encoded_sha256'] for r in known}
    write(out/'comparison.json',dict(encoded_hashes=sorted(actual),expected_hashes=sorted(expected),
        exact_replay=actual==expected,scope='encoder recipe replay only, no model/gate assessment'))
    print(json.dumps(dict(broad_bridge_exact=actual==expected)))


def augmentation(root):
    out=root/'encodes/augmentation';out.mkdir()
    tools=json.loads((root/'encodes/tools.json').read_text())
    assert sha(SWEEP)==tools['zenmetrics']['sha256'] and sha(MOZ)==tools['moz']['sha256']
    sources=json.loads((root/'encodes/augmentation-sources.json').read_text())
    by={r['origin']:r for r in sources}
    assert len(by)==128
    moz=out/'moz';moz.mkdir();(moz/'encoded').mkdir();(moz/'decoded').mkdir()
    normalized=[]
    for source in sources:
        for q in [20,50,80]:
            jpg=moz/'encoded'/f"{source['origin']}_q{q}.jpg"
            command([MOZ,'-quality',q,'-optimize','-outfile',jpg,source['reference']],moz/f"{source['origin']}_q{q}.log")
            normalized.append(dict(source=str(jpg),source_sha256=sha(jpg),destination=str(moz/'decoded'/(jpg.stem+'.png'))))
    write(moz/'normalize.json',dict(mode='normalize',longest_side=0,output=str(moz/'normalized.jsonl'),rows=normalized))
    pixels=Path(__file__).resolve().parents[2]/'target/debug/examples/e34_pixels'
    command([pixels,moz/'normalize.json'],moz/'normalize.log')
    decoded={r['destination']:r for r in map(json.loads,(moz/'normalized.jsonl').read_text().splitlines())}
    rows=[]
    for source in sources:
        for q in [20,50,80]:
            name=f"{source['origin']}_q{q}"
            png=moz/'decoded'/(name+'.png');jpg=moz/'encoded'/(name+'.jpg')
            row=decoded[str(png)]
            assert (row['width'],row['height'])==(source['width'],source['height'])
            rows.append(dict(origin=source['origin'],codec='moz',q=q,reference=source['reference'],reference_sha256=source['reference_sha256'],
                distorted=str(png),distorted_sha256=row['sha256'],encoded=str(jpg),encoded_sha256=sha(jpg),width=row['width'],height=row['height']))
    print(json.dumps(dict(codec='moz',bases=384)),flush=True)
    for codec,knobs in [('jxl',{'effort':[8],'butteraugli_iters':[0]}),('webp',{'method':[4]}),('avif',{'speed':[8]})]:
        encoded=native(codec,root/'encodes/augmentation-sources',out/codec,knobs=knobs)
        for r in encoded:
            origin=Path(r['ref_path']).stem;source=by[origin]
            png=Path(r['dist_path']);bitstream=out/codec/'encoded'/(png.stem+'.'+codec)
            assert bitstream.is_file()
            rows.append(dict(origin=origin,codec=codec,q=int(r['q']),reference=source['reference'],reference_sha256=source['reference_sha256'],
                distorted=str(png),distorted_sha256=sha(png),encoded=str(bitstream),encoded_sha256=sha(bitstream),width=source['width'],height=source['height']))
        print(json.dumps(dict(codec=codec,bases=len(encoded))),flush=True)
    order={c:i for i,c in enumerate(['moz','jxl','webp','avif'])}
    rows.sort(key=lambda r:(r['origin'],order[r['codec']],r['q']))
    for i,r in enumerate(rows):r['base_id']=i
    assert len(rows)==1536 and len({(r['origin'],r['codec'],r['q']) for r in rows})==1536
    assert sha(SWEEP)==tools['zenmetrics']['sha256'] and sha(MOZ)==tools['moz']['sha256']
    write(out/'bases.json',rows)
    print(json.dumps(dict(augmentation='COMPLETE',origins=128,bases=1536)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','probe','broad-bridge','augmentation'])
    p.add_argument('--root',required=True,type=Path)
    a=p.parse_args()
    {'prepare':prepare,'probe':probe,'broad-bridge':broad_bridge,'augmentation':augmentation}[a.mode](a.root)
if __name__=='__main__':main()
