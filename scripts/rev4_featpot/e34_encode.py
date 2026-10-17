"""Local E34 encode preparation and native-only, label-free E36 generation."""
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



# E36 uses native encoders exclusively. These local preparation functions invoke
# the existing zenmetrics sweep/plan/jobexec owners; they do not claim fleet work.
E36_CODECS = [
    ('jpeg', 'zenjpeg', {}),
    ('rav1e', 'zenavif', {'backend': ['zenravif']}),
    ('svt', 'zenavif', {'backend': ['svt-rs']}),
    ('webp', 'zenwebp', {}),
    ('jxl', 'zenjxl', {}),
]


def e36_disk(root):
    import shutil
    for path in [Path.home(), root]:
        if shutil.disk_usage(path).free < 20 * 1024 ** 3:
            raise RuntimeError(f'E36 disk floor: {path}')


def e36_run(argv, out, stdin=None, binary_output=None):
    """Measured invocation; binary stdout never shares its diagnostic log."""
    e36_disk(out.parent)
    rss = out.with_suffix('.rss.txt')
    started = time.monotonic()
    with out.open('xb') as log:
        if binary_output:
            with binary_output.open('xb') as payload:
                result = subprocess.run(['/usr/bin/time', '-v', '-o', str(rss), *map(str, argv)],
                    input=stdin, stdout=payload, stderr=log)
        else:
            result = subprocess.run(['/usr/bin/time', '-v', '-o', str(rss), *map(str, argv)],
                input=stdin, stdout=log, stderr=subprocess.STDOUT)
    values = {}
    for line in rss.read_text().splitlines():
        if 'Maximum resident set size (kbytes):' in line:
            values['peak_rss_kib'] = int(line.rsplit(':', 1)[1])
    receipt = dict(argv=list(map(str, argv)), returncode=result.returncode,
        wall_seconds=time.monotonic() - started, rss_file=str(rss), **values)
    write(out.with_suffix('.command.json'), receipt)
    if result.returncode:
        raise RuntimeError(f'E36 native invocation failed: {out}')
    return receipt


def e36_smoke(a):
    import gzip
    import shutil
    root = a.root
    registered_q = [*range(21), *range(22,101,2)]
    assert a.q == sorted(set(a.q)) and set(a.q) <= set(registered_q)
    tool = a.tool.resolve(strict=True)
    pin = json.loads(a.tool_pin.read_text())
    assert sha(tool) == pin['binary_sha256']
    assert pin['source_closed'] is True and pin['native_only'] is True
    assert sha(a.pixels) == pin['pixel_binary_sha256']
    assert sha(a.feature_spec) == pin['feature_spec_sha256']
    refs = json.loads(a.references.read_text())
    assert len(refs) > 0
    out = root / a.smoke_name
    out.mkdir(exist_ok=False)
    cas = root / 'artifacts'
    cas.mkdir(exist_ok=True)
    summary = []
    for ref in refs:
        assert ref['color_admitted'] is True
        assert sha(ref['destination']) == ref['sha256']
        src = out / ('source-' + ref['origin'] + '-' + str(ref['size']))
        src.mkdir()
        (src / (ref['origin'] + '.png')).symlink_to(ref['destination'])
        for name, codec, knobs in E36_CODECS:
            for q in a.q:
                cell = out / (ref['origin'] + '-' + str(ref['size']) + '-' + name + '-q' + str(q))
                cell.mkdir()
                argv = [tool, '--use-legacy-scheduler', '--bench-on-start', 'no', 'sweep',
                    '--codec', codec, '--sources', src, '--q-grid', str(q),
                    '--metric', 'butteraugli', '--metric', 'cvvdp', '--display-model', 'standard_fhd',
                    '--gpu-runtime', 'cpu', '--jobs', '1', '--output', cell/'scores.tsv',
                    '--encoded-out-dir', cell/'encoded', '--distorted-out-dir', cell/'decoded',
                    '--pairs-tsv', cell/'pairs.tsv']
                if knobs:
                    argv.extend(['--knob-grid', json.dumps(knobs, separators=(',', ':'))])
                timing = e36_run(argv, cell/'encode-score.log')
                pairs = list(csv.DictReader((cell/'pairs.tsv').open(), delimiter='\t'))
                assert len(pairs) == 1
                encoded = list((cell/'encoded').iterdir())
                assert len(encoded) == 1
                bitstream = encoded[0]
                digest = sha(bitstream)
                artifact = cas / (digest + bitstream.suffix)
                if artifact.exists():
                    assert sha(artifact) == digest
                else:
                    shutil.copy2(bitstream, artifact)
                pixels = Path(pairs[0]['dist_path'])
                assert pixels.is_file()
                spec = json.loads(a.feature_spec.read_text())
                fjob = dict(mode='features', output=str(cell/'features.jsonl'),
                    model=spec['model'], model_sha256=spec['model_sha256'],
                    width=1853, feature_ids=spec['feature_ids'],
                    metric_ids=[m+'@build:'+pin['binary_sha256'] for m in ['butteraugli','cvvdp@standard_fhd']], rows=[dict(
                        key=cell.name, reference=str(src/(ref['origin']+'.png')),
                        reference_sha256=ref['sha256'], distorted=str(pixels),
                        distorted_sha256=sha(pixels))])
                write(cell/'features-job.json', fjob)
                ft = e36_run([a.pixels, cell/'features-job.json'], cell/'features.log')
                feature_rows = list(map(json.loads, (cell/'features.jsonl').read_text().splitlines()))
                assert len(feature_rows) == 1 and len(feature_rows[0]['features']) == 1853
                frow = feature_rows[0]
                assert frow['reference_pixels_sha256'] == ref['pixels_sha256']
                maps = []
                for metric in ['butteraugli', 'cvvdp@standard_fhd']:
                    # Decode the exact already-retained native variant in jobexec.
                    job = dict(cell=dict(image_path=str(src/(ref['origin']+'.png')), codec=codec),
                        kind=dict(kind='diffmap', metric=metric, hdr=False), inputs=['file://' + str(artifact)])
                    blob = cell / (metric + '.pfm.gz')
                    mt = e36_run([tool, 'jobexec'], cell/(metric+'.log'),
                        stdin=json.dumps(job).encode(), binary_output=blob)
                    raw = gzip.decompress(blob.read_bytes())
                    assert raw.startswith(b'Pf\n')
                    lines = raw.split(b'\n', 3)
                    width, height = map(int, lines[1].split())
                    assert (width, height) == (ref['width'], ref['height'])
                    assert len(lines[3]) == width * height * 4
                    mh = sha(blob)
                    saved = cas / (mh + '.pfm.gz')
                    if saved.exists(): assert sha(saved) == mh
                    else: shutil.copy2(blob, saved)
                    pixel_key = frow['metric_pixel_keys'][metric+'@build:'+pin['binary_sha256']]
                    maps.append(dict(metric=metric, path=str(saved), sha256=mh,
                        pixel_pair_metric_key=pixel_key,
                        bytes=saved.stat().st_size, execution=mt))
                score = list(csv.DictReader((cell/'scores.tsv').open(), delimiter='\t'))
                assert len(score) == 1
                cvkeys = [k for k in score[0] if k.startswith('score_cvvdp') and 'standard_fhd' in k]
                assert len(cvkeys) == 1
                butter = float(score[0]['score_butteraugli_pnorm3'])
                cv = float(score[0][cvkeys[0]])
                assert math.isfinite(butter) and math.isfinite(cv)
                row = dict(origin=ref['origin'], family=ref['family'], split=ref['split'],
                    content=ref['content'], size=ref['size'], width=ref['width'], height=ref['height'],
                    codec=name, native_codec=codec, q=q, knobs=knobs,
                    reference_sha256=ref['sha256'], distorted_sha256=sha(pixels),
                    encoded_sha256=digest, encoded_path=str(artifact), encoded_bytes=artifact.stat().st_size,
                    distorted_path=str(pixels), maps=maps, scalars=score[0], execution=timing,
                    tool_sha256=pin['binary_sha256'], feature_execution=ft,
                    feature_path=str(cell/'features.jsonl'), feature_sha256=sha(cell/'features.jsonl'),
                    feature_layout=dict(width=1853, consumed_ids=spec['feature_ids'],
                        structural_zero_append=list(range(1825,1853))),
                    reference_pixels_sha256=frow['reference_pixels_sha256'],
                    distorted_pixels_sha256=frow['distorted_pixels_sha256'],
                    butteraugli_pnorm3=butter, cvvdp_standard_fhd=cv)
                write(cell/'artifacts-verified.json', row)
                summary.append(row)
                print(json.dumps(dict(completed=len(summary), origin=ref['origin'], size=ref['size'],
                    codec=name, q=q, encoded_bytes=row['encoded_bytes'], wall_seconds=timing['wall_seconds'])), flush=True)
                if len(summary) == 1:
                    write(out/'FIRST_CELL_VERIFIED.json', row)
    write(out/'complete.json', dict(schema='e36-native-smoke-v1', labels_read=False, fits=0,
        tool_pin_sha256=sha(a.tool_pin), references_sha256=sha(a.references), rows=summary))


def e36_plan(a):
    """Native finite mode-plan export; zero encodes and zero budget truncation."""
    tool = a.tool.resolve(strict=True)
    pin = json.loads(a.tool_pin.read_text())
    assert pin['source_closed'] and sha(tool) == pin['binary_sha256']
    out = a.root/('native-plans-main-effects' if a.main_effects else 'native-plans')
    out.mkdir(exist_ok=False)
    qgrid = ','.join(map(str, [*range(21), *range(22,101,2)]))
    for name, codec, knobs in E36_CODECS:
        plan = 'svt_doe_main' if name == 'svt' else 'modes_full'
        argv = [tool, 'sweep', '--codec', codec, '--sources', a.sources,
            '--q-grid', qgrid, '--plan', plan, '--dry-run',
            '--emit-cells', out/(name+'.jsonl'), '--output', out/(name+'.tsv')]
        if a.main_effects:
            argv.extend(['--max-deviations', '1'])
        e36_run(argv, out/(name+'.log'))


def e36_audit(a):
    """Read back stored native artifacts; independent decoders never set targets."""
    import gzip
    import numpy as np
    import shutil
    out = a.root/a.audit_name
    out.mkdir(exist_ok=False)
    runs = [json.loads(p.read_text()) for p in sorted(a.root.glob('smoke-*/complete.json'))]
    rows = [r for run in runs for r in run['rows']]
    assert rows
    tools = {name:shutil.which(name) for name in ['djpeg','dwebp','ffmpeg']}
    assert all(tools.values())
    assert a.djxl.is_file()
    refs = {r['sha256']:r for p in a.root.glob('smoke-references-*.json')
            for r in json.loads(p.read_text())}
    from PIL import Image
    import hashlib
    pngs = {}
    def png_readback(path, file_sha, pixels_sha, width, height):
        assert sha(path)==file_sha
        if file_sha not in pngs:
            with Image.open(path) as im:
                assert im.size==(width,height)
                digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            assert digest==pixels_sha, f'independent PNG pixels differ: {path}'
            pngs[file_sha]=digest
        else:
            assert pngs[file_sha]==pixels_sha
    for path in a.normalized_references:
        for r in map(json.loads,path.read_text().splitlines()):
            png_readback(r['destination'],r['sha256'],r['pixels_sha256'],r['width'],r['height'])
    checked, oracles = {}, []
    for i, row in enumerate(rows):
        assert sha(row['encoded_path']) == row['encoded_sha256']
        assert sha(row['distorted_path']) == row['distorted_sha256']
        assert sha(row['feature_path']) == row['feature_sha256']
        ref=refs[row['reference_sha256']]
        assert row['reference_pixels_sha256'] == ref['pixels_sha256']
        png_readback(ref['destination'],ref['sha256'],ref['pixels_sha256'],ref['width'],ref['height'])
        png_readback(row['distorted_path'],row['distorted_sha256'],row['distorted_pixels_sha256'],row['width'],row['height'])
        feature = json.loads(Path(row['feature_path']).read_text())
        assert np.isfinite(np.asarray(feature['features'], dtype=np.float32)).all()
        for m in row['maps']:
            assert sha(m['path']) == m['sha256']
            raw = gzip.decompress(Path(m['path']).read_bytes())
            magic, dims, scale, payload = raw.split(b'\n',3)
            assert magic == b'Pf' and scale == b'-1.0'
            assert tuple(map(int,dims.split())) == (row['width'],row['height'])
            floats = np.frombuffer(payload,dtype='<f4')
            assert len(floats)==row['width']*row['height'] and np.isfinite(floats).all()
        key = (row['native_codec'],row['encoded_sha256'])
        if key in checked:
            assert checked[key]['distorted_pixels_sha256']==row['distorted_pixels_sha256']
            continue
        checked[key] = row
        encoded = row['encoded_path']
        if row['codec']=='jpeg':
            argv = [tools['djpeg'],'-pnm',encoded]
        elif row['codec']=='webp':
            argv = [tools['dwebp'],encoded,'-o','/dev/null']
        elif row['codec'] in {'rav1e','svt'}:
            argv = [tools['ffmpeg'],'-v','info','-threads','1','-i',encoded,'-f','null','-']
        elif row['codec']=='jxl':
            argv = [a.djxl,encoded,'-','--output_format','ppm','--num_threads=1']
        else:
            raise ValueError('unregistered native codec')
        log = out/(str(i)+'.log')
        started = time.monotonic()
        with log.open('xb') as stream:
            result = subprocess.run(list(map(str,argv)),stdout=subprocess.DEVNULL,stderr=stream)
        record = dict(codec=row['codec'],encoded_sha256=row['encoded_sha256'],argv=list(map(str,argv)),
            returncode=result.returncode,wall_seconds=time.monotonic()-started,log=str(log))
        oracles.append(record)
        write(out/(str(i)+'.json'),record)
        assert result.returncode==0, f'independent readback failed: {log}'
    write(out/'complete.json',dict(labels_read=False,fits=0,rows=len(rows),
        unique_encoded_variants=len(checked),independently_read_pngs=len(pngs),oracles=oracles,
        oracle_binary_sha256={k:sha(v) for k,v in dict(tools,djxl=a.djxl).items()},
        scope='native artifact hashes/layout/finite maps and independent decode success; no oracle pixels or scores used as training targets'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','probe','broad-bridge','augmentation','e36-smoke','e36-plan','e36-audit'])
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--tool', type=Path)
    p.add_argument('--tool-pin', type=Path)
    p.add_argument('--references', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--main-effects', action='store_true')
    p.add_argument('--pixels', type=Path)
    p.add_argument('--feature-spec', type=Path)
    p.add_argument('--djxl', type=Path)
    p.add_argument('--audit-name',default='artifact-audit')
    p.add_argument('--normalized-references',action='append',type=Path,default=[])
    p.add_argument('--smoke-name', default='smoke')
    p.add_argument('--q', type=int, nargs='+', default=[0,1,20,60,100])
    a=p.parse_args()
    if a.mode in {'e36-smoke', 'e36-plan','e36-audit'}:
        {'e36-smoke':e36_smoke,'e36-plan':e36_plan,'e36-audit':e36_audit}[a.mode](a)
        return
    {'prepare':prepare,'probe':probe,'broad-bridge':broad_bridge,'augmentation':augmentation}[a.mode](a.root)
if __name__=='__main__':main()
