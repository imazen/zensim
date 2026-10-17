"""Native E36 derived-data executor for zenfleet; never fits or reads labels.

One view materializes a fixed quality ladder. Zenfleet owns concurrency,
ledger/gap/retries. This executor owns native calls, strict pixel bindings and
durable artifact acknowledgements; its output is a compressed feature receipt.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tarfile
import time
import traceback
import uuid

import numpy as np
from PIL import Image

from e34_encode import E36_CODECS, e36_run, e36_disk
from e36_store import Client, sha, StorageTransportError

Q=[*range(21),*range(22,101,2)]
PREFIX='e36-native-ladder-v1:'
CAPABILITY='e36-native-ladder-v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def write(path,value):
    with Path(path).open('x') as f:f.write(canonical(value)+'\n')


def png_sha(path,width,height):
    with Image.open(path) as im:
        if im.size!=(width,height) or im.mode not in ['RGB','RGBA','P','L','LA','1']:
            raise ValueError('native materialized pixels must be lossless 8-bit PNG at the registered dimensions')
        if im.convert('RGBA').getextrema()[3]!=(255,255):
            raise ValueError('decoded materialization is not opaque; no matte conversion is admitted')
        # PNG may losslessly store a constant decode using indexed/gray pixels.
        # This is independent readback only: the exact expanded RGB hash MUST
        # equal both native ScoreFile and native feature extraction below.
        return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def validate_scores(rows,reference_pixels,distorted_pixels):
    if len(rows)!=2 or {r.get('metric') for r in rows}!={'butteraugli','cvvdp@standard_fhd'}:
        raise ValueError('expected exactly the two registered native teacher rows')
    for r in rows:
        if 'error' in r or r.get('reference_pixels_sha256')!=reference_pixels or r.get('distorted_pixels_sha256')!=distorted_pixels:
            raise ValueError('native scalar/pixel binding failed')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in r['scores'].values()):
            raise ValueError('nonfinite native scalar variant')
    butter=next(r for r in rows if r['metric']=='butteraugli')
    keys=[k for k in butter['scores'] if k.endswith('pnorm3')]
    if len(keys)!=1:raise ValueError('butteraugli pnorm3 is not uniquely identified')
    cv=next(r for r in rows if r['metric']=='cvvdp@standard_fhd')
    values=float(butter['scores'][keys[0]]),float(cv['score'])
    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite active teachers')
    return values


def validate_map(path,width,height):
    raw=gzip.decompress(Path(path).read_bytes())
    magic,dims,scale,payload=raw.split(b'\n',3)
    if magic!=b'Pf' or scale!=b'-1.0' or tuple(map(int,dims.split()))!=(width,height):
        raise ValueError('native map header/dimensions changed')
    pixels=np.frombuffer(payload,dtype='<f4')
    if len(pixels)!=width*height or not np.isfinite(pixels).all():
        raise ValueError('native map has missing/nonfinite pixels')


class Runtime:
    def __init__(self,path):
        self.path=Path(path);self.spec=json.loads(self.path.read_text());self.spec_sha=sha(self.path)
        s=self.spec
        if s['schema']!='e36-native-data-runtime-v1' or s['fits']!=0:raise ValueError('wrong data-only runtime')
        self.root=Path(s['root']);self.tool=Path(s['native_tool']);self.pixels=Path(s['feature_tool'])
        self.feature_spec=json.loads(Path(s['feature_spec']).read_text())
        pin=json.loads(Path(s['native_pin']).read_text())
        if sha(self.tool)!=pin['binary_sha256'] or sha(self.pixels)!=pin['pixel_binary_sha256'] or sha(s['feature_spec'])!=pin['feature_spec_sha256']:
            raise ValueError('native producer pin changed')
        for name,expected in s['program_files'].items():
            if sha(Path(__file__).parent/name)!=expected:raise ValueError('executor program changed')
        self.pin=pin
        references=json.loads(Path(s['references']).read_text())
        if sha(s['references'])!=s['references_sha256']:raise ValueError('normalized reference index changed')
        self.references={r['reference_key']:r for r in references['rows']}
        os.environ['E36_STORE_CONNECT_GATE']=str(self.root/'storage-connect-gate')
        self.store=Client(s['storage']['command'],s['storage']['program_sha256'])
        self.plans={}
        for n in ['jpeg','rav1e','svt','webp','jxl']:
            p=Path(s['mode_inventory'])/(n+'.jsonl')
            if sha(p)!=s['mode_inventory_sha256'][n]:raise ValueError('native mode inventory changed')
            self.plans[n]=[json.loads(line) for line in p.read_text().splitlines()]
        for key,p in s['oracles'].items():
            if sha(p)!=s['oracle_sha256'][key]:raise ValueError('independent decoder changed')
        self.staging=self.root/'staging';self.staging.mkdir(exist_ok=True)
        (self.root/'progress').mkdir(exist_ok=True);(self.root/'errors').mkdir(exist_ok=True)

    def call(self,argv,log,deadline,stdin=None,binary_output=None):
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('registered generation wall budget expired')
        # GNU timeout owns and terminates the complete native command group.
        # The parent GNU time wrapper still records its return code/RSS.
        return e36_run(['/usr/bin/timeout','--signal=TERM','--kill-after=10',f'{remaining:.3f}',*map(str,argv)],
            log,stdin=stdin,binary_output=binary_output)

    def oracle(self,codec,encoded,cell,deadline):
        tools=self.spec['oracles']
        if codec=='jpeg':args=[tools['djpeg'],'-pnm',encoded]
        elif codec=='webp':args=[tools['dwebp'],encoded,'-o','/dev/null']
        elif codec in ['rav1e','svt']:args=[tools['ffmpeg'],'-v','error','-threads','1','-i',encoded,'-f','null','-']
        elif codec=='jxl':args=[tools['djxl'],encoded,'-','--output_format','ppm','--num_threads=1']
        else:raise ValueError('unknown native codec')
        # Decoder stdout is inspection output only and never enters targets.
        return self.call(args,cell/'oracle.log',deadline,binary_output=cell/'oracle-discard.bin')

    def execute_cell(self,ref,codec,native,knobs,q,cell,deadline,cache,mode):
        started=time.monotonic()
        cell.mkdir()
        src=cell/'source';src.mkdir();(src/(ref['origin']+'.png')).symlink_to(ref['destination'])
        encoded_dir=cell/'encoded';decoded_dir=cell/'decoded'
        if mode=='baseline':
            args=[self.tool,'--use-legacy-scheduler','--bench-on-start','no','sweep',
                '--codec',native,'--sources',src,'--q-grid',str(q),'--metric','psnr',
                '--gpu-runtime','cpu','--jobs','1','--output',cell/'encode.tsv',
                '--encoded-out-dir',encoded_dir,'--distorted-out-dir',decoded_dir,'--pairs-tsv',cell/'pairs.tsv']
            if knobs:args.extend(['--knob-grid',canonical(knobs)])
            encode_timing=self.call(args,cell/'encode.log',deadline)
            files=list(encoded_dir.iterdir());assert len(files)==1
            encoded=files[0]
            pairs=list(csv.DictReader((cell/'pairs.tsv').open(),delimiter='\t'));assert len(pairs)==1
            decoded=Path(pairs[0]['dist_path'])
            actual_knobs=json.loads(pairs[0]['knob_tuple_json'])
        else:
            encoded_dir.mkdir();ext={'jpeg':'.jpg','rav1e':'.avif','svt':'.avif','webp':'.webp','jxl':'.jxl'}[codec]
            encoded=encoded_dir/('variant'+ext)
            req=dict(kind=dict(kind='encode',codec=native,q=q,knobs=canonical(knobs),hdr=False),
                cell=dict(image_path=ref['destination'],codec=native,q=q,knob_tuple_json=canonical(knobs)),inputs=[])
            encode_timing=self.call([self.tool,'jobexec'],cell/'encode.log',deadline,
                stdin=canonical(req).encode(),binary_output=encoded)
            # Native lossless PNG materializes the same native RGB8 decode that
            # ScoreFile later hashes. No third-party pixels become a target.
            convert=[self.tool,'--use-legacy-scheduler','--bench-on-start','no','sweep','--codec','zenpng',
                '--sources',encoded_dir,'--q-grid','0','--knob-grid','{"compression":["none"]}',
                '--metric','psnr','--gpu-runtime','cpu','--jobs','1','--output',cell/'convert.tsv','--encoded-out-dir',decoded_dir]
            convert_timing=self.call(convert,cell/'convert.log',deadline)
            pixels=list(decoded_dir.iterdir());assert len(pixels)==1;decoded=pixels[0]
            actual_knobs=knobs
        encoded_sha=sha(encoded)
        group_knobs=copy.deepcopy(actual_knobs)
        if mode!='baseline':group_knobs.pop('fp',None)
        requested=dict(q=q,knobs=actual_knobs,mode_key=canonical(group_knobs),panel=mode)
        width,height=ref['width'],ref['height']
        dist_pixels=png_sha(decoded,width,height)
        if encoded_sha in cache:
            prior=copy.deepcopy(cache[encoded_sha]);prior.update(requested)
            if prior['distorted_pixels_sha256']!=dist_pixels:
                raise ValueError('identical encoded bytes produced different native pixels')
            prior['physical_alias']=True
            prior['physical_alias_of']=dict(q=cache[encoded_sha]['q'],mode_key=cache[encoded_sha]['mode_key'])
            prior['encode_execution']=encode_timing
            if mode!='baseline':prior['convert_execution']=convert_timing
            prior['encode_reused']=False
            prior['artifact_persistence_seconds']=0.0
            prior['cell_total_seconds']=time.monotonic()-started
            return prior
        # File URI input must be named by its content digest (native ingress
        # verifies it). This link is temporary; the canonical bundle follows.
        artifact=cell/(encoded_sha+encoded.suffix);os.link(encoded,artifact)
        if png_sha(ref['destination'],width,height)!=ref['pixels_sha256']:
            raise ValueError('normalized reference independent readback changed')
        score_job=dict(cell=dict(image_path=ref['destination'],codec=native),
            kind=dict(kind='score_file',metrics=['butteraugli','cvvdp@standard_fhd'],hdr=False),
            inputs=['file://'+str(artifact)])
        target=cell/'native-scores.jsonl'
        score_timing=self.call([self.tool,'jobexec'],cell/'score.log',deadline,
            stdin=canonical(score_job).encode(),binary_output=target)
        scores=[json.loads(x) for x in target.read_text().splitlines()]
        butter,cv=validate_scores(scores,ref['pixels_sha256'],dist_pixels)
        fjob=dict(mode='features',output=str(cell/'features.jsonl'),model=self.feature_spec['model'],
            model_sha256=self.feature_spec['model_sha256'],width=1853,feature_ids=self.feature_spec['feature_ids'],
            metric_ids=[m+'@build:'+self.pin['binary_sha256'] for m in ['butteraugli','cvvdp@standard_fhd']],
            rows=[dict(key=cell.name,reference=ref['destination'],reference_sha256=ref['sha256'],
                distorted=str(decoded),distorted_sha256=sha(decoded))])
        write(cell/'features-job.json',fjob)
        feature_timing=self.call([self.pixels,cell/'features-job.json'],cell/'features.log',deadline)
        features=[json.loads(x) for x in (cell/'features.jsonl').read_text().splitlines()]
        if len(features)!=1:raise ValueError('native feature row count changed')
        feature=features[0];values=feature['features']
        if len(values)!=1853 or not np.isfinite(np.asarray(values,dtype=np.float64)).all():
            raise ValueError('native feature layout/finite admission failed')
        if feature['reference_pixels_sha256']!=ref['pixels_sha256'] or feature['distorted_pixels_sha256']!=dist_pixels:
            raise ValueError('native feature/scalar decoded pixel hashes disagree')
        maps=[]
        for metric in ['butteraugli','cvvdp@standard_fhd']:
            req=dict(cell=dict(image_path=ref['destination'],codec=native),kind=dict(kind='diffmap',metric=metric,hdr=False),
                inputs=['file://'+str(artifact)])
            blob=cell/(metric+'.pfm.gz')
            timing=self.call([self.tool,'jobexec'],cell/(metric+'.log'),deadline,stdin=canonical(req).encode(),binary_output=blob)
            validate_map(blob,width,height)
            maps.append(dict(metric=metric,sha256=sha(blob),bytes=blob.stat().st_size,
                member='maps/'+sha(blob)+'.pfm.gz',execution=timing,
                pixel_pair_metric_key=feature['metric_pixel_keys'][metric+'@build:'+self.pin['binary_sha256']]))
        oracle_timing=self.oracle(codec,encoded,cell,deadline)
        # Inspection stdout can be enormous for large PPMs. Preserve its hash
        # and byte count, not an extra copy of oracle-produced image bytes.
        oracle_output=cell/'oracle-discard.bin'
        oracle_receipt=dict(stdout_sha256=sha(oracle_output),stdout_bytes=oracle_output.stat().st_size,
            purpose='independent decode inspection only; never a target')
        feature_gz=cell/'native-features.jsonl.gz'
        feature_gz.write_bytes(gzip.compress((cell/'features.jsonl').read_bytes(),compresslevel=1,mtime=0))
        info=dict(schema='e36-native-cell-v1',fits=0,labels_read=False,origin=ref['origin'],family=ref['family'],
            build_commit=self.spec['build_commit'],generator_runtime_sha256=self.spec_sha,
            generation_role=self.current_role,generation_view_sha256=self.current_view,
            content=ref['content'],split=ref['split'],size=ref['size'],width=width,height=height,
            reference_sha256=ref['sha256'],reference_pixels_sha256=ref['pixels_sha256'],
            codec=codec,native_codec=native,**requested,encoded_sha256=encoded_sha,
            encoded_member='encoded/'+encoded_sha+encoded.suffix,encoded_bytes=encoded.stat().st_size,
            distorted_sha256=sha(decoded),distorted_pixels_sha256=dist_pixels,
            distorted_member='pixels/'+sha(decoded)+'.png',feature_sha256=sha(cell/'features.jsonl'),
            feature_member='features/'+sha(feature_gz)+'.jsonl.gz',feature_layout_sha256=self.spec['feature_layout_sha256'],
            butteraugli_pnorm3=butter,cvvdp_standard_fhd=cv,scalar_precision='native JSON f64',
            native_scores=scores,maps=maps,encode_execution=encode_timing,score_execution=score_timing,
            feature_execution=feature_timing,oracle_execution=oracle_timing,oracle=oracle_receipt,tool_sha256=self.pin['binary_sha256'],
            feature_tool_sha256=self.pin['pixel_binary_sha256'],physical_alias=False,encode_reused=False)
        if mode!='baseline':info['convert_execution']=convert_timing
        write(cell/'metadata.json',info)
        diagnostics=cell/'diagnostics.tar.gz'
        with tarfile.open(diagnostics,'w:gz',compresslevel=1) as archive:
            for path in sorted(cell.iterdir()):
                if path.is_file() and path.name not in [diagnostics.name,'oracle-discard.bin',artifact.name,feature_gz.name] and path.suffix not in ['.pfm','.gz']:
                    archive.add(path,arcname=path.name)
        bundle=cell/'bundle.tar'
        members={info['encoded_member']:encoded,info['distorted_member']:decoded,
            info['feature_member']:feature_gz,'metadata.json':cell/'metadata.json','diagnostics.tar.gz':diagnostics}
        members.update({m['member']:cell/(m['metric']+'.pfm.gz') for m in maps})
        with tarfile.open(bundle,'w') as archive:
            for name,path in members.items():archive.add(path,arcname=name)
        storage_started=time.monotonic()
        ack=self.store.put(bundle)
        if not ack['ok']:raise ValueError('no durable artifact acknowledgement')
        if self.current_role=='first-producer':
            checked=self.store.verify(ack['sha256'])
            if checked['bytes']!=ack['bytes']:raise ValueError('first producer storage readback changed')
        info['bundle']=dict(sha256=ack['sha256'],bytes=ack['bytes'],relative_path=ack['relative_path'],
            root=self.spec['storage']['root'],host=self.spec['storage']['host'],storage_program_sha256=ack['program_sha256'])
        info['feature_ids']=self.feature_spec['feature_ids']
        info['features']=[values[i] for i in self.feature_spec['feature_ids']]
        info['artifact_persistence_seconds']=time.monotonic()-storage_started
        info['cell_total_seconds']=time.monotonic()-started
        cache[encoded_sha]=copy.deepcopy(info)
        return info

    def execute(self,job):
        ladder_started=time.monotonic()
        if job['kind']['kind']!='bake' or not job['kind']['view'].startswith(PREFIX):
            raise ValueError('executor accepts only the versioned data view, never fit jobs')
        view=json.loads(job['kind']['view'][len(PREFIX):])
        if view['runtime_sha256']!=self.spec_sha or self.spec_sha not in job['inputs']:
            raise ValueError('data view does not bind this runtime')
        if view['role']=='bulk' and view['q_grid']!=Q:raise ValueError('bulk quality thinning is forbidden')
        if not view['q_grid'] or not set(view['q_grid'])<=set(Q):raise ValueError('unregistered quality')
        ref=self.references[view['reference_key']]
        self.current_role=view['role']
        if ref['sha256'] not in job['inputs'] or sha(ref['destination'])!=ref['sha256']:
            raise ValueError('source input is not content-bound')
        codecs={n:(c,k) for n,c,k in E36_CODECS}
        native,base_knobs=codecs[view['codec']]
        if job['cell']['codec']!=native or job['cell']['image_path']!=ref['destination'] or job['cell']['q']!=-1:
            raise ValueError('cell metadata disagrees with its view')
        key=hashlib.sha256(job['kind']['view'].encode()).hexdigest()
        self.current_view=key
        stage=self.staging/(key+'-'+uuid.uuid4().hex);stage.mkdir()
        progress=self.root/'progress'/(key+'.jsonl')
        saved=self.root/'progress'/key;saved.mkdir(exist_ok=True)
        deadline=min(time.time()+view['deadline_seconds'],float(os.environ.get('E36_GENERATION_DEADLINE','inf')))
        if view['panel']=='baseline':
            cells=[dict(q=q,knobs=base_knobs,mode='baseline') for q in view['q_grid']]
        elif view['panel']=='mode-panel':
            cells=[dict(q=int(r['requested_q']),physical_q=int(r['q']),
                native_alias=r['native_alias'],mode_key=r['mode_key'],knobs=json.loads(r['knob_tuple_json']),mode='mode-panel')
                for r in self.plans[view['codec']] if int(r['requested_q']) in view['q_grid']]
        else:raise ValueError('unregistered panel')
        cache={};receipts=[];native_aliases={}
        for i,cell in enumerate(cells):
            checkpoint=saved/(str(i)+'.json.gz')
            if checkpoint.exists():
                result=json.loads(gzip.decompress(checkpoint.read_bytes()))
                if result['runtime_sha256']!=self.spec_sha or result['view_sha256']!=key or result['q']!=cell['q']:
                    raise ValueError('resume receipt is not bound to this exact data view')
                self.store.verify(result['bundle']['sha256'])
                cache[result['encoded_sha256']]=copy.deepcopy(result);receipts.append(result)
                continue
            physical_key=canonical([cell.get('physical_q',cell['q']),cell['knobs']])
            if physical_key in native_aliases:
                result=copy.deepcopy(native_aliases[physical_key]);result['q']=cell['q']
                result['native_alias']=cell.get('native_alias');result['physical_alias']=True
                result['encode_reused']=True
                result['artifact_persistence_seconds']=0.0
                result['cell_total_seconds']=0.0
            else:
                result=self.execute_cell(ref,view['codec'],native,cell['knobs'],cell.get('physical_q',cell['q']),stage/str(i),deadline,cache,cell['mode'])
                native_aliases[physical_key]=copy.deepcopy(result)
                result['q']=cell['q'];result['native_alias']=cell.get('native_alias')
            if 'mode_key' in cell:result['mode_key']=cell['mode_key']
            result.update(role=view['role'],view_sha256=key,runtime_sha256=self.spec_sha,physical_index=i)
            receipts.append(result)
            with checkpoint.open('xb') as f:f.write(gzip.compress(canonical(result).encode(),compresslevel=1,mtime=0))
            with progress.open('a') as f:f.write(canonical(dict(completed=i+1,total=len(cells),q=cell['q'],
                codec=view['codec'],origin=ref['origin'],size=ref['size'],bundle=result['bundle']['sha256'],utc=time.time()))+'\n')
            # Each cell's real bytes/maps are acknowledged in the canonical
            # store before its bounded staging copy is removed. Aliases point
            # to an already acknowledged identical physical variant.
            if (stage/str(i)).exists():shutil.rmtree(stage/str(i))
        output=dict(schema='e36-native-ladder-receipt-v1',fits=0,labels_read=False,view=view,
            rows=receipts,runtime_sha256=self.spec_sha,stage=str(stage),complete=True,
            ladder_wall_seconds=time.monotonic()-ladder_started,completed_unix=time.time(),
            alias_contract='native emitted physical cells; requested mode/q aliases are joined from the frozen native inventory')
        payload=gzip.compress(canonical(output).encode(),compresslevel=1,mtime=0)
        stage.rmdir()
        return payload


def read_exact(stream,n):
    result=bytearray()
    while len(result)<n:
        part=stream.read(n-len(result))
        if not part:raise EOFError('truncated native worker frame')
        result.extend(part)
    return bytes(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--serve',action='store_true');p.add_argument('command',nargs='?');a=p.parse_args()
    if a.command=='capabilities':print(CAPABILITY);return
    # All native kernels are single-threaded within the worker's admitted
    # concurrent slots; these variables do not change registered codec knobs.
    for key in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    os.environ['ZENSIM_FORMULA_REV']='5';os.environ['ZENSIM_MAX_TIER']='v3'
    os.environ['ZEN_JOBEXEC_PIXEL_HASH']='1';os.environ['ZENMETRICS_USE_LEGACY_SCHEDULER']='1'
    runtime=Runtime(a.runtime)
    source=sys.stdin.buffer;target=sys.stdout.buffer
    try:
        while True:
            if a.serve:
                header=source.read(4)
                if not header:return
                if len(header)!=4:raise EOFError('truncated frame header')
                size=struct.unpack('<I',header)[0]
                if not 0<size<=1024**2:raise ValueError('worker request frame is not bounded')
                raw=read_exact(source,size)
            else:raw=source.read()
            try:
                payload=runtime.execute(json.loads(raw));status=0
            except Exception as exc:
                trace=traceback.format_exc();sys.stderr.write(trace)
                path=runtime.root/'errors'/(str(time.time_ns())+'.json')
                write(path,dict(error=str(exc),traceback=trace,request_sha256=hashlib.sha256(raw).hexdigest(),fits=0))
                payload=canonical(dict(**{'class':'upload_fail' if isinstance(exc,StorageTransportError) else 'unknown'},detail=str(exc))).encode();status=3
            if a.serve:
                target.write(bytes([status])+struct.pack('<I',len(payload))+payload);target.flush()
            else:
                if status:raise RuntimeError(payload.decode())
                target.write(payload);target.flush();return
    finally:runtime.store.close()


if __name__=='__main__':main()
