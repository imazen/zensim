#!/usr/bin/env python3
"""Relocate pinned native-cache fixtures without changing pixels or numeric expectations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')


def prepare(args):
    out=args.root/'native-remote'
    out.mkdir(exist_ok=False)
    assets=out/'assets'
    assets.mkdir()
    mapping={}

    def asset(source):
        if source in mapping:
            return mapping[source]
        source=Path(source)
        digest=sha(source)
        rel=Path('assets')/(digest+source.suffix)
        target=out/rel
        if target.exists():
            assert sha(target)==digest
        else:
            shutil.copy2(source,target)
        mapping[str(source)]=str(args.remote_root/rel)
        return mapping[str(source)]

    manifest=dict(remote_root=str(args.remote_root), models={}, relocated_assets=mapping)
    for model in ['a','c']:
        original=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part6')/f'cached-gate-{model}'/'result.json'
        expected=out/model/'expected.json'
        expected.parent.mkdir()
        shutil.copy2(original,expected)
        jobs=[]
        for chunk in sorted((args.root/'native-cache'/model).glob('chunk-*')):
            if not chunk.is_dir():
                continue
            packet=read(chunk/'job.json')
            for case in packet['cases']:
                for key in ['reference','distorted','model']:
                    assert sha(case[key])==case[key+'_sha256']
                    case[key]=asset(case[key])
            cache=read(packet['repair_feature_cache'])
            assert sha(cache['producer_model'])==cache['producer_sha256']
            cache['producer_model']=asset(cache['producer_model'])
            rel=Path(model)/chunk.name
            write(out/rel/'cache.json',cache)
            packet['repair_feature_cache']=str(args.remote_root/rel/'cache.json')
            packet['output']=str(args.remote_root/rel/'result.json')
            write(out/rel/'job.json',packet)
            jobs.append(str(rel/'job.json'))
        assert len(jobs)==15, jobs
        manifest['models'][model]=dict(jobs=jobs,expected=str(Path(model)/'expected.json'),
                                      source_result=str(original),source_result_sha256=sha(original))
    manifest['input_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'MANIFEST.json',manifest)
    print(json.dumps(dict(models=2,assets=len(mapping),cases=270,input_files=len(manifest['input_sha256']))))


def compare(root, model, manifest):
    record=manifest['models'][model]
    expected=read(root/record['expected'])['rows']
    assert sha(root/record['expected'])==record['source_result_sha256']
    rows=[]
    for job_path in record['jobs']:
        job=read(root/job_path)
        result=read(Path(job['output']))
        assert result['packet_sha256']==sha(root/job_path)
        assert len(result['rows'])==len(job['cases'])==9
        for row in result['rows']:
            assert row['repair_feature_cache']==job['repair_feature_cache']
        rows.extend(result['rows'])
    assert len(rows)==len(expected)==135
    for actual, prior in zip(rows,expected):
        assert actual['key']==prior['key']
        actual=dict(actual)
        prior=dict(prior)
        # This path is explicitly relocated; both byte-pinned cache contents
        # and every numeric/bit output remain validated by the native owner.
        del actual['repair_feature_cache']
        del prior['repair_feature_cache']
        assert actual==prior, actual['key']
    proof=dict(status='PASS',cases=len(rows),original_result_sha256=record['source_result_sha256'],
               equality='every row field except the verified relocated repair_feature_cache path',
               native_model_and_pixel_sha_checks=True)
    write(root/model/'PARITY_PASS.json',proof)
    print(model,proof,flush=True)


def run(args):
    root=args.root
    manifest=read(root/'MANIFEST.json')
    assert str(root)==manifest['remote_root']
    assert sha(args.binary)==args.binary_sha256
    for rel,digest in manifest['input_sha256'].items():
        assert sha(root/rel)==digest, rel
    for model in ['a','c']:
        for rel in manifest['models'][model]['jobs']:
            path=root/rel
            job=read(path)
            log=path.parent/'native.log'
            assert not Path(job['output']).exists()
            env={**os.environ,'TMPDIR':str(Path.home()/'tmp'),'ZENSIM_FORMULA_REV':'5',
                 'ZENSIM_NEIGHBOUR_EXACT':'1','RAYON_NUM_THREADS':'1','STEERFIX_PACKET':str(path)}
            with log.open('x') as out:
                subprocess.run(['flock',str(Path.home()/'tmp/r7900x-heavy.lock'),
                    'flock',str(Path.home()/'tmp/zensim-paper/rev4/heavy.lock'),
                    str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','16G','--jobs','1','--',
                    str(args.binary),'metric::bake::steerfix_packet::engineering_packet','--exact','--nocapture'],
                    env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
            assert read(Path(job['output']))['packet_sha256']==sha(path)
            print('native cache chunk complete',rel,flush=True)
        compare(root,model,manifest)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--remote-root',type=Path)
    p.add_argument('--binary',type=Path)
    p.add_argument('--binary-sha256')
    args=p.parse_args()
    if args.mode=='prepare':
        assert args.remote_root and args.remote_root.is_absolute()
        prepare(args)
    else:
        assert args.binary and args.binary_sha256
        run(args)


if __name__=='__main__':
    main()
