"""Resolve immutable E36 bundle receipts through verified archival relocations."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from e36_store import digest, sha

from e36_archive_runs import native_runs


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_mkdir(path):
    path=Path(path)
    if path.is_dir():return
    durable_mkdir(path.parent)
    try:path.mkdir()
    except FileExistsError:
        if not path.is_dir():raise
    sync_directory(path)
    sync_directory(path.parent)


@contextmanager
def access(root,hash_value,exclusive=False):
    digest(hash_value)
    directory=Path(root)/'artifact-access-locks'
    durable_mkdir(directory)
    with (directory/(hash_value[:2]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def relocation_segments(root):
    return sorted((Path(root)/'owner-svt-ssd-v1/artifact-relocation-segments-v1').glob('*.jsonl'))


def relocation_records(root):
    journal=Path(root)/'owner-svt-ssd-v1/artifact-relocations.jsonl'
    paths=([journal] if journal.exists() else [])+relocation_segments(root)
    for path in paths:
        if path!=journal and sha(path)!=path.stem:raise ValueError('committed relocation segment changed')
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):raise ValueError('committed relocation history has an incomplete line')
                yield json.loads(line)


def resolve(bundle,root,remote_verify,storage,journal_pin=None):
    """Verify actual bytes at the recorded owner or a hash-bound relocation."""
    root=Path(root);hash_value=digest(bundle['sha256'])
    expected_relative='blobs/'+hash_value[:2]+'/'+hash_value+'.tar'
    if bundle['relative_path']!=expected_relative:
        raise ValueError('artifact path is not its canonical content address')
    if bundle['host']==storage['host'] and bundle['root']==storage['root']:
        answer=remote_verify(hash_value)
        if answer['bytes']!=bundle['bytes']:raise ValueError('legacy bundle size changed')
        return dict(host=storage['host'],root=storage['root'],**answer)
    local=root/'artifacts-local-v1'
    if bundle['host']!='dev' or bundle['root']!=str(local):
        raise ValueError('unknown artifact owner')
    with access(root,hash_value):
        path=local/expected_relative
        try:
            size=path.stat().st_size;actual_sha=sha(path)
        except FileNotFoundError:
            # Pack eviction journals the whole verified batch before unlink.
            # If unlink races this read, resolve through that durable journal.
            pass
        else:
            if size!=bundle['bytes'] or actual_sha!=hash_value:raise ValueError('local artifact bytes changed')
            return dict(ok=True,host='dev',root=str(local),sha256=hash_value,
                        bytes=bundle['bytes'],relative_path=expected_relative)
        journal=root/'owner-svt-ssd-v1/artifact-relocations.jsonl'
        if journal_pin is not None:
            if Path(journal_pin['path'])!=journal or sha(journal)!=journal_pin['sha256']:
                raise ValueError('frozen artifact relocation journal changed')
        found=None
        if journal_pin is not None and journal_pin.get('segments')!=[dict(path=str(p),sha256=sha(p)) for p in relocation_segments(root)]:raise ValueError('frozen relocation segments changed')
        for record in relocation_records(root):
            if record['sha256']!=hash_value:continue
            owner=record_owner(root,record,storage)
            if (record['sha_verified'] is not True or record['bytes']!=bundle['bytes']):raise ValueError('artifact relocation owner/hash/size differs')
            found=record
        if found is None:raise ValueError('missing local artifact has no verified relocation')
        owner=record_owner(root,found,storage)
        if found.get('encoding')=='tar-pack-v1':
            from e36_pack_archive import readback
            actual=readback(owner,found)
        elif owner.get('encoding')=='zstd':
            from e36_archive_store import ArchiveClient
            client=ArchiveClient(owner['command'],owner['program_sha256'])
            try:actual=client.verify(hash_value)
            finally:client.close()
            if any(actual[k]!=found[k] for k in ['stored_sha256','stored_bytes','encoding']):
                raise ValueError('archive stored-byte identity changed')
        else:actual=remote_verify(hash_value)
        if actual['bytes']!=bundle['bytes']:raise ValueError('relocated artifact byte count changed')
        return dict(host=owner['host'],root=owner['root'],**actual)


def record_owner(root,record,legacy):
    """Each immutable journal row chooses one strict, independently pinned owner."""
    if record.get('encoding')=='tar-pack-v1':
        path=Path(root)/('storage-cache-v1.json' if record['canonical_root']=='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache' else 'storage-archive-v2.json')
        owner=json.loads(path.read_text());digest(record['pack_id']);digest(record['pack_verifier_sha256']);digest(record['stored_sha256'])
        if (owner['host']!=record['canonical_host'] or owner['root']!=record['canonical_root'] or owner['program_sha256']!=record['program_sha256'] or record['relative_path']!='packs/'+record['pack_id']+'.tar' or record['stored_bytes']<=0):raise ValueError('unknown lossless pack owner')
        if path.name=='storage-cache-v1.json' and (owner.get('authorization')!='coordinator-2026-10-17-cache250GB' or owner['cache_budget_bytes']!=250000000000 or owner['cache_reserve_bytes']!=180*1024**3):raise ValueError('packed cache grant differs')
        return owner
    expected='blobs/'+record['sha256'][:2]+'/'+record['sha256']+'.tar'
    if record['canonical_host']==legacy['host'] and record['canonical_root']==legacy['root']:
        owner=legacy
        if record.get('encoding') not in [None,'raw']:raise ValueError('legacy encoding changed')
    else:
        path=Path(root)/('storage-cache-v1.json' if record['canonical_root']=='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache' else 'storage-archive-v2.json');owner=json.loads(path.read_text())
        if path.name=='storage-cache-v1.json' and (owner.get('authorization')!='coordinator-2026-10-17-cache250GB' or owner.get('cache_budget_bytes')!=250000000000 or owner.get('cache_reserve_bytes')!=180*1024**3):raise ValueError('cache authorization differs')
        if (owner.get('encoding')!='zstd' or owner['host']!='tower'
            or owner['root'] not in ['/mnt/disk1/coefficient/output/zensim/e36-bulk-2026-10-17-disk1','/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache']
            or record['canonical_host']!=owner['host'] or record['canonical_root']!=owner['root']
            or record.get('encoding')!='zstd' or type(record.get('stored_bytes')) is not int
            or record['stored_bytes']<=0):raise ValueError('unknown compressed archive owner')
        digest(record['stored_sha256']);expected+='.zst'
    if record['relative_path']!=expected or record['program_sha256']!=owner['program_sha256']:
        raise ValueError('archive owner/path/program differs')
    return owner


def require_pack_directory_proof(root):
    owner=Path(root)/'owner-svt-ssd-v1';path=owner/'pack-directory-durability-v1.json'
    proof=json.loads(path.read_text())
    if proof.get('admitted') is not True or len(proof.get('stores',[]))!=2:raise ValueError('durable pack-directory admission missing')
    for name in ['storage-cache-v1.json','storage-archive-v2.json']:
        storage=json.loads((Path(root)/name).read_text());matches=[r for r in proof['stores'] if r['storage']==name]
        if len(matches)!=1:raise ValueError('durable pack-directory owner missing')
        row=matches[0]
        if row['root']!=storage['root'] or row['container_id']!=storage['container_id'] or row['actual'].get('fsynced') is not True or set(row['actual']['directories'])!={'/data/packs','/data/.pack-incoming','/data'}:raise ValueError('durable pack-directory owner differs')
    return path


def retain_resolver(root,source):
    """Preserve earlier owners and retain this exact code by content hash."""
    import shutil
    source=Path(source);value=sha(source)
    path=Path(root)/'owner-svt-ssd-v1'/('artifact-location-final-'+value+'.py')
    if path.exists():
        if sha(path)!=value:raise ValueError('content-addressed artifact resolver changed')
    else:
        with source.open('rb') as incoming,path.open('xb') as output:
            shutil.copyfileobj(incoming,output);output.flush();os.fsync(output.fileno())
        sync_directory(path.parent)
    return path


def freeze(root,out):
    """Bind final relocation history and replay native eligibility plus readback."""
    import gzip
    import random
    import time
    import pyarrow.parquet as pq
    from e36_store import Client
    root=Path(root);out=Path(out)
    stop=root/'owner-svt-ssd-v1/mirror-stop.json'
    if not stop.exists():raise ValueError('stop archival writer before freezing provenance')
    owner_dir=root/'owner-svt-ssd-v1'
    pid_pins=list(owner_dir.glob('mirror*pid.json'))+list(owner_dir.glob('resource-budget4-v*/mirror-pid.json'))
    for pin in pid_pins:
        identity=json.loads(pin.read_text());pid=identity.get('pid',identity.get('new_pid'));p=Path('/proc')/str(pid)
        if p.exists() and b'e36_artifact_mirror' in (p/'cmdline').read_bytes():
            raise ValueError('artifact mirror still active')
    journal=root/'owner-svt-ssd-v1/artifact-relocations.jsonl'
    if not journal.exists():
        with journal.open('x') as stream:stream.flush();os.fsync(stream.fileno())
        sync_directory(journal.parent)
    journal_sha=sha(journal)
    rows=list(relocation_records(root));segments=[dict(path=str(p),sha256=sha(p)) for p in relocation_segments(root)]
    published=set();ledger_pins=[]
    for name in native_runs(root):
        for p in (root/'runs'/name).glob('ledger*.parquet'):
            for batch in pq.ParquetFile(p).iter_batches(columns=['status','attempts','output_sha']):
                published.update(r['output_sha'] for r in batch.to_pylist() if r['status']=='done' and r['attempts']==1 and r['output_sha'])
            ledger_pins.append(dict(path=str(p),sha256=sha(p)))
    storage_path=root/'storage-runtime.json';storage=json.loads(storage_path.read_text())
    unique={};by_view={}
    for row in rows:
        h=digest(row['sha256']);relative='blobs/'+h[:2]+'/'+h+'.tar'
        record_owner(root,row,storage)
        if (row['sha_verified'] is not True or row['completed_view_sha256'] not in published):
            raise ValueError('archived artifact lacks exact store/native DONE eligibility')
        by_view.setdefault(row['completed_view_sha256'],[]).append(row)
        prior=unique.setdefault(h,row)
        if any(prior[k]!=row[k] for k in ['bytes','canonical_host','canonical_root','relative_path','program_sha256']):
            raise ValueError('conflicting artifact relocations')
    for v,records in by_view.items():
        from e36_archive_runs import published_receipt
        doc=published_receipt(root,v)
        native={r['bundle']['sha256']:r['bundle'] for r in doc['rows']}
        for row in records:
            b=native.get(row['sha256'])
            if b is None or b['host']!='dev' or b['root']!=str(root/'artifacts-local-v1') or b['bytes']!=row['bytes']:
                raise ValueError('archived object absent from its exact native view')
    sampled=random.Random(36).sample(sorted(unique),min(3,len(unique)))
    client=Client(storage['command'],storage['program_sha256']);checks=[]
    try:
        for h in sampled:
            row=unique[h];owner=record_owner(root,row,storage)
            if row.get('encoding')=='tar-pack-v1':
                from e36_pack_archive import readback
                answer=readback(owner,row)
            elif owner.get('encoding')=='zstd':
                from e36_archive_store import ArchiveClient
                archive=ArchiveClient(owner['command'],owner['program_sha256'])
                try:answer=archive.verify(h)
                finally:archive.close()
                if any(answer[k]!=row[k] for k in ['stored_sha256','stored_bytes','encoding']):raise ValueError('final compressed archive changed')
            else:answer=client.verify(h)
            if answer['bytes']!=unique[h]['bytes']:raise ValueError('final archive readback size changed')
            checks.append(answer)
    finally:
        # EOF closes the idle verifier naturally; no process is terminated.
        if client.process is not None:
            client.process.stdin.close();client.process.wait(timeout=30);client.process.stdout.close()
    if sha(journal)!=journal_sha:raise ValueError('relocation journal changed during freeze')
    resolver_path=retain_resolver(root,Path(__file__))
    result=dict(schema='e36-artifact-location-admission-v1',admitted=True,
                journal=dict(path=str(journal),sha256=journal_sha,segments=segments),
                resolver=dict(path=str(resolver_path),sha256=sha(resolver_path)),
                storage=dict(path=str(storage_path),sha256=sha(storage_path)),
                native_ledgers=ledger_pins,unique_relocated_objects=len(unique),native_complete_views=len(by_view),
                actual_random_readbacks=checks,relocation_eligibility_replayed=True,utc=time.time(),fits=0)
    drain=root/'owner-svt-ssd-v1/archive-drain-final-v1.json'
    if drain.exists() or list(root.glob('native-archive-drain-*/launch.json')):
        from e36_archive_terminal import validate as validate_drain
        path,terminal=validate_drain(root)
        result['archive_drain_terminal']=dict(path=str(path),sha256=sha(path))
        result['archive_drain_inventory']=json.loads(Path(terminal['stop']['path']).read_text())['inventory']
    archive_path=root/'storage-archive-v2.json'
    if archive_path.exists():result['archive_storage']=dict(path=str(archive_path),sha256=sha(archive_path))
    cache_path=root/'storage-cache-v1.json'
    if cache_path.exists():
        result['cache_storage']=dict(path=str(cache_path),sha256=sha(cache_path))
        reservations=owner_dir/'cache-reservations-v1.jsonl'
        claims={}
        if reservations.exists():
            for line in reservations.read_text().splitlines():
                item=json.loads(line);previous=claims.setdefault(item['sha256'],item)
                if previous!=item:raise ValueError('conflicting final cache reservations')
        pack_claims={}
        for path in (owner_dir/'pack-intents-v1').glob('*.json'):
            item=json.loads(path.read_text())
            if item['storage']['root']=='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache':pack_claims[item['pack_id']]=item['reserved_bytes']
        probe=owner_dir/'pack-storage-admission-v1.json'
        probe_bytes=json.loads(probe.read_text())['probe_reserved_bytes'] if probe.exists() else 0
        reserved_total=sum(item['reserved_bytes'] for item in claims.values())+sum(pack_claims.values())+probe_bytes
        if reserved_total>250000000000:raise ValueError('final cache reservations exceed grant')
        for h,row in unique.items():
            if row['canonical_root']=='/mnt/user/coefficient/output/zensim/e36-bulk-2026-10-17-cache':
                if row.get('encoding')=='tar-pack-v1':
                    if row['pack_id'] not in pack_claims or row['stored_bytes']>pack_claims[row['pack_id']]:raise ValueError('cache pack lacks pre-upload reservation')
                elif h not in claims or row['stored_bytes']>claims[h]['reserved_bytes']:raise ValueError('cache object lacks pre-upload byte reservation')
        pack_pins=[];pack_documents={}
        for path in sorted((owner_dir/'pack-intents-v1').glob('*.json')):
            item=json.loads(path.read_text());pack_documents[item['pack_id']]=item
            pack_pins.append(dict(path=str(path),sha256=sha(path)))
        for row in unique.values():
            if row.get('encoding')!='tar-pack-v1':continue
            intent=pack_documents.get(row['pack_id'])
            if intent is None or any(row[k]!=intent['storage'][v] for k,v in [('canonical_host','host'),('canonical_root','root'),('program_sha256','program_sha256')]):raise ValueError('packed relocation intent owner differs')
            member=next((i for i in intent['items'] if i['sha256']==row['sha256']),None)
            if member is None or member['bytes']!=row['bytes'] or member['view_sha256']!=row['completed_view_sha256']:raise ValueError('packed relocation member differs from intent')
        result['pack_intents']=pack_pins
        review_closeout=owner_dir/'review-pin-recovery-v1.json'
        if review_closeout.exists():result['pack_review_closeout']=dict(path=str(review_closeout),sha256=sha(review_closeout))
        if any(row.get('encoding')=='tar-pack-v1' for row in unique.values()):
            durability=require_pack_directory_proof(root)
            result['pack_directory_durability']=dict(path=str(durability),sha256=sha(durability))
        if probe.exists():
            receipt=json.loads(probe.read_text());source=Path(receipt['program'])
            if sha(source)!=receipt['verifier_sha256']:raise ValueError('pack verifier source changed')
            result['pack_storage_admission']=dict(path=str(probe),sha256=sha(probe))
            result['pack_verifier']=dict(path=str(source),sha256=sha(source))
        assignment=owner_dir/'archive-assignments-v1.jsonl'
        if assignment.exists():result['archive_assignments']=dict(path=str(assignment),sha256=sha(assignment))
        result['archive_assignment_segments']=[dict(path=str(p),sha256=sha(p)) for p in sorted((owner_dir/'archive-assignment-segments-v1').glob('*.jsonl'))]
        census=owner_dir/'archive-assignment-census-v1/receipt.json'
        if census.exists():result['archive_assignment_census']=dict(path=str(census),sha256=sha(census))
        result['cache_reserved_bytes']=reserved_total
        if reservations.exists():result['cache_reservations']=dict(path=str(reservations),sha256=sha(reservations))
    readback=root/'owner-svt-ssd-v1/archive-program-readback-v1.json'
    if readback.exists():result['archive_program_readback']=dict(path=str(readback),sha256=sha(readback))
    with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result
