"""Bind the current ship panel to E36's existing native reference exclusion owner."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def native_admitted(native,cohort,reference_records):
    expected_training={r['source']:r['source_sha256'] for r in cohort['rows']}
    expected_refs={r['path']:r['sha256'] for r in reference_records}
    if (native['schema']!='native-png-dhash-audit-v1' or native['hash_era']!='bt709-encoded-luma-zenresize-lanczos-gray8-v1'
            or native['threshold']!=10 or len(native['training'])!=200 or len(native['holdout'])!=len(reference_records)
            or {r['path']:r['sha256'] for r in native['training']}!=expected_training
            or {r['path']:r['sha256'] for r in native['holdout']}!=expected_refs):
        raise ValueError('complete native current-panel hash coverage differs')
    return native['strict_flags']==0 and not any(r['strict_flag'] for r in native['close_pairs'])


def input_files(root,manifest,prepared,out):
    """Plan a byte-preserving remote reference-only audit; never decode labels."""
    out.mkdir();cohort=json.loads((root/'cohort-selection.json').read_text());panel=json.loads(manifest.read_text())
    if cohort['labels_read'] is not False or len(cohort['rows'])!=200:raise ValueError('frozen200-source cohort required')
    records={r['source']:dict(path=r['source'],sha256=r['source_sha256']) for r in cohort['rows']}
    for path in [manifest,prepared/'INPUTS.json',Path(cohort['current_family_v2']['path'])]:records[str(path)]=dict(path=str(path),sha256=sha(path))
    for index in range(len(panel['sources'])):
        for name in [f'source-{index:02d}.png',f'source-{index:02d}-recipe.json']:
            path=prepared/name;records[str(path)]=dict(path=str(path),sha256=sha(path))
    for r in records.values():r['bytes']=Path(r['path']).stat().st_size
    write(out/'files.json',dict(schema='e36-current-panel-native-input-files-v1',cohort_sha256=sha(root/'cohort-selection.json'),
        panel_manifest_sha256=sha(manifest),files=list(records.values()),labels_read=False,fits=0))
    with (out/'sources.files0').open('xb') as f:f.write(b'\0'.join(p.lstrip('/').encode() for p in sorted(records))+b'\0')
    print(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records.values()),labels_read=False,fits=0)))


def run(root,manifest,prepared,binary,out):
    cohort_path=root/'cohort-selection.json';cohort_raw=cohort_path.read_bytes();cohort=json.loads(cohort_raw)
    manifest_raw=manifest.read_bytes();panel=json.loads(manifest_raw)
    cohort_sha=hashlib.sha256(cohort_raw).hexdigest();manifest_sha=hashlib.sha256(manifest_raw).hexdigest()
    pins=json.loads((prepared/'INPUTS.json').read_text())
    if cohort['labels_read'] is not False or len(cohort['rows'])!=200 or pins['manifest_sha256']!=manifest_sha or pins['sources']!=len(panel['sources']):
        raise ValueError('current cohort/panel preparation binding differs')
    family_path=Path(cohort['current_family_v2']['path'])
    if sha(family_path)!=cohort['current_family_v2']['sha256']:raise ValueError('current canonical family owner changed')
    with family_path.open() as f:families={r['id']:r for r in csv.DictReader(f,delimiter='\t')}
    used={r['family'] for r in cohort['rows']};family_checks=[]
    for source in panel['sources']:
        match=re.fullmatch(r'imazen-screen-(\d+)',source['name'])
        if match:
            origin=match.group(1);family=families[origin]['family'] or 'origin:'+origin
            if family in used:raise ValueError('current ship reference family enters E36: '+origin)
            family_checks.append(dict(origin=origin,family=family,overlap=False))
        elif not source['name'].startswith('gb82-') or not source['source_family'].startswith('gb82-'):
            raise ValueError('current ship source requires explicit family lineage admission')
    out.mkdir();refs=out/'references';refs.mkdir();reference_records=[]
    for index,source in enumerate(panel['sources']):
        recipe=prepared/f'source-{index:02d}-recipe.json';recipe_raw=recipe.read_bytes();r=json.loads(recipe_raw)
        path=prepared/f'source-{index:02d}.png'
        if r['input_sha256']!=source['sha256'] or sha(path)!=r['output_sha256']:
            raise ValueError('current prepared reference differs from native source receipt')
        dest=refs/path.name;shutil.copyfile(path,dest)
        if sha(dest)!=r['output_sha256']:raise ValueError('prepared reference changed while copying')
        reference_records.append(dict(path=str(dest),sha256=r['output_sha256'],recipe=dict(path=str(recipe),sha256=hashlib.sha256(recipe_raw).hexdigest()),
            original_source=source))
    training=out/'training.csv'
    with training.open('x') as f:
        writer=csv.writer(f);writer.writerow(['source'])
        for row in cohort['rows']:
            if sha(row['source'])!=row['source_sha256']:raise ValueError('frozen cohort source bytes changed')
            writer.writerow([row['source']])
    with (out/'SOURCE_MANIFEST.json').open('xb') as f:f.write(manifest_raw)
    argv=[str(binary),'--native-png','--max-decode-pixels','120000000','--cid22-refs',str(refs),
        '--training-csv',str(training),'--threshold','10','--expected-training','200',
        '--expected-holdout',str(len(reference_records)),'--out-tsv',str(out/'audit.tsv')]
    with (out/'native.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    native_path=out/'audit.hashes.json';native=json.loads(native_path.read_text())
    if sha(manifest)!=manifest_sha or sha(cohort_path)!=cohort_sha or sha(family_path)!=cohort['current_family_v2']['sha256']:
        raise ValueError('cohort/panel/family metadata changed during native audit; frozen evidence retained')
    admitted=native_admitted(native,cohort,reference_records)
    write(out/'receipt.json',dict(schema='e36-current-panel-exclusion-v1',admitted=admitted,
        cohort_sha256=cohort_sha,panel_manifest_sha256=manifest_sha,panel_sources=len(reference_records),
        native_binary_sha256=sha(binary),native_hashes_sha256=sha(native_path),native_argv=argv,
        current_family_v2=cohort['current_family_v2'],family_checks=family_checks,
        reference_records=reference_records,strict_flags=native['strict_flags'],labels_read=False,fits=0,
        lineage_limit=cohort['inherited_lineage_limit'],
        gb82_lineage='declared source-family namespace plus native pixel screen; cross-corpus canonical lineage not independently reconstructed',
        note='Native pixel screen plus explicit canonical family closure; no universal recovery of unknown crop/parent lineage.'))
    if not admitted:raise ValueError('current ship panel has native dHash<=10 flags; E36 remains unqualified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','manifest','prepared','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--binary',type=Path);p.add_argument('--input-files-only',action='store_true')
    a=p.parse_args()
    if a.input_files_only:input_files(a.root,a.manifest,a.prepared,a.out)
    else:
        if a.binary is None:p.error('--binary required for actual native audit')
        run(a.root,a.manifest,a.prepared,a.binary,a.out)
