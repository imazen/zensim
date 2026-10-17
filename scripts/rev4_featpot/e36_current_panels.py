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


def run(root,manifest,prepared,binary,out):
    cohort_path=root/'cohort-selection.json';cohort=json.loads(cohort_path.read_text())
    panel=json.loads(manifest.read_text());pins=json.loads((prepared/'INPUTS.json').read_text())
    if cohort['labels_read'] is not False or len(cohort['rows'])!=200 or pins['manifest_sha256']!=sha(manifest) or pins['sources']!=len(panel['sources']):
        raise ValueError('current cohort/panel preparation binding differs')
    family_path=Path(cohort['current_family_v2']['path'])
    if sha(family_path)!=cohort['current_family_v2']['sha256']:raise ValueError('current canonical family owner changed')
    families={r['id']:r for r in csv.DictReader(family_path.open(),delimiter='\t')}
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
        recipe=prepared/f'source-{index:02d}-recipe.json';r=json.loads(recipe.read_text())
        path=prepared/f'source-{index:02d}.png'
        if r['input_sha256']!=source['sha256'] or sha(path)!=r['output_sha256']:
            raise ValueError('current prepared reference differs from native source receipt')
        dest=refs/path.name;shutil.copyfile(path,dest)
        reference_records.append(dict(path=str(dest),sha256=sha(dest),recipe=dict(path=str(recipe),sha256=sha(recipe)),
            original_source=source))
    training=out/'training.csv'
    with training.open('x') as f:
        writer=csv.writer(f);writer.writerow(['source'])
        for row in cohort['rows']:
            if sha(row['source'])!=row['source_sha256']:raise ValueError('frozen cohort source bytes changed')
            writer.writerow([row['source']])
    shutil.copyfile(manifest,out/'SOURCE_MANIFEST.json')
    argv=[str(binary),'--native-png','--max-decode-pixels','120000000','--cid22-refs',str(refs),
        '--training-csv',str(training),'--threshold','10','--expected-training','200',
        '--expected-holdout',str(len(reference_records)),'--out-tsv',str(out/'audit.tsv')]
    with (out/'native.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    native_path=out/'audit.hashes.json';native=json.loads(native_path.read_text())
    expected_training={r['source']:r['source_sha256'] for r in cohort['rows']}
    expected_refs={r['path']:r['sha256'] for r in reference_records}
    if (native['schema']!='native-png-dhash-audit-v1' or native['hash_era']!='bt709-encoded-luma-zenresize-lanczos-gray8-v1'
            or native['threshold']!=10 or len(native['training'])!=200 or len(native['holdout'])!=len(reference_records)
            or {r['path']:r['sha256'] for r in native['training']}!=expected_training
            or {r['path']:r['sha256'] for r in native['holdout']}!=expected_refs):
        raise ValueError('complete native current-panel hash coverage differs')
    admitted=native['strict_flags']==0 and not any(r['strict_flag'] for r in native['close_pairs'])
    write(out/'receipt.json',dict(schema='e36-current-panel-exclusion-v1',admitted=admitted,
        cohort_sha256=sha(cohort_path),panel_manifest_sha256=sha(manifest),panel_sources=len(reference_records),
        native_binary_sha256=sha(binary),native_hashes_sha256=sha(native_path),native_argv=argv,
        current_family_v2=cohort['current_family_v2'],family_checks=family_checks,
        reference_records=reference_records,strict_flags=native['strict_flags'],labels_read=False,fits=0,
        lineage_limit=cohort['inherited_lineage_limit'],
        note='Native pixel screen plus explicit canonical family closure; no universal recovery of unknown crop/parent lineage.'))
    if not admitted:raise ValueError('current ship panel has native dHash<=10 flags; E36 remains unqualified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','manifest','prepared','binary','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.root,a.manifest,a.prepared,a.binary,a.out)
