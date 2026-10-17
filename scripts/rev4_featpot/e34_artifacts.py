"""E34 artifact helpers run directly, without temporary shebang wrappers."""
import argparse


def e36_manifest(root, build_commit):
    """Index completed preparation evidence; this never releases fitting."""
    import hashlib
    import json
    from pathlib import Path
    root = Path(root).resolve(strict=True)
    assert build_commit
    cohort = json.loads((root/'cohort-data/_MANIFEST.json').read_text())
    audit = json.loads((root/'artifact-audit-final/complete.json').read_text())
    closure = json.loads((root/'cohort-closure/complete.json').read_text())
    assert cohort['fits'] == audit['fits'] == closure['fits'] == 0
    assert cohort['complete_for_fitting'] is False
    files = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p == root/'_MANIFEST.json':
            continue
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[str(p.relative_to(root))] = dict(sha256=digest, bytes=p.stat().st_size)
    result = dict(schema='e36-preparation-evidence-v1', build_commit=build_commit,
        labels_read=False, fits=0, complete_for_fitting=False, files=files,
        cohort_selection_sha256=closure['sha256'],
        feature_build_commit=cohort['build_commit'], teacher_build_commit=cohort['teacher_build_commit'],
        note='initial generation only; bulk coverage and trainer/evaluation admissions remain open')
    with (root/'_MANIFEST.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(files=len(files), fits=0, complete_for_fitting=False)))

def backup_check():
    import hashlib,json
    from pathlib import Path
    source=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5')
    target=Path('/mnt/tower/output/zensim/e34-prep-2026-10-15/part5')
    files=sorted(p for p in source.rglob('*') if p.is_file() and not p.is_symlink() and '.git' not in p.relative_to(source).parts and '.jj' not in p.relative_to(source).parts)
    def digest(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    for p in files:assert digest(p)==digest(target/p.relative_to(source)),p
    receipt=dict(files=len(files),all_sha256_equal=True,excluded=['input symlinks','source clone .git/.jj internal history'],source=str(source),mirror=str(target))
    print(json.dumps(receipt))

def archive_binaries():
    import hashlib,json,shutil
    from pathlib import Path
    dest=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/bin')
    paths={
      'panel-noise':Path('target/debug/panel'),
      'e34-pixels-probes':Path('target/debug/examples/e34_pixels'),
      'zensim-mlp-train-smokes':Path('target/debug/zensim_mlp_train'),
      'e34-reference-features':Path('tools/e34-reference-features/target/debug/e34-reference-features'),
      'e34-jxl-scalar':Path('tools/e34-jxl-scalar/target/debug/e34-jxl-scalar'),
      'native-rev5-extractor':Path('/var/tmp/rev5-extract/target/release/examples/extract_features_372col')}
    records={}
    for name,p in paths.items():
        target=dest/name
        assert not target.exists()
        shutil.copy2(p,target)
        with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        records[name]=dict(source=str(p.resolve()),sha256=digest,bytes=target.stat().st_size)
    with (dest/'archive.json').open('x') as f:json.dump(records,f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['backup-check','archive-binaries','e36-manifest'])
    p.add_argument('--root')
    p.add_argument('--build-commit')
    args=p.parse_args()
    if args.mode == 'e36-manifest':
        assert args.root and args.build_commit
        e36_manifest(args.root, args.build_commit)
    else:
        {'backup-check':backup_check,'archive-binaries':archive_binaries}[args.mode]()
