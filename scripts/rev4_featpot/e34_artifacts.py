"""E34 artifact helpers run directly, without temporary shebang wrappers."""
import argparse

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
    p.add_argument('mode',choices=['backup-check','archive-binaries'])
    mode=p.parse_args().mode
    {'backup-check':backup_check,'archive-binaries':archive_binaries}[mode]()
