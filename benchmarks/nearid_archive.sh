#!/usr/bin/env bash
set -euo pipefail
root=/mnt/v/output/zensim/nearid-2026-10-09
mirror=/mnt/tower/output/zensim/nearid-2026-10-09
if [[ -e "$root/NAS_VERIFIED.json" ]]; then
    echo "REFUSE: NEARID archive already completed" >&2
    exit 1
fi
cp benchmarks/nearid_archive.sh "$root/archive.executed.sh"
python3 scripts/lint_scripts.py > "$root/lint-archive.log"
if [[ ! -f "$root/curves.pre-layout.png" ]]; then
mv "$root/curves.png" "$root/curves.pre-layout.png"
mv "$root/curves.svg" "$root/curves.pre-layout.svg"
python3 - "$root" <<'PYIN'
import sys,json
from pathlib import Path
sys.path.insert(0, 'scripts')
from prodqual_label_free import nearid_plot
root=Path(sys.argv[1]);rows=[json.loads(line) for model in ('seed0','B','A') for line in (root/f'{model}.jsonl').read_text().splitlines()];nearid_plot(root,rows)
PYIN
fi
if [[ ! -f "$root/nearid-program-source.tar.gz" ]]; then
jj file list -r @- > "$root/tracked-files.txt"
tar -czf "$root/nearid-program-source.tar.gz" -T "$root/tracked-files.txt"
fi
cargo metadata --locked --format-version 1 --no-deps > "$root/cargo-metadata.json"
rustc -Vv > "$root/rustc.txt"
python3 - "$root" <<'PYIN'
import sys,hashlib,json
from pathlib import Path
root=Path(sys.argv[1]); previous=root/'SHA256SUMS.json'
if previous.exists():
    destination=root/('SHA256SUMS.previous-'+hashlib.sha256(previous.read_bytes()).hexdigest()+'.json')
    assert not destination.exists(), 'preserve every earlier manifest'
    previous.rename(destination)
files={str(p.relative_to(root)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS.json','NAS_VERIFIED.json')};(root/'SHA256SUMS.json').write_text(json.dumps(files,indent=2)+'\n');print('Evidence files',len(files),'bytes',sum((root/k).stat().st_size for k in files))
PYIN
mkdir -p "$mirror"
rsync -rt --no-owner --no-group --no-perms "$root/" "$mirror/"
python3 - "$root" "$mirror" <<'PYIN'
import sys,hashlib,json,random
from pathlib import Path
root,mirror=map(Path,sys.argv[1:]);hashes=json.loads((root/'SHA256SUMS.json').read_text()); digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for name,expected in hashes.items():assert digest(mirror/name)==expected,name
assert digest(root/'SHA256SUMS.json')==digest(mirror/'SHA256SUMS.json')
chosen=random.Random(20261009).sample(sorted(hashes),3);receipt={'pass':True,'verified_files':len(hashes),'verified_bytes':sum((root/k).stat().st_size for k in hashes),'mirror':str(mirror),'three_random_files':{k:hashes[k] for k in chosen},'source_archive_sha256':hashes['nearid-program-source.tar.gz'],'binary_sha256':hashes['bin/serve_custom_bake'],'build_commit':(root/'BUILD_COMMIT.txt').read_text().strip(),'evidence_source_commit':(root/'EVIDENCE_CODE_COMMIT.txt').read_text().strip()}
(root/'NAS_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n');(mirror/'NAS_VERIFIED.json').write_bytes((root/'NAS_VERIFIED.json').read_bytes());assert digest(root/'NAS_VERIFIED.json')==digest(mirror/'NAS_VERIFIED.json');print(json.dumps(receipt,indent=2))
PYIN
# The completed pipeline owns no live Cargo job. Preserve the replay binary above before cache deletion.
if [[ -d "$CARGO_TARGET_DIR" ]]; then
test -f "$CARGO_TARGET_DIR/.rustc_info.json"
du -sb "$CARGO_TARGET_DIR" > "$root/CARGO_TARGET_DELETED.txt"
rm -rf "$CARGO_TARGET_DIR"
fi
cp "$root/CARGO_TARGET_DELETED.txt" "$mirror/CARGO_TARGET_DELETED.txt"
