"""Verify three random immutable encoded/map payloads after an evidence mirror."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
from datetime import datetime, timezone


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# Receipts are written only after payload verification, so exclude their own
# self-referential hashes from the immutable payload inventory.
RECEIPTS = {"LAST_MIRROR_VERIFIED.json", "ARCHIVE_FILES.json"}


def verify_all(source, destination, complete_marker):
    """Compare a finished source tree byte-for-byte before moving its local copy."""
    source_real,destination_real=source.resolve(),destination.resolve()
    assert source_real != destination_real and source_real not in destination_real.parents and destination_real not in source_real.parents, "archive roots must be distinct and disjoint"
    assert not complete_marker.is_absolute() and ".." not in complete_marker.parts
    assert (source / complete_marker).is_file(), "source completion marker missing"
    assert (destination / complete_marker).is_file(), "destination completion marker missing"
    def inventory(root):
        return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
                if p.is_file() and p.name not in RECEIPTS}
    left, right = inventory(source), inventory(destination)
    assert set(left) == set(right), "archive file population differs"
    files = []
    for relative, src in sorted(left.items()):
        expected = sha(src)
        assert expected == sha(right[relative]) == sha(src), relative
        files.append(dict(relative=relative,sha256=expected,bytes=src.stat().st_size))
    return dict(schema="complete-evidence-archive-files-v1",complete_marker=str(complete_marker),files=files,
                file_count=len(files),total_bytes=sum(f["bytes"] for f in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source','destination']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--complete-marker', type=Path, help='also SHA256-verify the complete finished tree; path relative to source')
    args = parser.parse_args()
    choices = [p for p in args.source.rglob('*') if p.is_file() and
               (p.suffix in ['.jxl','.avif','.webp','.jpg','.zmap'] or p.name.endswith('.data.png'))]
    assert len(choices) >= 3
    chosen = secrets.SystemRandom().sample(choices, 3)
    rows = []
    for src in chosen:
        relative = src.relative_to(args.source)
        dst = args.destination/relative
        expected = sha(src)
        assert expected == sha(dst) == sha(src), relative
        rows.append(dict(relative=str(relative),sha256=expected,bytes=src.stat().st_size))
    full = verify_all(args.source,args.destination,args.complete_marker) if args.complete_marker else None
    if full is not None:
        for root in [args.source,args.destination]:
            with (root/'ARCHIVE_FILES.json').open('x') as stream:
                json.dump(full,stream,indent=2)
                stream.write('\n')
    receipt = dict(time_utc=datetime.now(timezone.utc).isoformat(), source=str(args.source),
        destination=str(args.destination),verified_immutable_payloads=rows,
        scope='incremental mirror while campaign is running; final completeness not claimed')
    if full is not None:
        receipt.update(scope='complete finished evidence tree SHA256-verified, plus three random encoded payloads',
                       complete_marker=str(args.complete_marker),file_count=full['file_count'],total_bytes=full['total_bytes'],
                       archive_files_sha256=sha(args.source/'ARCHIVE_FILES.json'))
    for root in [args.source,args.destination]:
        with (root/'LAST_MIRROR_VERIFIED.json').open('w') as stream:
            json.dump(receipt,stream,indent=2)
            stream.write('\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
