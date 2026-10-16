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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source','destination']:
        parser.add_argument('--'+name, type=Path, required=True)
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
    receipt = dict(time_utc=datetime.now(timezone.utc).isoformat(), source=str(args.source),
        destination=str(args.destination),verified_immutable_payloads=rows,
        scope='incremental mirror while campaign is running; final completeness not claimed')
    for root in [args.source,args.destination]:
        with (root/'LAST_MIRROR_VERIFIED.json').open('w') as stream:
            json.dump(receipt,stream,indent=2)
            stream.write('\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
