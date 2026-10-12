#!/usr/bin/env python3
"""Retire explicitly listed sidecars, retaining selected columns byte for byte.

The inventory is a JSON list of {source, destination, columns}. A null destination
removes the sidecar; otherwise columns are retained in a new Parquet file. This
performs storage maintenance only and does not assess, fit or select models.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def retire(item):
    import pyarrow.parquet as pq
    source = Path(item['source'])
    if source.is_symlink():
        raise ValueError(f'expected owned regular sidecar: {source}')
    destination = Path(item['destination']) if item['destination'] else None
    if destination:
        if destination.exists():
            raise FileExistsError(destination)
        columns = item['columns']
        old = pq.ParquetFile(source)
        import pyarrow as pa
        schema = pa.schema([old.schema_arrow.field(c) for c in columns])
        with pq.ParquetWriter(destination, schema, compression='zstd') as writer:
            for batch in old.iter_batches(batch_size=65536, columns=columns):
                writer.write_batch(batch.replace_schema_metadata(None))
        new = pq.ParquetFile(destination)
        if old.metadata.num_rows != new.metadata.num_rows or new.schema_arrow.names != columns:
            raise ValueError(f'row count/schema mismatch: {destination}')
        for a, b in zip(old.iter_batches(batch_size=65536, columns=columns),
                        new.iter_batches(batch_size=65536), strict=True):
            if not a.equals(b, check_metadata=False):
                raise ValueError(f'column bytes differ: {destination}')
        receipt = {'source': str(source), 'destination': str(destination),
                   'rows': new.metadata.num_rows, 'sha256': digest(destination)}
    else:
        receipt = {'source': str(source), 'destination': None}
    source.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    items = json.loads(args.inventory.read_text())
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    # Each completed destructive step gets a durable receipt immediately.
    with args.receipt.open('x') as receipt:
        for item in items:
            result = retire(item)
            receipt.write(json.dumps(result) + '\n')
            receipt.flush()
            print('Sidecar maintenance verified.', flush=True)


if __name__ == '__main__':
    main()
