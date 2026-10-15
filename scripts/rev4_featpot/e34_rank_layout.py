"""Explicitly embed native E34 rows into the frozen rank archive's layout."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from e34_augmentation import sha,write
from e34_p4_finish import ROOT
from v2_common import fx1_declaration

archive=Path('/mnt/v/output/zensim/e33-impl-2026-10-09/packet/container-scratch-1/fit-cell/9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008/rev4-featpot/v2e29')
receipt=archive/'wide/main/real/receipt.json'
width=json.loads(receipt.read_text())['width']
source=ROOT/'repairs/augmentation.parquet'
manifest=json.loads(source.with_suffix('.parquet.manifest.json').read_text())
raw_width=manifest['width'];assert (raw_width,width)==(1825,1853)
decl=fx1_declaration();consumed=set(decl['direct'])|{i for pair in decl['products'] for i in pair}
assert max(consumed)<raw_width
out=ROOT/'repairs/rank-inputs';out.mkdir();table=out/'augmentation.parquet'
n=0
with pq.ParquetWriter(table,pq.read_schema(source),compression='zstd') as writer:
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256):
        cols=list(batch.columns)
        for i in [2,3]:
            a=np.asarray(cols[i].values).reshape(batch.num_rows,raw_width)
            padded=np.zeros((batch.num_rows,width),dtype=np.float32);padded[:,:raw_width]=a
            assert np.array_equal(padded[:,:raw_width].view(np.uint32),a.view(np.uint32))
            assert not padded[:,raw_width:].view(np.uint32).any()
            cols[i]=pa.ListArray.from_arrays(pa.array(np.arange(batch.num_rows+1,dtype=np.int32)*width),pa.array(padded.reshape(-1)))
        writer.write_batch(pa.RecordBatch.from_arrays(cols,schema=batch.schema));n+=batch.num_rows
assert n==manifest['repair_rows']
result=dict(manifest,width=width,native_width=raw_width,table_sha256=sha(table),source_table_sha256=sha(source),rank_archive_receipt_sha256=sha(receipt),rank_archive_receipt=str(receipt),structural_zero_append=list(range(raw_width,width)),consumed_ids=sorted(consumed),layout_rule='exact native f32 prefix plus explicitly unconsumed structural-zero append; no feature identity inferred from width')
write(table.with_suffix('.parquet.manifest.json'),result)
print(json.dumps(dict(width=width,native_width=raw_width,repair_rows=n,table_sha256=result['table_sha256'],structural_zero_append=28)),flush=True)
