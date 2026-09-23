"""rev4 featbank: f32-cast and storage probe over an existing Rev3 944 feature cache.
Reads FEATURE COLUMNS ONLY (never a label column). Usage:
  nice -n19 python3 f32_cast_probe.py <cache.parquet> [--rows N]
"""
import sys, os, time, argparse, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument('parquet'); ap.add_argument('--rows', type=int, default=3000)
ap.add_argument('--tmp', default='/var/tmp/rev4-featbank'); a = ap.parse_args()
cols = [f'f{i}' for i in range(944)]
t = pq.read_table(a.parquet, columns=cols)
X = np.column_stack([t[c].to_numpy().astype(np.float64) for c in cols])
S = X[:a.rows]; f32 = S.astype(np.float32).astype(np.float64)
rel = np.where(S == 0, 0.0, np.abs(f32 - S) / np.maximum(np.abs(S), 1e-300))
for lo, hi, n in [(0,156,'basic'),(156,228,'peaks'),(228,300,'masked'),(300,372,'iw'),(372,720,'v2'),(720,924,'append'),(924,944,'append2')]:
    print(f'{n} f{lo}-f{hi-1}: f32-exact cells {(f32==S)[:,lo:hi].mean():.4f} max rel {rel[:,lo:hi].max():.3e}')
print('overall max rel', rel.max(), 'outside f32 range', bool((np.abs(S) > 3.4e38).any()),
      'f32-subnormal', int(((np.abs(S) < 1.18e-38) & (S != 0)).sum()))
live = [i for i in range(944) if np.abs(X[:, i]).max() > 0]
print('rows', X.shape[0], 'live cols', len(live), 'structural-zero ids', [i for i in range(944) if i not in set(live)])
tb = pa.table({f'f{i}': pa.array(X[:, i].astype(np.float32)) for i in live})
os.makedirs(a.tmp, exist_ok=True)
for name, kw in [('zstd3_dict', dict(compression='zstd', compression_level=3)),
                 ('zstd3_bss', dict(compression='zstd', compression_level=3, use_dictionary=False,
                                    column_encoding={f'f{i}': 'BYTE_STREAM_SPLIT' for i in live}))]:
    p = os.path.join(a.tmp, f'probe_{name}.parquet'); pq.write_table(tb, p, row_group_size=65536, **kw)
    r = pq.read_table(p); ok = all(np.array_equal(r[c].to_numpy(), tb[c].to_numpy()) for c in tb.column_names)
    print(name, os.path.getsize(p), 'B', round(os.path.getsize(p) / X.shape[0]), 'B/row', 'exact_roundtrip', ok); os.remove(p)
print('source parquet bytes', os.path.getsize(a.parquet))
