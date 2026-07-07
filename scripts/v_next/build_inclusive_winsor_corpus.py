#!/usr/bin/env python3
"""Build the near-lossless-INCLUSIVE winsor fit corpus for Profile B (2026-07-07).

B's original winsor fit corpus was hdr_v3mix alone (HDR JXL, 7,410 rows); its
[p0.1,p99.9] bounds don't cover the SDR near-lossless feature range, so
near-lossless SDR features fell below p0.1 and CLAMPED — 245/372 features went
constant across near-lossless, collapsing B's near-lossless dial to a ~91.5 pin.
Fix: add the zenjxl near-lossless SDR sweep to the fit corpus so the p0.1 bounds
cover it (340/372 features freed), while f155's p99.9 upper guard stays 0.776 —
still catches the 14,532 tiny-screen pathology (0.776 << 14,532).

Inputs (record shas in the methodology doc):
  hdr_v3mix : /mnt/v/output/zensim-multicodec-probe/hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet (f0..f371)
  near-lossless SDR sweep (feat_0..371): zensim-jxl-nearlossless/{refit,full}/features.parquet
Output: /mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet (zstd, f0..f371)
Then: bake_dial_refit add-winsor --in <raw> --fit-corpus <this> --lo-pct 0.1 --hi-pct 99.9
      bake_dial_refit extend-top --in <winsor> --anchor multiband_anchor_dial100.parquet
"""
import pyarrow.parquet as pq, pyarrow as pa, numpy as np
NF=372
hm=pq.read_table("/mnt/v/output/zensim-multicodec-probe/hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet", columns=[f"f{i}" for i in range(NF)])
Hm=np.column_stack([hm.column(f"f{i}").to_numpy(zero_copy_only=False).astype(float) for i in range(NF)])
nl=[]
for fp in ["/mnt/v/output/zensim-jxl-nearlossless/refit/features.parquet","/mnt/v/output/zensim-jxl-nearlossless/full/features.parquet"]:
    t=pq.read_table(fp, columns=[f"feat_{i}" for i in range(NF)])
    nl.append(np.column_stack([t.column(f"feat_{i}").to_numpy().astype(float) for i in range(NF)]))
inc=np.vstack([Hm]+nl)
pq.write_table(pa.table({f"f{i}": inc[:,i] for i in range(NF)}),
               "/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet", compression="zstd")
print(f"wrote inclusive winsor corpus: {inc.shape}")
