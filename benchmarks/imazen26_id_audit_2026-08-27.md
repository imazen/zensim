# imazen-26 ID audit — 2026-08-27

- `zenavif_lossy`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓
- `zenjpeg_lossy`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓
- `zenjxl_lossless`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓
- `zenjxl_lossy`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓
- `zenpng_lossless`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓
- `zenwebp_lossless`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓
- `zenwebp_lossy`: origins train=212 val=128 test=74 — disjoint ✓, parity ✓, fixtures-clean ✓

Eval id set (test-split ref basenames, all datasets): **808 ids**

## Cross-set hits (eval ids appearing in OTHER local training tables)
- /mnt/v/zen/zensim-training/2026-05-15-full-features/imazen26_test_120k_2026-07-16.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-05-15-full-features/imazen26_test_372col_2026-07-16.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_features_2026-07-03.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_features_qualified_2026-07-03.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_features_2026-07-02.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_pareto_2026-07-02.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_pareto_2026-07-03.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/canonical-2026-07-03/test.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext504-basic-v2-2026-07-23/ext_imazen26_720_nn_full.parquet: **792 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext504-foldable-2026-07-24/ext_imazen26_720_nn_full.parquet: **792 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/ext_hfnlproxy.parquet: **772 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext720-foldable-2026-07-24/ext_imazen26_720_nn_full.parquet: **792 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_hfnlproxy.parquet: **772 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_imazen26.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_nonphoto.parquet: **632 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/zenavif_lossy/test_944.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/zenjpeg_lossy/test_944.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/zenjxl_lossless/test_944.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/zenjxl_lossy/test_944.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/zenpng_lossless/test_944.parquet: **808 eval ids** — classify before any training use
- /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/zenwebp_lossless/test_944.parquet: **808 eval ids** — classify before any training use

## Verdict

**PASS (id half)**: splits disjoint, parity rule holds, no fixture id in any train/validate view, cross-set sweep clean.

**Remaining (registered, NOT done here): the dHash+eye half** — perceptual
near-duplicates under different filenames need the dHash-64 d≤10 screen +
user-eye verification per the 2026-05-14 policy; scheduled as a daylight pass.
