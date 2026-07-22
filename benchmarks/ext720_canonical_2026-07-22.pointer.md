# ext720-canonical-2026-07-22 — pointer (data lives on block storage, not git)

The 11 local-leg 720-feature extraction datasets (149,195 rows) were
consolidated into one canonical directory with a unified `_MANIFEST.json`
(per-corpus sha256, build_commit, pairs provenance, role, target semantics):

- Canonical: `/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/`
- R2: `s3://zentrain/ext720-canonical-2026-07-22/` (13 objects, 871 MiB)
- Tower: `/mnt/tower/output/zensim-ext720-canonical-2026-07-22/` (sha-verified 2026-07-22)
- Index entry: `~/work/zen/DATA_PROVENANCE.md` § ext720-canonical-2026-07-22

| corpus | rows | sha256 (first 16) |
|---|--:|---|
| ext_cid22val | 4,292 | ac532ccf2ad32bd2 |
| ext_aic3 | 600 | bc6ec37d8ee5f483 |
| ext_csiq | 866 | bb6a86044331edb4 |
| ext_live | 779 | 9442705ea8db0a35 |
| ext_kadid | 10,125 | e0fd550764f3ab4a |
| ext_tid | 3,000 | 1b56e3a3d93c3964 |
| ext_safesyn_full | 111,068 | 6b687e47ab602d5a |
| ext_cid22_train201 | 17,611 | 582fd85932445c5f |
| ext_aic4 | 300 | 0437ca05129558b7 |
| ext_konjnd_jpeg_val | 504 | a27088d975582db7 |
| ext_sdr25 | 50 | 33c45026d05900c9 |

Schema: `ref_basename, human_score, f0..f719` (f64, ZSTD, NaN/null-free);
f0..f371 = frozen v1 with-iw, f372..f719 = v2-348 bounded (append-only).
Waves: 2026-07-19 (@6f191264) + 2026-07-20 (@9e7516d7), both pre pool-SIMD
(numeric note in the manifest; drift policy record:
`benchmarks/v2_ref_reuse_perf_2026-07-21.md`).

Produced by `scripts/canonical_corpus/promote_ext720_canonical.py`
(idempotent; dated source dirs kept in place with POINTER.md breadcrumbs).
Fleet legs (T-big / T-safe / bigcodec valdigits) are separate write-backs —
index in DATA_PROVENANCE when they land.
