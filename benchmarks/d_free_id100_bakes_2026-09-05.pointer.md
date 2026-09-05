# D+free arms — block-storage pointer (2026-09-05)

Record: [`benchmarks/d_free_id100_2026-09-05.md`](d_free_id100_2026-09-05.md).
Nothing here is in git (bakes are small but the probes/anchors are not, and the
no->30 KB rule plus ML-pipeline discipline 7b put the whole set in block storage).

Root: `/mnt/v/output/zensim/dfree-2026-09-05/`

## Bakes (ZNPR v3, 944 declared inputs, f16, single additive layer)

| file | bytes | sha256 |
|---|--:|---|
| `bakes/Fctl.bin` | 8530 | `41a14a4f2e97a6a93cf84c54db76abc559104e64660edd06812637839cec7975` |
| `bakes/Fctl_id100.bin` | 8484 | `313f95b9e4c69eab79c24eb88f1ffefecee33fd82183262c4a2926f8abc21806` |
| `bakes/Fctl_id100_raw.bin` | 8484 | `313f95b9e4c69eab79c24eb88f1ffefecee33fd82183262c4a2926f8abc21806` |
| `bakes/Fctl_id100negrich.bin` | 8485 | `92242c93adb2f1684b1b7f0782107942424da4324f339d129c9075ead68e835d` |
| `bakes/Fctl_id100negrich_raw.bin` | 8485 | `92242c93adb2f1684b1b7f0782107942424da4324f339d129c9075ead68e835d` |
| `bakes/Ffree.bin` | 8693 | `8ad4b032b5a319a2bd7650244bfc22a3e4bccf3a312474d07da8824d77f98c6b` |
| `bakes/FfreeC.bin` | 8714 | `5d8ed51945876315e5a3fb4d731b85fe053682f49ef71fc7eaa943075e0e4763` |
| `bakes/FfreeC_id100.bin` | 8668 | `368a105ea75a9ff0e893dc6c94c3ac4216cac5604fdeefede6a62d64ffd94a8f` |
| `bakes/FfreeC_id100_raw.bin` | 8668 | `368a105ea75a9ff0e893dc6c94c3ac4216cac5604fdeefede6a62d64ffd94a8f` |
| `bakes/FfreeC_id100negrich.bin` | 8669 | `33d889c62e6b8d8d0477ab1f67b8916d30b1aff977a3b0823ff54275e695a32e` |
| `bakes/FfreeC_id100negrich_raw.bin` | 8669 | `33d889c62e6b8d8d0477ab1f67b8916d30b1aff977a3b0823ff54275e695a32e` |
| `bakes/Ffree_id100.bin` | 8648 | `0c722a01fff5c34fe12092658e89209eddf61c3d83a02c462ff2913af2132768` |
| `bakes/Ffree_id100_raw.bin` | 8648 | `0c722a01fff5c34fe12092658e89209eddf61c3d83a02c462ff2913af2132768` |
| `bakes/Ffree_id100negrich.bin` | 8649 | `8641c2106b36aac9509e37c1c4f7523324af3442fa480f14203daf8718b59a74` |
| `bakes/Ffree_id100negrich_raw.bin` | 8649 | `8641c2106b36aac9509e37c1c4f7523324af3442fa480f14203daf8718b59a74` |
| `bakes/Fpeaks.bin` | 8607 | `21450d2d1d8edddbcc029cdc0c05e6265fb65872c7d36e77e950483d85e5f3cf` |
| `bakes/Fpeaks_id100.bin` | 8562 | `733225a77876e0f93a863410d8bb80a20300a5abb4dc639018d49a9df1d0fbcf` |
| `bakes/Fpeaks_id100_raw.bin` | 8562 | `733225a77876e0f93a863410d8bb80a20300a5abb4dc639018d49a9df1d0fbcf` |
| `bakes/Fpeaks_id100negrich.bin` | 8563 | `557bc62bc462cbe01292abce442480d020260d7fce034867398acf0e269e671e` |
| `bakes/Fpeaks_id100negrich_raw.bin` | 8563 | `557bc62bc462cbe01292abce442480d020260d7fce034867398acf0e269e671e` |

## Instruments built by this lane (era `folded720append2pools`, r1b-pools944-2026-08-30)

| file | rows | sha256 |
|---|--:|---|
| `probes/identity_probe_944pools_2026-09-05.parquet` | 39 | `31fc440381a38d188ca8f91f9b5799daa2a19b9565f5a355cb6dcad86ebc1655` |
| `probes/negtail_probe_944pools_2026-09-05.parquet` | 2000 | `bafc89943befd195b0372e76afa91dab76a5c1f6c569bd53f33f942c3b8d67d8` |
| `anchors/anchor944_r1b_dial_clamped.parquet` | 1976 | `235dd65ca3f80255df0c696f8a0569ba3c5b7101f28375ae076f8ebd899d0cb4` |
| `anchors/anchor944_r1b_dial_clamped_id21.parquet` | 1997 | `2871bb4e4afd425e98a4477d836a30e226f8f56f7b269c1653d25e843f2e551c` |
| `anchors/anchor944_r1b_dial_negrich.parquet` | 1976 | `5742aa7bae758f82f1389fc58fd90d6b9b52b383083e5653c92a70618f188188` |
| `anchors/anchor944_r1b_dial_negrich_id21.parquet` | 1997 | `7af3e25e16b96f20a23ce50455d5b027bbc07baad3a45186478e589b483b115e` |

## Pinned inputs (NOT produced here; sha256 verified at run time)

- `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/ext_safesyn_full.parquet` — `e33ec91e8fa1989b38d0693293ea20da3a57cc54f07a9e9345c5441c4a3ff778`
- `/mnt/v/output/zensim/freefeats-2026-09-01/quality-probe/g_safe_human944.npz` — `f1301440d0fb33f29d291df72b77b6e654434ac279475f3a4749119d36170aea`
- `/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet` — `694e16c4520a5d41b986cebe5ab9c8390c1eabdfef15b44386fe71abcd3c5593`
- `/mnt/v/output/zensim/wlin7b-2026-08-30/v2_ab_extract_PREFIX_PINNED` — `fc0d780bf8b7739a6d7c5e6a4f35028e9250b69cedaaf52ebed662eb42e1afcd`
- `/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_944grid.tsv` — `f8a9bc17b50c016ab853cc2ec88487768ce5069098e1ca2559363d8bc92bb68e`

## Other directories

- `peer/` — the three `peer_ssim2` cell tables + the derived G-ADDR json/markdown + the round-trip control
- `verdicts/` — `bake_verdict --full-json` per arm (12 corpora, per-pair blocks)
- `gaddr/` — `--gaddr-json` per arm at full f64
- `perpair/` — per-corpus `(human, pred)` dumps feeding the paired bootstrap
- `speed/` — the W4 sweeps (`s2bar_{1,8}t.log`), the 10-start 2304/1T set (`starts/`),
  and `ffab/` (the owner-instrument 15c/15f re-measurement of section 8.1)
- `work/` — era-control extractions, contrib TSVs, bootstrap logs
