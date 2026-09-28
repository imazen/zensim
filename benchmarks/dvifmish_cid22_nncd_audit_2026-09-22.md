# CID22-49 duplicate audit, CID22-B(23) correction, NNCD registration audit — 2026-09-22

Lane `dvifmish` (Part 2 §2 and the NNCD registration of §5). All numbers measured
on 2026-09-22 with existing owners; artefacts under
`/mnt/v/output/zensim/dvifmish-eval-2026-09-22/audit/` and `.../cid22b23/`
(sha256 in that directory's `SHA256SUMS`).

## 1. CID22-49 against itself

Owner: `check_holdout_overlap --native-png` (binary `target/release/check_holdout_overlap`,
built 2026-09-20), 49 validation references as both holdout and "training" set,
threshold 10, review band 16. Pixel adjudication: `flag_confirm`
(`zensim-bench/target/release/examples/flag_confirm`, Mitchell to the reference's
geometry, luma RMSE on 0–255, RMSE/std(ref), NCC).

| pair (A-side first where applicable) | dHash d | luma RMSE | RMSE/std | NCC | verdict |
|---|---|---|---|---|---|
| `844297.png` (A) / `3316926_opo25u.png` (B) | 0 | 0.736 | 0.014 | 0.9999 | **same picture** |
| `373965.png` (A) / `2887497.png` (B) | 14 | 98.8 | 1.311 | −0.148 | unrelated |
| `3316926_opo25u.png` (B) / `2887497.png` (B) | 16 | 69.0 | 1.293 | 0.207 | unrelated |
| `3653963.png` (A) / `2887497.png` (B) | 16 | 74.1 | 1.231 | 0.403 | unrelated |
| `844297.png` (A) / `2887497.png` (B) | 16 | 69.0 | 1.294 | 0.207 | unrelated |
| `3316926.png` (A) / `3316926_opo25u.png` (B) | > 16 | 73.9 | 1.458 | −0.007 | unrelated (shared stem only) |

All 49 files are 512×512. One exact-content duplicate; no other near-duplicate.

## 2. CID22-B re-scored on the 23 clean references (correction of the spent read)

Same per-row scores as the single registered read of 2026-09-21
(`/mnt/v/output/zensim/dvifm-verdict-2026-09-20/scores/*__on__cid22b.csv`, the
era-corrected tables), the 89 rows of `3316926_opo25u.png` removed, statistics
from the verdict lane's own `build_verdict.metrics` / `ref_boot_delta`
(2,000 reference-cluster draws, seed 20260920). No model refit, no new label read.

| model | SROCC 24 | SROCC 23 | KROCC 24 | KROCC 23 | PLCC 24 | PLCC 23 |
|---|---|---|---|---|---|---|
| dvifm_gate | 0.7738 | 0.7797 | 0.5698 | 0.5765 | 0.7685 | 0.7735 |
| fast-ssim2 | 0.9131 | 0.9250 | 0.7395 | 0.7582 | 0.9174 | 0.9285 |
| zensim B | 0.8899 | 0.9000 | 0.7096 | 0.7250 | 0.8933 | 0.9046 |
| zensim D | 0.8795 | 0.8831 | 0.6910 | 0.6964 | 0.8798 | 0.8834 |
| R915_basic228_ens5 | 0.8968 | 0.9115 | 0.7163 | 0.7378 | 0.9011 | 0.9152 |
| R915_y60_ens5 | 0.8613 | 0.8737 | 0.6838 | 0.7016 | 0.8596 | 0.8711 |

n = 2,100 → 2,011. The 24-reference column reproduces the verdict record exactly.

| paired Δ SROCC dvifm_gate − peer | 24 refs: mean [CI95] | 23 refs: mean [CI95] |
|---|---|---|
| fast-ssim2 | −0.1344 [−0.1850, −0.0888] | −0.1407 [−0.1873, −0.0951] |
| zensim B | −0.1118 [−0.1656, −0.0648] | −0.1163 [−0.1642, −0.0705] |
| zensim D | −0.1025 [−0.1420, −0.0625] | −0.1004 [−0.1393, −0.0584] |
| R915_basic228 | −0.1187 [−0.1691, −0.0717] | −0.1278 [−0.1734, −0.0866] |
| R915_y60 | −0.0848 [−0.1396, −0.0210] | −0.0911 [−0.1414, −0.0268] |

Every model gains 0.004–0.015 SROCC once the duplicated picture is removed; the
ordering and every conclusion of `benchmarks/dvifm_verdict_2026-09-20.md` are
unchanged.

## 3. NNCD-IQA before its first read

- Archives verified against `SHA256SUMS`; extracted to
  `/var/tmp/dvifmish/datasets/nncd-iqa/`: 16 references (768×512 PNG), 64 per codec
  (JPEG 2000 as P6 PNM; four learned codecs as PNG) = 320.
- dHash, 16 NNCD references vs 4,544 training sources (joint-core-v1 references,
  SafeSyn fit + development sources, TID2013 and KADID-10k references, CID22 train 201
  and validation 49, KonFiG references): 37 pairs in the review band, 2 strict
  (`image6.png` vs two SafeSyn sources, d = 9). All five pairs with d ≤ 12 adjudicated
  unrelated by `flag_confirm` (NCC 0.21–0.43, RMSE/std ≥ 1.23).
- Crop containment (`research/2026-09-dvifm/dvifmish-eval/crop_containment.py`,
  unscaled crop search on a 4× luma grid, refined at full resolution): **16 of
  TID2013's 25 references are exact unscaled 512×384 crops (NCC 1.0000) of NNCD's 16
  references, one to one**: I03→image2, I04→image14, I06→image11, I07→image9,
  I09→image4, I10→image12, I11→image10, I12→image6, I15→image7, I16→image5,
  I17→image8, I19→image18, I20→image1, I21→image13, I22→image15, I23→image3.
  Full-frame comparison (resize, no crop) finds none (best NCC 0.78), which is why the
  whole-frame dHash did not flag them.

Consequence: NNCD is EVAL-only but shares every scene with TID2013. Any model fitted
on TID2013 rows has seen NNCD's photographs.
