# Wave-6 ensemble-teacher twins — block-storage pointer (2026-08-04)

Six parquets (~3.6 GB) built by `scripts/canonical_corpus/build_teacher944.py`
for SOTA-944 amendment 6 arm F. NOT in git (ML data-pipeline discipline §7b).
They are fully regenerable: the builder is committed, the member bakes are on
the campaign path, and the target rule is deterministic (no RNG, no fit).

Regenerate:

```sh
B=/mnt/v/output/zensim/bakes/sota944/bakes
scripts/canonical_corpus/build_teacher944.py --tag ensk2 \
  --members "$B/C_co3a_s1301.bin,$B/C_co2a_s1307.bin" \
  --out-dir /mnt/v/output/zensim/bakes/sota944/teacher_ensk2
scripts/canonical_corpus/build_teacher944.py --tag ensk5 \
  --members "$B/C_co3a_s1301.bin,$B/C_co2a_s1307.bin,$B/C_co3a_s1319.bin,$B/C_co1b_s1303.bin,$B/C_em944_s31.bin" \
  --out-dir /mnt/v/output/zensim/bakes/sota944/teacher_ensk5
```

Target rule (frozen, and verified BIT-EXACT against the committed EM4 teacher
before use — max|rule − stored| = 0.0 over all 369,237 rows of the three twins):
`lo,hi = quantile(raw_safesyn, 0.001), quantile(raw_safesyn, 0.999)`;
`human_score = clip((raw − lo)/(hi − lo), 0, 1)`; ONE affine for all three twins.

## `teacher_ensk2`  (k=2)

Members: `C_co3a_s1301.bin`, `C_co2a_s1307.bin`

Affine: `[-19.754818468093873, 12.413235281467463]`

| twin | rows | raw mean | teacher mean | clip frac | sha256 |
|---|---:|---|---|---|---|
| safesyn | 111068 | 1.242966 | 0.652807 | 0.002017 | `787025fae1c1dffad25bbf9c9688e4acd1bb9363f9a4505cd434d0823370a80e` |
| tbig | 208169 | 1.637993 | 0.665000 | 0.001412 | `c6d0d6d9e4a36d1d2a3bcb900c4c5879f2a7b3024cf61243c6eb60fcd389d585` |
| kadis | 50000 | -0.203086 | 0.607896 | 0.000220 | `1cef7a2617118727eee393e8927b49eea3dc533901fe0a0ca5e1849519b73f25` |

## `teacher_ensk5`  (k=5)

Members: `C_co3a_s1301.bin`, `C_co2a_s1307.bin`, `C_co3a_s1319.bin`, `C_co1b_s1303.bin`, `C_em944_s31.bin`

Affine: `[-15.57560635547638, 11.566778017616276]`

| twin | rows | raw mean | teacher mean | clip frac | sha256 |
|---|---:|---|---|---|---|
| safesyn | 111068 | 1.367033 | 0.624271 | 0.002017 | `9c6f46c9322e01c24adf20b2379fce049eb58e16d2d0003cfcd3935ec8141de9` |
| tbig | 208169 | 1.362109 | 0.624026 | 0.001504 | `517c64e618acd6d1fe422f7860fb04e3824b407e94e6d429dfca6926889a497c` |
| kadis | 50000 | -0.241952 | 0.565079 | 0.000360 | `7aa97870f6d06cc6d201a48ef4e21b212896cabac2844405882bd70d14d377ee` |

Teacher rank-agreement vs the amendment-3 EM4 teacher (SROCC between TARGET
columns, deterministic stride subsample, stats from `panel --batch`):

| twin | n | SROCC(ensk2, EM4) | SROCC(ensk5, EM4) | SROCC(ensk2, ensk5) |
|---|---:|---|---|---|
| safesyn | 22,214 | 0.98609 | 0.99172 | 0.99747 |
| tbig | 20,817 | 0.93036 | 0.97153 | 0.98172 |
| kadis | 25,000 | 0.73250 | 0.79014 | 0.93467 |
