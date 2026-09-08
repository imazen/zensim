# Canonical corruption serving screen — September 8, 2026

The frozen D + nonlinear HGB composition is **not shippable**. On canonical
validation inputs it detects 93.82% of unique corruption cases and improves
strict ordering below the same-source native JPEG q20 anchor from 36.33% to
98.64%. It also incorrectly sends eight near-lossless JXL outputs, originally
scoring 97.77–98.54, to zero. The newer `real_bug` family has only 64.35%
detection. This supports a canonical refit with honest native negatives; it
does not establish a release candidate or justify a hard high-score guard.

## Scope and chronology

This follows the [native input packet](canonical_corruption_2026-09-08.md)
and its 12 train / 8 validation origins. The frozen September 6 D artifact and
historical HGB are evaluated without fitting or threshold tuning. Their
historical training provenance does not establish independence from these
inputs. The role named `validate` is the new corpus role, **not a newly proven
held-out qualification of the historical models**. No terminal human labels
were evaluated. [Preregistration and amendments](../docs/CANONICAL_CORRUPTION_SERVING_2026-09-08.md)
preceded the instrument, identity API and f32 precision checks respectively.

All 14,300 catalog/anchor rows and 760 honest native JXL/AVIF bitstreams are
scored through complete Rust `BakeScorer` composition. The optional audit in
`zensim-bench/examples/extract_features_372col.rs` records independently decoded
RGB8/file hashes, consumed features, base score, composed pixel and cached
scores, raw head probability and explicit work counts. The existing
`scripts/v_next/corruption_gate_eval.py` analyzes these records; Python does
not implement another scorer. The baked probability threshold stays 0.9.

## Identity context and feature parity

The first full audit correctly refused 678 inert rows: raw feature composition
returned zero while pixel scoring returned 100. This was the documented API
distinction, not a regression in pixel identity behavior. Zero features alone
do not prove identical pixels. The failed prototype remains in the packet.

`BakeScorer::score_features_with_identity` now lets cached pair consumers carry
independently verified decoded-pixel identity. True returns exactly 100;
false calls the unchanged complete raw-feature scorer. SDR/HDR pixel scoring
uses the same owner. A test distinguishes genuine identity from an ordinary
zero row on a firing companion and preserves negative scores. Current native
spatial adapters still explicitly reject corruption companions.

Final instrument: **15,060/15,060 complete, zero failures**, including all 678
inert identities at 100. Measured maxima are all exactly zero:

- Consumed canonical versus pixel-surface feature difference.
- Identity-aware cached versus pixel composed score difference.
- Actual stored-f32 versus fresh-f64 composed score and head probability
  differences; head fire sets agree on every row.

All **5,602,320 f32 feature cells** match the original four input tables,
joined by recorded table and row key. Disabling the audit produces an exactly
identical feature CSV. This establishes compatibility for this instrument,
feature revision and model composition, not for arbitrary historical tables.

## Results after removing duplicate source/pixel pairs

Deduplication uses `(origin, reference pixel SHA, distorted pixel SHA)` within
each role, rejecting conflicting labels or scores. Raw attempts remain intact.
Per-family statistics deduplicate within each family; families can contain
the same pixels, so their denominators must not be summed as independent cases.

| Measurement | Train inputs | Validation inputs |
|---|---:|---:|
| Raw rows | 9,036 | 6,024 |
| Unique pairs / removed duplicates | 8,213 / 823 | 5,679 / 345 |
| Unique corruptions / honest negatives | 7,725 / 488 | 5,353 / 326 |
| Corruption detection | 7,302/7,725 (94.52%) | 5,022/5,353 (93.82%) |
| Raw head false positives | 41/488 (8.40%) | 14/326 (4.29%) |
| Honest final score lowered | 33/488 (6.76%) | 8/326 (2.45%) |
| Honest JXL outputs lowered | 25/252 (9.92%) | 8/168 (4.76%) |
| Honest AVIF outputs lowered | 8/204 (3.92%) | 0/136 |
| Base strictly below q20 anchor | 2,872/7,725 (37.18%) | 1,945/5,353 (36.33%) |
| Composed strictly below q20 anchor | 7,640/7,725 (98.90%) | 5,280/5,353 (98.64%) |
| Base strictly below q10 anchor | 2,402/7,725 (31.09%) | 1,600/5,353 (29.89%) |
| Composed strictly below q10 anchor | 7,635/7,725 (98.83%) | 5,265/5,353 (98.36%) |
| `real_bug` corruption detection | 324/473 (68.50%) | 204/317 (64.35%) |

The raw head fires on eight train and six validation unique identity pairs;
verified identity keeps all their final scores at 100. All 40 honest native
JPEG anchors remain unchanged. The eight validation codec failures are:

| Origin | JXL distance | Base score | Head probability | Composed score |
|---|---:|---:|---:|---:|
| 3311 | 0.010000 | 98.232048 | 0.995781 | 0 |
| 3311 | 0.014788 | 97.906261 | 0.985507 | 0 |
| 5315 | 0.010000 | 98.541533 | 0.985507 | 0 |
| 5315 | 0.014788 | 98.407285 | 0.993292 | 0 |
| 5315 | 0.021867 | 98.220421 | 0.985507 | 0 |
| 5315 | 0.032336 | 97.914178 | 0.918750 | 0 |
| 6823 | 0.010000 | 97.770967 | 0.918750 | 0 |
| 9413 | 0.010000 | 98.049616 | 0.985507 | 0 |

These are model errors, with exact pixel/cache/f32 agreement. Correct identity
handling therefore does not repair near-lossless behavior. Later native
`real_bug` catalog entries also expose gaps hidden by the older local catalog.

## Reproduction and verification

Artifact root: `/mnt/v/output/zensim/canonical-corruption-serving-2026-09-08/`.
Windows share directory: `~/work/zensim-validation-2026-09-08/canonical-corruption-serving/`.
`INPUTS.json` pins the four tables, exact source rows and input TSV.
`BUILD_VERIFIED.json`, source archive, model copies and verification receipts
pin the instrument. Accepted results are `all-verified.{csv,jsonl,log}` and
`report-verified.json`; prototypes are retained separately by name.

| Artifact | SHA-256 |
|---|---|
| D bake | `cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6` |
| HGB w372 companion | `c95bd5f84c235e485df30f98c2009d6dbc7186618998df9e5de69ca0bc4dce15` |
| Verified extractor | `72ce76e203c0729d0719f4fd4f2a78b6de1c1e57b7b8e729659fb61f8f32e763` |
| Accepted feature CSV | `77e3da77eb021adda44c72426ccd7ee929411590c810f3ef1a52b014f31be5ea` |
| Accepted audit JSONL | `090a6a2ccbdf1b8412186477b588748899aeb4f3408af26fcba33bfeeaf211b9` |

Build the nested benchmark workspace with `training,zen-decode`. Run the
extractor on `all-pairs.tsv` with `--corpus pairs`, fresh `--out`,
`--audit-jsonl`, `--audit-bake` and `--audit-corruption-head`, then the existing
gate owner with `--audit-jsonl`, `--inputs-json` and fresh `--out-json`. The
packet contains exact commands and independent table-parity verification.

Checks passed: 16 Rust bake-surface tests including identity, HDR, ensembles
and attribution; API snapshot/check; CI-exact root Clippy; exact extractor
Clippy; minimal-feature build; scoped rustfmt; script lint. Fifteen invalid
CLI controls refuse, and report controls accept one valid packet and reject
17 damaged packets. Legacy report/scorer function ASTs remain unchanged apart
from the explicit new-mode dispatch.

The accepted audit performs 15,060 canonical extractions, 30,120 additional
native audit decodes, 15,060 candidate pixel comparisons, 60,240 cached score
calls and 30,120 auxiliary head evaluations. Canonical extraction itself also
decodes both inputs. There are **zero full encodes and zero maps** in this
screen. Pilot, failed prototype, earlier complete replay and audit-disabled
control are additional engineering work, retained in their logs. The final
audit took about 8.5 seconds under eight threads and a 16 GiB cap, with 0.23 GiB
reported peak RSS; these are run receipts, not a controlled speedup benchmark.

## Next decision

Public-surface and stored-feature compatibility are now established for these
inputs. Before fitting, finish T0 source-content admission, produce explicit
source/pixel-deduplicated training views and preregister family-disjoint fit,
calibration and evaluation identities. Extend the existing corruption trainer
so it reports the exact exported single fit; its historical default reports
a CV ensemble while exporting a different fit. Preserve that legacy replay.

Then fit against native honest near-lossless outputs and correctly labeled
inert attempts, measure the newer real-bug family and evaluate packed bytes
through Rust. Rank, floors, target coverage, HDR/color/alpha and useful native
spatial allocation remain separate release gates. No model is qualified here.
