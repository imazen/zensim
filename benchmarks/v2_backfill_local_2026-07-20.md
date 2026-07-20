# v2 720-feature dataset backfill — LOCAL leg (2026-07-20)

Division of labor per `docs/V2_EXPERIMENT_PLAN_2026-07-20.md` §E1: the
zenmetrics session owns the FLEET backfill (T-big, T-safe, 40×cx43, in
flight, `.workongoing`: "launch 40x cx43 full backfill" — NOT touched by
this session in any way, per hard boundary). This session owns the LOCAL
leg: H-aic4, H/T-konjnd-JPEG, H-sdr25, T-cid201, + two time-boxed
investigations (hf_nearlossless, kadis_negrich).

All output under `/mnt/v/output/zensim/v2-backfill-2026-07-20/` (pairs
TSVs, `ext_<corpus>.{csv,parquet}`, `_MANIFEST.json` with per-corpus
`build_commit`). Extractor: `v2_ab_extract` (720 = frozen v1-372 ++
v2-348), built at zensim main `9e7516d7480165a1d561e3a3bb6a7d3431a29cd2`.
New pairs-TSV builders landed in the canonical owner file
`scripts/canonical_corpus/build_fr_corpus_pairs.py` (`build_aic4`,
`build_konjnd_jpeg_val`, `build_sdr25`, `build_cid22_train201`).

## Result

| corpus | rows | pairs in TSV | skipped | status | notes |
|---|--:|--:|--:|---|---|
| H-aic4 | 300 | 300 | 0 | DONE | matches plan's "~300" exactly |
| H-konjnd (JPEG val) | 504 | 504 | 0 | DONE | matches plan's "~500"; raw mean-PJND target, [22,70]-ish |
| H-sdr25 (JPEG-AI scoreable) | 50 | 50 | 0 | DONE | plan said "~95k" — CORRECTED, see below |
| T-cid201 | 17,611 | 17,611 | 0 | DONE | exact match to plan's pre-registered count |
| T-konjnd (active-mix train, ~10k) | — | — | — | **SKIPPED — flagged** | no raw-path discriminator + needs zenmetrics CVVDP |
| T-hfnl (hf_nearlossless, 900+300) | — | — | — | **SKIPPED — flagged** | pixels confirmed gone (ephemeral /tmp scratchpad) |
| T-negrich (kadis_negrich) | — | — | — | **SKIPPED — flagged** | provenance already documented as lost, in its own manifest |
| eval grids @720 | — | — | — | not attempted | infra item I-2, not one of this leg's 6 work items |

**Total: 18,465 rows extracted at 720 features, 0 skips on any built corpus.**
Wall time: ~2.5 min compute (aic4 3s + konjnd 4s + sdr25 1s + cid201 122s)
+ investigation/build time; session total including investigation ≈ 40 min.

## H-sdr25: the "~95k" pair count was never real (finding, not a shortfall)

The plan doc's Datasets table lists H-sdr25 as "95k raw triplets" and flags
"investigate triplet→pair layout first." Investigation found: the ~95k rows
in `JPEG_AI_SDR_subjective_data/*.csv` are **triplet comparison responses**
(pivot-relative "which side is more distorted" judgments), not independent
labeled pairs — you cannot turn one triplet response into one (ref,dist,score)
row. They only become absolute per-stimulus scores after an ordered-probit
joint MLE reconstruction, which `scripts/v_next/reconstruct_sdr25_jnd.py`
already performed (T0 eval anchor, BUILT 2026-07-02, `docs/DATA_SPLITS.md:108`)
— output: 116 reconstructed stimuli, 5 images × up to 6 codecs.

Of those 116, only the JPEG-AI codec subset (5 images × 10 levels = 50) has
locally-available pixels; the other 5 codecs are anchor stimuli whose
bitstreams are not in the public zip (confirmed via sparse/inconsistent
per-image `dlevel` coverage in the reconstructed parquet — e.g. AVIF has 1-5
dlevel samples per image, never all 10). **50 pairs is the entire scoreable
set, not a subsample of a larger extractable one.** `build_sdr25()` reformats
the existing reconstruction into pairs and was verified byte-identical
(paths + scores) to a pre-existing but undocumented
`/mnt/v/output/zensim-multicodec-probe/sdr25_eval_pairs.tsv` from a prior
session — independent confirmation the mapping is right, not a guess.

Runtime was therefore ~1s, not the "~30 min run-heavy" the plan anticipated
for a 95k-scale extraction. Docs already correctly describe this
(`DATA_SPLITS.md:108` says "Scoreable subset = 5×10 JPEG-AI PTC crops") — the
V2_EXPERIMENT_PLAN's Datasets table row is the one still carrying the stale
"95k" figure; flagging for correction there.

## T-cid201: reused an already-built raw-path workspace TSV

`canonical-2026-05-21/_workspace/cid22_train_ssim2.tsv` already had
`ref_path, dist_path, ref_basename, codec, q, ssim2_gpu` for all 17,611
pairs (built by `v11_extract_cid22_train.py` +
`v11_cid22_train_backfill_cvvdp_iwssim.py`). `build_cid22_train201()`
reformats it (human_score = ssim2_gpu/100, matching the documented
`cid22_train_norm.human_score == ssim2_gpu/100` invariant) rather than
re-deriving the 201-ref split. Verified programmatically (not assumed):
the 201 unique ref_basenames have **zero overlap** with the 49-ref
`CID22_validation_set.csv` `reference_img` basenames — the sacred human-MOS
holdout is untouched.

First extraction attempt hit the Bash tool's 2-minute default timeout
(17,611 pairs × 720-feature compute ≈ 122s, right at the boundary) and was
killed with no partial output (the extractor writes its CSV only once, at
the end — no incremental flush). Re-ran detached in the background; verify
long extractions have headroom past 120s before assuming a 2-min foreground
call is enough.

## Flags (skipped, not guessed) — full detail in `_MANIFEST.json`

**T-konjnd** (konjnd-dense active-mix, ~10k JPEG rows expected): investigated
all 3 local copies of the source parquet via direct pyarrow schema
inspection. None carry a `ref_path`/`dist_path` column, and none carry a
quality-level discriminator — only `ref_basename` duplicated 20× per ref
with no way to tell which of the 100 available JPEG quality levels backs a
given row. The target itself (`human_score`/`active_mix_raw`,
`mix_cv*_iw*` columns) is a CVVDP+IW-SSIM metric blend — CVVDP scoring is a
zenmetrics-adjacent GPU capability, out of bounds per this session's hard
boundary. A legitimate, fully-local, well-documented **alternative** exists
(`konjnd_full_scored.csv`, 50,400 JPEG rows, raw paths + `gpu_ssimulacra2`,
matches the v1 `load_konjnd_full` loader) but is a different corpus (5×
larger, ssim2-anchored not CVVDP+IW-SSIM-anchored) — not silently
substituted; flagged as a follow-up option instead.

**T-hfnl** (hf_nearlossless train, 900+300 rows expected): traced provenance
through `_MANIFEST_hf_nearlossless.json` → `zensim-jxl-nearlossless/refit/
pareto.tsv`, whose `image_path` column pointed at a **prior session's
per-session `/tmp/claude-*/.../scratchpad/` path**. Verified 2026-07-20 that
directory no longer exists — this is the exact failure mode CLAUDE.md's
"`/tmp` IS BANNED" rule documents (scratch paths get wiped unpredictably).
Additionally the `encoded_filename` column for the distorted bitstream is
empty on 100% of rows, and the reference format was `.jxl` (unsupported by
this session's `zen_io.rs`, which decodes PNG/JPEG/BMP only) — even without
the wipe, this corpus's pixels were never durably persisted. Not recoverable
without a fresh sweep re-run.

**T-negrich** (kadis_negrich, subset expected): its own manifest
(`_MANIFEST_kadis_negrich.json`, written 2026-07-15) already documents the
gap verbatim — "No script in the repo writes the source parquet... selection
rule (266,111 rows out of KADIS-700k) is unrecorded" and "source dropped
KADIS `source_id`... cannot be verified leak-free." This is a stronger case
than "R2-only" (the task's anticipated shape for this item): there is no key
anywhere, local or remote, that identifies which 266,111 of KADIS-700k's
700,000 cells these rows are. Pulling the full R2 canonical parquet would
build a *different* negative-rich sample, not recover this one.

## Provenance

`zensim` main commit for every extraction: `9e7516d7480165a1d561e3a3bb6a7d3431a29cd2`.
Builder source: `scripts/canonical_corpus/build_fr_corpus_pairs.py`
(`build_aic4`, `build_konjnd_jpeg_val`, `build_sdr25`, `build_cid22_train201`
added this session). Full per-corpus detail (source files, row-level
verification, safety checks) in
`/mnt/v/output/zensim/v2-backfill-2026-07-20/_MANIFEST.json`.
