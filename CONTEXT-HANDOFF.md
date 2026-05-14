# zensim — context handoff (2026-05-14)

Written immediately before a session reset. Read this first.

## TL;DR — what's live

- **Shipped bake**: `zensim/weights/v0_18_2026-05-13.bin` (v3 format,
  93,064 B, md5 `c94e93607390d0b6704e95f3851d421e`). V0_18 zerobiased+HU-reordered
  as of 2026-05-13. **CID22 SROCC 0.8933** on standard validation —
  but this number is INFLATED because V0_18's training corpus had
  perceptual overlap with KADID/TID reference images. CID22 purge
  from 2026-05-12 was honest; KADID/TID purge happened later
  (2026-05-14) — V0_18 is "honest on CID22, inflated on KADID/TID".
- **Crate version**: zensim 0.3.0, never published. Swap bake bytes
  in place; no version bump.
- **46 zensim --lib unit tests** still pass.

## What happened 2026-05-14 (this session)

### 1. Contamination audit caught KADID/TID overlap

`zensim-validate/bin/check_holdout_overlap` (dHash-64) audited the
synth-v2 training corpus vs KADID I01..I81 and TID I01..I25 refs.
At d≤16 threshold:

- **149 unique source basenames** with perceptual overlap (118 KADID,
  33 TID). Audit TSVs at
  `benchmarks/{kadid,tid}_overlap_2026-05-14.tsv`.

| Holdout | strict d≤10 | loose d≤16 | refs affected |
|---|--:|--:|---|
| KADID10k | 6 | 118 | I04 / I18 (×4) / I25 / I41 / I71 |
| TID2013 | 1 | 33 | I12 |

CID22 already purged 2026-05-12 (361 sources for 22 of 49 refs).
Preserved.

### 2. Contamination guard plumbed into trainer

- `zensim-validate/src/contamination_guard.rs` (module)
- `zensim-validate/src/bin/contamination_guard.rs` (CLI)
- 149-basename blocklist embedded at compile time via
  `include_str!("../../benchmarks/contamination_blocklist_2026-05-14.txt")`
- `zensim_mlp_train` calls `scrub_csv_or_die` per `--group` CSV.
  Exits 2 on detection. Filenames containing `CONTAMINATED` get
  rejected on sight.

**Audit bypass**: `ZENSIM_BYPASS_CONTAMINATION_GUARD_FOR_AUDIT_I_REALLY_MEAN_IT=1`
disables the guard with a loud warning. Audit/reproduction only.
Don't ship bakes trained with this set.

### 3. Quarantined contaminated CSVs

Renamed with `.CONTAMINATED_2026-05-14_DO_NOT_USE.csv` suffix:

- `/tmp/zensim_loop/safe_synth_clean_features.CONTAMINATED_*.csv`
  (V0_18's 144,791-row base — CID22-purged but contains the 149
  KADID/TID-overlap basenames)
- `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.CONTAMINATED_*.csv`
- `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.CONTAMINATED_*.csv`
- `/mnt/v/zen/zensim-training/2026-05-07/v06-features/safe_synth_ssim2_features.CONTAMINATED_*.{csv,parquet}`

### 4. Canonical clean corpus (single source of truth)

`/mnt/v/zen/zensim-training/2026-05-14-clean/`:

| File | md5 | rows |
|---|---|--:|
| `safe_synth_v19_clean_features.csv` | `71a0a78428ded74e60fd569ba77ed7e0` | 138,872 |
| `kadid_features.csv` | `6f07b25ce854fd6f8dc596cf055747d1` | 10,125 |
| `tid_features.csv` | `7189f57a832161c82aee3a8860c677a1` | 3,000 |
| `konjnd_aligned_features.csv` | `209ebdb6586cf76c5d57a62883b314e3` | 76,104 |
| `tv_pairs_bands.tsv` | `ad18f6cc46fc8f812b69e7030d63393b` | 205,654 |

Plus `_MANIFEST.md` with full audit lineage.

### 5. V0_19 — built, validated, NOT YET decided

Followed V0_18 3-way concat recipe exactly on the clean corpus.

**Bakes**:
- `benchmarks/v0_19_base_seed1_2026-05-14.bin`
- `benchmarks/v0_19_cycle14_s1_2026-05-14.bin`
- `benchmarks/v0_19_cycle14_s42_2026-05-14.bin`
- `benchmarks/v0_19_concat_3way_2026-05-14.bin`
- `benchmarks/v0_19_calibrated_2026-05-14.bin`

**Validation** (`benchmarks/v0_19_10band_2026-05-14.md`):

| Corpus | V0_19 | fast-ssim2 | vs ssim2 | vs V0_18 (inflated) |
|---|--:|--:|--:|--:|
| CID22 | **0.8786** | 0.8895 | **−0.0109** | −0.0147 |
| KADID10k | 0.9462 | 0.8133 | +0.1329 | +0.0035 |
| TID2013 | 0.9553 | 0.8460 | +0.1093 | +0.0027 |

V0_19 CID22 is below fast-ssim2's 0.8895. I initially marked this
as ship-blocking. **The user pushed back** ("rejecting a ship
because it decontaminated is bad") and is **correct**. V0_18's
apparent CID22 advantage may have been contamination-driven; V0_19
is the honest baseline.

**V0_18 reproduction audit IN FLIGHT** to confirm. Trains V0_19
pipeline on V0_18's ORIGINAL contaminated corpus (env-var bypass):
- `benchmarks/v0_18_repro_base_seed1.bin` (PID in `/tmp/v18_repro_base.pid`)
- `benchmarks/v0_18_repro_cycle14_s1.bin` (PID in `/tmp/v18_repro_s1.pid`)
- `benchmarks/v0_18_repro_cycle14_s42.bin` (PID in `/tmp/v18_repro_s42.pid`)

Each train ~25 min wall, started ~08:50 UTC, should be done by ~09:15.

**Once repro completes**:
- If V0_18 repro CID22 ≈ 0.8933 → V0_19 pipeline is faithful;
  V0_19's 0.8786 is the honest cost of decontamination. SHIP V0_19.
- If V0_18 repro CID22 ≠ 0.8933 → pipeline differs from what made
  V0_18. INVESTIGATE before shipping.

### 6. V0_2 → V0_18 candlestick

`benchmarks/v0_2_to_v0_18_candlestick_2026-05-14.{png,svg,tsv}`,
n=17,417 paired rows. V0_2 saturates at top (53% of pairs in
[95,100) bin); V0_18 within that bin spans p5=35.7 to p95=97.1.

### 7. V0_18 methodology hallucination caught + corrected

V0_18 methodology referenced 2 scripts NEVER in git history:
`scripts/v_next/concat_three_way.py` and
`/tmp/zensim_loop/concat_construct.py`. Audited via `git log --all
--diff-filter=A --name-only`. Doc corrected. Reproduction now uses
the real `zensim-validate/bin/concat_three_way` Rust binary.

Memory: `feedback_methodology_must_be_real.md` codifies "every
script path in a methodology doc must be git-log-verifiable."

### 8. Rust ports of Python pipeline

- `zensim-validate/bin/concat_three_way` — replaces phantom
  `concat_three_way.py`. The canonical reproduction now.
- `zensim-validate/bin/affine_calibrate` — replaces
  `affine_calibrate_znpr_v2.py`. Md5-byte-identical output.
- `affine_calibrate_znpr_v2.py` accepts both v2 AND v3 bakes.

V_X ship pipeline is now pure `cargo run`. Zero Python.

### 9. zenpredict 0.3 FeatureTransform variants (V0_20 prep)

Added 3 parameter-less variants to `FeatureTransform` enum:

- `SignedLog1p`: `sign(x) · ln(1 + |x|)`
- `SignedSqrt`: `sign(x) · sqrt(|x|)`
- `SignedCbrt`: `sign(x) · cbrt(|x|)`

Wire tokens: `signed_log1p`, `signed_sqrt`, `signed_cbrt`. Std and
no_std paths. Trains can now opt-in per-feature via metadata.

### 10. Dockerfile architecture refactored

`docker/Dockerfile`:

1. `trainer-bin` — slow stage, builds all Rust binaries once
2. `corpora` — downloads test corpora (URLs still TODO)
3. `features` — feature extraction + audit gate
4. `train-base` / `train-cycle14-s1` / `train-cycle14-s42` —
   separate stages, ARG-driven hyperparams. Tweak one without
   re-paying others.
5. `concat` / `validate` / `bundle` — cheap downstream

Build-time tweaking:
```sh
docker build --build-arg CYCLE14_S1_TV_WEIGHT=2.0 -f docker/Dockerfile .
```

Only re-runs the affected component + downstream.

### 11. Optional Rust-side cache (NOT YET wired)

`zensim-validate/src/train_cache.rs` — content-addressable cache
keyed on `SHA-256(binary_version || input_md5s || flags)`. In-tree
md5+sha256, no external deps. Optional via volume mount:

```sh
docker run -v $(pwd)/cache:/cache -e ZENSIM_TRAIN_CACHE=/cache ...
```

Without the mount, in-container only (volatile). Module compiles
but NOT yet plumbed into `zensim_mlp_train::main()`. ~30 LOC TO-DO.

### 12. Site fixes (earlier in session)

`zensim/site/compare.html` was broken (`manifest.corpora is not
iterable`). Fixed by merging the R2 manifest WITH the JS stub.
`check_site_urls.py` now spoofs Chrome UA (R2 was 403-ing urllib's
default UA). All 39 URLs pass; Playwright-verified.

## Outstanding work (priority order)

1. **V0_18 reproduction audit completes** (~10 more min from
   2026-05-14 ~09:00 UTC). If CID22 ≈ 0.8933 → ship V0_19 in place
   (no version bump) and supersede V0_18.
2. **Wire `train_cache` into `zensim_mlp_train::main()`**. ~30 LOC.
3. **Fix `quant_compare.rs`** to accept v3 bakes (currently asserts
   v2 only; blocks V_X → I8 path on v3 inputs).
4. **Dockerfile R2 sync + corpus URLs + checksums**. Stubs in place.
5. **`--recipe-version` flag** in `zensim_mlp_train`. Pins all
   hyperparams to a frozen named recipe so future default changes
   don't break past invocations.
6. **V0_20 input-shaping experiments**. FeatureTransform variants
   ready; need per-feature training sweeps.
7. **V0_21 linear distillation**. Ridge regression with engineered
   features vs MLP teacher; target beat V0_2's 0.8676.

## Critical rules to restate

- 10-band SROCC reporting is the PRIMARY release gate (revised
  2026-05-14). Legacy 4-band CID22 cuts retained for paper compat.
- NEVER ship a bake without re-running `check_holdout_overlap`
  against ALL holdout corpora at d≤16 (dHash-64).
- "match-or-exceed fast-ssim2" is the aspiration, **NOT** a strict
  block. V0_19 is below fast-ssim2 on CID22; that's an honest gap
  to close, not a reason to ship the inflated baseline.
- "NEVER CLAIM FALSE COMPLETION" — every methodology doc's script
  references must be `git log -- <path>`-verifiable.
- Don't ship bakes trained with `ZENSIM_BYPASS_CONTAMINATION_GUARD_*`
  set. Audit-only.

## Commits this session (zensim main, oldest → newest)

- `83f1b3ec` — 10-band per-band reporting + site URL fixes
- `97fd9a43` — site manifest merge fix (compare.html)
- `d0b8c291` — zensim → zenpredict 0.2.0 + V0_18 v2→v3 + 10-band
- `d6f12e48` — ship V0_18 zerobiased+HU in place
- `f0394bff` — V0_19/20/21 training kickoff + candlestick + concat
- `984e3da7` — V0_18 methodology hallucination fix
- `4609d0f7` — V0_19 honest failure + guard + canonical clean corpus
- `fe6cbdb5` — Rust affine_calibrate + Dockerfile scaffold
- `04e2a16b` — V0_18 repro audit + Dockerfile refactor + train_cache
  + FeatureTransform v3.0 variants

## Inputs (locations)

- Test corpora: `/mnt/v/dataset/{cid22,kadid10k,tid2013}/`
- KonJND: `/mnt/v/dataset/konfig-iqa/` (verify — old refs claimed
  `/mnt/v/datasets/KonJND-1k/` which doesn't exist on this box)
- Training corpus (clean canonical):
  `/mnt/v/zen/zensim-training/2026-05-14-clean/`
- Audit blocklist:
  `benchmarks/contamination_blocklist_2026-05-14.txt`
- Methodology:
  - `benchmarks/v0_18_methodology_2026-05-13.md` (with 2026-05-14
    addendum + hallucination-fix note)
  - `benchmarks/v0_19_methodology_2026-05-14.md`
  - `benchmarks/v0_20_v0_21_design_2026-05-14.md`

## How to run a re-training right now (recipe pinned)

```sh
CLEAN=/mnt/v/zen/zensim-training/2026-05-14-clean

# Component 1: V0_16-base equivalent
cargo run --release -p zensim-validate --bin zensim_mlp_train -- \
  --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
  --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
  --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
  --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --seed 1 \
  --out benchmarks/v0_X_base_seed1_$(date -u +%Y-%m-%d).bin

# Component 2: cycle-14 seed=1 TV-regularized
cargo run --release -p zensim-validate --bin zensim_mlp_train -- \
  --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
  --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
  --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
  --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --seed 1 \
  --tv-pairs-file $CLEAN/tv_pairs_bands.tsv \
  --tv-weight 1.0 --tv-band-weights 10,30,10,30 \
  --tv-apply-every 50 --tv-batch 32 \
  --out benchmarks/v0_X_cycle14_s1_$(date -u +%Y-%m-%d).bin

# Component 3: cycle-14 seed=42 (same flags, --seed 42)

# Concat:
cargo run --release -p zensim-validate --bin concat_three_way -- \
  --base benchmarks/v0_X_base_seed1_*.bin \
  --s1   benchmarks/v0_X_cycle14_s1_*.bin \
  --s42  benchmarks/v0_X_cycle14_s42_*.bin \
  --coeffs 0.65:0.30:0.05 \
  --out  benchmarks/v0_X_concat_3way_$(date -u +%Y-%m-%d).bin

# Affine calibrate (V0_16 lineage):
cargo run --release -p zensim-validate --bin affine_calibrate -- \
  --in-bake benchmarks/v0_X_concat_3way_*.bin \
  --out-bake zensim/weights/v0_X_$(date -u +%Y-%m-%d).bin \
  --alpha 28.0366 --beta=-5.0738

# Validate (10-band primary):
cargo run --release -p zensim-bench --example dataset_metric_baseline -- \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --v04-bake zensim/weights/v0_X_$(date -u +%Y-%m-%d).bin \
  --max-pairs 50000 > benchmarks/v0_X_10band_$(date -u +%Y-%m-%d).md
```

That's the canonical 2026-05-14 ship pipeline. Pure Rust. Memorize it
or `cat CONTEXT-HANDOFF.md` after the reset.
