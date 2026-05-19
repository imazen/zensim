# EXP-CHUNKC-PERGROUP — FALSIFIED (per-group standardizer salvage attempt)

**Status**: hypothesis FALSIFIED. Per-group standardization makes CID22
SROCC dramatically WORSE than the prior (already-falsified) chunk-c
attempt: median 0.1539 vs prior 5-group 0.5289 and ship Compression
0.8641 (Δ −0.71). AIC-3 also collapses to 0.3527 (Δ −0.47 vs ship
Compression 0.8183). This closes the EX-4 Chunk C feature line
definitively — the 19 features cannot be salvaged via simple
standardization changes.

## Hypothesis & falsification

| Aspect | Statement |
|---|---|
| Hypothesis | Adding a per-group standardizer for the 19 EX-4 Chunk C CVVDP-shape per-pair features (f324..f342) should fix the cross-corpus distribution shift identified in the prior falsification (`exp_chunkc_perpair_falsified_2026-05-18.md`) and restore the +0.005 to +0.015 CID22 lift the FEATURE-AUDIT estimated. |
| Falsification | Fails if CID22 5-seed median ≤ 0.8641 (no lift) AND KADID/TID/KonJND each within −0.05 of ship Compression. |
| Result | CID22 5-seed median **0.1539** — far below 0.8641, and also far below the prior chunk-c attempt's 0.5289 without standardization. KADID/TID/KonJND within tolerance (+0.0046 / +0.0009 / +0.1281). AIC-3 collapses to 0.3527 (Δ −0.47). |
| Conclusion | **Hypothesis falsified twice over**: not only does the per-group standardizer fail to lift CID22, it makes CID22 *worse than the prior un-standardized attempt*. The mechanism is documented below. |

## Method

### Per-group standardizer (Option B: pre-standardize at parquet-build time)

For each training group (safesyn, kadid, tid, konjnd,
cvvdp_iwssim_large), compute per-feature `(mu, sigma)` over the 19
f324..f342 columns, then z-score within the group. Output written to
`/mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup/{group}_per_group_std.parquet`.

For each validation corpus (cid22, kadid, tid, konjnd, aic3), compute
its own `(mu, sigma)` over f324..f342 and standardize. **Per-corpus
inference-time standardization** — each corpus's f324..f342 are
z-scored against its own distribution. This is the simplest answer to
cross-corpus distribution shift and is what the brief's hypothesis
predicts will fix the mechanism.

Zero-fill corpora (safesyn, cvvdp_iwssim_large) keep zeros — `(mu=0,
sigma=1)`, so `z = 0`.

Verification (sample feature f324):
- KADID train: mean=0.0000, std=1.0000, min=-4.74, max=9.95
- TID train: mean=0.0000, std=1.0000, min=-11.18, max=8.80
- KonJND train: mean=0.0000, std=1.0000, min=-7.60, max=8.05
- CID22 val (per-corpus std): mean=0.0000, std=1.0000, min=-10.45, max=4.53
- AIC-3 val (per-corpus std): mean=0.0000, std=1.0000, min=-2.44, max=1.34

All training + validation magnitudes now overlap; the global
standardize-once-fit-all problem from the prior attempt is
mathematically removed.

Build script: `scripts/exp_chunkc_pergroup/build_per_group_std_parquets.py`.
Standardizer metadata: `/mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup/standardizers.json`.

### Training recipe (5 seeds parallel)

Identical to `scripts/exp_chunkc_perpair/run_chunkc_perpair_seed.sh`
(the prior V_24 per-sample-α + chunk-c recipe), except the `--group`
paths point at per-group-standardized parquets and the binary was
rebuilt from `main@origin`:

```
zensim_mlp_train \
  --group safesyn:safesyn_per_group_std.parquet:1.0:0.0 \
  --group kadid:kadid_per_group_std.parquet:0.3:1.0 \
  --group tid:tid_per_group_std.parquet:0.3:1.0 \
  --group konjnd:konjnd_per_group_std.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:cvvdp_iwssim_large_per_group_std.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 343 \
  --target-column mix_cv40_iw60 --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed <s> --log-every 10 --early-stop-patience 0
```

Wall: ~10 min per seed, 5 seeds in parallel.

Bakes at `/mnt/v/zen/zensim-eval/exp_chunkc_pergroup_2026-05-18/pergroup_s{1..5}_h128.bin`.

### Evaluation

`bake_verdict` on each bake against the per-group-standardized
validation features-root (`/tmp/exp_chunkc_pergroup_features_root`).
Symlinks rename the val parquets to the 372col-2026-05-15 schema
filenames so `bake_verdict` finds them; the parquet_loader
auto-detects n_features=343 from f0..fN consecutive columns.

Verdict markdown files at
`/mnt/v/zen/zensim-eval/exp_chunkc_pergroup_2026-05-18/verdicts/pergroup_s{1..5}_verdict.md`.

`bake_compare` 1000-bootstrap also run for the median seed (s1) vs ship
Compression and ship Balanced. Reports at
`/mnt/v/zen/zensim-eval/exp_chunkc_pergroup_2026-05-18/bake_compare_pergroup_s1_vs_{compression,balanced,prior_chunkc_s1}.md`.

## Results — 5-seed CI

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| pergroup_s1 | 0.1539 | 0.9362 | 0.8902 | 0.9361 | 0.3573 |
| pergroup_s2 | 0.1876 | 0.9360 | 0.8893 | 0.9343 | 0.3643 |
| pergroup_s3 | 0.1547 | 0.9372 | 0.8910 | 0.9325 | 0.3484 |
| pergroup_s4 | 0.1531 | 0.9366 | 0.8902 | 0.9361 | 0.3527 |
| pergroup_s5 | 0.0912 | 0.9355 | 0.8912 | 0.9374 | 0.3382 |
| **seed median** | **0.1539** | **0.9362** | **0.8902** | **0.9361** | **0.3527** |
| ship_compression | 0.8641 | 0.9316 | 0.8893 | 0.8080 | 0.8183 |
| ship_balanced | 0.8324 | 0.9677 | 0.9729 | 0.8927 | 0.7845 |
| chunkc (no pergroup, 5gp) | 0.5289 | 0.9356 | 0.8903 | 0.9115 | 0.7403 |

### Δ tables

| Corpus | median vs ship_compression | vs ship_balanced | vs prior chunk-c |
|---|---:|---:|---:|
| CID22  | **−0.7102** | **−0.6785** | **−0.3750** |
| KADID  | +0.0046 | −0.0315 | +0.0006 |
| TID    | +0.0009 | −0.0827 | −0.0001 |
| KonJND | +0.1281 | +0.0434 | +0.0246 |
| AIC-3  | **−0.4656** | **−0.4318** | **−0.3876** |

CID22 is **catastrophic** — far worse than the prior chunk-c attempt's
already-falsified 0.5289. AIC-3 collapses similarly. KADID/TID/KonJND
all hit the training-side ceiling (essentially identical to prior
chunk-c because the standardizer doesn't change training-corpus
behavior fundamentally — the MLP sees scaled values from the same
ground-truth distribution within each group).

### bake_compare § A.9 — aggregate decisions (1000-bootstrap, seed=42)

**pergroup_s1 vs ship_compression** (`bake_compare_pergroup_s1_vs_compression.md`):

| Corpus | A SROCC | B SROCC | h_SROCC | Decision |
|---|---:|---:|---:|---|
| CID22 | 0.1539 | 0.8641 | -37.751 | **B>>A** |
| KADID | 0.9362 | 0.9316 | +93.105 | A>>B |
| TID | 0.8902 | 0.8893 | +19.524 | tied |
| KonJND | 0.9361 | 0.8080 | +26.314 | A>>B |
| AIC-3 | 0.3573 | 0.8183 | -20.753 | **B>>A** |

CID22 and AIC-3 both decisive B>>A. **Compression-trail gate FAILS**
(requires A>>B on ≥1 of {CID22, AIC-3}).

**pergroup_s1 vs ship_balanced** (`bake_compare_pergroup_s1_vs_balanced.md`):

| Corpus | A SROCC | B SROCC | Decision |
|---|---:|---:|---|
| CID22 | 0.1539 | 0.8324 | **B>>A** |
| KADID | 0.9362 | 0.9677 | **B>>A** |
| TID | 0.8902 | 0.9729 | **B>>A** |
| KonJND | 0.9361 | 0.8927 | A>>B |
| AIC-3 | 0.3573 | 0.7845 | **B>>A** |

**Balanced-trail gate FAILS** (any decisive B>>A is a ship blocker;
4 of 5 corpora are decisive B>>A).

### Smoking gun: pergroup_s1 vs prior chunk-c_s1

`bake_compare_pergroup_s1_vs_prior_chunkc_s1.md` — when BOTH bakes are
scored on the SAME per-group-standardized features-root:

| Corpus | pergroup A | chunkc B | Decision |
|---|---:|---:|---|
| CID22 | 0.1539 | 0.1497 | promising (tied) |
| KADID | 0.9362 | 0.2179 | A>>B |
| TID | 0.8902 | 0.1093 | A>>B |
| KonJND | 0.9361 | 0.2019 | A>>B |
| AIC-3 | 0.3573 | 0.0883 | promising |

The prior chunkc_s1's CID22 of 0.6011 (reported in
`exp_chunkc_perpair_falsified_2026-05-18.md`) was measured on the
ORIGINAL (un-standardized) features-root. When the same bake is
scored on per-group-standardized features-root, its CID22 collapses
to 0.1497 — the same neighborhood as pergroup_s1's 0.1539. The two
bakes are essentially indistinguishable in this scoring regime, AND
both are catastrophic.

This decisively shows:

- The prior chunk-c bake learned to **ignore** f324..f342 (the
  cross-corpus shift gave it no usable signal there); it scored 0.60
  on CID22 mostly from f0..f323.
- The pergroup bake learned to **use** f324..f342 because they were
  z-aligned across groups. But the resulting weight pattern doesn't
  generalize: it overfits to within-corpus rankings on KADID/TID/KonJND
  and produces wrong predictions on CID22.
- Both bakes are equally bad on CID22 when re-scored on
  per-group-standardized features — because the standardization
  itself washes out the cross-corpus signal-magnitude information
  that was the bake's only remaining purchase on out-of-distribution
  data.

## Mechanism (post-hoc analysis)

The per-group standardizer fixes the **mathematical scale-mismatch
problem** that the prior attempt's falsification identified
(KADID f324 std=13.9 vs CID22 std=0.21, an 80× scale gap). After
per-group standardize, every group's f324..f342 are at scale `(μ=0,
σ=1)`. So the MLP can no longer "see" magnitude as a corpus
discriminator.

But that **was the signal**. The cross-corpus magnitude carries
information about the kind of distortion: KADID's analytic
distortions produce large per-pair deltas; CID22's codec distortions
produce small deltas. When we z-normalize per-corpus, a "subtle"
CID22 codec artifact and a "severe" KADID JPEG-30 artifact both
look like `z=+2σ` to the MLP. The MLP learned the KADID `z=+2σ`
mapping (because most training pairs are from KADID + TID); applied
to CID22's `z=+2σ`, it predicts KADID-magnitude distortion where
CID22's MOS only marks subtle decline. Result: predictions are
miscalibrated against MOS.

The within-ref signal (the 13/19 features with ratio > 0.3 per the
prior audit) is preserved by per-group standardization — that's why
KADID/TID/KonJND scores are high (each corpus's within-ref ordering
is correctly modeled). But the **between-ref** discrimination across
corpora is destroyed.

In other words: **the cross-corpus magnitude was a feature, not a
bug**. The brief's hypothesis assumed removing it would help; it
hurts.

## What this rules out (definitively)

1. **EX-4 Chunk C features cannot be made cross-corpus-stable via
   standardization alone.** Neither global nor per-group works. The
   features themselves carry corpus-specific signal that any
   distribution-matching loses.
2. **The "anchor-only" path is also dead** (the prior chunk-c attempt's
   anchor-only seeds dropped CID22 to 0.03 — even worse than per-group).
3. **Distribution-matching loss (CORAL etc.)** would have the same
   problem: it can't simultaneously preserve the within-corpus signal
   and align cross-corpus distributions, because they are the same
   axis of variation.

## What's still open (lower priority)

1. **Feature-transform on raw f324..f342** before standardization
   (signed_log1p, sqrt, rank-transform). This compresses the dynamic
   range from 80× to ~5× but doesn't fully solve the magnitude-as-signal
   problem. Probably gives partial lift but won't reach the FEATURE-AUDIT
   estimate.
2. **Per-codec-family training** with a meta-picker at runtime: train
   one MLP on synth+KADID+TID, another on synth+CID22-like (e.g.,
   cvvdp_iwssim_LARGE), and route at inference based on a learned
   distortion-classifier. This is a multi-week engineering project.
3. **External cvvdp/iwssim/ssim2 scores as MLP inputs** (Pick 2 from
   `project_feature_expansion_audit`). Different mechanism — uses
   off-the-shelf metric outputs as features. Not affected by the
   per-pair-feature distribution-shift problem.

None of these are single-bake compression-trail levers. The EX-4
Chunk C line is closed.

## Verdict per § A.10 trail gates

### Balanced trail gate

- CID22: decisive B>>A (−0.6785).
- KADID, TID, AIC-3: decisive B>>A.
- KonJND: A>>B (+0.0434).
- **FAILS Balanced gate**: any decisive B>>A on a corpus is a ship blocker.

### Compression trail gate

- CID22: decisive B>>A (−0.7102).
- AIC-3: decisive B>>A (−0.4656).
- KADID, TID: tied / A>>B (within tolerance).
- KonJND: A>>B (+0.1281).
- **FAILS Compression gate**: requires A>>B on ≥1 of {CID22, AIC-3};
  we lose both decisively.

**Neither trail gate passes.** No shipping action; bakes stay in
`/mnt/v/zen/zensim-eval/exp_chunkc_pergroup_2026-05-18/` for forensic
archival.

## Bake artifacts (forensic, not for production)

```
/mnt/v/zen/zensim-eval/exp_chunkc_pergroup_2026-05-18/
  pergroup_s1_h128.bin                       seed 1 — CID22 0.1539
  pergroup_s2_h128.bin                       seed 2 — CID22 0.1876
  pergroup_s3_h128.bin                       seed 3 — CID22 0.1547
  pergroup_s4_h128.bin                       seed 4 — CID22 0.1531
  pergroup_s5_h128.bin                       seed 5 — CID22 0.0912
  verdicts/pergroup_s{1..5}_verdict.md       bake_verdict full panels
  bake_compare_pergroup_s1_vs_compression.md 1000-bootstrap A vs B
  bake_compare_pergroup_s1_vs_balanced.md    1000-bootstrap A vs B
  bake_compare_pergroup_s1_vs_prior_chunkc_s1.md  smoking gun comparison

/mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup/
  safesyn_per_group_std.parquet          196,086 rows × 343 features
  kadid_per_group_std.parquet             10,125 rows × 343 features
  tid_per_group_std.parquet                3,000 rows × 343 features
  konjnd_per_group_std.parquet             1,008 rows × 343 features
  cvvdp_iwssim_large_per_group_std.parquet 73,300 rows × 343 features
  val/cid22_per_group_std.parquet          4,292 rows × 343 features
  val/kadid_per_group_std.parquet         10,125 rows × 343 features
  val/tid_per_group_std.parquet            3,000 rows × 343 features
  val/konjnd_per_group_std.parquet         1,008 rows × 343 features
  val/aic3_per_group_std.parquet             600 rows × 343 features
  standardizers.json                       per-group (μ, σ) metadata
```

## Methodology & reproducibility

- Workspace: `~/work/zen/zensim--exp-chunkc-pergroup/`.
- New change on `main@origin` parented at `uzoqyuqx 6a4bcf13`.
- Trainer binary: `target/release/zensim_mlp_train` rebuilt from main.
- Per-group-standardized parquets built via
  `scripts/exp_chunkc_pergroup/build_per_group_std_parquets.py`
  from the 343-col extfeat parquets at
  `/mnt/v/zen/zensim-training/2026-05-18-extfeat/` (those parquets
  were produced by the prior chunk-c attempt and ARE preserved; this
  attempt did not regenerate them).
- Training script: `scripts/exp_chunkc_pergroup/run_pergroup_seed.sh`.
- Validation: `bake_verdict` (rebuilt from main) on temp features-root
  `/tmp/exp_chunkc_pergroup_features_root/` (symlinks renaming
  per-group-standardized val parquets to the 372col-2026-05-15 schema
  filenames; loader auto-detects n_features=343).
- bake_compare: 1000-bootstrap per § A.9, seed=42.

## CLAUDE.md learning entry (queued)

Append to "V_20 input-shaping + multi-bake runtime — learnings":

> ### EX-4 Chunk C with per-group standardizer — falsified (third attempt)
>
> Three attempts now falsified for the 19 EX-4 Chunk C CVVDP-shape
> per-pair features (`zensim/src/cvvdp_features.rs`, f324..f342):
>
> 1. `b94314f4` (2026-05-18 morning): 5-group with safesyn + cvvdp_LARGE
>    zero-filled, vanilla MLP without `--per-sample-alpha-head`.
>    CID22 0.5919.
> 2. `exp_chunkc_perpair_2026-05-18` (2026-05-18 night): re-trained with
>    `--per-sample-alpha-head`. 5-group CID22 median 0.5289; anchor-only
>    CID22 median 0.0323. Diagnosed as cross-corpus distribution shift
>    (KADID f324 std is 80× CID22's).
> 3. `exp_chunkc_pergroup_2026-05-18` (this attempt, 2026-05-18 night):
>    per-group standardizer (each corpus's f324..f342 z-scored to its
>    own distribution at parquet-build time). CID22 5-seed median
>    **0.1539** — WORSE than (2), far below ship Compression 0.8641.
>    AIC-3 collapses to 0.3527 (Δ −0.47 vs ship Compression).
>
> Root cause confirmed: cross-corpus magnitude IS the signal, not a
> bug. Per-group standardization preserves within-corpus signal
> (KADID/TID/KonJND all > 0.93) but destroys between-corpus
> generalization (CID22 0.15, AIC-3 0.35). The bakes overfit to
> standardized-within-corpus ranking and have no purchase on
> out-of-distribution data.
>
> The EX-4 Chunk C line is **closed**. Future attempts must
> change the feature design (not standardization). Candidate
> directions: feature-transforms on raw f324..f342 (signed_log1p,
> sqrt), per-codec-family multi-bake routing, OR external metric
> scores as MLP inputs (Pick 2 from project_feature_expansion_audit).
>
> Falsification doc: `benchmarks/exp_chunkc_pergroup_falsified_2026-05-18.md`.

## Outcome

Documented falsification. No shipping action. Per the brief's two-step
decision tree:

> If hypothesis is right: cross-corpus distribution shift no longer
> corrupts the trainer; the 19 features lift CID22 +0.005 to +0.015.
>
> If hypothesis is wrong: features are intrinsically not generalizable;
> closes the EX-4 line definitively.

**Hypothesis is wrong.** The EX-4 line is closed. Next compression-trail
levers per `project_feature_expansion_audit`:

- **Pick 2**: external metric scores (cvvdp/ssim2/iwssim) as MLP
  inputs. Medium-large expected lift. ~2 hours wall.
- **Pick 3**: true percentile pool features. Small-medium lift.
  ~6 hours wall.

Recommend Pick 2 as the next experiment.
