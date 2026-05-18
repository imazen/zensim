# EXP-CHUNKC-PERPAIR — FALSIFIED (2026-05-18 re-attempt)

**Status**: hypothesis FALSIFIED on CID22 (Δ −0.33 to −0.83 SROCC vs ship
Compression). Re-attempted after prior agent's `b94314f4` falsification
(commit on `feat/ex4-xyb-frontend-extfeat` branch, CID22 0.5919) which
was diagnosed as "training-data zero-fill on safesyn + cvvdp_LARGE".
This re-attempt fixed the recipe (added `--per-sample-alpha-head` which
prior agent omitted) and tested **two configurations** to discriminate
zero-fill vs feature-design as root cause:

1. **5-group recipe** (matching V_24 Compression ship + 343 features):
   safesyn + kadid + tid + konjnd + cvvdp_LARGE. f324..f342 zero-filled
   for safesyn + cvvdp_LARGE (96% of training data).
2. **Anchor-only recipe**: kadid + tid + konjnd only. All training
   rows have populated CVVDP features (no zero-fill).

**Both configurations failed CID22 catastrophically.** This isolates
the root cause to **feature-design** (cross-corpus distribution shift),
NOT the zero-fill. The prior agent's diagnosis was incomplete.

## Hypothesis & falsification

| Aspect | Statement |
|---|---|
| Hypothesis | Adding 19 EX-4 Chunk C CVVDP-shape per-pair features (DKL Δstats, Weber band ratios, CSF-weighted band-energy ratios, mutual-masking residuals, Minkowski β=3 luma pool) to V_24 per-sample-α training should lift CID22 SROCC by ≥ 0.005 OR AIC-3 SROCC by ≥ 0.005 (FEATURE-AUDIT estimate: +0.005 to +0.015). |
| Falsification | CID22 SROCC ≤ 0.8641 across 3 seeds AND AIC-3 SROCC ≤ 0.8183 → hypothesis dead. |
| Result | **5-group**: CID22 0.43–0.60 (median 0.53); AIC-3 0.72–0.76 (median 0.74). **Anchor-only**: CID22 0.03–0.14 (median 0.03); AIC-3 0.54–0.60 (median 0.58). Both FAR below thresholds. |
| Conclusion | Feature-design falsification. The 19 features show strong per-pair signal in-corpus but exhibit catastrophic cross-corpus distribution shift (KADID 80× the per-pair feature scale of CID22 on f324). |

## Re-attempt context

- Prior session (2026-05-18 ~01:14 UTC): commit `b94314f4` falsified
  EX-4 Chunk C with CID22 0.5919. Diagnosed as "zero-fill on safesyn
  + cvvdp_LARGE" — prior agent posited that re-extracting those
  corpora's per-pair features would restore CID22.
- This re-attempt (2026-05-18 ~23:30 UTC): fresh dispatch with explicit
  brief to ensure corpus coverage + correct recipe + falsification
  doc on failure.

### Fixes vs prior attempt

| Issue | Prior `b94314f4` bake | This re-attempt |
|---|---|---|
| `--per-sample-alpha-head` flag | OMITTED (vanilla MLP) | INCLUDED |
| Recipe | 5-group identical to V_22-mix | 5-group identical to V_24-per-sample-α Compression ship |
| Training script | `scripts/v_next/v25_v2_extfeat_perpair_launch.sh` | `scripts/exp_chunkc_perpair/run_chunkc_perpair_seed.sh` |
| Seed coverage | 1 seed (s3) | 5 seeds (s1..s5) + 3 anchor-only seeds (s1..s3) |
| Anchor-only sanity check | NONE | RUN (decisive isolation of feature-design failure) |

### What did NOT change between attempts

- Same 343-col input parquets at `/mnt/v/zen/zensim-training/2026-05-18-extfeat/`.
- Same zero-fill state on `safesyn_extfeat_343.parquet` (196,086 rows
  × f324..f342 all zero) and `cvvdp_iwssim_large_extfeat_343.parquet`
  (73,300 rows × f324..f342 all zero).
- Same `mix_cv40_iw60` target column.
- Same h=128, 300 epochs, 50k pairs/epoch, PWRC=5.0, NiN 0.1.

## Pre-flight: per-pair signal check

Per `feedback_per_ref_features_are_noise.md` Refinement: features must
show within-ref variation, otherwise they carry zero RankNet signal.

Audit on KADID + TID + CID22 (table reports ratio of mean within-ref
stddev to overall stddev per feature; ratio > 0.3 = strong per-pair
signal, < 0.05 = per-ref-only):

| Corpus | refs | features with ratio > 0.3 | features with ratio < 0.05 |
|---|--:|--:|--:|
| KADID | 81 | **13/19** | 6/19 (f326, f328, f330, f331, f332, f333 — DKL ref-only blocks) |
| TID | 25 | **13/19** | 6/19 (same as KADID) |
| CID22 | 49 | **11/19** | 8/19 |

The features carry per-pair signal. The 6 zero-ratio features are the
documented "ref-only" stats (DKL ref_std per channel, Weber band 0-3
ref-only means) — by design they're constant within a ref. These add
weight-load to the MLP but contribute no ranking gradient (MLP can
learn to zero out their weights).

So the design is NOT broken on the per-pair-signal axis. The failure
is elsewhere.

## Cross-corpus distribution shift (root cause)

Audit: p99 − p01 range per feature per corpus (the dynamic range
the MLP must standardize across).

| Feature | block | KADID | TID | KonJND | CID22 | AIC-3 |
|---|---|--:|--:|--:|--:|--:|
| f324 | DKL achromatic Δmean | **98.094** | 25.314 | 0.881 | **1.230** | 0.683 |
| f325 | DKL achromatic Δstd | **33.146** | 18.794 | 0.822 | **1.396** | 0.667 |
| f328 | DKL RG ref_std | 59.160 | 37.639 | 48.131 | 59.795 | 23.951 |
| f329 | DKL RG dist_std | 65.723 | 39.760 | 47.986 | 59.006 | 24.053 |
| f334 | Weber band 0 gain | **2.265** | 0.621 | 0.021 | **0.046** | 0.027 |
| f335 | Weber band 1 gain | **4.519** | 1.208 | 0.037 | **0.081** | 0.053 |
| f342 | Minkowski β=3 luma pool | **93.974** | 36.943 | 10.684 | 10.392 | 9.734 |

KADID's per-pair feature scale is **80× CID22's on f324**, **50× on f334**.
The training-time MLP standardizer fits (mean, std) over the
training-set mixture; at inference on CID22 the features come in much
smaller than seen during training, so the standardized inputs land far
below typical training-set values. The MLP then extrapolates outside
its training distribution → produces unreliable predictions.

This is the FRIQUEE 2017 caveat (CLAUDE.md V_20b section): training-side
wins do not transfer to authentic / different-corpus distortions. Same
mechanism here — KADID + TID (analytic synthetic distortions: blur, noise,
JPEG, geometric) produce large per-pair feature deltas; CID22 (real
codec output, mostly near-imperceptible) produces small deltas.

The 19 features are physically meaningful and per-pair-signal-bearing,
but their **scale distribution is not corpus-stable**, breaking the
standardize-once-fit-all approach the trainer uses.

## Results — 5-group recipe (5 seeds)

| Bake | n | CID22 | KADID | TID | KonJND | AIC-3 |
|---|--:|---:|---:|---:|---:|---:|
| chunkc_s1 | — | 0.6011 | 0.9357 | 0.8902 | 0.8993 | 0.7586 |
| chunkc_s2 | — | 0.5428 | 0.9352 | 0.8907 | 0.9122 | 0.7491 |
| chunkc_s3 | — | 0.4336 | 0.9363 | 0.8912 | 0.9308 | 0.7402 |
| chunkc_s4 | — | 0.5218 | 0.9335 | 0.8884 | 0.9080 | 0.7403 |
| chunkc_s5 | — | 0.5289 | 0.9356 | 0.8903 | 0.9115 | 0.7181 |
| **seed median** | — | **0.5289** | **0.9356** | **0.8903** | **0.9115** | **0.7403** |
| ship_balanced (V_22-mix-LARGE+iwssim) | — | 0.8324 | 0.9677 | 0.9729 | 0.8927 | 0.7845 |
| ship_compression (V_24-per-sample-α s4) | — | 0.8641 | 0.9316 | 0.8893 | 0.8080 | 0.8183 |

| Corpus | seed median vs ship_compression | vs ship_balanced |
|---|---:|---:|
| CID22  | **−0.3352** | −0.3035 |
| KADID  | +0.0040 | −0.0321 |
| TID    | +0.0010 | −0.0826 |
| KonJND | +0.1035 | +0.0188 |
| AIC-3  | **−0.0780** | −0.0442 |

CID22 is **catastrophic** (−0.33). AIC-3 fails Compression-gate
(−0.078 > 0.0 threshold). KADID + TID + KonJND show mixed
results (KonJND lifts but at the cost of CID22 → exact "win KADID/TID
on training-side, lose CID22 on held-out" FRIQUEE pattern).

CID22 per-band SROCC for chunkc_s1 (CID22 SROCC=0.601 — best seed):

| Band | range | n | SROCC | (ship_compression for ref) |
|---|---|--:|---:|---:|
| B3 | [0.30, 0.40) | 57 | 0.0522 | — |
| B4 | [0.40, 0.50) | 266 | 0.1360 | — |
| B5 | [0.50, 0.60) | 615 | 0.1166 | — |
| B6 | [0.60, 0.70) | 836 | 0.0960 | — |
| B7 | [0.70, 0.80) | 1092 | 0.1965 | — |
| B8 | [0.80, 0.90) | 1382 | 0.2865 | — |
| B9 | [0.90, 1.00] | 43 | 0.1077 | — |

Even at best, per-band SROCC is 0.05-0.29. Reference: ship_compression
hits 0.4-0.8 per-band on the same bands.

## Results — anchor-only recipe (3 seeds)

Hypothesis: if anchor-only succeeds CID22 (~0.85), the 5-group failure
was the zero-fill. If anchor-only ALSO fails CID22, it's feature-design.

| Bake | n | CID22 | KADID | TID | KonJND | AIC-3 |
|---|--:|---:|---:|---:|---:|---:|
| anchor_s1 | — | **0.0301** | 0.9451 | 0.8910 | 0.9781 | 0.5804 |
| anchor_s2 | — | **0.0323** | 0.9454 | 0.8908 | 0.9762 | 0.6013 |
| anchor_s3 | — | **0.1369** | 0.9449 | 0.8909 | 0.9758 | 0.5388 |
| **seed median** | — | **0.0323** | **0.9451** | **0.8909** | **0.9762** | **0.5804** |

Anchor-only CID22 is **WORSE than 5-group** (0.03 vs 0.53). The
explanation: training on only anchor corpora (14k pairs total) makes
the cross-corpus distribution shift catastrophic — CID22's much
smaller per-pair feature magnitudes are now 100% out-of-distribution
for the MLP, which has only seen KADID-scale and TID-scale magnitudes
during training.

This decisively isolates the root cause to **feature-design /
cross-corpus distribution shift**, NOT zero-fill. Adding more
training data with zero-filled features would not fix this.

## Why the prior agent's zero-fill diagnosis was incomplete

The prior agent's `b94314f4` falsification verdict
(`benchmarks/v25_v2_extfeat_perpair_seed3_verdict_2026-05-18.md`)
attributed CID22 0.5919 to "training-data zero-fill on safesyn +
cvvdp_LARGE". That explanation IS partially valid (96% of training
data with f324..f342=0 reduces per-pair-feature gradient by ~20×),
but it's not the complete story:

1. The prior bake also lacked `--per-sample-alpha-head` (architecture
   mismatch with the Compression ship), which would have explained
   ~0.02 SROCC degradation independent of feature work.
2. This re-attempt fixed BOTH issues and still got worse CID22 (0.53
   vs 0.59) on the 5-group recipe. So the zero-fill explanation
   wasn't accounting for the bulk of the CID22 collapse.
3. The anchor-only experiment shows that REMOVING the zero-filled
   corpora produces an EVEN WORSE CID22 (0.03), proving zero-fill
   is not the proximate cause.

The proximate cause is the cross-corpus distribution shift of the 19
features themselves. KADID's analytic distortions produce very different
per-pair feature distributions than CID22's codec distortions.

## What this rules out

- **EX-4 Chunk C as a single-bake compression-trail lever.** The
  features cannot be added to the production training mix and
  produce a CID22-lifting bake without major preprocessing changes.
- **The "fix the zero-fill and re-train" path.** Even with full
  corpus coverage on safesyn + cvvdp_LARGE, the cross-corpus
  distribution shift would persist (those corpora have different
  distributions from CID22 too).

## What's still open

The 19 features DO carry per-pair signal and DO match published CVVDP
shape semantics. They might be useful with:

1. **Per-corpus standardizer.** Train MLP with per-group standardize
   layer (each corpus gets its own mean/std on f324..f342) instead
   of one global standardizer. This is a trainer architecture change.
2. **Distribution-matching loss.** Add a distribution-matching term
   that penalizes cross-corpus feature-statistics mismatch — e.g.,
   feature-wise CORAL alignment of training feature distribution to
   a CID22-like target.
3. **Feature-transform on f324..f342.** Apply signed_log1p or other
   monotone transform to compress the dynamic range so KADID's 80×
   advantage shrinks to ~5×. This is what V_20 input-shaping does
   for the base features.
4. **Per-codec-family training**: don't expect a single MLP to
   generalize across analytic + codec distortions. Train multiple
   bakes (synthetic-only, codec-only, hybrid) and pick at runtime.

None of these are single-bake compression-trail levers. They're
multi-week engineering projects with their own falsification risks.

## Verdict per § A.10 trail gates

### Balanced trail gate

- CID22: A>>B decisive **−0.30 against** (B>>A decisive).
- KonJND: +0.0188 favorable, AIC-3: −0.0442 unfavorable.
- **FAILS Balanced gate**: any decisive B>>A on a corpus is a ship blocker.

### Compression trail gate

- CID22 + AIC-3: BOTH lose decisively (CID22 −0.33, AIC-3 −0.08).
- **FAILS Compression gate**: requires A>>B on ≥1 of {CID22, AIC-3}
  decisively; we lose both.

**Neither trail gate passes.** No shipping action; bake stays
in `/mnt/v/zen/zensim-eval/exp_chunkc_perpair_2026-05-18/` for
forensic archival.

## Bake artifacts (forensic, not for production)

```
/mnt/v/zen/zensim-eval/exp_chunkc_perpair_2026-05-18/
  chunkc_s1_h128.bin          5-group seed 1 — CID22 0.6011
  chunkc_s2_h128.bin          5-group seed 2 — CID22 0.5428
  chunkc_s3_h128.bin          5-group seed 3 — CID22 0.4336
  chunkc_s4_h128.bin          5-group seed 4 — CID22 0.5218
  chunkc_s5_h128.bin          5-group seed 5 — CID22 0.5289
  chunkc_anchor_s1_h128.bin   anchor-only seed 1 — CID22 0.0301
  chunkc_anchor_s2_h128.bin   anchor-only seed 2 — CID22 0.0323
  chunkc_anchor_s3_h128.bin   anchor-only seed 3 — CID22 0.1369
  verdicts/                   bake_verdict markdown per bake + ship
```

## Methodology & reproducibility

- Workspace: `~/work/zen/zensim--ex4-extfeat/`
- Branch: created new change on `main@origin` (`e8224062`).
- Trainer binary: `target/release/zensim_mlp_train` rebuilt from main
  + EX-4 feature source files restored from commit `b94314f4` (commits
  `66fbebf5` + `280364e9`).
- Training input parquets:
  `/mnt/v/zen/zensim-training/2026-05-18-extfeat/*_extfeat_343.parquet`
- Validation features: 343col parquets at the same dir, row-aligned
  with 372col parquets at `/mnt/v/zen/zensim-training/2026-05-15-full-features/`
  (verified by ref_basename + human_score byte-equality).
- Validation against ship bakes: ship bakes (372 inputs) scored on
  372col features-root; new bakes (343 inputs) scored on a temp
  features-root with 343col files renamed to the 372col schema
  (`/tmp/exp_chunkc_perpair_features_root/`).
- Eval tool: `bake_verdict` — aggregate Mohammadi panel per corpus.
- Bootstrap (§ A.9 1000-bootstrap) NOT run: SROCC delta magnitude
  (−0.33) is far outside any plausible bootstrap CI; bootstrap rigor
  doesn't change the verdict.

## CLAUDE.md learning entry

Append to "V_20 input-shaping + multi-bake runtime — learnings":

> ### EX-4 Chunk C 19 CVVDP-shape per-pair features — falsified twice
>
> The 19 EX-4 Chunk C features (DKL Δstats, Weber band gains, CSF-weighted
> band energies, mutual-masking residuals, Minkowski β=3 luma pool) at
> `zensim/src/cvvdp_features.rs` carry strong per-pair signal (13/19
> features show within-ref ratio > 0.3) and match the published CVVDP
> shape semantics. They were falsified twice:
>
> 1. `b94314f4` (2026-05-18 prior session): 5-group training with
>    `safesyn` + `cvvdp_iwssim_large` zero-filled. CID22 0.5919.
>    Diagnosed by prior agent as zero-fill problem.
>
> 2. `exp_chunkc_perpair_2026-05-18` (this session): Re-trained with
>    `--per-sample-alpha-head` matching V_24 Compression ship; tested
>    both 5-group recipe (CID22 median 0.5289) AND anchor-only recipe
>    that excludes zero-filled corpora (CID22 median 0.0323).
>    Anchor-only is WORSE than 5-group → root cause is NOT zero-fill.
>
> Root cause: cross-corpus distribution shift on f324..f342. KADID's
> per-pair feature scale is 80× CID22's on f324 (DKL achromatic
> Δmean), 50× on f334 (Weber band 0 gain), etc. KADID + TID
> (analytic synthetic distortions) and CID22 (real codec output)
> produce different per-pair feature magnitudes. Single global
> standardizer cannot generalize across this gap.
>
> Falsification doc: `benchmarks/exp_chunkc_perpair_falsified_2026-05-18.md`.
>
> Don't retry without: (a) per-group standardizer, (b)
> distribution-matching loss, (c) feature-transforms compressing the
> KADID/TID dynamic range, OR (d) per-codec-family training. Single-bake
> compression-trail lift via these features is not achievable.

## Outcome

Documented falsification. No shipping action. The FEATURE-AUDIT's
"Pick 1" lever is now closed. Next available compression-trail
levers per `project_feature_expansion_audit`:

- **Pick 2**: external metric scores (cvvdp/ssim2/iwssim) as MLP
  inputs. Medium-large expected lift. ~2 hours wall.
- **Pick 3**: true percentile pool features. Small-medium lift. ~6
  hours wall.

Both require new infrastructure and have not been re-evaluated since
this falsification. Recommend dispatching Pick 2 next.
