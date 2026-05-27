# SESSION-RESUME — read this first after every compact

**Last updated:** 2026-05-27 (axiom-clean QAT-native ship candidate staged;
#33 Approach-B validated; #35 resolved)

## 2026-05-27 session — what changed (read this, then the older state below)

**SHIPPED (commit 1fd645a7): `ZensimProfile::A` is now v47-strict-QAT-native**,
replacing the broken V39 bake (user-approved replace-at-Profile::A, 2026-05-27).
`zensim/weights/v47_strict_qat_native_2026-05-27.bin` is `include_bytes!`'d via
`mlp_bake_a_v47_qat()`; V39 bytes stay on disk (still back `PreviewV0_4`). The
metric_invariants `v39_known_limit_violations` test flipped → replaced by the
positive A gate (`a_v47_is_bounded_above_and_self_identity_maximal`,
`a_v47_is_degradation_monotone`). **The #1 non-speed goal is DONE**: Profile::A
is bounded-above + self-identity-maximal + degradation-monotone on all content.
Full workspace green. **Next per user**: (1) retune jxl-encoder's zensim loop
for the new Profile::A; (2) iterate on #33 as a diffmap candidate vs #32; (3) try
CVVDP-target training instead of ssim2 (note the V41 scalar-CVVDP dead-end —
diffmap-target may differ). The candidate that shipped:

- **`v47_strict_qat_native_2026-05-27.bin`** (27 KB, sha256 `d0ef7a30…`),
  produced by ONE `zensim_mlp_train --manifest
  zensim/weights/manifests/v47_strict_qat.toml` pass (QAT-native f16+zerobias,
  no Python post-step — the "rust workflow handles packing" deliverable).
- Masked-monotone-by-construction (0 inversions, identity=max=97.69), fixes
  the blur>identity AND identity=0 V39 defects.
- **Best codec dial measured**: q-sweep monotonicity 94.33% / 0.33% tied
  (V39 67.7% / 53.6%), clean monotone median q5→q95 (1.40→88.50).
- CID22 0.8657 (best of the v47 axiom-clean candidates). Costs vs V39 are
  rank-SROCC on KADID/TID (0.79 vs 0.93 — non-compression analytic
  distortions, integrity guards) + KonJND (f16 removes PJND precision).
- **Methodology doc** (the required pre-ship artifact, all 8 sub-points):
  `benchmarks/v0_qat_native_methodology_2026-05-27.md`. q-sweep:
  `benchmarks/qsweep_qat_native_vs_v39_2026-05-27.md`.

**⇒ USER-GATED ship-form decision (#32):** replace V39 at `Profile::A` (bake
rotation, permitted by shipping policy; recommended — V39 is broken) vs add a
sibling profile (new public enum variant, needs API approval) vs hold. The
swap is a one-line `include_bytes!` in `zensim/src/profile.rs` once decided.
The non-QAT `recal_negtail` (30 KB, KonJND 0.485) is the HF-priority alt.

**QAT/packing is the standard recipe path** (`qat_fine_tune_epochs` kept CLI-opt-in,
default 0 — the KonJND trade shouldn't be forced on HF bakes; the ship recipe
opts in). `out_dtype` CLI default stays f32; the canonical recipe sets f16.

**#33 (localized-defect detection) — Approach-B VALIDATED.** Perceptual
tile-min failed on screen content; a content-robust signal (per-tile error
RELATIVE to the source's own local activity, 3 channels mean/maxpix/chroma,
multi-scale 64∧16) catches **op100 (full-strength) 92.5% photo / 81.2%
screen** (~97% / ~85.5% excluding two corpus no-op families). Beats tile-min
(37%/24%) and is content-robust. Spike:
`scripts/v_next/structural_signature_spike.py`; findings:
`benchmarks/approach_b_structural_signal_spike_2026-05-27.md`. Next (non-gated):
Rust port → wire into zensim-regress, THEN propose the `ZensimLocal` public API
(gated). Surfaced 2 corpus generator bugs: **codec-corpus#9**
(chroma_boundary + block_repeat_neighbor are no-ops — change ~0 pixels).

**#35 RESOLVED:** the konjnd-agg "2× gradient bug" was a malformed FD test
(f32 forward + eps=1e-6 swamped by f32 noise + pure-relative gate on near-zero
gradients). Gradients are correct; ships unaffected. Fixed with eps=1e-2 +
atol+rtol gradcheck. New `backprop_heads_dl_dh` train-core test isolates +
confirms the head gradient.

## Current state (pre-2026-05-27 — V39 is still the shipped Profile::A)

**Shipped champion:** `PreviewV0_3` = `zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`
- Architecture: 372→128→64→heads (2-layer, per-sample α, tanh pin, PCHIP spline)
- Recipe: V32 ranking (normalized [0,1] group targets, hybrid MSE 0.6 +
  RankNet 0.6) + multi-band anchor at weight 0.01 for post-training
  spline calibration. seed=17.
- Beats prior V0_3 (v_tuner_v11) on **5 of 6** held-out corpora
  (full panel) AND the G1 dial:
  CID22 0.8793 / KADIK 0.9251 / TID 0.9317 / KonJND 0.4197 / AIC-3 0.8023;
  G1 dial 1.00 (prior 0.69). **AIC-4 (n=300) is the exception** —
  V0_3 wins SROCC 0.9284 vs V39 0.9051 — and it's STATISTICALLY REAL
  (paired bootstrap p=0.001, CI [+0.009,+0.040]), not noise. So
  "universally better" is FALSE. AIC-4 is HOLDOUT-ONLY — chasing it by
  recipe search = holdout-fishing (forbidden). V39 wins the other 5.
- **Goal status:** PASSES G1 + G7; soft-passes G8/G9; **FAILS G5**
  (KonJND HF-rank 0.42 < 0.70). G5 is now a CHARACTERIZED Pareto limit,
  not a data gap: the konjnd-aggregation lever (wired+gradient-verified
  this session) CAN reach KonJND 0.85 but craters CID22/KADIK/TID/AIC-3;
  a 2-bake regime-routed ensemble also fails (CID22's near-lossless tail
  overlaps the KonJND regime in feature space). Closing G5 needs a
  better HF feature REPRESENTATION (not more data — KonJND already in
  training; not CSF — falsified). Multi-session research.
- Bake bytes carry both `tanh_output_head` + `output_calibration_spline`.

**⚠ The old "0.885 AIC-3 ceiling / CVVDP gap needs CSF features" claim is
FALSIFIED** (see Findings below). The AIC-3 "gap" was a measurement
artifact; per-ref we already beat raw CVVDP. The real lever is
cross-ref absolute-scale calibration (the dial), not features.

## DECISION (user, 2026-05-26): ACCEPT V39 + phone bake; AIC-4/G5 are known-limits

The user accepted the shipped state as the deliverable: **V39** (PreviewV0_3,
beats V0_3 on 5/6 corpora + better dial/Z-RMSE) + **zensim-b-phone** (CVVDP
iPhone, shipped). **AIC-4 (V0_3's lone win) and G5 (KonJND HF) are accepted
KNOWN-LIMITS — do NOT re-grind them.** They are not closable by recipe or
architecture (v42/v43/v44 falsified); the only unblock is new JPEG-AI +
near-lossless training data (root cause below). Revisit ONLY if/when that
data is acquired.

## ROOT CAUSE unifying the 2 unmet goals (2026-05-26) — it's TRAINING DATA

The two unsolved deliverables share ONE cause: **the training distribution
doesn't cover the regimes they test.** Training = conventional codecs
(mozjpeg/jpegli/webp/JPEG/BPG) + synthetic analytic distortions
(kadid/tid). NOT covered: (a) **neural-codec artifacts (JPEG-AI = AIC-4)**
→ V39-vs-V0_3 on AIC-4 is a generalization LOTTERY we can't steer without
JPEG-AI training data (and AIC-4 is HOLDOUT-ONLY, so steering = fishing);
(b) **dense near-lossless JND (KonJND-HF = G5)** → underrepresented, so the
HF specialist can't rank it without cratering mid-fidelity.
Falsified in-session that it's NOT closable by recipe/architecture alone:
single-MLP (v42 Pareto), 2-bake ensemble (v43 overlap), {V39,V0_3} MoE
(v44 anti-corr + AIC-4 not route-separable). **The real unblock for BOTH
is acquiring a JPEG-AI + near-lossless training corpus** (the goals doc's
own flagged-unstarted work). Without it: lottery/research. Decision for
the user: acquire that corpus, accept V39(5/6)+phone-bake, or explicitly
relax the AIC-4 holdout rule.

## Monotone-by-construction `A` retrain — BLOCKED on 2 obstacles (2026-05-26 PM)

Goal: make shipped `A` itself correct-by-construction (≤100 + monotone +
self-identity, **negative allowed below** per user — "worse than a simple
lq encode → negative") while retaining V39's Mohammadi panel.

**DONE:** the **bounds half is shipped** — `apply_output_calibration_spline`
now clamps **≤100 (upper only)**, keeping the negative lower tail
(commit `24f93462`). So `A`/V39 no longer exceeds 100; `>100` bug fixed in
production. The remaining gap is **monotonicity** (the OOD inversion).

**BLOCKER 1 — monotone training collapses.** `--monotone-cbc` added
(`fa5c699d`, flag OFF by default). FIVE attempts to train a non-neg-weight
encoder all collapsed after a healthy epoch 0:
`v45` hard clamp; `v45b` clamp+lr1e-4; `v45c` clamp+momentum-reset;
`v45d` penalty λ10 + clamp (fully dead); penalty-only untested-for-monotonicity.
Unconstrained baseline (`v45base`) is STABLE (train val 0.92). → the hard
sign-clamp drives dead weights. **Fix = softplus reparam** `w=±softplus(θ)`
(weights→0 without dying, gradients always flow) — NOT yet implemented
(invasive: forward call sites + θ shadow vectors, OR keep w=w_eff and update
θ in `do_adam_step` only). Scripts: `scripts/v_next/run_v45*.sh`.

**BLOCKER 2 — RESOLVED: recipe recovered from prior-session transcript.**
My CID22 0.295 reconstruction used the WRONG recipe (reconstructed from
the g5 konjnd-agg script). The REAL zensim-a/V39-lineage command (from
session `133ab28d`, `scripts/v_next/repro_zensim_a_recipe_2026-05-26.sh`):
- **`--target-column mix_cv40_iw60`** (ONE consistent cvvdp×iwssim metric
  across all groups) — NOT `human_score` (per-group-inconsistent:
  safesyn=ssim2 / cid22=MCOS / kadid=DMOS → the overfit cause).
- **Only 2 `--group` inputs**: `safesyn:1.0:0.0` + `cid22_train:0.5:0.0`
  (cid22_train.parquet, NOT _norm) — NOT 5 explicit human-MOS groups.
- konjnd via `--konjnd-aggregation-weight 0.05` (not a --group), plus
  `--cross-codec-eq-*` + `--dynamic-range-floor-weight 0.3`.
- `--lr 5.66e-3 --tanh-output-head-scale 20.0 --anchor-loss-weight 0.5
  --anchor-target-score 60 --epochs 300 --hidden 128 --per-sample-alpha-head`.
- tuner_v11/V0_3 = `--mse 1.0 --ranknet 0.0 --seed 1` (CID22 0.860).
  V32/V39 = `--mse 0.6 --ranknet 0.6 --seed 17` (+ spline injection via
  `scripts/inject_spline.sh`) → CID22 0.8879/0.8793.
Reproduction running 2026-05-26; expect CID22 ~0.86 (seed-1 tuner_v11).
→ the monotone retrain (Blocker 1) must use THIS recipe, not human_score.

**HAZARD:** a concurrent agent edited `mlp_train/mod.rs` (panel-subcommand
work), clobbering my const + breaking main once (fixed `e507cf84`). Claim
the `.workongoing` marker before editing that file.

**Path:** (1) reproduce V32 0.88 unconstrained; (2) implement softplus
reparam; (3) train monotone, measure panel + invariant gate; (4) promote
to `A` only if panel retained AND gate passes. `LinearBounded` remains the
fully-monotone guaranteed alternate now (CID22 ~0.86).

## SHIPPED (2026-05-26): `ZensimProfile::LinearBounded` — correct-by-construction metric

First correct-by-construction zensim metric (commit `caf82c48`). V0_2's
non-negative weights × non-negative dissimilarity features → distance
`d ≥ 0` (`=0` iff identical), mapped by the bounded squash
`100·exp(−(a/100)·d^b)`. **Bounded `[0,100]`, self-identity = 100 = unique
max, monotone non-increasing in every error feature — by construction, on
the ENTIRE domain** (incl. 100σ-off-manifold synthetic content where V39
inverts). SROCC identical to `PreviewV0_2` (monotone transform of the same
`d`). New `bounded_squash` disposition flag (ProfileParams + ZensimConfig,
default false, ignored on MLP profiles). Invariant gate
`tests/metric_invariants.rs` verifies it + tracks A's violations. Use as
the guaranteed-safe metric / OOD fallback. The expressive member
(partial-monotone MLP per the redesign doc §5.1–5.4) is the multi-session
follow-on; swapping shipped champion `A` to a CbC MLP is a separate call.
Doc: `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`.

> ⚠ `zensim-validate` does NOT compile at HEAD (pre-existing, not from the
> CbC work): `main.rs` has a stale `mod loss_norm_in_norm;` (file moved to
> `mlp_train/` in `226aab3`) + `crate::adam_simd`/`panel` main-vs-lib path
> confusion. `bake_verdict`/trainer won't build until fixed.

## Finding (2026-05-26): V39 violates metric invariants — process flaw, not corner case

V39 (`ZensimProfile::A`) returns scores **>100** and ranks **heavier
degradation as higher quality** on ALL synthetic content (mandelbrot,
checker, noise, smooth color_blocks). Localized: linear V0_2 (shared
feature extraction, no MLP) stays monotone+bounded; the **trained MLP
itself inverts off its natural-photo training manifold**, and
`extrapolate_score:true` (metric.rs:2113) unbounds it. The fundamental
flaw is **process**: bake acceptance was optimized+gated solely on SROCC
(rank-only, scale-invariant) over narrow natural-photo MOS corpora, with
NO invariant gate (bounds / self-identity / degradation-monotonicity).
SROCC is mathematically blind to all three; the one invariant test
(`score_sanity_checks`) was outside the acceptance loop and silently red
since V39 shipped. **Fix = an invariant gate (`tests/metric_invariants.rs`
+ bake_verdict), NOT a clamp** (clamp hides bounds but leaves the
inversion). Full writeup + evidence table:
`benchmarks/ROOT_CAUSE_v39_invariant_violations_2026-05-26.md`. Open
decision: implement the gate (red until MLP fixed) vs dig the MLP
inversion mechanism vs leave documented.

## Findings & falsifications (2026-05-25 evening) — READ BEFORE RE-TRYING

1. **Broken dial from SROCC-chasing (the big one).** Bakes trained
   without an output calibration spline collapse the dial to a
   near-constant (~65, G1=0.00) even at high SROCC — UNUSABLE as a
   codec target. The shipped V0_3 had a spline; V13–V36 didn't.
   Fix: the PCHIP spline. Because SROCC is **rank-invariant under a
   monotone spline**, a well-ranking compressed bake + a multi-band-
   anchor spline = good rank AND a working dial (that's V39).
   Detail: `benchmarks/v5_vs_v03_comparison_2026-05-25.md`.
2. **Anchor weight causes rank collapse.** V37 used
   `--anchor-loss-weight 0.5` with per-row anchor targets → the anchor
   MSE fought RankNet and held-out CID22 SROCC collapsed 0.88→0.55.
   **The anchor is for SPLINE FITTING ONLY — use weight ≤ 0.01.** Any
   meaningful anchor weight destroys rank.
3. **Target-scale mismatch → training divergence.** 5-group MSE-only
   training diverges when group `human_score` scales differ
   (cid22_train raw MCOS [3-94], konjnd [-66,96], others [0,1]).
   Normalize all targets to a common scale.
4. **dynamic-range-floor overshoots** (V40): pushed output >100 →
   clamps to saturated top. Not a clean G1 fix on its own.
5. **CVVDP-emulator is a dead end** (V41): training toward CVVDP
   scores gives WORSE human-MOS (CID22 0.66). Emulating CVVDP's
   output ≠ having its accuracy.
6. **AIC-3 "0.80 vs 0.96 CVVDP gap" was an artifact.** CVVDP's 0.96
   was a 5-image subset; on the full 600-pair/10-ref set CVVDP is
   0.79 pooled / 0.93 per-ref. Our bake scores 0.9475 per-ref —
   ABOVE raw CVVDP (0.9342). Mixing real CVVDP in as an input
   feature: +0.004 (noise). **NOT feature-limited.** Detail:
   `benchmarks/aic3_cvvdp_feature_spike_2026-05-25.md`.
7. **DATA BUG: kadid/tid iwssim + ssim2_gpu corrupt.** `iwssim` was a
   100% copy of `human_score` (target leak); `ssim2_gpu` was joined
   ref-vs-ref (~0 corr). Shipped bakes SAFE (used `human_score`);
   multi-target `iwssim` experiments on kadid/tid INVALID. Fixed
   siblings written (`*_fixed_2026-05-25.parquet`), pending promotion.
   Detail: `benchmarks/DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`.

## Read order on resume

1. This doc (`SESSION-RESUME.md`) — current state, ~2 min
2. `CLAUDE.md` — methodology + workflow + gotchas
3. `docs/CODEC_TARGET_GOALS.md` — the goal set (G1-G11)
4. `benchmarks/v5_vs_v03_comparison_2026-05-25.md` — V39 lineage + findings
5. `RESEARCH.md` — corpus map + workflow recipes

## What shipped (29 commits, 2026-05-25)

### Speed
- 9× per-epoch training speedup (SIMD encoder + parallel validation)
- f32 SIMD encoder ready (`simd_encoder_f32.rs`, 1.36× over f64)

### Architecture
- 2-layer MLP (372→128→64→heads) + skip connection
- Full forward/backward with exact h1_pre caching
- 2-layer bake format (3 BakeLayer entries in ZNPR v3)

### Validation
- Mohammadi 2025 exact-methodology eval (`scripts/mohammadi_eval.py`)
- Multi-stat panel (SROCC+PLCC+PWRC), `--val-policy goals`
- NaN safety gate, per-sample Z-RMSE, output spline fitting
- DisplayProfile struct + iPhone 14 Tier 1 calibration

### Tests: 111+ across 3 crates

## What to do next (priority order)

1. **Cross-ref absolute-scale calibration** — the real lever for the
   AIC-3 pooled residual (NOT CSF features — that's falsified, see
   Finding 6). The dial is near-constant on near-imperceptible
   distortions; per-ref/absolute-scale anchoring is the open problem.
2. **Promote the kadid/tid data-bug fix** — review `*_fixed_2026-05-25.parquet`,
   `mv` over originals, rebuild `_MANIFEST.json`, re-sync R2, then re-run
   any multi-target `iwssim` bake (those used the leaked label).
3. **Smoother spline** — V39's spline is coarse (3 knots, raw p5=-89.7
   pre-clamp). More anchor bands surviving the strict-mono filter → a
   smoother dial. The dynamic-range-floor lever overshot (Finding 4);
   needs tuning, not abandonment.
4. **G5 HF representation research** — the only path to clearing G5
   (KonJND≥0.70 with the rest intact): a better HF feature
   representation so an HF-specialist can rank CID22's near-lossless
   tail too (the ensemble blocker). Single-MLP + 2-bake-ensemble both
   falsified (`v42_*`, `v43_*`). Not data, not CSF — representation.
5. **Wire cross-codec-eq + pjnd-passthrough aux losses for 2-layer** —
   still gated off for multi-layer/skip (konjnd-aggregation IS now
   wired; same arch_backward pattern applies to the other two). Blocks
   the G4 cross-codec goal at 2-layer.

DONE since (don't redo): zensim-b-phone CVVDP bake at
modern_oled_phone_indoor (≈110 ppd, SROCC 0.9342 phone-CVVDP tracking,
working dial, wired to DisplayTarget::Phone); zen-metrics `--display-model`
flag; konjnd-aggregation 2-layer wiring + gradient test; kadid/tid
data-bug fix (`*_fixed_2026-05-25.parquet`, pending promotion); G11 doc
physics correction (higher PPD → LESS visible, was backwards).

DONE this session (don't redo): σ-weighted MSE infra, modular refactor
(mlp_train → arch.rs/goals.rs/utils.rs), f32 encoder, DisplayTarget +
iPhone Tier-1 profile, auto-bake_verdict-after-train, DS-AUC + goals
scorecard in bake_verdict.

## Key files

| File | What |
|------|------|
| `zensim-train-core/src/simd_encoder.rs` | SIMD f64 encoder (production) |
| `zensim-train-core/src/simd_encoder_f32.rs` | SIMD f32 encoder (ready) |
| `zensim-train-core/src/per_sample_alpha_head.rs` | Head forward/backward + bake |
| `zensim-validate/src/mlp_train/` | trainer (mod.rs + arch.rs + goals.rs + utils.rs) |
| `zensim-validate/src/bin/bake_verdict.rs` | eval: Mohammadi panel + DS-AUC + goals scorecard; auto-runs after every train |
| `zensim-validate/src/panel.rs` | LightPanel + ValAggregate + stats |
| `zensim/src/display.rs` | DisplayProfile + DisplayTarget enum (desktop/phone/tv) |
| `zensim/src/profile.rs` | `mlp_bake_preview_v0_3` → V39 (shipped) |
| `docs/CODEC_TARGET_GOALS.md` | Goal set (G1-G11) |
