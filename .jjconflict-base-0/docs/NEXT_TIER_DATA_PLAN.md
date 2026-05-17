# Next-Tier Training Data Plan

Concrete data requirements for the next zensim tier. Read-only research
doc; produced 2026-05-04. Citations use absolute paths so every claim is
verifiable against code, not memory.

> **Important context.** The codebase already has substantial MLP
> infrastructure on experimental jj branches that has not yet landed on
> `main@origin`. V0_4 ships an MLP-shaped placeholder with V0_2-equivalent
> weights; V0_5/V0_6 retrained MLPs exist with measured holdout SROCC; a
> V0_7 plan with a low-q fill is committed. This doc reflects that
> reality. The "next tier" is therefore *condition-aware* training data
> (squintly), not the missing MLP runtime — that runtime exists.

---

## 1. Current zensim status

### 1.1 Model architecture

| Layer | What `main@origin` ships today | What experimental branches add |
|---|---|---|
| Public profiles | `PreviewV0_1`, `PreviewV0_2` | `PreviewV0_4` (MLP-dispatched, V0_2-equivalent placeholder) |
| Scoring path | 228-feature dot product → `100 - A·d^B` | 228-feature MLP forward → same mapping |
| MLP runtime | absent | `zensim::mlp` module (vendored from zenpicker v0.1.0, re-licensed MIT/Apache-2.0) |
| Binary format | n/a | `ZNPK` v1 — header (32 B) + scaler + per-layer (in_dim, out_dim, activation, weight_dtype f32/f16/i8, weights, scales, biases) |
| Activations | n/a | Identity, ReLU, LeakyReLU(α=0.01) |

Sources:
- `/home/lilith/work/zen/zensim/zensim/src/profile.rs:14-22, 92-108, 352+`
- `/home/lilith/work/zen/zensim/zensim/src/metric.rs:1642, 300-339, 1649`
- `/home/lilith/work/zen/zensim/CHANGELOG.md:12`
- jj `ykzptwqq` (`feat(zensim): vendor MLP runtime + V0_4 profile with V0_2-equivalent placeholder`)

| Item | Value | Source |
|---|---|---|
| Crate version on main | 0.2.7 | `CHANGELOG.md:12` |
| Feature count (linear path) | 228 (4 scales × 3 channels × 19 features) | `metric.rs:1642`, `lib.rs:113` |
| Linear params | 228 f64 + (A, B) | `profile.rs:53-66` |
| Non-zero linear weights | 127 of 228 | `CLAUDE.md:73` |
| Linear weights file size | 228 × 8 = 1 824 B | embedded `WEIGHTS_PREVIEW_V0_2` |
| Linear training algos | Nelder-Mead (default), CMA-ES, coord descent, RankNet | `zensim-validate/src/main.rs:1273, 2536, 2689` |
| MLP forward kernels | f32 / f16 / i8 SAXPY matmul | jj `ykzptwqq`, `mlp/inference.rs` |
| V0_4 placeholder shape | 228 → 1, single linear layer | jj `ykzptwqq`, `profile.rs` (V0_4 variant) |

### 1.2 SROCC against human IQA datasets (raw distance)

Numbers on the published 0.2.7 build (`README.md:35-39`):

| Dataset | Pairs | SROCC | KROCC |
|---|---:|---:|---:|
| CID22 | 4 292 | 0.8676 | 0.6786 |
| TID2013 | 3 000 | 0.8427 | 0.6657 |
| KADID-10k | 10 125 | 0.8192 | 0.6139 |

Numbers from `benchmarks/4metric_overnight_FINAL_2026-05-01.md` (jj
`nmmxmlmw`) on the experimental V0_5/V0_6 MLPs (n=1 500 per dataset
sample, `dataset_metric_baseline` per-pair CSVs):

| metric | CID22 | KADID-10k | TID2013 |
|---|---:|---:|---:|
| V0_4-smooth (RankNet+magnitude-matching) | 0.8910 | 0.8400 | 0.8336 |
| V0_5 (synthetic SSIM2 proxy, 80/20 source-disjoint) | **0.8934** | **0.8505** | **0.8492** |
| V0_4-smooth + KonJND-1k in train (NEGATIVE result) | 0.8221 | 0.7441 | 0.7566 |
| V0_6 (zenanalyze dct_hf features appended) | 0.8935 | 0.8496 | 0.8416 |
| reference SSIMULACRA 2 (same pairs) | 0.8928 | 0.8155 | 0.8525 |
| reference Butteraugli 3-norm | 0.7670 | 0.5864 | 0.6827 |

Caveats from that report (verbatim §1):
- 0 of 49 CID22 validation images leak into V0_4+ training; safe-synthetic
  filter lands a clean MCOS holdout.
- 196 of 215 CID22 *training* images appear as cropped tiles in synthetic
  training (5.2 % of pairs). V0_5/V0_6 inherit some SSIM2-content-specific
  behaviour on that subset.
- KADID-10k and TID2013 have **no source overlap** with synthetic or CID22
  → those SROCC numbers are clean independent generalization tests.
- V0_2's published 0.8676 CID22 number is **inflated** by an 8-image /
  475-pair leak that V0_4+ removed.

V0_5 is honestly described in that report as "a fast-SSIM2 surrogate, not
an independent perceptual model" — useful framing.

### 1.3 V0_7 plan (committed but not run)

Per the latest experimental commit (`nmmxmlmw`, 2026-05-01):

> V0_7 plan = V0_6 dct_hf retrained on the post-fill extended dataset
> (218k base + ~140k zenjpeg-420-e1 fills covering SSIM2 0-90), with
> --mlp-low-band-oversample 0.5. Includes a no-bias control to isolate
> the sampler-bias change from the data fill.

Per-band V0_6 baseline (218k):

| band | V0_6 SROCC | training share |
|---|---:|---:|
| ≤ 0     | 0.96 | 6.0 % |
| 0-25    | 0.91 | 7.4 % |
| 25-40   | 0.90 | 7.8 % |
| 40-60   | 0.96 | 17.4 % |
| 60-75   | 0.97 | 20.9 % |
| 75-90   | 0.98 | 28.0 % |
| ≥ 90    | 0.86 | 12.4 % |

Expected V0_7 numbers (after fill + sampler bias): 25-40 → ~0.95, 40-60 →
~0.97, 0-25 → ~0.94, 75-90 → ~0.97 (slight regression, "acceptable for
web target").

### 1.4 What's already done — what isn't

| Gap | Status |
|---|---|
| MLP forward (f32/f16/i8) | DONE (jj `ykzptwqq`) |
| ZNPK v1 binary format | DONE |
| V0_4 dispatch in `Zensim::compute*` | DONE |
| V0_4-smooth, V0_5, V0_6 weights | TRAINED, not on main yet |
| V0_7 (post-fill) | infrastructure ready, run not done |
| **Condition-aware inputs (dpr, viewing distance, …)** | **NOT STARTED** |
| Squintly TSV ingest in zensim-validate | not started |
| Per-content-class conditioning at metric layer | not started |

The condition-aware tier — what squintly was built to feed — is the
genuinely new work. Everything below sizes that.

---

## 2. Gap to the next tier

### 2.1 Viewing-condition gap (squintly's argument)

`/home/lilith/work/squintly/README.md:6-10`:

> "zensim plateaus around SROCC 0.82 on these — the residual is
> dominated by *how* an image is being viewed: device pixel ratio,
> intrinsic-to-device ratio, viewing distance, ambient light, gamut.
> Squintly collects pairwise judgments **with those conditions
> recorded**, so zensim can learn to condition on them."

The headline number is now ~0.85-0.89 on V0_5/V0_6 (per §1.2), so the
pure plateau argument is less sharp than it was. But:
- KADID/TID independent-holdout best is ~0.85 — still flat.
- V0_5 hits 0.89 on CID22 but tracks SSIMULACRA2 (its training target)
  almost exactly, indicating limited *headroom from the same 218k pairs*.
- Squintly's data is the only live source of *condition-conditioned*
  labels not yet ingested anywhere.

| Condition axis | Current handling | Squintly capture | Source |
|---|---|---|---|
| dpr | implicit "≈1" via training corpus | `devicePixelRatio` per session | `squintly/SPEC.md:134, 187` |
| intrinsic_to_device_ratio | not modeled | derived per trial | `SPEC.md:162-164` |
| viewing distance | implicit "screen-typical" | self-report + Li 2020 chinrest | `SPEC.md:144-153` |
| ambient light | not modeled | 3-bin self-report + ALS opportunistic | `SPEC.md:42-44, 147` |
| color gamut | sRGB normalized internally | `matchMedia(color-gamut)` | `SPEC.md:137` |
| pixels per degree | not modeled | derived from calibration | `SPEC.md:152-153` |

### 2.2 Other latent gaps from existing benchmarks

- **Scale invariance** (`docs/scale-invariance.md`): zensim |β|/oct =
  0.458 on codec, vs DSSIM 0.0001. Codec scores drift ~0.5 zenpoints per
  octave of pixel count change. Drift narrows at high q (line 76 onward).
  Worst at q ≤ 30 where web traffic lives.
- **Gaussian-blur scale drift**: |β|/oct = 1.881 — nearly worst-in-class.
- **Low-q SROCC**: V0_6 25-40 band at 0.90, ≥90 band at 0.86 (§1.3). The
  V0_7 fill plan is targeted at exactly this gap. Squintly low-q bias
  (`SPEC.md:122-124, 60`) is the human equivalent.

### 2.3 Content-class signal

`/home/lilith/work/zen/zenanalyze/benchmarks/per_class_signal_probe_bprime_2026-05-04.md`
falsifies the encoder-side per-class quant table hypothesis (36/36 cells
fail). But it confirms a **metric-side** asymmetry:

- At q=85, screens are 3× more sensitive (ΔBA +0.705) than photos
  (+0.075) to identical HF coarsening. Lines 184-189.
- Clusters tested: photo, screen, technical, synthetic.

Content class is therefore worth conditioning the metric on, even though
it doesn't justify per-class encoder tables. V0_6 already feeds zenanalyze
features into the MLP (`benchmarks/4metric_overnight_FINAL_2026-05-01.md`
§"V0_6 (zenanalyze features…) — best of 12 theoretically grounded
combos. dct_hf wins"). The next tier should keep this and add condition
inputs.

---

## 3. Training-data requirements

### 3.1 Sizing the next condition-aware MLP

V0_4 placeholder is 228 → 1 single linear layer (~228 params plus
scaler). V0_5/V0_6 architectures are not in the doc but per RankNet+
magnitude-matching defaults, likely 228 → 64 → 1 ≈ **14 720** params.

Three plausible next shapes, all building on V0_4's ZNPK runtime:

**Design A — V0_4 head + condition concat (recommended).**
228 zensim features + 10 condition features → 64 → 32 → 1.
- Params: `(238 × 64) + 64 + (64 × 32) + 32 + 32 + 1 = 17 377` ≈ **17.4 k**.

**Design B — FiLM-conditioned head.**
228 → 128 → FiLM-gated (γ, β from condition vector via 10 → 32 → 256) → 1.
- Params: `(228 × 128) + 128 + (10 × 32 + 32) + (32 × 256 + 256) + 256 + 1
  ≈ 38 786` ≈ **38.8 k**.

**Design C — minimal jump (matches V0_5/V0_6 capacity).**
238 → 64 → 1, no conditioning gate.
- Params: `(238 × 64) + 64 + 64 + 1 = 15 425` ≈ **15.4 k**.

| Design | Params | 10× rule | 30× comfort |
|---|---:|---:|---:|
| C (minimal) | 15.4 k | **154 k pairs** | 462 k |
| A (default) | 17.4 k | **174 k pairs** | 522 k |
| B (FiLM) | 38.8 k | **388 k pairs** | 1.16 M |

The 10× rule is the standard tabular-MLP literature floor. Pairwise BT
labels have lower noise than raw MOS, so 10–15× is realistic.

**Recommended target: 200 k pairs minimum, 500 k for comfort.** The
existing 218 k synthetic + 140 k V0_7 fill = 358 k condition-free pairs
covers Design A's training set if conditions are added by *augmenting*
(synthesizing pseudo-conditions from rendered DPI / display profile)
plus ~50-100 k real squintly trials with measured conditions.

### 3.2 Coverage requirements per axis

CLAUDE.md "Sweep / Calibration discipline" applies because these numbers
inform `WEIGHTS_PREVIEW_V0_5+`. Floors per bin = sample size needed to
fit a per-bin correction without overfit; for FiLM-style continuous
conditioning floors are *coverage* requirements rather than per-cell
counts.

| Axis | Bins | N per bin | Subtotal | Source |
|---|---|---:|---:|---|
| viewing distance | <25 cm, 25-50, 50-75, 75+ | 500 | 2 000 | `SPEC.md:145-147` |
| dpr | 1, 2, 3, 4 | 1 000 | 4 000 | `SPEC.md:134` |
| intrinsic_to_device_ratio | <0.5, 0.5-1.5, >1.5 | 1 000 | 3 000 | `SPEC.md:162-164` |
| ambient light | dim, normal, bright | 1 000 | 3 000 | `SPEC.md:42-44` |
| color gamut | sRGB, P3, Rec2020 | 1 000 | 3 000 | `SPEC.md:137` |
| content class | photo, screen, line-art, synthetic, mixed | 2 000 | 10 000 | per_class_signal_probe |
| codec | zenjpeg, zenwebp, zenavif, zenjxl, mozjpeg, cwebp, avifenc, cjxl | 1 500 | 12 000 | current synthetic codec set |
| quality (CLAUDE.md density) | step 5 q0-70 + step 2 q70-100 = 31 bins | 800 | 24 800 | CLAUDE.md |
| source size | <64, 256, 512-1024, 2048+ | 2 500 | 10 000 | CLAUDE.md 4-bucket rule |

**Multiplicative product (full crossing):**
4 × 4 × 3 × 3 × 3 × 5 × 8 × 31 × 4 = **10 712 064 cells**

At 5 pair opinions per cell (`squintly/docs/methodology.md:278-280`,
following CID22 Figure 7 calibration), full crossing ≈ **54 M pair
opinions** — unmanageable.

**Mitigations:**

1. **Latin-hypercube design.** Squintly's session sampler already
   inverse-weights by coverage (`SPEC.md:116-130`). Tighten to LHS at
   ~1-2 % of full crossing → ~150 k cells × 5 = **~750 k opinions**.
2. **Continuous condition inputs.** dpr, viewing-distance, ppd as
   continuous MLP inputs eliminates per-cell counting; we need *joint
   distribution* coverage. ~5 k-10 k observers naturally span this.
3. **Augment with rendered conditions.** Existing 218 k synthetic +
   140 k V0_7 pairs can be *replayed* with simulated conditions (varied
   DPR, simulated angular subtense via downsampling) to get a
   condition-augmented training set without new human labels.

**Realistic v0 target:** 200 k-500 k human pair opinions across the
natural device/distance distribution from ~5 k-10 k observers, plus
condition-augmentation of the existing 358 k synthetic pairs.

### 3.3 Stratification ratios

Squintly corpus (`motivation-and-compensation.md:92-103`) is photo-heavy
(~84 %). For zensim training, oversample screen + document at the
trial-sampling layer so the trained model gets equal-class gradient
signal even if observers see fewer screens:

| Class | Squintly corpus share | Training oversample | Effective weight |
|---|---:|---:|---|
| Photo | ~76 % | 0.5× | 0.45 |
| Screen / UI / line-art | ~9 % | 2.5× | 0.25 |
| Document / chart | ~6 % | 1.5× | 0.10 |
| Art / synthetic / mixed | ~9 % | 1.0× | 0.10 |
| Astrophoto / scientific | ~6 % | 1.5× | 0.10 |

Per-class signal-probe (`per_class_signal_probe_bprime_2026-05-04.md`)
shows screens are 3× more metric-sensitive at q=85 — undersampling them
is exactly the wrong move.

### 3.4 Quality-bin density (CLAUDE.md compliance)

Per global CLAUDE.md "Sweep / Calibration discipline":
- Step 5 q-grid 0..70 (15 points) + step 2 q-grid 70..100 (16 points) =
  **31 quality samples**, NOT 21.
- For per-q anchor calibration: step 1 between q75-q95 = 21 points;
  combined unique = **38 quality samples**.

Squintly implements this implicitly (`SPEC.md:122-124, 60`: "Quality
bias … the lower half of the quality grid 60% of the time"). The V0_7
fill plan (jj `nmmxmlmw`) targets ~140 k zenjpeg-420-e1 pairs covering
SSIM2 0-90 with low-band oversample 0.5 — exactly the coverage gap.

### 3.5 Source-image stratification (training data)

CLAUDE.md training rule: cluster sources by feature embedding (k-means
on `feat_*`), not random sampling. For zensim:

- Cluster the squintly corpus into K = 20 clusters using zenanalyze's
  per-image feature vector (entropy, edge density, variance, patch
  fingerprints — see `per_class_signal_probe_bprime_2026-05-04.md:240-247`).
- Pick centroid-nearest member of each cluster → 20 representative
  images.
- For each, generate ≥ 16 log-spaced sizes (32 → 4096 px) using
  Mitchell-Netravali resampling. Skip upscaling (synthetic upscale features
  mislead).
- Encode at 31-point quality grid for each codec.

20 sources × 16 sizes × 31 q × 8 codecs = **79 360 unique encodings**,
collapse to ~10 k after pareto-front filtering.

---

## 4. Acquisition timeline + cost

### 4.1 Per-pair cost

Per `motivation-and-compensation.md:155-178`:

| Mode | Cost / trial | Quality (PLOS 2023) | When available |
|---|---:|---|---|
| Volunteer | $0 | 92 % (AAAI vs paid 78 %) | v0.2 (now) |
| Prolific (cohort completion) | ~$0.04/trial quality-adjusted at £6/hr | 67.94 % | v0.4+ |
| Charity ($0.02-0.05/trial donated) | $0.02-0.05 | volunteer-tier | v0.3 |

The user's prompt cites "$0.50-$2.00 per pair" — that's the literature
*upper bound* for slow-paced annotation. Squintly trials run ~10-15 s
each, so at £6/hr per-trial rate is ~$0.025; the $0.04 quality-adjusted
figure already includes the failure rate.

### 4.2 Realistic budget

| Pairs target | Mode mix | Direct cost |
|---|---|---:|
| 200 k (Design C floor) | volunteer | $0 (UX + outreach only) |
| 500 k (Design A comfort) | 80 % volunteer, 20 % Prolific cohort fill | ~$4 000 |
| 1.6 M (LHS condition coverage, Design B) | 60 % volunteer, 30 % charity, 10 % Prolific | ~$25 000 charity + $6 400 Prolific |

### 4.3 Throughput

Per `motivation-and-compensation.md:104-107`: ~80 k candidate trials =
6 months at 1 k phone-observers × 100 trials. Scaled:

| Target | Months volunteer-only | Months with paid acceleration |
|---|---:|---:|
| 200 k | 12-15 | 4-6 |
| 500 k | 30-36 | 8-10 |
| 1.6 M | 60+ | 18-24 |

### 4.4 Existing public datasets — warm start

| Dataset | Pairs | Use as | Caveat |
|---|---:|---|---|
| CID22 | 4 292 | warm-start training, evaluation | 49 ref images blocked from synthetic |
| TID2013 | 3 000 | warm-start training, evaluation | 25 refs, no overlap with synthetic |
| KADID-10k | 10 125 | warm-start training, evaluation | 81 refs, no overlap with synthetic |
| Synthetic safe | 218 089 | full training | shipped, no condition data |
| V0_7 zenjpeg-420-e1 fill | ~140 000 | low-q coverage | infrastructure ready, not run |
| KonIQ-10k | 10 073 | warm-start (NOT used yet) | conditions partial |
| KonJND-1k PJND | per `zensim-validate` | warm-start (NEGATIVE result, see §1.2) | inflection-point objective conflicts |
| KonFiG-IQA, PIPAL, CSIQ | per `zensim-validate/src/main.rs:42` | training/validation | supported, not currently used |

**Warm-start strategy:**
1. V0_7 (no human labels, no conditions) — runs now from existing 358 k
   pairs once V0_7 plan executes.
2. V0_8 (V0_7 + CID22+KADID+TID fine-tune) — adds ~17 k human pairs, no
   conditions. Caps at the linear-architecture's headroom.
3. V0_9 (V0_8 + squintly condition-aware fine-tune) — first model that
   uses conditions. Starts shipping when squintly hits ≥ 50 k pairs.

Each stage validates on a held-out fraction (squintly `held_out=1` per
`SPEC.md` and `methodology.md:296-303`).

---

## 5. Sequencing / unblock

### 5.1 What needs to land first

| Dependency | Status | Source |
|---|---|---|
| MLP runtime in zensim | DONE on experimental jj branch (`ykzptwqq`) | not yet on main |
| V0_4 placeholder | DONE | not yet on main |
| V0_5/V0_6 trained weights | DONE | jj `nmmxmlmw` |
| V0_7 post-fill plan | scripts committed, not run | jj `nmmxmlmw`, `benchmarks/v07_postfill_run.sh` |
| Squintly v0.1 (works locally) | shipped | `squintly/README.md:67` |
| Squintly hosted deployment | **BLOCKING for condition data** | `squintly/docs/HANDOFF.md:1212-1217` |
| Coefficient deployment | **BLOCKING (squintly's image source)** | `HANDOFF.md:1212-1217` |
| Seed scripts (anchors, held-out, calibration) | not started | `HANDOFF.md:1219-1228` |
| ASAP runtime active sampling | not started, lowers N by 30-50 % | `HANDOFF.md:1230-1239` |
| Held-out export filter | infra ready, not wired | `HANDOFF.md:1250-1254` |
| Crowdsourcing channel | not chosen | `HANDOFF.md:1274-1278` |

Until squintly is hosted with coefficient image source, **no condition-
aware data flows.** The condition-aware tier is paused on §14.1 of
HANDOFF.

### 5.2 Training-loop status

| Component | Status | Path |
|---|---|---|
| Dataset loaders (7 formats) | DONE | `zensim-validate/src/main.rs` (TID, KADID, CSIQ, PIPAL, CID22, KonFiG, synthetic) |
| Linear training (NM/CMA-ES/coord/RankNet) | DONE | `zensim-validate/src/main.rs:1273-2723` |
| Feature cache (`ZSFC` magic) | per `TODO.md:64-69` | not yet split out |
| Cross-validation (k-fold, leave-one-out) | per `TODO.md:89-94` | functional |
| MLP forward inference | DONE | `zensim/src/mlp/inference.rs` (jj `ykzptwqq`) |
| MLP bake (write ZNPK) | DONE for single-layer linear | `zensim/src/mlp/bake.rs` |
| MLP training (RankNet+magnitude-matching, V0_4-smooth) | DONE | jj `nmmxmlmw`, plus zenanalyze-feature variants V0_5/V0_6 |
| Low-band oversample | DONE | `--mlp-low-band-oversample 0.5` flag |
| **Condition-aware loss** | **NOT STARTED** | |
| **Squintly TSV ingest** | **NOT STARTED** | |
| **Multi-layer bake (ReLU + I8)** | partial — runtime supports it, bake helper only does single linear | `mlp/bake.rs` |

### 5.3 v0 dataset format

Squintly already exports in zenanalyze/zentrain pareto schema with a BT-
fitted `quality` column (`squintly/README.md:58-61`). For zensim training:

- One row per (source_hash, encoding_id, condition_bucket).
- Columns: 228 zensim features + ≤ 25 zenanalyze features (per V0_6) +
  10 condition features + BT quality scalar from squintly + bootstrap CI
  + observer count + held-out flag.
- Filter: `held_out = 0`, `weight > 0` (warmup discarded), `is_golden = 0`
  (anchors used for evaluation only).

Add to zensim:
- `zensim-validate` loader for squintly `pareto.tsv` (`load_squintly`).
- Multi-layer bake helper in `zensim::mlp::bake` (currently only single
  linear).
- New `ZensimProfile::PreviewV0_9` (or whatever number is current after
  V0_7/V0_8 ship) with condition-aware MLP weights.

### 5.4 Minimum-viable / dream

**v0.4-on-main (immediate):**
- Land jj `ykzptwqq` (MLP runtime + V0_4 placeholder) on main@origin.
- Public API additions are additive only; V0_2 default unchanged.
- Cost: 0; data exists.
- Unblocks: enables V0_5/V0_6/V0_7 to be trained against and shipped.

**v0.5-v0.7 (over the next ~weeks):**
- Run V0_7 post-fill harness (`benchmarks/v07_postfill_run.sh`).
- Decide V0_5 vs V0_6-dct_hf vs V0_7 based on 4-metric report.
- Ship as `PreviewV0_5` with retrained ZNPK weights.
- Target: KADID ≥ 0.85, TID ≥ 0.85, CID22 ≥ 0.89.
- No squintly dependency.

**v0.8 (condition-aware MVP):**
- Design A (~17 k params, 228+10 → 64 → 32 → 1).
- Trained on 358 k synthetic+fill + 17 k human + initial 50 k squintly
  pairs.
- Conditions: dpr, viewing_distance, ambient_light, gamut, content_class
  (5 axes, ~10 features after one-hot).
- Target: SROCC > 0.88 on CID22, > 0.85 on TID/KADID, **> 0.92 on
  squintly held-out**.
- Unblock: squintly hosted + 6-12 months of data + multi-layer bake +
  condition-aware loss.

**Dream tier (v1.x):**
- Design B (~38 k params, FiLM-gated).
- 500 k-1.6 M condition-conditioned pairs.
- Pixels-per-degree as continuous input (requires squintly calibration
  uptake).
- Per-content-class adaptive frequency weighting.
- Target: SROCC > 0.92 on every dataset, < 0.5 zenpoint scale-invariance
  drift per octave.

---

## 6. Open questions

1. **Land V0_4 on main first.** The experimental jj branch (`ykzptwqq`)
   has been parked since 2026-04-30. Folding the MLP runtime into main is
   prerequisite to everything else and creates no risk (V0_2 default
   path unchanged).
2. **V0_5 vs V0_6 vs V0_7 — pick one.** The 4-metric overnight report
   (jj `nmmxmlmw`) shows V0_5 ≈ V0_6 within noise on every holdout.
   Choosing requires a smoothness comparison on q-sweeps
   (`benchmarks/4metric_smoothness_addendum_2026-05-01.md`) — V0_6 wins
   smoothness per the commit message.
3. **Squintly architecture choice — concat vs FiLM.** Defer until ≥
   50 k squintly pairs exist for ablation.
4. **Held-out evaluation overlap.** Squintly's 20 % held-out subset and
   CID22's validation split must not share source images. Need an
   automated check at training time (extension of `CLAUDE.md` dataset
   contamination rules to squintly sources).
5. **Bootstrap CI propagation.** Per `methodology.md:262-270`, squintly
   produces 5th/95th percentile per row. A heteroscedastic loss
   (weighted by 1/CI-width) converges faster on noisy human data than
   unweighted MSE. The MLP training harness already supports per-pair
   weighting via `weight` column — wire it in.

---

## Appendix A — files referenced

- `/home/lilith/work/zen/zensim/README.md`
- `/home/lilith/work/zen/zensim/CLAUDE.md`
- `/home/lilith/work/zen/zensim/CHANGELOG.md`
- `/home/lilith/work/zen/zensim/docs/jxl-encoding.md`
- `/home/lilith/work/zen/zensim/docs/scale-invariance.md`
- `/home/lilith/work/zen/zensim/zensim/src/lib.rs`
- `/home/lilith/work/zen/zensim/zensim/src/profile.rs`
- `/home/lilith/work/zen/zensim/zensim/src/metric.rs`
- `/home/lilith/work/zen/zensim/zensim-validate/src/main.rs`
- `/home/lilith/work/zen/zensim/zensim-validate/TODO.md`
- `/home/lilith/work/zen/zensim/zensim-validate/weights/gpu_ssim2_v2_163k.txt`
- `/home/lilith/work/squintly/README.md`
- `/home/lilith/work/squintly/SPEC.md`
- `/home/lilith/work/squintly/docs/methodology.md`
- `/home/lilith/work/squintly/docs/motivation-and-compensation.md`
- `/home/lilith/work/squintly/docs/HANDOFF.md`
- `/home/lilith/work/zen/zenanalyze/benchmarks/per_class_signal_probe_bprime_2026-05-04.md`

## Appendix B — relevant experimental jj changes (not on main)

| Change ID | Description |
|---|---|
| `ykzptwqq` | feat(zensim): vendor MLP runtime + V0_4 profile with V0_2-equivalent placeholder |
| `mluvyporz` | feat(bench/validate): butteraugli 3-norm gen + trainer target |
| `srltslklu` | feat(bench): V0_4 score-mapping calibration |
| `mnmylulnu` | feat(bench): V0_2 vs V0_4 profile compatibility report |
| `umrwvvmns` | experiment(v04-3norm): retrained V0_4 MLP on butteraugli 3-norm with perfect holdout |
| `nmmxmlmw` | experiment(v04-ssim2-holdout): SSIM2 target with perfect holdout BEATS V0_2 on CID22 + V0_7 plan |
| `pukvkqzks` | wip |
| `tmrtwxrk` | ci: fmt + clippy + wasm checksum updates |

These changes carry the in-flight V0_4-V0_7 work that informs §1.2-§1.3.
Landing them is the immediate path forward; condition-aware training (§3)
is the longer-horizon work.
