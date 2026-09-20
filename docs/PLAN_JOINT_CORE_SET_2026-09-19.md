# One compact training core, and the zensim-vs-DVIFM-pyramid tradeoff (2026-09-19)

User direction 2026-09-19: "establish a training set that we know is diverse but also good enough to train zensim
and this model both, and small enough"; "we can stop doing luma only work"; "evaluate the tradeoffs of zensim vs
yuv binomial 5 planes as described in the talk with empirical data".

Companions: [fitted-constant guards](FITTED_CONSTANT_GUARDS_2026-09-19.md),
[scales/planes preregistration](PREREG_SCALES_PLANES_2026-09-19.md),
[spatial steering plan](PLAN_SPATIAL_STEERING_DVIFM_2026-09-19.md). Split rules:
[DATA_SPLITS](DATA_SPLITS.md) and [`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) still govern; this plan adds a
view over admitted TRAIN data, never a new role for anything.

## Why one core set

The evidence base is split across legs that were never sized as a whole, and both extremes hurt:

- **Too small.** The admitted human-labelled TRAIN estate is 8,327 rows (KADID train refs 5,000 + TID 3,000 +
  KonFiG 327). Measured at that size: every arm's development metric DECLINED between 50 and 100 epochs, and
  30 extra inputs cost ~0.003 SROCC whether or not they carried signal
  (`benchmarks/dvifm_screen2b_2026-09-19.md`). Any 228-input head is data-limited there.
- **Too big / too scattered.** The leaders' recipe reads 168,142 fit rows across four legs; a DVIFM constants fit
  over the same rows would need ~520 GB of block cache at the measured 971 KB per row per plane, against ~102 GB
  free. Phase 2d already spent 60 GB on 6.5k plane-rows.

So: one named, versioned, **compact** core that is large enough for a 228+-input head, diverse enough that a
content class cannot be silently missing, and cheap enough that a constants fit and a full zensim fit can both run
on it repeatedly on one box.

## Composition targets (all TRAIN-role, reference-level disjoint from every eval corpus)

Target **≈50,000 pairs**, ~1,500–2,500 distinct reference variants, with these legs and minimum shares. Shares are
of pairs; every leg keeps its own target column and orientation, and legs are never column-mixed.

| Leg | Role in the core | Target share | Target column |
|---|---|---|---|
| SafeSyn (synthetic distortions) | breadth of distortion types, dense low quality | 25% | ssim2-derived (teacher) |
| CID22-train-201 | photographic content at human-study scale | 15% | ssim2-derived (teacher) |
| Modern-codec sweeps (JPEG, WebP, AVIF, JXL; q5–q100, **denser below q60**) | the product's own distortions, per-codec floors | 25% | two-reference proxy (ssim2 ∧ butteraugli) |
| imazen-26 crops (photo / screen / text / line-art) | content classes zensim is weakest on | 15% | ssim2-derived (teacher) |
| KADID + TID train refs | human labels, distortion-type coverage | 10% | human MOS/DMOS, mapped to one 0–100 quality scale |
| KonFiG originsplit-train | JND-scale anchoring near threshold | 5% | JND-derived quality |
| hdr_v3mix train origins | the HDR leg that B's head liked | 5% | cvvdp-mix (teacher) |

Diversity is enforced, not hoped for:

- **Stratify by clustering**, not at random: k-means over the existing `feat_*` embedding, pick centroid-nearest
  members per cluster, keep singleton clusters (they are the outliers a model fails on). Record cluster sizes.
- **Content-class floor:** each of photo / screen / text / line-art / HDR holds ≥10% of its leg's references, and
  **real camera photography is the plurality of the core, not a minority.** Measured on the imazen26 leg that the
  phase-2d constants fit actually used (all 212 origins / 12,246 pairs, gallery
  `/mnt/v/output/zensim/demos/imazen26-subset-2026-09-19/origins.html`): AI-generated products/illustrations/clipart
  34%, screenshots + plots 31%, document scans 15%, **camera photography only 19% from 40 origins**. Defensible for a
  codec picker; wrong for fitting masking constants that model human vision on photographic texture. **Target ≥75%
  camera photography by pairs** (user, 2026-09-19: "perhaps real photos should be 75% tbh"), drawn from the imazen-26
  photo classes (1000-photos-general, 1200-interiors, 1400-nature, 1600-food, 2000-people, 3000/3300-museum) plus
  CID22-train and the KADID/TID photographic references. The remaining ≤25% keeps a **floor of 5% each** for screen
  content, document/text and line-art/synthetic, so a regression on those classes is still detectable — they are
  guard classes here, not training mass. AI-generated imagery is not a content class for metric fitting: cap it at
  5% and label it, since its statistics are a generator's, not a camera's or a renderer's.
  Note the consequence: with only 40 photographic origins in today's imazen26 leg, a 75% photo core needs MORE photo
  origins (the canonical corpus has 111 in the photo classes at a 12 MP median) and more renditions per origin, not
  a reweighting of what already exists.
- **Scale ladder (user, 2026-09-19): balanced, Mitchell, capped near 1 MP.**
  - **Cap ≈1 MP** (1024 px long edge). Nothing above it: the compute cost is not worth it, and the existing top step
    already sits at 0.8–1.05 MP. "More pixels of photos" means moving pairs UP the existing ladder, not extending it.
  - **Do not let tiny thumbnails dominate.** Measured today: the clean-picker rendition set is 1,656 tiny / 828 small
    / 1,224 medium / 789 large of 4,497, and the imazen26 subset spreads ~1,100 pairs over each of 11 steps, so 5 of
    11 steps (45% of pairs) sit at ≤256 px. Target instead: **≥55% of pairs at 384–1024 px**, ~25% at 192–256, and a
    ~20% tail at 64–128 kept only because the sweep discipline needs the fixed-overhead end of the size axis.
  - **Mitchell–Netravali is the standard downscaler** for every rendition and every gallery thumbnail — better for
    human visual preference than Lanczos (which rings) and than Triangle (which blurs; the invalid 2026-09-13 scale
    study used Triangle only). Via `zenresize`; record the kernel in `variants.tsv` per the codec-corpus
    VARIANTS-SPEC. Identical kernel on both sides of every pair, always.
- **Quality-axis floor:** within each codec leg, q5–q60 carries at least the same pair density as q60–q100.
- **Near-lossless and identity anchors** present by construction (the dial's ends), plus the corruption-gate
  negatives as a small tail so the integrity head keeps a training signal.
- **Never clip a teacher target** (user, 2026-09-19: "don't clip ssim2 scores!"). SSIMULACRA2 is signed and goes
  below 0 on badly damaged pairs; clipping to [0,1] pins those rows at a floor with no gradient and biases every
  constant fitted against them toward whatever explains the flat region. Measured on the imazen26 leg as built for
  the phase-2d constants fit: **885 of 12,246 rows (7.2%) had a negative raw score clipped to 0** — all at the
  aggressive end, exactly where the dial has to work. The core keeps the RAW SIGNED teacher value. If a bounded
  target is required by a loss, use a strictly monotone squashing map (order-preserving, invertible) and record
  it; never a clamp. Rows are dropped only when the teacher itself is undefined, never because it is negative.
  The guards doc's saturation detector fires on any fit where >5% of rows sit at a target bound.
  **If a consumer genuinely cannot take a negative target** (user, 2026-09-19: "if the system can't do negatives
  well - try skipping those rows"), DROP those rows from that consumer's view and record the count and their
  distribution over codec, quality and content class — never clamp them into the fit. Dropping loses the rows;
  clamping corrupts every row's constant. Measured negative shares in the leaders' own legs: SafeSyn 8,195/141,054
  (5.8%, minimum −743.9), codec 450/7,947 (5.7%, minimum −64.4); CID22-train and the human legs have none.
  Separately, `codec_fit` has **1,227 of 7,947 rows (15.4%) pinned at exactly 100** — a ceiling saturation that
  trips the same detector at the other end and needs the near-lossless rows spread, not stacked on the bound.
- **Dedup and audit:** exact-pixel dedup within legs, dHash audit against every T0 eval corpus (CID22-49, AIC-3,
  AIC-4, AIC2026, SDR25, KonJND val, KonFiG test) with flags adjudicated, not auto-quarantined.

## Why it is small enough

- Feature tables: ~50k rows × ~1,000 f32 ≈ **200 MB** per feature-set identity (Parquet, zstd).
- DVIFM constants fitting: per (plane, level) **2-D histograms** of (C̃, m) in the integer log domain — a few
  hundred KB per domain, and exact for the whole C₀ × β grid, because each grid cell's loss is a sum of per-block
  terms. Per-block records only on a capped, quantised subsample (≤40 KB per row per plane).
- Pixels are referenced by content hash through the existing store; the core set copies no image bytes.
- Whole core, including caches for one DVIFM variant: **≤25 GB**, which fits alongside everything else on `/mnt/v`.

## The one-hour gate (user, 2026-09-19: "let's try to make it something that can run in an hour")

The core is only useful if a full cycle is cheap enough to repeat. Acceptance requires, on this box, measured:

- **Constants fit ≤10 min** for all (plane, level) of a 3-plane DVIFM variant. This is what forces the histogram
  design: each C₀ × β grid cell's loss is a sum of per-block terms, so a 2-D histogram of (C̃, m) per (plane, level)
  gives the identical grid from a few hundred KB. The 2026-09-19 phase-2d run took **8 hours and did not finish**
  because it re-read per-block records for every cell — 16 minutes per plane-level on the imazen26 domain alone.
- **Five-seed model fit ≤40 min** for `basic228/h128` on the core, run ≤2 concurrent under run-heavy.
- **Extraction is separate and amortised** (once per feature-set identity), but must be ≤2 h for the whole core and
  produce a reusable table; the histograms are built in the same pass.
- **Whole repeat cycle (fit constants → re-bake → fit model → panel) ≤1 h.** Anything slower is a design bug, not a
  budget request: shrink the row count before shrinking seeds or budgets, and keep the fairness contract intact.

## How it is validated before anyone trains on it

1. **Reproduce a known model on it.** Refit the leaders' `basic228/h128` recipe on the core (same seeds, same
   budget rule) and compare to the frozen R915 result on the leaders' own tables. The core is admissible when the
   development-leg metrics land within seed noise, or when the gap is measured and stated.
2. **Convergence.** With the core, the development metric must still be rising or flat — not falling — between the
   50- and 100-epoch checkpoints for `basic228`. That is the test the 8.3k estate failed.
3. **Permuted-column control.** 30 permuted inputs must cost less than the seed noise on the core. If they still
   cost ~0.003 SROCC, the core is too small and the target grows before any feature is screened.
4. **Class coverage report** per content class and codec, with the metric per class, so a later regression can be
   attributed.

Recorded as `docs/DATA_SPLITS.md` addendum + `benchmarks/joint_core_2026-09-XX.md` + a `_MANIFEST.json` carrying
`build_commit`, per-input sha256 and the cluster/seed rules. Name and version it (`joint-core-v1`); every later
screen cites that name.

## The tradeoff study: zensim's pyramid vs the talk's YUV binomial 5-plane

Empirical, on the core set, with the leaders' recipe and paired seeds. **No luma-only arms** — chroma is settled as
the real DVIFM gain (+0.027 SROCC on CID22-A, +0.028 on TID/KADID, in-sample), while Y′-vs-XYB-Y as a single plane
did not replicate (+0.006 one way, −0.003 the other). The comparison is between complete designs:

| Arm | Planes | Scales | Band | Pooling |
|---|---|---|---|---|
| Z (production) | XYB (X, Y, B) | 4 × 2×2 box | 11×11 box-mean residual | global moments |
| D (the talk) | Y′CbCr | 5 × binomial [1 2 1] ↓2 + low-pass | Laplacian G−E(G↓) | 5×5 block peak × visibility, mean over blocks |
| D-local | Y′CbCr | as D | local band G−B²G | as D |
| H1 hybrid | XYB luma + Y′CbCr chroma | per source | per source | per source |
| H2 hybrid | XYB | zensim's box scales | zensim residual | DVIFM block pooling |
| H3 hybrid | Y′CbCr | binomial | Laplacian | global moments |

H2 and H3 separate the two ideas that are conflated today: **the decomposition** (which planes, which filters,
which bands) and **the pooling** (global moments vs block peak × masking). Each arm is judged on:

- accuracy on the core's development legs (registered composite, within-reference/local ordering, per codec, per
  content class), five paired seeds;
- **measured** cost — the planar-filter matrix (`~/tmp/devin/planar_filter_cost_prompt.md`): time and peak RSS at
  64²/256²/1024²/2048²/4096², 1T and 8T, `α + β·pixels` with both terms, never extrapolated;
- label-free stability: 1-px shift, codec-grid phase 0..7, crop-vs-full;
- streaming shape: halo rows, live planes, whether output is bit-identical across strip sizes;
- steering suitability: is the image score an exact sum over spatial units (DVIFM: yes; global moments: no).

Deliverable: one table where a reader can trade accuracy against milliseconds and bytes per pixel, plus a named
recommendation. A cheaper arm that ties on accuracy wins; a more accurate arm must state its cost.

## Order of work

1. Build and validate `joint-core-v1` (steps 1–4 above). Until it exists, screens keep using the 2d tables and say so.
2. Run the planar-filter cost matrix on a quiet box.
3. Run the six-arm tradeoff study on the core.
4. Only then revisit the i16 kernel, the fusion phases and spatial steering with real numbers on both axes.
