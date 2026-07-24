# The corruption 2nd head — rigorous, negrich-trained (2026-07-24)

The closed-loop metric is **one perceptual model** (rank + smooth dial + coherent
diffmap) **+ a separate corruption head** (see `ideal_clean_model_2026-07-24.md`
RESOLUTION). The perceptual metric is *blind to ~75% of structural corruptions* —
channel-swaps, flips, block-garbage look sharp/fine, scoring >40 ("looks OK"). The
corruption head catches them. Deploy: `final = min(perceptual, gate)`, `gate = 100`
unless `P(corruption) > T`.

## Corpus (the missing multi-source structural positives)

The prior corruption corpus was a **single image** (gb82_dog), so detection
*generalization to unseen images* was untestable. This builds the multi-source
positives: **141 imazen-26 sources** (photo/text/screen, 14 categories) × the
codec-corpus catalog (44 structural families × region × severity) → **95,424
corruption rows** + matched honest q10/q20 anchors, 720-feat, source-held-out.
`build_corruption_corpus.py` (streaming generate→extract→discard; Lanczos-caps
sources at 1024px — imazen-26 runs to 146 MP). Corpus:
`/mnt/v/output/zensim/corruption-corpus-2026-07-24/im26_corruption_720.parquet`.
(142/174 refs; 32 skipped/uncompleted — a screen image tripped a truncated-CSV
guard, now handled; 141 sources is ample for source-held-out.)

## negrich — the breakthrough hard negatives, regenerated LEAK-FREE

Structural corruptions have extreme, corruption-*looking* features — but so do
**severe-but-honest** degradations (heavy blur/noise/motion). negrich is the
severe-honest set that teaches the boundary. The original `kadis_sample_negrich`
had **no builder and dropped `source_id`** (unverifiable splits). Wrote the missing
builder (`build_kadis_negrich_from_canonical.py`): the negative-rich subset
(`score_zensim < 0`, 280,384 rows, 113,871 unique KADIS sources) of the 700k KADIS
canonical, **source_id restored** → leak-free split. The 24 KADIS IQA distortion
types confirmed (blur/noise/compress/color/pixelate). Output:
`/mnt/v/zen/zensim-training/kadis-negrich-regen-2026-07-24/kadis_negrich_srcid.parquet`.

## Two measured findings

**1. 372 ≈ 720 — v2 adds ~1% (noise).** Clean feature comparison (same data, no
negrich): 372 → 98.2% detection / 0.55% broad-FP; 720 → 99.4% / 0.38%. Corruption
is coarse-structural (v1 SSIM craters on it), so v1 suffices. **The head is 372**
(`f0..f371`, a free subset of the deployed 720 extraction) — equivalent, and it
composes with the full native-372 negrich with no 720-regen pass.

**2. negrich is ESSENTIAL — not optional.** The *no-negrich* head, scored on
severe-honest, **false-positives at 82%** (fires on 82% of honest blur/noise as
"corruption") — its 98% "detection" was hollow (it fires on anything extreme). With
negrich in training: **0.06% severe-honest FP**. Without the boundary, the head is
useless for `min()`.

## The shipped head (source-held-out, T=0.9)

372-feat logistic + isotonic calibration, `35k negrich + 40k broad-honest + matched
anchors` negatives, split leak-free on both corruption `ref_id` and KADIS `source_id`:

| metric (held-out sources) | value |
|---|--:|
| corruption detection | **84.6%** (photo 84 / screen 82 / text 87) |
| FP — severe-honest (negrich) | **0.06%** |
| FP — broad-honest | **0.34%** |
| FP — matched q10/q20 anchor | **0.00%** |
| perceptual-miss value-add | perceptual misses 75.8%; head catches **89.3%** of those |

Detection is 85% (not 99%) *because* negrich correctly makes it conservative — the
missed 15% are corruptions that genuinely resemble severe-honest, better missed than
false-firing on honest content in the loop. Deadband `T=0.9` → ~0.3% honest FP.

**Persisted** (durable — block storage + Tower + manifest):
`/mnt/v/output/zensim/corruption-head-2026-07-24/`
- `corruption_head_372.json` — portable head: weights + isotonic calibration +
  deadband T + deploy formula (`P=isotonic(sigmoid(clip((feat[:372]−mean)/scale)·
  coef+intercept))`; scores without sklearn).
- `metrics.json`, `_MANIFEST.json` (build_commit + input paths + split policy).
- `head_{372,720}_nonegrich.json` — the feature-comparison heads.
Tower: `/mnt/tower/output/zensim-corruption-head-2026-07-24/`.

## Feature ablation (perf question)

`train_corruption_head.py --ablate` sweeps top-K features (by |coef|). Detection +
FP vs K (held-out sources, T=0.9):

| K | detection | severe-honest FP | broad FP |
|--:|--:|--:|--:|
| 32 | 77.8% | 0.17% | 0.41% |
| 48 | 80.6% | 0.11% | 0.28% |
| **64** | **81.8%** | **0.04%** | 0.34% |
| 156 | 85.1% | 0.06% | 0.37% |
| 372 | 84.8% | 0.06% | 0.33% |

**64 features ≈ full (81.8% vs 84.8%)** — a 6× smaller head for ~3% detection. But
the perf payoff is limited, and honestly so: the top features span **all scales
(s0..s3), both chroma channels, AND both blocks** (basic `f17-21` at the *finest*
scale + mask/iw/peak `f255-334`). Consequences:
- **Deployment is already free:** the head reads features the perceptual model
  already extracts → one dot-product; K doesn't change deployment cost.
- **No cheap standalone gate:** the signal needs the finest (most expensive) scale
  *and* the mask/iw/peak block, so corruption detection cannot run without the full
  multi-scale pipeline. A fast regression-test gate that skips the perceptual
  extraction is NOT reachable from this feature set.
So ablation buys a smaller/less-overfit head + the knowledge that 64 features
suffice — not a speedup. (A genuinely-cheap gate would need corruption-specific
*cheap* features, e.g. a coarse-scale structural-signature à la
`structural_signature_spike.py`, which is a different design.)

## Follow-ons

- Detection 85% + broad-FP 0.34% is a solid v1; a small MLP (vs logistic) or more
  broad-honest could lift detection at equal FP.
- 720-negrich (via kadis-distort on imazen-26, matched content) only if a future
  measurement shows v2 helps — this one says it doesn't, materially.
- Finish the last 32 corpus sources (builder now guards truncated CSVs).

## The dial+diffmap subset detects corruption BEST (2026-07-24, reverses "372")

The perceptual (dial+diffmap) model uses only its **foldable subset** — basic-156 +
228 foldable v2, and ZERO of the f156..371 mask/iw/peak block. Asked whether
corruption can run on THAT subset (→ one shared feature extraction for the whole
system). Measured (source-held-out, corpus honest negatives, no negrich, T=0.9):

| subset | nfeat | detection | broad-FP |
|---|--:|--:|--:|
| basic-156 | 156 | 94.6% | 0.39% |
| **perceptual foldable (basic+v2)** | **384** | **98.0%** | **0.18%** |
| native-372 (basic+v1-mask) | 372 | 96.9% | 0.35% |
| full-720 | 720 | 89.8% (overfits) | 0.44% |

**The perceptual foldable subset is the BEST for corruption** — the foldable v2
features are more discriminative than the v1 mask/iw/peak block, and full-720
overfits on the source-held-out split. So:
- **One shared feature extraction** for dial + diffmap + corruption (the 384-feat
  foldable subset); the f156..371 block can be dropped entirely — neither head needs
  it. Corruption is then truly free AND enables the extraction optimization.
- This REVERSES the earlier "372≈720, v2 marginal" (which compared basic+v1-mask vs
  all-720; the right axis is basic+v2, which wins).
- **To ship it, negrich must be 720** (its severe-honest hard negatives need v2). The
  regen path: kadis-distort the 24 IQA types on the imazen-26 sources (matched
  content) → extract 720 → the severe-honest boundary on the shared subset. This is
  the payoff that justifies the 720-negrich regen deferred earlier.
Subset scan: `corruption-head-2026-07-24/compare_subsets.py` + `foldable_idx.npy`.

## Unified head (foldable-384 subset + MATCHED 720 severe-honest) — 2026-07-24

Regenerated severe-honest at 720 the right way — matched content, no confound —
via `build_severe_honest_720.py`: kadis-distort's 24 IQA types × severe levels
{3,4,5} on the SAME imazen-26 sources → 11,664 severe-honest rows, 161 sources,
all 720-feat (`severe-honest-720-2026-07-24/`). Trainer gained `--feat-subset`
(arbitrary feature indices) + `--severe-720` (720 severe negatives, leak-free on
ref_id) so the corruption head can train on the dial+diffmap **foldable-384**
subset.

Trained the unified head (foldable-384 + matched-720 severe-honest, source-held-out):
detection 89.7% (photo/text 89.7/89.8), broad-honest FP 0.04%, matched-anchor 0.00%,
value-add: catches 92.6% of perceptual-missed corruptions. **The 384-subset is
viable** — corruption runs on exactly the dial+diffmap features, one shared
extraction, mask/iw/peak dropped.

BUT the corruption-vs-severe-honest boundary is **under-trained**: severe-honest FP
= 8.77% at T=0.9 (foldable-384) / 6.11% (full-720) — the mask/iw/peak block barely
helps (so keep it dropped), but 11k matched severe-honest is too little vs 95k
corruption. (The earlier 0.06% was a CONTENT artifact — KADIS-content negrich ≠
imazen corruptions; this matched test is the honest, harder one.)

**What closes it — the full 720 negrich (the lilith-lianli job):** the KADIS-700k
severe-honest at 720 (280k rows, v2-populated) gives the volume + KADIS content
diversity to nail the boundary on the shared subset. Re-extract 720 on the
GPU-canonical `distorted_url` PNGs, or regenerate via kadis-distort on the 140k
KADIS refs. This is the standing "all datasets → 720" population, routed to
lilith-lianli. Heads persisted: `corruption_head_foldable384.json` (with `feat_idx`),
`head_full720_matched.json`.
