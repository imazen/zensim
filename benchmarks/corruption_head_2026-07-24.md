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

## Follow-ons

- Detection 85% + broad-FP 0.34% is a solid v1; a small MLP (vs logistic) or more
  broad-honest could lift detection at equal FP.
- 720-negrich (via kadis-distort on imazen-26, matched content) only if a future
  measurement shows v2 helps — this one says it doesn't, materially.
- Finish the last 32 corpus sources (builder now guards truncated CSVs).
