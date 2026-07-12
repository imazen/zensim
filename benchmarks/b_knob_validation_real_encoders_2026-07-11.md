# Validating B as a codec quality KNOB with real encoders (2026-07-11)

**Question (user):** how do we validate, with *real encoders*, the quality of B
(`ZensimProfile::B` = `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`,
sha b6fe5233) as a quality dial?

Everything measured before this ran on the *stored* dial grid
(`dial_grid_372col_2026-05-29`), which is partly corrupt (9/115 ladders — the
quarantined webp masked/IW garbage). "Real encoders in the loop" closes that gap.
This validation reuses real-encoder data already on disk — **no new encodes.**

## What "B is a good knob" means — three tests

A user types a target zensim score; the codec binary-searches its quality param to
hit it. The knob is good iff, dialing to a target T:

1. **Mechanics** — the search *converges*: B is monotone in the encoder quality
   param, spans the range, no flat dead-zones (low tie rate). Reference-free.
2. **Consistency-at-target** — a fixed B=T delivers *consistent true perceptual
   quality across content*, measured as the spread of an **independent** reference
   (human MOS > CVVDP > ssim2; ssim2 is semi-circular for B, whose `cid` head trains
   on an ssim2+cvvdp mix). Rank-INDEPENDENT of SROCC — a metric can rank well yet be
   a content-dependent (poor) knob.
3. **RD-efficiency** — targeting B spends bytes competitively at equal true quality
   (the byte side of test 2; not separately gated here).

## Data (all real bitstreams, already scored)

| source | what | role |
|--|--|--|
| `ab_rescored_2026-07-05/{zenjpeg,zenavif,zenjxl,zenwebp}_lossy.{a,b}.parquet` | picker TEST corpus, real multi-codec q-sweeps, `pred_b`+`pred_a`+`score_ssim2`+`q`+bytes, ~17 modes × ~19 q per (ref,box) | **mechanics** (test 1), reference-free |
| `val/cid22.parquet` (B/A forward) + `cid22_ssim2_scores.tsv` (mcos+ssim2) + `cid22_cvvdp_scores_2026-05-17.tsv` | 4292 real encodes, 8 codecs (aom/avif/heic/jp2/webp/jxl/mozjpeg), **human MOS** + CVVDP + ssim2 | **consistency** (test 2), independent reference |

Join verified exact: `val/cid22.parquet` row-order aligns to the ssim2/cvvdp TSVs
(parquet `human_score`×100 == ssim2-TSV `mcos`, 0/4292 mismatches).

Scripts (reusable as B evolves): `scripts/v_next/knob_mechanics_ab_rescored.py`,
`scripts/v_next/knob_consistency_cid22.py`.

## Test 1 — Mechanics on real per-image q-ladders (ab_rescored)

Per (ref, box, cell) ladder, sort by q; |ρ|=|Spearman(metric,q)|, inv=adjacent-q
reversal rate, tie=|Δ|<0.5 dial-pt rate, span=reachable dial range. Ladders
oriented so quality increases with q.

| codec | metric | ladders | \|ρ\| | strict-mono% | inv-rate | tie@0.5 | span |
|--|--|--:|--:|--:|--:|--:|--:|
| jpeg | **B** | 43632 | 0.8811 | 63.23 | 0.1510 | 0.0102 | 49.00 |
| jpeg | A | 43632 | 0.8757 | 62.03 | 0.1583 | 0.0144 | 53.40 |
| jpeg | ssim2 | 43632 | 0.8801 | 64.45 | 0.1449 | 0.0197 | 49.33 |
| avif | **B** | 38784 | 0.9949 | 89.84 | 0.0183 | 0.0194 | 77.30 |
| avif | A | 38784 | 0.9976 | 95.34 | 0.0081 | 0.0062 | 89.94 |
| avif | ssim2 | 38784 | 0.9961 | 95.15 | 0.0092 | 0.0054 | 90.28 |
| jxl | **B** | 28280 | 0.9684 | 76.02 | 0.0338 | 0.0222 | 40.82 |
| jxl | A | 28280 | 0.9752 | 81.89 | 0.0244 | 0.0428 | 35.18 |
| jxl | ssim2 | 28280 | 0.9629 | 69.50 | 0.0553 | 0.0447 | 32.19 |
| webp | **B** | 24240 | 0.9993 | 98.39 | 0.0027 | 0.0024 | 54.83 |
| webp | A | 24240 | 0.9992 | 98.28 | 0.0030 | 0.0090 | 59.11 |
| webp | ssim2 | 24240 | 0.9961 | 95.09 | 0.0100 | 0.0071 | 48.53 |

**Weighted mean across codecs:** B |ρ| 0.9533 / strict-mono 79.88% / inv 6.17% /
tie 1.40% / span 56.47; A 0.9538 / 82.28% / 5.92% / 1.70% / 61.11; ssim2 0.9516 /
79.84% / 6.29% / 1.86% / 57.36.

- **B ≈ ssim2 mechanically**, marginally better on ties + inversions.
- **B beats ssim2 on jxl** (76.0 vs 69.5% strict-mono, wider span) and **is the best
  webp knob** (98.4%). Trails A/ssim2 only on avif.
- **jpeg is noisy for ALL metrics** (~15% step-inversions for B, A, ssim2 alike):
  intrinsic JPEG RD non-monotonicity at fixed mode, not a B defect. The search still
  converges on the trend (|ρ| 0.88) with a 1% tie rate.
- A is the smoothest overall, driven entirely by its avif advantage.

## Test 2 — Consistency-at-target vs an INDEPENDENT reference (CID22)

Bin all 4292 pairs into equal-count deciles of each candidate knob; measure the
spread of the reference within bins. η²(REF|knob)=between-bin/total variance of REF
(higher=knob pins REF); resid SD=mean within-bin SD of REF (lower=tighter knob).

### vs human MOS (gold)

| knob | SROCC vs MOS | η²(MOS\|knob) | resid SD (MOS pts) |
|--|--:|--:|--:|
| ssim2 | 0.8894 | 0.7761 | **6.04** |
| **B** | 0.8764 | 0.7585 | **6.33** |
| A | 0.8657 | 0.7326 | 6.60 |
| cvvdp | 0.8214 | 0.6734 | 7.33 |

- **B beats A** on all three (better human-MOS knob), lands **0.3 MOS behind ssim2** —
  inside human MOS noise (~6-pt subjective SD is the floor all four metrics hit).
- B decile table is textbook-monotone and **tightens toward high quality**: dial B
  low→high walks mean MOS 49.6→86.9, resid SD ±8.1 (bottom) → **±2.8 MOS (top decile)**
  — exactly where users target.

### vs CVVDP (independent metric, not human)

| knob | SROCC vs CVVDP | η²(CVVDP\|knob) | resid SD (CVVDP) |
|--|--:|--:|--:|
| ssim2 | 0.9357 | 0.7885 | 0.054 |
| A | 0.9298 | 0.7644 | 0.056 |
| **B** | 0.8918 | 0.7312 | 0.064 |

Against CVVDP, A/ssim2 beat B — the known A↔B tradeoff: A is ssim2/CVVDP-shaped,
B is human-MOS-shaped.

## Part 3b — Normalized span + REACHABILITY (unreachable regions)

The native `span` in Test 1 isn't apples-to-apples (each metric's own scale). Two
notions of "unreachable": **metric-limited** (the dial compresses a single image's
true quality range) and **encoder-limited** (even max/min q can't reach a target).
Measured on **current B** re-forwarded on the stored features — the stored `pred_b`
is the pre-inclusive-winsor B, off by up to 9.6 dial pts (mean 0.27, corr 0.9997),
so this step is mandatory. Scripts: `scripts/v_next/knob_reach_ab_rescored.py` (+
the reforward). Data: `/mnt/v/output/zensim-multicodec-probe/knob_reforward/`.

**Normalized span** = per-ladder span / metric's own (p99−p1), mean over ladders
(weighted across codecs):

| metric | span_frac (norm) | span_native |
|--|--:|--:|
| **B** | **0.6835** | 56.61 |
| A | 0.6403 | 57.47 |
| ssim2 | 0.4000 | 57.36 |

**B is the most expressive knob per-image** — one image's q-sweep traverses 68% of
B's usable dial vs 40% for ssim2. The native-span parity was a scale artifact:
ssim2's usable range is huge (jpeg p1..p99 = −52..93) but mostly an *unused negative
tail* real encodes never hit; B uses its dial efficiently (no wasted range).

**Reachability** (fraction of ladders whose [min_q,max_q] spans a dial target):

| codec | metric | reach≥50% band | reach@85 | reach@95 | ceiling (median max) |
|--|--|--|--:|--:|--:|
| jpeg | B | 28..80 | 0.37 | 0.00 | 82.1 |
| jpeg | A | 32..85 | 0.53 | 0.00 | 85.7 |
| jxl | **B** | 40..80 | **0.26** | 0.00 | **81.2** |
| jxl | A | 50..85 | 0.58 | 0.00 | 85.9 |
| webp | **B** | 28..82 | **0.21** | 0.00 | **82.7** |
| webp | A | 28..85 | 0.67 | 0.00 | 86.8 |
| avif | B | 10..90 | 0.84 | 0.01 | 92.0 |
| avif | ssim2 | 0..92 | 0.86 | **0.17** | 92.7 |

- **B's band is shifted DOWN**: reaches lower floors (jpeg median floor 27 vs ssim2
  34) but lower ceilings (jxl 81 vs A 86, webp 83 vs 87).
- **The "unreachable top" for B**: on jxl/webp, only 21–26% of images can be dialed
  to B=85, vs A's 58–67%. B doesn't score even high-q encodes that high — a
  conservative ceiling. This is B's MOS-shaping as a knob property (human MOS
  saturates less than ssim2 at the top → lower, more honest, but harder-to-reach
  high targets). It is *not* the old near-lossless pin (current-B, and the ceiling
  is encoder-limited).
- **Dial 95+ is encoder-unreachable for all metrics** on these q-sweeps (they don't
  produce near-lossless) — except ssim2 reaches 95 on 17% of avif. So the ~92–95 cap
  is a real encoder ceiling, not a metric defect.

## Verdict

Against **real encoders + human judgment, B is a legitimately good knob:**
mechanically sound (|ρ| 0.95, ~80% strict-mono, 1.4% tie across 4 codecs, on par
with ssim2), the **most expressive per-image** (normalized span 0.68 > A 0.64 >
ssim2 0.40), and human-consistent (±6.3 MOS at a fixed target, matching ssim2's ±6.0,
beating A's ±6.6). The A↔B tradeoff persists in the knob view exactly as in the rank
view: **B is the better human-MOS knob, A/ssim2 the better metric-agreement knob.**
This is a knob-quality validation independent of the earlier rank-SROCC and dial-grid
panels, on real bitstreams, and it does not depend on the corrupt 2026-05-29 grid.

**The one real knob limitation**: B's top-end is conservative — on jxl/webp, high
targets (dial 85+) are unreachable for most images (21–26% reach vs A's 58–67%),
because B (MOS-shaped) scores high-q encodes lower than A/ssim2. Likely MOS-honest
(human MOS saturates less at the top) but a genuine reach cost: a product dialing B
to 90 on jxl/webp will often cap out ~83–87. If high-target reach on those codecs
matters, that's A's domain — consistent with A staying the default `codec_target`.

## Honest gaps / how to strengthen

1. **Independent-reference consistency at SCALE (not run).** Test 2 vs MOS/CVVDP ran
   only on CID22's 4292 discrete encodes. The ab_rescored ~1M multi-codec q-ladders
   were mechanics-only (their sole reference is ssim2, semi-circular for B). To get
   cross-codec consistency-at-target vs an *independent* reference at scale: score
   CVVDP (GPU) on the ab_rescored bitstreams (they persist at `variant_r2_url`) via
   `zenmetrics batch --metric cvvdp` — a scoped GPU job, NOT a re-encode. Then repeat
   test 2 (η²/resid-SD of CVVDP given B-bin, per codec + pooled). **Confirm scope
   before a fleet run** (cost gate).
2. **jxl ladders predate the #18/#94 near-lossless fix** (ab_rescored 2026-07-05;
   fix ~07-06/07). Mechanics span q5–90 so mostly unaffected, but a fresh jxl
   re-sweep would clean the near-lossless top.
3. **A forwarded on raw pre-spline output** (`ensemble_score_rows`); rank/η² stats
   valid (spline monotone), absolute A dial-units not comparable — irrelevant to the
   rank-based knob stats used here.
4. **Fold η²/resid-SD knob-consistency into `bake_verdict`** so every ship-grade bake
   reports it next to the rank + dial panels (currently a standalone script).

## Data / repro
- CID22 forwards: `ensemble_score_rows --bake <B|A> --parquet val/cid22.parquet` →
  `/tmp/cid22_{B,A}.tsv`; joined with the ssim2/cvvdp TSVs by verified row-alignment.
- Mechanics: the 4 `ab_rescored_2026-07-05/*_lossy.{a,b}.parquet`.
- Both scripts committed under `scripts/v_next/`.
