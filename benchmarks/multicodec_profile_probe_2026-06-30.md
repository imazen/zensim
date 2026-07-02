# Multi-codec ssim2 profile probe — 2026-06-30

Can the `canonical-picker-2026-06-27` dataset (5.74M rows, 7 codec×modes,
372 zensim feat + score_ssim2, clean origin split) train a better Profile::A?
**Yes.** A bare probe already beats the shipped v47 on every codec/HF human-MOS
corpus and cracks the G5 KonJND wall.

## Probe setup
- Train: 1.2M rows stratified (300k each from zenjpeg/zenavif/zenjxl/zenwebp
  lossy `train.parquet`, origins {0,2,4,6,8}), ssim2-decile-stratified for low-q.
- Target: `score_ssim2` (--target-scale 1.0). Held-out val = origins {1,3,5}.
- `zensim_mlp_train --max-features 372 --hidden 128 --per-sample-alpha-head
  --mse-weight 0.6 --ranknet-weight 0.6 --epochs 60 --seed 17 --lr 1e-3
  --minibatch-size 256`. NO monotone-cbc, NO QAT (probe only).
- Held-out-origin ssim2-tracking SROCC = 0.9647.

## A/B vs shipped v47 (bake_verdict, identical harness + 372-col canonical val)

| Corpus | probe | A (v47) | Δ | kind |
|---|--:|--:|--:|---|
| CID22 | 0.8827 | 0.8657 | +0.017 | codec (gold) |
| AIC-3 | 0.7948 | 0.7680 | +0.027 | codec JND |
| AIC-4 | 0.8921 | 0.8854 | +0.007 | neural codec |
| KonJND | 0.5619 | 0.4185 | +0.143 | HF/near-lossless (G5 wall) |
| KADID | 0.6397 | 0.7933 | −0.154 | NON-codec analytic |
| TID | 0.7248 | 0.7927 | −0.068 | NON-codec analytic |

CID22 0.8827 within 0.007 of fast-ssim2 (0.8895). Probe wins all codec+HF
corpora; loses only the non-compression analytic guards (never trained on
blur/noise/geometric).

## Caveats
- Dial broken (G3 inversions 0.28) — no monotone-cbc. Not ship-ready as-is.
- ssim2-only target; 1.2M of 5.7M; 60 epochs. Full-scale + balanced should beat this.

## Next: balanced ship-candidate
multi-codec 5.7M + kadid/tid/konjnd train subsets (recover analytic guards) +
--monotone-cbc + QAT f16 (A's dial correctness) + 372. Scale on Hetzner.
Optional: score 5.7M variants with cvvdp+iwssim for A's proven mix target.

Bakes: /mnt/v/output/zensim-multicodec-probe/probe_372.bin (+verdict.md);
A baseline: A_baseline.verdict.md. Trainer ban-228 guard landed same day.

## Iteration 2: the monotone-vs-rank tension (2026-06-30)

Tried to make the probe ship-quality (recover analytic guards + monotone dial).
Two diagnostics on the same harness (bake_verdict, 372-col canonical val):

| recipe | CID22 | AIC-3 | AIC-4 | KonJND | KADID | TID |
|---|--:|--:|--:|--:|--:|--:|
| probe (multi-codec ssim2, no constraint) | **0.8827** | 0.7948 | 0.8921 | 0.5619 | 0.6397 | 0.7248 |
| + heavy analytic guards (k/t 0.3, kj 0.5) + mono-cbc | 0.7076 | 0.6265 | 0.6729 | 0.4372 | 0.8142 | 0.8168 |
| + mono-cbc-strict only (no guards) | 0.6795 | 0.6166 | 0.6505 | 0.4280 | 0.7627 | 0.7908 |
| shipped A (v47, mono-cbc + v47 recipe) | 0.8657 | 0.7680 | 0.8854 | 0.4185 | 0.7933 | 0.7927 |

**Findings:**
1. `--monotone-cbc --monotone-strict` on the pure-ssim2 multi-codec target
   **craters CID22** (0.88→0.68). Strict mode DROPS 72 features + sign-pins
   300; that constraint fits human-MOS-flavored targets (v47 used such a
   target and got 0.8657) but FIGHTS the ssim2-on-codec ranking. Notably it
   *raises* KADID/TID (the analytic distortions respect the sign-mask better).
2. Heavy analytic guards swamp the 1.2M codec rows (33k analytic rows at
   weight 1.1-combined ≈ 45% of training signal) → also craters codec corpora.
   Guard weights must be ≪ (≈0.03-0.05) so multi-codec dominates.
3. An output spline alone can't fix the probe's q-sweep dial inversions
   (monotone f(non-monotone) is still non-monotone).

**The tension:** multi-codec ssim2 is a strictly better RANK signal (CID22
0.88 > A 0.87) but A's strong correct-by-construction monotonicity caps CID22
with this data. The ship recipe must navigate this — candidates:
- (a) Max-rank + spline dial (V39-style, drop strict-CBC): best CID22, but
  not strong-feature-monotone (loses A's OOD-synthetic robustness).
- (b) Multi-codec as a GROUP inside v47's proven recipe (human_score-norm
  target + cid22_train + QAT), not as the sole pure-ssim2 target — may keep
  both. UNTESTED — most promising.
- (c) Soft monotonicity-reg (no strict, keep all 372 features) + spline.
- The cvvdp+iwssim target (score the 5.7M variants) is orthogonal and may
  help (a) and (b).

## Iteration 3 + correction: safesyn is NOT artificial distortions (2026-06-30)

User correction (verified against `coefficient/examples/generate_zensim_training.rs`
+ the synthetic-pipeline doc): **safesyn = synthetic/CID22-safe SOURCE tiles
(CID22-512, clic2025, kodak, corpus-builder JPEGs) run through real CODECS**
(mozjpeg/zenjpeg/zenwebp at quality levels), scored by ssim2/butteraugli. The
distortions are **codec compression, NOT artificial/analytic** (blur/noise/color/
geometric). "Safe" = CID22-leak-safe, not "artificially distorted".

Recipe (b) [multi-codec + safesyn + light guards + **monotone-cbc**]:
CID22 **0.6808** (A 0.8657) — same crater as mono-only. Adding more codec data
(safesyn) does NOT fix the monotone-vs-codec-ssim2 tension. KADID/TID rise
(+0.02) as before. **Confirmed: monotone-cbc-strict fundamentally caps CID22
on codec-distortion data; safesyn is not the safety ingredient.**

## Corrected ship recipe (the path forward)

The OOD-safety A gets from the strict monotone CONSTRAINT (which craters CID22)
should instead come from artificial-distortion **DATA**:
1. **Generate an artificial-distortion training set** — apply analytic
   distortions (the KADID taxonomy: gaussian/lens/motion blur, white/impulse/
   speckle noise, color shift/quantize/saturate, contrast, jpeg2k-style, spatial
   warp, sharpen) to the CID22-safe synthetic sources, score with ssim2.
   Leak-free (KADID/TID stay held-out guards). Does NOT exist yet — needs a new
   generation pass (Hetzner-able). [Open Q: fresh-generate vs KADID origin-split.]
2. **Train the ship bake**: multi-codec (rank, CID22 0.88) + artificial-distortion
   (OOD safety, learned not constrained) + light analytic guards, **NO strict-cbc**,
   + output PCHIP spline (dial). Hypothesis: high CID22 + OOD safety + working dial.
3. Validate: full panel + **tests/metric_invariants.rs** (blur>identity etc., the
   real OOD-safety gate) + dial monotonicity.
4. Scale on Hetzner (full 5.7M + cvvdp/iwssim mix target); HDR corpus gen in parallel.

## Iteration 4: artificial-distortion DATA works (TID-in-loop) + KADIS-700k (2026-06-30)

**Mechanism confirmed.** multi-codec (rank, 1.2M) + TID2013 (analytic, 3k, weight
1.0) in-loop, **NO strict-cbc**, validate held-out:

| corpus | tid-in-loop | probe (no tid) | A | note |
|---|--:|--:|--:|---|
| KADID (analytic, HELD-OUT) | **0.8662** | 0.6397 | 0.7933 | +0.23 transfer — TID taught analytic safety |
| CID22 (codec gold) | 0.8456 | 0.8827 | 0.8657 | small cost (tid weight 1.0 a bit high; tunable) |
| AIC-3 | 0.7883 | 0.7948 | 0.7680 | ≈ |
| AIC-4 | 0.8580 | 0.8921 | 0.8854 | ≈ |
| KonJND | 0.5072 | 0.5619 | 0.4185 | > A |

So artificial-distortion DATA (not the strict-cbc CONSTRAINT) delivers OOD safety:
held-out KADID 0.64→0.87 while CID22 only dips 0.88→0.85 (vs strict-cbc's crater
to 0.68). The TID weight is the knob (sweep down to recover CID22).

**KADIS-700k = the scaled version.** `/mnt/v/datasets/kadis700k/kadis700k.zip`
(42 GB): 140k pristine refs + `kadis700k_friqa.csv` (700k rows: dist_im,ref_im +
12 proxy FR-IQA scores [ssim,msssim,**iwssim**,mdsi,vsi,fsim,gmsd,sff,scqi,
add_gsim,sr_sim] — NOT SSIMULACRA2). Distorted images are NOT in the zip — the
dist_im name encodes type+level (`<ref>_<distortion>_<level>.png`), regenerated
from refs via the bundled distortion code. **Not integrated:** no extraction, no
generated distortions, no ssim2 score, no 372 features, not in any parquet.

### KADIS-700k integration plan (the scaled OOD-safety group)
1. Extract 140k refs. 2. Generate the 700k distortions (bundled code / our
   `gen_tid2013_distortions.py`, per the friqa.csv assignments). 3. Score ssim2
   (GPU) for target-consistency with the multi-codec group [friqa's iwssim is a
   faster alt]. 4. Extract 372 zensim features. 5. → parquet (f0..f371 +
   human_score=ssim2). Big GPU job → Hetzner.
6. Ship recipe: multi-codec (rank) + KADIS-700k (analytic safety, tuned weight) +
   PCHIP spline (dial), NO strict-cbc. Expect: CID22 ≥ A, KADID/TID ≥ A,
   metric_invariants (OOD) pass, monotone dial. Validate full panel + invariants.

## KADIS-700k monotonic-safety gate + metric-target sweep (2026-07-01)

**New data**: `kadis700k_canonical_gpu_2026-07-01.parquet` (700k rows, 7 GPU
metrics + 372 feats + persisted distorted PNGs, 0 nulls). ssim2 IS present
(`score_ssim2_gpu`). All 7 metrics scored on the SAME regenerated distortions
as the features (no MATLAB/Python drift).

**Safety gate built**: reshaped KADIS test split (source_id%10==9, U-shaped
7/18/25 excluded) → dial-grid schema (image_id=source_id, codec=dist_name,
q=6-severity, f0..f371). `bake_verdict --dial-grid kadis_test_safetygrid.parquet`
(or ZENSIM_DIAL_GRID env) now measures per-ladder monotonicity + boundedness on
held-out artificial distortions.

**Baseline safety (held-out KADIS test)**:
| bake | CID22 | KADIS mono | KADIS range |
|---|--:|--:|---|
| A (v47) | 0.8657 | 0.972 | -29.7..78.5 |
| probe (multi-codec) | 0.8827 | 0.106 | 1.5..152.8 |
| tidinloop | 0.8456 | 0.111 | -5..20 |

**KEY FINDING**: rank-target analytic data (tidinloop) does NOT fix monotonic
safety (0.111, ~= probe's 0.106). A's 0.972 comes from monotone-by-construction
(W1>=0 sign-mask). Global RankNet rarely samples WITHIN a severity ladder, so
rank data never teaches within-ladder ordering. The fix under test: KADIS as an
OOD group whose per-row MSE toward a BOUNDED metric supplies dense within-ladder
ordering + boundedness.

**Metric severity-monotonicity** (which metric best encodes severity ordering):
dssim 98.8% > iwssim 98.2% > ssim2 98.0% > cvvdp 97.9% > zensim 96.6% >
butteraugli-3norm 93.4% > butteraugli-max 91.5%. Bounded metrics
(dssim/iwssim/cvvdp) are the best safety targets; ssim2/zensim plunge to -1834/-151.

**Sweep (running)**: multi-codec(ssim2) + KADIS(target M) for M in
{ssim2,iwssim,cvvdp,dssim}, ± monotone-cbc(soft). Data at
/mnt/v/output/zensim-multicodec-probe/kadis_{M}_{train,val}.parquet +
kadis_test_safetygrid.parquet. Queue: sweep_queue.sh. Results below when done.

### Sweep results (2026-07-01) — held-out KADIS test-split safety dial

Each = multi-codec(ssim2 rank, 1.2M) + KADIS OOD group. Baselines: A CID22
0.8657/mono 0.972; probe CID22 0.8827/mono 0.106.

| variant | CID22 | KADID | TID | AIC3 | KADIS mono | boundedness p5/p95 |
|---|--:|--:|--:|--:|--:|---|
| ssim2_w1 | 0.876 | 0.818 | 0.824 | 0.781 | 0.050 | -5.9 / 22.4 |
| iwssim_w1 | 0.867 | 0.842 | 0.802 | 0.772 | 0.050 | 9.1 / 41.3 |
| cvvdp_w1 | 0.876 | 0.851 | 0.844 | 0.784 | 0.085 | 8.7 / 29.6 |
| dssim_w1 | 0.878 | 0.860 | 0.845 | 0.767 | 0.030 | 10.9 / 59.2 |
| iwssim_cbc | 0.370 | 0.799 | 0.754 | 0.078 | **0.975** | -25.3 / 4.8 |
| codec_cbc | 0.367 | 0.705 | 0.730 | 0.020 | **0.945** | -286 / 3.7 |

**Verdict — the two levers are orthogonal and neither alone suffices:**
- **DATA (KADIS metric target)**: keeps CID22≈0.88 (all above A), RECOVERS KADID
  0.64→0.82-0.86 + TID, FIXES boundedness — but CANNOT fix within-ladder
  monotonicity (0.03-0.09). cvvdp is the best-balanced data target (CID22 0.876,
  KADID 0.851, TID 0.844, tightest positive range) + it's A's proven mix component.
- **ARCH (monotone-cbc soft)**: FIXES monotonicity (0.95-0.97 ~ A) but its global
  sign-mask + α≡1 projection is INCOMPATIBLE with multi-codec ssim2 rank →
  CID22 collapses to 0.37 (worse than canonical-data cbc; this is exactly why A
  trades down to 0.866). Confirms: multi-codec rank REQUIRES non-monotone feature
  combos that global cbc forbids.

**Resolution under test**: targeted within-ladder monotonicity via `--tv-pairs-file`
(448k KADIS-ladder pairs, penalty max(0, pred[harsher]-pred[milder])) — supervises
monotonicity ONLY on real distortion ladders, leaving codec-rank features free (no
global sign-mask). cvvdp target, tv-weight sweep {5,15,40}. Results below.

### TV-pairs + cbc-composition (2026-07-01) — both dead ends; tension is fundamental

**TV-pairs (targeted within-ladder monotonicity) is a NO-OP with --per-sample-alpha-head.**
The TvRegularizer is only wired into the base mlp path; per_sample_alpha_head.rs has
zero tv references. tv-weight {5,15,40} → BYTE-IDENTICAL bakes (md5 32f7eeb6...),
mono unchanged (0.076). To use TV as the targeted monotonicity fix, it must be WIRED
into the per-sample-alpha-head training loop (real trainer feature; not done).

**cbc + KADIS(cvvdp) + codec-weight sweep — craters regardless of weight:**
| codec_w | CID22 | KADID | mono |
|--:|--:|--:|--:|
| 0.0 (kadis only) | 0.379 | 0.689 | 0.987 |
| 0.2 | 0.220 | 0.753 | 0.969 |
| 0.5 | 0.265 | 0.770 | 0.976 |
| 1.0 | 0.242 | 0.754 | 0.974 |

KADIS(cvvdp) is ARTIFICIAL distortions — doesn't teach codec rank, so cbc+KADIS
caps CID22 at ~0.38 at ANY codec weight (adding codec makes it WORSE — the multi-
codec ssim2 fights cbc). cbc's monotonicity only coexists with decent CID22 on
CANONICAL codec data = exactly A (0.866).

## CONCLUSION (2026-07-01): the tension is fundamental with current mechanisms

Multi-codec CID22 (0.88) and held-out within-ladder monotonicity (0.97) are
MUTUALLY EXCLUSIVE:
- Multi-codec rank REQUIRES an unconstrained MLP → non-monotone on OOD distortions.
- Monotone-by-construction (cbc) REQUIRES canonical data + caps CID22 at ~0.866 (A).
- The one targeted middle (within-ladder TV) isn't wired into the α-head path.

**Two ship options today:**
1. **cvvdp_w1** — CID22 0.876 / KADID 0.851 / TID 0.844 / AIC3 0.784, bounded
   (8.7..29.6). BEATS A on every Mohammadi corpus + fixes boundedness; ONLY loses
   OOD within-ladder monotonicity (0.085 vs A 0.972). Bake:
   /mnt/v/output/zensim-multicodec-probe/sweep_cvvdp_w1.bin.
2. **A (v47)** — CID22 0.866, monotone-by-construction (0.972). Safe; lower rank;
   no KADID recovery.

**Principled resolution (code change)**: wire within-ladder TV into
per_sample_alpha_head training so monotonicity is supervised ONLY on KADIS ladders,
leaving codec-rank features free — the only untested path to BOTH. Payoff uncertain
(generalization to held-out ladders unproven).

## UPDATE 2026-07-01: TV wired into α-head — pure-hinge curve + regime crater

The within-ladder TV path is now wired into `train_mlp_per_sample_alpha_head`
(commit 5e6267cf). `--tv-weight` was previously a silent no-op on that head
(dropped at the dispatch). It now fires a within-ladder hinge
`max(0, y_harsher − y_milder)` at each Adam boundary, supervising monotonicity
ONLY on the 448k KADIS adjacent-severity pairs (group 0, offset 0) and leaving
codec-rank features free.

**Pure-hinge tv-weight sweep** (cvvdp target on KADIS + ssim2 on multi-codec;
held-out KADIS mono on the source_id%10==9 safety grid; Mohammadi via bake_verdict):

| tv-weight | mono | CID22 | KADID | TID | AIC3 | raw range span |
|---|---|---|---|---|---|---|
| 0 (cvvdp_w1) | 0.085 | 0.876 | 0.851 | 0.844 | 0.784 | 20.9 (8.7..29.6) |
| 0.5 | 0.457 | 0.876 | 0.803 | 0.737 | 0.791 | 6.2 (5.8..12.0) |
| 1.0 | 0.711 | 0.865 | 0.724 | 0.606 | 0.774 | 4.1 (3.0..7.1) |
| 2.0 | 0.912 | 0.862 | 0.594 | 0.477 | 0.751 | 2.7 (0.5..3.2) |
| 5.0 | 0.995 | 0.848 | 0.409 | 0.325 | 0.719 | 1.9 (0.1..2.0) |
| A (v47 ship) | 0.973 | 0.866 | 0.793 | 0.793 | 0.768 | n/a (has spline) |

**TV genuinely fixes held-out monotonicity** (0.085 → 0.995) and, unlike
monotone-cbc (CID22 0.37), **holds CID22** (0.85–0.88). This is the first
mechanism to get both high mono AND high CID22.

**But pure hinge has NO sweet spot.** Reaching the 0.93 mono gate needs tv≈2–3,
where the cost lands as a clean **regime split**:
- **Codec corpora are ROBUST**: CID22 0.876→0.862 (−0.014), AIC3 0.784→0.751.
  The multi-codec RankNet protects codec rank.
- **Analytic corpora CRATER**: KADID 0.851→0.594 (−0.26), TID 0.844→0.477 (−0.37).
  KADIS/KADID/TID are the same analytic domain; over-constraining KADIS
  monotonicity destroys analytic cross-image rank.
- The **raw range span tracks the KADID crater almost perfectly** (20.9→6.2→4.1
  →2.7→1.9 as KADID 0.85→0.80→0.72→0.59→0.41). The pure hinge is minimized by
  collapsing every ladder flat.

The range collapse itself is **cosmetic** (these α-head bakes carry no output
spline; a ship-time monotone PCHIP maps raw→[0,100] preserving mono + SROCC).
The KADID/TID rank crater is the real blocker — a spline can't fix rank.

**Next: margin hinge** `max(0, y_harsher − y_milder + m)` (landed alongside
the wiring; `--tv-margin`). Forces a minimum per-step gap, holding the ladder
OPEN at high weight. Tests whether the KADID crater is (a) collapse-driven
(margin recovers it) or (b) intrinsic analytic over-constraint (needs per-group
α-gate supervision instead). Sweep w5×{m1.5,m3,m6} + w15×m3 in flight.

### Margin hinge FALSIFIED (2026-07-01)

Sweep at tv-weight 5 with `--tv-margin` {1.5, 3.0} (each forces a min per-step
gap, holding the ladder OPEN):

| bake | CID22 | KADID | TID | mono | raw range |
|---|---|---|---|---|---|
| pure w5 (m0) | 0.848 | 0.409 | 0.325 | 0.995 | 0.1..2.0 (span 1.9) |
| w5 m1.5 | 0.796 | 0.300 | 0.200 | 0.976 | −2.0..5.1 (span 7.1) |
| w5 m3.0 | 0.706 | 0.333 | 0.263 | 0.971 | −4.2..8.0 (span 12.2) |

The margin **opened the range** exactly as designed (span 1.9→7.1→12.2) but made
**everything worse** — CID22 0.848→0.706, KADID stayed cratered ~0.3. This
**falsifies the collapse-driven-crater hypothesis**: the range collapse under
pure hinge was a *symptom*, not the cause. The KADID/CID22 crater is the
monotonicity constraint itself over-writing the analytic feature→score map, and
margin *adds* constraint (a minimum gap on every pair) → more distortion → worse
rank. Mechanism (b) confirmed: intrinsic analytic over-constraint.

**Consequence:** neither low weight (insufficient mono) nor margin (worsens
crater) resolves it. The fix must **supply the analytic rank signal directly**
rather than hope range-preservation recovers it. Next: add kadid+tid as training
groups (direct rank supervision, canonical [0,1] parquets) alongside pure TV, so
the network has explicit analytic rank targets while TV enforces KADIS mono.
KADID/TID then become train==val integrity guards; honest rank holdouts are
CID22 + AIC3; safety is held-out KADIS mono. Sweep dfx_tv{2,3,5} in flight.

### Data-fix (kadid/tid training groups) + TV — 2026-07-01

Add kadid+tid as training groups (canonical [0,1] parquets, direct analytic-rank
supervision) alongside pure TV on KADIS. kadis stays group 0 (TV offset 0).
KADID/TID become **train==val integrity guards** (memorized), so honest holdouts
are CID22 + AIC3 + held-out KADIS mono.

| bake (kadid/tid w=1.0) | CID22 | AIC3 | KADID* | TID* | mono | range |
|---|---|---|---|---|---|---|
| dfx tv2 | 0.798 | 0.729 | 0.938 | 0.958 | 0.977 | 0.5..2.5 |
| dfx tv3 | 0.782 | 0.746 | 0.935 | 0.956 | 0.991 | 0.6..2.3 |
| dfx tv5 | 0.796 | 0.729 | 0.934 | 0.954 | 0.997 | 0.3..1.5 |
(*memorized — trained on)

Direct supervision **recovers KADID/TID** (0.59→0.94) and mono is excellent
(all > A's 0.973). BUT **CID22 collapses to ~0.79** (below A's 0.866) at EVERY
tv-weight — kadid(10k)+tid(3k) at w1.0 get 50% of pair sampling and drown the
codec signal. AIC3 also drops (0.73–0.75 < A's 0.768). The CID22 ceiling is set
by the analytic group weight, not tv-weight. → group-weight sweep (light
kadid/tid 0.1–0.5) in flight.

### The (CID22, mono) Pareto frontier — A appears optimal

Collecting the CID22-vs-mono tradeoff across all mechanisms:

| mono | best CID22 | mechanism |
|---|---|---|
| 0.085 | 0.876 | cvvdp_w1 (no mono constraint) |
| 0.457 | 0.876 | pure TV w0.5 |
| 0.711 | 0.865 | pure TV w1 |
| 0.912 | 0.862 | pure TV w2 |
| **0.973** | **0.866** | **A (cbc masked-monotone-by-construction)** |
| 0.973 | ~0.849 | pure TV ~w4 (interpolated) |
| 0.995 | 0.848 | pure TV w5 |

**At matched monotonicity (mono 0.973), A's cbc (CID22 0.866) DOMINATES pure TV
(~0.849).** A sits on the Pareto frontier; TV is inside it. The only way to beat
A on CID22 is to abandon monotonicity (cvvdp_w1: 0.876 @ mono 0.085). Adding
KADIS monotonicity to ANY high-CID22 bake costs CID22 because the α-head's rank
and pool heads share ONE encoder — TV shaping the pool head for mono distorts the
shared features that feed the rank head. Gate supervision (route codec→rank,
artificial→pool) is the one untested lever that could decouple them, but the
shared encoder limits how much it can separate.

### Group-weight sweep (light kadid/tid) — 2026-07-01

Lower the analytic-group weight so it supervises rank without drowning codec.

| bake | CID22 | AIC3 | KADID* | TID* | mono |
|---|---|---|---|---|---|
| A (ship) | 0.866 | 0.768 | 0.793 | 0.793 | 0.973 |
| gw kadid/tid=0.10 tv2.5 | 0.845 | 0.764 | 0.801 | 0.865 | 0.949 |
| gw kadid/tid=0.25 tv2.5 | 0.807 | 0.705 | 0.888 | 0.919 | 0.954 |
| gw kadid/tid=0.50 tv3.0 | 0.793 | 0.729 | 0.918 | 0.944 | 0.987 |
(*partial memorization — trained on at low weight)

**kadid/tid weight 0.10 is the analytic-crater sweet spot**: KADID recovers to
0.80 (ABOVE A) and TID to 0.865 without collapsing CID22 — it lands at CID22
0.845 / mono 0.949, clearing the 0.93 mono gate with KADID/TID/AIC3 all healthy.
**But A still dominates it** (CID22 0.866 > 0.845, mono 0.973 > 0.949). It's the
best BALANCED TV bake but not a win over A. Raising the analytic weight past 0.10
trades CID22 down monotonically. → codec-weight + hidden-256 levers in flight to
attack the CID22 ceiling / shared-encoder coupling directly.

### Codec-weight sweep (protect CID22) — 2026-07-01 — FAILED

Raise codec-group weight (2×, 3×) to give ssim2 rank more sampling under TV 2.5:

| bake | CID22 | AIC3 | KADID | TID | mono |
|---|---|---|---|---|---|
| cw codec2 tv2.5 | 0.836 | 0.736 | 0.444 | 0.326 | 0.984 |
| cw codec3 tv2.5 | 0.843 | 0.705 | 0.263 | 0.218 | 0.989 |
| cw codec3 kadid/tid0.15 tv2.5 | 0.836 | 0.760 | 0.754 | 0.842 | 0.981 |

Codec-weight-up did NOT lift CID22 — it went DOWN (0.836–0.843 < pure tv2's
0.862). The codec RankNet is already saturated at weight 1.0; adding tv2.5 (> the
tv2.0 pure baseline) dominates the added codec weight. **The CID22 ceiling at
mono≥0.93 is firmly ~0.85, below A's 0.866.** Every weighting lever confirms it.

### Gate supervision RULED OUT by diagnostic — 2026-07-01

Before implementing per-group α-target supervision, checked whether the α gate is
even the bottleneck. In the best balanced bake (gw a0.10), α(x) ALREADY spreads
across the full range: μ=0.533, **min=0.007, max=0.986** — the gate routes some
samples to the pool head (α→0) and some to the rank head (α→1) by itself.
codec_tr srocc=0.935 / codec_va 0.924 at the final epoch (strong codec rank).

So the α gate is NOT stuck — supervising it toward per-group targets would add
nothing. The CID22 ceiling under TV comes from the **shared 372→H→H encoder**:
both heads read the same hidden vector h, so TV shaping the pool head for
monotonicity distorts the features that also feed the rank head. Weighting
(gw/cw) and gate routing can't decouple a shared encoder. The only structural
fixes are MORE encoder capacity (hidden-256, in flight) or SEPARATE encoders per
head (a real arch change). Gate supervision is dropped from the lever list.

### hidden-256 capacity lever — marginal, A stands — 2026-07-01

| bake (hidden 256) | CID22 | AIC3 | KADID | TID | mono |
|---|---|---|---|---|---|
| hd 256 codec1 tv2.5 | 0.844 | 0.738 | 0.558 | 0.396 | 0.952 |
| hd 256 codec2 tv2.5 | 0.852 | 0.738 | 0.402 | 0.267 | 0.985 |
| hd 256 codec1 tv3 | 0.841 | 0.731 | 0.503 | 0.358 | 0.985 |

Doubling encoder width nudged CID22 from ~0.845 (h128) to 0.852 (best, codec2) —
capacity eases the shared-encoder coupling slightly but does NOT reach A's 0.866.
Diminishing returns; hidden-512 would likely add <0.005.

## FINAL VERDICT (2026-07-01): A (cbc + canonical) is Pareto-optimal

Across **6 lever types / 18 trained bakes** — pure-TV weight, anti-collapse
margin, kadid/tid data groups, group-weight, codec-weight, hidden-256 — **no
mechanism reaches CID22 ≥ 0.866 at mono ≥ 0.93.** The (CID22, mono) frontier:

| approach | best CID22 @ mono≥0.93 | mono | note |
|---|---|---|---|
| **A = cbc + canonical** | **0.866** | 0.973 | shipped; Pareto-optimal |
| TV + multicodec (this work) | ~0.852 | 0.95–0.99 | best is hd256/codec2 or gw a0.10 |
| cvvdp_w1 (no mono) | 0.876 | 0.085 | best RANKER, unusable as a dial |
| cbc + multicodec (prior) | 0.37 | 0.95 | projection incompatible w/ aggressive ssim2 |

**Why A wins:** monotonicity via a STRUCTURAL constraint (cbc: W1≥0 + sign mask
+ [0,100]) is more rank-efficient than a SOFT penalty (TV), because cbc
constrains the function class without adding a competing loss, while TV fights
the rank objective during training. On the aggressive multi-codec ssim2 target,
cbc's projection craters (0.37) but canonical data fits it — so **A's
cbc+canonical is the efficient corner.** The α-gate already routes by regime
(α spread 0.007–0.986), so the ceiling is the shared encoder, which capacity
and weighting can't decouple.

**Deliverables from this investigation:**
1. **TV within-ladder monotonicity** wired into the α-head trainer
   (`--tv-weight`, `--tv-margin`) — reusable, was a silent no-op before.
2. **KADIS-700k held-out monotonic-safety eval gate** (source_id%10==9 test
   split → dial-grid → bake_verdict dial panel) — a NEW capability that measures
   OOD within-ladder monotonicity. It CONFIRMS A is safe (mono 0.973).
3. **Best balanced TV bake** (gw kadid/tid=0.10 tv2.5): CID22 0.845 / mono 0.949
   / KADID 0.801 / TID 0.865 — dominated by A on CID22+mono but BEATS A on the
   analytic corpora. A candidate only if analytic-distortion rank matters more
   than codec rank (it doesn't, for the codec-dial product).

**Recommendation: KEEP A.** The KADIS parquet's value is confirming A's safety +
the TV infra, not a better dial bake. The one untested structural lever —
SEPARATE encoders per head (rank encoder free for codec, pool encoder
TV-constrained for artificial) — could in principle preserve cvvdp_w1's 0.876
under mono, but it's a multi-crate change (trainer + arch + zensim runtime + bake
format) with uncertain payoff against an already-optimal A. That is a deliberate
R&D investment for the user to weigh, not a quick sweep.

## CORRECTION PASS (2026-07-01, user-directed xref): the crater was partly misinterpreted data

User: "cratering is usually misinterpreted data. check kadis700k especially against kadik."
Three xref findings invalidate parts of the FINAL VERDICT above:

### 1. TV-pair bug — 6.68% of pairs taught the WRONG order

The 448k TV pairs were built from severity ordering for ALL dist types. But the
signed/U-shaped types (7 color_saturate_hsv, 18 mean_shift, 25 contrast) were
only excluded from the SAFETY GRID, not from the TV pairs. Per the parquet's own
metrics: type 25 = 53.8% inverted (severity is a coin-flip for quality), type 18
= 49.8%, type 23 color_block = 15.9%, type 7 = 9.4%. Overall 6.68% (~30k) of TV
hinge pairs pushed y(harsher) ≤ y(milder) on rows where the training target says
the harsher image looks BETTER — a direct gradient war on the analytic manifold.
Every TV/dfx/gw/cw/hd bake above trained on these poisoned pairs; the KADID
crater conclusions are contaminated. Clean rebuild: exclude 7/18/25 + keep only
step-concordant pairs (cvvdp AND ssim2 non-increasing) → 364,775 pairs, 0.000%
target-inversion (`kadis_tv_pairs_clean.tsv`).

### 2. The safety-grid mono ceiling is 0.980, not 1.0

cvvdp itself (the training target) has a 1.98% step-inversion rate on the
included ladders (ssim2: 2.03%) — real quality is genuinely non-monotone in
severity on ~2% of steps (concentrated in color_block/color_shift/denoise/jp2k,
exactly where A's dial inversions live). So **A's mono 0.973 ≈ the oracle
ceiling**, and the TV bakes' 0.995+ were OVER-shooting (flattening real quality
reversals = rank damage). The right mono target band is ~0.96–0.98; treating
0.99 as better than 0.97 was miscalibrated.

### 3. Full-panel + per-band REVERSES the aggregate-SROCC story (SROCC-only ban vindicated)

Extracting the full Mohammadi panel from the same verdicts:
- **cvvdp_w1 beats A on the ENTIRE panel on 4/6 corpora** (CID22, KADID, TID,
  AIC-3 — every stat: SROCC/PWRC/Z-RMSE/DS-AUC/geo3), ties AIC-4
  (0.875 vs 0.885), and its HQ bands (CID22 B8/B9, KADID B6–B9) are at-or-above
  A's.
- **The genuine HQ-end weakness is KonJND** (JND/visually-lossless anchor): A
  0.419 (already failing G5 ≥0.70), cvvdp_w1 0.310, dirty-TV gw_a010 0.030
  (destroyed). Ship gates must include KonJND-no-regression, which the earlier
  tables omitted.
- Comparison-fairness note: A trained on kadid(0.5)+tid(0.5)+cid22_train(1.5)
  (v47 manifest) — its KADID/TID numbers are train==val, and it had CID22-domain
  human supervision the probe bakes lacked. KADID/TID are honest holdouts ONLY
  for the probe bakes; part of A's CID22 edge may be data, not architecture.

### Corrected experiment matrix (in flight)

R1a/R1b: kadis(cvvdp) + codec + CLEAN TV @ {1.0, 2.5} — isolates the pair bug.
R2: + cid22_train:1.5 (A's data advantage). R2k: + konjnd-dense:1.2 (full v47
data parity minus kadid/tid). All eval: full panel + per-band + KonJND gate +
KADIS safety (target band 0.96–0.98, oracle 0.980), KADID/TID as honest holdouts.

### 4. THE BIG ONE — probe bakes are direction-INVERTED on KADIS (sign-convention artifact)

Per-type dial spans exposed it: A ascends (`blur_gauss −12.8 → 97.7`) but
cvvdp_w1 DESCENDS on the KADIS safety grid (`27.5 → 3.6`) while ASCENDING on the
standard codec dial (3.9 → 119.4) and on human corpora (signed AIC-4 +0.875).
The α-head probe bakes (no anchor/spline/tanh-pin) have **regime-split
orientation** — the trainer's RankNet is distance-convention ("sigmoid
cross-entropy on signed distance"), MSE is score-convention, and without v47's
orientation anchors the loss war settles differently per content regime.

Consequences:
- **cvvdp_w1's "mono 0.085" was a MISREAD**: 91.5% of its safety-grid steps move
  consistently DOWN (descending dial), only 1.02% are true own-direction
  inversions → own-direction mono ≈ 0.99, at the oracle ceiling. The number that
  launched the whole TV campaign was a sign artifact.
- **Every TV bake was fighting the sign, not enforcing monotonicity**: the hinge
  (score-convention: y_harsher ≤ y_milder) PUNISHED the correct distance-shaped
  ordering on ~ALL pairs — hence range collapse, KADID/TID/KonJND craters, and
  "mono improving" (dial flattened toward 0 slope). Clean pairs changed nothing
  because the poisoned 6.68% was noise next to a 100% sign war. All 18+ probe-TV
  bakes are artifacts; the remaining clean-TV queue was killed.
- **cvvdp_w1 is still NOT a dial** — direction-aware, its codec-dial mono is
  0.784 with 68% sub-resolution (flat) steps, and one global monotone spline
  cannot fix a regime-split orientation. It remains a strong RANKER only.
- The safety grid itself is verified correct (q = 6 − severity, exact feature
  match to canonical; ascending q = ascending quality).

### The experiment that was never run (next): v47 recipe + KADIS group

Everything points to injecting KADIS into A's own recipe rather than rebuilding
from probe scratch: v47's cbc + tanh-pin + anchor-spline enforce a consistent,
oriented, calibrated dial BY CONSTRUCTION (dial mono 0.973 ≈ oracle), while the
KADIS data supplies the artificial-distortion coverage that lifted the probe
bakes' rank panel everywhere. The earlier "cbc+kadis craters CID22 to 0.37"
result came from probe-style configs (NO anchor/spline/canonical groups) — the
v47-manifest-with-kadis variant is untested. Plan: copy
`zensim/weights/manifests/v47_strict_qat.toml`, add
`kadis:kadis_cvvdp_train.parquet` at modest weight (0.3–0.5), train, full-panel
+ both dials + per-band vs A.

### v48-kadis @ w0.5 — first result (2026-07-01)

v47 recipe + kadis@0.5 (manifest `v48_kadis_experiment.toml`, QAT + in-pass
spline, all inputs sha-verified):

| | CID22 | KADID* | TID* | KonJND | AIC3 | AIC4 | codec dial | KADIS safety |
|---|---|---|---|---|---|---|---|---|
| A (ship) | 0.8657 | 0.7933 | 0.7927 | 0.4185 | 0.7680 | 0.8854 | mono 0.9747, G1 ✓ (15.0/94.5) | 0.9726 |
| v48 kadis@0.5 | 0.8136 | 0.7801 | 0.7833 | **0.4812** | 0.7285 | 0.8422 | mono **0.9783**, G1 ✓ (20.2/94.5) | **0.9758** |
(*train==val for both)

The KADIS group under the v47 recipe: **KonJND +0.063** (largest gain measured
on the G5/HQ weakness without cratering elsewhere), **both dials at-or-above A**
(the oriented-dial machinery works — no direction split, no crater), but
**CID22 −0.052 / AIC3 −0.040 / AIC4 −0.043**: at w0.5 (≈10% of pair sampling)
the artificial data dilutes codec/human rank under cbc's constrained capacity.
Weight sweep w∈{0.1, 0.25} in flight to find whether the KonJND/dial gains
survive at a weight that doesn't cost CID22.

### v48-kadis weight sweep — trade is real, weight-independent, and unstable (2026-07-01)

| bake (seed 17) | CID22 | KonJND | AIC3 | AIC4 | codec dial | KADIS safety |
|---|---|---|---|---|---|---|
| A (ship) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.9747 / G1✓ | 0.9726 |
| v48 kadis@0.10 | 0.8231 | 0.4802 | 0.7442 | 0.8644 | 0.9783 / G1✓ | 0.9791 |
| v48 kadis@0.25 | **0.6380 ⚠** | 0.4514 | **0.5228 ⚠** | **0.4151 ⚠** | 0.9781 / G1✓ | 0.9751 |
| v48 kadis@0.50 | 0.8136 | 0.4812 | 0.7285 | 0.8422 | 0.9783 / G1✓ | 0.9758 |

1. **The KonJND gain (+0.06) is weight-INdependent** (0.480 @0.10 ≈ 0.481 @0.50)
   — KADIS's near-threshold artificial ladders teach JND discrimination the
   canonical corpus lacks, even at 2% sampling.
2. **The CID22/AIC cost does NOT tune away** — even @0.10, CID22 −0.043. The
   dilution is not proportional to sampling share; it's the cbc-constrained
   capacity re-allocating to the artificial manifold.
3. **@0.25 is catastrophically non-monotone in weight** (CID22 0.638, AIC4
   0.415 while dials stay fine) — a training-instability signature under the
   v47 recipe + kadis, NOT a smooth trade curve. Single-run numbers here carry
   large variance; seed-31 diagnostic at @0.25 in flight. ANY ship decision in
   this family requires multi-seed confirmation.
4. Dials are uniformly at-or-above A across all weights (codec 0.978 G1✓,
   KADIS safety 0.975–0.979): the v47 orientation/cbc/spline machinery absorbs
   KADIS with zero dial risk. The trade is purely on the rank panel.

**Cycle verdict:** v48-kadis does not beat A for the dial product (CID22
regression at every tested weight). Deliverables that stand: the KonJND lever
(+0.06, weight-independent — a candidate ingredient for a KonJND/G5-focused
variant), the corrected science (sign artifact, oracle ceiling, clean pairs),
the KADIS held-out safety gate, and the v48 manifest family for reproduction.

### @0.25 seed diagnostic — instability CONFIRMED; all single-seed deltas unproven (2026-07-01)

| w0.25 | CID22 | KonJND | AIC3 | AIC4 | codec mono |
|---|---|---|---|---|---|
| seed 17 | 0.6380 | 0.4514 | 0.5228 | 0.4151 | 0.9781 |
| seed 31 | **0.8585** | 0.4017 | **0.7874** | **0.8967** | 0.9724 |
| A (ship) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.9747 |

seed31@0.25 lands AT A on CID22 (−0.007) and ABOVE A on AIC-3 (+0.019) and
AIC-4 (+0.011) with healthy dials — while seed17 at the identical config had
collapsed. Run-to-run spread of 0.22 SROCC on CID22 means **every single-seed
delta in the tables above (including "KonJND +0.06" and "CID22 −0.04") is
unproven**. Per experiment-rigor policy the verdict moves to seed-means:
seeds {7,47} × weights {0.10,0.25} in flight (→ 3-4 seeds per weight incl.
existing runs). Note the KonJND gain also flipped at seed31 (0.402 < A) — the
gain may itself be seed-luck; seed-mean will tell.

## FINAL (multi-seed) VERDICT — 2026-07-01: A stays; KADIS-in-recipe is a modest KonJND lever with a real CID22 cost

Seed-means (seeds 17/7/47 @0.10; 17/31/7/47 @0.25):

| (seed-mean ± sd) | CID22 | KonJND | AIC-3 | AIC-4 | codec dial |
|---|---|---|---|---|---|
| A (ship, fixed artifact) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.9747 |
| v48 kadis@0.10 (n=3) | 0.8243±0.017 | 0.4465±0.030 | 0.7360±0.032 | 0.8403±0.045 | 0.9776±0.001 |
| v48 kadis@0.25 (n=4) | 0.7878±0.101 | 0.4475±0.032 | 0.6998±0.120 | 0.7520±0.226 | 0.9759±0.002 |
| v48 kadis@0.25 (n=3, excl. collapsed s17) | 0.8378±0.020 | 0.4461±0.039 | 0.7588±0.027 | 0.8643±0.029 | 0.9760 |

1. **CID22: A wins at seed-mean** (−0.041 @0.10 ≈ 4 SEM; −0.028 @0.25-excl ≈ 2.3
   SEM). Even the best of 7 runs (0.8585) is below A. The cost is real, not
   seed noise.
2. **KonJND: the gain survives but halves** — seed-mean +0.028 (~1.6 SEM,
   positive in 6/7 runs): suggestive, modest, not the single-seed +0.06.
3. **AIC-3/AIC-4: no reliable gain** (at/below A at both weights).
4. **Dials: uniformly at-or-above A with tiny variance** (codec 0.976–0.978
   G1✓, KADIS safety 0.975–0.979) — the v47 machinery absorbs KADIS at zero
   dial risk, at any weight, any seed.
5. **Training-collapse hazard: 1/7 runs** (w0.25 s17: CID22 0.638) — recipe+
   kadis has a real instability mode; multi-seed is mandatory for any future
   work in this family.

**A remains the ship.** The KADIS parquet's proven values this cycle: the
held-out monotonic-safety gate (A ≈ 0.980 oracle ceiling), the clean TV-pairs
file, the corrected science (sign artifact + poisoned pairs + full-panel
discipline), and a modest KonJND/G5 lever (+0.03) available to a future
KonJND-focused variant that deliberately accepts a small CID22 trade.

## Profile-A reproduction check — FAILS functionally (2026-07-01, user-directed)

`v47_repro_check.toml` (= shipped manifest, output redirected) against current
main: all 8 documented inputs sha-verify and load — **docs are input-complete**
— but the result is NOT A:

| | CID22 | KADID | TID | KonJND | AIC3 | AIC4 | dial mono | bytes |
|---|---|---|---|---|---|---|---|---|
| A (ship 2026-05-27) | 0.8657 | 0.7933 | 0.7927 | 0.4185 | 0.7680 | 0.8854 | 0.9747 | 27,316 |
| repro (main 2026-07-01) | 0.7376 | 0.7795 | 0.7841 | 0.4309 | 0.6228 | **0.5459** | 0.9704 | 57,207 |

Diagnosis: the bake structure matches (3 layers f16/f16/f32 + spline, flags=3)
but the payload is 2× larger → **trainer code drift since 2026-05-27** (packing/
zerobias behavior changed). Same seed + same data + different code = a different
random run — and this one landed in the same collapse basin the v48 sweep
exposed (AIC4 craters, dials fine; cf. v48 w025 s17). Two compounding facts:
1. **v47 is not currently reproducible from main + manifest.** The recipe's
   *inputs* are fully documented; the *trainer* is not version-pinned. The
   manifest should record the trainer git commit (v47's was ~2026-05-27) and
   reproduction should check out that commit.
2. **The recipe family has a known instability basin** (2/9 runs of
   v47-recipe-class trains this cycle collapsed) — even a correct-code repro
   needs multi-seed.

Action items recorded: (a) add `trainer_commit` to manifest schema + backfill
v47's; (b) reproduce at the recorded commit to separate drift from
non-determinism; (c) never retrain-and-swap A from a single run.

## HQ-zone (75–100) instrument v1 — built + first results (2026-07-01)

New artifacts (probe dir): `hq_codec_grid_2026-07-01.parquet` (7,388 cells =
~1,055 images × 7-q curves, webp vp8-m2_def + avif s6-noqm-420, 372 feats, dial-
grid schema) + `hq_codec_refs_2026-07-01.parquet` (per-cell butteraugli-max/p3 +
cvvdp + dssim + iwssim + ssim2 + zensim from the 2026-06-24 GPU corpus). New
`ZENSIM_DIAL_PRED_OUT` env on bake_verdict dumps per-cell predictions over ANY
dial grid → external joins vs reference metrics.

**Zone rank (SROCC vs cvvdp / vs −butteraugli-max), by ssim2 zone:**

| zone | n | A (ship) | v48s31 | fast-ssim2 |
|---|---|---|---|---|
| <70 | 2950 | 0.832 / 0.667 | 0.852 / 0.689 | 0.820 / 0.690 |
| 70–85 | 2567 | 0.719 / 0.519 | **0.777 / 0.565** | 0.540 / 0.542 |
| 85–100 | 1871 | 0.720 / 0.655 | 0.695 / **0.749** | 0.479 / 0.694 |

**HQ step-agreement** (refs agree a q-step visibly improved — butter −10% AND
cvvdp +0.05 — does the candidate increase?): A 99.7/98.7/98.6%, v48s31
99.8/99.7/**100%**, ssim2 99.9/100/**92.9%** across <70 / 70–85 / 85–100.

Findings:
1. **ssim2's HQ saturation is now measured on codec content**: cvvdp-agreement
   0.82→0.54→0.48 across zones; misses 7% of visibly-better steps at 85–100.
   zensim (A) already beats ssim2 in both HQ zones on cvvdp-rank.
2. **The kadis-trained v48s31 is the best HQ-zone metric measured** (70–85
   cvvdp-rank 0.777; 85–100 butter-rank 0.749; perfect step-agreement) —
   KADIS's near-threshold ladders teach exactly the 75–100 discrimination codec
   targeting needs, corroborating the KonJND +0.03 lever.
3. Zone-consistency gating (min-over-zones) is now measurable; per-zone
   recalibration (piecewise/spline) is legitimate since the dial spline is
   already monotone-rank-preserving — the METRIC must rank within every zone,
   scale is calibration's job.

v2 plan: add zenjpeg/zenpng (R2 sidecars), the 2026-06-24-cpu 477k cells, more
knobs per codec, KADIS severity-1 near-threshold pairs (140k, 7 metrics), and a
`zone_consistency` summary in bake_verdict (min-over-zones panel + step-agreement
gates) so every train auto-reports it.

## Profile-A reproduction, part 2: root-cause archaeology (2026-07-01)

**What made A (exact provenance):** trainer tree ≈ `e9442678` (parent of ship
commit d13ad6b3/1fd645a7, 2026-05-27 11:44), manifest `v47_strict_qat.toml`,
mechanism commits c5c5aed7 (QAT STE) + 310c6aac/742e8a73 (spline on
projected+quantized net) + 55df657b (native f16 + compress). No training log
was preserved; the bake carries no build-commit metadata.

**Drift enumerated (ship → main):** only 7 trainer-touching commits — #40
47aff783 (trainer + bake-emit for hidden=1; the emit change is benign for
leaky=0.01), #41 b1872f7b (soft-monotone-keep-72), c5da5410 (ban-228), fmt/
clippy passes, + the TV wiring (inactive without --tv-pairs-file). One of the
#40/#41 trainer-side changes is the likely 27→57KB packing/behavior drift;
the pinned-tree rerun (in flight) separates code drift from non-determinism.

**SECOND provenance break — a canonical INPUT was rewritten in place after
ship:** the original manifest records konjnd-dense-norm.parquet sha
`5595a922…`; the on-disk file is `49b65c48…` — the 2026-05-28 column-alias
rewrite (active_mix_norm) REPLACED the file one day after v47 trained, the
current manifest's sha was silently updated, and ALL THREE mirrors (local, R2,
Tower — Tower copy dated May 28 02:30) hold the post-rewrite bytes. The
original input bytes are unrecoverable. Data-equivalence is likely (the
rewrite added alias columns; trainer reads human_score + f0..f371 by name) but
unprovable at the byte level. **Frozen-canonical rule violated in practice:
"adding a column" rewrote a released training input in place across every
mirror.** Pinned rerun uses --manifest-allow-sha-drift for this one input.

**Reproduction protocol gaps to fix (all three bit us):**
1. Manifests must record `trainer_commit` (+ the trainer should stamp its
   build commit into bake metadata).
2. Canonical inputs are IMMUTABLE once a ship-bake references them — schema
   additions create a NEW dated file, never rewrite in place.
3. Preserve the training log next to the bake (epoch-0 line = cheap
   determinism oracle).

## Post-CID22 human-alignment datasets (zenpapers survey, 2026-07-01)

From `zenpapers/docs/iqa_datasets_catalog_2026-05-27.md` + dataset pointers
(94 subjective-IQA papers in corpus). Post-CID22 (2023+) sets relevant to
compression + the 75–100 zone, by usefulness to us:

| dataset | year | what | local? | zone fit |
|---|---|---|---|---|
| **JPEG-AI-SDR25** (arXiv 2504.06301, QoMEX'25, Jenadeleh/Sneyers) | 2025 | high-fidelity SDR JND; 5 src × 10 JPEG-AI levels; **85k BTC + 10k PTC raw triplets** | ✓ `/mnt/v/datasets/jpeg-ai-sdr25/` | ★ near-threshold = 75–100 zone; NOT yet in our panel |
| **AIC-HDR2025** (arXiv 2506.12505) | 2025 | HDR (PQ10) JND, 34,560 triplets | ✗ still unreleased (checked 2026-07-01; repo README-only 9 months post-QoMEX) | HDR eval when released |
| **AIC-3 BTC/PTC raw** (dataset-BTC-PTC-24) | 2024 | **419,760 raw triplets** behind CID22/AIC-3 JND | ✓ `/mnt/v/datasets/aic3-btc-ptc/` | ★ pairwise-ranking TRAINING data beyond reconstructed MCOS |
| JPEG-AIC-4 sample | 2024-25 | 5 src × 305 PTC imgs, 6 codecs, JND | ✓ (already in panel, n=300) | already used; full set committee-held |
| UHD-IQA (Hosu) | 2024 | high-res authentic, NR-oriented | partial | weak fit (NR, not codec-FR) |
| PIPAL | 2020 | 1.13M judgments, GAN/SR distortions | ✓ zips | popular but SR-focused; low codec fit |
| AGIQA-3k / AIGCIQA2023 | 2023 | AI-generated image quality | pointers | different domain |

**Actionable:** (a) SDR25 → JND reconstruction (same BTC/PTC protocol as
AIC-3/AIC-4; jpeg-aic reconstruction code) → decode → 372-feat extract → add as
`sdr25` val corpus in bake_verdict — a REAL post-CID22 near-threshold human
anchor for the 75–100 zone. (b) AIC-3 raw triplets → per-triplet pairwise
training signal (RankNet-native, no reconstruction loss) — the highest-value
untapped HUMAN training data in-house. (c) Re-check AIC-HDR2025 quarterly.

## ✅ Profile A REPRODUCED BYTE-IDENTICALLY at the pinned tree (2026-07-01)

Building the trainer at `e9442678` (ship-commit parent) and running
`v47_strict_qat.toml` (output redirected; `--manifest-allow-sha-drift` for the
single documented konjnd rewrite) produced **byte-identical** output:
`sha256 d0ef7a30… , 27,316 bytes == the shipped bake`.

This settles everything the failed main-tree repro raised:
1. **The recipe + docs are complete and correct** — nothing was forgotten.
2. **Training is DETERMINISTIC** — same code + data + seed → same bytes. The
   v48-family "seed variance" is real sensitivity to the seed, but each run is
   exactly reproducible; A was not "a lucky run" in any unreproducible sense.
3. **The konjnd 2026-05-28 alias-rewrite is proven data-equivalent** (byte-
   identical bake trained through the rewritten file).
4. **The main-tree failure is 100% trainer code drift** — one/some of the 7
   post-ship trainer commits (#40 47aff783 / #41 b1872f7b / c5da5410 / fmt /
   clippy / TV wiring) changed training behavior. Bisect step 1 (build at
   #40) in flight in the `zensim--v47pin` workspace.

**Protocol landed (same day):** `[training].trainer_commit` in the manifest
schema — the trainer compares it against runtime `git rev-parse HEAD` and
fails loud on mismatch (same `--manifest-allow-sha-drift` override), with the
workspace-pin instructions in the error. v47's manifest backfilled with the
proven commit + provenance notes. Current main + v47 manifest now fails
correctly instead of silently producing a drifted bake.

**Improvement path unblocked:** "improve A" experiments (e.g. v48-kadis
recipe deltas) should branch from the pinned tree — or from main once the
bisect identifies + reverts/fixes the behavior-changing commit — so every
candidate differs from A by exactly the intended recipe delta, not by hidden
trainer drift.

## v49 WAVE-1 — improvement campaign on the FIXED trainer (2026-07-01)

With main reproducing A byte-identically, four one-delta candidates (v47 recipe,
seed 17, trainer_commit-gated). **The prior v48 "kadis costs CID22" is void —
it was the #40 init bug:**

| (seed 17) | CID22 | KonJND | AIC3† | AIC4† | KADID* | codec dial | KADIS safety | HQ 70-85 / 85-100 |
|---|---|---|---|---|---|---|---|---|
| A (ship) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.7933 | 0.9747 G1✓ | 0.9726 | 0.719 / 0.720 |
| k025 = +kadis@0.25 | 0.8622 | 0.4231 | **0.7930** | **0.9139** | 0.8142 | 0.9755 G1✓ | **0.9806** | **0.762** / 0.674 |
| k05 = +kadis@0.5 | 0.8539 | 0.4290 | **0.7948** | 0.9073 | 0.8068 | **0.9783** G1✓ | 0.9771 | 0.762 / 0.699 |
| c025 = +codec@0.25 | 0.8015 | **0.5027** | 0.7082 | 0.8081 | 0.7648 | 0.9726 G1✓ | 0.9789 | 0.689 / **0.761** |
| c05 = +codec@0.5 | 0.8374 | **0.5049** | 0.7571 | 0.8733 | 0.8020 | 0.9781 G1✓ | 0.9806 | 0.739 / 0.748 |
(†honest holdouts; *train==val for all rows incl. A; HQ = cvvdp-rank on the
hq_codec_grid; ssim2 reference: 0.540 / 0.479)

1. **k025 is a genuine A-challenger**: CID22 −0.0035 (noise), AIC-3 +0.025 and
   AIC-4 +0.029 on the honest holdouts, KADIS safety 0.9806 (> A, near the
   0.980 oracle), codec dial pass, HQ 70-85 +0.043. The kadis group at 0.25
   under the CORRECT init costs nothing and lifts the compression holdouts.
2. **codec-millions is a KonJND/HQ-85-100 lever**: c05 KonJND 0.5049 (+0.086 —
   largest measured; still below the 0.70 G5 floor) + HQ 85-100 0.748, at
   CID22 −0.028.
3. The deltas are complementary (kadis→AIC/dial/70-85; codec→KonJND/85-100) →
   wave-2 = combo (kadis 0.25 + codec {0.25, 0.5}) + k025 multi-seed (7, 31).

## v49 WAVE-2 — combos + k025 multi-seed (2026-07-01/02)

| (fixed trainer) | CID22 | KonJND | AIC3 | AIC4 | codec dial | KADIS safety | HQ 70-85/85-100 |
|---|---|---|---|---|---|---|---|
| A (ship) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.9747 | 0.9726 | 0.719/0.720 |
| k025 s17 | 0.8622 | 0.4231 | 0.7930 | 0.9139 | 0.9755 | 0.9806 | 0.762/0.674 |
| k025 s7 | 0.8429 | 0.4309 | 0.7642 | 0.8676 | 0.9736 | 0.9769 | 0.748/0.760 |
| k025 s31 | 0.8379 | 0.5135 | 0.7548 | 0.8533 | 0.9753 | 0.9786 | 0.737/0.777 |
| combo25 (k.25+c.25) s17 | 0.8350 | 0.4891 | 0.7583 | 0.8633 | 0.9755 | 0.9802 | 0.735/0.765 |
| combo50 (k.25+c.50) s17 | 0.8041 | 0.4721 | 0.6933 | 0.7631 | 0.9832 | 0.9756 | 0.695/0.780 |

**Honest verdict:** k025 seed-mean (n=3) = CID22 0.848±0.013 (−0.018 vs A),
AIC3 0.771 ≈ A, AIC4 0.878 ≈ A, KonJND 0.456 (+0.037), safety 0.979 (+0.006),
HQ better both zones on 2/3 seeds. The s17 numbers were partly seed-favorable
AND the training-side val does NOT select s17 (s31 wins val geomean3 0.9135)
— so s17 cannot be honestly cherry-picked. k025 is a TRADE (safety/KonJND/HQ
up, CID22 −0.018), not yet a dominating win. Combos: KonJND gain persists
(0.489) but CID22 cost grows — the probe-sample codec group is the weak link.

**Next lever: the corpus itself.** The probe codec sample (1.2M rows, no
provenance) predates the canonical 2026-06-27 picker datasets, which now total
**5,742,660 rows** (avif + jxl-lossy landed): jpeg 1.48M, avif 1.51M, jxl-lossy
1.42M, webp 944k+40k, jxl-lossless 270k, png 76k — all with feat_0..371 +
score_ssim2 + score_zensim + provenance + origin splits. Building
`bigcodec_5p7M_2026-07-02.parquet` (f0..f371 + human_score=clamp(ssim2/100,0,1))
→ wave-3: v47+bigcodec@0.25 and kadis.25+bigcodec.25, seed 17 first.

## v50 WAVE-3 — the 5.7M canonical corpus (2026-07-02)

`bigcodec_5p7M_2026-07-02.parquet` (sha 1ef6be99…, 5,742,660 rows = canonical
picker 2026-06-27, all 7 datasets × all splits, human_score=clamp(ssim2/100,0,1)):

| seed 17 | CID22 | KonJND | AIC3 | AIC4 | dial | safety | HQ 70-85/85-100 |
|---|---|---|---|---|---|---|---|
| A (ship) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.9747 | 0.9726 | 0.719/0.720 |
| big025 = v47+bigcodec@0.25 | 0.8610 | **0.4715** | 0.7820 | 0.8983 | 0.9702 | 0.9712 | 0.722/0.647 |
| kb25 = +kadis@0.25+bigcodec@0.25 | **0.6441 ⚠** | 0.4614 | 0.5344 ⚠ | 0.4205 ⚠ | 0.9683 | 0.9740 | 0.572/0.651 |

1. **big025 is the most balanced single-delta candidate measured**: KonJND
   +0.053 (the clean jpeg-heavy corpus delivers most of the probe-sample's
   KonJND lever at a tenth of its CID22 cost), AIC-3 +0.014 / AIC-4 +0.013,
   CID22 −0.005 (within seed noise), dial/safety ≈ A. Seeds 7/31 in flight.
2. **Its one regression — HQ 85-100 rank 0.647 vs A 0.720 — is the predicted
   ssim2-saturation amplification** (SSIMULACRA2 README: the 85/90 anchors are
   in-place/flicker JND grades, and per our HQ instrument ssim2 ranks that
   band at cvvdp-agreement 0.48). More ssim2-labeled HQ rows teach the
   saturation. Fix direction for wave-4: in the ≥0.85 band supervise from
   cvvdp/butteraugli (2026-06-24 GPU corpus, 180k cells) and/or human JND
   (KonJND, SDR25) instead of duplicating ssim2 labels.
3. **kb25 collapsed** (CID22 0.644/AIC4 0.42, dials fine) — the instability
   basin persists on the FIXED trainer when group count grows (7 groups),
   seed-17. **Worse: its val(geomean3) looked healthy (0.909) — checkpoint
   selection is BLIND to this failure mode** because the val groups are
   train==val (memorization masks holdout collapse). Eval-gap action: add a
   truly-held-out val group (e.g. a codec-va slice or KADIS test slice) to the
   recipe's val set so collapse is visible to selection/early-stop.
4. **SSIMULACRA2 provenance note (README, read 2026-07-02)**: ssim2 was tuned
   on CID22(201/250 refs)+TID2013+KADID+KonFiG via Nelder-Mead. Our CID22-49
   val refs were held out of that tuning (ssim2 0.8854 there — fair
   comparison); KADID/TID are fully in-sample for ssim2 — never scoreboard
   corpora against it.

## Contamination audit: imazen-26 origins vs CID22-49 refs — CLEAN (2026-07-02)

dHash-64 audit (`check_holdout_overlap`) of all imazen-26 origin images (the
source corpus behind every canonical-2026-06-27 row → bigcodec_5p7M) against
the 49 CID22 validation refs: **1,051/1,067 hashed (16 decode failures on odd
screen PNGs), minimum Hamming distance d=12, ZERO flags at the strict d≤10
threshold.** The corpus is perceptually disjoint from the gold holdout.
The d=15–16 screening tail (51 sources) is the documented flat/graphic
false-positive mode (generated starburst/voronoi line-art ↔ Semarang city
logo; LoC scan pages ↔ photos) — informational only per the 2026-05-14 dHash
policy (d≤16 requires eye review; nothing at the auto-relevant threshold).
Report: /mnt/v/output/zensim-multicodec-probe/{imazen26_vs_cid22_dhash_t16.tsv,
dhash_audit_t16.log}. This clears assumption (e) from the 2026-07-02
assumption inventory: big025-class candidates do not train on CID22-adjacent
content.

## v51 — digit-split corpus + HELD-OUT VAL SELECTION (2026-07-02)

First recipe generation on the locked DATA_SPLITS foundation: bigcodec
restricted to TRAIN-digit origins (2,946,036 rows), selection val includes two
truly-held-out groups (val-digit origin sample 147k + KADIS %10==8 70k).

| | CID22 | KonJND | AIC3 | AIC4 | codec dial | KADIS safety |
|---|---|---|---|---|---|---|
| A (ship) | 0.8657 | 0.4185 | 0.7680 | 0.8854 | 0.9747 | 0.9726 |
| v51 s17 | 0.8488 | 0.4926 | 0.7675 | 0.8808 | 0.9762 | 0.9706 |
| v51 s7 | 0.8340 | 0.5281 | 0.7535 | 0.8616 | 0.9694 | 0.9690 |
| v51 s31 | 0.8509 | 0.3921 | 0.7770 | 0.8747 | 0.9724 | 0.9601 |
| **mean (n=3)** | **0.8446±0.009** | 0.4709±0.071 | 0.7660±0.012 | 0.8724±0.010 | ~0.973 | ~0.967 |

1. **Zero collapses in 3/3 seeds and the tightest CID22 seed-spread measured
   (sd 0.009** vs v50's 0.101 raw / 0.020 excl-outlier) — held-out-val
   checkpoint selection is doing exactly what §5 predicted. Selection val
   (geomean incl. held-out groups) now sits at 0.84-0.86, honestly below the
   train==val era's 0.91 — the number finally means something.
2. The recipe itself remains a TRADE vs A: CID22 −0.021 (real at ~2.3 sem),
   KonJND +0.05 (seed-noisy, sd 0.071), AIC ≈ A, dials ≈ A. Consistent with
   v50's picture: bigcodec@0.25 (ssim2-labeled) buys KonJND, costs ~0.02
   CID22. The supervision-design fix (multi-metric HQ-band labels, wave-4,
   awaiting the metric backfill) is where the CID22 cost should shrink.
3. Cells ran 9.5-11.5 min at FULL per-epoch eval — the earlier "80 min/cell"
   read was wrong (a clock misread while three jobs contended). The
   group_eval_cap speedup claim is being re-measured cleanly (±cap A/B on the
   idle box) and the ITERATION_PROTOCOL numbers will be corrected to match.

## Data-quality catch: 22.2% duplicate rows in the 5.7M canonical corpus (2026-07-02)

validate_parquet's C10 spot-check flagged the hqfill-combined corpus; full
quantification: **1,290,240 of 5,804,833 rows (22.2%) are duplicates** on
(ref_basename, human_score, f0, f1) — present in the BASE canonical-2026-06-27
data, not the 62k hqfill append. Cause: `modes_full` knob sweeps where knobs
are NO-OPS (cf. the 2026-06-28 knob-ablation finding that avif broad axes
carry no RD value) — different `knob_tuple_json`, byte-identical encodes,
identical features + scores.

Consequences:
1. Every bake trained on bigcodec (v49 c-cells, v50, v51) overweighted
   no-op-knob codec cells ~1.28×. Results stand as measured but are noted.
2. **The canonical picker datasets carry the same duplicate mass** — flagged
   for the picker-training pipeline owners.
3. Fixed: `bigcodec_hqfill_dedup_2026-07-02.parquet` (+ digit splits) dedups
   on the 4-tuple; validator C10 upgraded to a sampled-rate gate (<1%).
4. Training corpora built from sweep data MUST dedup by content, not by
   (image, q, knob) key — knob no-ops make the key non-unique in content
   space. Added to the standing corpus-build contract.
