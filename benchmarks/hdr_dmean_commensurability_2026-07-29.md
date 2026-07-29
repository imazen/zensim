# CSFW tier-1 — the HDR cross-route commensurability claim: **NO MEASURABLE VALUE — recommend CLOSING the family** (2026-07-29)

**VERDICT (Q4): (b).** The one claim left alive after the G6 SDR LOO FAIL
(`csfw_g6_loo_2026-07-29.md` caveat #1 — "DMEAN-only the shape worth
re-testing" on the HDR side) was priced here with pre-registered
measurements on real human-labeled HDR data, and it does not survive:

- **Q1 (UPIQ, zero-fit):** the weighted lanes improve NEITHER within-study
  SROCC (W_DMEAN is *worse* than its unweighted twin on Narwaria at all 4
  scales, −0.015..−0.019) NOR cross-study scale alignment (the
  within-vs-pooled gap does not shrink: +0.060 unweighted → +0.058..+0.067
  weighted). Registered lane-level improvement bar: not met at any scale.
- **Q3 (registered transfer probes, the head-level read):** adding the 4
  DMEAN lanes to a 944 head *degrades* the out-of-manifold HDR stratum —
  Narwaria SROCC 0.7688 → 0.7412 (**Δ −0.0273, 10k-paired-bootstrap
  p(Δ≥0) = 0.0000**), Korshunov flat (−0.0006, p 0.18), with SDR-side CV
  identical (0.9363 vs 0.9361) — the head can't even use the lanes
  in-domain. The full 12-lane block is the same story (nar −0.0234,
  p 0.0000). This is the G6 signature reproduced on the HDR side: the
  lanes get absorbed to the model's detriment.
- **Q2 (V3 construction):** the G1 lane-level cross-route win REPRODUCES
  exactly at tip (DMEAN 0.850 → 0.918/0.952/0.942/0.878) — the core
  mechanism is real — but it does not propagate to ANY consumer: probe-head
  cross-route score SROCC is 0.9693 (944) vs 0.9681 (956) vs 0.9678
  (944+DMEAN) — no improvement, sign mildly negative, noise-scale at n=90.

**Recommended ledger disposition: f956..f979 (chroma tiers + tier B) FREED;
tier-1 (f944..f955) stays landed-but-default-OFF as a complete negative
result** — SDR LOO harmful (G6), HDR within-study harmful-or-neutral (this
study), cross-route lane win real but valueless to every consumer tested.
Stage-2 / chroma-tier work on this family should not proceed; the A6
(append3) closure is the model.

## Protocol (pre-registered before any number existed)

`/mnt/v/output/zensim/hdr-dmean-2026-07-29/PROTOCOL.md`, registered at
build_commit **c4632d6257b1** (= `7bfd511d`/`2eebd76a` feature code +
fmt-only + examples-only harness commits; nothing that touches feature
values). Harness landed in-tree: `examples/upiq_features_extract.rs` (UPIQ
full-vector extraction, both strata, both routes — the chunk-2 V4
pair-enumeration protocol verbatim) + `ZENSIM_CSFW_DUMP` on
`hdr_sdr_consistency`.

Data (all local, none substituted): UPIQ HDR 380 EXR pairs (Narwaria
n=140 wavelet/JPEG2000, Korshunov n=240 JPEG-XT/DCT; validation tier,
never fit on), UPIQ SDR 3,779 PNG pairs (TID2013+LIVE, same unified JOD
scale — train side of the transfer probe only), the chunk-2 V3 ladder (10
aic3 refs × 9 distortions, both routes, n=90). The imazen-26-derived HDR
sets were deliberately NOT the fit vehicle: they are HDR-route-only (no
cross-route content) and `bhdr_improvement_split_lineage_2026-07-12.md`
§8.15 established synthetic-HDR→UPIQ head transfer fails family-wide — a
null through that vehicle would be unattributable. Split hygiene: SDR
train content (TID/LIVE, 38 merged groups incl. the 19 TID↔LIVE repeats)
is fully disjoint from HDR eval content (verified: no
`repeating_content_id` touches Narwaria/Korshunov); the imazen-26
source-id rule is n/a (no imazen-26 rows used). UPIQ-HDR look budget spent
here: 3 registered looks (one per feature set), zero selection on the HDR
axis.

**Gates (all PASS):** E1(956) vs E2(944) first-944 **bit-equality 380/380
on the EXR path** (the tier-1 V2 guarantee, now verified on UPIQ);
`score228` bit-equal 380/380 across 944/956/control — the chunk-2 V4
protocol readout **reproduces the recorded 0.7145 exactly** (0.7145259);
the E4 run reproduces the G1 lane table and V3 headline stats digit-for-digit.

## Q1 — UPIQ within-study decomposition (chunk-2 residual #5, landed)

Readout (fixed V0_2/228, the chunk-2 V4 protocol — identical at 944 and
956 by construction, verified bit-equal):

| leg | pooled | narwaria | korshunov |
|---|--:|--:|--:|
| **streamed route (this, = chunk-2's 0.7145)** | **0.7145** | **0.7145** | **0.9456** |
| zensim A (old PU path, §8.1) | 0.6935 | 0.7173 | 0.9086 |
| shipped BHdr (§8.13, PU-linear 372) | 0.7536 | 0.7834 | 0.9175 |
| structural family (HDR-VDP-2 / PU-iwssim, §8.1) | ~0.81 | ~0.88 | ~0.95 |

The streamed route's +0.021 pooled over zensim A decomposes as: Korshunov
ranking +0.037 (0.9456 — above shipped BHdr, near cvvdp's 0.9686), Narwaria
flat (0.7145 vs 0.7173), remainder cross-study alignment. The G2 gate's
pre-registered within-study baseline is hereby **nar 0.7145 / kor 0.9456**.

Per-lane, zero-fit, |SROCC| vs JOD (all lanes LIVE on both strata — std
0.13..0.34 for DMEAN, 2.7e-2/6.6e-2 CGAIN/CLOSS pooled; no aic3-style
dead-lane caveat applies). W = weighted, U = unweighted twin; gap =
mean(within-study) − pooled (bigger = worse cross-study alignment):

| scale | U\|S\|nar | W\|S\|nar | U\|S\|kor | W\|S\|kor | U gap | W gap |
|---|--:|--:|--:|--:|--:|--:|
| DMEAN s0 | 0.168 | 0.153 | 0.368 | 0.405 | +0.060 | +0.058 |
| DMEAN s1 | 0.168 | 0.151 | 0.368 | 0.383 | +0.060 | +0.066 |
| DMEAN s2 | 0.168 | 0.149 | 0.368 | 0.371 | +0.060 | +0.066 |
| DMEAN s3 | 0.168 | 0.150 | 0.368 | 0.371 | +0.060 | +0.067 |

Weighting makes Narwaria *worse* at every scale, Korshunov better only at
s0 (+0.037), and the commensurability gap does not shrink. CGAIN/CLOSS
twins: same pattern, smaller amplitudes (full table in `results.json`).
Registered bar (≥+0.02 on BOTH studies at ≥2 scales): **0 of 4 scales
qualify — answer: NEITHER.**

Structural note (measured): the four unweighted GLOBAL_DMEAN slots are
numerically ONE feature — max cross-scale |Δ| ≤ 2.4e-5 over 380 pairs
(mean-of-averages scale invariance). Only the weight field differentiates
the scales, which is the §3.1 z-score-no-op argument made flesh: the
unweighted lane family was already 1-dimensional, and the weighting adds
scale structure that then measures as harmful-or-neutral.

## Q2 — cross-route commensurability (the V3 construction)

E4 = the chunk-2 harness verbatim at tip, 956 both routes (headline stats
reproduce: within-ref 0.9867 / pooled 0.9777 / min 0.9667). Lane level —
the G1 table reproduces exactly; set level (registered definitions):

| feature set | DMEAN-channel cross-route SROCC per scale (s0/s1/s2/s3) | mean over GLOBAL lanes |
|---|---|--:|
| 944 (unweighted only) | 0.850 / 0.850 / 0.850 / 0.851 | 0.820 |
| 956 (all 12 weighted added) | **0.918 / 0.952 / 0.942 / 0.878** | 0.824 |
| 944+DMEAN (4 lanes added) | **0.918 / 0.952 / 0.942 / 0.878** | 0.846 |

Probe-head cross-route score SROCC (the consumer-level statistic: each Q3
head scores both routes' vectors of the same 90 ladder pairs):

| scorer | pooled | within-ref mean | within-ref min |
|---|--:|--:|--:|
| fixed readout (V0_2/228, chunk-2 reference) | 0.9777 | 0.9867 | 0.9667 |
| head[944] | 0.9693 | 0.9683 | 0.9333 |
| head[956] | 0.9681 | 0.9667 | 0.9333 |
| head[944+DMEAN] | 0.9678 | 0.9650 | 0.9333 |

**Answer: the lane-level improvement is real and reproduced, but including
csfw lanes does NOT reduce cross-route disagreement for any scoring
function** — the deltas are ≤0.003 in the wrong direction (noise-scale at
n=90). A head with 940+ features routes around the one commensurable lane;
commensurability of a single statistic is not a scarce resource the model
needed.

## Q3 — the registered refit (transfer probes; splits documented above)

Ridge on z-scored features, target JOD, trained ONLY on UPIQ-SDR (3,779
rows); λ by 5-fold GroupKFold CV (38 content groups) per feature set over
the registered grid {1e-2..100}; ONE HDR look per feature set:

| feature set | λ | SDR CV SROCC | HDR pooled | narwaria | korshunov |
|---|--:|--:|--:|--:|--:|
| 944 | 100 | 0.9363 | **0.7597** | **0.7688** | 0.9346 |
| 956 | 100 | 0.9360 | 0.7518 | 0.7452 | 0.9339 |
| 944+DMEAN | 100 | 0.9361 | 0.7513 | **0.7412** | 0.9340 |

Paired bootstrap on the deltas (10k resamples, seed 20260729):
nar Δ(944+DMEAN − 944) = **−0.0273, p(Δ≥0) = 0.0000**; nar Δ(956 − 944) =
−0.0234, p = 0.0000; kor Δ(944+DMEAN − 944) = −0.0006, p = 0.18.
Registered verdict path (i) required ≥ +0.02 on one stratum, ≥ −0.005 on
the other: **the measurement is a signed, significant negative instead.**

Secondary observation (non-claimable, one look, recorded for chunk 4): the
**944 transfer probe itself** — an SDR-JOD-trained ridge on the new-regime
streaming features, applied cross-route — reaches HDR pooled 0.7597 / nar
0.7688 / kor 0.9346, beating the fixed readout's narwaria by +0.054 and
approaching shipped BHdr (0.7834 nar) with zero HDR training rows. Human-
labeled cross-domain training data moves the axis that synthetic HDR mass
(§8.15) could not; the representation transfers. Caveats: UPIQ-SDR shares
the JOD realignment protocol with the eval stratum, λ hit the registered
grid edge (CV monotone rising to λ=100 on all three sets — symmetric, so
the A/B is fair, but the absolute numbers are not a tuned ceiling), and
UPIQ's look budget means no follow-up selection on this axis.

## Honest caveats

1. n=90 ladder statistics are small-sample; Q2's head-level deltas are
   sign-informative, not significance claims. The Q3 bootstrap negatives
   are the load-bearing numbers.
2. The probe family is ONE registered construction (ridge/λ-grid/GroupKFold
   as specified). No other family was tried — no axis mining; but a
   different head family could in principle read the lanes differently.
   Given G6 (BVLS, SDR) and this probe (ridge, HDR) agree in sign, the
   burden of proof is now on any proposer.
3. What was NOT run: an HDR-backfill-regime LOO with imazen-26-derived
   training mass (scoped out for cause — §8.15 vehicle invalidity +
   HDR-route-only content; see PROTOCOL.md). If the backfill wave ever
   re-opens the family, that instrument exists; nothing here claims it
   would change the sign, and three independent instruments now agree.
4. UPIQ-380 absorbed 3 more registered looks (budget noted in-doc; no
   selection occurred on the HDR axis).
5. E3's "944 set" is the first 944 columns of a csfw-ON extraction —
   covered by the tier-1 V2 first-944 guarantee (verified bit-exact on the
   380-pair EXR leg here and on 149,195 rows by the G6 byte-gate).

## Reproduce

Everything (registered protocol, exact commands, extraction CSVs + sha256s,
analysis script, results.json, logs):
`/mnt/v/output/zensim/hdr-dmean-2026-07-29/` — `PROTOCOL.md`, `COMMANDS.md`,
`analyze.py`, `results.json`, `upiq_hdr_{944,956}.csv`, `upiq_sdr_956.csv`,
`upiq_hdr924_control.csv`, `v3_ladder/`. Logs: `~/tmp/hdr-dmean-*.log`.
Build: zensim `c4632d6257b1`, `--features feature-regime-v2,threads,training`.
