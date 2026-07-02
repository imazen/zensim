# PLAN_BEAT_A.md — beating Profile::A as a KNOB and as an RD-LOOP metric

Locked 2026-07-02 with the user. This is the durable plan-of-record; it
survives compaction so no session re-derives it. Companion docs:
`docs/DATA_SPLITS.md` (split registry), `docs/ITERATION_PROTOCOL.md` (how to
run cells/grids), `benchmarks/multicodec_profile_probe_2026-06-30.md` (the
full evidence trail this plan is built on).

## The two roles and where A stands (all numbers measured this cycle)

**Knob** (user types "zensim 85", codec binary-searches q): A is structurally
sound — codec dial mono 0.9747 + G1 pass, KADIS held-out safety 0.9726 ≈ the
0.980 oracle ceiling, QAT-native spline-calibrated 27 KB artifact,
byte-reproducible. Its knob weakness is **near-threshold discrimination**:
KonJND 0.4185 (G5 floor is 0.70), HQ-zone cvvdp-agreement 0.719/0.720
(70-85/85-100), step-agreement misses concentrated at 85-100 — exactly the
q90-vs-q93 decisions production knobs make.

**RD loop** (rank encodes, drive encoder decisions): A's CID22-49 is 0.8657
while our own unconstrained data model (cvvdp_w1) reached 0.8759 — rank is on
the table. We don't yet even evaluate the regime RD loops actually use
(same-image pairwise, the CID22 paper's Table-6 regime).

## The three bets, in evidence order

**Bet 1 — fix the LABEL in the HQ zone (wave-4; GATED ON the multi-metric
backfill of the 5.7M canonical corpus, in flight 2026-07-02).** The HQ
instrument measured that ssim2's own label is the binding constraint at
85-100 (cvvdp-agreement 0.48; adding more ssim2-labeled HQ rows made bakes
WORSE there). When the backfill lands: rebuild
`bigcodec_multimetric_<date>.parquet` with per-zone targets — cvvdp/
butteraugli-derived labels where ssim2 ≥ 0.85, ssim2 in its reliable 30-85
band — plus kadis-cvvdp as-is. Expected (from measured lever effects):
HQ-zone + KonJND up, CID22 neutral-to-up. Prereq (b) below must land first so
the HQ instruments are content-holdout.

**Bet 2 — the untapped HUMAN pairwise signal.** In-house, unused: 419,760 raw
AIC-3 BTC/PTC triplets + 95k SDR25 triplets (`/mnt/v/datasets/aic3-btc-ptc/`,
`/mnt/v/datasets/jpeg-ai-sdr25/`) — human judgments in exactly the JND regime
both roles need. Triplets are order-pairs; the TV-hinge plumbing
(`--tv-pairs-file`, wired 2026-07-01) can carry human pairs directly.
**Splits amendment required (DATA_SPLITS §3):** AIC-3 moves T0→T2 (its
triplets train); **SDR25 (JND-reconstructed) becomes the new untouched
compression T0** alongside CID22-49 + AIC-4. This is the direct attack on
KonJND/G5 — metric distillation has never cracked it because it needs human
near-threshold data, which this is.

**Bet 3 — separate rank/pool encoders (structural; ONLY if 1+2 stall).** The
shared 372→128→64 encoder couples dial-monotonicity pressure to rank features
(measured: α already routes per-regime, capacity doesn't decouple). Keep A's
knob machinery (cbc + QAT + spline) untouched; the split-encoder is a
multi-crate spike (trainer + arch + zensim runtime + bake format) scheduled
only on evidence that labels + human pairs weren't enough.

## Pre-registered gates (ONE artifact must pass BOTH columns; 5 seeds; paired tests)

| Knob | RD loop |
|---|---|
| codec dial mono ≥ 0.975 + G1 pass | CID22-49 ≥ 0.876 (cvvdp_w1's proven level) |
| KADIS safety in 0.96-0.98 (oracle 0.980) | AIC-4 ≥ 0.89; SDR25 ≥ fast-ssim2 on it |
| HQ zones ≥ A both; step-agreement ≥ 99% @ 85-100 | within-ref pairwise ≥ fast-ssim2 (new eval) |
| KonJND ≥ 0.50 now (0.70 = G5 goal) | full Mohammadi panel; KADID/TID guards only |
| identity ≥ 97, tied ≤ 5%, spline invertible | no HQ-band regression vs A per-band |

Never scoreboard vs ssim2 on KADID/TID (in-sample for it). Single-seed deltas
< 0.02 CID22 are noise. Verdicts from bake_verdict only, never training-side
diagnostics.

## Instrument prerequisites (cheap, before/parallel to Bet 1)

a. **SDR25 JND reconstruction** → `sdr25` T0 corpus in bake_verdict
   (features via decode + extract; audit on ingest). Unblocks Bet 2's holdout
   swap. Gated on nothing.
b. **Rebuild HQ + dial grids on TEST-digit origins ({7,9})** → true
   content-holdout instruments (current ones are in-domain for
   bigcodec-trained bakes; documented in DATA_SPLITS §4).
c. **Within-ref pairwise eval in bake_verdict** (Table-6 regime): for each
   ref, all same-ref pair orderings vs human; report accuracy + per-band.
   This is the RD-loop-native measurement we currently lack.

## Current in-flight state (2026-07-02)

- zen-train-1 (Hetzner ccx63) bootstrapping → auto-handoff grid: v51box_s17
  (cross-machine determinism check vs local bytes) + v51 s47/s63. Then
  `hz.sh pull` + **`hz.sh retire`** (mandatory lifecycle).
- Multi-metric backfill of the 5.7M corpus: USER-side fleet, in flight → Bet 1.
- v51 (local, 3 seeds): held-out-val selection PROVEN (0 collapses, CID22 sd
  0.009); recipe = trade vs A (CID22 −0.021, KonJND +0.05) pending Bet 1
  labels.
- Ship path when gates pass: QAT-native artifact → methodology doc
  (`benchmarks/` template per CLAUDE.md shipping policy) → `include_bytes!`
  swap proposal → USER sign-off.
