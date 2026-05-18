# V_22-mix-LARGE+konjnd@0.02 with real iwssim — methodology + results

**Date:** 2026-05-18
**Status:** packed bake produced, NOT wired as a Profile (per user direction:
crate version bumps should not follow every nice bake).

## Result

5-seed CI head-to-head vs prior ship `konjnd@0.02`:

| Bake | CID22 | KADID | TID | KonJND | AIC-3 | Packed KB |
|---|---|---|---|---|---|---|
| konjnd@0.02 (prior ship, 4-grp 372feat) | 0.8241±0.009 | 0.8996±0.010 | 0.8853±0.006 | 0.8797±0.027 | 0.7904±0.004 | 44 |
| **LARGE+iwssim (5-grp 300feat)**         | **0.8339**±0.007 | **0.9673**±0.0002 | **0.9726**±0.0004 | **0.8869**±0.003 | 0.7872±0.008 | **41** |

CID22 +0.010, KADID +0.068, TID +0.087, KonJND +0.007. AIC-3 −0.003
(paired-t p=0.56 — not statistically significant). KonJND CI tightens
9× because the prior recipe's seed-5 outlier disappears under the
larger corpus.

Packed bake:
`/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin`
(41,695 B; i8 + zerobias 0.005 + lz4; CID22 drift +0.0001).

## Training recipe

5 groups (vs prior 4):

| Group | Rows | Target | Train_w | Val_w |
|---|---|---|---|---|
| safesyn | 196,086 | mix_cv40_iw60 | 1.0 | 1.0 |
| kadid   |  10,125 | mix_cv40_iw60 | 0.3 | 1.0 |
| tid     |   3,000 | mix_cv40_iw60 | 0.3 | 1.0 |
| konjnd  |   1,008 | PJND          | 0.02 | 0.0 |
| **cvvdp_iwssim_LARGE** | **73,300** | **mix_cv40_iw60** | **0.5** | **0.0** |

Hyperparams: hidden=128, epochs=300 (full, no early-stop), lr=1e-3
cosine to 0, l2=1e-5, leaky-α=0.01, minibatch=256, val-policy=min,
PWRC + Norm-in-Norm 0.1, 300-feature input (no auto-transforms).

Mix target: `mix_cv40_iw60 = 0.4·cvvdp_log_norm + 0.6·iwssim_log_norm`,
where both normalisations use the safesyn-anchored (lo,hi) extracted
when the safesyn 372-col parquet was built (cvvdp lo=-2.1188 hi=13.8155,
iwssim max_log=13.7202). LARGE-group rows use those same anchors so
the per-group target distributions are commensurable.

## Corpus dynamics

The LARGE group was built from:
- `/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/cvvdp_imazen_consolidated.parquet`
  (1,169,500 CVVDP scores from vast.ai backfill of v12/v13/v14/v15r sweeps)
- `/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet`
  (75,300 iwssim scores from the 754-chunk vast.ai fleet, 2026-05-18 —
  see below)
- Joined on `(basename, codec, q, knob_tuple_json)` against the
  features parquets at `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/`.
- Final: 73,300 rows surviving the iwssim∩CVVDP∩features intersection.

Why iwssim only covers 75k of the 1.17M CVVDP rows: iwssim-gpu paper
constraint `min(W,H) >= 176`. The v15r_zenjpeg corpus is dominated by
small Wikipedia/Giphy thumbnails (88-220px) that violate the floor.
Filter `png-24-32__ | gif-static__ | png-8__` prefix drop leaves 754
chunks out of 12,000 in the iwssim-supported set. The full 1.17M
CVVDP corpus IS available for cvvdp-only training (see V_22-CVVDP-
LARGE result below).

### Where the lift came from

CID22+0.010 and KonJND+0.007 are modest because both corpora were
already well-covered by safesyn. KADID+0.068 and TID+0.087 are the
load-bearing wins — those anchor corpora share no content with
safesyn, so the LARGE group taught cross-codec dispersion the
safesyn-only training missed.

**Implication:** marginal corpus value is in *content types* not seen
in training, not in raw row count. A future v16 sweep covering
screen content + line art + synthetic gradients + face crops would
likely move CID22 in 0.01-0.03 increments.

## What was tried and rejected

### V_22-CVVDP-LARGE (cvvdp-only target, no iwssim)

Same 1.17M CVVDP rows but with cvvdp_log_norm as the only target.
5-seed CI: CID22 0.805±0.006, KonJND 0.337±0.022 (CATASTROPHIC),
AIC-3 0.799±0.009. Pure-CVVDP supervision on compression distortions
gives no signal for JND ordering. Confirms T11.6's prior result —
CVVDP-only multi-codec extension fails on KonJND.

### V_22-5GRP candlestick sweep (cvvdp_large_w × konjnd_w, BEFORE real iwssim)

9-cell sweep at single-seed with the bogus all-zero iwssim sidecars
(mix_cv40_iw60 reduced to 0.4·cvvdp_log_norm in this corpus).
No cell was strict-Pareto over konjnd@0.02 — every CID22 gain was
within seed noise (±0.009 std), every KonJND regression was real
(−0.08 to −0.14, well outside KonJND's seed noise). Conclusion was
"konjnd@0.02 stays ship" — VALID for the bogus-data corpus, but the
later proper-mix result demonstrates the bogus iwssim was masking
real lift.

## Open questions worth measuring before shipping

1. **α-sweep on the LARGE-included corpus.** The α=0.40 (40% cvvdp,
   60% iwssim) was tuned on the smaller safesyn-only corpus. Pre-
   computed mix columns at every 0.05 step from 0.25 to 0.75 exist
   in the per-corpus 372-col parquets — α retrain is ~5 min × 5
   seeds × 6 α values = ~150 min total. Likely worth running before
   declaring 0.40 optimal.

2. **LARGE-group TRAIN_WEIGHT sweep.** 0.5 is the default; the
   candlestick sweep showed weight sensitivity. With real iwssim,
   the LARGE×konjnd grid may have a Pareto-better cell at
   `cvvdp_large_w` ∈ {0.3, 0.5, 0.7, 1.0} × `konjnd_w` ∈ {0.02,
   0.05, 0.10}.

3. **Early-stop @ epoch 240.** Convergence curves show val_mean
   plateauing 240-300. `--early-stop-patience 30` would save ~20%
   wall-clock with no fidelity loss.

4. **Statistical significance of AIC-3 drop.** Paired-t p=0.56 says
   it's noise, but n=5 seeds is thin. A 10-seed CI would tighten the
   CI and confirm AIC-3 is preserved.

## Provenance

- Training corpus: `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet`
- Trainer: `/home/lilith/work/zen/zensim/target/release/zensim_mlp_train` (commit `04e2a16` at run time)
- iwssim binary in fleet workers: docker `ghcr.io/imazen/zen-metrics-sweep:0.6.4-iwssim-fixed-6227c1a` built from master `6227c1a` (post-78b162f NaN-on-identical fix + adaptive small-image variant)
- Fleet: 26 boxes × ~25 min wall, ~$1.10 total at $0.10/hr cap
- Trainer logs: `/tmp/v22_iwssim_LARGE_logs/seed{1..5}.log`
- Verdicts: `/tmp/v22_iwssim_LARGE_verdicts/seed{1..5}.md`

## Decision

Per user direction: NOT shipping as a `PreviewV0_N` variant. Bake
remains an artifact in block storage. Memory entry at
`~/.claude/projects/-home-lilith-work-zen/memory/project_candlestick_fix_konjnd_weight.md`
tracks this as the strongest candidate to date; next bake that
beats it ships first, no version bumps in between.
