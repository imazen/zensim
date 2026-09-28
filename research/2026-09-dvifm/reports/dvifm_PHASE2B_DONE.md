# DVIFM PHASE 2b — DONE (2026-09-19)

## Verdict: INFO-NOT-USEFUL (preregistered rule, evaluated at N_max / E=100, 10 paired seeds)

| pair | Δ per-ref mean SROCC (primary) | SD | 2·SD/√10 | signs | Δ pooled SROCC |
|---|---:|---:|---:|---:|---:|
| B−A (dvifm − basic228) | −0.00293 | 0.00236 | ±0.00149 | +0/−10 | −0.00503 |
| B−C (dvifm − permuted) | +0.00636 | 0.00286 | ±0.00181 | +10/−0 | +0.00427 |
| D−A (y60 − basic228) | −0.00185 | 0.00146 | — | +1/−9 | −0.00441 |

Bootstrap over 25 dev refs (10k resamples): B−A frac-positive 0.5%
[−0.0053,−0.0006]; B−C frac-positive 100% [+0.0034,+0.0093].

Reading: the 30 DVIFM columns carry REAL within-image ordering signal (they
beat their own row-permuted twins decisively) but it is REDUNDANT on top of
basic228 — net-harmful at every measured scale. The overfitting signature
from Phase 2 persists at full power (fit SROCC +0.0062, loss −28%, dev
−0.0029). Family stays registered, default-OFF, no optimization, no
five-seed confirmation, no frozen EVAL.

INCOMPLETE rule did not fire: every arm's dev per-ref mean DECLINED
E=50→E=100 at N_max (A −0.0036, B −0.0046, C −0.0067, D −0.0032 mean paired
Δ). No E=150 run.

## N_max and provenance

N_max = 8,327 admitted human-labelled TRAIN rows (target ≥30,000 NOT met —
this is the whole admitted estate; stated as a MISSING in the report):
- KADID-10k train-origin refs {0,2,4,6,8}: 5,000 rows (dev = refs {1,3,5}
  3,125 rows, Phase-2 segment verbatim; terminal {7,9} untouched)
- TID2013 all 25 refs: 3,000 rows (train-only ruling)
- KonFiG originsplit-train {SRC06,SRC28,SRC50}: 327 rows
- + KonFiG originsplit-val dev2 leg {SRC01,SRC03,SRC31,SRC45}: 436 rows
  (reported separately; KonFiG test sources untouched)
Extraction: single `extract_features_372col --full-986` pass, 11,888 rows,
spec `dvifm-local-fitted-final.json` sha256
`1b8283987bb47559a0d6fceac8f88e68132d32f01f16d386a24aa6ffe45d3028`.

## Commits (jj, on top of 84d2a70e; NOT pushed)

- `c0ebca00` docs: DVIFM phase-2b preregistration (committed before any fit)
- `02f84632` feat(screen): phase-2b matrix extensions in
  feature_screen_ceiling (epochs_list, data_scales, permuted_arms, eval
  legs, final-epoch checkpoint promotion, bounded audit, report2b)
- `8bff0ff7` docs: phase-2b record (report md/json/pointer, block-gates
  outcome update, board_discussion_sets append — role train-development)

## Learning curve (E=100, dev per-ref mean)

| fit N | B−A (signs) | B−C (signs) |
|---:|---|---|
| 2,000 | −0.0018 ±0.0057 (+5/−5) | +0.0091 ±0.0060 (+9/−1) |
| 8,000 | −0.0021 ±0.0025 (+2/−8) | +0.0057 ±0.0018 (+10/−0) |
| 8,327 | −0.0029 ±0.0024 (+0/−10) | +0.0064 ±0.0029 (+10/−0) |

Deficit does not shrink with data. dev2 (KonFiG, 4 refs) ordering agrees:
A 0.8715 > C 0.8626 > B 0.8510 at E=100.

## What was NOT done

- No E=150 (incompleteness rule did not trigger)
- No DVIFM optimization, X/B channels, no frozen EVAL / CID22 / AIC /
  KonFiG-test / terminal data opened
- No production qualification, no public API change, no push
- Arm C uses ONE fixed permutation (seed 6619, as preregistered) — bounds
  the control to "matched marginals", not a permutation distribution
- dev2 is 4 references — directional only

## Compliance

- 240 fits, ~35 min, ≤8 concurrent single-threaded under run-heavy 16G
- Bounded pixel audit PASS after post-hoc `bake_dial_refit append-meta`
  stamping of formula_revision=3 + feature_set_id on the 240
  checkpoint-derived bakes (recorded in RESULT.json `post_hoc_stamp`)
- Artifacts 969 MB (< 40 GB cap); /mnt/v retains ~102 GB free (≥80 GB)
- `cargo fmt --all -- --check`, `just clippy`, `just lint-scripts`: all green
- Evidence root: /mnt/v/output/zensim/dvifm-screen2b-2026-09-19/
  (RESULT `6d9359c7…`, SUMMARY `dd1d7ab4…`, recipe `d95ec9ea…`)
- Record: benchmarks/dvifm_screen2b_2026-09-19.{md,json,pointer.md}
