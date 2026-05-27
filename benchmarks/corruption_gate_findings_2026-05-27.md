# Corruption-corpus gate + q-sweep G3 — validation findings (2026-05-27)

Both gates run on `v47-strict-recal-negtail` (the dial-fixed monotone bake)
vs V39 (shipped Profile::A), per the user's "validate gates first" decision
before any ship. Corpus: codec-corpus#7 / PR#8 structural-corruption
generators (gb82/dog photo ref, 672 entries × {corruption, q20, q10}).

## TL;DR

1. **G3 q-sweep monotonicity (codec dial): recal WINS decisively** —
   94.0% monotone (4 dead-zone ties) vs V39 67.7% (482 ties). recal is a
   clean, usable codec dial; V39 is not.
2. **Identity (self-similarity): recal is SANE, V39 is BROKEN.** recal scores
   `score(ref,ref)=97.8` on every reference (photo/kadid/screen). **V39 scores
   identity = 0.0 on every reference** (raw ≈ −90 → clamp). V39 cannot tell a
   perfect decode from a broken one — both clamp to 0.
3. **Corruption gate (localized defect < honest-lq): UNSOLVED by any global
   metric.** recal gets the GLOBAL ordering right (identity > honest-q20 >
   whole-image-corruption) but a localized 8×8 corruption is globally
   negligible → scores ~74–96 (above q20's 44). V39's apparent 95% "pass" is a
   calibration ARTIFACT (it scores everything low, including identity).
   Localized-defect detection needs a LOCAL / max-pooled signal — a separate
   head, not a global mean-pooled perceptual metric.

## G3 q-sweep monotonicity (real JPEG q5–q100, 50 imgs × 19 q)

| bake | monotone q-steps | dead-zone ties | dial shape |
|---|--:|--:|---|
| **recal-negtail** | **0.9400** | 4 | clean monotone median 4.6→88.4 (q5→q95) |
| V39 | 0.6767 | 482 | median collapses to 0 for q55–q95 |

recal's top-of-dial hypersensitivity (q85→95 in a 0.002-wide tanh-pin window)
compresses high-q increments but stays MONOTONE — not a dial break.
Report: `benchmarks/qsweep_v47recal_vs_v39_2026-05-27.md`.

## Corruption gate — production clamp [0,100], gb82/dog

Sane ordering should be: identity > q20-honest > corruption.

| pair | recal | V39 |
|---|--:|--:|
| identity (ref,ref) | **97.8** ✓ | **0.0** ✗ (not the max!) |
| q20 honest (whole) | 43.9 | 76.2 |
| channel_invert WHOLE | 17.2 (✓ < q20) | 62.9 |
| channel_invert sq8 (local) | 74.0 (✗ > q20) | 0.0 |

- **recal**: identity is the max everywhere; whole-image corruption ranks
  below honest q20 (correct). Misses only *localized* (8×8/16×16) corruption,
  which is globally subtle → scores high. Gate(vs q20): whole 35.6%, frac4
  21%, sq16 0%, sq8 0% → 17.3% overall. This is the honest behavior of a
  global perceptual metric, not a bug.
- **V39**: identity = 0 → ranks a perfect decode below a q20 lossy encode AND
  below a channel-inverted image. Its 95.2% gate "pass" is meaningless — it
  scores ~everything deeply negative (identity raw −90, corruptions −95..−128,
  q20 +76), so corruptions land below q20 only because identity does too.

## What this means for the ship

- **recal-negtail is the SANE metric** and the better codec dial. It fixes
  the V39 blur>identity defect AND the (more severe) V39 identity=0 defect,
  is 94% monotone on real codecs, keeps CID22 0.855 with better calibration.
  Safe to ship — replacing V39 at Profile::A is now well-justified (V39 is
  genuinely broken at the identity / near-identity regime that regression
  tests live in).
- **Localized-corruption detection is a SEPARATE, unsolved problem.** No
  global mean-pooled perceptual metric (recal or V39) can rank an 8×8 defect
  below a global q20 encode — the defect is globally negligible. The
  regression-test use case ("catch a broken decode") needs a LOCAL signal:
  max-pooled / tiled zensim, or a dedicated structural-defect head trained on
  this corpus. → new task. The corruption corpus did its job: it surfaced
  exactly this limitation (the spec's "measured, not asserted on the subtle
  end").

## Caveats / follow-ups

- Single photo reference (gb82/dog) for the full 672-entry gate; the
  identity=0 finding is confirmed across 8 refs (photo/kadid/screen). A
  multi-ref full-gate sweep is a cheap follow-up but the conclusions are
  robust (corroborated by the q-sweep median-collapse across 50 imgs + the
  blur-ladder above_identity=31 for V39).
- Several block/whole generator entries score ≈ identity for all severities
  (e.g. block_repeat_neighbor__whole) — possible generator subtlety worth a
  glance, but orthogonal to the global-metric finding.
- Scripts: `scripts/v_next/corruption_gate_eval.py`,
  `scripts/v_next/recal_v47_dial.py`. Full per-family table:
  `benchmarks/corruption_gate_v47recal_dog_2026-05-27.md`.
