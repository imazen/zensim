# v2-vs-v1 trainability A/B — pre-registration (2026-07-19)

**The decisive experiment of the feature-v2 program** (task #36): do the 264 bounded
"perfectable" v2 features carry at least as much TRAINABLE human-rank signal as v1's 372?
Per-feature correlations (spec §A.12.6) are necessary-but-insufficient — this trains real
models on both feature sets under identical conditions and compares held-out rank.

## Hypothesis + falsification (pre-registered BEFORE any training)

- **H**: a same-recipe MLP trained on v2 features matches or beats its v1 twin on held-out
  human-rank corpora. Rationale: v2 keeps every signal family (D1-fixed SSIM, bounded HF,
  canonical pooling) and ADDS validated carriers (GMS 0.536, transducer bank 0.45).
- **WIN**: v2 arm ≥ v1 arm − 0.010 on the MEAN of {CID22-val, CSIQ, LIVE-R2} SROCC AND on
  each corpus individually ≥ v1 − 0.020. Then v2 wins the program outright (its bounds +
  structural spatializability come free).
- **KILL**: v2 arm ≤ v1 arm − 0.030 on any held-out corpus → trainable-signal regression;
  stop, per-feature ablate before any further v2 investment.
- Between the bands: seed-7 replicate decides; if still between, report as "parity unproven,"
  not spun either way.

## Design (identical across arms — features are the ONLY variable)

- **Train**: KADID-10k + TID2013 — the two corpora with pixels on disk, labels aligned to
  the SAME rows in both arms (v1 arm = the canonical kadid/tid parquets' human_score
  convention; v2 arm joins the identical (ref,dist,label) rows). Groups
  `kadid:1.0:1.0:both` + `tid:1.0:1.0:both`, seed 13 (seed 7 only if needed), epochs 120,
  pairs-per-epoch 50k, target human_score, MLP default width. v1 arm uses all 372 / v2 arm
  all 264 — full sets, since "which set is the better substrate" is the question.
- **Eval (held-out, pixels on disk, never trained)**: CID22-val (4,292; MCOS), CSIQ (866),
  LIVE-R2 (779). Scoring via the canonical stats path (`panel`/zenstats) on forwarded
  predictions — NOT bake_verdict (its corpus registry is v1-372-specific).
- **KonJND**: stretch goal (extraction of the exact val pair-list for v2 is extra tooling);
  if not run, the transducer-bank KonJND question stays open and is SAID to be open.
- **Steer gate**: NOT re-measured here — v2's spatializability is structural (every feature
  is a mean of a per-pixel map); the runtime v2 diffmap fold is future work. Noted, not
  claimed as measured M3.

## Known limitations (stated up front)

1. This is a FEATURE-SET comparison under a lab recipe, NOT the production recipe (safesyn
   is absent — its images require bitstream re-decode; deliberately out of scope). A v2 win
   here justifies building the v2 production data path; it does not itself produce a ship
   candidate.
2. KADID/TID are train==val-overlap corpora in the canonical registry — fine HERE because
   both arms train on them identically and ALL verdicts come from the disjoint held-out
   trio.
3. v1 arm trains on the frozen v1 extraction; v2 arm on the phase-5 kernel (post-SIMD/strip,
   ≤5e-4 vs scalar reference).

## Mechanics

1. `v2_ab_extract` (extractor example, extends the v2_helpfulness_screen machinery): dumps
   per-pair v2 feature parquets (f0..f263 + human_score + ref_basename) for
   kadid/tid/cid22val/csiq/live from the on-disk pair lists — SAME lists as the v1 canonical
   parquets / the 2026-07-18 FR-corpus builders.
2. v1 arm data: canonical kadid/tid train parquets + the existing 372-col val parquets.
3. Train both arms with `zensim_mlp_train` (identical argv apart from --group paths +
   feature count). Spline via `add-spline` (rank-invariant; for dial sanity only).
4. Forward both bakes over both arms' val parquets (`predict_features_with_bake`), stats via
   `panel`. Report the full Mohammadi panel, verdict per the pre-registered bands.
5. Everything lands in `benchmarks/v2_trainability_ab_2026-07-19.md` + sidecars.

Execution is serialized AFTER feature-v2 phase 6's bench windows complete (no load
contamination of its gate measurements).
