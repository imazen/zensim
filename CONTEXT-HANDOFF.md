# Context handoff — 2026-05-27 (QAT native packing + correctness-defect ship candidates)

Read this first, then [SESSION-RESUME.md](SESSION-RESUME.md),
[RESEARCH.md](RESEARCH.md), [CLAUDE.md](CLAUDE.md). (The 2026-05-24 Tuner-v11
handoff is superseded; see git history if needed.)

## What shipped to main this session (all committed + pushed)

1. **Structural-corruption corpus** (codec-corpus): issue #7 (40 mined real
   decoder/render bugs) + PR #8 (10 synthetic distortion-family generators).
   Spec: `docs/structural_corruption_corpus_spec_2026-05-27.md`.

2. **QAT fine-tune in the trainer (task #34, DONE+VERIFIED)** — the "rust
   workflow handles packing" deliverable. `--qat-fine-tune-epochs N` +
   `--out-dtype f16`: last N epochs train quantization-aware (f16+zerobias
   STE), the dial spline fits on the PROJECTED+QUANTIZED (shipped) net, the
   2-layer bake stores f16+compressed. ONE `zensim_mlp_train --manifest` pass
   → **27 KB** bake, identity 97.7 (max), 0 above-identity, CID22 **0.8657**.
   No Python post-step. STANDARD packing path in CLAUDE.md (opt-in).
   `benchmarks/qat_fine_tune_2026-05-27.md`.

3. **TOML recipes** cover masked-monotone + QAT (`zensim/weights/manifests/
   v47_strict.toml`, `v47_strict_qat.toml`) — retired the 25-flag bash script.

4. **Standard pack-then-calibrate** (non-QAT fallback): `scripts/v_next/
   pack_and_calibrate.py`. Rule (both paths): quantize-then-calibrate, fit the
   spline on the projected+quantized net.
   `benchmarks/standard_bake_packing_2026-05-27.md`.

5. **Task #33 tile-min local-defect scorer**: `score_tiles_with_bake` +
   `corruption_gate_eval.py TILE_MIN=1`. tile-min (tile=64) doubles the
   corruption gate on PHOTO (17%→37%), fixes channel/block 8×8 localized
   defects. CONTENT-DEPENDENT (no gain on screen — q20's own tiles crater).
   `benchmarks/local_defect_head_design_2026-05-27.md`.

## SHIP CANDIDATES (both validated; the choice is the USER's call)

| | QAT-native | non-QAT recal-negtail | V39 (current ship) |
|---|--:|--:|--:|
| CID22 SROCC | **0.8657** | 0.8564 | 0.8793 |
| CID22 Z-RMSE | 0.512 | 0.541 | 0.493 |
| KonJND | 0.418 | 0.485 | 0.420 |
| KADID / TID | 0.79 / 0.79 | 0.80 / 0.80 | 0.93 / 0.93 |
| identity | 97.7 ✓ | 97.8 ✓ | **0.0 (BROKEN)** |
| blur>identity | FIXED (0 above-id) | FIXED | VIOLATED (31 above-id) |
| dial monotonicity (q-sweep) | (QAT, monotone) | 94% | 68% |
| size | **27 KB native** | 30 KB (2 post-steps) | 257 KB f32 |

Bakes on `/mnt/v/output/zensim/bakes/`:
`v47_strict_qat_native_2026-05-27.bin` (md5 802f0c46),
`v47_strict_recal_negtail_packed30k_2026-05-27.bin` (md5 4c6cfc67).

## TWO USER-GATED DECISIONS (could not do autonomously)

1. **Ship-form**: replace V39 at `Profile::A` vs add a sibling
   `Profile::A_Strict`. Strongly indicated to REPLACE — V39 scores a perfect
   decode at 0 (broken at identity, confirmed across photo+screen+8 refs); the
   candidates fix blur>identity + give a monotone dial. Cost: KADID/TID rank
   (−0.12/−0.14, the strict-monotonicity price; integrity guards, not the
   compression target). Recommend QAT-native (27 KB, best CID22, native).
   Needs: commit the weight to `zensim/weights/` + `include_bytes!` flip in
   `profile.rs` + archive V39 + CHANGELOG + a methodology doc (template:
   `benchmarks/v47_strict_recal_methodology_2026-05-27.md`). Weight <30 KB so
   under the binary gate, but a user-facing default change → needs your OK.
2. **#33 public API**: a `ZensimLocal` profile / `compute_local` (global +
   min-tile) + zensim-regress wiring. Public API → needs approval. Use a
   content-robust gate (min-tile < absolute T, NOT min-to-min — fragile on
   screen). Internal tile-min scorer is ready.

## Key findings / lessons

- **V39 is BROKEN at identity** (scores 0 on every ref) — 0.879 CID22 is
  rank-only; absolute dial broken at the near-identity regime regression tests
  live in. Strong replace signal.
- **QAT trade is intrinsic** (not tau-tunable): +CID22/+Z-RMSE, −KonJND (f16
  removes PJND precision). Both candidates fail G5's 0.70 (HF Pareto limit).
- **Quantize-then-calibrate**: fit the dial spline on the projected+quantized
  SHIPPED net, else it inverts (blur scored 2184) or identity drops 97.8→93.4.
- **Probe arc (V1/V2/V3) was recipe-confounded** — hand-rolled f64 probe can't
  train unconstrained (diverges); use the production recipe for real numbers.
- Localized-defect detection needs a LOCAL signal (tile-min); global
  perceptual metrics structurally can't rank an 8×8 defect below honest q20.

## Open / next (not user-gated)

- Multi-content tile-min calibration (content-robust threshold T).
- Re-pack the other `zensim/weights/` F32 bakes to f16 (SROCC-neutral cleanup).
- Close G5 (KonJND HF) — characterized Pareto limit, needs HF representation.
