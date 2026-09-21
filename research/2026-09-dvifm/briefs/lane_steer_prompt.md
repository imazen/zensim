# LANE `steer` — is the DVIFM block map better than what we steer with today?

Read `~/tmp/devin/LANE_PREAMBLE.md` first; it binds you. Then `docs/PLAN_SPATIAL_STEERING_DVIFM_2026-09-19.md`
(stages S1–S3 are this lane) and `docs/PRODUCTION_PRIORITIES_2026-09-15.md` P3 (its three questions and controls
still govern). Outputs `/mnt/v/output/zensim/dvifm-steer-2026-09-20/` (≤15 GB).

S1 **map owner.** Extend the existing `attribution.rs` / `diffmap` owners (no second map pipeline) to emit, per
plane and level, the block field ε_b = v_b·m_b^P painted as ε_b/N_b over each block's pixels, plus v_b and m_b
separately. Gates: Σ map equals the score contribution exactly (integer path) or ≤1e-6 (f64); a rectangle query
equals the sum of its block terms, with a stated rule for blocks the rectangle cuts; identity → all-zero map;
bit-identical across strip sizes.

S2 **rectangle-gain prediction.** On TRAIN pairs only, replace rectangle R of the distorted image with the
reference, predict Δscore from the map, measure the actual Δscore. Report median and p90 relative prediction error
by rectangle size (16/32/64/128/256 px) and by pyramid level, and the same for the CURRENT attribution-density map
as the control. Also test additivity on two disjoint rectangles and report the pyramid leak (an edit inside R
changes coarse blocks outside R — quantify it, do not assume it away). Use the existing
`diffmap_block_coherence` / native intervention instruments.

S3 **judge agreement.** Per-block severity rank agreement within each image against two-reference truth
(`fast-ssim2` per-scale error map ∧ `butteraugli` diffmap through our own crates — a block is "truly bad" only where
both agree), per codec, with text/screen content reported separately. Control: the current steering map on the same
blocks.

Verdict paragraph: is the DVIFM map more faithful (S2) and better aligned with independent judges (S3) than the map
we steer with today? If yes, name the smallest change that would let a codec loop consume it. Do NOT implement a
codec loop in this lane. Records `benchmarks/dvifm_steer_2026-09-20.{md,json}`; terminal file
`~/tmp/devin/LANE_STEER_DONE.md`, progress `~/tmp/devin/lane_steer.log`.
