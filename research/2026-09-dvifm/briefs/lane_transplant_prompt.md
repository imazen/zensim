# LANE `transplant` — X4/X5/X7: adopt DVIFM's mechanisms into zensim, and grow the core enough to decide

Read `~/tmp/devin/LANE_PREAMBLE.md` first; it binds you. Then `docs/PLAN_DVIFM_VERDICT_2026-09-20.md` (X4, X5, X7
and §4-§5), `docs/PLAN_JOINT_CORE_SET_2026-09-19.md`, `benchmarks/joint_core_v1_2026-09-20.md`. Outputs
`/mnt/v/output/zensim/dvifm-transplant-2026-09-20/` (≤25 GB).

**Step 0 — grow the core, because the screens below cannot be decided otherwise.** `joint-core-v1` failed its
permuted-column gate: 30 permuted columns cost +0.0021 against a ~0.003 noise floor. Extend it (same rules: ≥75%
camera photography, ≤1 MP, ≥55% of pairs at 384-1024 px, plain Mitchell via `zenresize`, signed unclipped targets,
clustered selection, kernels recorded per leg) until the permuted-column cost is BELOW the noise floor with margin,
measuring the gate at each size so we learn the sensitivity curve: report the gate at ~53k (already measured), and
at your new sizes. Name it `joint-core-v2` with its own manifest and DATA_SPLITS addendum; do not modify v1.

**X4 — pooling transplant (the highest-value experiment in the programme).** zensim's OWN XYB decomposition (its
box scales, its 11×11 residual) with DVIFM's 5×5 block-peak × two-state-visibility pooling replacing global
moments, as an opt-in feature family. Arms on the leaders' recipe, 5 paired seeds, each with a size-matched
permuted control: production zensim; zensim + block-pooled terms; zensim + permuted terms. This tests the MECHANISM
without adopting the pyramid — the add-on screens already showed 30 DVIFM columns on top of basic228 are redundant.

**X5 — chroma transplant.** zensim + DVIFM chroma terms only (Cb, Cr), since both constants domains put 71-83% of
the channel weight on chroma. Same recipe, seeds and permuted control.

**X7 — block-edge contrast**, three terms, each a single change, per §5 of the verdict doc: (1) the 5×5 block peak
on the UNFILTERED level-0 difference (no band, no blur — the only term that sees a codec block step at full
amplitude); (2) an across-boundary vs within-block contrast discriminator; (3) the same run over all 8 codec-grid
phases keeping the maximum, against the fixed-phase control (zensim's v2 idx 25 is oriented but fixed-phase).
Judge per codec: JPEG and WebP are where blocking dominates.

Report every arm with per-seed paired differences, sign counts, and its permuted control; a gain that does not beat
its permuted control is not a gain. Records `benchmarks/dvifm_transplant_2026-09-20.{md,json}`; terminal file
`~/tmp/devin/LANE_TRANSPLANT_DONE.md`, progress `~/tmp/devin/lane_transplant.log`.
