# feature-cost lane — full `bake_verdict --full-json` outputs (block storage)

The seven `*.fulleval.json` files behind
`benchmarks/feature_cost_frontier_2026-08-31.md` §4.2 / §4.2b / §4.3 carry
per-pair scatter blocks and run 2.0-2.3 MB each — well over the 30 KB git
limit — so they live in block storage. The committed
`model_class_quality.tsv` in this directory is the reduction the doc quotes
(every scalar, plus each bake's `bake_sha256`), so no number in the note
depends on fetching these.

**Block storage:** `/mnt/v/output/zensim/feature-cost-2026-08-31/verdicts/`
(`*.json` + the human-readable `*.md` verdicts).

**Provenance.** `bake_verdict --cross-regime --regime 944 --features-root
/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30` for the five `*_r1b` rows
(the one root carrying live `f0..372` AND `f372..944`, so 372-wide and
944-wide bakes read the same pairs), and `--regime 372 --features-root
/mnt/v/zen/zensim-training/2026-08-30-full-features-372` for the two `*_372`
cross-root controls. Bakes: `ADD156_safesyn_only_raw_lasso`,
`b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07`,
`Q7b_pools_g0.2_a0.2_b0.97`, `Q7b_pools_g0.25_a0.2_b0.97`,
`c_sdr_purity944_2026-08-29`. The ssim2 floor row is the board's own
`/mnt/v/output/zensim/reports/fulleval/peer_ssim2.fulleval.json`, read
unmodified.

Also here: `contrib_*.tsv`, the per-input `bake_contrib` dumps whose
family-range summaries are the `ablate_*.log` files committed alongside.
