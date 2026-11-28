# zensim-psy

Unpublished `zenpsy-api` adapter for `zensim`, defaulting to Revision-7
`ZensimProfile::A` (`zensim-a`).

- Implements `Metric`, `PrepareReference`, `PlanarMetric`, and
  `PreparePlanarReference` with owned `Send` prepared state.
- Declares `headline_score_key = Some("score")` (`Direction::Higher`,
  `perfect = 100.0`, `unit = "zensim-a"` for `ZensimProfile::A`) and
  `steering_map_key = Some("diffmap")` (`Direction::Lower`, `perfect = 0.0`,
  `unit = "zensim-diffmap"`).
- Uses `Zensim::compute_with_ref_and_diffmap_linear_planes_with` inside
  `PlanarWarmReference::with_evaluation` to reuse scratch and lend the native
  diffmap directly to encoder perceptual loops without an extra full-image copy.
- Reports exact pre-allocation reference storage through
  `estimated_reference_bytes` and declares `is_deterministic() == true`.
