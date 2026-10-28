# Cross-path reuse and exactness audit

Baseline implementation: `d15a5486`. Scores and maps must retain their existing
bits during this refactor. No public API or model changes are included.

## Routes and reuse boundaries

- `Zensim::compute` selects the streaming fold for supported model plans. Serial
  and parallel folds share the same canonical strip reduction order.
- `compute_with_ref` and `compute_with_ref_into` feed the fold from a cached
  reference, with a buffered fallback when the cache cannot feed it. The latter
  also reuses caller-owned scratch.
- RGB and planar linear diffmaps share the scale-band implementation, but currently
  duplicate final score selection, contrast masking, square root, and trimming.
  Both recompute the score plan twice. These are the first consolidation targets.
- Planar canonical diffmap scoring currently materializes interleaved RGBA. Removing
  that allocation requires a compatible row-source design; the present ImageSource
  trait returns borrowed packed rows. It is not a free interface substitution.

## Existing mismatch

Before any production refactor, profile A produced different score bits between
interleaved and planar linear diffmaps on a 65×63 checkerboard with strided planes:
`4635359462469191635` versus `4635359462376882550`. The new cross-path test exposed
this. Changing either result conflicts with preserving every existing score.
The audit records this limitation rather than silently changing the score contract.

## Validation design

`tests/cross_path.rs` exercises subminimum images, odd dimensions, poisoned row
padding, masking/sqrt combinations, cached-reference scratch reuse and SIMD tiers.
`examples/path_audit.rs` supplies output fingerprints and zenbench timings for the
compute, cached, scratch, map and planar routes. Scalar token dispatch and explicit
thread counts are separate axes. Memory observations use fresh-process maximum RSS,
including fixture and reference setup; timings exclude that setup and fingerprinting.

Measurements and final findings will be recorded here after the paired runs.
