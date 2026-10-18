# Cached candidate scoring

Concrete caller: an encoder's targeting or rate-distortion loop scores many
reconstructions against one immutable source and one loaded bake.

Proposed additive API, behind the existing custom-profiles and
feature-regime-v2 features:

```text
BakeScorer::compute_with_ref(
    &mut self,
    source: &impl ImageSource,
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
    codec_hint: Option<&str>,
) -> Result<ZensimResult, ZensimError>
```

Create the cache with the same scorer's existing `precompute_reference`.
The source argument preserves the canonical pixel-identity check. Dimensions
and sampling contracts are checked; matching source/cache contents and
arithmetic revision remain the caller's responsibility, as with the existing
cached attribution entry. A supported fold feeds its reference pyramid
directly from the cache. Unsupported feed geometries/families use the existing
uncached scorer, preserving the entire result and composition.

No density, gradient, SAT or refinement snapshot is built for score-only calls.
Features, score, raw distance, mean offsets and identity status must match
independent uncached calls bit for bit. Cache creation is outside repeated-call
timings and reported separately.

Cache construction checks both logical dimensions and minimum-64 padded compute
dimensions against the existing pixel limit before allocation or pixel access.
Reuse tests cover strided odd rectangles, identity/reconstruction reuse, both
parallel settings, and 1/8/16-worker pools; the complete result includes its
profile tag.
