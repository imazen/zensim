# Exact block gains for the STEER research experiment

`ZENSIM_STEER_EXACT=all-blocks-v1` is an explicit research mode, selected when
`BakeScorer::prepare_steering` creates a session. Absent or `off` preserves the
existing approximate map. Unknown values refuse. This is diagnostic steering,
not production adoption or a change to legacy linear M2.

The mode requires an isolated Rev5 SDR sRGB RGB8 bake and bin1. On every
`SteeringSession::compute`, it eagerly pastes the reference into each rectangle
on the8/16/32/64 grids, independently from the original reconstruction, then
runs full native extraction and the complete score. `refinement_gain` returns
that finite gain for registered rectangles. Unsupported rectangles return NaN;
no approximate fallback or subset-delta forward is substituted. Ordinary
comparison score, features and explanatory density retain their existing paths.
Input rows may be strided; padding is never treated as pixels. Codec routing,
HDR, other pixel formats, ensembles and score dispositions are refused.

All repair extraction, forwards, allocation and gain-table construction occur
inside the cached map call. Include them in the unchanged spatial/RSS collector.
No performance or qualification claim follows from exactness. The complete
native reference deliberately favors an auditable repair over unmeasured cost.

`just steer-exact-tests` checks91 synthetic edge-clipped rectangles against
independently constructed native repair forwards, including strided input,
identity and unsupported queries. These deterministic generated images are
outside the fixed135-case steering panel. They read no corpus, model or labels.
