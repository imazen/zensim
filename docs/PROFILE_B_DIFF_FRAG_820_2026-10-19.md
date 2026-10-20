# B: diff+frag-820 (2026-10-19)

The owner promoted the frozen **diff+frag-820** model to slot **B** and the
external name **zensim-b**. `codec_target()` and `latest_preview()` return B.
This intentionally changes B's SDR scores; pin the crate revision and bake hash
when reproducing an older result. No new fit or calibration accompanies this
rotation.

The artifact is `zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin`,
214,904 bytes, SHA-256
`89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.
It is the packed full seed-0 model with its existing output calibration.
Its declared formula revision is **5** and its declared read set contains
**420 feature IDs**, expanded into **820 derived model inputs**. The recipe
label “820” is not the number of extracted feature IDs. The immutable source training-manifest hash is recorded in the
[weight manifest](../zensim/weights/manifests/b_sdr_diff_frag_820_rev5_2026-10-19.toml).

The named scorer runs the bake's Rev5 basic/peak/v2 plan without an environment
pin, including in minimal builds. Revision flows through color conversion,
blur, accumulation, dense/gradient kernels, finalization and reference caching.
The public research/engine APIs retain their feature gates. Raw cross-revision
research requests retain their admission guards.

The extended masked/IW extraction API refuses Rev5's unsupported families.
Strip entry points delegate to canonical B scoring. Legacy diffmaps retain
their basic-plane proxy meaning while returning the complete B scalar score;
they are not MLP sensitivity maps. Model-specific maps use the existing
`BakeScorer` attribution surface and its support diagnostics. Spatial quality
qualification remains separate from serving parity.

No corruption head is attached. The old linear-B JND map is detached: this
bake has no measured JND calibration. HDR entry points continue routing B to
the existing BHdr profile. The old explicit-ID linear B artifact remains
available as a historical regression control, with its dense/wide parity gates.

## Evidence and limits

The [matched prior-model comparison](../benchmarks/gauntlet_legacy_comparison_2026-10-19.md)
records the frozen external results and their exposure. Promotion is an owner
decision, not a declaration that every registered gate passed. Existing FAIL
and INCOMPLETE findings remain in force.

In particular, severe LIVE color noise can score above a less damaged version
of the same image. The [matrix trace](../benchmarks/gauntlet_live_color_trace_2026-10-18.md)
records the omitted finest-scale chroma and signed-network contributions.
The [repair comparison](../benchmarks/color_noise_repair_2026-10-18.md) retains
this inversion. Those exposed examples were not used for a new fit here.

Runtime regression checks pin the exact bake, compare named and dynamic
scoring, check identity and cached references, and compare complete feature and
score bits across process revision defaults 1–5. The build-feature serving
matrix checks supported feature combinations. These checks establish serving
parity; they do not fill missing perceptual, dial or spatial qualification.

Commands and runtime evidence are in the [validation record](../benchmarks/default_b_diff_frag_2026-10-19.md).
