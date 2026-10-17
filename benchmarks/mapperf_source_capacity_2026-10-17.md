# Source plane capacity: complete matched grid (2026-10-17)

All three candidates beat matched butteraugli at1024² and2048², with
all six paired upper95% bounds below zero. All still lose at64²/256².
This is the registered fixture protocol, with one pair per geometry.

Same deterministic RGB8 pairs, complete retained outputs, cached references, CPU10/v4x/one thread. First32cleanrounds of at most64; unchanged SPEEDQ quiet admission. Paired p95 intervals use the existing Rust owner: 10000round resamples, seed20261016, nominal95%, type7. Exploratory engineering evidence on one deterministic pair per geometry; no content-wide claim.

| model | size | before p95 ms | after p95 ms | butter p95 ms | after−before95% CI ms | after−butter95% CI ms |
|---|---|---:|---:|---:|---|---|
| diff-410 | 64x64 | 1.273889 | 1.120758 | 0.382106 | [-0.160584, -0.118465] | [0.728103, 0.768229] |
| diff-410 | 256x256 | 5.781038 | 4.380138 | 3.387122 | [-4.783814, -1.175021] | [0.824105, 2.479909] |
| diff-410 | 1024x1024 | 88.105491 | 56.264235 | 62.951645 | [-32.199787, -30.816430] | [-8.154366, -5.815371] |
| diff-410 | 2048x2048 | 345.660576 | 224.101496 | 354.602796 | [-123.042947, -120.841297] | [-134.504002, -125.996215] |
| diff-frag-820 | 64x64 | 2.045975 | 1.912081 | 0.389680 | [-0.219367, -0.104007] | [1.491953, 1.538492] |
| diff-frag-820 | 256x256 | 6.043747 | 4.665119 | 3.304996 | [-1.491582, -1.294508] | [1.269221, 1.406061] |
| diff-frag-820 | 1024x1024 | 75.051863 | 55.616835 | 62.504205 | [-19.757370, -14.606948] | [-7.838636, -1.999086] |
| diff-frag-820 | 2048x2048 | 309.207371 | 226.593789 | 356.542870 | [-83.983283, -79.245658] | [-133.604373, -125.541332] |
| w-c | 64x64 | 1.090190 | 0.955334 | 0.383278 | [-0.312824, -0.104378] | [0.561867, 0.599328] |
| w-c | 256x256 | 5.612017 | 4.184386 | 3.414404 | [-1.515649, -1.041438] | [0.675452, 1.087526] |
| w-c | 1024x1024 | 86.507306 | 56.176689 | 63.465621 | [-31.346507, -30.013906] | [-7.982077, -6.433517] |
| w-c | 2048x2048 | 347.145365 | 224.858072 | 355.993387 | [-124.235591, -121.986340] | [-132.934400, -126.679693] |

Descriptive unconstrained OLS of median time_ns=alpha_ns+beta_ns_per_pixel*pixels through the existing speed_matrix_report.least_squares owner. Negative intercepts are fit artifacts.

| model | arm | alpha ms | beta ns/pixel |
|---|---|---:|---:|
| diff-410 | e34_butter_map | -7.855938 | 83.166411 |
| diff-410 | e34_candidate_map_retained | 0.523488 | 52.696064 |
| diff-410 | e34_candidate_map_before | 0.312684 | 81.887297 |
| diff-frag-820 | e34_butter_map | -8.248020 | 83.970415 |
| diff-frag-820 | e34_candidate_map_retained | 0.770381 | 53.214546 |
| diff-frag-820 | e34_candidate_map_before | 0.366353 | 73.052991 |
| w-c | e34_butter_map | -7.991100 | 83.521537 |
| w-c | e34_candidate_map_retained | 0.470958 | 52.817967 |
| w-c | e34_candidate_map_before | 0.023100 | 82.283948 |

The after build retains inactive source-vector capacity while keeping its logical length zero, then copies the current cached reference with clone_from. No feature selection or floating-point operation changed. Its independent oracle checks source and distorted allocation reuse, a changed reference at the same dimensions, concurrent live outputs, cross-thread drop and changed geometry against fresh ordinary walks.

Normal after SHA-256: `26fe769038bc9d4cfc02d9b46f24a2525638cf6ca956bd77bdc11671955c40eb`. Native test SHA-256: `395226f5a6ddf43af1653737ab6ef8eefbf801754046f54b76d18f9566be6984`. Before SHA-256: `ce2756a21230a8d5b626896434aa88cc546d05d9620a8bea6f18b0b164a1e97f`. Rust1.99.0, empty normal rustflags, matched dependency/features/release profile; no native CPU flags. CPU10/sibling22 were sampled99.50%/100% idle before starting. CPU2 and sibling14 are excluded. Measurements are on r7900x; not measured on Zen5.

Complete cycle run-heavy: C107s, diff-410117s, W-C255s including admission/queue waits. All peak RSS1.03GiB. Minimum available RAM20781/20800/20675MiB and peak load2.46/2.21/2.25 respectively. Timing/report owners and all numerical/quiet/round/statistical gates are unchanged. Existing global-lock turns wait for the current job to release naturally, then release after one new accepted cell or300seconds; no process is signalled.

Current source checks:12/12 full score/map/gradient/query/refinement audit cells; the four-size forced-original subset oracle;38 scorer and15 refinement checks (three pre-existing ignores unchanged). The final source-capacity binary passes all270/270 original native-cache rows, including every original output field and the original repair-cache path (135 rows per fixture). [Artifact hashes and paths](mapperf_2026-10-17.pointer.md).


Final source commit: `2c6cfad25204f065d92e8e3ee89097a5852d8c30`, verified on origin/main. The final native-cache receipt is `src0-final/FULL_NATIVE_CACHE_PARITY.json`.

Final instrumented release-stage medians at1024² on CPU10, ten samples after three warmups (milliseconds):

| Candidate | Scalar extraction | Scalar forward | Map extraction | Map forward | Gradient | Density | Snapshot |
|---|---:|---:|---:|---:|---:|---:|---:|
| diff-410 | 27.3336615 | 0.009318 | 35.280959 | 0.0175135 | 0.384871 | 21.278758 | 0.671470 |
| diff+frag-820 | 27.3862575 | 0.016622 | 35.2085215 | 0.027322 | 0.6984615 | 19.681150 | 0.663695 |
| W-C | 27.409301 | 0.006573 | 32.8858605 | 0.012649 | 0.1976205 | 20.0112465 | 0.678699 |

Scalar extraction uses the uncached production pair; maps use a cached reference. Stage medians are separate from paired p95 qualification and need not add to a median total. Stage binary SHA-256: `3f7b21075f620d1751a1b6cc2e902405f7cdf919b05c09b019e033317e371ea7`. Stage run-heavy:64s including admission, peak RSS0.17GiB, minimum available23137MiB, peak load4.84.

The final C process counter diagnostic (100 complete calls plus setup) records803063 minor faults, zero major faults,4.793281s user and1.019059s system time,5.809872915s elapsed;31194432833 cycles and57340854692 instructions. This diagnostic does not replace paired per-call timing. Its wrapper reports6s, peak RSS0.17GiB, minimum available22490MiB, peak load1.52.

Final verification wrapper receipts: original135-row packets143s each, peak RSS0.18GiB, minimum available27055/31927MiB and peak load28.60/30.01; these ran on dev and are correctness evidence only. The subset oracle reports1s/0.86GiB/min37273MiB/load31.27; scorer/refinement checks15s/0.13GiB/min34213MiB/load33.06. Strict source clippy passes (4s/0.55GiB/min22884MiB/load1.43). No thresholds or existing ignores changed.

Across these four-size fits, each candidate has a lower after slope and a higher fitted intercept. These fixture measurements support a reduction that grows with image area; they do not show a reduction in the fitted fixed cost. The small-image losses remain visible. The earlier C profile recorded16.25% in the feature-extraction dense block kernel and9.85% in memmove (zero lost samples); the final counter and stage measurements above qualify the subsequent allocation experiment.

All19 final-source CI jobs passed on exact source2c6cfad25204f065d92e8e3ee89097a5852d8c30, including Windows ARM64, macOS ARM64, i686 and Ubuntu ARM64. [CI run39008241042](https://github.com/imazen/zensim/actions/runs/39008241042). The receipt is pinned in the artifact pointer.
