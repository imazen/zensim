# Retired-snapshot storage: complete matched grid (2026-10-17)

This variant still fails C at1024². Diff-410 and W-C pass both required sizes.
The MAPPERF goal remains incomplete.

Same deterministic RGB8 pairs, complete retained outputs, cached references, CPU10/v4x/one thread. First32cleanrounds of at most64; unchanged SPEEDQ quiet admission. Paired p95 intervals use the existing Rust owner: 10000round resamples, seed20261016, nominal95%, type7. Exploratory engineering evidence on one deterministic pair per geometry; no content-wide claim.

| model | size | before p95 ms | after p95 ms | butter p95 ms | after−before95% CI ms | after−butter95% CI ms |
|---|---|---:|---:|---:|---|---|
| diff-410 | 64x64 | 1.316591 | 1.133323 | 0.381304 | [-12.071233, -0.120469] | [0.718526, 0.763782] |
| diff-410 | 256x256 | 11.942127 | 4.353811 | 3.313224 | [-14.226877, -1.328053] | [0.986916, 1.088367] |
| diff-410 | 1024x1024 | 92.572578 | 61.896690 | 66.748384 | [-32.041731, -28.472613] | [-6.010965, -2.625687] |
| diff-410 | 2048x2048 | 364.470995 | 252.707280 | 353.371907 | [-124.325798, -94.773841] | [-112.111871, -94.296763] |
| diff-frag-820 | 64x64 | 2.045577 | 1.922483 | 0.400611 | [-0.239746, -0.014117] | [1.480212, 1.597354] |
| diff-frag-820 | 256x256 | 6.011100 | 4.444583 | 3.266917 | [-1.591484, -1.463831] | [1.152699, 1.403186] |
| diff-frag-820 | 1024x1024 | 79.461818 | 63.383343 | 62.912017 | [-17.539491, -13.214854] | [-4.653741, 0.994580] |
| diff-frag-820 | 2048x2048 | 307.830111 | 247.758000 | 363.923967 | [-61.631314, -46.500920] | [-117.963992, -91.063799] |
| w-c | 64x64 | 1.154613 | 0.878178 | 0.376646 | [-0.355785, -0.186003] | [0.483077, 0.503497] |
| w-c | 256x256 | 5.557866 | 4.082915 | 3.281192 | [-1.551737, -1.132381] | [0.659843, 0.902354] |
| w-c | 1024x1024 | 87.736482 | 59.527734 | 63.882175 | [-30.150186, -26.920872] | [-6.295879, -2.895219] |
| w-c | 2048x2048 | 353.852810 | 257.066209 | 360.413822 | [-119.667270, -86.537374] | [-114.080589, -90.870845] |

Descriptive unconstrained OLS of median time_ns=alpha_ns+beta_ns_per_pixel*pixels through the existing speed_matrix_report.least_squares owner. Negative intercepts are fit artifacts.

| model | arm | alpha ms | beta ns/pixel |
|---|---|---:|---:|
| diff-410 | e34_butter_map | -7.851027 | 83.135880 |
| diff-410 | e34_candidate_map_retained | -0.315790 | 57.633759 |
| diff-410 | e34_candidate_map_before | -0.131516 | 83.251536 |
| diff-frag-820 | e34_butter_map | -7.793475 | 82.847404 |
| diff-frag-820 | e34_candidate_map_retained | 0.276922 | 57.868446 |
| diff-frag-820 | e34_candidate_map_before | 0.446325 | 72.628570 |
| w-c | e34_butter_map | -7.919044 | 82.942312 |
| w-c | e34_candidate_map_retained | -0.906760 | 58.330052 |
| w-c | e34_candidate_map_before | -0.011979 | 83.414253 |

Runtime implementation: `b42a73d91868`. Normal after binary: `fedf543a67818a8ac207a5c3931775563b86d96523596b7662a146960c3e42a7`; before binary: `ce2756a21230a8d5b626896434aa88cc546d05d9620a8bea6f18b0b164a1e97f`. Rust1.99.0, empty rustflags, matched dependency closure. Measurements are on r7900x; not measured on Zen5.

Full-cycle run-heavy receipts include queue, quiet waits and excluded attempts: diff-4103309s, C3734s, W-C3011s; each peak RSS1.03GiB; min available16601/16388/16601MiB respectively; peak load5.95 each. Counters and instrumented stage probes are separate diagnostics. Three-model four-size score/map/gradient/query/refinement audits pass; original native cache270/270 rows match exactly. [Artifact hashes and paths](mapperf_2026-10-17.pointer.md).

Initially excluded foreign-build attempts remain archived. Later cells also used bounded turns on the existing global heavy-work lock: the current job finished naturally, and a turn ended on one new accepted cell or after300seconds. No process was signalled, no worker affinity or numerical/quiet/round/statistical gate changed, and no excluded samples were reused.
