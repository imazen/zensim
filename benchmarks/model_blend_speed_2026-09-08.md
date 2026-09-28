# A/D complete inference cost and benchmark gate repair

The fitted A/D blend does not advance as-is. The existing benchmark now measures
its actual `BakeScorer::ensemble` path, with each calibrated member as a control.
The two-member forward adds negligible cost over A; extracting the shared
feature regime is the important cost. No scoring, model, feature or spatial
arithmetic changed in this continuation.

## Measured cost and limits

Ryzen 9 9950X3D, CPU 8 pinned, one worker, optimized non-native-CPU build,
fast-ssim2 rayon feature off. Same deterministic synthetic texture/edge/ramp
inputs as the historical benchmark; complete uncached reference+comparison.
Forty randomized interleaved rounds at each size, one call per sample.

| Complete surface | 1024² mean / MAD ms | 2048² mean / MAD ms |
|---|---:|---:|
| A/D blend | 88.048 / 0.842 | 194.447 / 1.739 |
| A member | 87.977 / 0.735 | 194.122 / 1.574 |
| D member | 76.412 / 0.794 | 136.044 / 2.220 |
| Named D | 76.516 / 0.857 | 136.109 / 2.085 |
| fast-ssim2 | 67.726 / 0.660 | 289.154 / 1.606 |
| Named B | 95.140 / 0.354 | 223.143 / 0.883 |

All arms have MAD/median below 5%; every paired comparison retains at least
37 non-outlier rounds. At 1 MP the blend's **minimum 87.098 ms** exceeds both
the 50 ms bar and SSIM2's **maximum 69.050 ms**. At 4 MP its minimum
**192.253 ms** exceeds 1.25 times D's maximum **139.835 ms**. These are
conservative inequalities on observed individual-call extrema, not fabricated
p95 estimates. Zenbench's retained result schema does not contain p95/raw rounds.

**Formal performance qualification remains INCOMPLETE.** One resource advisory
and nonzero background samples on CPU 8/24 violate the registration's strict
no-competition condition. The corrected run is retained as low-dispersion
engineering evidence, not an accepted quiet-machine release measurement.
No incremental per-worker RSS, cached-reference, HDR or representative-corpus
performance claim is made. The combined process's 0.66 GiB peak RSS cannot
qualify a per-worker cache/scratch budget. No train/validation/terminal images
were consumed; this screen cannot establish perceptual quality.

The next useful action is to profile the actual common extraction path and
full-pool increment, using this owner and the shipping SIMD tier. A cheaper
head alone cannot remove the measured cost. At 1 MP even D exceeds the absolute
bar; at 4 MP A's pool regime adds substantial cost. Preserve the successful
scalar preference result while deciding between an arithmetic-preserving
extraction optimization and a retrained competitive model in a cheaper regime.
Do not implement every unsupported spatial pool before resolving this cost.
The corruption head's honest false positives and all codec RD/targeting gates
remain required product work.

## Repair that made measurement possible

The first run collected only four rounds per size and recorded 240 waits in
263 seconds. Its `unreliable=false` flag did not make it usable. Live stack and
syscall evidence located the waits; Linux sysinfo includes task IDs in the
process scan, and the lock's own `zenbench-exclusive-heartbeat` thread matched
the rival-benchmark filter. Excluding only the process leader was insufficient.

The canonical zenbench owner now excludes its own task IDs, while retaining
other processes and their tasks in the scan. A live named-thread regression
failed with 29 waits before the fix and passes with zero afterward. The fix is
pushed as `1bf8a6509fce2ff4baddf1e2252639acd78d576c`; the unpublished bench pins
that git revision. Its existing newer ancestor/argv0 and CPU-sampling repairs
are also included. No gate was disabled or release threshold relaxed. The
replacement run collected all 80 rounds in 82 seconds. Original timings,
debugger interventions and the preregistered correction are preserved.

## Instrument and verification

`ZEN_S2_ENSEMBLE` accepts ordered comma-separated bake paths;
`ZEN_S2_ENSEMBLE_WEIGHTS` requires explicit weights. `BakeScorer` validates the
complete composition. `bake_ensemble` and `bake_member_N` arms use that public
surface; legacy explicit-walk diagnostics remain separate and were disabled.
Startup checks complete weighted-score parity on each benchmark input, outside
timing. Both builds produce exactly the same startup scores. `ZEN_S2_SINGLE_CALL=1`
caps each sample at one call. `ZENBENCH_RESULT_PATH` saves structured evidence
and refuses an existing output path before measurement.

Thirteen negative controls pass on both builds: missing members/weights,
empty/single/duplicate/unreadable members, malformed/nonfinite/negative/wrong-count/
wrong-sum weights, and existing-result protection. Root CI-exact Clippy,
the exact nested bench Clippy, scoped Rust formatting and 605-script lint pass.
Zenbench's library suite passes 142 tests with one ignored, and its three local
Clippy configurations pass. No public API changed in either repository.

Evidence: `/mnt/v/output/zensim/model-blend-speed-2026-09-08/`, with the corrected
run in `corrected/`; mirror `~/work/zensim-validation-2026-09-08/model-blend-speed/`.
The bundle retains registrations, both binaries, exact model/weight hashes,
Cargo lockfiles and source copies, commands, CPU/process monitors, raw structured
benchmark results, controls and checks. `RESULT.json` explicitly distinguishes
the observed cost inequalities from failed strict measurement admission and
incomplete release qualification.
