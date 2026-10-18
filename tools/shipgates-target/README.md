# Native scalar target validation

This unpublished tool reuses `zensim-target/src/native_probe.rs` for TRAIN-only
calibration, source-family admission, offline witnesses and the frozen 1/2/3-probe
search. It runs frozen codec-owned global encoders, then checks the emitted RGB
through a separately linked first-party decoder and an independent decode oracle.
Every probe retains its bytes, native PNG, RGB hash and work record.

The scalar-only schema explicitly leaves active/neutral-map targeting unmeasured.
The existing three-arm instrument and its public signatures are unchanged. Numeric
bars, source roles, target units and actual full-encode budgets remain unchanged.
The controller retains its nominal-direction proposal heuristic; observed ladder
inversions remain in the data. Witness admission uses actual measured scores,
never a monotonicity assumption or an interpolated feasibility claim.

Build with `just gtarget-scalar-check CONFIG` and `just gtarget-scalar-build CONFIG`.
Use an empty Cargo configuration for the committed dependency pins. An external
output directory via `CARGO_TARGET_DIR` keeps generated build output separate.
Runtime requires explicit `ZENSIM_FORMULA_REV=5` and uses the recorded encoder
and oracle binaries. The local dispatcher is `just gtarget-scalar-panel`; it
releases the shared heavy lock after each fit/evaluation phase. Missing or
incomplete phases refuse; a numerical target miss is retained and reported.

The October 18 development evidence and its limits are recorded in
[the measurement report](../../benchmarks/gtarget_2026-10-18.md). These runs
neither select a shipping candidate nor qualify spatial steering or performance.
