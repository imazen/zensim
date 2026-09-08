# Actual target-loop validation — September 7, 2026

**September 8 interpretation correction:** these measurements mix feasible
and unwitnessed requests. Their aggregate error screen does not establish a
model's targeting failure. The [replacement protocol and experiment](target_steering_bounds_2026-09-08.md)
measure bounds first and evaluate 1/2/3-shot train-calibrated policies only on
witnessed targets. Raw September 7 observations below remain unchanged;
the legacy median <=2 bar is not a validated perceptual product tolerance.

**360/360 registered cells completed**: four original 512×512 PNGs, JPEG/WebP/
AVIF, B/D/H-control seed 4004, targets −10/30/70/90/99, tolerance 1, budgets
3 and 8. Registration `ab14a6787361`; instrument `898ac8589a43`.
Every output bitstream is retained with its SHA, reconstruction SHA, probe
history, bytes, target error, pass count, timing and independent judgments.
Sources, candidate, binary and dependency lock are content-pinned. The Rust
measurement owner verifies the returned reconstruction and exact bitstream
against the codec adapter, then rechecks the final score through the same surface.

This measures the existing generic bisection controller and each model's own
dial. It does not qualify codec-native starting-quality predictors. Fixed B,
CPU SSIMULACRA2 and Butteraugli pnorm3 independently judge the reconstruction.
All targets, including failures and targets outside the observed score range,
remain in the denominator. Endpoints were not exhaustively probed, so that
last category is not a proof that a target is mathematically unreachable.

## Results

### Budget 3

| codec | model | hits / n | median abs error | p95 abs error | median passes | median loop ms | median score ms |
|---|---|---:|---:|---:|---:|---:|---:|
| jpeg | B | 2/20 | 21.796 | 39.626 | 3.0 | 17.72 | 2.717 |
| jpeg | D | 2/20 | 14.654 | 44.779 | 3.0 | 16.86 | 2.715 |
| jpeg | H_anchorlad_s4004_packed | 1/20 | 14.043 | 46.626 | 3.0 | 17.77 | 2.862 |
| webp | B | 1/20 | 22.535 | 40.262 | 3.0 | 52.76 | 2.761 |
| webp | D | 1/20 | 15.832 | 38.715 | 3.0 | 52.12 | 2.746 |
| webp | H_anchorlad_s4004_packed | 1/20 | 16.019 | 48.856 | 3.0 | 52.51 | 2.897 |
| avif | B | 0/20 | 7.901 | 19.009 | 3.0 | 3271.76 | 2.820 |
| avif | D | 1/20 | 6.503 | 13.420 | 3.0 | 3261.23 | 2.861 |
| avif | H_anchorlad_s4004_packed | 2/20 | 4.228 | 20.871 | 3.0 | 3266.24 | 3.023 |


### Budget 8

| codec | model | hits / n | median abs error | p95 abs error | median passes | median loop ms | median score ms |
|---|---|---:|---:|---:|---:|---:|---:|
| jpeg | B | 7/20 | 15.136 | 26.791 | 8.0 | 42.56 | 2.813 |
| jpeg | D | 9/20 | 8.896 | 22.395 | 8.0 | 40.71 | 2.928 |
| jpeg | H_anchorlad_s4004_packed | 9/20 | 4.881 | 25.641 | 8.0 | 40.03 | 2.985 |
| webp | B | 5/20 | 16.457 | 30.251 | 8.0 | 142.09 | 2.745 |
| webp | D | 8/20 | 6.550 | 22.590 | 8.0 | 127.38 | 3.034 |
| webp | H_anchorlad_s4004_packed | 8/20 | 7.276 | 23.861 | 8.0 | 127.25 | 2.903 |
| avif | B | 11/20 | 0.871 | 16.721 | 5.0 | 6381.98 | 2.856 |
| avif | D | 15/20 | 0.513 | 5.830 | 6.0 | 6018.40 | 3.037 |
| avif | H_anchorlad_s4004_packed | 13/20 | 0.781 | 5.492 | 6.5 | 7308.63 | 3.137 |


Every codec/model group fails the registered three-pass median-error ≤2
screen in this diagnostic. Eight passes improve convergence, especially AVIF,
but JPEG/WebP retain substantial error. The one-number product remains a
separate engineering/scientific problem after competitive training is preserved.

The current JPEG adapter searches q=5..99, so this loop cannot establish the
full JPEG q=0 floor contract; the separate full-range ladder instrument owns
that finding. Negative targets expose actual adapter/dial limits here. No
settings were retuned after seeing results. The useful next controller work
must distinguish target bracketing, codec floors and nonmonotonic curves before
attributing misses to the model alone.

## Cost and independently judged bytes

The three-pass matrix took 314 seconds, peak process RSS approximately 0.10 GiB;
the eight-pass matrix took 613 seconds. The three-pass scoring medians are
approximately 2.7–3.0 ms per 512-square pair at eight threads, including candidate
planning/extraction/scoring. This is one bounded geometry, not a general latency
budget or before/after speedup. `BakeScorer` reuses prediction scratch but still
constructs a pixel plan per pair. `VmHWM` is process-cumulative, not isolated
per-model memory. It is labeled that way in the raw JSON.

Training was paused for the loop measurement. The three-pass run's peak system
load was 1.91; the eight-pass run reached 6.94 and an unrelated Miri process was
observed at full CPU afterward. Eight-pass latency is retained as execution
cost but excluded from clean performance evidence. No contended timing is used
to justify feature or implementation removal.

Matched-quality comparisons reuse `rd_probe_analyze_2026-07-18.py`'s existing
log-byte interpolation, separately by codec, source class and judge; they do
not extrapolate. This changes only codec q, so the underlying deterministic
codec frontier is shared. Apparent savings from sparse interpolation (including
large screen-content values with only two matches) are **not encoder-RDO gains**.
At three passes the medians are approximately zero. The complete overlap counts
and eight-pass diagnostics are retained in the JSON; no qualified G-RD claim
is made, and screen evidence from one reference has no content-wide precision.

## Artifacts and checks

[Result JSON](cleanup_target_loop_2026-09-07.json) contains all aggregate numbers,
matched-judge overlap samples and the run identity. Original artifacts are at
`/mnt/v/output/zensim/cleanup-target-loop-2026-09-07/{budget3,budget8}/`:
`INPUTS.json`, `measurements.jsonl`, `COMPLETE`, and every encoded output. The
parent directory contains frozen binaries, Cargo.lock, RUN.json and logs.
`cleanup-floor-control-2026-09-07/INPUTS_AFTER.json` verifies all 360 emitted
bitstreams and the pinned sources/candidate after execution.

The analyzer refuses missing COMPLETE, missing cells and duplicate cells,
including a duplicate replacing a missing cell at the correct total row count.
These negative controls were executed. Default-feature target tests/clippy and
all-feature/all-target compilation (including JXL) pass. This run measures
three codecs; it makes no JXL/PNG target-convergence claim. Default profiles
remain B and D; candidate results have no false named-profile identity.
