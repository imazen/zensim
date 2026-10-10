# STEERPATH_PERF: prepared-steering spatial cost and memory (2026-10-10)

QUAL-A (`benchmarks/qual_a_2026-10-10.md`) found that every candidate fails two release gates in the
shared prepared-steering path (`BakeScorer::prepare_steering` once, then `SteeringSession::compute`
per call: score plus map, bin 1):

| gate | bar | QUAL-A, model A |
|---|---|---|
| spatial cost, cached-reference score+map p95 | ≤ 3× the uncached p95 | 4.07× at 1024², 3.15× at 2048² |
| peak incremental RSS with a map | ≤ 128 B/pixel + 64 MiB | 196,908 KiB at 1024² (bar 196,608), 738,064 KiB at 2048² (bar 589,824) |

This record finds where the time and memory go and brings both under their bars. No served score,
map value or steering output changes, and no public API changes.

## Where the memory went

Measured with `/usr/bin/time -v` (fresh process, model A, v4x, 1 thread) and `heaptrack --print-peaks`.

1. **The local-refinement snapshot cloned the session's retained planes.**
   - `LocalRefineSnapshot::capture_with_encoding` copied the scale 1–3 pyramids, the phase-A planes and
     the scale-0 planes out of the steering session's `FoldRetention`.
   - The capture ran before the attribution map pass, so the session copy, the snapshot copy and the
     map pass's own planes all coexisted at the call's peak.
2. **The snapshot converted both images to XYB at once** to fill its unretained scale-0 channels,
   holding six full-resolution planes as a transient.

## Where the time went

`perf record`, model A, v4x, 1 thread:

- **The f16 first layer.** On the QUAL-A instrument at 1024², 34.6% of the map arm's samples are in
  `zenpredict::saxpy_matmul_f16`. The map's finite-difference sensitivities run two model forwards per
  read feature, and the pinned zenpredict decodes every f16 weight on every forward with a branchy
  software decode.
- **The source was converted again on every call.** On a build with the moved retention and the f16
  cache, at 2048², sRGB→XYB conversion is 11.7% of the map arm (248.5–250.7 ms per call in diagnostic timing) and
  15.6% of the uncached arm (87.7–87.9 ms per call), although the map arm's reference is prepared.
  - On the Rev5 SDR route, `Fused944Session::planned_features` falls through to
    `compute_folded_v1_372_streaming_impl(source, distorted, ..)`. That walk converts and downscales
    the source on every call.
  - The snapshot then converted the source once more for its unretained scale-0 channels.

## Fixes (zensim, no public API change)

1. **`416b1e8f` — the snapshot moves the retention after the map pass.**
   - `LocalRefineSnapshot::capture_admits` is the single owner of the capture's refusal rule.
     `compute_attribution_input` decides up front whether a snapshot will replace the frozen v2
     density, runs the map pass, then captures with `capture_taking`.
   - `capture_taking` moves the planes out of `FoldRetention` instead of cloning them.
     `FoldRetention::ensure` re-creates exactly the moved buffers on the next walk, which rewrites
     every element before any read.
   - The unretained scale-0 channels are filled one image at a time.
2. **`d4a6864c` — the walk and the snapshot take the source from the prepared reference.**
   - `feature_v2::ref_feed_admits` is now the single admission test of the ref-cached feed (shared
     with `compute_folded_v1_372_with_ref_impl`). It also refuses plans with a restored-cut side pass
     (mapdev / z1max), where a ref-fed walk would assert.
   - `retained_ref_feed_admits` adds a Rev5+ plan whose revision is the process revision. Both
     steering reference constructors then converted at the walk's revision, and from Rev4 on the
     conversion is per pixel (`convert_chunk_rows_is_semantics_not_a_knob`), so the cached planes are
     the bytes the walk would convert.
   - `compute_folded_with_ref_retained` runs the retaining walk with the source side copied from the
     reference; the snapshot copies its unretained scale-0 source channels from the same cache.

The f16 cost is zenpredict's. zenanalyze `417cc785` (E33C_RUNTIME) decodes each f16 layer once at
load and is bit-identical by its own kernel test; it is held FIX-FIRST pending E33C_FIX (the profile
path parses a bake per compare). This record measures both the pinned zenpredict and that patch.

## The gate, re-run

- **Harness:** QUAL-A's own (`costcmp_run.py --qual-grid` parity → timing → fresh-process RSS, the
  QUAL-A runtime pins `runtime-candidates.json`, v4x, 1 thread), classified by
  `qual_a_report.runtime` with the report's spatial and memory bars.
- **Tree:** `d4a6864c` (both fixes) on main `0f630e61`, built twice: against the pinned zenpredict
  `05de3cbc` (`instrument-gate-pinned`, `95054c9c…`) and against zenanalyze `417cc785` through a local,
  uncommitted path patch (`instrument-gate-f16`, `a9804ecf…`).
- **Run quality:** quiet gate unchanged (load1 < 2, no foreign build or training); every cell kept its
  first 32 rounds, 0 excluded, `zenbench_gate_clean`; admitted load1 1.80–1.99. Parity
  (`PREFLIGHT_PASS.json`) passed for both builds, including the harness's own check that each map arm
  serves its model's uncached score bits. `run-heavy` peak RSS 0.61 GiB.
- **Before:** QUAL-A's registered run (same harness, same rules), on the QUAL-A instrument.

Spatial cost, cached-reference score+map p95 ÷ uncached p95 (bar ≤ 3×):

| size | model | before (QUAL-A) | after, pinned zenpredict | after, zenpredict 417cc785 |
|---|---|---:|---:|---:|
| 1024² | A | 90.53 / 22.24 ms = **4.07×** | 89.64 / 22.15 = **4.05×** | 58.31 / 22.14 = **2.63×** |
| 1024² | C | 125.84 / 22.38 = **5.62×** | 125.17 / 22.14 = **5.66×** | 60.69 / 22.21 = **2.73×** |
| 1024² | seed 0 | 91.28 / 22.28 = **4.10×** | 91.73 / 22.20 = **4.13×** | 58.22 / 21.90 = **2.66×** |
| 2048² | A | 261.62 / 83.09 = **3.15×** | 260.05 / 83.63 = **3.11×** | 226.32 / 83.22 = **2.72×** |
| 2048² | C | 302.91 / 83.68 = **3.62×** | 298.68 / 83.35 = **3.58×** | 232.43 / 83.19 = **2.79×** |
| 2048² | seed 0 | 259.59 / 83.09 = **3.12×** | 259.75 / 84.16 = **3.09×** | 225.49 / 84.14 = **2.68×** |

Peak incremental RSS with a map, fresh-process peak − baseline with inputs, model and scorer loaded
(bar 128 B/pixel + 64 MiB = 196,608 KiB at 1024², 589,824 KiB at 2048²):

| size | model | before (QUAL-A) | after, pinned | after, 417cc785 |
|---|---|---:|---:|---:|
| 1024² | A | 196,908 | 156,416 | 156,196 |
| 1024² | C | 196,832 | 156,396 | 155,936 |
| 1024² | seed 0 | 196,832 | 156,168 | 156,148 |
| 2048² | A | 738,064 | 575,968 | 575,696 |
| 2048² | C | 737,820 | 575,468 | 575,640 |
| 2048² | seed 0 | 738,100 | 575,976 | 575,312 |

**Verdict.**
- **Memory passes on zensim alone**, for every candidate at both sizes. The margin is 40,192–40,672
  KiB at 1024² and 13,848–14,512 KiB (2.3–2.5% of the bar) at 2048². QUAL-A measured a 248–1,008 KiB spread between
  repeated RSS observations.
- **Spatial cost passes only with zenpredict `417cc785`**: 2.63–2.79× for every candidate at both
  sizes. On the pinned zenpredict it still fails (3.09–5.66×). The f16 decode in the finite-difference
  forwards alone costs more than the 1024² headroom: a `perf` profile of the QUAL-A instrument puts it
  at 34.6% of the map call (about 90 ms), and A's map arm must reach 66.4 ms (3 × 22.15).
- So the spatial gate is met on `main` once E33C_FIX lands zenanalyze `417cc785` and the zensim
  zenpredict rev bump. Neither zensim commit here depends on it.

Diagnostic, not gate evidence (contended per-call means, v4x 1 thread, model A, process start
included): the reference feed alone moved the map arm 62.1 → 58.6 ms at 1024² and 248.5 → 232.2 ms
at 2048² with the f16 cache, and 275.9 → 265.3 ms at 2048² on the pinned zenpredict. Moving the
retention instead of cloning it did not reduce time on its own (2048², pinned: QUAL-A 266.1 ms,
moved-retention build 275.9 ms).

## Bit identity

| check | result |
|---|---|
| G-STEER, the registered 135-case engineering packet (`steerfix_packet::engineering_packet`, Rev5, exact neighbour replay), final tree, pinned zenpredict, vs the QUAL-A outputs | **A, C and seed 0: 135/135 rows each byte-identical** as serialized (55,380 blocks per model) |
| same, zenpredict `417cc785` | A and C: 135/135 byte-identical |
| same, moved-retention commit alone (f16 patch) vs the registered E33 and QUAL-A outputs | A and C: 135/135 byte-identical |
| The gate's own cases (the harness's synthetic pair; A, C, seed 0; 1024², 2048²; 1 and 16 threads; three consecutive calls of one session): score, uncached score, feature row, density map, block sums, sensitivities, refinement gains (64×64 grid + 8×8 blocks over the top-left 256²), pre-fix tree vs final tree | **36/36 identical**; also pre-fix vs moved-retention, with and without `417cc785` (36/36 each) |
| QUAL-A parity (`PREFLIGHT_PASS.json`): each map arm serves its model's uncached score bits, strict across the grid | PASS, both builds |
| `prepared_steering_ref_feed_is_bit_identical` (new): one session served with the feed on and refused, 3 calls, serial and parallel, 137×301, 64², 203×97, 256×136; hit counter | PASS; negative controls (one flipped bit in the snapshot's cached copy; one in the walk's feed) both FAIL it |
| `cargo test -p zensim --all-features`, pinned zenpredict and `417cc785` | **923 passed, 0 failed** (27 pre-existing ignores) in both |
| CI-exact `just clippy`, `just lint-scripts`, `cargo fmt --check` | clean |

## Pointers

Raw root `/mnt/v/output/zensim/steerperf-2026-10-10/`:
- `gate-gate-pinned/runtime/`, `gate-gate-f16/runtime/`: parity, timing (zenbench rounds, paired
  analyses, quiet waits) and RSS; `GATE_SUMMARY.json` classifies both.
- `gsteer/`: the G-STEER packets, outputs and logs. `mapdump/`: the map/steering dumps and
  `BINARIES.sha256` (the dump probe source is `provenance/mapdump/`).
- `provenance/`: instrument builds and hashes, `GATE_BUILD_PARENT`, the test/clippy/lint logs,
  `verify-chain.log`, `gate-chain.log` and the diagnostic timing/RSS tables.
