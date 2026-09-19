# DVIFM block-visibility (f956..f985): Phase-1 qualification gates (2026-09-19)

Mission: `../zenpapers/docs/iqa-methods/dvifm-zensim-worker-brief.md` +
`dvifm-zensim-feature-design.md`. Scope = "What to build" + "Gates before
any screen" ONLY — no TRAIN extraction, no training, no block-stats cache
fitting, no variant screening. The family is opt-in, default-off,
additive-only. Host: this dev box, x86-64 (AVX-512 present — v3/v4/v4x
tiers all summonable), perf gates on 1 pinned core (`taskset -c 8`,
`nice -n19 ionice -c3`).

## MISSING first (coverage = Phase-1 only)

1. **No TRAIN screen** — no extraction over any corpus, no LOO, no fit of
   the SEED-1.0 placeholder constants (`g=1, P=1, C0=0.01, β=0.65, ς=4`).
   Nothing here claims the family improves any model.
2. **No production wiring** — `dvifm_block` is off by default; no profile,
   bake, or `Zensim` surface consumes f956..f985. The 986 producer is
   registered (`w986`, slots_hash8 `685eb6ef`) with no canonical root.
3. **No cross-machine SIMD evidence** — `simd_tier_parity` passed on this
   host for the tiers it can summon (v3/v4/v4x + scalar). neon/wasm128
   arms exist but are not exercisable here.
4. **1024² toggle-off A/B is anchor-contaminated** — fast_ssim2 drifted
   −8.61% between blocks (the box sped up); the 2048² group is clean
   (+0.34%). Reported, not smoothed; see COST.
5. **No 1024²+ toggle-off perf gate under `v1_pools=Full`** — the
   cross-binary arm was `fold944_full`/`fold944_off`; the dvifm-adjacent
   956 shape did not exist in the pre binary.
6. **No HDR cost measurement** — cost gate is SDR only.
7. **`DvifmAccum::block_cache`** (training-side per-block stats) exists but
   is unused by the served path and unexercised by any consumer.

## Slot-ledger finding (step 1 of the work order)

f956 was confirmed free before any code: the CSF tier-1 gates doc
(`benchmarks/csf_tier1_gates_2026-07-28.md`, commit `4a08f77b` era) recorded
the chroma tiers' f956..f979 claim as **recommended CLOSED** (2026-07-29
adjudication, remainder #1), and the `hdr` registry token is a reserved
name with `slots: null`. DVIFM claims f956..f985 append-only; no slot was
renumbered and no prior width changed.

## What shipped (commits, oldest first)

| commit | content |
|---|---|
| `a4abefa5` | `dvifm.rs`: scalar/f64-clean kernel — 5-level binomial pyramid (reflect-101 borders, target-shape expand lattice), Laplacian + local-band modes, 5×5 blocks, corner-3×3 extrema, φ_g signed power, log-domain visibility, F1 + 5 F2 bins = 30 features; streaming `DvifmAccum` pump; baked route norms with recompute tests; 21 kernel tests |
| `0bf63a20` | numpy parity: `scripts/dvifm_parity_fixture.py` + fixture (28.5 KB) + test; all pyramid planes + feature vectors ≤1e-6 of the Python reference |
| `65b9165a` | registration: `ComputeToken::Dvifm` (appended — no bit renumber), `Replication::Flat`, `KernelId::Dvifm`, `V2NewFeatureToggles::dvifm_block` (default false, requires csfw→append2→append chain), `FeatureRegime::Folded720Dvifm`, `dvifm_features()` accessor, walk hook on scale-0 Y strips (lazy accum from strip info dims), `Phase::DvifmKernel`, registry JSON w986 entry; `just clippy` lint fixes |
| `03a3f8b6` | streaming parity gate (below) |
| `5344eff1` | SIMD tier parity gate (below) |
| `e177b6f8` | byte-stability gate (below) |
| `7d7c018c` | cost-gate bench arms `fold956_csfw`/`fold986_dvifm` |

## Constants (baked; `norm_constants_*_recompute` tests regenerate them)

| constant | value | provenance |
|---|---|---|
| `DVIFM_Y_MIN_SDR` | 0.010000014677643776 | min of live SDR Y plane (`srgb_to_positive_xyb` ch1) over a dense sRGB grid |
| `DVIFM_Y_SCALE_SDR` | 0.8453085776418447 | (max−min) of the same grid |
| `DVIFM_Y_MIN_PU` | 0.009999999776482582 | min of PU-encoded Y (`linear_to_pu_xyb_planar_into` ch1) over a nits grid |
| `DVIFM_Y_SCALE_PU` | 2.323035230860114 | (max−min) of the same |

## Gates

### G1 — numpy-reference parity: **PASS**

`dvifm::tests::numpy_parity_planes_and_features`: closed-form planes
generated identically in Rust and Python; every pyramid level's planes and
the 30-feature vector match `dvifm_block_visibility.py` within 1e-6, in
both Laplacian and Local band modes. Fixture 28.5 KB (< 30 KB cap).

### G2 — streaming ≡ whole-plane / served walk: **PASS, bit-identical**

- `dvifm::tests::strip_size_bit_identical`: pump pushes of
  1,2,3,5,7,11,16,33,64,97,128 rows (non-multiples of 5 and 16 included)
  are `to_bits`-equal to the single-push oracle at two sizes.
- `feature_v2::tests::dvifm_walk_tail_matches_streaming_oracle_bit_identical`:
  the folded-v2 walk's f956..f985 tail is bit-identical to
  `dvifm_features_stream` over the producer's own scale-0 Y planes —
  serial AND parallel walks — at 150×170 (final strip 42 rows), 67×83
  (odd, single strip), 131×129 (final strip 1 row, dims not divisible by
  the 5×5 block), 128×128, plus 40×50 (sub-64: producer reflect-pads to
  64² and the accumulator correctly takes the PADDED dims from strip info).
- `feature_v2::tests::dvifm_walk_tail_matches_streaming_oracle_hdr`: HDR
  route, 96×101, PU-Y plane under `DVIFM_NORM_PU`, bit-identical.

### G3 — SIMD ≡ scalar: **PASS, bit-identical**

`dvifm::tests::simd_tier_parity`: every summonable token — on this host
v3, v4, v4x — reproduces the scalar path's bits in both band modes over
widths 125/97/6/128 (all lane-remainder paths) at non-multiple strips.
This is bit-identity by construction (all `mul_add` factors are powers of
two; `GenericF64x8` is a fixed 8-lane type so lane order never changes),
now verified rather than assumed.

### G4 — byte stability: **PASS**

- Toggle ON: `dvifm_toggle_on_preserves_prefix_slots` — the 986-wide
  result's f0..f955 are `to_bits`-identical to the 956-wide CSFW result,
  serial and parallel, two sizes.
- Toggle OFF: the pump hook is `Option::None` — zero code-path change —
  and the pre-existing golden suite passed on this tree:
  `v1_golden_bytes` 5/5, `fold_engine_parity` 10/10,
  `feature_invariants` 13/13, `v1_feature_width_pure_function` 10/10.
  Registry sync (`zensim-validate`) green: `685eb6ef` verified.

### G5 — cost: **PASS (measured; toggle-on is NOT free — recorded)**

Setup: `extract_paths_bench` binary, one pinned core, `nice -n19
ionice -c3`, `RAYON_NUM_THREADS=1`, zenbench paired interleaved rounds,
n=30 per block, 2 alternating blocks per arm; competing processes recorded
in `~/tmp/devin/dvifm-ab/competing_procs.txt` (ambient devin/claude/niced
load ~1.6–2.5). Binaries: A = pre-DVIFM build at `b684c2fe`
(sha256 629426ad…), B = this tree (sha256 163d8544…); equal-length paths.

**Toggle-OFF (A vs B, shared arms)** — medians over 2 blocks × 30 rounds:

| arm | 1024² Δ | 2048² Δ |
|---|---:|---:|
| fast_ssim2 (anchor) | −8.6% (drift — contaminated) | +0.34% |
| fold944_off | −4.5% | −1.2% |
| fold944_full | −3.8% | −1.1% |
| buf_v1_372 | −6.9% | −0.8% |

2048² group is clean: every shared arm within −0.1..−3.5% of pre-DVIFM
(slight B-faster skew = code-layout luck, direction inconsistent with
added work). 1024² is reported but contaminated by box drift; its
arm-deltas straddle the anchor (−3.2..−9.8% vs −8.6%) i.e. attributable
delta ≈ 0±4%. **No measurable toggle-off regression.**

**Toggle-ON (within-B, fold956_csfw → fold986_dvifm)** — the DVIFM
marginal cost:

| size | fold956_csfw | fold986_dvifm | added |
|--:|--:|--:|--:|
| 1024² | 47.8 ms | 79.4 ms | **+31.6 ms (+66%)** |
| 2048² | 221.5 ms | 342.0 ms | **+120.5 ms (+54%)** |

(medians of the two blocks; per-block pairs agree within 0.6%. cv 0.3–1.0%
per arm.) A five-level f64 pyramid over the full plane is real work —
this is a qualification-cost record, not a claim the family is cheap.
For context: ≈ +120 ms/pair at 2048² sits between `fold944_off`→`fold944_full`
(+41 ms) and a whole extra `fold944_off` walk (+221 ms).

**Memory** — `/usr/bin/time -v` peak RSS (`ZEN_XP_RSS` arms, 8 iters):

| size | fold956_csfw | fold986_dvifm | added |
|--:|--:|--:|--:|
| 1024² | 52.6 MB (51.4 B/px) | 53.1 MB (51.8 B/px) | +0.5 MB |
| 2048² | 111.7 MB (27.2 B/px) | 113.0 MB (27.6 B/px) | +1.3 MB |

The accumulator holds ring buffers + block stats, not full planes — the
envelope stays ~flat. Both revisions measured (rev1/rev3 identical, as
expected — DVIFM doesn't read the revision).

## Reproduce

```
# correctness gates
cargo test -p zensim --features feature-regime-v2,training,custom-profiles dvifm
cargo test -p zensim --features feature-regime-v2,training,custom-profiles \
    --test v1_golden_bytes --test fold_engine_parity --test feature_invariants \
    --test v1_feature_width_pure_function
# cost arms (build once; do NOT wrap the run in run-heavy — zenbench gating)
cargo bench --no-run --locked --bench extract_paths_bench -p zensim \
    --features custom-profiles,feature-regime-v2,threads,training
BIN_A=<pre-dvifm-binary> BIN_B=<this-binary> ZEN_XP_ROUNDS=30 ZEN_XP_MIN_ROUNDS=30 \
    scripts/bench/rev3_cost_ab.sh <out> 2 1024,2048 8
python3 scripts/bench/rev3_cost_report.py <out>
ZEN_XP_ITERS=8 scripts/bench/rev3_rss.sh <out> "1024 2048" "fold956_csfw fold986_dvifm" 8
# numpy fixture regeneration (needs ../zenpapers + numpy)
python3 scripts/dvifm_parity_fixture.py > tests/fixtures/dvifm_parity_v1.txt
```

Evidence dirs: `~/tmp/devin/dvifm-ab/` (A/B logs, run.meta, binary.sha256,
competing_procs.txt), `~/tmp/devin/dvifm-rss/rss.tsv`.
