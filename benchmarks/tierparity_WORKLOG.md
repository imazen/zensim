# tierparity lane worklog — SIMD tiers must not change results

Lane: `tierparity` | Workspace: `~/work/zen/zensim--tierparity` | Bookmark: `quarantine/devin/tierparity`
Base: `83a205ad` (`quarantine/devin/optmlp` tip — kept untouched).
Brief: `~/tmp/devin/fitopt/BRIEF_tierparity.md`. Oracle: `~/tmp/devin/fitopt/bitexact.sh`
(golden `best_e3.bin` + `epochs_e3.txt`, 3-epoch H32 fold, produced under `ZENSIM_MAX_TIER=v3`).
Host: AVX-512 dev box (`X64V4xToken`, `X64V4Token`, `X64V3Token` all summonable).
All heavy commands under `~/tmp/devin/heavy --mem 16G --jobs 8 --`,
`CARGO_TARGET_DIR=/var/tmp/tierparity/target`.

## Rule under test

> "the SIMD tier must not change results" — AVX-512/scalar/neon/wasm128 must
> reproduce AVX2 (v3) bit-for-bit: lane structure, FMA-vs-mul+add, reduction
> order, epoch output, serialized weights. AVX2 arithmetic unchanged.

## Step 1 — enumeration + measurement

Tier-dispatched sites (incant!/arcane/magetypes + manual gates):

| crate | file | kernels |
|---|---|---|
| zensim-train-core | `src/simd_encoder.rs` | `accumulate_rows_f32`, `dot_product_f32`, `apply_leaky_relu_f32` (+ f64 encoder forward/backprop wrappers) |
| zensim-validate | `src/simd_mlp.rs` | `forward`, `backprop` (manual `avx512_allowed()` v4 gate + `_avx2`/`_scalar`) |
| zensim-validate | `src/adam_simd.rs` | `adam_update` (+ rsqrt variant) |
| zensim-validate | `src/mlp_train/mod.rs` | `add_l2_grad_layer1` — `is_x86_feature_detected!("avx")` gate, **already tier-invariant** (deliberate mul+add, no FMA) |
| zensim-validate | `src/pool_head.rs`, `hybrid_head.rs`, `per_sample_alpha_head.rs` | no SIMD dispatch — scalar only |

Root cause measured: `magetypes` generic `mul_add` is **unfused** `a*b+c` on
`ScalarToken`/`Wasm128Token` backends (`scalar.rs:189/573`, `wasm128.rs`) but
fused on neon (`vfmaq`) and x86 v3/v4 (`vfmadd`). `reduce_add` association order
also differs per backend (v4: `_mm512_reduce_add_ps` 16-lane tree; v3 `f32x16`:
`tree8(lo)+tree8(hi)`; scalar f32x8: `((a0+a4)+(a1+a5))+((a2+a6)+(a3+a7))`).
Two orthogonal defect classes:

- **Boundary/tail structure**: v4 kernels processed `n%8` remainder with fused
  ops where v3 uses fused domain `[0, n-n%4)` + mul+add tail of `n%4`
  (adam), or processed `nh%16`/`nh%8` tails differently (encoder/MLP).
- **Arithmetic class**: scalar/wasm128 `mul_add` unfused vs v3 fused;
  generic `reduce_add` order differs.

## Step 2 — failing tests (written BEFORE fixes; all failed pre-fix)

- `simd_encoder::tests::encoder_all_tiers_bit_identical`,
  `encoder_dispatch_bit_identical_under_token_permutations` — accumulate_rows +
  dot_product over `#[magetypes]` variants (v4/v3/neon/wasm128/scalar hand+dummy)
  across `archmage::testing::for_each_token_permutation` (9 host perms).
- `simd_mlp::tests::forward_all_tiers_bit_identical`,
  `backprop_all_tiers_bit_identical`,
  `dispatch_bit_identical_under_token_permutations` — direct per-tier calls +
  dispatch rerouting; `n_hidden ∈ {8, 20, 32}` to hit `nh%8∈{4}` boundaries.
- `adam_simd::tests::adam_all_tiers_bit_identical`,
  `adam_dispatch_bit_identical_under_token_permutations` — `n ∈ {1..17}` sweep +
  aligned/misaligned/late-step cases.
- `tests/tier_parity_e2e.rs::train_bit_identical_across_all_host_tiers` — full
  `train_mlp` 24×20 model, 240 rows, 4 epochs, both `per_sample_alpha_head`
  settings; serialized weight bytes + epoch lines (with `| t=` normalized)
  compared across all 9 permutations.

Pre-fix measured divergence (negative controls):

- encoder `accumulate_rows`/`dot_product`: v4 tail blocks and scalar/wasm
  unfused `mul_add` diverged; dispatch perms changed `y` bits
  (`0xc0f666b7 → 0xc0f666bc → 0xc0f666bd`); `leaky_relu` clean.
- adam: v4 diverged exactly when `n%8 >= 4` (masked-256 remainder group);
  scalar/wasm/neon-generic diverged on unfused `mul_add`.
- mlp: `forward_avx512` y-reduction (16-lane reduce vs v3 2×8 tree) diverged
  at nh=32; backprop gw2/gw1 tails likewise.
- e2e pre-fix: alpha-head weight bytes differed across perms (3 distinct
  hashes: v4/v3/scalar); plain head bytes coincided at 16×8 (f32 bake
  sub-ulp), so the e2e model uses `n_hidden=20` to hit the v4 boundary.

## Step 3 — fixes (AVX2 arithmetic untouched)

`zensim-train-core/src/simd_encoder.rs`:
- `accumulate_rows_f32_v4`: 16-chunk fused domain → match v3's
  `[0, nh-nh%8)` fused + `nh%8` mul+add tail via masked ops for the +8 group.
- `dot_product_f32_v4`: lane accumulation + pairwise reduction order made
  identical to v3 (`((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7))` per 8-lane half,
  halves combined, sequential tail).
- `#[magetypes(neon)]` → `#[magetypes(neon, -scalar)]` + hand-written
  `_scalar`/wasm128 variants using scalar `f32::mul_add` over the canonical
  fused domain (suppresses the macro's auto scalar variant).
- wasm128 hand-written fns gated `#[cfg(target_arch = "wasm32")]`.

`zensim-validate/src/simd_mlp.rs`:
- `forward_scalar`/`backprop_scalar`: v3-shaped fused domain + mul+add tail
  (was: different tail/arithmetic).
- `forward_avx512`/`backprop_avx512`: masked-512 ops for the +4 f64x4 group of
  `n%8`, y-reduction restructured to v3's 2×8-lane tree, gw1/gw2 tails aligned.
- `add_l2_grad_layer1` verified already invariant; untouched.

`zensim-validate/src/adam_simd.rs`:
- v4 remainder: 8-chunks then, when `n%8 >= 4`, a per-element scalar
  `f64::mul_add` group of 4 (X64V4Token has no `F64x4Backend`), then `n%4`
  mul+add tail — bit-identical since Adam is elementwise.
- `#[magetypes(neon, -scalar)]` + scalar-fused fallback; wasm128 gated.

`zensim-validate/benches/adam_bench.rs`: `#[allow(dead_code)]` on the
`#[path]`-included test helpers (bench targets compile `cfg(test)` items;
`#[test]` liveness doesn't propagate to helpers in the bench's non-harness
compile).

## Post-fix results

- `cargo test -p zensim-train-core --lib`: 44 passed, 0 failed, 1 ignored.
- `cargo test -p zensim-validate --lib`: 265 passed, 0 failed.
- Direct parity tests: encoder 4 pass (+1 ignored corpus), adam/simd_mlp
  14 pass — every permutation now yields canonical bits (encoder
  `bits=0xc0f666bc`; mlp `y=4.61471064194085e1`, `0x404712d462163212`).
- e2e: weight bytes identical across all 9 perms, both head paths
  (plain hash `d97effab934fc76c`, alpha `8f7da716e521bb5f`); epoch lines
  identical after normalizing the `| t=` wall-clock field (tiers differ in
  speed by design).

## Step 4 — oracle

- capped (`ZENSIM_MAX_TIER=v3`): `epoch lines identical: True | weights identical: True`
- uncapped (script minus the export): `True | True` — AVX-512 dispatch now
  reproduces the AVX2 golden.
- negative control: pre-fix base binary (`zensim--tb-base` @ `83a205ad`)
  uncapped → `epoch lines identical: False | weights identical: False`, rc=1.

## Step 5 — wall-clock before/after (uncapped fold, taskset 1 core)

Fixed fold = same command the oracle runs, minus weight-compare. First set
(taskset -c 9, min of 3): before `25.93/47.69/61.23` → min 25.93 s;
after `49.40/34.90/45.93` → min 34.90 s. Interleaved A/B set:
before `51.14/52.18/46.54`, after `36.45/100.05/36.15`.

**Caveat: inconclusive.** Host loadavg ~23–25 with many concurrent users;
within-set spread is ~2× both ways (before max/min = 2.36, after = 2.74 in
the interleaved set). Medians: before ~47.7, after ~36.5 — sign conflicts
with the mins. Treat as "no reliable regression measured"; kernel-level the
AVX-512 path now does strictly more scalar work at tails (the price of bit
parity), bounded by tail sizes `n%8`/`nh%8` — O(tail) per call, not O(n).

Repro commands:
```
taskset -c 9 /var/tmp/tb-base/target/release/zensim_mlp_train \
  --train-parquet ~/tmp/devin/fitopt/exp/fit.parquet \
  --val-parquet   ~/tmp/devin/fitopt/exp/val.parquet \
  --max-features 944 --hidden 32 --epochs 3 --pairs-per-epoch 50000 \
  --init-seed 1101 --sample-seed 101 --no-auto-eval --out ~/tmp/devin/tierparity/pb.bin
# same with /var/tmp/tierparity/target/release/zensim_mlp_train → pa.bin
```
logs `/var/tmp/tierparity/logs/perf_before_{1..3}.log`, `perf_after_{1..3}.log`,
interleaved `pb_t_{4..6}`, `pa_t_{4..6}` (same directory; moved there from `/tmp` at landing, sha256 in
`SHA256SUMS.moved_from_tmp`).

## Step 6 — `bench_mlp_kernels` clippy E0433

Verified at base: `zensim-validate/benches/bench_mlp_kernels.rs` already
contains `mod tier_cap { pub fn avx512_allowed() -> bool { true } }` —
identical to base rev (diff empty). The CI-exact invocation
`cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings`
**passes on base `83a205ad`** and on the fixed tree; the described E0433 does
not reproduce at this revision (shim predates it). My own additions surfaced
latent bench warnings (fixed: `#[allow(dead_code)]` on adam test helpers,
`needless_range_loop` → iterator form in `forward_avx512` tail,
`type_complexity` annotations in tests). Current tree: CI-exact clippy clean.

## Step 7 — feature-extraction tier audit (MEASUREMENT ONLY)

Harness: `zensim-validate/src/bin/tier_audit_features.rs` — decodes PNG pairs,
`research::extract(&Request::everything(), ref, dist)` (1502 slots, current
revision = Rev3 via `ZENSIM_FORMULA_REV=3`), cumulative token disables
(`v4x`→`v4`→`v3` baseline→`scalar`), bit-compares each slot vs v3.

Sample (restated at landing from the independent review; the first version of this entry said "8 real TRAIN
pairs, 8 distinct sizes", which was wrong): 8 pairs, 7 distinct sizes.
- 4 TRAIN-role pairs: kadid 512×384, tid 512×384, konfig 384×512, konjnd 640×480. Two of them are
  pixel-identical to their reference (kadid `I01_01_01.png` and konfig `SRC01_colordiffusion_0.png`), so
  "zero v4 differences" on those two says nothing about v4; only tid (512×384) and konjnd (640×480) are
  informative TRAIN pairs.
- 4 AIC-3 CTC pairs (853×945, 945×840, 1192×832, 2000×2496): bank role `fold+potential`, not TRAIN. Pixels only were read.
- 512×384 occurs twice (kadid, tid), hence 7 distinct sizes.
The audit therefore has 2 informative TRAIN pairs, not ≥6. Not re-run at landing: the featcanon lane owns the
feature-tier audit.
The harness decodes with the `image` crate, not the imazen decode route, and does not check its decoded pixels
against the bank's recorded pixel hashes (open, handed to featcanon).
Artifacts: the pair list, log and 883 KB TSV were never committed (over the 30 KB cap). They live in
`~/tmp/zensim-paper/rev4/tierparity_audit/` (`audit_pairs.tsv`, `tier_audit_rev3.log`, `tier_audit_rev3.tsv`;
sha256 in the landing DONE file). The pair list was first written to `/tmp/audit_pairs.tsv`, now
`/var/tmp/tierparity/logs/audit_pairs.tsv`.

Repro:
```
cargo build --release -p zensim-validate --bin tier_audit_features
RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=3 taskset -c 9 \
  .../tier_audit_features benchmarks/…pairs.tsv > out.tsv
```

Per-family rollup (slot-rows = per-pair differing slots; pairs = #pairs
with ≥1 diff; max rel vs v3):

| family | v4x | v4 | scalar |
|---|---|---|---|
| basic | 685 diffs 6pr 1.34e-5 | same | 839 6pr 3.36e-3 |
| iw | 432 6pr 8.38e-9 | same | 432 6pr 3.53e-4 |
| masked | 432 6pr 7.33e-9 | same | 432 6pr 3.46e-4 |
| peaks | 216 6pr 9.84e-9 | same | 393 6pr 8.23e-4 |
| v2 | clean | clean | 2112 8pr 1.47e-3 |
| append | clean | clean | 1098 8pr 9.31e290* |
| append2 | clean | clean | 78 8pr 1.62e-4 |
| arttype | clean | clean | 117 6pr 1.08e-2 |
| csfw | clean | clean | 48 6pr 1.98e-4 |
| dvifm | clean | clean | 180 6pr 4.72e-6 |
| gmsbank | clean | clean | 1080 6pr 2.01e-5 |
| gridblk | clean | clean | 519 6pr 4.28e-2 |
| ringbasis | clean | clean | 401 6pr 2.68e-2 |
| tailhist | clean | clean | 242 6pr 5.74e-1 |

*9.31e290 = denormal-vs-denormal ratio on `append_texture_dissim_s3_b`;
substantive max ≈ 0.67 (`tailhist_art_p99`/`append_texture_dissim`).

Kernel attribution:

- **v4/v4x (identical outputs — simd_ops has no v4x tier; blur's `+v4x`
  variant produced bit-identical results to `+v4` on these inputs)**:
  confined to SSIM-derived families. Responsible: generic `f32x16` bodies in
  `zensim/src/simd_ops.rs` — `ssim_signal_inline_{both,mask,iw}` (lines
  3300–3460, Rev3 retained-signal pools called at `ssim_form.rs:1674-1676`
  and `fused.rs:4098`), `ssim_channel_inline_{both,mask,iw}`,
  `edge_diff_channel_inline_both`, `build_inline_mse`. Mechanism:
  `f32x16::reduce_add` = `_mm512_reduce_add_ps` on v4 vs `tree8+tree8` on v3 —
  per-16-chunk sum order differs; `mul_add` is fused on both so element
  values are identical. `basic_var_loss/tex_loss/contrast_*` additionally
  ride the moment planes from `blur.rs` box_blur + `sq_sum_into`/`mul_into`.
  Content-dependent: kadid/konfig (mildest distortions) showed 0 diffs.
- **scalar (all 14 families)**: same reduce-order issue PLUS unfused generic
  `mul_add` on `ScalarToken` (`a*b+c`, no fma) — changes element values in
  every kernel that calls it (mask weights `1/(1+k·a)`, IW `1+k·a`, dissim
  terms), so planes themselves differ and every downstream pool/stat shifts.
  Dispatched sites involved: `simd_ops.rs` inline pools + `blur.rs`
  `box_blur_{h,v}_*` (`incant! [v4x,v4,v3,neon,wasm128,scalar]`) +
  `feature_v2.rs` `n_block_kernel*`/`#[magetypes]` pools +
  `dvifm.rs` `dvifm_{push_rows,finish}_entry` + `streaming.rs` callers.

Proposed fixes (NOT implemented — formula-revision decision required). Read the tradeoff on #1 before acting:
the Rev4 feature bank was extracted on an AVX-512 host with `extract-native-admission` `7c7ffbbf` (uncapped), so it carries
the AVX-512 (v4) reduction order in basic/peaks/masked/iw. The independent review reproduced 6 stored rows exactly
with the native binary (0 of 5,664 cells differ) and found the AVX2 path differing in 288–298 of 944 f64 values per
non-identity pair (3–17 cells per pair after the bank's f32 storage rounding). Canonicalizing to the AVX2 (v3) order,
as #1 proposes, would change those values on every non-identity row and mean re-extracting the ~249k-stimulus bank and
refitting everything trained on it. The alternative that keeps the bank valid is to make the v3, scalar, NEON and wasm
tiers reproduce the v4 tree (`_mm512_reduce_add_ps` order) by hand.
1. Canonicalize each generic body's accumulation to the v3 structure:
   fixed-width two-level reduction (e.g. always reduce 16-chunks as
   `tree8(lo)+tree8(hi)` regardless of native width) + identical scalar tail
   boundaries — the same recipe applied to the trainer kernels.
2. Make scalar/wasm128 `mul_add` paths fused (explicit `f32::mul_add` scalar
   fallbacks, as done in `simd_encoder.rs`/`adam_simd.rs`) — requires deciding
   that fused is the canonical semantic for those kernels.
3. `blur.rs` box_blur tails: fix column-group remainder handling to one
   canonical width order (v3's) across tiers.

## Notes for reviewers

- `quarantine/devin/optmlp` untouched (workspaces are separate checkouts).
- Timing numbers above are the honest record; do not quote the mins as a
  regression without the load caveat.
- The audit binary is measurement-only; it panics loudly if a token cannot
  be disabled (`x86-64` SSE baseline is compile-time-guaranteed and is not
  needed — disabling v3 already routes to scalar).
