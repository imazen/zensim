# Task: DVIFM — PHASE 3: make the block-visibility family cheap (measure first)

Repo `/home/lilith/work/zen/zensim` (jj-colocated). DVIFM (f956..f985, `zensim/src/dvifm.rs`) is on main,
opt-in. Measured cost (benchmarks/dvifm_block_gates_2026-09-19.md): +31.6 ms (+66%) at 1024², +120.5 ms at
2048², single pinned core. Budget bar: a complete candidate must stay p95 ≤ 50 ms at 1024², ≤ 200 ms at 2048².
Read that gates doc, `dvifm.rs`, and the spec
`/home/lilith/work/zen/zenpapers/docs/iqa-methods/dvifm-zensim-feature-design.md` (§Rust implementation).
`/home/lilith/.claude/CLAUDE.md` (Performance Optimization, archmage/magetypes) and `/home/lilith/work/zen/CLAUDE.md` bind you.
DO NOT start if `~/tmp/devin/dvifm_PHASE2_DONE.md` does not exist — benchmarks need a quiet box.

## Step 0 — measure the split (no optimisation before this)
Using the existing `Phase::DvifmKernel` timing owner + zenbench arms `fold956_csfw`/`fold986_dvifm`
(`zensim/benches/extract_paths_bench.rs`), report at 256²/1024²/2048², 1 pinned thread, ≥30 interleaved rounds:
time in (a) normalise f32→f64, (b) pyramid (blur/downsample/expand/subtract), (c) block scan max/min,
(d) per-block math (`phi_g` powf, `visibility` ln/exp/ln_1p, `hat_memberships`, `m.powf(p)`), (e) ring/alloc.
Also count transcendental calls per block. Write the table to `benchmarks/dvifm_perf_2026-09-19.md` FIRST and
commit it. Let the numbers order the remaining steps; skip a step whose stage is <10% of the added time.

## Step 1 — exact shortcuts (output must stay BIT-IDENTICAL; existing parity/stability tests unchanged)
- g == 1.0 → `phi_g` is identity: no powf. P == 1.0 → no powf; P == 0.5 → sqrt.
- Tied curves (θ_ref == θ_dist, no c_hi): max(v(C_r), v(C_d)) == v(min(C_r,C_d)) for non-increasing v — one
  visibility evaluation on min C, and reuse its `ln` for the F2 bin memberships (careful: F2 uses
  ln(minC + 1e-6); keep each expression's exact arithmetic so bits do not move — if a reuse would change bits,
  do not do it in this step).
- Hoist `c0.ln()`, `c_hi.ln()`, `beta*sharp` into per-level precomputed fields.
- Remove per-row allocation in the ring if Step 0 shows it.
Gate: every existing dvifm test passes untouched; `to_bits` equality vs the pre-change binary on the parity
fixtures and 3 real image pairs at 5 sizes incl. odd dims.

## Step 2 — the INTEGER definition (user directive 2026-09-19: "f64 should not be needed at all, and int log
## mapping and int domain should work just fine with i16")
Build `DvifmMath::Int16` (crate-private, selected by params/bake). It is the intended serving definition; the
f64 path stays only as the research oracle. NO f64/f32 arithmetic anywhere between the input quantiser and the
final per-level division that turns integer sums into the emitted feature value.
- Quantise: G_0 = Y normalised to [0,1] → Q14 in i16 (0..=16384), one rounding rule, stated. Bands
  G_l − E(G_{l+1}) ∈ [−1,1] then fit i16 signed Q14 without saturation; prove the bound in a test.
- Pyramid: [1 2 1]/4 per axis with a fixed integer rounding rule — either widening `(a + 2b + c + 2) >> 2` or
  the two-step rounding average; pick ONE, define it in the Python int reference, never mix. Decimate ↓2;
  expand = zero-insert + 4·B in the same integer rule. i16 lanes (i16x32 on AVX-512, i16x16 AVX2, i16x8
  SSE/NEON/WASM) through magetypes generics; `#[rite]` helpers, one `#[arcane]` entry, no `wide`, no unsafe.
- Block stage entirely integer: |ρ_s−ρ_d| as u16, 5-wide/3-wide sliding max/min, m_b and the 8 corner extrema
  per side as i16; contrast C̃ = min over corners of (φ(max) − φ(min)) as u16/i32.
- INTEGER LOG MAPPING: idx(C) = f(leading-zero count, top k mantissa bits) — shifts/masks only. One small table
  per level (target ≤ 256–1024 entries, L1-resident) holds, at that log index, v(C) in Q15 and the five hat
  memberships in Q15, with integer linear interpolation on the remaining bits. φ_g (general g) and m^P are 1-D
  functions of an i16/u16 → the same log-indexed integer tables (g=1, P=1 bypass the table). Tables are built
  at param-load time from the exact real-valued functions and carried/baked with provenance; C = 0 → v = Q15 one.
  Choose k / table size by MEASURED max error vs the real function over the full u16 domain; report it.
- Accumulate Σ v_b·m_b^P and Σ h_j·m_b^P as integer products in i64 (exactly associative ⇒ bit-identical across
  strip sizes, thread counts, SIMD tiers and architectures — assert this, including an i686 `cross` run if
  `cross` is available). Emit feature = sum / N_b once at the end.
- Reference: write the integer definition as a small pure-Python/numpy-int script under `zensim/scripts/`
  (do NOT edit zenpapers). Parity gate is EXACT integer equality of pyramids, block records, table indices and
  sums — not a tolerance.
- Report deviation of Int16 features vs ExactF64 per feature (max, p99 relative) over ≥200 TRAIN-side pairs
  across sizes (never CID22 validation/gold, AIC-3/4, AIC2026), and whether feature RANK order across pairs is
  preserved (Spearman per feature). Identity → exactly 0; monotone in noise; toggle-off and ExactF64 outputs
  unchanged bit-for-bit.
- The block-stats training cache must be able to emit the Int16 block records too, so constants can be fitted in
  the domain that is served.

## Step 3 — cost gate
Same protocol as the Phase-1 cost gate (two-binary A/B, pinned core, ≥30 paired interleaved rounds, dispersion,
competing processes recorded, anchor drift reported not smoothed; memory via `/usr/bin/time -v`). Report added
ms and B/px at 64², 256², 1024², 2048², 4096² for ExactF64-after-Step-1 and Int16; fit and
report `α + β·pixels` with both terms; never extrapolate between sizes. State plainly whether each variant
meets the 50 ms / 200 ms bars as part of a fold956+DVIFM extraction.

## Hard rules
`jj` only, small commits, **DO NOT PUSH**; `cargo fmt -p zensim` before every commit (`cargo fmt --all --
--check` clean); `just clippy` + `just lint-scripts` clean; heavy commands only through
`~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- … 2>&1 | tee ~/tmp/devin/<n>.log`, one at a time, never
head/tail-truncated; scratch only in `~/tmp/devin/`, never `/tmp`; a second build needs its own
`CARGO_TARGET_DIR`; no public API (pub(crate)/cfg(test)); never relax or `#[ignore]` a test (the 5
`zensim-validate/tests/bake_surface.rs` failures are pre-existing — leave them); refresh
`/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-dvifm <activity>`) every ≤2 min; touch no other
repo; no GitHub writes; zenbench not criterion; benchmark without `-C target-cpu=native`; no `pgrep -f`.

## Reporting
Progress lines → `~/tmp/devin/dvifm3_progress.log`. Terminal file `~/tmp/devin/dvifm_PHASE3_DONE.md`: commits,
the Step-0 split table, per-variant speed + deviation + bar verdict, what was NOT done. Measured numbers only.
