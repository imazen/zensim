# DENSEPOOL brief: bit-identical speedup of Rev5 f64 pool updates (2026-10-24)

Self-contained brief for a worker lane. Evidence and rationale:
[dense_pool_precision_2026-10-24.md](dense_pool_precision_2026-10-24.md). Prototype diff:
[dense_pool_proto_2026-10-24.patch](dense_pool_proto_2026-10-24.patch) (applies cleanly to main `524a04c2`;
`git apply --check` verified).

## Goal

Make Rev5 scoring faster **without changing a single output bit** on any tier, by restructuring how
the sixteen-virtual-lane f64 pools (`featcanon::Lanes16F64`) are updated. No formula revision, no
model replay, no test expectation changes.

Out of scope: f32 lanes or any other value change (that is a future Rev6 decision for the owner);
Rev1-Rev4 arithmetic; public API; the scalar/NEON/wasm128 sibling bodies unless the same change is
trivially bit-identical there too.

## Rules

Read `CLAUDE.md` (repo) first. Work on main in the primary checkout with `.workongoing`; push only
via `scripts/safe_push.sh -r <rev>`; heavy builds/tests through
`~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- ...`. Profile with `perf`, not println timers.
Never relax a test. If any parity check fails, the change is wrong: fix it, do not adjust the test.

## Why it works

The pool update in `dense_block_kernel_canon64_vec` (`zensim/src/feature_v2_oracle.rs`, ~line 1533;
generated for v4x/v4/v3 by `#[magetypes]`) does, for every 8 columns, 29 x (widen 8 f32 -> f64,
load 2 ymm from a stack pool, add, store). The 29 x 16 f64 lanes (3.7 KB) cannot live in registers,
so ~62% of the kernel's samples are this load/add/store traffic (perf annotate, i265).

Each lane's value is a sequential sum: lane `l` of pool `j` adds `(f64)term_j[x]` for every column
`x` with `x & 15 == l`, in ascending `x`, starting from 0.0 at the start of each row, then the
`dense_elem_canon` tail adds, then `fin()`'s fixed tree. Any schedule that preserves that per-lane
sequence is bit-identical. The prototype buffers the 29 term vectors of `ILV_K = 16` chunks in an
L1-resident stack array, then for each pool loads its two `f64x8` halves once, adds the buffered
chunks in order (even chunk -> lanes 0..8, odd chunk -> lanes 8..16, since the block start is even),
and stores once. Pool memory traffic drops by ~16x; the widening and adds are unchanged.

A whole-row buffer (29 x width) was measured SLOWER (40.6 ms vs 33.7 ms at 1 MP); keep the block in L1.

## Measured prototype result (i265, v3, 1 thread, default B `89cb4861`, `ZENSIM_FORMULA_REV=5`)

| Geometry | main | blocked interchange |
|---|---:|---:|
| 256^2 | 2.262 ms | 2.073 ms (-8.4%) |
| 1024^2 | 33.70 ms | 30.85 ms (-8.5%) |
| 4096x2048 | ~284 ms | 260.3 ms (-8.6%) |

Score + all 720 emitted slots bit-identical on 10/10 pairs (5 synthetic geometries incl. odd widths,
5 gb82 photos). v4/v4x NOT measured (i265 has no AVX-512).

## Tasks, in order

1. **Land the dense kernel change properly.** Replace the prototype's compile-time
   `ZENSIM_PROTO_POOL` switch with an unconditional path taken when `P::LANES == 16` (Rev5);
   Rev4's `LanesF64` (8 lanes) path keeps its current code byte for byte. Keep the pool updates
   as `f64x8` repr ops: the AVX-512 build otherwise turns per-lane RMW into
   `vgatherqpd`/`vscatterqpd` (a measured 4.3x regression, see the comment above the kernel).
   The tail (`width % 8`) must keep running after the last block is flushed, so its adds stay last.
   Try `ILV_K` 8/16/32 and keep the fastest; record the numbers.
2. **Same treatment for the other two Rev5 f64-pool kernels** (each ~5-10% of score time):
   `fused_vblur_ssim_canon64_vec` (`zensim/src/fused.rs` ~5635; pool updates via
   `row.<name>.chunk8(x0 & (P::LANES - 1))` ~5786-5971, some conditional on feature flags) and
   `gradient_block_kernel_canon64_vec` (`zensim/src/feature_v2_kernels.rs` ~4928). Profile first;
   do it only where the pool RMW is the hot part.
3. **Optional, measure first: `OnlineMoments::block`** (`feature_v2_kernels.rs` ~217) becomes ~10%
   of score time after task 1. Its two-pass 16-element block is a serial f64 dependency chain.
   Computing several blocks at once with one block per SIMD lane, each lane running the identical
   scalar sequence, is bit-identical; the Chan/Pebay `merge` must stay serial in raster order.

## Acceptance gates (all required, record commands and output in the report)

* Bits: `cargo test --release -p zensim --features feature-regime-v2 --test featcanon_rev5_parity`
  (every token permutation; geometries 64^2, 97x63, 131x65, 255x129). Also `featcanon_tier_parity`,
  `featcanon_rev4_contract`, `fold_engine_parity`, `research_engine_parity`, `cross_tier`.
* Run the parity tests on an **AVX-512 host** (dev = Zen 5, or r7900x) so v4x/v4 are exercised,
  and on i265 or another AVX2 host. Say explicitly which tiers were covered.
* Probe bit check (appendix): base vs patched dumps `cmp`-identical on all ten pairs, on both hosts.
* Full `cargo test -p zensim --all-features --release`, CI-exact `just clippy`, `just lint-scripts`.
* Speed: interleaved before/after on a quiet host (load < 2), pinned core, at 64^2, 256^2, 1024^2
  and 4096x2048, 1 and 8 threads. For a landing claim use the paired SPEEDQ owner
  (`zensim-bench/benches/speedq/`, `just bench-speed-matrix`) rather than the probe alone.
  Report run-heavy peak RSS. No "faster" claim for a cell whose paired interval includes zero.
* Commit message and a short results doc `benchmarks/densepool_<date>.md` (commit hashes,
  host, tier, numbers, failures). Update the CHANGELOG `[Unreleased]` with a perf entry.

## Report back

Per task: landed commit (verified with `git merge-base --is-ancestor <sha> origin/main`), tiers and
hosts covered by parity, before/after table, and anything that did not reproduce. Missing evidence
is reported as missing, not as a pass.

## Appendix: probe used for the prototype measurements

Out-of-repo crate (build it anywhere outside the workspace, e.g. `~/tmp/zensim-probe`). Edit the
zensim `path` to the checkout. The B bake is the shipped `zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin`, sha256 `89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.

Run: `ZENSIM_FORMULA_REV=5 PROBE_BAKE=<bake> PROBE_W=1024 PROBE_H=1024 PROBE_N=20 taskset -c 2 ./zensim-probe`
(timing), add `PROBE_N=1 PROBE_DUMP=<file>` for a score+feature bit dump, and
`PROBE_IMG=<png>` for a real image (deterministic distortion built in). gb82 photos come from the
`codec-corpus` cache (`gb82/{baby,bulb,city,dog,flowers}-lossless.png`).

`Cargo.toml`:

```toml
[package]
name = "zensim-probe"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
image = { version = "0.25", default-features = false, features = ["png"] }
zensim = { path = "<checkout>/zensim", features = ["custom-profiles", "feature-regime-v2"] }
zenpredict = { version = "0.2.0", git = "https://github.com/imazen/zenanalyze", rev = "f0903489300aea240ffae2e3f1359f8f2b9a0ba4" }
zenpredict-bake = { git = "https://github.com/imazen/zenanalyze", rev = "f0903489300aea240ffae2e3f1359f8f2b9a0ba4" }

[profile.release]
debug = 1
lto = "thin"
codegen-units = 1
```

`src/main.rs`:

```rust
//! Diagnostic probe: production seed-0 bake at the process formula revision.
//! PROBE_MODE=score|map|both, PROBE_W/H, PROBE_N calls, PROBE_PAR=0|1, PROBE_BIN.
use std::time::Instant;
use zensim::{BakeScorer, RgbSlice};

fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    // Byte-identical to zensim-bench/benches/ssim2_speed_bar.rs::test_pair.
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [base.wrapping_add(tex), base.wrapping_add(edge), (255 - base).wrapping_add(tex / 2)];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let bake = std::env::var("PROBE_BAKE").expect("PROBE_BAKE");
    let mut bytes = std::fs::read(bake).unwrap();
    if let Ok(rev) = std::env::var("PROBE_STAMP") {
        bytes = zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.formula_revision", &rev).unwrap();
    }
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mode = std::env::var("PROBE_MODE").unwrap_or_else(|_| "score".into());
    let (w, h): (usize, usize) = (env("PROBE_W", 1024), env("PROBE_H", 1024));
    let n: usize = env("PROBE_N", 20);
    let par = env::<u32>("PROBE_PAR", 0) == 1;
    let bin: usize = env("PROBE_BIN", 8);
    let (src, dst, w, h) = match std::env::var("PROBE_IMG") {
        Ok(path) => {
            let img = image::open(&path).unwrap().to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let src: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
            // Deterministic distortion: quantize plus a hashed +/-6 dither.
            let dst = src
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let hsh = (i as u32).wrapping_mul(2654435761) >> 28;
                    let n = hsh as i32 - 8;
                    let f = |v: u8| ((((v / 10) * 10) as i32 + n).clamp(0, 255)) as u8;
                    [f(p[0]), f(p[1]), f(p[2])]
                })
                .collect();
            (src, dst, w, h)
        }
        Err(_) => {
            let (s, d) = test_pair(w, h);
            (s, d, w, h)
        }
    };
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(par);
    let ids = scorer.consumed_feature_ids().unwrap();
    eprintln!("consumed ids: {} (first {:?} last {:?})", ids.len(), &ids[..4], &ids[ids.len() - 4..]);
    let mut times = Vec::new();
    let mut last = 0.0;
    if mode == "named" {
        let z = zensim::Zensim::new(zensim::ZensimProfile::B).with_parallel(par);
        last = z.compute(&s, &d).unwrap().score();
        for _ in 0..n {
            let t = Instant::now();
            last = z.compute(&s, &d).unwrap().score();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        report("named B score", &mut times, last);
        return;
    }
    if mode == "score" || mode == "both" {
        last = scorer.compute(&s, &d, None).unwrap().score();
        for _ in 0..n {
            let t = Instant::now();
            last = scorer.compute(&s, &d, None).unwrap().score();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        report("score", &mut times, last);
        if let Ok(path) = std::env::var("PROBE_DUMP") {
            let r = scorer.compute(&s, &d, None).unwrap();
            let mut out = format!("{:016x}\n", r.score().to_bits());
            for f in r.features() {
                out.push_str(&format!("{:016x}\n", f.to_bits()));
            }
            std::fs::write(path, out).unwrap();
        }
    }
    if mode == "map" || mode == "both" || mode == "maponly" {
        let pre = scorer.precompute_reference(&s).unwrap();
        let mut session = zensim::Fused944Session::default();
        let r = scorer.compute_with_ref_and_attribution(&s, &pre, &d, None, &mut session, bin).unwrap();
        last = r.result().score();
        times.clear();
        for _ in 0..n {
            let t = Instant::now();
            let r = scorer.compute_with_ref_and_attribution(&s, &pre, &d, None, &mut session, bin).unwrap();
            last = r.result().score();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        report("cached-ref score+map", &mut times, last);
        times.clear();
        for _ in 0..n.min(10) {
            let t = Instant::now();
            let p = scorer.precompute_reference(&s).unwrap();
            std::hint::black_box(&p);
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        report("precompute_reference", &mut times, last);
    }
}

fn report(name: &str, t: &mut [f64], score: f64) {
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = t[t.len() / 2];
    eprintln!("{name}: n={} min={:.3} med={:.3} max={:.3} ms score={score}", t.len(), t[0], med, t[t.len() - 1]);
}
```
