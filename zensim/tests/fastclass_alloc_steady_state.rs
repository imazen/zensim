// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! **Does the fast-class walk allocate once per compare, after warm-up?**
//!
//! The fast class (v1 basic `f0..155` plus the cheap extras that ride the same
//! walk — `V1PoolsMode::Peaks`, `V1FreeExtras::RawMoments` /
//! `::RawMomentsPlusBoundedErr`) is the product's throughput path: a codec
//! tuning loop calls it thousands of times against a reusable [`V2Scratch`].
//! Its steady-state allocation count is therefore a *contract*, not a
//! statistic — an allocation per compare is a malloc per compare in every
//! caller's hot loop, and it will not show up in a wall-clock A/B against the
//! noise floors this repo measures (±0.3 % at 1T, up to 6.5 % at 8T).
//!
//! So it is asserted directly instead of being inferred from timings, which is
//! the whole reason this file exists rather than another bench arm.
//!
//! # What is counted, and what is deliberately not
//!
//! The counter wraps the global allocator and counts `alloc` / `alloc_zeroed`
//! calls only. `realloc` and `dealloc` are not counted: a `Vec::resize` that
//! grows a scratch buffer to its steady-state size is *warm-up*, and the whole
//! question here is what remains after warm-up.
//!
//! The measurement window is opened AFTER several untimed warm-up walks
//! against the same `V2Scratch`, so every lazily-grown band/pool/rolling-plane
//! buffer has already reached its final size. What the window then sees is the
//! per-compare residue.
//!
//! # Threads
//!
//! Deliberately SERIAL (`with_parallel(false)`). rayon's own worker-thread
//! bookkeeping allocates on the first steal of a fresh job, which is a
//! property of the pool, not of the walk — counting it would make this test a
//! flaky assertion about rayon's internals. The serial path is where the
//! walk's own allocation behaviour is visible without that confound; the
//! parallel band scratch has its own regression already
//! (`benchmarks/fold_footprint_2026-08-31.md`: `map_init`-per-band
//! re-allocated ~580 KB per worker per strip per channel and cost 7.75 →
//! 10.00 ms at 3T, which is exactly the class of defect this test would have
//! caught at zero cost).
#![cfg(feature = "feature-regime-v2")]

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use zensim::feature_v2::{V1FreeExtras, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimProfile};

static COUNT: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct CountingAlloc;

// SAFETY-equivalent note: this forwards every call to the system allocator
// unchanged and only adds a relaxed counter. `#![forbid(unsafe_code)]` is a
// crate-level rule for `zensim` itself; a `GlobalAlloc` impl cannot be written
// in safe Rust, and this is test-only code in a separate integration-test
// crate that never ships.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { std::alloc::System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { std::alloc::System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        unsafe { std::alloc::System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

/// `zensim-bench/benches/ssim2_speed_bar.rs::test_pair`, verbatim — the same
/// content family every other fast-class instrument feeds its kernels.
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
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

/// Allocations attributable to `walks` steady-state compares on `toggles`,
/// after `WARM` warm-up compares against the same scratch.
fn steady_state_allocs(toggles: V2NewFeatureToggles, w: usize, h: usize, walks: usize) -> usize {
    const WARM: usize = 4;
    let (src, dst) = test_pair(w, h);
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let mut scratch = V2Scratch::new();

    let mut sink = 0.0f64;
    for _ in 0..WARM {
        let r = z
            .compute_folded720_append_features_streaming(&s, &d, toggles, &mut scratch)
            .expect("warm-up walk");
        sink += r.features()[0];
    }

    COUNT.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    for _ in 0..walks {
        let r = z
            .compute_folded720_append_features_streaming(&s, &d, toggles, &mut scratch)
            .expect("measured walk");
        sink += r.features()[0];
    }
    ARMED.store(false, Ordering::Relaxed);
    core::hint::black_box(sink);
    COUNT.load(Ordering::Relaxed)
}

fn basic_peaks() -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        v1_only: true,
        v1_pools: V1PoolsMode::Peaks,
        ..Default::default()
    }
}

/// The 944-LAYOUT fast-class shapes — `15c` / `15f` / `15x` in
/// `zensim/examples/foldapp_stream_bigpair.rs`.
fn layout944(free: V1FreeExtras) -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        v1_only: true,
        v1_pools: V1PoolsMode::Peaks,
        append_block: true,
        append2_block: true,
        free_extras: free,
        ..Default::default()
    }
}

/// **A RATCHET, at its measured baseline — not a statement that the ideal is
/// met.** A warmed fast-class walk against a reused [`V2Scratch`] should
/// allocate O(1) times per compare; it currently allocates **O(strips)**, and
/// this test exists to stop that number growing while it is driven down.
///
/// Measured 2026-09-05 at 1152², serial, `156_basic_peaks`
/// (`benchmarks/kernel_fastclass_2026-09-05.md` §3.2):
///
/// | | allocations / walk |
/// |---|---:|
/// | before the `starts` fix | 232.0 |
/// | after | **175.0** |
/// | the ideal this bar is NOT | ~1 (the returned feature `Vec`) |
///
/// The fixed source was `fold_v1_basic_bands`'s `let mut starts = Vec::new()`,
/// which ran once per (strip, channel, scale) and cost one allocation plus up
/// to two reallocs each — 57 of the 232 at this size, exactly the strip ×
/// channel count. The remaining ~175 are still per-strip and are **not
/// located**: they are not in `fold_v1_one_band`, not in `FoldPoolScratch::
/// ensure`/`ensure_h` (both only grow, so they are warm after warm-up), and
/// not in the producer's constructor (once per walk). Locating them is open
/// work, and this bar is what keeps the next lane honest about it.
///
/// **Lower this number; never raise it.** Raising it to make a change pass is
/// how the defect this test was written to catch gets re-admitted. The bar is
/// per-walk at a fixed 1152², so it is comparable across runs.
///
/// The bar cannot be a small constant today, and pretending otherwise by
/// deleting the test would be worse than an honest ratchet: the class of defect
/// it guards (per-strip / per-band / per-channel allocation) cost 7.75 → 10.00
/// ms at 3T once already (`map_init`-per-band, ~580 KB per worker per strip per
/// channel) and is invisible to a wall-clock A/B against this repo's noise
/// floors (±0.3 % at 1T, up to 6.5 % at 8T).
///
/// ONE test, deliberately. [`COUNT`] and [`ARMED`] are process-global, and
/// `cargo test` runs test functions on concurrent threads by default — two
/// armed tests in this binary would count each other's allocations and report
/// nonsense. Merging them is more robust than a mutex, because a mutex still
/// leaves the counter armed while another test's `Drop`s run.
#[test]
fn fast_class_steady_state_allocations_are_bounded_per_walk() {
    const WALKS: usize = 8;
    /// The measured 2026-09-05 baseline (175.0/walk on the worst of the four
    /// shapes) plus headroom for shape-to-shape variation. NOT the ideal.
    const BAR_PER_WALK: usize = 200;

    // Measure every shape BEFORE asserting, so a failure reports the whole
    // picture rather than aborting on whichever shape happens to be first.
    let mut measured: Vec<(&str, usize, f64)> = Vec::new();
    for (name, toggles) in [
        ("156_basic_peaks", basic_peaks()),
        ("15c_layout944", layout944(V1FreeExtras::Off)),
        ("15f_raw_moments", layout944(V1FreeExtras::RawMoments)),
        (
            "15x_class_c",
            layout944(V1FreeExtras::RawMomentsPlusBoundedErr),
        ),
    ] {
        let n = steady_state_allocs(toggles, 1152, 1152, WALKS);
        measured.push((name, n, n as f64 / WALKS as f64));
    }
    for (name, n, per) in &measured {
        eprintln!("{name}: {n} allocations over {WALKS} walks = {per:.1}/walk");
    }
    for (name, n, per) in &measured {
        assert!(
            *n <= BAR_PER_WALK * WALKS,
            "{name}: {per:.1} allocations/walk exceeds the {BAR_PER_WALK}/walk \
             ratchet. This bar is a measured baseline, not a target — if a \
             change raised it, fix the change; if a change LOWERED the whole \
             table, lower the bar in the same commit."
        );
    }

    // --- second half: steady state means *steady*. Doubling the measured
    // walks must not more than double the count — this is what distinguishes
    // "a constant per walk" from "state accumulating across compares against a
    // reused V2Scratch", which a single-count bar cannot see.
    let t = layout944(V1FreeExtras::RawMomentsPlusBoundedErr);
    let a = steady_state_allocs(t, 768, 768, 4);
    let b = steady_state_allocs(t, 768, 768, 8);
    assert!(
        b <= 2 * a + 4,
        "allocations grew superlinearly with walk count: 4 walks -> {a}, \
         8 walks -> {b}. A per-walk-constant allocator profile satisfies \
         b <= 2a (+slack); growth beyond it means state accumulating across \
         compares against a reused V2Scratch."
    );
    eprintln!("linearity: 4 walks -> {a} allocs, 8 walks -> {b} allocs");
}
