// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! **Diagnostic-only** per-phase timing for the streaming fold walk.
//!
//! The fold-MT lane's first instruction is "profile the parallel walk, don't
//! guess". Instruction-count profiles (callgrind) answer a *serial* question —
//! they cannot see a thread idling at a barrier. This module answers the
//! parallel one directly: for each phase of the strip loop it records both the
//! **wall** span (which is the critical path) and the summed **busy** time of
//! the tasks inside it, so `busy / (wall × threads)` is a measured occupancy
//! rather than an inferred one.
//!
//! Nothing here can change a feature byte: every hook is a timestamp and an
//! atomic add, and the recorded values are never read by any kernel.
//!
//! # Enabling
//!
//! `ZENSIM_FOLD_TIMING=<N>` — dump a report to stderr and reset every `N`
//! completed walks (`ZENSIM_FOLD_TIMING=1` dumps per walk). Unset or `0`
//! disables, and then every hook is one relaxed atomic load of an already-
//! resolved `OnceLock` plus a predictable branch.
//!
//! # Reading the report
//!
//! | column | meaning |
//! |---|---|
//! | `wall` | summed wall time of that phase across every strip — the phase's share of the **critical path** |
//! | `busy` | summed per-task time inside that phase |
//! | `occ` | `busy / (wall × RAYON_NUM_THREADS)` — 1.00 means every thread was busy for the whole phase |
//!
//! A phase with `occ` near `1/threads` is running at degree 1 (serial); a
//! phase whose `wall` is large and whose `occ` is small is where the threads
//! are idling, which is the number this lane exists to move.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// One accumulator slot. Indices are `(phase, scale)`; see [`Phase`].
const N_PHASE: usize = 12;
const N_SCALE: usize = 8;

macro_rules! zeroed_atomics {
    ($n:expr) => {{
        const Z: AtomicU64 = AtomicU64::new(0);
        [Z; $n]
    }};
}

static NANOS: [AtomicU64; N_PHASE * N_SCALE] = zeroed_atomics!(N_PHASE * N_SCALE);
static COUNTS: [AtomicU64; N_PHASE * N_SCALE] = zeroed_atomics!(N_PHASE * N_SCALE);
static WALKS: AtomicU64 = AtomicU64::new(0);

/// Phases of the strip loop, in the order they execute.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Phase {
    /// `producer.next_strip()` — decode + XYB + the downscale cascade. Serial
    /// by construction today; this is the row the "parallelise the producer"
    /// candidate is judged on.
    Producer = 0,
    /// Wall of the phase-A fan-out (3-way over channels).
    PhaseAWall = 1,
    /// Summed per-channel busy time inside phase A.
    PhaseABusy = 2,
    /// Wall of the phase-B fan-out (3-way over channels, bands nested inside).
    PhaseBWall = 3,
    /// Summed per-channel busy time inside phase B.
    PhaseBBusy = 4,
    /// Summed per-BAND busy time inside `fold_v1_basic_bands`.
    BandBusy = 5,
    /// Serial work between the two fan-outs (mean-offset side channel,
    /// `refy` binding, retention copies).
    Between = 6,
    /// Whole-walk wall, recorded at scale slot 0.
    Walk = 7,
    /// Wall of the strip-local `fold_v1_basic_bands` call (one per channel;
    /// summed across channels, so it exceeds `PhaseBWall` when parallel).
    FoldWall = 8,
    /// Wall of `fused_blur_h_ssim` (one per channel, summed).
    BlurHWall = 9,
    /// Scale-0 XYB conversion inside the producer.
    ProdConvert = 10,
    /// Downscale cascade inside the producer.
    ProdDownscale = 11,
}

#[inline(always)]
fn interval() -> u64 {
    static IV: OnceLock<u64> = OnceLock::new();
    *IV.get_or_init(|| {
        std::env::var("ZENSIM_FOLD_TIMING")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}

/// True when timing is enabled. One resolved-`OnceLock` load when off.
#[inline(always)]
pub(crate) fn on() -> bool {
    interval() != 0
}

/// Start a span. Returns `None` (and does no work) when timing is off.
#[inline(always)]
pub(crate) fn start() -> Option<Instant> {
    if on() { Some(Instant::now()) } else { None }
}

/// Close a span opened by [`start`] into `(phase, scale)`.
#[inline(always)]
pub(crate) fn stop(t: Option<Instant>, phase: Phase, scale: usize) {
    if let Some(t0) = t {
        record(phase, scale, t0.elapsed().as_nanos() as u64);
    }
}

#[inline(always)]
fn record(phase: Phase, scale: usize, nanos: u64) {
    let i = (phase as usize) * N_SCALE + scale.min(N_SCALE - 1);
    NANOS[i].fetch_add(nanos, Ordering::Relaxed);
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
}

fn sum(phase: Phase) -> (u64, u64) {
    let base = (phase as usize) * N_SCALE;
    let mut n = 0;
    let mut c = 0;
    for k in 0..N_SCALE {
        n += NANOS[base + k].load(Ordering::Relaxed);
        c += COUNTS[base + k].load(Ordering::Relaxed);
    }
    (n, c)
}

fn per_scale(phase: Phase) -> [u64; 4] {
    let base = (phase as usize) * N_SCALE;
    core::array::from_fn(|k| NANOS[base + k].load(Ordering::Relaxed))
}

/// Called once per completed walk; dumps + resets on the configured interval.
pub(crate) fn walk_done(walk_nanos: u64) {
    if !on() {
        return;
    }
    record(Phase::Walk, 0, walk_nanos);
    let n = WALKS.fetch_add(1, Ordering::Relaxed) + 1;
    if n % interval() == 0 {
        dump(n);
        reset();
    }
}

fn threads() -> f64 {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1.0, |v| v.get() as f64))
}

fn dump(walks: u64) {
    let thr = threads();
    let (walk, _) = sum(Phase::Walk);
    let w = walk.max(1) as f64;
    let ms = |v: u64| v as f64 / 1e6 / walks as f64;
    let pct = |v: u64| 100.0 * v as f64 / w;

    eprintln!(
        "\n=== ZENSIM FOLD TIMING === walks={walks} threads={thr:.0} \
         walk={:.3} ms/walk",
        ms(walk)
    );
    eprintln!(
        "{:<16} {:>10} {:>8} {:>10} {:>7} {:>9}  {}",
        "phase", "ms/walk", "% wall", "busy ms", "occ", "calls", "per-scale ms (0..3)"
    );
    let row = |name: &str, wall: Phase, busy: Option<Phase>| {
        let (wn, wc) = sum(wall);
        let (bn, occ) = match busy {
            Some(b) => {
                let (bn, _) = sum(b);
                (bn, bn as f64 / (wn.max(1) as f64 * thr))
            }
            None => (0, f64::NAN),
        };
        let ps = per_scale(wall);
        eprintln!(
            "{:<16} {:>10.3} {:>8.1} {:>10.3} {:>7} {:>9}  {:.2} {:.2} {:.2} {:.2}",
            name,
            ms(wn),
            pct(wn),
            ms(bn),
            if occ.is_nan() {
                "-".to_string()
            } else {
                format!("{occ:.3}")
            },
            wc,
            ms(ps[0]),
            ms(ps[1]),
            ms(ps[2]),
            ms(ps[3]),
        );
    };
    row("producer", Phase::Producer, None);
    row("  convert", Phase::ProdConvert, None);
    row("  downscale", Phase::ProdDownscale, None);
    row("phaseA", Phase::PhaseAWall, Some(Phase::PhaseABusy));
    row("  blur_h(sum)", Phase::BlurHWall, None);
    row("between", Phase::Between, None);
    row("phaseB", Phase::PhaseBWall, Some(Phase::PhaseBBusy));
    row("  fold(sum)", Phase::FoldWall, Some(Phase::BandBusy));
    let (p, _) = sum(Phase::Producer);
    let (a, _) = sum(Phase::PhaseAWall);
    let (bt, _) = sum(Phase::Between);
    let (b, _) = sum(Phase::PhaseBWall);
    let acct = p + a + bt + b;
    eprintln!(
        "accounted {:.3} ms/walk = {:.1} % of walk; unaccounted {:.3} ms/walk",
        ms(acct),
        pct(acct),
        ms(walk.saturating_sub(acct))
    );
}

fn reset() {
    for s in NANOS.iter().chain(COUNTS.iter()) {
        s.store(0, Ordering::Relaxed);
    }
}
