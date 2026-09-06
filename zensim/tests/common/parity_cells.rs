#![allow(dead_code)]
//! **The one owner of the parity GEOMETRY MATRIX.**
//!
//! Extracted from `fold_engine_parity.rs` (2026-09-05) so a second gate over
//! the same walks cannot drift onto its own hand-picked width list. Two files
//! that both claim to test "every geometry" while testing different ones is
//! exactly the shape the no-duplicate-implementations rule exists to stop —
//! and a geometry list is where that drift is invisible, because both suites
//! stay green while the union of what they cover shrinks.

/// The geometry matrix. `v1_372_bit_exact_to_fold_at_every_width`'s cells
/// (tight, non-tight even, non-tight odd, and the three `h = 93` cells that
/// were the last residual under the option-A pre-pad workaround) plus the two
/// procedural golden geometries and a sub-64 cell that exercises the shared
/// reflect-pad.
pub const CELLS: &[(usize, usize)] = &[
    // WIDER THAN `H_TILE_WIDTH` (1024) — the only cells here that exercise the
    // era-2 column tile at its SHIPPED width. Every other cell is narrower
    // than the tile and so runs the untiled path by construction
    // (`width > tile` guards every H entry), which means that without these
    // the whole parity suite would leave the shipped configuration untested.
    // 1153 is deliberately odd and crosses the boundary with a 129-column
    // remainder tile; 2049 crosses two boundaries with a 1-column remainder,
    // the narrowest tile the loop can emit.
    (1153, 72),
    (2049, 40),
    // the procedural golden fixtures
    (64, 64),
    (200, 150),
    // formerly "tight"
    (96, 64),
    (208, 144),
    (592, 80),
    (128, 93),
    // formerly divergent — even, non-tight
    (200, 151),
    (576, 96),
    (100, 96),
    // formerly divergent — odd, non-tight
    (127, 64),
    (129, 96),
    (255, 96),
    (577, 80),
    // the h = 93 cells
    (126, 93),
    (127, 93),
    (255, 93),
    // sub-64: the SHARED reflect-pad runs before either walk
    (48, 40),
    (17, 96),
];

/// The four extra cells the pool-sweeping tests add to [`CELLS`]: squares and
/// tall/wide aspect ratios whose band layout a rayon pool size can change.
///
/// Was written out identically at THREE call sites in `fold_engine_parity.rs`
/// before the extraction — the same list, retyped, with nothing checking that
/// the three stayed equal.
pub const POOL_SWEEP_EXTRA: &[(usize, usize)] = &[(256, 256), (96, 320), (320, 96), (577, 385)];

/// [`CELLS`] plus [`POOL_SWEEP_EXTRA`] — 24 cells.
pub fn pool_sweep_cells() -> Vec<(usize, usize)> {
    CELLS.iter().chain(POOL_SWEEP_EXTRA).copied().collect()
}
