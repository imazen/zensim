//! Resource caps that protect the layout engine from malicious or
//! pathological trees — important if a [`super::Node`] ever lands via
//! deserialization (JSON, MessagePack, …).
//!
//! Limits live in a single [`LayoutLimits`] struct and are propagated
//! per-render via a thread-local. Set via the public
//! [`super::render_with_limits`] entry point; everything inside reads
//! the active limits via [`limits`] / [`clamp_dim`] / etc.
//!
//! Trees within the active limits behave normally; trees outside are
//! silently clamped — the renderer never panics on input data alone.

use std::cell::Cell;

use super::geom::Size;

/// Knobs for hostile-input defense — pass to
/// [`super::render_with_limits`] to override the defaults for a single
/// render.
///
/// All fields are simple counts with hard upper bounds. Increasing
/// them costs memory; lowering them clamps more aggressively.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LayoutLimits {
    /// Maximum width or height in pixels. Bounds *per-axis* allocation
    /// (e.g., a 100K-wide text strip). Default 8192.
    pub max_dim: u32,

    /// Maximum total canvas pixel count. The output is scaled down
    /// (preserving aspect) when the measured tree exceeds this. At
    /// RGBA8 the default 20 MP works out to 80 MB.
    pub max_pixels: u32,

    /// A constraint above this is treated as "unbounded": [`super::Track::Fr`]
    /// tracks fall back to [`super::Track::Auto`] semantics so they
    /// don't try to fill an effectively infinite axis (e.g., the
    /// `u32::MAX/2` we hand the root vertical constraint).
    pub unbounded_threshold: u32,

    /// Maximum recursion depth across measure / paint passes. Tracked
    /// via a thread-local; deeper trees return zero size or paint
    /// nothing. Defends against pathologically nested modifiers.
    pub max_depth: u16,

    /// Maximum siblings honored in a [`super::Stack`] / [`super::Layers`]
    /// container. Children past this index are silently ignored.
    pub max_children: usize,

    /// Maximum grid cells honored.
    pub max_cells: usize,

    /// Maximum number of grid tracks (cols or rows).
    pub max_tracks: usize,

    /// Maximum character pixel-height for text rasterization. Bounds
    /// the *scaled font-strip* allocation (the engine resizes the
    /// embedded glyph strip to `char_w × CHAR_COUNT × char_h` before
    /// stamping), independent of the output canvas.
    pub max_char_h: u32,
}

impl LayoutLimits {
    /// Sensible defaults for diff-montage-style work — 20 MP canvas
    /// budget, 8K per axis, 1024-px max single-character height. Set as
    /// `LayoutLimits::default()`.
    pub const DEFAULT: Self = Self {
        max_dim: 8192,
        max_pixels: 20_000_000,
        unbounded_threshold: 1 << 20,
        max_depth: 64,
        max_children: 4096,
        max_cells: 4096,
        max_tracks: 256,
        max_char_h: 1024,
    };

    /// Loose limits — for trusted local rendering or debugging.
    /// 8K-tall characters, 200 MP canvas, deeper recursion.
    pub const PERMISSIVE: Self = Self {
        max_dim: 32_768,
        max_pixels: 200_000_000,
        unbounded_threshold: 1 << 22,
        max_depth: 256,
        max_children: 65_536,
        max_cells: 65_536,
        max_tracks: 4096,
        max_char_h: 4096,
    };

    /// Tight limits — for definitely-untrusted input on a small
    /// machine. 4 MP canvas, 32-deep recursion, 256 children/cells.
    pub const STRICT: Self = Self {
        max_dim: 2048,
        max_pixels: 4_000_000,
        unbounded_threshold: 1 << 18,
        max_depth: 32,
        max_children: 256,
        max_cells: 256,
        max_tracks: 64,
        max_char_h: 256,
    };
}

impl Default for LayoutLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Thread-local propagation ──────────────────────────────────────────

/// Default value of `em(1.0)` when no [`super::RenderConfig::base_em`]
/// is in scope. Matches CSS's typical 16-px baseline.
pub const DEFAULT_BASE_EM: u32 = 16;

thread_local! {
    static LIMITS: Cell<LayoutLimits> = const { Cell::new(LayoutLimits::DEFAULT) };
    static DEPTH: Cell<u16> = const { Cell::new(0) };
    static BASE_EM: Cell<u32> = const { Cell::new(DEFAULT_BASE_EM) };
}

/// Snapshot the limits active on this thread.
pub fn limits() -> LayoutLimits {
    LIMITS.with(|l| l.get())
}

/// Run `f` with `lim` as the active limits, restoring the previous
/// value (and resetting the depth counter) on return.
pub fn with_limits<R>(lim: LayoutLimits, f: impl FnOnce() -> R) -> R {
    let prev = LIMITS.with(|l| l.replace(lim));
    DEPTH.with(|d| d.set(0));
    let r = f();
    LIMITS.with(|l| l.set(prev));
    r
}

// ── Clamp / cap helpers (read the active limits) ──────────────────────

/// Clamp a single dimension to `limits().max_dim`.
pub fn clamp_dim(v: u32) -> u32 {
    v.min(limits().max_dim)
}

/// Clamp both axes of a [`Size`] to `limits().max_dim`.
pub fn clamp_size(s: Size) -> Size {
    Size::new(clamp_dim(s.w), clamp_dim(s.h))
}

/// Scale `s` down preserving aspect so `w × h <= limits().max_pixels`.
/// Returned unchanged if already in budget.
pub fn clamp_to_pixel_budget(s: Size) -> Size {
    let cap = limits().max_pixels as u64;
    let total = s.w as u64 * s.h as u64;
    if total == 0 || total <= cap {
        return s;
    }
    let ratio = (cap as f64 / total as f64).sqrt();
    Size::new(
        ((s.w as f64 * ratio).floor() as u32).max(1),
        ((s.h as f64 * ratio).floor() as u32).max(1),
    )
}

/// `true` when the value is large enough to be treated as "unbounded".
pub fn is_unbounded(v: u32) -> bool {
    v >= limits().unbounded_threshold
}

pub fn cap_children(n: usize) -> usize {
    n.min(limits().max_children)
}
pub fn cap_cells(n: usize) -> usize {
    n.min(limits().max_cells)
}
pub fn cap_tracks(n: usize) -> usize {
    n.min(limits().max_tracks)
}
/// Clamp a char-height to `limits().max_char_h`. Apply at any entry
/// point that converts char-height into a pixel buffer.
pub fn clamp_char_h(h: u32) -> u32 {
    h.min(limits().max_char_h)
}

/// Run `f` with the depth counter incremented. Returns `Some(f())`
/// when depth ≤ `limits().max_depth`, `None` when the cap is hit.
pub fn with_depth<R>(f: impl FnOnce() -> R) -> Option<R> {
    let max = limits().max_depth;
    DEPTH.with(|d| {
        let v = d.get();
        if v >= max {
            return None;
        }
        d.set(v + 1);
        let r = f();
        d.set(v);
        Some(r)
    })
}

/// Reset the depth counter — called by the public render entry points
/// so a panicking previous render doesn't poison subsequent calls on
/// the same thread.
pub fn reset_depth() {
    DEPTH.with(|d| d.set(0));
}

// ── em() — text-relative units ────────────────────────────────────────

/// Active `em(1.0)` in pixels for this thread. Set via
/// [`super::RenderConfig::with_base_em`]; defaults to
/// [`DEFAULT_BASE_EM`].
pub fn base_em() -> u32 {
    BASE_EM.with(|e| e.get())
}

/// Run `f` with `base` as the active em-baseline, restoring the
/// previous value on return.
pub fn with_base_em<R>(base: u32, f: impl FnOnce() -> R) -> R {
    let prev = BASE_EM.with(|e| e.replace(base));
    let r = f();
    BASE_EM.with(|e| e.set(prev));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_are_reasonable() {
        let d = LayoutLimits::default();
        assert_eq!(d.max_pixels, 20_000_000);
        assert_eq!(d.max_dim, 8192);
        assert_eq!(d.max_char_h, 1024);
    }

    #[test]
    fn clamp_dim_respects_active_limits() {
        with_limits(LayoutLimits::STRICT, || {
            assert_eq!(clamp_dim(u32::MAX), 2048);
        });
        with_limits(LayoutLimits::DEFAULT, || {
            assert_eq!(clamp_dim(u32::MAX), 8192);
        });
    }

    #[test]
    fn clamp_to_pixel_budget_scales_aspect() {
        with_limits(LayoutLimits::DEFAULT, || {
            // 10000 × 10000 = 100 MP, budget 20 MP → ~4472 × 4472.
            let out = clamp_to_pixel_budget(Size::new(10000, 10000));
            assert!(out.w as u64 * out.h as u64 <= 20_000_000);
            assert!(out.w >= 4400 && out.w <= 4500);
        });
    }

    #[test]
    fn clamp_to_pixel_budget_preserves_small() {
        with_limits(LayoutLimits::DEFAULT, || {
            let s = Size::new(100, 50);
            assert_eq!(clamp_to_pixel_budget(s), s);
        });
    }

    #[test]
    fn unbounded_detects_root_sentinel() {
        with_limits(LayoutLimits::DEFAULT, || {
            assert!(is_unbounded(u32::MAX / 2));
            assert!(!is_unbounded(1024));
        });
    }

    #[test]
    fn depth_caps_recursion() {
        with_limits(LayoutLimits::DEFAULT, || {
            fn recurse(n: u32) -> Option<u32> {
                super::with_depth(|| {
                    if n == 0 {
                        Some(0)
                    } else {
                        recurse(n - 1).map(|v| v + 1)
                    }
                })
                .flatten()
            }
            assert!(recurse(100).is_none()); // > MAX_DEPTH (64)
            assert_eq!(recurse(32), Some(32));
        });
    }

    #[test]
    fn permissive_allows_deeper() {
        with_limits(LayoutLimits::PERMISSIVE, || {
            assert!(limits().max_depth >= 200);
            assert!(limits().max_pixels >= 100_000_000);
        });
    }
}
