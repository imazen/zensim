//! Array-style children macros for declarative layout construction.
//!
//! **Internal API.** The macros are gated behind the `_internal_api`
//! cargo feature: enabling the feature also exports `column!`, `row!`,
//! and `layers!` at the crate root so consumers can write
//! `zensim_regress::column![...]`. With the feature off, the macros
//! are not defined at all — keeping the public API surface small for
//! downstream consumers that pin a stable subset.
//!
//! Each macro is a thin wrapper over the existing fluent builder —
//! `column![a, b, c]` desugars to `column().child(a).child(b).child(c)`.
//! Modifiers chain as before *after* the macro:
//!
//! ```ignore
//! use zensim_regress::layout::*;
//!
//! let m = column![
//!     row![ image(exp), image(act) ],
//!     row![ image(diff), image(heat) ],
//! ]
//! .gap(8)
//! .padding(8)
//! .background(hex("#0e0e10"));
//! ```
//!
//! No new types or trait methods — the macros expand to existing
//! `column()` / `row()` / `layers()` builder calls. Total API
//! footprint: 3 macros, zero added types.

/// Vertical stack literal — `column![child, child, ...]` desugars to
/// `column().child(child).child(child)…`. Modifiers (`.gap()`,
/// `.padding()`, etc.) chain after the macro. Available only with
/// the `_internal_api` cargo feature.
#[cfg(feature = "_internal_api")]
#[macro_export]
macro_rules! column {
    [ $($child:expr),* $(,)? ] => {
        $crate::layout::column()$(.child($child))*
    };
}

/// Horizontal stack literal — see [`column!`]. Available only with
/// the `_internal_api` cargo feature.
#[cfg(feature = "_internal_api")]
#[macro_export]
macro_rules! row {
    [ $($child:expr),* $(,)? ] => {
        $crate::layout::row()$(.child($child))*
    };
}

/// Layered overlap literal — `layers![bg, content]` paints children
/// back-to-front into the same rect. Available only with the
/// `_internal_api` cargo feature.
#[cfg(feature = "_internal_api")]
#[macro_export]
macro_rules! layers {
    [ $($child:expr),* $(,)? ] => {
        $crate::layout::layers()$(.child($child))*
    };
}
