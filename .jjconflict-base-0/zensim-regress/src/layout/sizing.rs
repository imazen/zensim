//! Sizing rules and stack-distribution enums — the vocabulary for "how
//! big" and "how to fill remaining space."

/// Main-axis distribution within a stack. CSS `justify-content`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MainAlign {
    #[default]
    Start,
    Center,
    End,
    /// Equal gaps between children, none at edges.
    SpaceBetween,
    /// Half-gaps at edges, full gaps between.
    SpaceAround,
    /// Equal gaps everywhere (between, before, after).
    SpaceEvenly,
}

/// Cross-axis alignment of children within a stack. CSS `align-items`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CrossAlign {
    #[default]
    Start,
    Center,
    End,
    /// Stretch each child to fill the cross axis.
    Stretch,
}

/// How an image leaf scales into its allotted rect — CSS `object-fit`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum Fit {
    /// Place at original size, top-left aligned, clipped if larger
    /// (CSS `none` with `object-position: 0 0`).
    #[default]
    None,
    /// Scale preserving aspect, fit entirely (letterbox).
    Contain,
    /// Scale preserving aspect, fill rect (crop overflow).
    Cover,
    /// Stretch to rect ignoring aspect.
    Stretch,
}

/// Per-axis sizing rule for a node — CSS `width: auto/100%/<n>px`.
///
/// [`SizeRule::Grow`] is CSS `flex-grow: n` — among Grow children of a
/// stack, remaining main-axis space is split proportionally.
///
/// [`SizeRule::Percent`] takes a unit fraction in `[0.0, 1.0]` —
/// e.g. `Percent(0.5)` is CSS `width: 50%`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum SizeRule {
    /// Hug content (CSS `auto`).
    #[default]
    Hug,
    /// Fixed pixel size.
    Fixed(u32),
    /// Fill the parent constraint (CSS `100%`).
    Fill,
    /// Weighted fill — CSS `flex-grow: n`.
    Grow(u32),
    /// Fraction of parent constraint, `[0.0, 1.0]`.
    Percent(f32),
}

impl SizeRule {
    pub(super) fn grow_weight(self) -> u32 {
        match self {
            SizeRule::Fill => 1,
            SizeRule::Grow(w) => w,
            _ => 0,
        }
    }

    /// Multiply the fixed-pixel quantity by `scale`. Hug / Fill / Grow
    /// / Percent are unchanged (they're already relative).
    pub(super) fn scaled(self, scale: f32) -> Self {
        match self {
            SizeRule::Fixed(v) => SizeRule::Fixed(((v as f32) * scale).round() as u32),
            other => other,
        }
    }
}

/// Grid track sizing — CSS `grid-template-columns` / `-rows`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Track {
    /// Fixed pixel size.
    Px(u32),
    /// Fraction of remaining space (CSS `<n>fr`).
    Fr(u32),
    /// Hug the maximum content size in this track (CSS `auto`).
    Auto,
    /// Fraction of total available track-axis space, `[0.0, 1.0]`.
    Percent(f32),
    /// Fraction of remaining space, never shrinking below `min_px` —
    /// analogous to CSS `minmax(<min_px>px, <weight>fr)`. Use when you
    /// need a hard minimum gap regardless of overflow content (e.g.,
    /// the Fr column between left/right groups in a segmented label
    /// strip — guarantees the groups stay at least `min_px` apart).
    FrMin { weight: u32, min_px: u32 },
}

impl Track {
    /// Multiply the fixed-pixel quantity by `scale`. Fr / Auto /
    /// Percent are unchanged (they're already relative); for `FrMin`
    /// only `min_px` scales, the weight stays.
    pub(super) fn scaled(self, scale: f32) -> Self {
        match self {
            Track::Px(v) => Track::Px(((v as f32) * scale).round() as u32),
            Track::FrMin { weight, min_px } => Track::FrMin {
                weight,
                min_px: ((min_px as f32) * scale).round() as u32,
            },
            other => other,
        }
    }
}
