//! Geometric primitives: sizes, rectangles, insets, and alignment enums.

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub w: u32,
    pub h: u32,
}

impl Size {
    pub const fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }
    pub const ZERO: Self = Self { w: 0, h: 0 };
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
    pub const fn size(&self) -> Size {
        Size::new(self.w, self.h)
    }
}

/// Padding-style insets — one value per edge.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Insets {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Insets {
    pub const fn all(v: u32) -> Self {
        Self {
            left: v,
            top: v,
            right: v,
            bottom: v,
        }
    }
    /// Symmetric `(x, y)` — same horizontal, same vertical.
    pub const fn xy(x: u32, y: u32) -> Self {
        Self {
            left: x,
            top: y,
            right: x,
            bottom: y,
        }
    }
    /// CSS shorthand order: top, right, bottom, left.
    pub const fn each(top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }
    pub const fn horizontal(&self) -> u32 {
        self.left + self.right
    }
    pub const fn vertical(&self) -> u32 {
        self.top + self.bottom
    }

    /// Multiply each edge by `scale` (rounded).
    pub(super) fn scaled(self, scale: f32) -> Self {
        let s = |v: u32| ((v as f32) * scale).round() as u32;
        Self {
            left: s(self.left),
            top: s(self.top),
            right: s(self.right),
            bottom: s(self.bottom),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Axis {
    Horizontal,
    Vertical,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum HAlign {
    Left,
    #[default]
    Center,
    Right,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum VAlign {
    Top,
    #[default]
    Center,
    Bottom,
}
