//! Closure-builder layout — lexical nesting matches the box tree.
//!
//! Each container takes a `FnOnce(&mut Ctx)` closure. The closure
//! configures the container (gap / padding / etc.) and adds children
//! by calling helper methods on the context. Lexical indentation =
//! tree depth, so the source reads top-down.
//!
//! ```ignore
//! use zensim_regress::layout::compose::*;
//!
//! let m = column_block(|c| {
//!     c.gap(8).padding(8).background(hex("#0e0e10"));
//!     c.row(|r| {
//!         r.image(exp).label("EXPECTED");
//!         r.image(act).label("ACTUAL");
//!     });
//!     c.row(|r| {
//!         r.image(diff).label("DIFF");
//!         r.layers(|l| {
//!             l.fill(DARK_GRAY);
//!             l.text(line("HEATMAP", WHITE).center());
//!         });
//!     });
//! });
//! ```
//!
//! All `Ctx` types wrap the existing fluent builders — no IR change.
//! Refactoring a sub-tree into `fn add_panels(c: &mut RowCtx, ...)` is
//! the killer feature: layout fragments become reusable functions
//! with type-checked container affinity.

use super::color::Color;
use super::layers as layers_builder;
use super::node::Node;
use super::sizing::{CrossAlign, MainAlign};
use super::stack::Stack;

// ── ColumnCtx / RowCtx ─────────────────────────────────────────────────

/// Mutable handle inside a `column_block(|c| { … })` body. Children
/// added via the per-leaf and per-container helpers; modifiers
/// chain on the whole stack via the Ctx-level `.gap()` / `.padding()`
/// methods (these mutate the wrapped `Stack`).
pub struct ColumnCtx {
    stack: Stack,
}

/// Same as [`ColumnCtx`] but for a `row_block(|r| { … })` body.
pub struct RowCtx {
    stack: Stack,
}

/// Mutable handle inside a `layers_block(|l| { … })` body.
pub struct LayersCtx {
    layers: super::layers::Layers,
}

macro_rules! impl_stack_ctx {
    ($Ctx:ident, $build:expr) => {
        impl $Ctx {
            // ── Container settings (mutate the stack) ────────────────
            pub fn gap(&mut self, g: u32) -> &mut Self {
                self.stack = std::mem::replace(&mut self.stack, $build).gap(g);
                self
            }
            pub fn justify(&mut self, j: MainAlign) -> &mut Self {
                self.stack = std::mem::replace(&mut self.stack, $build).justify(j);
                self
            }
            pub fn align_items(&mut self, a: CrossAlign) -> &mut Self {
                self.stack = std::mem::replace(&mut self.stack, $build).align_items(a);
                self
            }
            pub fn shrink_on_overflow(&mut self, on: bool) -> &mut Self {
                self.stack = std::mem::replace(&mut self.stack, $build).shrink_on_overflow(on);
                self
            }

            // ── Add a pre-built child ───────────────────────────────
            pub fn push(&mut self, child: impl Into<Node>) -> &mut Self {
                self.stack = std::mem::replace(&mut self.stack, $build).child(child);
                self
            }

            // ── Child-constructor helpers ────────────────────────────
            /// Push an image leaf.
            pub fn image(&mut self, img: crate::pixel_ops::Bitmap) -> &mut Self {
                self.push(super::node::image(img))
            }

            /// Push a single-line text leaf with the given color.
            pub fn line(&mut self, text: impl Into<String>, fg: Color) -> &mut Self {
                self.push(super::node::line(text, fg))
            }

            /// Push a pre-built [`Node`]. Useful when the child has
            /// already been wrapped with modifiers.
            pub fn node(&mut self, node: impl Into<Node>) -> &mut Self {
                self.push(node)
            }

            /// Open a nested column — its closure receives a fresh
            /// [`ColumnCtx`].
            pub fn column(&mut self, f: impl FnOnce(&mut ColumnCtx)) -> &mut Self {
                let mut c = ColumnCtx::new();
                f(&mut c);
                self.push(c.into_node())
            }

            /// Open a nested row.
            pub fn row(&mut self, f: impl FnOnce(&mut RowCtx)) -> &mut Self {
                let mut r = RowCtx::new();
                f(&mut r);
                self.push(r.into_node())
            }

            /// Open a nested layers stack.
            pub fn layers(&mut self, f: impl FnOnce(&mut LayersCtx)) -> &mut Self {
                let mut l = LayersCtx::new();
                f(&mut l);
                self.push(l.into_node())
            }

            fn into_node(self) -> Node {
                self.stack.into()
            }
        }
    };
}

impl ColumnCtx {
    fn new() -> Self {
        Self {
            stack: super::stack::column(),
        }
    }
}
impl RowCtx {
    fn new() -> Self {
        Self {
            stack: super::stack::row(),
        }
    }
}
impl_stack_ctx!(ColumnCtx, super::stack::column());
impl_stack_ctx!(RowCtx, super::stack::row());

impl LayersCtx {
    fn new() -> Self {
        Self {
            layers: super::layers::layers(),
        }
    }
    pub fn push(&mut self, child: impl Into<Node>) -> &mut Self {
        self.layers = std::mem::replace(&mut self.layers, layers_builder()).child(child);
        self
    }
    pub fn fill(&mut self, color: Color) -> &mut Self {
        self.push(super::node::fill(color))
    }
    pub fn image(&mut self, img: crate::pixel_ops::Bitmap) -> &mut Self {
        self.push(super::node::image(img))
    }
    pub fn text(&mut self, node: impl Into<Node>) -> &mut Self {
        self.push(node)
    }
    pub fn column(&mut self, f: impl FnOnce(&mut ColumnCtx)) -> &mut Self {
        let mut c = ColumnCtx::new();
        f(&mut c);
        self.push(c.into_node())
    }
    pub fn row(&mut self, f: impl FnOnce(&mut RowCtx)) -> &mut Self {
        let mut r = RowCtx::new();
        f(&mut r);
        self.push(r.into_node())
    }
    fn into_node(self) -> Node {
        self.layers.into()
    }
}

// ── Top-level entry points ─────────────────────────────────────────────

/// Open a vertical stack and configure it inside a closure. Returns
/// the resulting [`Node`]; chain layout modifiers (`.padding(...)`,
/// `.background(...)`, etc.) on the result.
pub fn column_block(f: impl FnOnce(&mut ColumnCtx)) -> Node {
    let mut c = ColumnCtx::new();
    f(&mut c);
    c.into_node()
}

/// Open a horizontal stack and configure it inside a closure.
pub fn row_block(f: impl FnOnce(&mut RowCtx)) -> Node {
    let mut r = RowCtx::new();
    f(&mut r);
    r.into_node()
}

/// Open a layered overlap and configure it inside a closure.
pub fn layers_block(f: impl FnOnce(&mut LayersCtx)) -> Node {
    let mut l = LayersCtx::new();
    f(&mut l);
    l.into_node()
}

#[cfg(test)]
mod tests {
    use super::super::color::{BLACK, WHITE};
    use super::*;
    use crate::pixel_ops::Bitmap;

    fn solid(w: u32, h: u32, c: Color) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn column_block_produces_node() {
        let n = column_block(|c| {
            c.gap(4).align_items(CrossAlign::Center);
            c.image(solid(10, 10, WHITE));
            c.image(solid(20, 5, WHITE));
        });
        if let Node::Stack {
            gap,
            align_items,
            children,
            ..
        } = n
        {
            assert_eq!(gap, 4);
            assert_eq!(align_items, CrossAlign::Center);
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected Stack");
        }
    }

    #[test]
    fn nested_blocks_compose_into_tree() {
        let n = column_block(|c| {
            c.row(|r| {
                r.image(solid(10, 10, WHITE));
                r.image(solid(10, 10, BLACK));
            });
            c.row(|r| {
                r.image(solid(20, 10, WHITE));
            });
        });
        // Outer column has 2 children, each a row.
        if let Node::Stack { children, .. } = &n {
            assert_eq!(children.len(), 2);
            for c in children {
                assert!(matches!(c, Node::Stack { .. }));
            }
        } else {
            panic!("expected Stack");
        }
    }

    /// Demonstrates the killer feature: a sub-tree builder is just a
    /// function over a Ctx.
    fn add_two_panels(r: &mut RowCtx, a: Bitmap, b: Bitmap) {
        r.image(a).line("A", WHITE);
        r.image(b).line("B", WHITE);
    }

    #[test]
    fn factored_subtree_via_function() {
        let n = column_block(|c| {
            c.row(|r| add_two_panels(r, solid(10, 10, WHITE), solid(10, 10, BLACK)));
        });
        if let Node::Stack { children, .. } = &n {
            assert_eq!(children.len(), 1);
            if let Node::Stack {
                children: inner, ..
            } = &children[0]
            {
                assert_eq!(inner.len(), 4); // image, text, image, text
            } else {
                panic!("expected nested Stack");
            }
        } else {
            panic!("expected Stack");
        }
    }

    #[test]
    fn layers_block_stacks_children() {
        let n = layers_block(|l| {
            l.fill([10, 20, 30, 255]);
            l.image(solid(20, 20, WHITE));
        });
        if let Node::Layers(children) = n {
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected Layers");
        }
    }
}
