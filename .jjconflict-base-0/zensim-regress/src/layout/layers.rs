//! Z-stack container — children render at the same rect in painter's
//! order. Roughly CSS `position: absolute` siblings.

use crate::pixel_ops::Bitmap;

use super::geom::{Rect, Size};
use super::node::Node;
use super::safety;

#[derive(Clone, Debug, Default)]
pub struct Layers {
    children: Vec<Node>,
}

impl Layers {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn child(mut self, c: impl Into<Node>) -> Self {
        self.children.push(c.into());
        self
    }
    pub fn children<I, N>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<Node>,
    {
        self.children.extend(items.into_iter().map(Into::into));
        self
    }
}

/// Free-fn entry point for an empty [`Layers`] builder.
pub fn layers() -> Layers {
    Layers::new()
}

impl From<Layers> for Node {
    fn from(l: Layers) -> Node {
        Node::Layers(l.children)
    }
}

pub(super) fn measure(children: &[Node], max: Size) -> Size {
    let n = safety::cap_children(children.len());
    let mut out = Size::ZERO;
    for c in children.iter().take(n) {
        let s = c.measure(max);
        out.w = out.w.max(s.w);
        out.h = out.h.max(s.h);
    }
    out
}

pub(super) fn paint(children: &[Node], rect: Rect, canvas: &mut Bitmap) {
    let n = safety::cap_children(children.len());
    for c in children.iter().take(n) {
        c.paint(rect, canvas);
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::WHITE;
    use super::super::modifiers::LayoutMod;
    use super::super::node::{empty, image as image_node};
    use super::*;
    use crate::pixel_ops::Bitmap;

    fn solid(w: u32, h: u32, c: super::super::color::Color) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn layers_max_size() {
        let a = image_node(solid(50, 30, WHITE));
        let b = image_node(solid(20, 80, WHITE));
        let l = layers().child(a).child(b);
        assert_eq!(
            Node::from(l).measure(Size::new(200, 200)),
            Size::new(50, 80)
        );
    }

    #[test]
    fn layers_painters_order() {
        let bg = empty().background([255, 0, 0, 255]).fill();
        let dot = image_node(solid(4, 4, [0, 255, 0, 255])).center();
        let img = layers().child(bg).child(dot).size(20, 20).render(20);
        assert_eq!(img.get_pixel(0, 0), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(10, 10), [0, 255, 0, 255]);
    }
}
