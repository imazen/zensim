//! 2D grid container — CSS `display: grid` with `grid-template-columns`
//! / `-rows`. Tracks can be `Px`, `Fr` (weighted fill of remainder), or
//! `Auto` (hug content). Supports cell spanning and CSS-style ASCII
//! `grid-template-areas`.

use std::collections::HashMap;

use crate::pixel_ops::Bitmap;

use super::geom::{Rect, Size};
use super::node::Node;
use super::safety;
use super::sizing::Track;

/// Position of a child within a [`Grid`] — column/row origin and span.
///
/// `colspan == 0` and `rowspan == 0` are normalized to `1` during
/// layout.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GridSpan {
    pub col: u32,
    pub row: u32,
    pub colspan: u32,
    pub rowspan: u32,
}

impl GridSpan {
    pub const fn cell(col: u32, row: u32) -> Self {
        Self {
            col,
            row,
            colspan: 1,
            rowspan: 1,
        }
    }
    pub const fn span(col: u32, row: u32, colspan: u32, rowspan: u32) -> Self {
        Self {
            col,
            row,
            colspan,
            rowspan,
        }
    }
    fn cs(&self) -> u32 {
        self.colspan.max(1)
    }
    fn rs(&self) -> u32 {
        self.rowspan.max(1)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Grid {
    cols: Vec<Track>,
    rows: Vec<Track>,
    gap: (u32, u32),
    pad: u32,
    cells: Vec<(GridSpan, Node)>,
    areas: HashMap<String, GridSpan>,
}

impl Grid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn columns(mut self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.cols = tracks.into_iter().collect();
        self
    }
    pub fn rows(mut self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.rows = tracks.into_iter().collect();
        self
    }
    /// Alias for [`Grid::columns`] — reads better paired with [`Grid::row_heights`].
    pub fn col_widths(self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.columns(tracks)
    }
    pub fn row_heights(self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.rows(tracks)
    }
    /// Equal-weight Fr columns: `vec![Track::Fr(1); n]`.
    pub fn cols(mut self, n: u32) -> Self {
        self.cols = (0..n).map(|_| Track::Fr(1)).collect();
        self
    }
    pub fn equal_rows(mut self, n: u32) -> Self {
        self.rows = (0..n).map(|_| Track::Fr(1)).collect();
        self
    }

    pub fn gap(mut self, g: u32) -> Self {
        self.gap = (g, g);
        self
    }
    pub fn gap_xy(mut self, x: u32, y: u32) -> Self {
        self.gap = (x, y);
        self
    }
    pub fn padding(mut self, p: u32) -> Self {
        self.pad = p;
        self
    }

    pub fn cell(mut self, col: u32, row: u32, n: impl Into<Node>) -> Self {
        self.cells.push((GridSpan::cell(col, row), n.into()));
        self
    }
    pub fn span(
        mut self,
        col: u32,
        row: u32,
        colspan: u32,
        rowspan: u32,
        n: impl Into<Node>,
    ) -> Self {
        self.cells
            .push((GridSpan::span(col, row, colspan, rowspan), n.into()));
        self
    }

    /// Define named areas via an ASCII-art template — like CSS
    /// `grid-template-areas`. Each row is whitespace-separated; `.` is
    /// empty. Repeated tokens form a single area whose bounding box is
    /// the contiguous extent of those tokens. All rows must have the
    /// same column count (panics otherwise).
    ///
    /// If `cols`/`rows` haven't been set explicitly, equal-weight Fr
    /// tracks are inferred from the template's dimensions.
    pub fn areas(mut self, rows: &[&str]) -> Self {
        if rows.is_empty() {
            return self;
        }
        let row_tokens: Vec<Vec<&str>> = rows
            .iter()
            .map(|r| r.split_whitespace().collect())
            .collect();
        let cols_count = row_tokens[0].len() as u32;
        for (i, r) in row_tokens.iter().enumerate() {
            assert_eq!(
                r.len() as u32,
                cols_count,
                "Grid::areas row {i} has {} cols, expected {cols_count}",
                r.len()
            );
        }
        let rows_count = rows.len() as u32;

        let mut bbox: HashMap<String, (u32, u32, u32, u32)> = HashMap::new();
        for (r, line) in row_tokens.iter().enumerate() {
            for (c, tok) in line.iter().enumerate() {
                if *tok == "." {
                    continue;
                }
                let key = (*tok).to_string();
                let entry = bbox
                    .entry(key)
                    .or_insert((c as u32, r as u32, c as u32, r as u32));
                entry.0 = entry.0.min(c as u32);
                entry.1 = entry.1.min(r as u32);
                entry.2 = entry.2.max(c as u32);
                entry.3 = entry.3.max(r as u32);
            }
        }
        self.areas = bbox
            .into_iter()
            .map(|(name, (c0, r0, c1, r1))| {
                (name, GridSpan::span(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
            })
            .collect();

        if self.cols.is_empty() {
            self.cols = (0..cols_count).map(|_| Track::Fr(1)).collect();
        }
        if self.rows.is_empty() {
            self.rows = (0..rows_count).map(|_| Track::Fr(1)).collect();
        }
        self
    }

    /// Place a child in a named area defined by [`Grid::areas`].
    /// Panics if `name` is unknown.
    pub fn place(mut self, name: &str, n: impl Into<Node>) -> Self {
        let span = *self
            .areas
            .get(name)
            .unwrap_or_else(|| panic!("Grid::place: no area named {name:?}"));
        self.cells.push((span, n.into()));
        self
    }
}

impl From<Grid> for Node {
    fn from(g: Grid) -> Node {
        Node::Grid {
            cols: g.cols,
            rows: g.rows,
            gap: g.gap,
            pad: g.pad,
            cells: g.cells,
        }
    }
}

/// Free-fn entry point for an empty [`Grid`] builder.
pub fn grid() -> Grid {
    Grid::new()
}

// ── Track sizing ───────────────────────────────────────────────────────

/// Solve track sizes given total available space along that axis.
///
/// Order: Px tracks consume their fixed size, Auto tracks call back to
/// `auto_size` for their content size, then the remainder is split among
/// Fr tracks proportionally to their weight (last Fr gets any rounding
/// leftover so we don't lose pixels to integer division).
///
/// **Safety:** when `available` is large enough to be considered
/// unbounded (e.g., the root render's `u32::MAX/2` for the vertical
/// axis), Fr tracks fall back to [`Track::Auto`] semantics rather
/// than expanding to fill an effectively infinite axis.
#[allow(clippy::manual_checked_ops)]
fn solve_tracks(tracks: &[Track], available: u32, auto_size: impl Fn(u32) -> u32) -> Vec<u32> {
    let mut sizes = vec![0u32; tracks.len()];
    let mut consumed = 0u32;
    let mut total_fr = 0u32;
    let unbounded = safety::is_unbounded(available);
    let n = safety::cap_tracks(tracks.len());

    for (i, t) in tracks.iter().take(n).enumerate() {
        match t {
            Track::Px(v) => {
                let v = safety::clamp_dim(*v);
                sizes[i] = v;
                consumed = consumed.saturating_add(v);
            }
            Track::Auto => {
                let v = safety::clamp_dim(auto_size(i as u32));
                sizes[i] = v;
                consumed = consumed.saturating_add(v);
            }
            Track::Percent(p) => {
                let p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
                let v = safety::clamp_dim(((available as f32) * p).round() as u32);
                sizes[i] = v;
                consumed = consumed.saturating_add(v);
            }
            Track::Fr(w) => {
                if unbounded {
                    // Fall back to Auto in unbounded constraints so a
                    // hostile tree can't request a 4-TB canvas.
                    let v = safety::clamp_dim(auto_size(i as u32));
                    sizes[i] = v;
                    consumed = consumed.saturating_add(v);
                } else {
                    total_fr = total_fr.saturating_add(*w);
                }
            }
            Track::FrMin { weight, min_px } => {
                if unbounded {
                    // Unbounded constraint: take just the min_px (no
                    // remainder to distribute fractionally).
                    let v = safety::clamp_dim(*min_px);
                    sizes[i] = v;
                    consumed = consumed.saturating_add(v);
                } else {
                    total_fr = total_fr.saturating_add(*weight);
                }
            }
        }
    }
    if total_fr > 0 {
        // Iterative freeze pass — for each FrMin track whose proportional
        // share of the remainder would fall below its min_px, freeze it
        // at min_px and reduce the remainder + total_fr accordingly.
        // Repeat until stable (each freeze can push another FrMin below
        // its min). For [FrMin] only, this terminates in 1 iteration;
        // for mixed Fr/FrMin, in at most `count(FrMin)` iterations.
        let mut frozen = vec![false; tracks.len()];
        let mut remainder = available.saturating_sub(consumed);
        loop {
            let mut any = false;
            for (i, t) in tracks.iter().take(n).enumerate() {
                if frozen[i] {
                    continue;
                }
                if let Track::FrMin { weight, min_px } = t {
                    let share = if total_fr > 0 {
                        ((remainder as u64) * (*weight as u64) / (total_fr as u64)) as u32
                    } else {
                        0
                    };
                    let min = safety::clamp_dim(*min_px);
                    if share < min {
                        sizes[i] = min;
                        consumed = consumed.saturating_add(min);
                        remainder = remainder.saturating_sub(min);
                        total_fr = total_fr.saturating_sub(*weight);
                        frozen[i] = true;
                        any = true;
                    }
                }
            }
            if !any {
                break;
            }
        }

        // Distribute the (possibly reduced) remainder among the still-
        // unfrozen Fr / FrMin tracks proportional to weight. Last
        // unfrozen track absorbs any rounding remainder.
        let mut allotted = 0u32;
        let mut last_idx: Option<usize> = None;
        for (i, t) in tracks.iter().take(n).enumerate() {
            if frozen[i] {
                continue;
            }
            let weight = match t {
                Track::Fr(w) => *w,
                Track::FrMin { weight, .. } => *weight,
                _ => continue,
            };
            let share = if total_fr > 0 {
                ((remainder as u64) * (weight as u64) / (total_fr as u64)) as u32
            } else {
                0
            };
            sizes[i] = safety::clamp_dim(share);
            allotted = allotted.saturating_add(sizes[i]);
            last_idx = Some(i);
        }
        if let Some(i) = last_idx {
            sizes[i] =
                safety::clamp_dim(sizes[i].saturating_add(remainder.saturating_sub(allotted)));
        }
    } else if !unbounded {
        // No Fr/FrMin tracks at all and content overflowed `available`.
        // Proportionally shrink Auto tracks so the row fits in
        // `available` (Px and Percent are honored as-is — they were
        // explicitly sized by the caller). This trades CSS-grid
        // max-content semantics for graceful degradation when content
        // would otherwise paint past the row.
        let total: u32 = sizes.iter().sum();
        if total > available && available > 0 {
            let mut auto_total = 0u32;
            for (i, t) in tracks.iter().take(n).enumerate() {
                if matches!(t, Track::Auto) {
                    auto_total = auto_total.saturating_add(sizes[i]);
                }
            }
            let non_auto: u32 = total.saturating_sub(auto_total);
            let auto_budget = available.saturating_sub(non_auto);
            if auto_total > auto_budget && auto_total > 0 {
                let mut allotted = 0u32;
                let mut last_auto: Option<usize> = None;
                for (i, t) in tracks.iter().take(n).enumerate() {
                    if matches!(t, Track::Auto) {
                        let share =
                            ((sizes[i] as u64) * (auto_budget as u64) / (auto_total as u64)) as u32;
                        sizes[i] = share;
                        allotted = allotted.saturating_add(share);
                        last_auto = Some(i);
                    }
                }
                if let Some(i) = last_auto {
                    sizes[i] = sizes[i].saturating_add(auto_budget.saturating_sub(allotted));
                }
            }
        }
    }

    // FrMin/Fr-with-Auto overflow case: same shrink logic, applied
    // after the Fr distribution pass. If sum still exceeds available,
    // shrink Auto tracks proportionally to fit (leaving Fr/FrMin sizes
    // alone — they've already negotiated their minimums).
    if !unbounded && total_fr > 0 {
        let total: u32 = sizes.iter().sum();
        if total > available && available > 0 {
            let mut auto_total = 0u32;
            for (i, t) in tracks.iter().take(n).enumerate() {
                if matches!(t, Track::Auto) {
                    auto_total = auto_total.saturating_add(sizes[i]);
                }
            }
            let non_auto: u32 = total.saturating_sub(auto_total);
            let auto_budget = available.saturating_sub(non_auto);
            if auto_total > auto_budget && auto_total > 0 {
                let mut allotted = 0u32;
                let mut last_auto: Option<usize> = None;
                for (i, t) in tracks.iter().take(n).enumerate() {
                    if matches!(t, Track::Auto) {
                        let share =
                            ((sizes[i] as u64) * (auto_budget as u64) / (auto_total as u64)) as u32;
                        sizes[i] = share;
                        allotted = allotted.saturating_add(share);
                        last_auto = Some(i);
                    }
                }
                if let Some(i) = last_auto {
                    sizes[i] = sizes[i].saturating_add(auto_budget.saturating_sub(allotted));
                }
            }
        }
    }

    sizes
}

/// For Auto-track sizing — find the max natural size of any cell whose
/// span includes track index `idx`, divided by span length so a child
/// spanning N tracks contributes 1/N to each.
///
/// The relevant axis is measured with effectively unbounded space so
/// children report their *content* size, not a size clamped to the
/// container's constraint — that's the CSS-grid `auto` (max-content)
/// semantics. The cross axis stays bounded.
fn max_natural_in_track(
    cells: &[(GridSpan, Node)],
    idx: u32,
    is_col: bool,
    gap: (u32, u32),
    max: Size,
) -> u32 {
    let mut out = 0u32;
    let n = safety::cap_cells(cells.len());
    let big = safety::limits().max_dim;
    let measure_max = if is_col {
        Size::new(big, max.h)
    } else {
        Size::new(max.w, big)
    };
    for (span, child) in cells.iter().take(n) {
        let (start, len) = if is_col {
            (span.col, span.cs())
        } else {
            (span.row, span.rs())
        };
        if idx < start || idx >= start + len {
            continue;
        }
        let s = child.measure(measure_max);
        let dim = if is_col { s.w } else { s.h };
        let internal_gap = if is_col { gap.0 } else { gap.1 };
        let spread = dim
            .saturating_sub(internal_gap.saturating_mul(len.saturating_sub(1)))
            .div_ceil(len.max(1));
        out = out.max(spread);
    }
    safety::clamp_dim(out)
}

// ── Measure / paint ────────────────────────────────────────────────────

pub(super) fn measure(
    cols: &[Track],
    rows: &[Track],
    gap: (u32, u32),
    pad: u32,
    cells: &[(GridSpan, Node)],
    max: Size,
) -> Size {
    if cols.is_empty() || rows.is_empty() {
        return Size::ZERO;
    }
    let inner_w = max
        .w
        .saturating_sub(pad * 2)
        .saturating_sub(gap.0.saturating_mul(cols.len().saturating_sub(1) as u32));
    let inner_h = max
        .h
        .saturating_sub(pad * 2)
        .saturating_sub(gap.1.saturating_mul(rows.len().saturating_sub(1) as u32));
    let col_widths = solve_tracks(cols, inner_w, |c| {
        max_natural_in_track(cells, c, true, gap, max)
    });
    let row_heights = solve_tracks(rows, inner_h, |r| {
        max_natural_in_track(cells, r, false, gap, max)
    });
    let total_w =
        pad * 2 + col_widths.iter().sum::<u32>() + gap.0.saturating_mul(cols.len() as u32 - 1);
    let total_h =
        pad * 2 + row_heights.iter().sum::<u32>() + gap.1.saturating_mul(rows.len() as u32 - 1);
    Size::new(total_w.min(max.w), total_h.min(max.h))
}

pub(super) fn paint(
    cols: &[Track],
    rows: &[Track],
    gap: (u32, u32),
    pad: u32,
    cells: &[(GridSpan, Node)],
    rect: Rect,
    canvas: &mut Bitmap,
) {
    if cols.is_empty() || rows.is_empty() {
        return;
    }
    let inner_w = rect
        .w
        .saturating_sub(pad * 2)
        .saturating_sub(gap.0.saturating_mul(cols.len().saturating_sub(1) as u32));
    let inner_h = rect
        .h
        .saturating_sub(pad * 2)
        .saturating_sub(gap.1.saturating_mul(rows.len().saturating_sub(1) as u32));
    let col_widths = solve_tracks(cols, inner_w, |c| {
        max_natural_in_track(cells, c, true, gap, rect.size())
    });
    let row_heights = solve_tracks(rows, inner_h, |r| {
        max_natural_in_track(cells, r, false, gap, rect.size())
    });

    // Cumulative track offsets — col_off[i] is the start of column i
    // (after preceding gaps); col_off[cols.len()] is the end of the
    // last column with no trailing gap.
    let mut col_off = vec![0u32; cols.len() + 1];
    for i in 0..cols.len() {
        col_off[i + 1] = col_off[i] + col_widths[i] + if i + 1 < cols.len() { gap.0 } else { 0 };
    }
    let mut row_off = vec![0u32; rows.len() + 1];
    for i in 0..rows.len() {
        row_off[i + 1] = row_off[i] + row_heights[i] + if i + 1 < rows.len() { gap.1 } else { 0 };
    }

    let n_cells = safety::cap_cells(cells.len());
    for (i, (span, child)) in cells.iter().take(n_cells).enumerate() {
        if span.col >= cols.len() as u32 || span.row >= rows.len() as u32 {
            continue;
        }
        let cs = span.cs().min(cols.len() as u32 - span.col);
        let rs = span.rs().min(rows.len() as u32 - span.row);
        let end_col = (span.col + cs) as usize;
        let end_row = (span.row + rs) as usize;
        // Subtract trailing gap only when ending mid-grid.
        let x_end = if end_col < cols.len() {
            col_off[end_col].saturating_sub(gap.0)
        } else {
            col_off[end_col]
        };
        let y_end = if end_row < rows.len() {
            row_off[end_row].saturating_sub(gap.1)
        } else {
            row_off[end_row]
        };
        let x0 = rect.x + pad + col_off[span.col as usize];
        let y0 = rect.y + pad + row_off[span.row as usize];
        let x1 = rect.x + pad + x_end;
        let y1 = rect.y + pad + y_end;
        let w = x1.saturating_sub(x0);
        let h = y1.saturating_sub(y0);
        let child_rect = Rect::new(x0, y0, w, h);
        super::diagnostics::check_contained(
            super::diagnostics::ContainerKind::Grid,
            i,
            child.kind(),
            rect,
            child_rect,
        );
        child.paint(child_rect, canvas);
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::BLACK;
    use super::super::modifiers::LayoutMod;
    use super::super::node::{empty, image as image_node};
    use super::*;
    use crate::pixel_ops::Bitmap;

    fn solid(w: u32, h: u32, c: super::super::color::Color) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn grid_uniform_fr_fills_constraint() {
        let img = grid()
            .cols(2)
            .equal_rows(2)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 255, 0, 255]).fill())
            .cell(0, 1, empty().background([0, 0, 255, 255]).fill())
            .cell(1, 1, empty().background([255, 255, 0, 255]).fill())
            .size(80, 80)
            .render(80);
        assert_eq!(img.get_pixel(20, 20), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(60, 20), [0, 255, 0, 255]);
        assert_eq!(img.get_pixel(20, 60), [0, 0, 255, 255]);
        assert_eq!(img.get_pixel(60, 60), [255, 255, 0, 255]);
    }

    #[test]
    fn grid_track_px_takes_fixed_size() {
        let img = grid()
            .columns([Track::Px(20), Track::Fr(1)])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 0, 255, 255]).fill())
            .size(100, 10)
            .render(100);
        assert_eq!(img.get_pixel(10, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(50, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn grid_track_auto_hugs_content() {
        let img = grid()
            .columns([Track::Auto, Track::Fr(1)])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, image_node(solid(15, 10, [255, 0, 0, 255])))
            .cell(1, 0, empty().background([0, 0, 255, 255]).fill())
            .size(100, 10)
            .render(100);
        assert_eq!(img.get_pixel(7, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(50, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn frmin_guarantees_minimum_when_remainder_collapses() {
        // [Auto(80), FrMin(1, 10), Auto(80)] in 100 px → FrMin's
        // proportional share = 0 (no remainder), so it must clamp to
        // its 10-px minimum. Total width = 80 + 10 + 80 = 170 (overflow,
        // but FrMin held its minimum).
        let img = grid()
            .columns([
                Track::Px(80),
                Track::FrMin {
                    weight: 1,
                    min_px: 10,
                },
                Track::Px(80),
            ])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 255, 0, 255]).fill())
            .cell(2, 0, empty().background([0, 0, 255, 255]).fill())
            .size(200, 10)
            .render(200);
        // Track positions: red 0..80, green 80..90, blue 90..170.
        // Even though we asked for 100 of available, FrMin took its
        // 10 anyway and pushed blue past col 1's natural end.
        assert_eq!(img.get_pixel(40, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(85, 5), [0, 255, 0, 255]); // FrMin's min
        assert_eq!(img.get_pixel(120, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn auto_tracks_shrink_proportionally_under_overflow() {
        // [Auto(60), FrMin(1, 6), Auto(120)] in 100 px.
        // Sum non-Fr = 180; budget after FrMin's 6 px min = 94.
        // Auto pair shrinks to 94 px split 60:120 → ~31:63.
        let img = grid()
            .columns([
                Track::Auto,
                Track::FrMin {
                    weight: 1,
                    min_px: 6,
                },
                Track::Auto,
            ])
            .equal_rows(1)
            .gap(0)
            .cell(
                0,
                0,
                empty().background([255, 0, 0, 255]).fill().size(60, 10),
            )
            .cell(1, 0, empty().background([0, 255, 0, 255]).fill())
            .cell(
                2,
                0,
                empty().background([0, 0, 255, 255]).fill().size(120, 10),
            )
            .size(100, 10)
            .render(100);
        // Total within available (100). Right cell (blue) ends at x=99.
        assert_eq!(img.get_pixel(99, 5), [0, 0, 255, 255]);
        // Far-left red, far-right blue, with green strip somewhere in between.
        assert_eq!(img.get_pixel(0, 5), [255, 0, 0, 255]);
    }

    #[test]
    fn frmin_distributes_remainder_when_above_min() {
        // [Auto(40), FrMin(1, 6), Auto(40)] in 100 px → remainder 20
        // > min 6, so FrMin gets the full 20.
        let img = grid()
            .columns([
                Track::Px(40),
                Track::FrMin {
                    weight: 1,
                    min_px: 6,
                },
                Track::Px(40),
            ])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 255, 0, 255]).fill())
            .cell(2, 0, empty().background([0, 0, 255, 255]).fill())
            .size(100, 10)
            .render(100);
        // Red 0..40, green 40..60, blue 60..100.
        assert_eq!(img.get_pixel(20, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(50, 5), [0, 255, 0, 255]); // FrMin = 20
        assert_eq!(img.get_pixel(80, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn grid_with_gap_last_col_full_width() {
        // Regression: the last column was being clipped by gap (off-by-pad bug).
        // 2 cols of Px(20), gap 8, padding 8 → total 8 + 20 + 8 + 20 + 8 = 64.
        let img = grid()
            .columns([Track::Px(20), Track::Px(20)])
            .equal_rows(1)
            .gap(8)
            .padding(8)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 0, 255, 255]).fill())
            .size(64, 36)
            .render(64);
        assert_eq!(img.get_pixel(8, 18), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(27, 18), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(36, 18), [0, 0, 255, 255]);
        assert_eq!(img.get_pixel(55, 18), [0, 0, 255, 255]);
        assert_eq!(img.get_pixel(56, 18), BLACK);
    }

    #[test]
    fn grid_areas_parses_template() {
        let img = grid()
            .areas(&["title title", "exp   act"])
            .row_heights([Track::Px(10), Track::Px(20)])
            .gap(0)
            .place("title", empty().background([255, 0, 0, 255]).fill())
            .place("exp", empty().background([0, 255, 0, 255]).fill())
            .place("act", empty().background([0, 0, 255, 255]).fill())
            .size(40, 30)
            .render(40);
        assert_eq!(img.get_pixel(20, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(10, 20), [0, 255, 0, 255]);
        assert_eq!(img.get_pixel(30, 20), [0, 0, 255, 255]);
    }

    #[test]
    fn grid_span_via_areas() {
        let img = grid()
            .areas(&["banner banner", "a a"])
            .equal_rows(2)
            .gap(0)
            .place("banner", empty().background([0, 255, 0, 255]).fill())
            .place("a", empty().background([255, 0, 0, 255]).fill())
            .size(40, 40)
            .render(40);
        assert_eq!(img.get_pixel(10, 10), [0, 255, 0, 255]);
        assert_eq!(img.get_pixel(30, 10), [0, 255, 0, 255]);
        assert_eq!(img.get_pixel(20, 30), [255, 0, 0, 255]);
    }
}
