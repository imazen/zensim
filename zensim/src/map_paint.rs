//! Research painting modes for the candidate attribution map
//! (`BakeScorer::compute_with_ref_and_attribution`; exploration record
//! `benchmarks/byconstruction_2026-10-22.md`, "Maps").
//!
//! The map adds, per pyramid scale, a pixel-allocated plane (`id`) and a
//! window-supported plane (`win`: signals that summarize an 11-tap window,
//! such as the SSIM error), then paints the scale onto the source grid. Two
//! choices shape what the map looks like:
//!
//! - **window** — where a windowed signal's mass goes. `box` (the default,
//!   unchanged bits) spreads it uniformly over its window, so one defect
//!   becomes an 11-tap plateau at its scale (88 source pixels at scale 3).
//!   `center` keeps it at the window's own pixel, as DVIFM keeps a block's
//!   term at its block. `guided` sends it back to the window's pixels in
//!   proportion to their own squared difference (each channel normalized by
//!   the reference's variance at that scale), so the mass lands where the two
//!   images actually differ. Every mode conserves the plane's sum.
//! - **paint** — how a coarse pixel covers its `2^s × 2^s` source footprint.
//!   `nearest` (the default) spreads it uniformly, as DVIFM paints a block.
//!   `guided` splits it in proportion to the source-resolution squared
//!   difference inside the footprint (per-pixel refinement); it applies only
//!   to a per-pixel map (`bin == 1`).
//!
//! A scorer selects them per map (`BakeScorer::with_map_paint`); research
//! command lines can set `ZENSIM_MAP_WINDOW=box|center|guided` and
//! `ZENSIM_MAP_PAINT=nearest|guided` instead. Nothing in the score path reads
//! them.

/// Where a window-supported signal's mass goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowMode {
    Box,
    Center,
    Guided,
}

/// How a coarse pixel covers its source footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PaintMode {
    Nearest,
    Guided,
}

/// The two painting choices of one map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MapPaint {
    pub(crate) window: WindowMode,
    pub(crate) paint: PaintMode,
}

impl Default for MapPaint {
    /// The uniform window spread and nearest-neighbour fold: today's map.
    fn default() -> Self {
        Self {
            window: WindowMode::Box,
            paint: PaintMode::Nearest,
        }
    }
}

impl MapPaint {
    /// Parse `box|center|guided` and `nearest|guided`.
    pub(crate) fn parse(window: &str, paint: &str) -> Option<Self> {
        let window = match window {
            "box" => WindowMode::Box,
            "center" => WindowMode::Center,
            "guided" => WindowMode::Guided,
            _ => return None,
        };
        let paint = match paint {
            "nearest" => PaintMode::Nearest,
            "guided" => PaintMode::Guided,
            _ => return None,
        };
        Some(Self { window, paint })
    }

    /// The process's research selection (`ZENSIM_MAP_WINDOW`,
    /// `ZENSIM_MAP_PAINT`), read once; unset means [`Self::default`] and an
    /// unknown value panics.
    pub(crate) fn from_env() -> Self {
        use std::sync::OnceLock;
        static MODES: OnceLock<MapPaint> = OnceLock::new();
        *MODES.get_or_init(|| {
            let window = std::env::var("ZENSIM_MAP_WINDOW").unwrap_or_else(|_| "box".into());
            let paint = std::env::var("ZENSIM_MAP_PAINT").unwrap_or_else(|_| "nearest".into());
            Self::parse(&window, &paint).unwrap_or_else(|| {
                panic!("ZENSIM_MAP_WINDOW={window:?} / ZENSIM_MAP_PAINT={paint:?}: expected box|center|guided and nearest|guided")
            })
        })
    }
}

/// Clipped box sum of radius `r` (each output sums the in-bounds inputs of
/// its `(2r+1)²` window), in f64.
fn box_sum(src: &[f64], w: usize, h: usize, r: usize, out: &mut Vec<f64>) {
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut acc = 0.0f64;
        let mut lo = 0usize;
        let mut hi = 0usize; // window [lo, hi)
        for x in 0..w {
            let want_hi = (x + r + 1).min(w);
            while hi < want_hi {
                acc += row[hi];
                hi += 1;
            }
            let want_lo = x.saturating_sub(r);
            while lo < want_lo {
                acc -= row[lo];
                lo += 1;
            }
            tmp[y * w + x] = acc;
        }
    }
    out.clear();
    out.resize(w * h, 0.0);
    for x in 0..w {
        let mut acc = 0.0f64;
        let mut lo = 0usize;
        let mut hi = 0usize;
        for y in 0..h {
            let want_hi = (y + r + 1).min(h);
            while hi < want_hi {
                acc += tmp[hi * w + x];
                hi += 1;
            }
            let want_lo = y.saturating_sub(r);
            while lo < want_lo {
                acc -= tmp[lo * w + x];
                lo += 1;
            }
            out[y * w + x] = acc;
        }
    }
}

/// The guide: per-pixel squared difference, each channel normalized by the
/// reference's variance on this plane, plus a small uniform floor so a
/// window without any difference still keeps (and spreads) its mass.
pub(crate) fn difference_guide(src: [&[f32]; 3], dst: [&[f32]; 3], n: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; n];
    for c in 0..3 {
        let (s, d) = (src[c], dst[c]);
        if s.len() < n || d.len() < n {
            continue; // a channel this walk did not keep (scale-0 chroma)
        }
        let mean = s[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = s[..n]
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let k = 1.0 / (var + 1e-6);
        for i in 0..n {
            let e = (s[i] - d[i]) as f64;
            g[i] += k * e * e;
        }
    }
    let mean = g.iter().sum::<f64>() / n as f64;
    let floor = 1e-3 * mean + 1e-30;
    for v in &mut g {
        *v += floor;
    }
    g
}

/// Merge `win` into `id` under `mode`. `Box` is the existing fused spread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_window(
    mode: WindowMode,
    win: &mut [f32],
    id: &mut [f32],
    w: usize,
    h: usize,
    radius: usize,
    guide: impl FnOnce() -> Vec<f64>,
    spread_tmp: &mut Vec<f32>,
    spread_out: &mut Vec<f32>,
    parallel: bool,
) {
    let n = w * h;
    match mode {
        WindowMode::Box => crate::blur::box_spread_merge_f32(
            &mut win[..n],
            &mut id[..n],
            w,
            h,
            radius,
            spread_tmp,
            spread_out,
            parallel,
        ),
        WindowMode::Center => {
            for i in 0..n {
                id[i] += win[i];
            }
            win[..n].fill(0.0);
        }
        WindowMode::Guided => {
            // out(q) = g(q) · Σ_{p ∋ q} win(p) / Σ_{q' ∈ W(p)} g(q').
            let g = guide();
            let mut s = Vec::new();
            box_sum(&g, w, h, radius, &mut s);
            let ratio: Vec<f64> = (0..n).map(|i| win[i] as f64 / s[i]).collect();
            let mut back = Vec::new();
            box_sum(&ratio, w, h, radius, &mut back);
            for i in 0..n {
                id[i] += (g[i] * back[i]) as f32;
            }
            win[..n].fill(0.0);
        }
    }
}

/// Paint a scale plane onto a `w0 × h0` source grid, splitting each coarse
/// pixel over its `2^scale` footprint in proportion to `guide0` (source
/// resolution); footprint cells past the image edge are dropped, as the
/// nearest-neighbour fold drops them.
pub(crate) fn paint_guided(
    plane: &[f32],
    ws: usize,
    hs: usize,
    scale: usize,
    guide0: &[f64],
    w0: usize,
    h0: usize,
) -> Vec<f32> {
    let f = 1usize << scale;
    let mut out = vec![0.0f32; w0 * h0];
    for cy in 0..hs {
        let y0 = cy * f;
        if y0 >= h0 {
            break;
        }
        let y1 = (y0 + f).min(h0);
        for cx in 0..ws {
            let x0 = cx * f;
            if x0 >= w0 {
                break;
            }
            let x1 = (x0 + f).min(w0);
            let m = plane[cy * ws + cx] as f64;
            if m == 0.0 {
                continue;
            }
            let mut total = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    total += guide0[y * w0 + x];
                }
            }
            for y in y0..y1 {
                for x in x0..x1 {
                    out[y * w0 + x] += (m * guide0[y * w0 + x] / total) as f32;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(w: usize, h: usize) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
        let s: [Vec<f32>; 3] = core::array::from_fn(|c| {
            (0..w * h)
                .map(|i| 0.2 + 0.01 * c as f32 + ((i * 7 % 13) as f32) * 0.01)
                .collect()
        });
        let mut d = s.clone();
        d[1][(h / 2) * w + w / 2] += 0.3; // one defect pixel
        (s, d)
    }

    #[test]
    fn every_window_mode_conserves_mass() {
        let (w, h) = (37, 29);
        let (s, d) = planes(w, h);
        let win0: Vec<f32> = (0..w * h).map(|i| ((i * 31 % 17) as f32) * 0.01).collect();
        let total: f64 = win0.iter().map(|&v| v as f64).sum();
        for mode in [WindowMode::Box, WindowMode::Center, WindowMode::Guided] {
            let mut win = win0.clone();
            let mut id = vec![0.0f32; w * h];
            let (mut t, mut o) = (Vec::new(), Vec::new());
            let sr = [&s[0][..], &s[1][..], &s[2][..]];
            let dr = [&d[0][..], &d[1][..], &d[2][..]];
            merge_window(
                mode,
                &mut win,
                &mut id,
                w,
                h,
                5,
                || difference_guide(sr, dr, w * h),
                &mut t,
                &mut o,
                false,
            );
            let got: f64 = id.iter().map(|&v| v as f64).sum();
            assert!(
                (got - total).abs() <= 1e-4 * total,
                "{mode:?}: {got} vs {total}"
            );
        }
    }

    #[test]
    fn guided_window_mass_concentrates_on_the_difference() {
        let (w, h) = (41, 41);
        let (s, d) = planes(w, h);
        // A plateau: every pixel whose window holds the defect carries error.
        let (cx, cy) = (w / 2, h / 2);
        let mut win = vec![0.0f32; w * h];
        for y in cy - 5..=cy + 5 {
            for x in cx - 5..=cx + 5 {
                win[y * w + x] = 1.0;
            }
        }
        let mut id = vec![0.0f32; w * h];
        let (mut t, mut o) = (Vec::new(), Vec::new());
        let sr = [&s[0][..], &s[1][..], &s[2][..]];
        let dr = [&d[0][..], &d[1][..], &d[2][..]];
        merge_window(
            WindowMode::Guided,
            &mut win,
            &mut id,
            w,
            h,
            5,
            || difference_guide(sr, dr, w * h),
            &mut t,
            &mut o,
            false,
        );
        let total: f32 = id.iter().sum();
        assert!(
            id[cy * w + cx] > 0.9 * total,
            "defect pixel holds {} of {total}",
            id[cy * w + cx]
        );
    }

    #[test]
    fn guided_paint_conserves_each_footprint() {
        let (w0, h0) = (21, 13);
        let scale = 2;
        let (ws, hs) = (w0 >> scale, h0 >> scale);
        let plane: Vec<f32> = (0..ws * hs).map(|i| 1.0 + i as f32).collect();
        let guide0: Vec<f64> = (0..w0 * h0).map(|i| 0.5 + (i % 5) as f64).collect();
        let out = paint_guided(&plane, ws, hs, scale, &guide0, w0, h0);
        let f = 1 << scale;
        for cy in 0..hs {
            for cx in 0..ws {
                let mut sum = 0.0f64;
                for y in cy * f..cy * f + f {
                    for x in cx * f..cx * f + f {
                        sum += out[y * w0 + x] as f64;
                    }
                }
                let want = plane[cy * ws + cx] as f64;
                assert!(
                    (sum - want).abs() <= 1e-5 * want,
                    "cell {cx},{cy}: {sum} vs {want}"
                );
            }
        }
    }
}
