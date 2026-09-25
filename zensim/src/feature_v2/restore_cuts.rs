//! Restored-cut side passes (COST_CUTS_AUDIT A1 `mapdev`, B2 `z1max`).
//!
//! Both families read the SAME eight per-pixel maps the v1 fused band kernel
//! pools into [`V1BasicSums`] (`sd`, `art`, `det`, the raw squared error
//! `mse`, and the four HF maps `(s−mu1)²`, `(d−mu2)²`, `|s−mu1|`, `|d−mu2|`),
//! re-derived from the kernel's own stored side outputs (`mu1`, `mu2`, `sd`)
//! with the kernel's own f32 formulas — the recipe of the zgeom lane's
//! `v1_channel_maps_and_sums` (research branch `keep/transplant-zgeom-block5`).
//!
//! * **mapdev** — the population standard deviation of `mse`, `hfsq_src`,
//!   `hfsq_dst`, `hfabs_src`, `hfabs_dst` per (scale, channel). Per-row
//!   Welford in f64 (`x` ascending), rows merged in row order with Chan's
//!   formula. Never `Σx² − mean²`. The SSIM/ART/DET map deviations are
//!   functions of the mean and L2 columns already in the surface
//!   (`benchmarks/gmsd_2026-09-22.md`) and are not carried.
//! * **z1max** — the 13 basic + 6 peak slots pooled over the
//!   (0,0)-anchored ungated 5×5 block-MAX lattice (partial border blocks
//!   dropped, `n` = surviving block count) and finalized through
//!   [`V1BasicSums`]'s own finalizers.
//!
//! Both consume rows in plane order inside a serial band loop per (scale,
//! channel) cell, so every result is independent of thread count, SIMD tier
//! and input stride; cells run in parallel when asked. The pyramids are the
//! production materializer's (`build_v2_ref_scales`).

use super::{
    BLUR_RADIUS, MAPDEV_PER_CELL, V1_BAND_OVERLAP, V1_BAND_ROWS, V1BasicSums, Z1MAX_PER_CELL,
    prepare_v2_reference_impl,
};
use crate::feature_defs::FormulaRevision;
use crate::source::ImageSource;

/// Which restored-cut side passes to run.
#[derive(Clone, Copy)]
pub(super) struct Work {
    pub(super) mapdev: bool,
    pub(super) z1max: bool,
}

impl Work {
    pub(super) fn any(self) -> bool {
        self.mapdev || self.z1max
    }
}

/// Running count / mean / sum of squared deviations (Welford), mergeable with
/// Chan's parallel formula. Same shape as C8's `GmsBankCell` deviation state.
#[derive(Clone, Copy, Default)]
struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    /// One row, pixels in ascending `x`. `recip[k] == 1.0 / k as f64`.
    fn of_row(row: &[f32], recip: &[f64]) -> Self {
        let (mut mean, mut m2) = (0.0f64, 0.0f64);
        for (i, &v) in row.iter().enumerate() {
            let x = f64::from(v);
            let delta = x - mean;
            mean += delta * recip[i + 1];
            m2 += delta * (x - mean);
        }
        Self {
            n: row.len() as u64,
            mean,
            m2,
        }
    }

    fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        let total = self.n + other.n;
        let shift = other.mean - self.mean;
        self.m2 += other.m2 + shift * shift * (self.n as f64 * other.n as f64 / total as f64);
        self.mean += shift * (other.n as f64 / total as f64);
        self.n = total;
    }

    /// Population standard deviation.
    fn std(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.m2.max(0.0) / self.n as f64).sqrt()
        }
    }
}

/// The eight per-pixel maps of one row, in `V1BasicSums` accumulator order.
const MAP_SD: usize = 0;
const MAP_ART: usize = 1;
const MAP_DET: usize = 2;
const MAP_MSE: usize = 3;
const MAP_HFSS: usize = 4;
const MAP_HFSD: usize = 5;
const MAP_HFAS: usize = 6;
const MAP_HFAD: usize = 7;

/// Block edge of the z1max lattice (`dvifm::DVIFM_BLOCK`'s convention).
const Z1_GRID: usize = 5;

/// Streaming block-max pooling state for one cell.
struct Z1Acc {
    nbx: usize,
    nby: usize,
    /// Running per-block max of the rows seen so far in the current block
    /// row, per map. Zero-initialised, like the record's `mx = [0.0; 8]`.
    bm: [Vec<f32>; 8],
    sums: V1BasicSums,
}

impl Z1Acc {
    fn new(w: usize, h: usize) -> Self {
        let nbx = w / Z1_GRID;
        Self {
            nbx,
            nby: h / Z1_GRID,
            bm: std::array::from_fn(|_| vec![0.0f32; nbx]),
            sums: V1BasicSums::default(),
        }
    }

    fn push_row(&mut self, y: usize, rows: &[Vec<f32>; 8]) {
        if y >= self.nby * Z1_GRID {
            return;
        }
        for (bm, row) in self.bm.iter_mut().zip(rows) {
            for (m, chunk) in bm.iter_mut().zip(row.as_chunks::<Z1_GRID>().0) {
                let mut v = *m;
                for &d in chunk {
                    v = v.max(d);
                }
                *m = v;
            }
        }
        if y % Z1_GRID == Z1_GRID - 1 {
            for bx in 0..self.nbx {
                let m: [f32; 8] = std::array::from_fn(|i| self.bm[i][bx]);
                accumulate_block(&mut self.sums, m);
            }
            for bm in &mut self.bm {
                bm.fill(0.0);
            }
        }
    }

    fn finalize(&self, revision: FormulaRevision, out: &mut [f64]) {
        debug_assert_eq!(out.len(), Z1MAX_PER_CELL);
        let n = self.nbx * self.nby;
        if n == 0 {
            return;
        }
        self.sums.finalize_into(n, &mut out[0..13], revision);
        let mut peaks = [0.0f64; 6];
        let (mut masked, mut iw) = ([0.0f64; 6], [0.0f64; 6]);
        self.sums
            .finalize_pools_into(n, &mut peaks, &mut masked, &mut iw, revision);
        out[13..19].copy_from_slice(&peaks);
    }
}

/// One block's eight maxima into the pooled sums — the record's `accumulate`
/// closure at gate weight `v = 1` (a multiply by exactly 1.0 is the identity,
/// so it is omitted).
fn accumulate_block(s: &mut V1BasicSums, m: [f32; 8]) {
    let [m_sd, m_art, m_det, m_mse, m_hss, m_hsd, m_has, m_had] = m;
    let sd2 = m_sd * m_sd;
    let sd4 = sd2 * sd2;
    s.ssim_d += f64::from(m_sd);
    s.ssim_d4 += f64::from(sd4);
    s.ssim_d2 += f64::from(sd2);
    s.ssim_d8 += f64::from(sd4 * sd4);
    s.ssim_max = s.ssim_max.max(m_sd);
    let a2 = m_art * m_art;
    let a4 = a2 * a2;
    s.edge_art += f64::from(m_art);
    s.edge_art4 += f64::from(a4);
    s.edge_art2 += f64::from(a2);
    s.edge_art8 += f64::from(a4 * a4);
    s.edge_art_max = s.edge_art_max.max(m_art);
    let d2 = m_det * m_det;
    let d4 = d2 * d2;
    s.edge_det += f64::from(m_det);
    s.edge_det4 += f64::from(d4);
    s.edge_det2 += f64::from(d2);
    s.edge_det8 += f64::from(d4 * d4);
    s.edge_det_max = s.edge_det_max.max(m_det);
    s.mse += f64::from(m_mse);
    s.hf_sq_src += f64::from(m_hss);
    s.hf_sq_dst += f64::from(m_hsd);
    s.hf_abs_src += f64::from(m_has);
    s.hf_abs_dst += f64::from(m_had);
}

/// One (scale, channel) cell's outputs.
struct CellOut {
    dev: [f64; MAPDEV_PER_CELL],
    z: [f64; Z1MAX_PER_CELL],
}

/// Run the v1 band loop over one plane pair and feed every inner row's eight
/// maps to the requested consumers, in plane row order.
fn run_cell(
    sp: &[f32],
    dp: &[f32],
    width: usize,
    height: usize,
    revision: FormulaRevision,
    work: Work,
    recip: &[f64],
) -> CellOut {
    let mut out = CellOut {
        dev: [0.0; MAPDEV_PER_CELL],
        z: [0.0; Z1MAX_PER_CELL],
    };
    if width == 0 || height == 0 {
        return out;
    }
    let band_cap_n = (V1_BAND_ROWS + 2 * V1_BAND_OVERLAP).min(height.max(1)) * width;
    let mut h: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0f32; band_cap_n]);
    let mut mu1_b = vec![0.0f32; band_cap_n];
    let mut mu2_b = vec![0.0f32; band_cap_n];
    let mut sd_b = vec![0.0f32; band_cap_n];
    let mut rows: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0.0f32; width]);
    let free = crate::fused::FreeExtrasWork {
        revision: Some(revision),
        ..Default::default()
    };
    let mut dev = [Welford::default(); MAPDEV_PER_CELL];
    let mut z1 = work.z1max.then(|| Z1Acc::new(width, height));

    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + V1_BAND_ROWS).min(height);
        let top = b0.saturating_sub(V1_BAND_OVERLAP);
        let bot = (b1 + V1_BAND_OVERLAP).min(height);
        let h_local = bot - top;
        let band_n = h_local * width;
        let span = top * width..bot * width;
        let [h0, h1, h2, h3] = &mut h;
        crate::blur::fused_blur_h_ssim_at_revision(
            &sp[span.clone()],
            &dp[span.clone()],
            &mut h0[..band_n],
            &mut h1[..band_n],
            &mut h2[..band_n],
            &mut h3[..band_n],
            width,
            h_local,
            BLUR_RADIUS,
            revision,
        );
        let inner_start = b0 - top;
        let inner_h = b1 - b0;
        // The kernel writes `mu1`, `mu2` and `sd` for the inner rows only.
        let _ = crate::fused::fused_vblur_features_ssim(
            &h0[..band_n],
            &h1[..band_n],
            &h2[..band_n],
            &h3[..band_n],
            &sp[span.clone()],
            &dp[span.clone()],
            width,
            h_local,
            inner_start,
            inner_h,
            BLUR_RADIUS,
            &mut mu1_b[..band_n],
            &mut mu2_b[..band_n],
            true,
            &mut sd_b[..band_n],
            true,
            &mut [],
            &mut [],
            false,
            free,
            crate::fused::ExtPoolsWork::default(),
            &[],
        );
        for lr in inner_start..inner_start + inner_h {
            let y = top + lr;
            let (brow, prow) = (lr * width, y * width);
            #[allow(clippy::needless_range_loop)] // x derives offsets across eight distinct arrays
            for x in 0..width {
                let (bi, pi) = (brow + x, prow + x);
                let (sv, dv) = (sp[pi], dp[pi]);
                let (mu1, mu2) = (mu1_b[bi], mu2_b[bi]);
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let pd = sv - dv;
                let vs = sv - mu1;
                let vd = dv - mu2;
                rows[MAP_SD][x] = sd_b[bi];
                rows[MAP_ART][x] = ed.max(0.0);
                rows[MAP_DET][x] = (-ed).max(0.0);
                rows[MAP_MSE][x] = pd * pd;
                rows[MAP_HFSS][x] = vs * vs;
                rows[MAP_HFSD][x] = vd * vd;
                rows[MAP_HFAS][x] = diff1;
                rows[MAP_HFAD][x] = diff2;
            }
            if work.mapdev {
                for (k, map) in [MAP_MSE, MAP_HFSS, MAP_HFSD, MAP_HFAS, MAP_HFAD]
                    .into_iter()
                    .enumerate()
                {
                    dev[k].merge(&Welford::of_row(&rows[map], recip));
                }
            }
            if let Some(z) = z1.as_mut() {
                z.push_row(y, &rows);
            }
        }
        b0 = b1;
    }
    if work.mapdev {
        for (o, w) in out.dev.iter_mut().zip(&dev) {
            *o = w.std();
        }
    }
    if let Some(z) = &z1 {
        z.finalize(revision, &mut out.z);
    }
    out
}

/// Fill the `mapdev` (60) and `z1max` (228) feature slices. Both are laid out
/// cell-major (`scale * 3 + channel`), matching their `PerChannel` registry
/// blocks. A slice for a family that is not requested is left untouched.
pub(super) fn run(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    parallel: bool,
    revision: FormulaRevision,
    work: Work,
    mapdev_out: &mut [f64],
    z1max_out: &mut [f64],
) {
    let n_scales = crate::NUM_SCALES;
    let src = prepare_v2_reference_impl(source, None, parallel, false)
        .expect("restore-cuts source pyramid");
    let dst = prepare_v2_reference_impl(distorted, None, parallel, false)
        .expect("restore-cuts distorted pyramid");
    let max_w = src.scales.iter().map(|s| s.1).max().unwrap_or(0);
    let recip: Vec<f64> = (0..=max_w)
        .map(|n| if n == 0 { 0.0 } else { 1.0 / n as f64 })
        .collect();
    let cells: Vec<(usize, usize)> = (0..n_scales * 3).map(|i| (i / 3, i % 3)).collect();
    let run_one = |&(scale, ch): &(usize, usize)| {
        let (sp, w, h) = &src.scales[scale];
        let (dp, _, _) = &dst.scales[scale];
        run_cell(&sp[ch], &dp[ch], *w, *h, revision, work, &recip)
    };
    #[cfg(feature = "threads")]
    let outs: Vec<CellOut> = if parallel {
        use rayon::prelude::*;
        cells.par_iter().map(run_one).collect()
    } else {
        cells.iter().map(run_one).collect()
    };
    #[cfg(not(feature = "threads"))]
    let outs: Vec<CellOut> = cells.iter().map(run_one).collect();
    for (i, o) in outs.iter().enumerate() {
        if work.mapdev {
            mapdev_out[i * MAPDEV_PER_CELL..(i + 1) * MAPDEV_PER_CELL].copy_from_slice(&o.dev);
        }
        if work.z1max {
            z1max_out[i * Z1MAX_PER_CELL..(i + 1) * Z1MAX_PER_CELL].copy_from_slice(&o.z);
        }
    }
}

/// Private qualification instrument for the NumPy mirror
/// (`scripts/restore_cuts/numpy_mirror.py`): built only with
/// `RUSTFLAGS='--cfg restore_cuts_instrument'`. Dumps the XYB pyramid planes
/// of four synthetic pairs and the side pass's own outputs for them.
#[cfg(restore_cuts_instrument)]
#[test]
fn restore_cuts_plane_dump() {
    use std::io::Write;
    let dir = std::path::PathBuf::from(std::env::var("RESTORE_CUTS_DUMP_DIR").unwrap());
    std::fs::create_dir(&dir).unwrap();
    let revision = crate::ssim_form::active_revision();
    assert_eq!(
        revision,
        FormulaRevision::Rev3,
        "dump under ZENSIM_FORMULA_REV=3"
    );
    let mut index = std::fs::File::create(dir.join("index.tsv")).unwrap();
    writeln!(index, "case\tside\tscale\tchannel\twidth\theight\tfile").unwrap();
    let mut csv = std::fs::File::create(dir.join("features.csv")).unwrap();
    write!(csv, "case").unwrap();
    for i in 0..60 {
        write!(csv, ",mapdev{i}").unwrap();
    }
    for i in 0..228 {
        write!(csv, ",z1max{i}").unwrap();
    }
    for i in 0..30 {
        write!(csv, ",gmsnative{i}").unwrap();
    }
    for i in 0..5 {
        write!(csv, ",dvifmgate{i}").unwrap();
    }
    writeln!(csv).unwrap();
    for &(w, h) in &[(64usize, 64usize), (97, 73), (200, 150), (333, 257)] {
        let mut state = 0x9E37_79B9u32 ^ (w as u32 * 31 + h as u32);
        let mut rnd = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        };
        let src: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let edge = if (x / 23 + y / 17) % 2 == 0 { 40 } else { 0 };
                let v = ((x * 5 + y * 3) % 200) as u8 / 2 + 30 + edge;
                [
                    v,
                    v.wrapping_add(rnd() % 12),
                    v.wrapping_mul(2) / 2 + rnd() % 9,
                ]
            })
            .collect();
        let dst: Vec<[u8; 3]> = src
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let blk = ((i % w) / 8 + (i / w) / 8) as u8 % 3;
                [
                    (p[0] & 0xF8).saturating_add(blk),
                    p[1].saturating_add(rnd() % 7),
                    p[2] / 2 * 2,
                ]
            })
            .collect();
        let (s, d) = (
            crate::RgbSlice::new(&src, w, h),
            crate::RgbSlice::new(&dst, w, h),
        );
        let case = format!("{w}x{h}");
        let (ps, pd) = (
            prepare_v2_reference_impl(&s, None, false, false).unwrap(),
            prepare_v2_reference_impl(&d, None, false, false).unwrap(),
        );
        for (side, prep) in [(0, &ps), (1, &pd)] {
            for (scale, (planes, pw, ph)) in prep.scales.iter().enumerate() {
                for (ch, plane) in planes.iter().enumerate() {
                    let file = format!("{case}_{side}_{scale}_{ch}.f32");
                    let mut bytes = Vec::with_capacity(plane.len() * 4);
                    for v in plane {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(dir.join(&file), bytes).unwrap();
                    writeln!(index, "{case}\t{side}\t{scale}\t{ch}\t{pw}\t{ph}\t{file}").unwrap();
                }
            }
        }
        // The values under test come from the production walk (all four
        // families on), not from the side pass alone.
        let toggles = super::V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            csfw_block: true,
            dvifm_block: true,
            rev4_gridblk: true,
            rev4_ringbasis: true,
            rev4_tailhist: true,
            rev4_arttype: true,
            gmsbank: true,
            mapdev: true,
            z1max: true,
            gmsnative: true,
            dvifmgate: true,
            v1_pools: super::V1PoolsMode::Full,
            ..Default::default()
        };
        let mut scratch = super::V2Scratch::new();
        let f = super::compute_folded720_streaming_impl(
            &s,
            &d,
            None,
            false,
            toggles,
            &mut scratch,
            None,
        )
        .expect("walk")
        .into_features();
        assert_eq!(f.len(), 1825);
        write!(csv, "{case}").unwrap();
        for v in &f[1502..] {
            write!(csv, ",{v:e}").unwrap();
        }
        writeln!(csv).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32) / (1u64 << 24) as f32
    }

    fn recip(n: usize) -> Vec<f64> {
        (0..=n)
            .map(|k| if k == 0 { 0.0 } else { 1.0 / k as f64 })
            .collect()
    }

    /// Row Welford + Chan row merge agrees with an f64 two-pass reference on
    /// data with a large common offset, where `Σx²−mean²` loses its digits,
    /// and is exactly zero on constant rows. The noise sits on the f32 grid
    /// (multiples of 2^-10), so every value is exactly representable.
    #[test]
    fn welford_row_merge_matches_two_pass_and_survives_an_offset() {
        let (w, h) = (37usize, 29usize);
        let r = recip(w);
        let mut seed = 7u64;
        let q = 1.0f32 / 1024.0;
        for offset in [0.0f32, 1.0e3, 1.0e4] {
            let plane: Vec<f32> = (0..w * h)
                .map(|_| offset + (lcg(&mut seed) * 4.0).floor() * q)
                .collect();
            let mut acc = Welford::default();
            for row in plane.chunks_exact(w) {
                acc.merge(&Welford::of_row(row, &r));
            }
            let x: Vec<f64> = plane.iter().map(|&v| f64::from(v)).collect();
            let n = x.len() as f64;
            let mean = x.iter().sum::<f64>() / n;
            let two_pass = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
            let rel = (acc.std() - two_pass).abs() / two_pass;
            // Both estimates carry ~eps*|mean| absolute error in the mean, i.e.
            // eps*offset/spread relative to the spread: the bound scales with it.
            let bound = 1e-12 + 8.0 * f64::EPSILON * (f64::from(offset) / two_pass);
            assert!(rel < bound, "offset {offset}: rel {rel:e} >= {bound:e}");
            if offset >= 1.0e4 {
                // The discriminating case: the raw-moment formula is wrecked here.
                let raw = ((x.iter().map(|v| v * v).sum::<f64>() / n) - mean * mean)
                    .max(0.0)
                    .sqrt();
                let raw_rel = (raw - two_pass).abs() / two_pass;
                assert!(
                    raw_rel > 1e-5,
                    "offset case must break Σx²−mean² (rel {raw_rel:e})"
                );
            }
        }
        let flat = vec![0.75f32; w * h];
        let mut acc = Welford::default();
        for row in flat.chunks_exact(w) {
            acc.merge(&Welford::of_row(row, &r));
        }
        assert_eq!(acc.std(), 0.0);
    }

    /// The streaming block-max pooler equals a naive whole-plane block-max
    /// pooling over the same maps, bit for bit, for sizes with partial border
    /// blocks in both axes and for a plane smaller than one block.
    #[test]
    fn streaming_block_max_matches_naive_lattice() {
        for (w, h) in [(23usize, 17usize), (40, 40), (4, 9), (11, 5)] {
            let mut seed = 99u64 ^ (w * 1000 + h) as u64;
            let maps: Vec<Vec<f32>> = (0..8)
                .map(|_| (0..w * h).map(|_| lcg(&mut seed)).collect())
                .collect();
            let mut z = Z1Acc::new(w, h);
            let mut rows: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0.0; w]);
            for y in 0..h {
                for (k, r) in rows.iter_mut().enumerate() {
                    r.copy_from_slice(&maps[k][y * w..(y + 1) * w]);
                }
                z.push_row(y, &rows);
            }
            let (nbx, nby) = (w / 5, h / 5);
            let mut want = V1BasicSums::default();
            for by in 0..nby {
                for bx in 0..nbx {
                    let m: [f32; 8] = std::array::from_fn(|k| {
                        let mut v = 0.0f32;
                        for r in by * 5..by * 5 + 5 {
                            for c in bx * 5..bx * 5 + 5 {
                                v = v.max(maps[k][r * w + c]);
                            }
                        }
                        v
                    });
                    accumulate_block(&mut want, m);
                }
            }
            let (mut a, mut b) = ([0.0f64; Z1MAX_PER_CELL], [0.0f64; Z1MAX_PER_CELL]);
            z.finalize(FormulaRevision::Rev3, &mut a);
            if nbx * nby > 0 {
                want.finalize_into(nbx * nby, &mut b[0..13], FormulaRevision::Rev3);
                let (mut p, mut m, mut i) = ([0.0; 6], [0.0; 6], [0.0; 6]);
                want.finalize_pools_into(nbx * nby, &mut p, &mut m, &mut i, FormulaRevision::Rev3);
                b[13..19].copy_from_slice(&p);
            }
            for k in 0..Z1MAX_PER_CELL {
                assert_eq!(a[k].to_bits(), b[k].to_bits(), "{w}x{h} slot {k}");
            }
        }
    }

    /// The registry's per-cell widths are the ones the walk allocates.
    #[test]
    fn registry_widths_match_the_walk_constants() {
        use crate::feature_set_id::ComputeToken as T;
        let ns = crate::NUM_SCALES;
        let w = |t| {
            crate::feature_defs::block_base(t, ns)
                .expect("registered")
                .1
                .width(ns)
        };
        assert_eq!(w(T::Mapdev), ns * 3 * MAPDEV_PER_CELL);
        assert_eq!(w(T::Z1max), ns * 3 * Z1MAX_PER_CELL);
        assert_eq!(w(T::Gmsnative), super::super::GMSNATIVE_WIDTH);
        assert_eq!(
            crate::feature_defs::block_base(T::Mapdev, ns).unwrap().0,
            1502
        );
        assert_eq!(
            crate::feature_defs::block_base(T::Z1max, ns).unwrap().0,
            1562
        );
        assert_eq!(
            crate::feature_defs::block_base(T::Gmsnative, ns).unwrap().0,
            1790
        );
    }
}
