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
//! channel) cell, so every result is independent of thread count and input
//! stride, and repeatable within a SIMD tier; cells run in parallel when
//! asked. Results are NOT bit-identical ACROSS SIMD tiers: the XYB planes and
//! the tier-dispatched v1 blur/kernel stages they read differ at float
//! precision between tiers (measured in `tests/restore_cuts_parity.rs`; the
//! stage responsible was not isolated). The pyramids are the
//! production materializer's (`build_v2_ref_scales`).

use super::{
    BLUR_RADIUS, C_HF, MAPDEV_PER_CELL, SATSIGN_PER_SCALE, TEXGAIN_PER_CELL, V1_BAND_OVERLAP,
    V1_BAND_ROWS, V1BasicSums, Z1MAX_PER_CELL, prepare_v2_reference_impl,
};
use crate::feature_defs::FormulaRevision;
use crate::source::ImageSource;

/// Which restored-cut side passes to run.
#[derive(Clone, Copy)]
pub(super) struct Work {
    pub(super) mapdev: bool,
    pub(super) z1max: bool,
    /// SIGNEDFEAT S1: texture-magnitude gain, accumulated in the same band loop.
    pub(super) texgain: bool,
    /// SIGNEDFEAT S2: signed chroma-saturation change, a separate (X, B) pass per scale.
    pub(super) satsign: bool,
}

impl Work {
    pub(super) fn any(self) -> bool {
        self.mapdev || self.z1max || self.texgain || self.satsign
    }

    /// Does any consumer of the per-band map loop run?
    fn band(self) -> bool {
        self.mapdev || self.z1max || self.texgain
    }
}

/// SIGNEDFEAT S2 chroma normalisation (neutral-axis centring, see `satsign_scale`): C8's mid chromaticity stabilisers
/// (`GMSBANK_CS_C[2]`, frozen from TRAIN pixels by the C8 lane). `m = sqrt(Xc²/cx + Bc²/cb)`.
const SAT_CX: f64 = crate::gmsbank_constants::GMSBANK_CS_C[2][0];
const SAT_CB: f64 = crate::gmsbank_constants::GMSBANK_CS_C[2][1];
/// Stabiliser of the saturation gain/loss form: the unit that `chromaticity_loss` adds to normalised
/// chroma energy, in the same units as `m`.
const C_SAT: f64 = 1.0;

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
    /// `row` is `f64` — production values are `f64::from` of the f32 maps
    /// (lossless), so the arithmetic is unchanged.
    fn of_row(row: &[f64], recip: &[f64]) -> Self {
        let (mut mean, mut m2) = (0.0f64, 0.0f64);
        for (i, &x) in row.iter().enumerate() {
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
    /// featacc: f64 so `exact` mode can carry the f64 map evaluation through
    /// the max without narrowing; production values were f32, and f32→f64 is
    /// lossless, so production bits are unchanged.
    bm: [Vec<f64>; 8],
    sums: V1BasicSums,
    /// Oracle element precision: `exact` squares in f64, the rest reproduce
    /// the production f32 square-then-widen.
    exact: bool,
}

impl Z1Acc {
    fn new(w: usize, h: usize) -> Self {
        let nbx = w / Z1_GRID;
        Self {
            nbx,
            nby: h / Z1_GRID,
            bm: std::array::from_fn(|_| vec![0.0f64; nbx]),
            sums: V1BasicSums::meas_f32(),
            exact: crate::featcanon::compute_mode().exact(),
        }
    }

    fn push_row(&mut self, y: usize, rows: &[Vec<f64>; 8]) {
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
                let m: [f64; 8] = std::array::from_fn(|i| self.bm[i][bx]);
                accumulate_block(&mut self.sums, m, bx & 7, self.exact);
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
///
/// featacc: `lane` is the canonical pool lane (block column mod 8).
/// `exact = false` reproduces the production element form — powers computed
/// in f32 then widened — while `true` evaluates them in f64 from the f64
/// block maxima (the oracle's element axis).
fn accumulate_block(s: &mut V1BasicSums, m: [f64; 8], lane: usize, exact: bool) {
    let [m_sd, m_art, m_det, m_mse, m_hss, m_hsd, m_has, m_had] = m;
    // Production squares in f32 (`sd2 = m_sd * m_sd` on f32 operands) and
    // widens; `v as f32` below is a no-op for f32-sourced maxima and is the
    // production element value for f64-sourced ones — which `exact` then
    // deliberately does NOT take.
    let p2 = |v: f64| {
        if exact {
            v * v
        } else {
            let v32 = v as f32;
            f64::from(v32 * v32)
        }
    };
    let p1 = |v: f64| {
        if exact { v } else { f64::from(v as f32) }
    };
    let sd2 = p2(m_sd);
    let sd4 = p2(sd2);
    s.ssim_d.add64(lane, p1(m_sd));
    s.ssim_d4.add64(lane, sd4);
    s.ssim_d2.add64(lane, sd2);
    s.ssim_d8.add64(lane, p2(sd4));
    s.ssim_max = s.ssim_max.max(m_sd);
    let a2 = p2(m_art);
    let a4 = p2(a2);
    s.edge_art.add64(lane, p1(m_art));
    s.edge_art4.add64(lane, a4);
    s.edge_art2.add64(lane, a2);
    s.edge_art8.add64(lane, p2(a4));
    s.edge_art_max = s.edge_art_max.max(m_art);
    let d2 = p2(m_det);
    let d4 = p2(d2);
    s.edge_det.add64(lane, p1(m_det));
    s.edge_det4.add64(lane, d4);
    s.edge_det2.add64(lane, d2);
    s.edge_det8.add64(lane, p2(d4));
    s.edge_det_max = s.edge_det_max.max(m_det);
    s.mse.add64(lane, p1(m_mse));
    s.hf_sq_src.add64(lane, p1(m_hss));
    s.hf_sq_dst.add64(lane, p1(m_hsd));
    s.hf_abs_src.add64(lane, p1(m_has));
    s.hf_abs_dst.add64(lane, p1(m_had));
}

/// One (scale, channel) cell's outputs.
struct CellOut {
    dev: [f64; MAPDEV_PER_CELL],
    z: [f64; Z1MAX_PER_CELL],
    tex: [f64; TEXGAIN_PER_CELL],
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
        tex: [0.0; TEXGAIN_PER_CELL],
    };
    if width == 0 || height == 0 {
        return out;
    }
    let band_cap_n = (V1_BAND_ROWS + 2 * V1_BAND_OVERLAP).min(height.max(1)) * width;
    let mut h: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0f32; band_cap_n]);
    let mut mu1_b = vec![0.0f32; band_cap_n];
    let mut mu2_b = vec![0.0f32; band_cap_n];
    let mut sd_b = vec![0.0f32; band_cap_n];
    // featacc: the per-pixel maps are f64 so `exact` mode can evaluate the
    // map formulas in f64; every other mode stores `f64::from` of the
    // production f32 evaluation (lossless — production bits unchanged).
    let mut rows: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0f64; width]);
    let free = crate::fused::FreeExtrasWork {
        revision: Some(revision),
        ..Default::default()
    };
    // featacc: under a measurement mode the mapdev pools run as
    // `WelfordVar` fed PER PIXEL (`Seq` = the canonical sequential Welford,
    // `Lanes` = 8 substreams on x mod 8 Chan-merged pairwise, `Neum` =
    // compensated). Production (`off`/unset) keeps the row-Welford + Chan
    // row merge exactly as shipped. `exact` also re-evaluates the eight
    // maps' element formulas in f64; every other mode keeps the production
    // f32 evaluation (widened losslessly into the f64 rows).
    let mode = crate::featcanon::compute_mode();
    let exact = mode.exact();
    // rev4canon: `for_mode` maps `Canon64` (the Rev4 canon) to `Lanes`, the
    // `c32` oracle arm to `Seq`, and `Off` leaves `dev_var` unset so the
    // shipped row-merge path runs byte-for-byte.
    let mut dev_var = mode
        .active()
        .then(|| [crate::featcanon::WelfordVar::for_mode(mode); MAPDEV_PER_CELL]);
    let mut dev = [Welford::default(); MAPDEV_PER_CELL];
    let mut z1 = work.z1max.then(|| Z1Acc::new(width, height));
    // S1: ascending-row f64 sum of the per-row ascending-x f64 sums.
    let mut tex_sum = 0.0f64;

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
                let sd = sd_b[bi];
                if exact {
                    // f64 element evaluation of the same map formulas — the
                    // oracle's element axis (the f32 blur planes stay the
                    // terms' inputs; their precision is the blur axis's
                    // separate measurement).
                    let (sv, dv) = (f64::from(sv), f64::from(dv));
                    let (mu1, mu2) = (f64::from(mu1), f64::from(mu2));
                    let diff1 = (sv - mu1).abs();
                    let diff2 = (dv - mu2).abs();
                    let ed = (1.0f64 + diff2) / (1.0f64 + diff1) - 1.0f64;
                    let pd = sv - dv;
                    let vs = sv - mu1;
                    let vd = dv - mu2;
                    rows[MAP_SD][x] = f64::from(sd);
                    rows[MAP_ART][x] = ed.max(0.0);
                    rows[MAP_DET][x] = (-ed).max(0.0);
                    rows[MAP_MSE][x] = pd * pd;
                    rows[MAP_HFSS][x] = vs * vs;
                    rows[MAP_HFSD][x] = vd * vd;
                    rows[MAP_HFAS][x] = diff1;
                    rows[MAP_HFAD][x] = diff2;
                } else {
                    // Production's f32 element evaluation, widened losslessly.
                    let diff1 = (sv - mu1).abs();
                    let diff2 = (dv - mu2).abs();
                    let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                    let pd = sv - dv;
                    let vs = sv - mu1;
                    let vd = dv - mu2;
                    rows[MAP_SD][x] = f64::from(sd);
                    rows[MAP_ART][x] = f64::from(ed.max(0.0));
                    rows[MAP_DET][x] = f64::from((-ed).max(0.0));
                    rows[MAP_MSE][x] = f64::from(pd * pd);
                    rows[MAP_HFSS][x] = f64::from(vs * vs);
                    rows[MAP_HFSD][x] = f64::from(vd * vd);
                    rows[MAP_HFAS][x] = f64::from(diff1);
                    rows[MAP_HFAD][x] = f64::from(diff2);
                }
            }
            if work.texgain {
                // `bounded_excess(|hf_d|, |hf_s|, C_HF)` on the f32 maps' widened `|s-mu1|`, `|d-mu2|`.
                let mut row_sum = 0.0f64;
                for (&a, &b) in rows[MAP_HFAS].iter().zip(&rows[MAP_HFAD]) {
                    row_sum += (b - a).max(0.0) / (b + a + C_HF);
                }
                tex_sum += row_sum;
            }
            if work.mapdev {
                if let Some(devv) = dev_var.as_mut() {
                    for (k, map) in [MAP_MSE, MAP_HFSS, MAP_HFSD, MAP_HFAS, MAP_HFAD]
                        .into_iter()
                        .enumerate()
                    {
                        for (x, &v) in rows[map].iter().enumerate() {
                            devv[k].push(x, v);
                        }
                    }
                } else {
                    for (k, map) in [MAP_MSE, MAP_HFSS, MAP_HFSD, MAP_HFAS, MAP_HFAD]
                        .into_iter()
                        .enumerate()
                    {
                        dev[k].merge(&Welford::of_row(&rows[map], recip));
                    }
                }
            }
            if let Some(z) = z1.as_mut() {
                z.push_row(y, &rows);
            }
        }
        b0 = b1;
    }
    if work.mapdev {
        if let Some(devv) = dev_var.as_ref() {
            for (o, w) in out.dev.iter_mut().zip(devv) {
                let s = w.stats();
                *o = if s.n == 0 {
                    0.0
                } else {
                    (s.m2.max(0.0) / s.n as f64).sqrt()
                };
            }
        } else {
            for (o, w) in out.dev.iter_mut().zip(&dev) {
                *o = w.std();
            }
        }
    }
    if let Some(z) = &z1 {
        z.finalize(revision, &mut out.z);
    }
    if work.texgain {
        out.tex[0] = (tex_sum / (width * height) as f64).clamp(0.0, 1.0);
    }
    out
}

/// SIGNEDFEAT S2 for one scale: the (X, B) chroma-magnitude gain/loss. `[sat_gain, sat_loss,
/// gsat_gain, gsat_loss]`. `xs/bs` are the reference X and B planes, `xd/bd` the distorted ones.
/// All arithmetic is scalar f64 in a fixed order (ascending x within a row, rows ascending), so the
/// result does not depend on the dispatch tier or the thread count.
fn satsign_scale(
    xs: &[f32],
    bs: &[f32],
    xd: &[f32],
    bd: &[f32],
    width: usize,
    height: usize,
) -> [f64; SATSIGN_PER_SCALE] {
    if width == 0 || height == 0 {
        return [0.0; SATSIGN_PER_SCALE];
    }
    // Neutral-axis centring: on a gray R=G=B the opsin B-plane value is `0.55 - (-cbrt(K_B0))` for every
    // luminance (the absorbance bias), so subtracting it makes `m = 0` on the neutral axis.
    let (x0, b0) = (
        f64::from(0.42_f32),
        f64::from(0.55_f32) + f64::from(crate::color::K_B0).cbrt(),
    );
    let m = |x: f32, b: f32| {
        let (xc, bc) = (f64::from(x) - x0, f64::from(b) - b0);
        (xc * xc / SAT_CX + bc * bc / SAT_CB).sqrt()
    };
    let (mut gain, mut loss, mut sum_r, mut sum_d) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..height {
        let (mut rg, mut rl, mut rr, mut rd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in y * width..(y + 1) * width {
            let (mr, md) = (m(xs[i], bs[i]), m(xd[i], bd[i]));
            let recip = 1.0 / (md + mr + C_SAT);
            rg += (md - mr).max(0.0) * recip;
            rl += (mr - md).max(0.0) * recip;
            rr += mr;
            rd += md;
        }
        gain += rg;
        loss += rl;
        sum_r += rr;
        sum_d += rd;
    }
    let n = (width * height) as f64;
    let (gr, gd) = (sum_r / n, sum_d / n);
    let grecip = 1.0 / (gd + gr + C_SAT);
    [
        (gain / n).clamp(0.0, 1.0),
        (loss / n).clamp(0.0, 1.0),
        ((gd - gr).max(0.0) * grecip).clamp(0.0, 1.0),
        ((gr - gd).max(0.0) * grecip).clamp(0.0, 1.0),
    ]
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
    texgain_out: &mut [f64],
    satsign_out: &mut [f64],
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
    let cells: Vec<(usize, usize)> = if work.band() {
        (0..n_scales * 3).map(|i| (i / 3, i % 3)).collect()
    } else {
        Vec::new()
    };
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
        if work.texgain {
            texgain_out[i * TEXGAIN_PER_CELL..(i + 1) * TEXGAIN_PER_CELL].copy_from_slice(&o.tex);
        }
    }
    if work.satsign {
        for scale in 0..n_scales {
            let (sp, w, h) = &src.scales[scale];
            let (dp, _, _) = &dst.scales[scale];
            let v = satsign_scale(&sp[0], &sp[2], &dp[0], &dp[2], *w, *h);
            satsign_out[scale * SATSIGN_PER_SCALE..(scale + 1) * SATSIGN_PER_SCALE]
                .copy_from_slice(&v);
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
            let plane64: Vec<f64> = plane.iter().map(|&v| f64::from(v)).collect();
            let mut acc = Welford::default();
            for row in plane64.chunks_exact(w) {
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
        let flat = vec![0.75f64; w * h];
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
            let mut rows: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; w]);
            for y in 0..h {
                for (k, r) in rows.iter_mut().enumerate() {
                    for (d, &v) in r.iter_mut().zip(&maps[k][y * w..(y + 1) * w]) {
                        *d = f64::from(v);
                    }
                }
                z.push_row(y, &rows);
            }
            let (nbx, nby) = (w / 5, h / 5);
            let mut want = V1BasicSums::meas_f32();
            for by in 0..nby {
                for bx in 0..nbx {
                    let m: [f64; 8] = std::array::from_fn(|k| {
                        let mut v = 0.0f64;
                        for r in by * 5..by * 5 + 5 {
                            for c in bx * 5..bx * 5 + 5 {
                                v = v.max(f64::from(maps[k][r * w + c]));
                            }
                        }
                        v
                    });
                    accumulate_block(&mut want, m, bx & 7, false);
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

/// Private qualification instrument for the SIGNEDFEAT NumPy mirror
/// (`scripts/signedfeat/numpy_mirror.py`): built only with
/// `RUSTFLAGS='--cfg restore_cuts_instrument'`, run under `ZENSIM_FORMULA_REV=4`. Dumps the XYB pyramid
/// planes of three synthetic pairs and of the 256² TRAIN pairs named by `SIGNEDFEAT_PARITY_DIR`
/// (`scripts/signedfeat/prep_parity_pairs.py`), plus the production walk's own `texgain`/`satsign`
/// values for each.
#[cfg(restore_cuts_instrument)]
#[test]
fn signedfeat_plane_dump() {
    use std::io::Write;
    let dir = std::path::PathBuf::from(std::env::var("SIGNEDFEAT_DUMP_DIR").unwrap());
    std::fs::create_dir(&dir).unwrap();
    assert_eq!(crate::ssim_form::active_revision(), FormulaRevision::Rev4);
    let mut index = std::fs::File::create(dir.join("index.tsv")).unwrap();
    writeln!(index, "case\tside\tscale\tchannel\twidth\theight\tfile").unwrap();
    let mut csv = std::fs::File::create(dir.join("features.csv")).unwrap();
    write!(csv, "case").unwrap();
    for i in 0..12 {
        write!(csv, ",texgain{i}").unwrap();
    }
    for i in 0..16 {
        write!(csv, ",satsign{i}").unwrap();
    }
    writeln!(csv).unwrap();
    type Case = (String, usize, usize, Vec<[u8; 3]>, Vec<[u8; 3]>);
    let mut cases: Vec<Case> = Vec::new();
    for &(w, h) in &[(64usize, 64usize), (97, 73), (200, 150)] {
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
                    v.wrapping_add(rnd() % 40),
                    v.wrapping_mul(2) / 2 + rnd() % 60,
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
                    p[1].saturating_add(rnd() % 30),
                    p[2] / 2 * 2,
                ]
            })
            .collect();
        cases.push((format!("syn{w}x{h}"), w, h, src, dst));
    }
    let pdir = std::env::var("SIGNEDFEAT_PARITY_DIR").expect("SIGNEDFEAT_PARITY_DIR");
    for line in std::fs::read_to_string(format!("{pdir}/index.tsv"))
        .unwrap()
        .lines()
        .skip(1)
    {
        let f: Vec<&str> = line.split('\t').collect();
        if f[3] != "256" {
            continue;
        }
        let rd = |suffix: &str| -> Vec<[u8; 3]> {
            std::fs::read(format!("{pdir}/{}_{suffix}.rgb", f[0]))
                .unwrap()
                .as_chunks::<3>()
                .0
                .to_vec()
        };
        cases.push((f[0].to_string(), 256, 256, rd("ref"), rd("dst")));
    }
    for (case, w, h, src, dst) in cases {
        let (s, d) = (
            crate::RgbSlice::new(&src, w, h),
            crate::RgbSlice::new(&dst, w, h),
        );
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
        // Values under test: the production walk with every family on.
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
            texgain: true,
            satsign: true,
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
        assert_eq!(f.len(), 1853);
        write!(csv, "{case}").unwrap();
        for v in &f[1825..] {
            write!(csv, ",{v:e}").unwrap();
        }
        writeln!(csv).unwrap();
    }
}
