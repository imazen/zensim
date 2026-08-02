//! Streaming strip-plane producer for the folded-720+append walk.
//!
//! C1 of `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md` (plan:
//! `STREAMING_FOLDAPP_PLAN_2026-07-26.md`): produce, per pair, the exact
//! per-kernel-strip plane windows the v2 walk consumes — WITHOUT ever
//! materializing a full-image f32 plane on either side.
//!
//! # Architecture (design note §3)
//!
//! - **Rolling planes:** 2 sides × 3 channels × [`crate::NUM_SCALES`]
//!   scales of row-windowed buffers ([`RollingPlane`]: linear rows,
//!   `[lo, hi)` window, memmove compaction — kernels get ordinary
//!   contiguous `&[f32]` row slices, never ring indexing).
//! - **Scale-0 fill** converts source rows in chunks through
//!   [`crate::streaming::convert_source_to_xyb_into_slices`] with
//!   `abs_row_offset` so even translucent-alpha noise-background phase
//!   matches the whole-image conversion bit-for-bit.
//! - **Pyramid fill:** scale s+1 rows come from
//!   [`crate::blur::downscale_2x_into`] over even row pairs of scale s —
//!   the same arithmetic `build_v2_ref_scales` documents as bit-identical
//!   to the materialized walk's `downscale_2x_inplace`.
//! - **Per-scale kernel-strip cursors:** scale s emits its next
//!   [`STRIP_ROWS`]-row kernel strip (the 128-row tiling of the FULL
//!   plane, rooted at row 0 — identical to
//!   `compute_channel_scale_v2_with_fold`'s loop) as soon as the rolling
//!   window covers the strip's `HALO_P`-padded wide window. Strips of a
//!   scale always emit in ascending order; scales interleave (shallowest
//!   ready first) — irrelevant to parity because every consumer
//!   accumulates per-(scale, channel).
//!
//! # Parity contract (C1 unit gate)
//!
//! For every emitted strip, [`StripPlaneProducer::fill_wide`] writes ONE
//! channel-side's wide window with bytes EQUAL to
//! [`gather_strip_halo`]`(full_plane, …)` run against the materialized
//! pyramid of the same image — including `reflect_101` rows at the true
//! top/bottom. [`StripPlaneProducer::wide_window`] additionally returns
//! the window as a zero-copy slice when no reflection is needed
//! (interior strips — the fast path the C2 walk reads in place).

#[cfg(test)]
use crate::feature_v2::gather_strip_halo;
use crate::feature_v2::{HALO_P, HdrEncoding, STRIP_ROWS, reflect_101};
use crate::source::{ImageSource, PixelFormat, SubsetView};

/// Which front-end fills the scale-0 rolling planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FrontEnd {
    /// The SDR path: `convert_source_to_xyb_into_slices` (sRGB → linear →
    /// opsin → cube-root XYB) — byte-identical to the pre-HDR producer.
    Sdr,
    /// The HDR path (HDR_PLAN chunk 2): pixel values → absolute cd/m² per
    /// the declared [`HdrEncoding`] → the UPIQ-validated PU-XYB front-end
    /// (`color::linear_to_pu_xyb_planar_into`: opsin mix → PU21
    /// banding_glare per channel ÷ PU21(100) → opponent axes, X×4).
    Hdr(HdrEncoding),
}

/// Rows appended to scale 0 per production step. A multiple of
/// `2^(NUM_SCALES-1)` keeps every deeper scale's fill cadence in whole
/// even row pairs (16 rows/step at scale 3), so downscale block parity
/// never depends on the step size.
const ADVANCE_ROWS: usize = 128;

/// One side (source or distorted) of the pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Source,
    Distorted,
}

/// A row-windowed plane buffer: rows `[lo, hi)` of a conceptual
/// `width × height` plane, stored contiguously starting at buffer row
/// `base` (compacted to 0 whenever appending would overflow capacity).
struct RollingPlane {
    buf: Vec<f32>,
    width: usize,
    /// First plane row currently retained.
    lo: usize,
    /// One past the last plane row produced.
    hi: usize,
    /// Buffer row index where plane row `lo` lives.
    base: usize,
}

impl RollingPlane {
    /// Build from a recycled buffer (see [`StripPlaneProducer::recycle`])
    /// — resized up if the pooled buffer is too small; contents are
    /// irrelevant (every row is fully written before it is read).
    fn from_pooled(mut buf: Vec<f32>, width: usize, cap_rows: usize) -> Self {
        let need = width * cap_rows;
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        Self {
            buf,
            width,
            lo: 0,
            hi: 0,
            base: 0,
        }
    }

    #[inline]
    fn cap_rows(&self) -> usize {
        self.buf.len() / self.width
    }

    /// Contiguous rows `[r0, r1)`. Panics (debug) if outside the window.
    #[inline]
    fn rows(&self, r0: usize, r1: usize) -> &[f32] {
        debug_assert!(
            self.lo <= r0 && r0 <= r1 && r1 <= self.hi,
            "rows [{r0}, {r1}) outside window [{}, {}) (base {})",
            self.lo,
            self.hi,
            self.base
        );
        let start = (self.base + (r0 - self.lo)) * self.width;
        &self.buf[start..start + (r1 - r0) * self.width]
    }

    /// One plane row as a slice.
    #[inline]
    fn row(&self, r: usize) -> &[f32] {
        self.rows(r, r + 1)
    }

    /// Make room for `n_rows` new rows at `hi` and return the writable
    /// region for them (compacting or growing first as needed). Caller
    /// fills every element of the returned slice.
    fn append_rows(&mut self, n_rows: usize) -> &mut [f32] {
        let held = self.hi - self.lo;
        if self.base + held + n_rows > self.cap_rows() {
            // Compact the live window to the buffer front.
            self.buf
                .copy_within(self.base * self.width..(self.base + held) * self.width, 0);
            self.base = 0;
            if held + n_rows > self.cap_rows() {
                // Capacity fallback — correctness-safe, should not happen
                // with the sizing in `scale_capacity_rows` (asserted by
                // the C1 tests via `max_held_rows`).
                self.buf.resize((held + n_rows) * self.width, 0.0);
            }
        }
        let start = (self.base + held) * self.width;
        self.hi += n_rows;
        &mut self.buf[start..start + n_rows * self.width]
    }

    /// Raise `lo` (retire rows below `new_lo`). No-op if already ≥.
    fn retire_below(&mut self, new_lo: usize) {
        let new_lo = new_lo.min(self.hi);
        if new_lo > self.lo {
            self.base += new_lo - self.lo;
            self.lo = new_lo;
        }
    }
}

/// Identity of one ready kernel strip.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StripInfo {
    pub scale: usize,
    /// Strip's first plane row (`ks_idx * STRIP_ROWS`).
    pub y0: usize,
    /// Rows in the strip (`STRIP_ROWS`, truncated at the plane bottom).
    pub strip_h: usize,
    /// Plane dims at this scale.
    pub plane_w: usize,
    pub plane_h: usize,
}

impl StripInfo {
    /// Buffer height of the strip's `HALO_P`-padded wide window.
    #[inline]
    pub fn wide_h(&self) -> usize {
        self.strip_h + 2 * HALO_P
    }

    /// True when the wide window needs no boundary reflection — i.e.
    /// `[y0 − HALO_P, y0 + strip_h + HALO_P)` lies fully inside the
    /// plane, so it exists as contiguous real rows.
    #[inline]
    pub fn interior(&self) -> bool {
        self.y0 >= HALO_P && self.y0 + self.strip_h + HALO_P <= self.plane_h
    }
}

/// Per-scale production/emission cursors.
struct ScaleState {
    plane_w: usize,
    plane_h: usize,
    /// Next kernel strip to emit (index into the 128-row tiling).
    next_ks: usize,
}

/// The streaming producer: rolling XYB pyramids for both sides of a pair
/// plus the per-scale kernel-strip emission cursors.
pub(crate) struct StripPlaneProducer<'a, S: ImageSource, D: ImageSource> {
    source: &'a S,
    distorted: &'a D,
    /// `[side][channel][scale]` rolling planes; side 0 = source.
    planes: [[Vec<RollingPlane>; 3]; 2],
    scales: Vec<ScaleState>,
    parallel: bool,
    front_end: FrontEnd,
    /// Row scratch for the HDR front-end (decode + PU conversion input).
    hdr_row: Vec<[f32; 3]>,
    /// Peak `hi − lo` observed per scale (capacity-budget telemetry for
    /// the C1 tests; also what G-RAM's O(width) claim rests on).
    max_held_rows: Vec<usize>,
}

/// Capacity (in rows) budgeted for one scale's rolling window: one
/// kernel strip + both halos + one production step + slack for the
/// downscale-consumption floor. Exceeding it merely grows the buffer
/// (correctness-safe); the C1 tests assert it is never exceeded.
fn scale_capacity_rows(scale: usize, plane_h: usize) -> usize {
    let step = (ADVANCE_ROWS >> scale).max(2);
    (STRIP_ROWS + 2 * HALO_P + step + 32).min(plane_h.max(1) + 2 * HALO_P)
}

impl<'a, S: ImageSource, D: ImageSource> StripPlaneProducer<'a, S, D> {
    /// Build a producer for a dimension-matched pair. Caller has already
    /// applied entry-point validation (dims match, ≥ MIN_PYRAMID_DIM —
    /// sub-64 inputs are reflect-padded BEFORE constructing, exactly like
    /// the materialized entries).
    /// `pool` recycles rolling-plane buffers across pairs (the caller's
    /// [`crate::feature_v2::V2Scratch`] owns it): construction drains it,
    /// [`Self::recycle`] refills it — steady-state batch extraction
    /// performs zero producer allocation (measured ~1.7k page
    /// faults/pair from fresh per-pair Vecs before this).
    #[cfg_attr(not(test), allow(dead_code))] // production sites pass FrontEnd explicitly (new_with_front_end)
    pub(crate) fn new(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
    ) -> Self {
        Self::new_with_front_end(source, distorted, parallel, pool, FrontEnd::Sdr)
    }

    pub(crate) fn new_with_front_end(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
        front_end: FrontEnd,
    ) -> Self {
        let (w0, h0) = (source.width(), source.height());
        debug_assert_eq!(w0, distorted.width());
        debug_assert_eq!(h0, distorted.height());

        let mut scales = Vec::with_capacity(crate::NUM_SCALES);
        let (mut w, mut h) = (w0, h0);
        for _ in 0..crate::NUM_SCALES {
            scales.push(ScaleState {
                plane_w: w,
                plane_h: h,
                next_ks: 0,
            });
            w /= 2;
            h /= 2;
        }

        let planes = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                scales
                    .iter()
                    .enumerate()
                    .map(|(s, st)| {
                        RollingPlane::from_pooled(
                            pool.pop().unwrap_or_default(),
                            st.plane_w,
                            scale_capacity_rows(s, st.plane_h),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        });

        Self {
            source,
            distorted,
            planes,
            scales,
            parallel,
            front_end,
            hdr_row: match front_end {
                FrontEnd::Sdr => Vec::new(),
                FrontEnd::Hdr(_) => vec![[0.0f32; 3]; w0],
            },
            max_held_rows: vec![0; crate::NUM_SCALES],
        }
    }

    #[inline]
    fn plane(&self, side: Side, ch: usize, scale: usize) -> &RollingPlane {
        let si = match side {
            Side::Source => 0,
            Side::Distorted => 1,
        };
        &self.planes[si][ch][scale]
    }

    /// Emit the next ready kernel strip (shallowest ready scale first;
    /// strictly ascending strip order within a scale), producing more
    /// pyramid rows as needed. Returns `None` when every kernel strip of
    /// every scale has been emitted. The returned windows (via
    /// [`Self::wide_window`] / [`Self::fill_wide`] / [`Self::rows`]) stay
    /// valid until the NEXT `next_strip` call.
    pub(crate) fn next_strip(&mut self) -> Option<StripInfo> {
        // Retire rows consumed by everything emitted so far, then look
        // for a ready strip; if none, produce more rows and repeat.
        loop {
            self.retire();
            if let Some(info) = self.find_ready() {
                self.scales[info.scale].next_ks += 1;
                return Some(info);
            }
            if !self.produce() {
                debug_assert!(
                    self.scales
                        .iter()
                        .all(|st| { st.next_ks * STRIP_ROWS >= st.plane_h }),
                    "production exhausted with unemitted strips"
                );
                return None;
            }
        }
    }

    /// Shallowest scale with a ready, unemitted kernel strip.
    fn find_ready(&self) -> Option<StripInfo> {
        for (scale, st) in self.scales.iter().enumerate() {
            let y0 = st.next_ks * STRIP_ROWS;
            if y0 >= st.plane_h {
                continue; // scale fully emitted
            }
            let strip_h = STRIP_ROWS.min(st.plane_h - y0);
            let need_hi = (y0 + strip_h + HALO_P).min(st.plane_h);
            let hi = self.plane(Side::Source, 0, scale).hi;
            if hi >= need_hi {
                return Some(StripInfo {
                    scale,
                    y0,
                    strip_h,
                    plane_w: st.plane_w,
                    plane_h: st.plane_h,
                });
            }
        }
        None
    }

    /// Advance production: convert up to [`ADVANCE_ROWS`] new scale-0
    /// rows on both sides and cascade downscales. Returns false when
    /// scale 0 was already complete (nothing produced).
    fn produce(&mut self) -> bool {
        let h0 = self.scales[0].plane_h;
        let hi0 = self.planes[0][0][0].hi;
        if hi0 >= h0 {
            return false;
        }
        let n_new = ADVANCE_ROWS.min(h0 - hi0);

        // --- Scale-0 conversion, both sides. ---
        let front_end = self.front_end;
        for si in 0..2 {
            let width = self.scales[0].plane_w;
            let [c0, c1, c2] = &mut self.planes[si];
            let (p0, p1, p2) = (
                c0[0].append_rows(n_new),
                c1[0].append_rows(n_new),
                c2[0].append_rows(n_new),
            );
            match front_end {
                FrontEnd::Sdr => match si {
                    0 => {
                        let sub = SubsetView::new(self.source, hi0, n_new);
                        crate::streaming::convert_source_to_xyb_into_slices(
                            &sub,
                            p0,
                            p1,
                            p2,
                            width,
                            self.parallel,
                            hi0,
                        );
                    }
                    _ => {
                        let sub = SubsetView::new(self.distorted, hi0, n_new);
                        crate::streaming::convert_source_to_xyb_into_slices(
                            &sub,
                            p0,
                            p1,
                            p2,
                            width,
                            self.parallel,
                            hi0,
                        );
                    }
                },
                FrontEnd::Hdr(encoding) => {
                    for k in 0..n_new {
                        let y = hi0 + k;
                        let row_off = k * width;
                        if si == 0 {
                            hdr_source_row_to_nits(self.source, y, encoding, &mut self.hdr_row);
                        } else {
                            hdr_source_row_to_nits(self.distorted, y, encoding, &mut self.hdr_row);
                        }
                        crate::color::linear_to_pu_xyb_planar_into(
                            &self.hdr_row[..width],
                            &mut p0[row_off..row_off + width],
                            &mut p1[row_off..row_off + width],
                            &mut p2[row_off..row_off + width],
                        );
                    }
                }
            }
        }

        // --- Cascade downscales: fill each deeper scale from complete
        //     even row pairs of its parent. ---
        for scale in 1..crate::NUM_SCALES {
            let (plane_w, plane_h) = (self.scales[scale].plane_w, self.scales[scale].plane_h);
            let parent_w = self.scales[scale - 1].plane_w;
            for si in 0..2 {
                for ch in 0..3 {
                    let parent_hi = self.planes[si][ch][scale - 1].hi;
                    let child_hi = self.planes[si][ch][scale].hi;
                    let target = (parent_hi / 2).min(plane_h);
                    if target <= child_hi {
                        continue;
                    }
                    let n_out = target - child_hi;
                    // Split borrow: parent (read) vs child (write).
                    let (head, tail) = self.planes[si][ch].split_at_mut(scale);
                    let parent = &head[scale - 1];
                    let child = &mut tail[0];
                    let src_rows = parent.rows(child_hi * 2, child_hi * 2 + n_out * 2);
                    let dst = child.append_rows(n_out);
                    crate::blur::downscale_2x_into(src_rows, parent_w, dst, plane_w, n_out);
                }
            }
        }

        for (scale, held) in self.max_held_rows.iter_mut().enumerate() {
            let p = &self.planes[0][0][scale];
            *held = (*held).max(p.hi - p.lo);
        }
        true
    }

    /// Retire rows no longer needed by (a) the next unemitted kernel
    /// strip's wide window at each scale, or (b) the parent rows the next
    /// downscale production step still has to read.
    fn retire(&mut self) {
        for scale in 0..crate::NUM_SCALES {
            let st = &self.scales[scale];
            // (a) Wide-window floor for the next strip to emit. The top
            //     reflection only ever reads rows ≥ 0 that a top strip
            //     still holds (lo starts at 0 and this floor is 0 until
            //     the first strip retires).
            let ks_floor = if st.next_ks * STRIP_ROWS >= st.plane_h {
                st.plane_h // scale fully emitted — no kernel need
            } else {
                (st.next_ks * STRIP_ROWS).saturating_sub(HALO_P)
            };
            // (b) Downscale floor: the child's next output row `j` reads
            //     parent rows [2j, 2j+2).
            let ds_floor = if scale + 1 < crate::NUM_SCALES {
                let child_hi = self.planes[0][0][scale + 1].hi;
                let child_h = self.scales[scale + 1].plane_h;
                if child_hi >= child_h {
                    st.plane_h
                } else {
                    child_hi * 2
                }
            } else {
                st.plane_h
            };
            let floor = ks_floor.min(ds_floor);
            for si in 0..2 {
                for ch in 0..3 {
                    self.planes[si][ch][scale].retire_below(floor);
                }
            }
        }
    }

    /// Zero-copy wide window for an INTERIOR strip (no reflection):
    /// contiguous rows `[y0 − HALO_P, y0 + strip_h + HALO_P)`. `None`
    /// when the strip touches the true top/bottom — use
    /// [`Self::fill_wide`] there.
    pub(crate) fn wide_window(&self, side: Side, ch: usize, info: &StripInfo) -> Option<&[f32]> {
        if !info.interior() {
            return None;
        }
        Some(
            self.plane(side, ch, info.scale)
                .rows(info.y0 - HALO_P, info.y0 + info.strip_h + HALO_P),
        )
    }

    /// Materialize the strip's wide window into `dst`
    /// (`plane_w × wide_h()` elements) with [`gather_strip_halo`]'s exact
    /// semantics: buffer row `i` = plane row
    /// `reflect_101(y0 − HALO_P + i, plane_h)`.
    pub(crate) fn fill_wide(&self, side: Side, ch: usize, info: &StripInfo, dst: &mut [f32]) {
        let w = info.plane_w;
        let wide_h = info.wide_h();
        debug_assert!(dst.len() >= w * wide_h);
        let plane = self.plane(side, ch, info.scale);
        for i in 0..wide_h {
            let gy = info.y0 as isize - HALO_P as isize + i as isize;
            let gy_r = reflect_101(gy, info.plane_h);
            dst[i * w..(i + 1) * w].copy_from_slice(plane.row(gy_r));
        }
    }

    /// Contiguous plane rows `[r0, r1)` at a scale (for the strip-fed
    /// blockiness/gradient consumers — rows must lie inside the current
    /// window, which the emission/retire order guarantees for
    /// `[y0 − 1, y0 + strip_h + 1)` of the just-emitted strip).
    pub(crate) fn rows(&self, side: Side, ch: usize, scale: usize, r0: usize, r1: usize) -> &[f32] {
        self.plane(side, ch, scale).rows(r0, r1)
    }

    /// Peak rows held per scale (test telemetry).
    #[cfg(test)]
    pub(crate) fn max_held_rows(&self) -> &[usize] {
        &self.max_held_rows
    }

    /// Return every rolling-plane buffer to `pool` for the next pair.
    pub(crate) fn recycle(self, pool: &mut Vec<Vec<f32>>) {
        for side in self.planes {
            for ch in side {
                for plane in ch {
                    pool.push(plane.buf);
                }
            }
        }
    }
}

/// Read one source row as absolute-linear cd/m² RGB triples per the
/// declared [`HdrEncoding`]. Accepted pixel formats (validated at the HDR
/// entry): `LinearF32Rgba` (f32 values used directly — cd/m² for
/// `Linear`, code values for `Pq`/`Hlg`; alpha ignored, Opaque required)
/// and `Srgb16Rgba` (u16 code values normalized by 65535 — `Pq`/`Hlg`
/// code-value containers like cICP-spliced 16-bit PNG).
fn hdr_source_row_to_nits(
    src: &impl ImageSource,
    y: usize,
    encoding: HdrEncoding,
    out: &mut [[f32; 3]],
) {
    let width = src.width();
    let row_bytes = src.row_bytes(y);
    match src.pixel_format() {
        PixelFormat::LinearF32Rgba => {
            let px: &[[f32; 4]] = bytemuck::cast_slice(row_bytes);
            for (o, p) in out[..width].iter_mut().zip(&px[..width]) {
                *o = [p[0], p[1], p[2]];
            }
        }
        PixelFormat::Srgb16Rgba => {
            const INV: f32 = 1.0 / 65535.0;
            for (x, o) in out[..width].iter_mut().enumerate() {
                let off = x * 8;
                let r = u16::from_ne_bytes([row_bytes[off], row_bytes[off + 1]]);
                let g = u16::from_ne_bytes([row_bytes[off + 2], row_bytes[off + 3]]);
                let b = u16::from_ne_bytes([row_bytes[off + 4], row_bytes[off + 5]]);
                *o = [r as f32 * INV, g as f32 * INV, b as f32 * INV];
            }
        }
        other => unreachable!(
            "HDR route accepts LinearF32Rgba/Srgb16Rgba only (validated at the entry); got {other:?}"
        ),
    }
    match encoding {
        HdrEncoding::Linear => {}
        HdrEncoding::Pq { peak_nits } => {
            crate::transfer::decode_pq_row(&mut out[..width], peak_nits)
        }
        HdrEncoding::Hlg {
            peak_nits,
            ambient_lux,
        } => crate::transfer::decode_hlg_row(&mut out[..width], peak_nits, ambient_lux),
    }
}

// ============================================================================
// C1 unit gate: producer windows byte-equal the materialized-path planes.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::RgbSlice;

    /// Deterministic textured content (same family as feature_v2's tests).
    fn textured_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut px = vec![[0u8; 3]; w * h];
        let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for y in 0..h {
            for x in 0..w {
                let n = next();
                let gx = ((x * 5) & 0x7F) as u8;
                let gy = ((y * 3) & 0x7F) as u8;
                px[y * w + x] = [
                    gx.wrapping_add((n & 0x3F) as u8),
                    gy.wrapping_add(((n >> 8) & 0x3F) as u8),
                    ((x ^ y) & 0xFF) as u8 ^ ((n >> 16) & 0x1F) as u8,
                ];
            }
        }
        px
    }

    fn distort(src: &[[u8; 3]]) -> Vec<[u8; 3]> {
        src.iter()
            .map(|&[r, g, b]| [(r & !7) | 4, g.saturating_add(3), b & !3])
            .collect()
    }

    /// Materialized pyramid for one side, built EXACTLY like the
    /// prepared-reference path (`build_v2_ref_scales`: whole-plane XYB +
    /// `downscale_2x_into` per level). The materialized walk's distorted
    /// side uses `downscale_2x_inplace` instead — documented (and
    /// asserted here) bit-identical arithmetic.
    fn materialize(img: &impl ImageSource) -> Vec<([Vec<f32>; 3], usize, usize)> {
        let mut width = img.width();
        let mut height = img.height();
        let mut scales: Vec<([Vec<f32>; 3], usize, usize)> = Vec::new();
        scales.push((
            crate::streaming::convert_source_to_xyb(img, width, false),
            width,
            height,
        ));
        for _ in 1..crate::NUM_SCALES {
            let new_w = width / 2;
            let new_h = height / 2;
            let mut next: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; new_w * new_h]);
            {
                let (prev, _, _) = scales.last().unwrap();
                for ch in 0..3 {
                    crate::blur::downscale_2x_into(&prev[ch], width, &mut next[ch], new_w, new_h);
                }
            }
            scales.push((next, new_w, new_h));
            width = new_w;
            height = new_h;
        }
        scales
    }

    /// In-place downscale (the materialized walk's distorted-side
    /// operator) produces the same bytes as `downscale_2x_into`.
    #[test]
    fn downscale_into_matches_inplace() {
        let (w, h) = (127, 93);
        let src = textured_image(w, h, 3);
        let img = RgbSlice::new(&src, w, h);
        let mut planes = crate::streaming::convert_source_to_xyb(&img, w, false);
        let reference = materialize(&img);
        let (mut cw, mut ch_) = (w, h);
        #[allow(clippy::needless_range_loop)] // scale drives geometry AND indexes reference/planes
        for scale in 1..crate::NUM_SCALES {
            for p in planes.iter_mut() {
                crate::blur::downscale_2x_inplace(p, cw, ch_);
            }
            cw /= 2;
            ch_ /= 2;
            let (ref_planes, rw, rh) = &reference[scale];
            assert_eq!((cw, ch_), (*rw, *rh));
            for c in 0..3 {
                assert_eq!(planes[c].len(), ref_planes[c].len());
                for (i, (a, b)) in planes[c].iter().zip(ref_planes[c].iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "scale {scale} ch {c} idx {i}");
                }
            }
        }
    }

    /// THE C1 gate: every emitted strip's wide window — both sides, all
    /// 3 channels — is byte-equal to `gather_strip_halo` run against the
    /// materialized planes; interior strips' zero-copy windows equal the
    /// filled ones; every kernel strip of every scale is emitted exactly
    /// once, ascending per scale; capacity budget holds.
    #[test]
    fn producer_windows_byte_equal_materialized() {
        // Odd dims, sub-one-strip, multi-strip, tall-thin, and >2-strip
        // heights; all ≥ MIN_PYRAMID_DIM (sub-64 pads before the
        // producer, covered by entry-level tests).
        let cases = [
            (96usize, 64usize),
            (127, 93),
            (208, 144),
            (200, 150),
            (96, 1030),
            (320, 517),
        ];
        for (w, h) in cases {
            let src = textured_image(w, h, 11);
            let dst = distort(&src);
            let s_img = RgbSlice::new(&src, w, h);
            let d_img = RgbSlice::new(&dst, w, h);
            let mat_s = materialize(&s_img);
            let mat_d = materialize(&d_img);

            let mut pool = Vec::new();
            let mut prod = StripPlaneProducer::new(&s_img, &d_img, false, &mut pool);
            let mut emitted: Vec<Vec<usize>> = vec![Vec::new(); crate::NUM_SCALES];
            let mut wide = Vec::new();
            let mut wide_ref = Vec::new();
            while let Some(info) = prod.next_strip() {
                emitted[info.scale].push(info.y0);
                let n_wide = info.plane_w * info.wide_h();
                wide.resize(n_wide, 0.0);
                wide_ref.resize(n_wide, 0.0);
                for (side, mat) in [(Side::Source, &mat_s), (Side::Distorted, &mat_d)] {
                    let (planes, pw, ph) = &mat[info.scale];
                    assert_eq!((*pw, *ph), (info.plane_w, info.plane_h));
                    #[allow(clippy::needless_range_loop)] // ch feeds fill_wide AND indexes planes
                    for ch in 0..3 {
                        prod.fill_wide(side, ch, &info, &mut wide);
                        gather_strip_halo(
                            &planes[ch],
                            *pw,
                            *ph,
                            info.y0,
                            info.wide_h(),
                            HALO_P,
                            &mut wide_ref,
                        );
                        for i in 0..n_wide {
                            assert_eq!(
                                wide[i].to_bits(),
                                wide_ref[i].to_bits(),
                                "{w}x{h} scale {} strip y0={} side {side:?} ch {ch} idx {i}",
                                info.scale,
                                info.y0
                            );
                        }
                        if let Some(win) = prod.wide_window(side, ch, &info) {
                            assert!(info.interior());
                            for i in 0..n_wide {
                                assert_eq!(
                                    win[i].to_bits(),
                                    wide[i].to_bits(),
                                    "in-place window mismatch at idx {i}"
                                );
                            }
                        } else {
                            assert!(!info.interior());
                        }
                    }
                }
            }

            // Full, ordered coverage of every scale's strip tiling.
            let (mut pw, mut ph) = (w, h);
            #[allow(clippy::needless_range_loop)] // scale drives tiling AND indexes emitted
            for scale in 0..crate::NUM_SCALES {
                let expect: Vec<usize> = (0..ph.div_ceil(STRIP_ROWS))
                    .map(|k| k * STRIP_ROWS)
                    .collect();
                assert_eq!(
                    emitted[scale], expect,
                    "{w}x{h} scale {scale} emission order/coverage"
                );
                pw /= 2;
                ph /= 2;
            }
            let _ = pw;

            // Capacity budget: rolling windows never outgrew their
            // construction-time budget (no fallback growth).
            for (scale, &held) in prod.max_held_rows().iter().enumerate() {
                let budget = scale_capacity_rows(scale, if scale == 0 { h } else { h >> scale });
                assert!(
                    held <= budget,
                    "{w}x{h} scale {scale}: held {held} rows > budget {budget}"
                );
            }
        }
    }

    /// Translucent RGBA: the producer's strip conversion must keep the
    /// alpha noise-background in the PARENT image's row phase
    /// (`abs_row_offset`), matching whole-image conversion bit-for-bit.
    #[test]
    fn producer_alpha_phase_matches_whole_image() {
        let (w, h) = (96, 517);
        let base = textured_image(w, h, 29);
        let rgba: Vec<[u8; 4]> = base
            .iter()
            .enumerate()
            .map(|(i, &[r, g, b])| [r, g, b, (((i * 37) % 256) as u8)])
            .collect();
        let img = crate::source::RgbaSlice::new(&rgba, w, h);
        let mat = materialize(&img);

        let mut pool = Vec::new();
        let mut prod = StripPlaneProducer::new(&img, &img, false, &mut pool);
        let mut wide = Vec::new();
        let mut wide_ref = Vec::new();
        while let Some(info) = prod.next_strip() {
            let n_wide = info.plane_w * info.wide_h();
            wide.resize(n_wide, 0.0);
            wide_ref.resize(n_wide, 0.0);
            let (planes, pw, ph) = &mat[info.scale];
            #[allow(clippy::needless_range_loop)] // ch feeds fill_wide AND indexes planes
            for ch in 0..3 {
                prod.fill_wide(Side::Source, ch, &info, &mut wide);
                gather_strip_halo(
                    &planes[ch],
                    *pw,
                    *ph,
                    info.y0,
                    info.wide_h(),
                    HALO_P,
                    &mut wide_ref,
                );
                for i in 0..n_wide {
                    assert_eq!(
                        wide[i].to_bits(),
                        wide_ref[i].to_bits(),
                        "alpha-phase mismatch: scale {} y0 {} ch {ch} idx {i}",
                        info.scale,
                        info.y0
                    );
                }
            }
        }
    }
}
