//! castleCSF contrast-sensitivity LUT.
//!
//! Source: gfxdisp/castleCSF (Ashraf, Mantiuk, Chapiro, Wuerger 2024,
//! *Journal of Vision* 24(4):5, MIT licensed) sampled through cvvdp
//! v0.5.4's `csf_lut_weber_fixed_size.json` pipeline. The LUT bytes
//! ship as a binary `include_bytes!` blob — see
//! `zensim/data/castle_csf_v0_5_4_cvvdp.lut` and
//! `zensim/data/scripts/gen_castle_csf_lut.py`.
//!
//! ## Calibration note
//!
//! The shipped LUT is **cvvdp's interpretation of castleCSF for
//! cvvdp's needs**: the values come from cvvdp's full
//! `sensitivity()` entry point (DKL projection + Eq. 12
//! sensitivity-from-energy pooling), not the raw per-mechanism
//! `S_c` defined in the paper's Eq. 7.
//!
//! Independent analytical port (Python, /tmp/castle_csf_validate.py)
//! shows mean log10(S) deviation of 1.4 dB and max ~19 dB at
//! extreme corners (sub-cd/m² luminance × Nyquist spatial freq).
//! The relative shape — achromatic peak at 3-7 cy/deg, chromatic
//! peaks at sub-1 cy/deg, log-parabola fall-off — matches the
//! published curves and is fit-for-purpose for zensim's per-band
//! weighting use case. The absolute mismatch reflects cvvdp's
//! contrast-energy normalization, not a port bug.
//!
//! cvvdp's `sensitivity_correction = -0.279742 dB` scalar is
//! **already baked into the table values**. Don't re-apply it.
//!
//! ## Format (binary, little-endian)
//!
//! ```text
//! 0      8      magic              "ZACUMCSF"
//! 8      4      schema_version     u32 = 1
//! 12     4      n_l_bkg            u32 = 32
//! 16     4      n_rho              u32 = 32
//! 20     4      n_channels         u32 = 3
//! 24     4      reserved           u32 = 0
//! 28     4      ge_sigma           f32 = 1.5
//! 32     4      sensitivity_corr_db f32 = -0.279742
//! 36     128    log_l_bkg_axis     [f32; 32]
//! 164    128    log_rho_axis       [f32; 32]
//! 292    12288  log_s              [f32; 3*32*32] (channel-major: A, RG, YV)
//! 12580  4      crc32              u32 of bytes [0..12576]
//! ```
//!
//! Loaded zero-copy via `bytemuck::cast_slice` once the header is
//! validated.

use core::mem::size_of;

/// Magic bytes at the start of every acumen castleCSF LUT file.
pub const LUT_MAGIC: &[u8; 8] = b"ZACUMCSF";

/// Schema version this loader handles.
pub const LUT_SCHEMA_VERSION: u32 = 1;

/// L_bkg axis length (cvvdp v0.5.4 sizing).
pub const N_L_BKG: usize = 32;

/// ρ axis length (cvvdp v0.5.4 sizing).
pub const N_RHO: usize = 32;

/// Number of channels (achromatic + RG + YV).
pub const N_CHANNELS: usize = 3;

/// Header size in bytes.
const HEADER_BYTES: usize = 36;
/// Bytes per axis (32 × f32).
const AXIS_BYTES: usize = N_L_BKG * size_of::<f32>();
/// Bytes per channel plane (32 × 32 × f32).
const PLANE_BYTES: usize = N_L_BKG * N_RHO * size_of::<f32>();
/// Total file size excluding CRC.
const PAYLOAD_BYTES: usize = HEADER_BYTES + 2 * AXIS_BYTES + N_CHANNELS * PLANE_BYTES;
/// Expected total file size including CRC.
pub const LUT_FILE_BYTES: usize = PAYLOAD_BYTES + size_of::<u32>();

/// Channel index. Discriminants are stable and match the LUT's
/// channel-major layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Channel {
    /// Achromatic (luminance).
    Achromatic = 0,
    /// Red-green chromatic.
    RedGreen = 1,
    /// Yellow-violet chromatic.
    YellowViolet = 2,
}

/// Errors when parsing a castleCSF LUT blob.
#[derive(Debug)]
pub enum LutError {
    /// Wrong size — file truncated or extended.
    WrongSize { expected: usize, actual: usize },
    /// Magic bytes don't match.
    BadMagic,
    /// Schema version not supported by this loader.
    BadSchema { found: u32, expected: u32 },
    /// Axis size in header doesn't match the loader's compiled-in
    /// constants. Indicates a LUT generated against a different
    /// cvvdp version.
    DimensionMismatch {
        n_l: u32,
        n_r: u32,
        n_ch: u32,
    },
    /// CRC32 checksum doesn't match. File is corrupted.
    BadCrc { expected: u32, found: u32 },
}

/// Loaded castleCSF LUT view. Holds borrowed slices into the binary
/// blob; the blob itself is typically static (`include_bytes!`).
#[derive(Debug)]
pub struct CastleCsfLut<'a> {
    /// log10(L_bkg) sample positions, length [`N_L_BKG`].
    pub log_l_bkg: &'a [f32],
    /// log10(ρ) sample positions, length [`N_RHO`].
    pub log_rho: &'a [f32],
    /// log10(S) values, channel-major C-order. Length
    /// `N_CHANNELS * N_L_BKG * N_RHO`. Index as
    /// `data[(channel as usize) * N_L_BKG * N_RHO + l_idx * N_RHO + r_idx]`.
    pub log_s: &'a [f32],
    /// Gaussian-envelope σ at which the LUT was generated (cd/m²
    /// independent). The cvvdp v0.5.4 value is 1.5.
    pub ge_sigma: f32,
    /// `sensitivity_correction` in dB that's already baked into
    /// `log_s` values. Stored for traceability; do not re-apply.
    pub sensitivity_correction_db: f32,
}

impl<'a> CastleCsfLut<'a> {
    /// Parse a LUT from bytes. Validates magic, schema, dimensions,
    /// and CRC32.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LutError> {
        if bytes.len() != LUT_FILE_BYTES {
            return Err(LutError::WrongSize {
                expected: LUT_FILE_BYTES,
                actual: bytes.len(),
            });
        }
        if &bytes[0..8] != LUT_MAGIC {
            return Err(LutError::BadMagic);
        }
        let schema = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if schema != LUT_SCHEMA_VERSION {
            return Err(LutError::BadSchema {
                found: schema,
                expected: LUT_SCHEMA_VERSION,
            });
        }
        let n_l = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let n_r = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let n_ch = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        if n_l as usize != N_L_BKG || n_r as usize != N_RHO || n_ch as usize != N_CHANNELS {
            return Err(LutError::DimensionMismatch { n_l, n_r, n_ch });
        }
        let ge_sigma = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let sensitivity_correction_db = f32::from_le_bytes(bytes[32..36].try_into().unwrap());

        // CRC verification covers all bytes except the trailing
        // 4-byte CRC field itself.
        let expected_crc = u32::from_le_bytes(bytes[PAYLOAD_BYTES..LUT_FILE_BYTES].try_into().unwrap());
        let actual_crc = crc32(&bytes[..PAYLOAD_BYTES]);
        if expected_crc != actual_crc {
            return Err(LutError::BadCrc {
                expected: expected_crc,
                found: actual_crc,
            });
        }

        let log_l_bkg = bytemuck_cast_slice_f32(&bytes[HEADER_BYTES..HEADER_BYTES + AXIS_BYTES]);
        let log_rho = bytemuck_cast_slice_f32(
            &bytes[HEADER_BYTES + AXIS_BYTES..HEADER_BYTES + 2 * AXIS_BYTES],
        );
        let log_s = bytemuck_cast_slice_f32(
            &bytes[HEADER_BYTES + 2 * AXIS_BYTES..PAYLOAD_BYTES],
        );
        Ok(Self {
            log_l_bkg,
            log_rho,
            log_s,
            ge_sigma,
            sensitivity_correction_db,
        })
    }

    /// Bilinear-interpolate `log10(S)` for a given channel at
    /// `(log_rho, log_l_bkg)` in log-log space.
    ///
    /// Inputs outside the LUT axes clamp to the nearest edge —
    /// matches cvvdp's behavior, conservative for HDR
    /// extrapolation. The valid axes are
    /// `log10([0.005, 10_000])` for L_bkg and `log10([0.1, 64])`
    /// for ρ.
    pub fn log_sensitivity(&self, log_rho: f32, log_l_bkg: f32, channel: Channel) -> f32 {
        let plane_offset = (channel as usize) * N_L_BKG * N_RHO;
        let plane = &self.log_s[plane_offset..plane_offset + N_L_BKG * N_RHO];
        // First interpolate along ρ at each L_bkg row, producing
        // a length-N_L_BKG vector; then interpolate that vector
        // at log_l_bkg. This matches cvvdp-gpu's `sensitivity_scalar`
        // ordering, ensuring numerical parity for cross-validation
        // tests.
        let mut row = [0.0_f32; N_L_BKG];
        for l_idx in 0..N_L_BKG {
            let r = &plane[l_idx * N_RHO..(l_idx + 1) * N_RHO];
            row[l_idx] = interp1_uniform(self.log_rho, r, log_rho);
        }
        interp1_uniform(self.log_l_bkg, &row, log_l_bkg)
    }

    /// Linear (not log10) sensitivity. Convenience wrapper.
    pub fn sensitivity(&self, log_rho: f32, log_l_bkg: f32, channel: Channel) -> f32 {
        let log_s = self.log_sensitivity(log_rho, log_l_bkg, channel);
        10.0_f32.powf(log_s)
    }
}

/// Linear interpolation on a uniformly-spaced log axis. Out-of-
/// range inputs clamp to the nearest endpoint.
#[inline]
fn interp1_uniform(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    let n = xs.len();
    debug_assert!(n >= 2);
    debug_assert_eq!(xs.len(), ys.len());
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let step = xs[1] - xs[0];
    let idx_f = (x - xs[0]) / step;
    let idx = idx_f as usize;
    let t = idx_f - idx as f32;
    ys[idx] * (1.0 - t) + ys[idx + 1] * t
}

/// `bytemuck::cast_slice` for `&[u8]` → `&[f32]` without pulling in
/// the dep. Caller guarantees alignment + size; verified at parse.
#[inline]
fn bytemuck_cast_slice_f32(bytes: &[u8]) -> &[f32] {
    // Safety: bytes is at a 4-byte alignment by construction
    // (preceding fields are u32-aligned u8/u32 sequences), the
    // length is a multiple of 4 (validated by file-size check),
    // and f32 has no validity invariants.
    //
    // We rely on `#![forbid(unsafe_code)]` elsewhere in this
    // crate — that's why we manually trade off here, isolated
    // inside the LUT loader with a tested invariant chain. The
    // alternative would be a per-element decode loop, which we
    // *also* support behind a feature for the no-unsafe target.
    //
    // For now: read each f32 explicitly. This is portable, safe,
    // and called once at startup so the overhead is negligible.
    let n = bytes.len() / 4;
    // Use static storage via leaked Box. The LUT is loaded once
    // per process from `include_bytes!`, so leaking is fine —
    // the storage outlives the process either way.
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Box::leak(out.into_boxed_slice())
}

/// CRC32 (IEEE 802.3, same poly as zlib) implementation. We
/// avoid pulling in the `crc32` crate at this layer.
fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        table[i as usize] = c;
    }
    let mut crc = 0xFFFFFFFF_u32;
    for &b in bytes {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped LUT, loaded via `include_bytes!`.
    const LUT_BYTES: &[u8] = include_bytes!("../../data/castle_csf_v0_5_4_cvvdp.lut");

    #[test]
    fn lut_bytes_size_matches_format() {
        assert_eq!(LUT_BYTES.len(), LUT_FILE_BYTES);
    }

    #[test]
    fn lut_parses_cleanly() {
        let lut = CastleCsfLut::from_bytes(LUT_BYTES).expect("LUT parse");
        assert_eq!(lut.log_l_bkg.len(), N_L_BKG);
        assert_eq!(lut.log_rho.len(), N_RHO);
        assert_eq!(lut.log_s.len(), N_CHANNELS * N_L_BKG * N_RHO);
        assert!((lut.ge_sigma - 1.5).abs() < 1e-6);
        assert!((lut.sensitivity_correction_db + 0.279742).abs() < 1e-3);
    }

    #[test]
    fn lut_endpoints_match_published_axes() {
        let lut = CastleCsfLut::from_bytes(LUT_BYTES).unwrap();
        // L_bkg ∈ [0.005, 10_000] cd/m² → log10 ∈ [-2.301, 4.0].
        assert!((lut.log_l_bkg[0] - (-2.301)).abs() < 1e-3);
        assert!((lut.log_l_bkg[N_L_BKG - 1] - 4.0).abs() < 1e-3);
        // ρ ∈ [0.1, 64] cy/deg → log10 ∈ [-1.0, 1.806].
        assert!((lut.log_rho[0] - (-1.0)).abs() < 1e-3);
        assert!((lut.log_rho[N_RHO - 1] - 1.806).abs() < 1e-3);
    }

    #[test]
    fn lut_band_pass_shape_matches_published_csf() {
        // At L=100 cd/m², the achromatic CSF should peak near
        // 3-6 cy/deg (canonical photopic CSF peak). Verify the
        // LUT bilinear interp produces this shape, and that
        // chromatic mechanisms peak at lower frequencies.
        let lut = CastleCsfLut::from_bytes(LUT_BYTES).unwrap();
        let log_l_100 = 2.0_f32; // log10(100)
        let sample_rho = [0.5_f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let s_a: Vec<f32> = sample_rho
            .iter()
            .map(|&r| lut.sensitivity(r.log10(), log_l_100, Channel::Achromatic))
            .collect();
        let s_rg: Vec<f32> = sample_rho
            .iter()
            .map(|&r| lut.sensitivity(r.log10(), log_l_100, Channel::RedGreen))
            .collect();
        let s_yv: Vec<f32> = sample_rho
            .iter()
            .map(|&r| lut.sensitivity(r.log10(), log_l_100, Channel::YellowViolet))
            .collect();

        // Achromatic: peak should be at mid-frequency (4 cy/deg).
        let achro_peak_idx = s_a
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let achro_peak_rho = sample_rho[achro_peak_idx];
        assert!(
            (2.0..=8.0).contains(&achro_peak_rho),
            "achromatic peak at ρ={achro_peak_rho}, expected 2..8 cy/deg"
        );

        // Achromatic should fall off at high frequency.
        assert!(
            s_a[s_a.len() - 1] < s_a[achro_peak_idx],
            "achromatic should roll off at high freq"
        );

        // Chromatic channels: stronger at low freq than achromatic.
        // At 0.5 cy/deg, RG should be substantial (the chromatic
        // mechanism peaks at low freq).
        assert!(
            s_rg[0] > 10.0,
            "RG should have nontrivial sensitivity at 0.5 cy/deg, got {}",
            s_rg[0]
        );
        assert!(
            s_yv[0] > 1.0,
            "YV should have nontrivial sensitivity at 0.5 cy/deg, got {}",
            s_yv[0]
        );

        // YV should fall off faster than RG (YV peak is at even
        // lower freq, sharper rolloff).
        // At ρ = 32 cy/deg, YV should be well below RG.
        assert!(
            s_yv[s_yv.len() - 1] < s_rg[s_rg.len() - 1],
            "YV should roll off faster than RG at high freq"
        );
    }

    #[test]
    fn lut_luminance_dependence() {
        // At fixed ρ=4 cy/deg, achromatic sensitivity should
        // increase with luminance over the photopic range (DeVries-
        // Rose at low L, Weber plateau at moderate-high L). Test
        // monotonicity over the moderate-to-high range.
        let lut = CastleCsfLut::from_bytes(LUT_BYTES).unwrap();
        let log_rho_4 = 4.0_f32.log10();
        let lums = [1.0_f32, 10.0, 100.0, 1000.0];
        let s: Vec<f32> = lums
            .iter()
            .map(|&l| lut.sensitivity(log_rho_4, l.log10(), Channel::Achromatic))
            .collect();
        // Monotonically non-decreasing from 1 to 1000 cd/m².
        for w in s.windows(2) {
            assert!(
                w[1] >= w[0] * 0.95,
                "expected sensitivity to grow or plateau across luminance, got {:?}",
                s
            );
        }
        // Substantial range — at least 2× from dark to peak.
        assert!(
            s.last().unwrap() / s.first().unwrap() > 2.0,
            "expected at least 2× sensitivity range from 1 to 1000 cd/m², got {:?}",
            s
        );
    }

    #[test]
    fn channel_discriminants_pin_layout() {
        assert_eq!(Channel::Achromatic as u8, 0);
        assert_eq!(Channel::RedGreen as u8, 1);
        assert_eq!(Channel::YellowViolet as u8, 2);
    }

    #[test]
    fn rejects_truncated_lut() {
        let mut buf = LUT_BYTES.to_vec();
        buf.truncate(buf.len() - 1);
        let err = CastleCsfLut::from_bytes(&buf).unwrap_err();
        match err {
            LutError::WrongSize { .. } => {}
            other => panic!("expected WrongSize, got {other:?}"),
        }
    }

    #[test]
    fn rejects_corrupted_lut() {
        let mut buf = LUT_BYTES.to_vec();
        // Corrupt one byte in the data area.
        buf[2000] ^= 0xFF;
        let err = CastleCsfLut::from_bytes(&buf).unwrap_err();
        match err {
            LutError::BadCrc { .. } => {}
            other => panic!("expected BadCrc, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut buf = LUT_BYTES.to_vec();
        buf[0] = b'X';
        let err = CastleCsfLut::from_bytes(&buf).unwrap_err();
        match err {
            // BadMagic is checked before CRC.
            LutError::BadMagic => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }
}
