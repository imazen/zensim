//! Shared zensim profile used across the regression harness.
//!
//! `zensim` removed the built-in `PreviewV0_1` / `PreviewV0_2` linear
//! profiles (A-only ship). The regression tests and benches here score
//! through a linear profile as a *perceptual similarity tool* — the
//! `> 90` / `> 70` orientation/resize thresholds were calibrated against
//! the linear `100 − 18·d^0.7` score distribution, not the MLP profile
//! `A`. To keep those thresholds valid without changing any expected
//! value, this reconstructs the exact former `PreviewV0_2` params via the
//! stable [`zensim::profile::ProfileParams::builder`] extension point:
//! the builder defaults (`WEIGHTS_PREVIEW_V0_2`, `blur_radius = 5`,
//! `blur_passes = 1`, `num_scales = 4`, `100 − 18·d^0.7`, all dispositions
//! off, no MLP) are byte-for-byte the old `PROFILE_PREVIEW_V0_2` static,
//! so scores are identical to the removed built-in.

use std::sync::OnceLock;
use zensim::ZensimProfile;
use zensim::profile::ProfileParams;

/// The former built-in `ZensimProfile::PreviewV0_2`, reconstructed as a
/// [`ZensimProfile::Custom`] with bit-identical params. Use this anywhere
/// the harness previously named `ZensimProfile::PreviewV0_2`.
pub fn legacy_linear() -> ZensimProfile {
    static PARAMS: OnceLock<ProfileParams> = OnceLock::new();
    ZensimProfile::Custom {
        params: PARAMS.get_or_init(|| ProfileParams::builder().build()),
        name: "zensim-preview-v0.2",
    }
}
