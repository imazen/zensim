//! Mohammadi 2025 statistical panel + MRR significance test +
//! decisive-rule machinery.
//!
//! As of 2026-05-26 (commit landing the `zenstats` crate
//! consolidation, see imazen/zenmetrics@36d71ca3) this module is a
//! **thin re-export shim** over the canonical home at
//! [`zenstats::panel`](https://github.com/imazen/zenmetrics). The
//! same statistical math was reimplemented across zensim, zenanalyze,
//! coefficient, zenmetrics, and jxl-encoder — sometimes correctly,
//! often subtly wrong — and the cross-repo audit at
//! `benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md` (Tier-2 #7)
//! flagged it as the single highest-reach consolidation target. This
//! crate's panel.rs body was extracted verbatim into the new
//! `zenstats` crate; the wildcard re-export below keeps every
//! existing `zensim_validate::panel::*` import working without
//! changes.
//!
//! All stats follow the same polarity-tolerant convention used by
//! `bake_verdict`: SROCC / KROCC are taken `.abs()` at the aggregate
//! level because bake outputs can be distance- or score-shaped
//! depending on the trainer's target convention. PLCC / OR / PWRC /
//! Z-RMSE are computed after a 4-parameter logistic rescale
//! (Mohammadi 2025 § IV-A convention) which absorbs both polarity
//! and saturation.
//!
//! `mrr_h` (Meng-Rosenthal-Rubin paired-correlation test) takes
//! r_AZ / r_BZ as raw signed correlations — callers MUST pass the
//! polarity-aligned correlation (apply `polarity_factor`'s sign) so
//! that the Fisher z-transform and the (1 − r²) denominators are
//! computed on the correct shape. Passing the `.abs()` value would
//! silently break the MRR for distance-shaped bakes.

pub use zenstats::panel::*;

/// Maximum number of rows fed into [`compute_light_panel_subsampled`].
/// `zenstats::panel::sa_st_curve` (called by `pwrc_sa_st_auc` inside
/// `compute_light_panel`) preallocates a `Vec<(f64, bool)>` of capacity
/// `n·(n−1)/2`, so for the safesyn training group (n=196086) the
/// preallocation alone is 307 GB and OOMs every trainer run that
/// validates against the full group (introduced 2026-05-26 by
/// commit `1e7f42a` swapping the O(n) PWRC proxy for the paper-correct
/// O(n²) SA-ST AUC inside `compute_light_panel`).
///
/// The SA-ST AUC is a deterministic functional of the pair-gap
/// distribution; a uniform random subsample is an unbiased Monte
/// Carlo estimator. At n=4096 the subsample yields `n(n−1)/2 ≈ 8.4 M`
/// pairs — more than enough resolution for training-time validation
/// logging (final paper-grade evaluation in `bake_verdict` runs on
/// canonical val parquets where n ≤ 10125 and the full O(n²) PWRC
/// is computed verbatim).
pub const LIGHT_PANEL_PWRC_SUBSAMPLE_CAP: usize = 4096;

/// Train-time wrapper around [`compute_light_panel`] that caps the
/// input to [`LIGHT_PANEL_PWRC_SUBSAMPLE_CAP`] via a deterministic
/// stride-decimation subsample. SROCC / PLCC / PWRC remain valid
/// estimators of the underlying ranking; only PWRC's CI widens
/// slightly at the subsample limit.
///
/// When `scores.len() ≤ LIGHT_PANEL_PWRC_SUBSAMPLE_CAP`, behaves
/// identically to `compute_light_panel`.
pub fn compute_light_panel_subsampled(
    scores: &[f64],
    humans: &[f64],
) -> zenstats::panel::LightPanel {
    let n = scores.len().min(humans.len());
    if n <= LIGHT_PANEL_PWRC_SUBSAMPLE_CAP {
        return zenstats::panel::compute_light_panel(scores, humans);
    }
    // Deterministic stride-decimation. A random subsample would also
    // work but stride is reproducible without a per-call RNG and
    // preserves coverage across the whole row range.
    let stride = n.div_ceil(LIGHT_PANEL_PWRC_SUBSAMPLE_CAP);
    let mut s_sub: Vec<f64> = Vec::with_capacity(LIGHT_PANEL_PWRC_SUBSAMPLE_CAP + 1);
    let mut h_sub: Vec<f64> = Vec::with_capacity(LIGHT_PANEL_PWRC_SUBSAMPLE_CAP + 1);
    let mut i = 0usize;
    while i < n {
        s_sub.push(scores[i]);
        h_sub.push(humans[i]);
        i += stride;
    }
    zenstats::panel::compute_light_panel(&s_sub, &h_sub)
}
