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
