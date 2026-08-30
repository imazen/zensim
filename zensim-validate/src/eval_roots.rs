//! Canonical eval feature-root paths, and the extractor ERA each one belongs to.
//!
//! **One owner.** Before 2026-08-30 the 372 root path was a string literal repeated in ten
//! `.rs` files, so "which root does a verdict read?" had ten answers and flipping it meant
//! finding all ten. Every binary now names a constant from here, and [`era_of`] turns a root
//! path into the one-line era label a report can print, so a verdict is self-describing.
//!
//! ## The eras
//!
//! The v1-372 masked (`f228..299`) + IW (`f300..371`) blocks were once a function of
//! `RAYON_NUM_THREADS` (`docs/DATASET_HISTORY.md` §3.27). Two fixes bracket the window:
//! `2dab8f30` (activity map, 2026-05-17) and `6af83b60` (geometry-only band layout,
//! 2026-06-09). A table extracted inside that window does not reproduce at its own build
//! commit; `basic f0..155` and `peaks f156..227` are unaffected on every corpus.
//!
//! * [`DEFAULT_FEATURES_ROOT_372`] — **the default since 2026-08-30.** Re-extracted at
//!   `build_commit ea16c7ee`; its `_MANIFEST.json` carries per-file sha256, row accounting,
//!   per-corpus era and the per-slot drift vs the stored tables. Deliberately reuses the OLD
//!   file names (`bake_verdict` hardcodes them per corpus), so the ROOT carries the date.
//!   Eight of the fourteen default corpora are current-extractor reads; six are byte-copies
//!   whose distorted material is no longer on this box (`aic4` is PRE-FIX and unrefreshable),
//!   which `benchmarks/eval_annotations.json` records as
//!   `eval372-current-root-copied-corpora-2026-08-30`.
//! * [`STORED_FEATURES_ROOT_2026_05_15`] — the previous default. **Still on disk, still a
//!   valid STORED-ERA read** — flipping the default rewrites nothing; it only changes what a
//!   flagless invocation means going forward. Probes and trainers that must keep reading the
//!   old tables name this constant, so their era choice is visible rather than accidental.
//!
//! Measured on this pair (`benchmarks/eval372_current_root_2026-08-30.md`): the era shift is
//! **model-specific, not a constant offset** — exactly 0.00000 on all 15 corpora for a bake
//! that reads only `f0..155`, up to |0.489| SROCC for one that leans on the drifted block —
//! so a published number cannot be corrected, only re-verdicted.
//!
//! The 720/944 roots are separate feature spaces, not eras of this one; they are named here
//! only so [`era_of`] can label them.

use std::path::Path;

/// `--regime 372` (the default regime) feature root — **current extractor**, 2026-08-30.
pub const DEFAULT_FEATURES_ROOT_372: &str =
    "/mnt/v/zen/zensim-training/2026-08-30-full-features-372";

/// The pre-2026-08-30 372 root. Valid as a STORED-ERA read; not the default since 2026-08-30.
pub const STORED_FEATURES_ROOT_2026_05_15: &str =
    "/mnt/v/zen/zensim-training/2026-05-15-full-features";

/// `--regime 720` feature root (folded+append 720-wide re-extraction).
pub const FEATURES_ROOT_720: &str = "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22";

/// `--regime 944` feature root (the SOTA-944 campaign root).
pub const FEATURES_ROOT_944: &str = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01";

/// One-line era label for a features root, for the note a report prints so every verdict
/// says which ruler produced it. Unrecognized roots are reported as unknown rather than
/// guessed — a wrong era label is worse than none.
pub fn era_of(root: &Path) -> &'static str {
    // Compare on the trailing component: callers pass the root with or without a trailing
    // slash, and an absolute path is the only form any of these constants take.
    let s = root.to_string_lossy();
    let s = s.trim_end_matches('/');
    match s {
        _ if s == DEFAULT_FEATURES_ROOT_372 => {
            "current-extractor 372 (2026-08-30 root, build_commit ea16c7ee)"
        }
        _ if s == STORED_FEATURES_ROOT_2026_05_15 => {
            "STORED-ERA 372 (2026-05-15 root — masked/IW f228..371 from the thread-dependent window)"
        }
        _ if s == FEATURES_ROOT_720 => "720 folded+append (ext720-canonical-2026-07-22)",
        _ if s == FEATURES_ROOT_944 => "944 campaign root (ext944-canonical-2026-08-01)",
        _ => "era UNKNOWN (not a registered root — check its _MANIFEST.json)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The flip of 2026-08-30 is PINNED here: a silent revert to the stored root would
    /// change every future 372-regime number without changing a single call site.
    #[test]
    fn default_372_root_is_the_current_extractor_root() {
        assert_eq!(
            DEFAULT_FEATURES_ROOT_372,
            "/mnt/v/zen/zensim-training/2026-08-30-full-features-372"
        );
        assert_ne!(DEFAULT_FEATURES_ROOT_372, STORED_FEATURES_ROOT_2026_05_15);
    }

    #[test]
    fn era_of_labels_every_registered_root_distinctly() {
        let roots = [
            DEFAULT_FEATURES_ROOT_372,
            STORED_FEATURES_ROOT_2026_05_15,
            FEATURES_ROOT_720,
            FEATURES_ROOT_944,
        ];
        let labels: Vec<&str> = roots.iter().map(|r| era_of(Path::new(r))).collect();
        for (i, a) in labels.iter().enumerate() {
            assert!(!a.contains("UNKNOWN"), "{} labeled unknown", roots[i]);
            for b in labels.iter().skip(i + 1) {
                assert_ne!(a, b, "two registered roots share an era label");
            }
        }
        // A trailing slash is the same root.
        assert_eq!(
            era_of(&PathBuf::from(format!("{DEFAULT_FEATURES_ROOT_372}/"))),
            era_of(Path::new(DEFAULT_FEATURES_ROOT_372))
        );
        // Anything else is honestly unknown, never guessed.
        assert!(
            era_of(Path::new("/mnt/v/zen/zensim-training/some-other-root")).contains("UNKNOWN")
        );
    }
}
