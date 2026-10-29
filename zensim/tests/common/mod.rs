pub mod distortions;
pub mod generators;
pub mod parity_cells;

/// A profile whose extraction emits the full 372-feature v1 vector
/// (basic, peaks, masked and IW pools at the first formula revision).
#[allow(dead_code)]
pub fn legacy_full_profile() -> zensim::ZensimProfile {
    zensim::ZensimProfile::A
}
