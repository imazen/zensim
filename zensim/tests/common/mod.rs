pub mod distortions;
pub mod generators;
pub mod parity_cells;

/// The adopted serving profile.
#[allow(dead_code)]
pub fn serving_profile() -> zensim::ZensimProfile {
    zensim::ZensimProfile::A
}
