pub mod distortions;
pub mod generators;
pub mod parity_cells;

/// Fixed 372-feature fixture, independent of future default-model rotations.
#[allow(dead_code)]
pub fn legacy_full_profile() -> zensim::ZensimProfile {
    #[cfg(feature = "custom-profiles")]
    {
        static PARAMS: std::sync::OnceLock<zensim::profile::ProfileParams> =
            std::sync::OnceLock::new();
        fn bytes() -> &'static [u8] {
            include_bytes!(
                "../../weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
            )
        }
        zensim::ZensimProfile::Custom {
            name: "historical-linear-B",
            params: PARAMS.get_or_init(|| {
                zensim::profile::ProfileParams::builder()
                    .mlp(bytes)
                    .extended_features(true)
                    .compute_iw_features(true)
                    .skip_score_mapping(true)
                    .extrapolate_score(true)
                    .build()
            }),
        }
    }
    #[cfg(not(feature = "custom-profiles"))]
    {
        zensim::ZensimProfile::BHdr
    }
}
