pub fn profile(name: &str) -> zensim::ZensimProfile {
    match name {
        "A" | "zensim-a" => zensim::ZensimProfile::A,
        #[cfg(feature = "legacy-models")]
        "PreviewV0_1" => zensim::ZensimProfile::PreviewV0_1,
        #[cfg(feature = "legacy-models")]
        "PreviewV0_2" => zensim::ZensimProfile::PreviewV0_2,
        _ => panic!("unknown profile: {name}"),
    }
}
