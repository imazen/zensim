pub fn profile(name: &str) -> zensim::ZensimProfile {
    match name {
        "E" => zensim::ZensimProfile::E,
        "ECodec" => zensim::ZensimProfile::ECodec,
        "A" => zensim::ZensimProfile::A,
        "PreviewV0_1" => zensim::ZensimProfile::PreviewV0_1,
        "PreviewV0_2" => zensim::ZensimProfile::PreviewV0_2,
        _ => panic!("unknown profile: {name}"),
    }
}
