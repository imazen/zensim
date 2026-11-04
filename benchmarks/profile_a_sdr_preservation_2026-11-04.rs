// Reproduce the Phase 3 same-host SDR score comparison. The scratch Cargo
// package forwards its `legacy-models` feature to zensim.
mod pairs {
    include!(env!("ZENSIM_SDR_PAIRS"));
}
use zensim::{RgbSlice, Zensim, ZensimProfile};
const IDS: &[&str] = &[
    "64x64-photo-noise",
    "64x64-edges-blur",
    "65x63-text-blocks",
    "128x96-photo-posterize",
    "200x150-edges-shift",
    "200x150-photo-blur",
    "256x256-text-noise",
    "256x256-photo-pixels",
    "333x251-edges-blocks",
    "333x251-photo-identity",
    "16x16-photo-noise",
    "31x17-text-shift",
    "512x384-photo-blocks",
    "1024x768-photo-noise",
];
fn main() {
    let pairs = pairs::split_parity_pairs();
    println!("profile\tpair\tscore_bits");
    let profiles = [
        ZensimProfile::A,
        #[cfg(feature = "legacy-models")]
        ZensimProfile::PreviewV0_1,
        #[cfg(feature = "legacy-models")]
        ZensimProfile::PreviewV0_2,
    ];
    for profile in profiles {
        for id in IDS {
            let p = pairs.iter().find(|p| p.id == *id).unwrap();
            let src = RgbSlice::new(bytemuck::cast_slice(&p.src), p.width, p.height);
            let dst = RgbSlice::new(bytemuck::cast_slice(&p.dst), p.width, p.height);
            let result = Zensim::new(profile).compute(&src, &dst).unwrap();
            println!("{}\t{id}\t{:016x}", profile.name(), result.score().to_bits());
        }
    }
}
