//! Byte capture for base-versus-palette comparison at each frozen revision.
#![cfg(all(feature = "training", feature = "feature-regime-v2"))]
use zensim::{
    RgbSlice,
    feature_set_id::SlotSet,
    research::{self, Request},
};
#[test]
fn palette_legacy_vectors() {
    let rev = std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_| "1".into());
    let want = if rev == "5" {
        SlotSet::from_ranges([(0, 228), (372, 720)])
    } else {
        SlotSet::from_ranges([(0, 1825)])
    };
    let req = Request::for_slots(want, 1825);
    let mut bytes = Vec::new();
    for (w, h) in [(17, 19), (64, 64), (97, 63), (131, 65)] {
        let src: Vec<_> = (0..w * h)
            .map(|i| [(i % 251) as u8, (i * 13 % 251) as u8, (i * 31 % 251) as u8])
            .collect();
        let dst: Vec<_> = src.iter().map(|p| [p[0] / 2, p[1], p[2]]).collect();
        let values =
            research::extract(&req, &RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
                .unwrap();
        assert_eq!(values.values().len(), 1825);
        for value in values.values() {
            bytes.extend(value.to_bits().to_le_bytes());
        }
    }
    if let Some(path) = std::env::var_os("PALETTE_LEGACY_CAPTURE") {
        use std::io::Write;
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }
    if let Some(path) = std::env::var_os("PALETTE_LEGACY_EXPECT") {
        assert_eq!(
            bytes,
            std::fs::read(path).unwrap(),
            "frozen old-family bytes changed at Rev{rev}"
        );
    }
}
