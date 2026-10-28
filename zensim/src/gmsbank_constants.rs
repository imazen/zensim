//! C8 GMSBANK stabilisers, frozen from TRAIN pixels only on 2026-09-23.
//!
//! 652 CID22 TRAIN and SafeSyn TRAIN pairs, 175,324,296 eligible co-sited
//! reference/distortion sites. Pair-median Prewitt gamma-luma / scale-1 XYB Y
//! central-difference ratio: p25=1.0395610843687322,
//! p50=1.104453582332472, p75=1.184353457359629. Thus
//! c_mid=0.0026/p50²=0.002131466010785597. 20/32 size/content strata were
//! empty, including every tiny/small stratum; the workspace source-constant
//! size sweep is not met. Full 32-stratum and input
//! provenance: benchmarks/gmsbank_calibration_2026-09-23.md and
//! the committed ratio/XYB producer zensim/src/gmsbank_calibration_instrument.rs
//! (invoked by scripts/gmsbank/calibration_instrument.sh). Raw summary:
//! /var/tmp/gmsbank/calibration/report.json (SHA256
//! a0f0d183ce47f0f9cb46c15d3ee792b15cce029a9b6f8c11269ddae1dad2c5fb).
//! The five literals are c_mid·4^(k−2), k=0..4. The hot path constructs no
//! constants with powf, exp, log, sqrt, or any other transcendental.

pub(crate) const GMSBANK_C: [f64; 5] = [
    0.00013321662567409982,
    0.0005328665026963993,
    0.002131466010785597,
    0.008525864043142388,
    0.03410345617256955,
];

// C8 chroma revision, 2026-09-24. Preregistered TRAIN pixels only.
// 1,520 native/box-derived records; quartiles/empty counts retained.
// Raw report SHA256 a6aecdf036d89b7afef10e9c8aaa6c935836fbc69ed12950e56f41eabcf9c633
// X/B gradient C1=140 and chromaticity C3=550 mapped independently.
pub(crate) const GMSBANK_X_C: [f64; 5] = [
    6.341568337543687e-05,
    0.0002536627335017475,
    0.00101465093400699,
    0.00405860373602796,
    0.01623441494411184,
];
pub(crate) const GMSBANK_B_C: [f64; 5] = [
    8.239404165903782e-05,
    0.00032957616663615127,
    0.001318304666544605,
    0.00527321866617842,
    0.02109287466471368,
];
pub(crate) const GMSBANK_CS_C: [[f64; 2]; 5] = [
    [9.29736082289899e-05, 0.04837252392816359],
    [0.0003718944329159596, 0.19349009571265435],
    [0.0014875777316638384, 0.7739603828506174],
    [0.0059503109266553535, 3.0958415314024696],
    [0.023801243706621414, 12.383366125609879],
];
