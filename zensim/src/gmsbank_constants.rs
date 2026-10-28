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

pub(crate) const GMSBANK_CS_C: [[f64; 2]; 5] = [
    [9.29736082289899e-05, 0.04837252392816359],
    [0.0003718944329159596, 0.19349009571265435],
    [0.0014875777316638384, 0.7739603828506174],
    [0.0059503109266553535, 3.0958415314024696],
    [0.023801243706621414, 12.383366125609879],
];
