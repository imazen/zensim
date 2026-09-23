//! C8 GMSBANK stabilisers, frozen from TRAIN pixels only on 2026-09-23.
//!
//! 652 CID22 TRAIN and SafeSyn TRAIN pairs, 175,324,296 eligible co-sited
//! reference/distortion sites. Pair-median Prewitt gamma-luma / scale-1 XYB Y
//! central-difference ratio: p25=1.0395610843687322,
//! p50=1.104453582332472, p75=1.184353457359629. Thus
//! c_mid=0.0026/p50²=0.002131466010785597. Full 32-stratum and input
//! provenance: benchmarks/gmsbank_calibration_2026-09-23.md and
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
