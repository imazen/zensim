//! Shared adjacent-rung classification for bake and arbitrary-metric panels.
//! Buckets: forward, material inversion, identical samples, dead zone,
//! sub-resolution. The caller supplies materiality in its own score units.
pub(crate) fn bucket(delta: f64, material: f64, same_samples: bool) -> u8 {
    if delta > material {
        0
    } else if delta < -material {
        1
    } else if same_samples {
        2
    } else if delta.abs() <= 1e-9 {
        3
    } else {
        4
    }
}
