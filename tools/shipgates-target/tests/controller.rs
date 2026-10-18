//! Analytical peer-unit oracle for the actual shared controller.
use anyhow::Result;
use std::cell::Cell;
use zensim::{ImageSource, RgbSlice};
use zensim_target::{CodecKind, ProbeRecord, TargetResult, TargetSpec, codec};
#[path = "../../../zensim-target/src/controller.rs"]
mod controller;

struct Linear {
    calls: Cell<usize>,
}
impl codec::CodecBackend for Linear {
    fn quality_range(&self) -> (f32, f32) {
        (0., 100.)
    }
    fn lower_quality_means_higher_score(&self) -> bool {
        false
    }
    fn encode_decode(
        &self,
        _rgb: &[u8],
        _w: u32,
        _h: u32,
        knob: f32,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        self.calls.set(self.calls.get() + 1);
        Ok((knob.to_le_bytes().to_vec(), vec![knob as u8; 3]))
    }
}
fn run(budget: u32, alpha: f64, beta: f64) -> TargetResult {
    let backend = Linear {
        calls: Cell::new(0),
    };
    let result = controller::search(
        &[0; 3],
        1,
        1,
        CodecKind::Jpeg,
        TargetSpec {
            target: 75.,
            tolerance: 1.,
            max_iterations: budget,
            ..TargetSpec::default()
        },
        None,
        &backend,
        |_: &RgbSlice<'_>, b: &RgbSlice<'_>| {
            // Lower butteraugli distance is higher quality. The registered inverse
            // affine mapping must recover the same score from those native units.
            let distance = -(alpha * f64::from(b.row_bytes(0)[0]) + beta);
            Ok(((-distance - beta) / alpha) as f32)
        },
    )
    .unwrap();
    assert_eq!(backend.calls.get(), result.iterations as usize);
    assert!(backend.calls.get() <= budget as usize);
    result
}
#[test]
fn native_units_preserve_all_budget_proposals_and_best_bytes() {
    for budget in [1, 2, 3] {
        let base = run(budget, 1., 0.);
        let peer = run(budget, 0.25, -25.);
        assert_eq!(base.encoded, peer.encoded);
        assert_eq!(base.achieved_score, peer.achieved_score);
        assert_eq!(
            base.probes.iter().map(|p| p.knob).collect::<Vec<_>>(),
            peer.probes.iter().map(|p| p.knob).collect::<Vec<_>>()
        );
        assert_eq!(base.achieved_score, if budget == 1 { 50. } else { 75. });
        assert_eq!(base.converged, budget >= 2);
    }
}
