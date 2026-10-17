//! Explicit external-model audit; compiled only by steerfix_instrument.
#[cfg(test)]
use super::*;
#[cfg(test)]
use crate::RgbSlice;
#[cfg(test)]
use serde_json::json;
use std::time::Instant;

pub(super) fn start() -> Option<Instant> {
    std::env::var_os("MAPPERF_STAGE_TIMING").map(|_| Instant::now())
}
pub(super) fn stage(t: &mut Option<Instant>, name: &str) {
    if let Some(previous) = t.as_mut() {
        eprintln!("MAPPERF_STAGE {} {}", name, previous.elapsed().as_nanos());
        *previous = Instant::now();
    }
}

pub(super) fn total(t: Option<Instant>, name: &str) {
    if let Some(t) = t {
        eprintln!("MAPPERF_RUNTIME_TOTAL {} {}", name, t.elapsed().as_nanos());
    }
}

#[test]
fn declared_plan_and_stages() {
    use sha2::{Digest, Sha256};
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
    let root =
        std::path::PathBuf::from(std::env::var("MAPPERF_AUDIT_ROOT").expect("explicit packet"));
    let job: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("JOB.json")).unwrap()).unwrap();
    let selected = std::env::var("MAPPERF_AUDIT_MODEL").expect("explicit model");
    let m = &job["models"][&selected];
    let bytes = std::fs::read(root.join(m["path"].as_str().unwrap())).unwrap();
    assert_eq!(
        Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        m["sha256"].as_str().unwrap()
    );
    let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
    let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(true);
    let plan = scorer.plan().unwrap();
    let declared = crate::feature_layout::declared_ids(&model).unwrap();
    let reads = crate::feature_plan::bake_read_slots(&model).unwrap();
    let populated = plan
        .compute
        .populated_slots(crate::NUM_SCALES, plan.walk_width());
    println!(
        "MAPPERF_PLAN {}",
        json!({"model":selected,"sha256":m["sha256"],
        "model_inputs":model.n_inputs(),"declared_raw_count":declared.iter_slots().count(),
        "read_raw_count":reads.iter_slots().count(),"computed_raw_count":populated.iter_slots().count(),
        "emitted_count":plan.emit.iter_slots().count(),"walk_width":plan.walk_width(),
        "declared_ids":declared.iter_slots().collect::<Vec<_>>(),"reads":reads.iter_slots().collect::<Vec<_>>(),
        "computed_ids":populated.iter_slots().collect::<Vec<_>>(),"compute":format!("{:?}",plan.compute)})
    );
    if std::env::var_os("MAPPERF_COUNTS_ONLY").is_some() {
        return;
    }
    if std::env::var_os("MAPPERF_SUBSET_ORACLE").is_some() {
        assert!(
            plan.compute.omit_pjnd_fragility,
            "explicit diff-410 oracle only"
        );
        for side in [64, 256, 1024, 2048] {
            let (pixels, distorted) = test_pair(side, side);
            let src = RgbSlice::new(&pixels, side, side);
            let dst = RgbSlice::new(&distorted, side, side);
            let mut session = scorer.prepare_steering(&src, 1).unwrap();
            let narrowed = session.compute(&dst, None).unwrap();
            let mut original = plan.clone();
            original.compute.omit_pjnd_fragility = false;
            original.emit = original
                .compute
                .populated_slots(crate::NUM_SCALES, original.walk_width())
                .intersect(&original.layout.ids());
            let before = session
                .scorer
                .compute_attribution_input(
                    session.source,
                    &session.reference,
                    &dst,
                    None,
                    &mut session.scratch,
                    1,
                    None,
                    Some(original),
                )
                .unwrap();
            assert_eq!(
                narrowed.result().score().to_bits(),
                before.result().score().to_bits()
            );
            assert_eq!(
                narrowed
                    .sensitivities()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                before
                    .sensitivities()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                narrowed
                    .attribution()
                    .density()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                before
                    .attribution()
                    .density()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            let extras: Vec<_> = populated
                .iter_slots()
                .filter(|id| !reads.contains(*id))
                .collect();
            assert!(extras.is_empty(), "narrow plan has excess slots");
            let mut omitted = Vec::new();
            for (id, (a, b)) in narrowed
                .result()
                .features()
                .iter()
                .zip(before.result().features())
                .enumerate()
            {
                if a.to_bits() != b.to_bits() {
                    assert_eq!(a.to_bits(), 0.0f64.to_bits());
                    assert!(!reads.contains(id));
                    assert_eq!(
                        crate::feature_defs::def_at(id, crate::NUM_SCALES)
                            .unwrap()
                            .signal
                            .name,
                        "pjnd_fragility"
                    );
                    omitted.push(id);
                }
            }
            assert_eq!(omitted, [422, 480, 509, 538, 567, 596, 625, 654, 683, 712]);
            for block in [1, 8, 16, 32, 64] {
                for y in (0..side).step_by(block) {
                    for x in (0..side).step_by(block) {
                        let rect = (x, y, (x + block).min(side), (y + block).min(side));
                        assert_eq!(
                            narrowed
                                .attribution()
                                .query_rect(rect.0, rect.1, rect.2, rect.3)
                                .to_bits(),
                            before
                                .attribution()
                                .query_rect(rect.0, rect.1, rect.2, rect.3)
                                .to_bits()
                        );
                    }
                }
            }
            for (x, y) in [(0, 0), (side / 2, side / 2), (side - 8, side - 8)] {
                assert_eq!(
                    narrowed.refinement_gain(x, y, x + 8, y + 8).to_bits(),
                    before.refinement_gain(x, y, x + 8, y + 8).to_bits()
                );
            }
            println!("MAPPERF_SUBSET_ORACLE_PASS {side} omitted={omitted:?}");
        }
        return;
    }

    let (src, dst) = test_pair(1024, 1024);
    let src = RgbSlice::new(&src, 1024, 1024);
    let dst = RgbSlice::new(&dst, 1024, 1024);
    let mut session = scorer.prepare_steering(&src, 1).unwrap();
    for i in 0..12 {
        eprintln!("MAPPERF_REPEAT {i}");
        let t = Instant::now();
        let result = session.compute(&dst, None).unwrap();
        let elapsed = t.elapsed().as_nanos();
        eprintln!("MAPPERF_TOTAL {elapsed}");
        std::hint::black_box(result);
    }
    drop(session);
    for i in 0..12 {
        let t = Instant::now();
        let result = scorer.compute(&src, &dst, None).unwrap();
        eprintln!("MAPPERF_SCALAR {i} {}", t.elapsed().as_nanos());
        std::hint::black_box(result);
    }
}

// Mechanically copied from ssim2_speed_bar::test_pair to preserve the matched fixture.
#[cfg(test)]
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}
