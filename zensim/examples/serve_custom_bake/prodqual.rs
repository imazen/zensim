//! Label-free probe of explicitly named packed models. No corpus or verdict reader.
//! Run with `ZENSIM_FORMULA_REV=5 serve_custom_bake --prodqual MODEL...`.
//! JSON records every dispatch permutation and every seed, including failures;
//! the caller compares the recorded bits and evaluates identity bands. Synthetic
//! ladder monotonicity is an observation, never an admission threshold here.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use serde_json::json;
use zensim::research::{self, Request};
use zensim::{BakeScorer, RgbSlice};

const GEOMETRIES: [(usize, usize); 4] = [(64, 64), (97, 63), (131, 65), (255, 129)];
const DROPPED_BITS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

// Same deterministic texture used by featcanon_rev5_parity, without a corpus.
fn reference(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let g = ((x * 7 + y * 13) % 251) as u8;
                let t = (((x * 31) ^ (y * 17)) % 67) as u8;
                let s = g.wrapping_add(t);
                [s, g.wrapping_add(t / 2), 255 - s]
            })
        })
        .collect()
}

pub(super) fn run(paths: &[String]) {
    assert!(
        !paths.is_empty(),
        "--prodqual requires explicit model paths"
    );
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
    let _guard = archmage::testing::lock_token_testing();
    let mut seeds = Vec::new();
    for (seed, path) in paths.iter().enumerate() {
        eprintln!("prodqual: seed {seed} model {path}");
        let bytes = std::fs::read(path).expect("read explicitly named model");
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse packed model");
        assert_eq!(
            model
                .metadata()
                .get_utf8("zentrain.formula_revision")
                .unwrap(),
            "5"
        );
        let request = Request::for_bake_bytes(&bytes).expect("declared extraction plan");
        let mut permutations = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            eprintln!("prodqual: seed {seed} permutation {}", perm.label);
            let mut scorer = BakeScorer::new(&model)
                .expect("public scorer")
                .with_parallel(false);
            let mut rows = Vec::new();
            for (w, h) in GEOMETRIES {
                let pixels = reference(w, h);
                let source = RgbSlice::new(&pixels, w, h);
                let mut steering = scorer
                    .prepare_steering(&source, 8)
                    .expect("prepare steering");
                for dropped in DROPPED_BITS {
                    let mask = (255u32 << dropped) as u8;
                    let distorted: Vec<[u8; 3]> =
                        pixels.iter().map(|p| p.map(|c| c & mask)).collect();
                    let dest = RgbSlice::new(&distorted, w, h);
                    let cached = steering
                        .compute(&dest, None)
                        .expect("cached steering score");
                    // A separate scorer avoids aliasing the session's mutable borrow.
                    let mut direct = BakeScorer::new(&model).unwrap().with_parallel(false);
                    let served = direct
                        .compute(&source, &dest, None)
                        .expect("public pixel score");
                    let extracted =
                        research::extract(&request, &source, &dest).expect("canonical extractor");
                    let feature_score = direct
                        .score_features(extracted.values(), w as u32, h as u32, None)
                        .expect("feature score");
                    let identity_aware = direct
                        .score_features_with_identity(
                            extracted.values(),
                            w as u32,
                            h as u32,
                            None,
                            pixels == distorted,
                        )
                        .expect("identity-aware feature score");
                    let read_bits: Vec<u64> = request
                        .want()
                        .iter_slots()
                        .map(|id| extracted.values()[id].to_bits())
                        .collect();
                    let mismatches: Vec<usize> = request
                        .want()
                        .iter_slots()
                        .filter(|&id| {
                            served.features()[id].to_bits() != extracted.values()[id].to_bits()
                        })
                        .collect();
                    let scores = [
                        served.score(),
                        cached.result().score(),
                        feature_score,
                        identity_aware,
                    ];
                    rows.push(json!({
                        "width":w, "height":h, "dropped_bits":dropped,
                        "pixel_score":served.score(), "pixel_bits":served.score().to_bits(),
                        "cached_bits":cached.result().score().to_bits(),
                        "feature_score":feature_score, "feature_bits":feature_score.to_bits(),
                        "identity_aware_bits":identity_aware.to_bits(),
                        "finite":scores.iter().all(|s| s.is_finite()),
                        "feature_mismatches":mismatches, "read_bits":read_bits,
                        "density_cells":cached.attribution().density().len(),
                    }));
                }
            }
            permutations.push(json!({"label":perm.label, "rows":rows}));
        });
        seeds.push(
            json!({"seed":seed, "model":path, "caller_width":model.caller_input_width(),
            "declared_reads":request.want().iter_slots().count(),
            "permutations_run":report.permutations_run, "permutations":permutations}),
        );
    }
    println!(
        "{}",
        json!({"schema":1, "formula_revision":5,
        "input":"synthetic integer texture; cumulative low-bit truncation",
        "geometries":GEOMETRIES, "dropped_bits":DROPPED_BITS, "seeds":seeds})
    );
}
