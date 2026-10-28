//! Reusable candidate serving surface. Inference and metadata live in the
//! parent metric owner, shared with every named profile.

use super::*;
use std::sync::Arc;

#[path = "bake_gradient.rs"]
mod gradient;

/// A loaded bake's reusable scoring state.
///
/// The caller owns the parsed model; no bytes are leaked or installed into a
/// global loader. Create one scorer per worker and reuse it across image pairs
/// or feature rows. All baked heads, output splines and codec calibration run
/// through the same dispatch used by named profiles.
///
/// By default the bake's calibrated output is returned unchanged, including
/// negative scores. Model authors may select a different final disposition
/// with [`Self::with_score_disposition`]. This is model configuration; end users
/// still control one target score.
pub struct BakeScorer<'a> {
    model: &'a crate::mlp::Model,
    predictor: crate::mlp::Predictor<'a>,
    metadata: Arc<ScoreMetadata>,
    layout: crate::feature_layout::Layout,
    gathered: Vec<f64>,
    inputs: Vec<f32>,
    standardized: Vec<f64>,
    pixel_scratch: crate::feature_v2::V2Scratch,
    disposition: Option<&'a ProfileParams>,
    parallel: bool,
    finite_moments: bool,
    members: Vec<BakeScorer<'a>>,
    weights: Option<Vec<f64>>,
    /// `Plan::for_bake(self.model)`, built on first use. It depends only on
    /// the model (and the process formula revision, fixed per process), and a
    /// scorer's model never changes, so the cache needs no invalidation.
    /// Composition (members, weights, companion) stays per call in [`Self::plan`].
    model_plan: std::sync::OnceLock<crate::feature_plan::Plan>,
    /// [`Self::plan`]'s result, built on first use. Unlike `model_plan` it
    /// depends on the composition, so every change to `members`, `weights` or
    /// `corruption` resets it ([`Self::reset_plan`]).
    served_plan: std::sync::OnceLock<crate::feature_plan::Plan>,
}

impl<'a> BakeScorer<'a> {

    pub(super) fn with_metadata(
        model: &'a crate::mlp::Model,
        metadata: Arc<ScoreMetadata>,
    ) -> Result<Self, ZensimError> {
        crate::feature_layout::formula_revision(model)?;
        crate::sampling::Sampling::from_model(model)?;
        // A derived-input bake declares its read set, not one id per input;
        // `metadata` already validated its entries against that read set and
        // the caller width (`derived_inputs::from_model`).
        let declared_width = match metadata.derived_inputs.as_deref() {
            Some(d) => d.gather_width(),
            None => model.caller_input_width(),
        };
        if model.metadata().get("zentrain.feature_ids").is_some()
            && crate::feature_layout::declared_ids(model)
                .is_none_or(|ids| ids.len() != declared_width)
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "invalid zentrain.feature_ids declaration or input count",
            });
        }
        if let Some(head) = metadata.minmax_head.as_ref()
            && (head.n != model.n_inputs() || model.caller_input_width() != model.n_inputs())
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "min-max head width differs from model input width",
            });
        }
        Ok(Self {
            model,
            predictor: crate::mlp::Predictor::new(model),
            metadata,
            layout: crate::feature_layout::declared_layout(model),
            gathered: Vec::new(),
            inputs: Vec::with_capacity(model.caller_input_width()),
            standardized: Vec::new(),
            pixel_scratch: crate::feature_v2::V2Scratch::default(),
            disposition: None,
            parallel: true,
            finite_moments: false,
            members: Vec::new(),
            weights: None,
            model_plan: std::sync::OnceLock::new(),
            served_plan: std::sync::OnceLock::new(),
        })
    }

    /// Score an identity-layout feature row using the bake's declared IDs.
    ///
    /// The caller must supply features at the bake's extraction revision and
    /// decoder era. Table admission validates that contract before a batch.
    /// Dimensions supply optional size axes; a codec hint selects any baked
    /// per-codec calibration. A zero row is not evidence of identical pixels.
    ///
    /// # Errors
    /// Returns a named model error if the row cannot supply the declared inputs
    /// or the predictor cannot execute the model.
    pub fn score_features(
        &mut self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Result<f64, ZensimError> {
        let primary = if self.weights.as_ref().is_some_and(|w| w[0] == 0.0) {
            0.0
        } else {
            forward_model_with_codec(
                self.model,
                &mut self.predictor,
                &self.metadata,
                &self.layout,
                &mut self.gathered,
                &mut self.inputs,
                &mut self.standardized,
                features,
                width,
                height,
                codec_hint,
            )?
        };
        let raw = if let Some(weights) = &self.weights {
            let mut total = 0.0;
            if weights[0] != 0.0 {
                total += weights[0] * primary;
            }
            for (member, &weight) in self.members.iter_mut().zip(&weights[1..]) {
                if weight != 0.0 {
                    total += weight * member.score_features(features, width, height, codec_hint)?;
                }
            }
            total
        } else if self.members.is_empty() {
            primary
        } else {
            let mut total = 0.0;
            total += primary;
            for member in &mut self.members {
                total += member.score_features(features, width, height, codec_hint)?;
            }
            total / (self.members.len() + 1) as f64
        };
        let score = self.disposition.map_or(raw, |p| dispose_mlp_raw(raw, p));
        Ok(score)
    }

}

#[cfg(test)]
mod revision_contract_tests {

    use crate::feature_defs::FormulaRevision;
    use crate::ssim_form::SsimLumaForm;
    // Everything that scores PIXELS through a bake needs `feature-regime-v2`:
    // `bake_declaring` fixes a v2 extraction requirement, so `BakeScorer::new`
    // refuses it without the feature and the pixel-path tests are gated.
    use crate::ssim_form::run_at_revision;
    use crate::{RgbSlice, ZensimError};

    #[test]
    #[ignore = "requires registered candidate IDs and external full bakes"]
    fn external_by_v2fy_declared_ids_registration_audit() {
        let candidate = std::env::var("ZENSIM_PLAN_AUDIT_IDS_JSON").expect("candidate IDs path");
        let candidate: serde_json::Value =
            serde_json::from_slice(&std::fs::read(candidate).unwrap()).unwrap();
        let expected: Vec<u16> =
            serde_json::from_value(candidate["candidates"]["by_v2fy"].clone()).unwrap();
        assert_eq!(expected.len(), 420);
        assert!(expected.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(expected.last(), Some(&719));
        let paths = std::env::var("ZENSIM_PLAN_AUDIT_BAKES").expect("colon-separated bake paths");
        assert!(!paths.is_empty());
        for path in paths.split(':') {
            let model = zenpredict::Model::from_bytes(&std::fs::read(path).unwrap()).unwrap();
            // Use the serving metadata owner, never a second bake parser.
            let actual = crate::declared_feature_ids(&model).expect("declared dense bake");
            assert_eq!(actual, expected, "candidate/bake ID mismatch: {path}");
            println!(
                "R5INTEG3_IDS {}",
                serde_json::json!({"path": path, "ids": actual})
            );
        }
    }

    /// A bake over the given feature IDs with small deterministic negative weights (declared revision = the
    /// process default, so the wide-family arithmetic contract matches).
    fn bake_over(ids: &[usize]) -> zenpredict::Model {
        bake_over_sampling(ids, None)
    }

    fn bake_over_sampling(ids: &[usize], sampling: Option<&str>) -> zenpredict::Model {
        zenpredict::Model::from_bytes(&bake_bytes_over_sampling(ids, sampling)).unwrap()
    }

    fn bake_bytes_over_sampling(ids: &[usize], sampling: Option<&str>) -> Vec<u8> {
        let mut metadata = vec![
            serde_json::json!({"key":"zentrain.feature_ids","type":"utf8",
                "text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")}),
            serde_json::json!({"key":"zentrain.formula_revision","type":"utf8",
                "text":(crate::ssim_form::active_revision() as u8 + 1).to_string()}),
        ];
        if let Some(sampling) = sampling {
            metadata
                .push(serde_json::json!({"key":"zentrain.sampling","type":"utf8","text":sampling}));
        }
        let weights: Vec<f64> = ids
            .iter()
            .map(|&id| -(0.002 + 0.0003 * ((id * 7) % 13) as f64))
            .collect();
        let recipe = serde_json::json!({
            "schema_hash":1,"scaler_mean":vec![0.0;ids.len()],"scaler_scale":vec![1.0;ids.len()],
            "metadata":metadata,
            "layers":[{"in_dim":ids.len(),"out_dim":1,"activation":"identity","dtype":"f32",
                "weights":weights,"biases":[100.0]}]
        });
        zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap()
    }

    /// COSTSET2: v2-bearing plans over basic/peaks/v2 skip X/B work at scales whose chroma slots they never read
    /// (the channel mask), and every consumed feature stays bit-identical to the unrestricted extraction.
    /// The mask must engage for the Y-only and coarse-chroma sets (otherwise the saving is not real), must NOT
    /// engage when an append (cross-channel) block is read, and `edge_width_change` of a chroma channel must
    /// keep the next scale's chroma gradients alive.
    #[test]
    fn v2_plans_skip_unread_chroma_work_without_changing_any_feature() {
        use crate::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
        let v2 = |s: usize, c: usize, k: usize| 372 + s * 87 + c * 29 + k;
        // Basic with X/B only at scales >= 1 (fine-Y) and basic Y-only: the chroma slots a plan reads decide the mask.
        let basic_y: Vec<usize> = (0..156).filter(|id| (id / 13) % 3 == 1).collect();
        let basic_fine_y: Vec<usize> = (0..156)
            .filter(|id| id / 39 >= 1 || (id / 13) % 3 == 1)
            .collect();
        let v2_y_all: Vec<usize> = (0..4)
            .flat_map(|s| (0..29).map(move |k| v2(s, 1, k)))
            .collect();
        let sets: Vec<(&str, Vec<usize>, bool)> = vec![
            (
                "basic Y + v2 Y-only, all scales",
                basic_y
                    .iter()
                    .copied()
                    .chain(v2_y_all.iter().copied())
                    .collect(),
                true,
            ),
            (
                "fine-Y basic + v2 Y at scale 0, all channels at 1-3",
                basic_fine_y
                    .iter()
                    .copied()
                    .chain((0..4).flat_map(|s| {
                        (0..3)
                            .filter(move |&c| s > 0 || c == 1)
                            .flat_map(move |c| (0..29).map(move |k| v2(s, c, k)))
                    }))
                    .collect(),
                true,
            ),
            (
                "fine-Y basic + peaks + v2 Y-only scales 1-3",
                basic_fine_y
                    .iter()
                    .copied()
                    .chain((156..228).filter(|id| ((id - 156) / 6) % 3 == 1))
                    .chain((1..4).flat_map(|s| (0..29).map(move |k| v2(s, 1, k))))
                    .collect(),
                true,
            ),
            (
                "v2 only, chroma edge-width at scale 1",
                v2_y_all.iter().copied().chain([v2(1, 0, 28)]).collect(),
                true,
            ),
            (
                "full basic + Y-only v2: chroma still read by basic",
                (0..156).chain(v2_y_all.iter().copied()).collect(),
                false,
            ),
            (
                "full v2 + basic (no mask possible)",
                (0..156).chain(372..720).collect(),
                false,
            ),
            (
                "append read: complete walk",
                basic_y
                    .iter()
                    .copied()
                    .chain([v2(0, 1, 3), 720 + 17 * 3 + 9])
                    .collect(),
                false,
            ),
        ];
        for (w, h) in [(64usize, 64usize), (97, 65), (128, 96)] {
            let (r, d) = crate::serving::pair(w, h);
            let (rs, ds) = (RgbSlice::new(&r, w, h), RgbSlice::new(&d, w, h));
            let full = crate::feature_v2::compute_folded720_streaming_impl(
                &rs,
                &ds,
                None,
                true,
                V2NewFeatureToggles {
                    v1_pools: V1PoolsMode::Full,
                    append_block: true,
                    append2_block: true,
                    csfw_block: true,
                    ..Default::default()
                },
                &mut V2Scratch::new(),
                None,
            )
            .unwrap();
            for (name, ids, masked) in &sets {
                let mut ids = ids.clone();
                ids.sort_unstable();
                ids.dedup();
                let model = bake_over(&ids);
                let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
                let compute = scorer.plan().unwrap().compute;
                assert_eq!(
                    !compute.full_res_xb || compute.coarse_y_only_scales != 0,
                    *masked,
                    "{name}: channel mask engagement"
                );
                if name.starts_with("v2 only, chroma edge-width") {
                    // chroma at scale 1 is read (edge width), so scale 2 keeps its chroma gradients too.
                    assert!(
                        compute.channel_active(1, 0) && compute.channel_active(2, 0),
                        "{name}"
                    );
                    assert!(
                        !compute.channel_active(3, 0) && !compute.channel_active(0, 0),
                        "{name}"
                    );
                }
                let served = scorer.compute(&rs, &ds, None).unwrap();
                for &id in &ids {
                    assert_eq!(
                        served.features()[id].to_bits(),
                        full.features()[id].to_bits(),
                        "{name} {w}x{h} f{id}"
                    );
                }
            }
        }
    }

    #[test]
    fn consumed_ids_include_replacement_minmax_inputs() {
        // Placeholder network reads dense position zero (f13); replacement
        // min-max reads position one (f300). Its pixel result must agree with
        // canonical extraction, not a structural zero left by the network.
        let mut payload = Vec::new();
        for n in [1u32, 1, 2] {
            payload.extend(n.to_le_bytes());
        }
        for value in [0.0f32, -2.0, 80.0] {
            payload.extend(value.to_le_bytes());
        }
        let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
        let recipe = serde_json::json!({
            "schema_hash":1,"scaler_mean":[0.0,0.0],"scaler_scale":[1.0,1.0],
            "metadata":[
                {"key":"zentrain.feature_ids","type":"utf8","text":"13\n300"},
                {"key":"zentrain.minmax_monotone_head","type":"bytes","hex":hex}],
            "layers":[{"in_dim":2,"out_dim":1,"activation":"identity","dtype":"f32",
                "weights":[1.0,0.0],"biases":[0.0]}]
        });
        let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
        assert_eq!(scorer.consumed_feature_ids().unwrap(), [300]);
        let mut row = vec![0.0; 372];
        row[13] = 7.0;
        row[300] = 3.0;
        assert_eq!(scorer.score_features(&row, 96, 96, None).unwrap(), 74.0);
        assert_eq!(
            scorer
                .score_features_fd_gradient(&row, 96, 96, None)
                .unwrap()[13],
            0.0
        );
        assert!(
            scorer
                .score_features_fd_gradient(&row, 96, 96, None)
                .unwrap()[300]
                < -1.9
        );
        let src: Vec<_> = (0..96 * 96).map(|i| [(i % 251) as u8; 3]).collect();
        let mut dst = src.clone();
        dst[100..160].fill([255, 0, 255]);
        let rs = RgbSlice::new(&src, 96, 96);
        let ds = RgbSlice::new(&dst, 96, 96);
        let reference = zenpredict::Model::from_bytes(&bake_declaring(None, 300)).unwrap();
        let canonical = crate::BakeScorer::new(&reference)
            .unwrap()
            .compute(&rs, &ds, None)
            .unwrap();
        let actual = scorer.compute(&rs, &ds, None).unwrap();
        assert_ne!(canonical.features()[300], 0.0);
        assert_eq!(actual.features()[300], canonical.features()[300]);
        assert_eq!(
            actual.score(),
            scorer
                .score_features(canonical.features(), 96, 96, None)
                .unwrap()
        );
    }

    #[test]
    fn exact_gradient_matches_weighted_linear_coefficients() {
        let mut weights = vec![0.0; 944];
        weights[13] = -0.7;
        weights[52] = -1.3;
        let recipe = serde_json::json!({
            "schema_hash":1,"scaler_mean":vec![0.0;944],"scaler_scale":vec![1.0;944],
            "layers":[{"in_dim":944,"out_dim":1,"activation":"identity","dtype":"f32",
                "weights":weights,"biases":[100.0]}]
        });
        let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
        let models = vec![
            zenpredict::Model::from_bytes(&bytes).unwrap(),
            zenpredict::Model::from_bytes(&bake_declaring(None, 91)).unwrap(),
        ];
        let mut row: Vec<_> = (0..944).map(|i| (i as f64 + 1.0) / 945.0).collect();
        for weights in [None, Some([0.0, 1.0]), Some([0.25, 0.75])] {
            let mut scorer =
                crate::BakeScorer::ensemble(&models, weights.as_ref().map(|v| v.as_slice()))
                    .unwrap();
            let mut reads = vec![false; 944];
            scorer.mark_feature_reads(&mut reads);
            assert_eq!(
                scorer.consumed_feature_ids().unwrap(),
                if weights == Some([0.0, 1.0]) {
                    vec![91]
                } else {
                    vec![13, 52, 91]
                }
            );
            assert_eq!(
                reads.iter().filter(|v| **v).count(),
                if weights == Some([0.0, 1.0]) { 1 } else { 3 }
            );
            let mut exhaustive = vec![0.; 944];
            let w = weights.unwrap_or([0.5, 0.5]);
            exhaustive[13] = w[0] * f64::from(-0.7_f32);
            exhaustive[52] = w[0] * f64::from(-1.3_f32);
            exhaustive[91] = w[1];
            for parallel in [false, true] {
                scorer.parallel = parallel;
                let skipped = scorer
                    .score_features_fd_gradient(&row, 96, 96, Some("jxl"))
                    .unwrap();
                assert_eq!(skipped, exhaustive);
            }
            row[900] = f64::MAX;
            assert!(
                scorer
                    .score_features_fd_gradient(&row, 96, 96, None)
                    .is_err()
            );
            row[900] = 901.0 / 945.0;
        }
    }

    /// A minimal one-input identity bake reading feature `id`, optionally
    /// declaring an arithmetic revision. Same recipe shape the cross-tier
    /// attribution gate uses; the campaign bakes are 500 KB+ external
    /// artifacts, so the committed contract tests build their own.
    fn bake_declaring(revision: Option<&str>, id: usize) -> Vec<u8> {
        let mut metadata = vec![serde_json::json!({
            "key": "zentrain.feature_ids", "type": "utf8", "text": id.to_string()
        })];
        if let Some(rev) = revision {
            metadata.push(serde_json::json!({
                "key": "zentrain.formula_revision", "type": "utf8", "text": rev
            }));
        }
        let recipe = serde_json::json!({
            "schema_hash": 1, "scaler_mean": [0.0], "scaler_scale": [1.0],
            "metadata": metadata,
            "layers": [{"in_dim":1,"out_dim":1,"activation":"identity",
                        "dtype":"f32","weights":[1.0],"biases":[0.0]}]
        });
        zenpredict_bake::bake_from_json_str(&recipe.to_string()).expect("bake the recipe")
    }

    fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut src = vec![[200u8, 190, 180]; w * h];
        let mut dst = vec![[200u8, 190, 180]; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if (x / 5 + y / 7) % 3 == 0 {
                    src[i] = [30, 40, 50];
                }
                let t = ((x * 7 + y * 11) % 5) as i32 - 2;
                for c in 0..3 {
                    dst[i][c] = (src[i][c] as i32 + t * 3).clamp(0, 255) as u8;
                }
            }
        }
        (src, dst)
    }

    /// The premise the mismatch test below rests on, asserted rather than
    /// assumed: revisions 2 and 3 select the SAME luminance form. A refusal
    /// built on comparing luminance forms would wave this pair straight
    /// through, which is exactly why `check_pixel_revision` compares the
    /// REVISION and not only the form it selects.
    #[test]
    fn revisions_two_and_three_select_the_same_luminance_form() {
        assert_eq!(
            SsimLumaForm::for_revision(FormulaRevision::Rev2),
            SsimLumaForm::for_revision(FormulaRevision::Rev3),
            "the two-revisions-one-form case this contract exists for is gone; \
             re-derive the mismatch control against a pair that still shares a form"
        );
        assert_eq!(
            SsimLumaForm::for_revision(FormulaRevision::Rev3),
            SsimLumaForm::Clamp
        );
    }

    /// **Bake/process revision mismatch is refused — including between two
    /// revisions that share the Clamp luminance form.**
    ///
    /// A Rev2 bake served by a Rev3 process would read Rev2 coefficients
    /// against Rev3 pixels. Both select `Clamp`, so the pre-existing
    /// luminance-form comparison could not see the difference; the revision
    /// comparison can.
    #[test]
     // `bake_declaring` fixes a v2 extraction requirement
    fn rev3_process_refuses_a_rev2_bake_despite_the_shared_clamp_form() {
        if !run_at_revision(
            "3",
            "metric::bake::revision_contract_tests::rev3_process_refuses_a_rev2_bake_despite_the_shared_clamp_form",
            "REV3-BAKE-MISMATCH-RAN",
        ) {
            return;
        }
        assert_eq!(crate::ssim_form::active_revision(), FormulaRevision::Rev3);
        let (w, h) = (96usize, 96usize);
        let (src, dst) = pair(w, h);
        let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));

        for declared in ["1", "2"] {
            let bytes = bake_declaring(Some(declared), 400);
            let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
            let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
            let err = scorer.compute(&rs, &ds, None).expect_err(
                "a mismatched-revision bake must not be served by a revision-3 process",
            );
            assert!(
                matches!(err, ZensimError::ModelLoadFailed { .. }),
                "declared {declared}: expected a load refusal, got {err:?}"
            );
        }

        // The matching revision is served, so the refusal is about agreement
        // and not about revision 3 being unservable.
        let bytes = bake_declaring(Some("3"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
        let score = scorer
            .compute(&rs, &ds, None)
            .expect("a revision-3 bake is served by a revision-3 process");
        assert!(
            score.score().is_finite(),
            "served bake produced a non-finite score: {}",
            score.score()
        );
        println!("REV3-BAKE-MISMATCH-RAN");
    }

    /// The same contract from the other side: the SHIPPED process refuses a
    /// revision-3 bake. An old bake relabelled `3` therefore cannot be served
    /// as if it had been refit, and a genuine revision-3 bake cannot be
    /// served against revision-1 pixels.
    #[test]
     // `bake_declaring` fixes a v2 extraction requirement
    fn the_shipped_process_refuses_a_rev3_bake() {
        if crate::ssim_form::active_revision() != crate::ssim_form::SHIPPED_REVISION {
            return;
        }
        let (w, h) = (96usize, 96usize);
        let (src, dst) = pair(w, h);
        let bytes = bake_declaring(Some("3"), 400);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
        let err = scorer
            .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h), None)
            .expect_err("the shipped process must not serve a revision-3 bake");
        assert!(
            matches!(err, ZensimError::ModelLoadFailed { .. }),
            "{err:?}"
        );
    }

    /// An undeclared bake is the registered pre-stamp era, and a bake naming
    /// a revision this build does not know is refused at LOAD — never quietly
    /// treated as the default.
    #[test]
    fn undeclared_is_the_shipped_era_and_an_unknown_revision_is_refused() {
        let bytes = bake_declaring(None, 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        assert_eq!(
            crate::feature_layout::formula_revision(&model).expect("undeclared resolves"),
            crate::ssim_form::SHIPPED_REVISION
        );
        // `4` is registered since featcanon (Rev4, the `tiercanon` era), `5`
        // since rev5 (the `localwin` era), `6` since the by-construction
        // `ringhpm` era and `7` since the `rowlanes` era; the first
        // unregistered value is `8`.
        let bytes = bake_declaring(Some("4"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        assert_eq!(
            crate::feature_layout::formula_revision(&model).expect("4 resolves"),
            crate::feature_defs::FormulaRevision::Rev4
        );
        let bytes = bake_declaring(Some("5"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        assert_eq!(
            crate::feature_layout::formula_revision(&model).expect("5 resolves"),
            crate::feature_defs::FormulaRevision::Rev5
        );
        let bytes = bake_declaring(Some("6"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        assert_eq!(
            crate::feature_layout::formula_revision(&model).expect("6 resolves"),
            crate::feature_defs::FormulaRevision::Rev6
        );
        let bytes = bake_declaring(Some("7"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        assert_eq!(
            crate::feature_layout::formula_revision(&model).expect("7 resolves"),
            crate::feature_defs::FormulaRevision::Rev7
        );
        let bytes = bake_declaring(Some("8"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        assert!(
            crate::feature_layout::formula_revision(&model).is_err(),
            "an unregistered revision must be refused, not defaulted"
        );
        assert!(
            crate::feature_v2::bake_formula_revision(&model).is_err(),
            "the feature layer must refuse it too, not default to the shipped era"
        );
    }
}

