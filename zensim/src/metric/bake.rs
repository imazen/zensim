//! Reusable candidate serving surface. Inference and metadata live in the
//! parent metric owner, shared with every named profile.

use super::*;
use std::sync::Arc;

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
    #[cfg(feature = "feature-regime-v2")]
    pixel_scratch: crate::feature_v2::V2Scratch,
    disposition: Option<&'a ProfileParams>,
    members: Vec<BakeScorer<'a>>,
    weights: Option<Vec<f64>>,
    #[cfg(feature = "corruption-head")]
    corruption: Option<Companion<'a>>,
}

#[cfg(feature = "corruption-head")]
enum Companion<'a> {
    Tree(&'a crate::corruption_head::CorruptionHead, f64),
    Linear(Box<BakeScorer<'a>>, f64),
}

impl<'a> BakeScorer<'a> {
    /// Validate a parsed model and prepare reusable inference buffers.
    ///
    /// # Errors
    /// Returns [`ZensimError::ModelLoadFailed`] for malformed or mutually
    /// incompatible head metadata, or an invalid explicit feature declaration.
    pub fn new(model: &'a crate::mlp::Model) -> Result<Self, ZensimError> {
        let scorer = Self::with_metadata(model, Arc::new(parse_bake_metadata(model)?))?;
        scorer.check_servable()?;
        Ok(scorer)
    }

    pub(super) fn with_metadata(
        model: &'a crate::mlp::Model,
        metadata: Arc<ScoreMetadata>,
    ) -> Result<Self, ZensimError> {
        crate::feature_layout::formula_revision(model)?;
        if model.metadata().get("zentrain.feature_ids").is_some()
            && crate::feature_layout::declared_ids(model)
                .is_none_or(|ids| ids.len() != model.caller_input_width())
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
            #[cfg(feature = "feature-regime-v2")]
            pixel_scratch: crate::feature_v2::V2Scratch::default(),
            disposition: None,
            members: Vec::new(),
            weights: None,
            #[cfg(feature = "corruption-head")]
            corruption: None,
        })
    }

    /// The validated wire metadata used by model-author diagnostics.
    #[doc(hidden)]
    pub fn metadata(&self) -> &crate::bake_metadata::ScoreMetadata {
        &self.metadata
    }

    /// Remove the output spline and codec affine for calibration fitting.
    /// The fitted artifact must be evaluated again with its complete metadata.
    #[doc(hidden)]
    pub fn without_output_calibration(mut self) -> Self {
        let md = Arc::make_mut(&mut self.metadata);
        md.output_spline = None;
        md.per_codec_calibration = None;
        self
    }

    /// Diagnostic tail for a perturbed network output (feature contribution).
    /// Uses this bake's exact head, pin, spline and codec calibration. This
    /// cannot stand in for the complete model's evaluated score.
    ///
    /// # Errors
    /// Refuses min-max replacement heads and composed models, or wrong output width.
    #[doc(hidden)]
    pub fn score_network_output(
        &self,
        out: &[f32],
        codec_hint: Option<&str>,
    ) -> Result<f64, ZensimError> {
        if self.metadata.minmax_head.is_some()
            || !self.members.is_empty()
            || out.len() != self.model.n_outputs()
        {
            return Err(ZensimError::ModelForwardFailed {
                reason: "network-output diagnostic requires one MLP and its full output width",
            });
        }
        #[cfg(feature = "corruption-head")]
        if self.corruption.is_some() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "network-output diagnostic cannot apply a feature-dependent corruption gate",
            });
        }
        let affine = self
            .metadata
            .per_codec_calibration
            .as_ref()
            .and_then(|c| codec_hint.and_then(|hint| lookup_per_codec_affine(c, hint)));
        let raw = finish_network_output(
            out,
            self.metadata.per_sample_alpha.as_deref(),
            self.metadata.hybrid_head.as_deref(),
            self.metadata.tanh_pin_scale,
            self.metadata.output_spline.as_deref(),
            affine,
        )?;
        Ok(self.disposition.map_or(raw, |p| dispose_mlp_raw(raw, p)))
    }

    /// Serve an equal mean or a convex blend of calibrated member scores.
    /// Members are accumulated in their supplied order. A single member with
    /// no weights is exactly [`Self::new`], with no extra arithmetic.
    ///
    /// # Errors
    /// Requires at least one valid model. Explicit weights must be finite,
    /// nonnegative, match the member count and sum to one within `1e-12`.
    pub fn ensemble(
        models: &'a [crate::mlp::Model],
        weights: Option<&[f64]>,
    ) -> Result<Self, ZensimError> {
        let Some((first, rest)) = models.split_first() else {
            return Err(ZensimError::ModelLoadFailed {
                reason: "ensemble has no models",
            });
        };
        if let Some(w) = weights
            && (w.len() != models.len()
                || w.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (w.iter().sum::<f64>() - 1.0).abs() > 1e-12)
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "invalid ensemble weights",
            });
        }
        let mut scorer = Self::new(first)?;
        scorer.members = rest.iter().map(Self::new).collect::<Result<_, _>>()?;
        scorer.weights = weights.map(<[f64]>::to_vec);
        scorer.check_servable()?;
        Ok(scorer)
    }

    /// Attach a Rust tree corruption head. Its baked deadband is used unless
    /// the model author explicitly supplies an override. The returned score
    /// includes the gate on every scoring surface.
    ///
    /// # Errors
    /// The deadband must be finite and within 0–100 score units.
    #[cfg(feature = "corruption-head")]
    pub fn with_corruption_head(
        mut self,
        head: &'a crate::corruption_head::CorruptionHead,
        deadband: Option<f64>,
    ) -> Result<Self, ZensimError> {
        let t = deadband.unwrap_or_else(|| head.deadband_score());
        check_deadband(t)?;
        self.corruption = Some(Companion::Tree(head, t));
        self.check_servable()?;
        Ok(self)
    }

    /// Attach the historical ZNPR corruption head with an explicit deadband.
    /// Its own head and spline execute through this same Rust surface.
    ///
    /// # Errors
    /// Rejects malformed head models or a nonfinite/out-of-range deadband.
    #[cfg(feature = "corruption-head")]
    pub fn with_linear_corruption_head(
        mut self,
        model: &'a crate::mlp::Model,
        deadband: f64,
    ) -> Result<Self, ZensimError> {
        check_deadband(deadband)?;
        self.corruption = Some(Companion::Linear(Box::new(Self::new(model)?), deadband));
        self.check_servable()?;
        Ok(self)
    }

    /// Select the final score disposition using the existing profile contract.
    /// Only mapping/clamping fields are used; the bake and its heads remain
    /// those passed to [`Self::new`].
    ///
    /// # Errors
    /// A calibrated spline cannot also be treated as an uncalibrated distance.
    pub fn with_score_disposition(
        mut self,
        params: &'a ProfileParams,
    ) -> Result<Self, ZensimError> {
        if (self.metadata.output_spline.is_some()
            || self
                .members
                .iter()
                .any(|m| m.metadata.output_spline.is_some()))
            && !params.skip_score_mapping
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "a calibrated spline cannot also use distance mapping",
            });
        }
        self.disposition = Some(params);
        Ok(self)
    }

    /// Score extracted features with independently verified pixel identity.
    ///
    /// Set `pixels_identical` only after proving byte-identical decoded pixels
    /// in the same encoding and geometry. A zero feature row, equal source
    /// names or a perceptual hash is not such proof. True returns exactly 100
    /// before inference, matching [`Self::compute`] and [`Self::compute_hdr`].
    /// False delegates to [`Self::score_features`], including the complete
    /// configured model, calibration, ensemble and corruption composition.
    ///
    /// This entry supports cached pair records that preserve identity evidence.
    /// Call [`Self::score_features`] when only the feature row is known.
    ///
    /// # Errors
    /// For nonidentical pairs, returns the same errors as [`Self::score_features`].
    pub fn score_features_with_identity(
        &mut self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
        pixels_identical: bool,
    ) -> Result<f64, ZensimError> {
        if pixels_identical {
            Ok(100.0)
        } else {
            self.score_features(features, width, height, codec_hint)
        }
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
        #[cfg(feature = "corruption-head")]
        if let Some(head) = self.corruption.as_mut() {
            let (value, threshold) = match head {
                Companion::Tree(h, t) => {
                    let row = features.get(..h.caller_input_width()).ok_or(
                        ZensimError::ModelForwardFailed {
                            reason: "feature row is shorter than the corruption head's input",
                        },
                    )?;
                    (
                        h.score_f64(row)
                            .map_err(|_| ZensimError::ModelForwardFailed {
                                reason: "corruption head could not score the feature row",
                            })?,
                        *t,
                    )
                }
                Companion::Linear(h, t) => {
                    (h.score_features(features, width, height, codec_hint)?, *t)
                }
            };
            return Ok(crate::corruption_head::gate_score(score, value, threshold));
        }
        Ok(score)
    }

    /// Local feature sensitivities of this complete candidate's served score.
    ///
    /// The result uses the input row's identity layout, including zero
    /// sensitivity for unread IDs. Each component is the central difference
    /// of [`Self::score_features`] with step `max(abs(feature) * 1e-3, 1e-5)`.
    /// Predictor buffers are reused within each worker. Unread IDs skip their
    /// forwards; threaded builds evaluate large read sets in independent
    /// column groups with unchanged per-forward arithmetic. All heads,
    /// splines, codec calibration, ensemble members, final disposition and
    /// corruption gates participate; negative scores are preserved.
    ///
    /// These are local finite secants. A tree boundary, clamp or other
    /// nonsmooth head need not have a derivative, and a zero local sensitivity
    /// does not prove that a finite pixel intervention has no effect. Pixel
    /// attribution must also account for the feature integrands it supports.
    /// Dimensions and codec hint stay fixed for all probes. The pixel-identity
    /// override belongs to [`Self::compute`], not an arbitrary feature row.
    ///
    /// # Errors
    /// Rejects a nonfinite row, base score, perturbed input or probe result,
    /// and propagates model/feature-layout errors without replacing them with
    /// zero sensitivities.
    pub fn score_features_fd_gradient(
        &mut self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Result<Vec<f64>, ZensimError> {
        let invalid = || ZensimError::ModelForwardFailed {
            reason: "candidate sensitivity requires finite inputs, scores and probes",
        };
        if features.iter().any(|f| !f.is_finite())
            || !self
                .score_features(features, width, height, codec_hint)?
                .is_finite()
        {
            return Err(invalid());
        }
        // Validate every requested perturbation before skipping an unread ID.
        // The shortcut cannot hide overflow in an otherwise unused column.
        for &value in features {
            let eps = (value.abs() * 1e-3).max(1e-5);
            if !(value + eps).is_finite() || !(value - eps).is_finite() {
                return Err(invalid());
            }
        }
        let mut reads = vec![false; features.len()];
        self.mark_feature_reads(&mut reads);
        let indices: Vec<usize> = reads
            .iter()
            .enumerate()
            .filter_map(|(i, read)| read.then_some(i))
            .collect();
        let mut values = vec![0.0; indices.len()];
        #[cfg(feature = "threads")]
        let parallel = if indices.len() >= 128 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            let chunk = indices.len().div_ceil(rayon::current_num_threads().min(8));
            values
                .par_chunks_mut(chunk)
                .enumerate()
                .try_for_each(|(i, out)| {
                    let mut worker = self.fork_predictor_state()?;
                    worker.fd_gradient_into(
                        features,
                        &indices[i * chunk..i * chunk + out.len()],
                        out,
                        (width, height),
                        codec_hint,
                    )
                })?;
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let parallel = false;
        if !parallel {
            self.fd_gradient_into(features, &indices, &mut values, (width, height), codec_hint)?;
        }
        let mut gradient = vec![0.0; features.len()];
        for (id, value) in indices.into_iter().zip(values) {
            gradient[id] = value;
        }
        Ok(gradient)
    }

    // Conservative declaration-based read set. Do not infer zero influence
    // from an observed zero gradient, input value or layer coefficient.
    fn mark_feature_reads(&self, reads: &mut [bool]) {
        if self.weights.as_ref().is_none_or(|w| w[0] != 0.0) {
            for pos in 0..self.layout.width() {
                if let Some(id) = self.layout.slot_at(pos)
                    && let Some(read) = reads.get_mut(usize::from(id))
                {
                    *read = true;
                }
            }
        }
        for (i, member) in self.members.iter().enumerate() {
            if self.weights.as_ref().is_none_or(|w| w[i + 1] != 0.0) {
                member.mark_feature_reads(reads);
            }
        }
        #[cfg(feature = "corruption-head")]
        if let Some(head) = &self.corruption {
            match head {
                Companion::Tree(h, _) => {
                    for &id in h.declared_feature_ids() {
                        if let Some(read) = reads.get_mut(usize::from(id)) {
                            *read = true;
                        }
                    }
                }
                Companion::Linear(h, _) => h.mark_feature_reads(reads),
            }
        }
    }

    // Each parallel column group owns its predictor buffers. Model bytes and
    // parsed metadata stay shared; no pixel scratch or stale numeric state is
    // copied. Reconstruct the whole composition, including its disposition.
    #[cfg(feature = "threads")]
    fn fork_predictor_state(&self) -> Result<Self, ZensimError> {
        let mut fork = Self::with_metadata(self.model, Arc::clone(&self.metadata))?;
        fork.disposition = self.disposition;
        fork.weights = self.weights.clone();
        fork.members = self
            .members
            .iter()
            .map(Self::fork_predictor_state)
            .collect::<Result<_, _>>()?;
        #[cfg(feature = "corruption-head")]
        {
            fork.corruption = match &self.corruption {
                Some(Companion::Tree(h, t)) => Some(Companion::Tree(h, *t)),
                Some(Companion::Linear(h, t)) => {
                    Some(Companion::Linear(Box::new(h.fork_predictor_state()?), *t))
                }
                None => None,
            };
        }
        Ok(fork)
    }

    fn fd_gradient_into(
        &mut self,
        features: &[f64],
        indices: &[usize],
        output: &mut [f64],
        dimensions: (u32, u32),
        codec_hint: Option<&str>,
    ) -> Result<(), ZensimError> {
        let mut probe = features.to_vec();
        for (&k, result) in indices.iter().zip(output) {
            let value = features[k];
            let eps = (value.abs() * 1e-3).max(1e-5);
            probe[k] = value + eps;
            let up = self.score_features(&probe, dimensions.0, dimensions.1, codec_hint)?;
            probe[k] = value - eps;
            let down = self.score_features(&probe, dimensions.0, dimensions.1, codec_hint)?;
            probe[k] = value;
            let sensitivity = (up - down) / (2.0 * eps);
            if !up.is_finite() || !down.is_finite() || !sensitivity.is_finite() {
                return Err(ZensimError::ModelForwardFailed {
                    reason: "candidate sensitivity requires finite inputs, scores and probes",
                });
            }
            *result = sensitivity;
        }
        Ok(())
    }

    #[cfg(feature = "feature-regime-v2")]
    fn plan(&self) -> Result<crate::feature_plan::Plan, ZensimError> {
        use crate::feature_plan::Plan;
        let mut combined: Option<Plan> = None;
        for (i, model) in std::iter::once(self.model)
            .chain(self.members.iter().map(|m| m.model))
            .enumerate()
        {
            if self.weights.as_ref().is_some_and(|w| w[i] == 0.0) {
                continue;
            }
            let other = Plan::for_bake(model).map_err(|_| ZensimError::ModelLoadFailed {
                reason: "bake reads features unavailable to the extraction plan",
            })?;
            combined = Some(match combined {
                Some(plan) => {
                    if !plan.revisions_agree(&other) {
                        return Err(ZensimError::ModelLoadFailed {
                            reason: "ensemble members require different feature revisions",
                        });
                    }
                    plan.union(&other)
                }
                None => other,
            });
        }
        let mut plan = combined.ok_or(ZensimError::ModelLoadFailed {
            reason: "ensemble has no active member",
        })?;
        // Emit an identity row reaching all materialized slots, including free
        // slots a corruption companion may use. No second extraction is run.
        plan = Plan::widened_to_identity(&plan, plan.walk_width().max(372));
        #[cfg(feature = "corruption-head")]
        if let Some(companion) = &self.corruption {
            let needed = match companion {
                Companion::Tree(h, _) => crate::feature_set_id::SlotSet::from_slots(
                    h.declared_feature_ids().iter().map(|&id| usize::from(id)),
                ),
                Companion::Linear(h, _) => {
                    let p = h.plan()?;
                    if !plan.revisions_agree(&p) {
                        return Err(ZensimError::ModelLoadFailed {
                            reason: "corruption head requires another feature revision",
                        });
                    }
                    crate::feature_plan::bake_read_slots(h.model).ok_or(
                        ZensimError::ModelLoadFailed {
                            reason: "corruption head has no readable feature declaration",
                        },
                    )?
                }
            };
            if !plan.covers(&needed) {
                return Err(ZensimError::ModelLoadFailed {
                    reason: "corruption head reads features not computed by the model's extraction plan",
                });
            }
            let width = match companion {
                Companion::Tree(h, _) => h.caller_input_width(),
                Companion::Linear(h, _) => h.layout.walk_width(),
            };
            plan = Plan::widened_to_identity(&plan, plan.walk_width().max(width));
        }
        Ok(plan)
    }

    fn check_servable(&self) -> Result<(), ZensimError> {
        #[cfg(feature = "feature-regime-v2")]
        {
            self.plan()?;
        }
        #[cfg(not(feature = "feature-regime-v2"))]
        {
            if self.layout.walk_width() > 372
                || crate::feature_layout::formula_revision(self.model)?
                    != crate::ssim_form::active_revision()
            {
                return Err(ZensimError::ModelLoadFailed {
                    reason: "bake requires feature-regime-v2 for its extraction requirements",
                });
            }
            for member in &self.members {
                member.check_servable()?;
            }
            #[cfg(feature = "corruption-head")]
            if let Some(head) = &self.corruption {
                match head {
                    Companion::Tree(h, _) if h.caller_input_width() > 372 => {
                        return Err(ZensimError::ModelLoadFailed {
                            reason: "corruption head requires feature-regime-v2",
                        });
                    }
                    Companion::Linear(h, _) => h.check_servable()?,
                    _ => (),
                }
            }
        }
        Ok(())
    }

    // The global-contrast finalizer is per-plan. The existing SSIM kernels
    // still select their luminance form per process; refuse a mixed formula
    // instead of claiming that a partially honored revision is correct.
    fn check_pixel_revision(&self) -> Result<(), ZensimError> {
        let model = std::iter::once(self.model)
            .chain(self.members.iter().map(|m| m.model))
            .enumerate()
            .find(|(i, _)| self.weights.as_ref().is_none_or(|w| w[*i] != 0.0))
            .map(|(_, m)| m)
            .ok_or(ZensimError::ModelLoadFailed {
                reason: "ensemble has no active member",
            })?;
        let revision = crate::feature_layout::formula_revision(model)?;
        if crate::ssim_form::active_luma_form()
            != crate::ssim_form::SsimLumaForm::for_revision(revision)
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "pixel kernels use another formula revision; set ZENSIM_FORMULA_REV to the bake's declared revision before starting the process",
            });
        }
        Ok(())
    }

    /// Score declared HDR pixels through the canonical PU front end.
    /// `encoding` supplies the display model; supported containers and alpha
    /// modes are those of [`Zensim::compute_folded720_features_hdr`].
    /// The bake's revision and extraction requirements are honored, and the
    /// same complete head/spline/composition as `score_features` is returned.
    ///
    /// # Errors
    /// Rejects invalid dimensions, unsupported HDR formats, or model failures.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_hdr(
        &mut self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        encoding: crate::feature_v2::HdrEncoding,
        codec_hint: Option<&str>,
    ) -> Result<f64, ZensimError> {
        self.check_pixel_revision()?;
        let plan = self.plan()?;
        let features = crate::feature_v2::compute_folded720_hdr_streaming_impl(
            source,
            distorted,
            encoding,
            Some(120_000_000),
            true,
            plan.toggles(),
            &mut self.pixel_scratch,
        )?;
        self.score_features_with_identity(
            features.features(),
            source.width() as u32,
            source.height() as u32,
            codec_hint,
            images_byte_identical(source, distorted),
        )
    }

    /// Score an SDR image pair through the canonical pixel front end and the
    /// same inference method as [`Self::score_features`]. Identical pixels
    /// return exactly 100 before model inference, as in [`Zensim::compute`].
    ///
    /// # Errors
    /// Rejects invalid dimensions, HDR input, unservable feature requirements
    /// or a failed model forward. The normal 120-million-pixel limit applies.
    /// SSIM kernels currently require the process `ZENSIM_FORMULA_REV` to
    /// match the bake; a mismatch refuses before extraction.
    pub fn compute(
        &mut self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        codec_hint: Option<&str>,
    ) -> Result<ZensimResult, ZensimError> {
        self.check_pixel_revision()?;
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), Some(120_000_000))?;
        // B supplies the established four-scale SDR pixel front end. The
        // candidate's own plan determines which feature families run; B's
        // weights only fill the intermediate result and are never returned.
        let params = ZensimProfile::B.params();
        let config = config_from_params(params, true);
        #[cfg(feature = "feature-regime-v2")]
        let plan = self.plan()?;
        #[cfg(not(feature = "feature-regime-v2"))]
        if self.layout.walk_width() > 372 {
            return Err(ZensimError::ModelLoadFailed {
                reason: "this bake requires feature-regime-v2 for image extraction",
            });
        }
        let mut result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            None,
            true,
            #[cfg(feature = "feature-regime-v2")]
            Some(&plan),
        );
        result.score = self.score_features_with_identity(
            result.features(),
            source.width() as u32,
            source.height() as u32,
            codec_hint,
            result.is_identical(),
        )?;
        Ok(result)
    }

    /// Cache the SDR reference for [`Self::compute_with_ref_and_attribution`].
    /// Reuse this cache for comparisons against the same source image.
    ///
    /// # Errors
    /// Refuses a formula mismatch, HDR input or invalid source dimensions.
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    pub fn precompute_reference(
        &self,
        source: &impl ImageSource,
    ) -> Result<crate::PrecomputedReference, ZensimError> {
        self.check_pixel_revision()?;
        Zensim::new(ZensimProfile::B).precompute_reference(source)
    }

    /// Compare SDR pixels and spatialize this complete candidate's score.
    ///
    /// Uses the same declared feature plan and complete scoring composition
    /// as [`Self::compute`], with retained extraction buffers and a cached
    /// reference. No caller-supplied or first-image gradient is used.
    /// `precomputed` must come from [`Self::precompute_reference`] for this
    /// same `source`; dimensions are checked, source identity is the caller's
    /// contract. Reuse `session` across comparisons to reuse scratch buffers.
    ///
    /// `bin` sets the map grid in source pixels. Aligned rectangle integrals
    /// remain exact for that density; unaligned queries interpolate within
    /// bins. Inspect the result's unsupported feature IDs and corruption-gate
    /// flag: local finite sensitivities and supported integrands do not prove
    /// accuracy for a finite pixel edit. Identity returns score 100 and a zero
    /// map. Negative scores retain their original scale.
    /// The candidate map includes the L8 terms in f156-227, with the same
    /// removal-based moment linearization as L2/L4. Hard maxima and masked/IW
    /// pools remain explicitly unsupported; large removals retain root-curvature
    /// and blur-neighborhood approximation errors.
    ///
    /// # Errors
    /// Refuses `bin == 0`, invalid inputs/cache dimensions, HDR, a formula
    /// mismatch, or failed/nonfinite candidate scoring and sensitivities.
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_with_ref_and_attribution(
        &mut self,
        source: &impl ImageSource,
        precomputed: &crate::PrecomputedReference,
        distorted: &impl ImageSource,
        codec_hint: Option<&str>,
        session: &mut crate::Fused944Session,
        bin: usize,
    ) -> Result<crate::ScoredAttribution, ZensimError> {
        if bin == 0 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "attribution bin must be nonzero",
            });
        }
        self.check_pixel_revision()?;
        validate_pair(source, distorted)?;
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(source.width(), source.height(), Some(120_000_000))?;
        let plan = self.plan()?;
        let params = ZensimProfile::B.params();
        let config = config_from_params(params, true);
        #[cfg(feature = "corruption-head")]
        let has_corruption_gate = self.corruption.is_some();
        #[cfg(not(feature = "corruption-head"))]
        let has_corruption_gate = false;
        if images_byte_identical(source, distorted) {
            let result = identical_result_at(&config, plan.walk_width());
            return Ok(crate::ScoredAttribution {
                sensitivities: vec![0.0; result.features().len()],
                result,
                attribution: crate::attribution::zero_attribution(
                    source.width(),
                    source.height(),
                    bin,
                ),
                unsupported_feature_ids: Vec::new(),
                has_corruption_gate,
            });
        }
        let (mut features, mean_offset) = session.planned_features(source, distorted, &plan)?;
        features.truncate(
            plan.walk_width()
                .max(crate::fold_engine::v1_feature_width(&config)),
        );
        let (_, raw_distance) =
            score_v1_layout_features(&mut features, params.weights, &config, config.num_scales);
        let score = self.score_features(
            &features,
            source.width() as u32,
            source.height() as u32,
            codec_hint,
        )?;
        let sensitivities = self.score_features_fd_gradient(
            &features,
            source.width() as u32,
            source.height() as u32,
            codec_hint,
        )?;
        let (spatial, unsupported_feature_ids) =
            crate::attribution::candidate_map_sensitivities(&plan, &sensitivities);
        let (_, attribution) = Zensim::new(ZensimProfile::B).attribution_from_retention_binned(
            precomputed,
            distorted,
            &spatial,
            spatial.get(156..spatial.len().min(228)).unwrap_or(&[]),
            session,
            bin,
        )?;
        Ok(crate::ScoredAttribution {
            result: ZensimResult::new(score, raw_distance, features, ZensimProfile::B, mean_offset),
            attribution,
            sensitivities,
            unsupported_feature_ids,
            has_corruption_gate,
        })
    }
}

#[cfg(feature = "corruption-head")]
fn check_deadband(t: f64) -> Result<(), ZensimError> {
    if t.is_finite() && (0.0..=100.0).contains(&t) {
        Ok(())
    } else {
        Err(ZensimError::ModelLoadFailed {
            reason: "corruption deadband must be finite and in 0..=100",
        })
    }
}
