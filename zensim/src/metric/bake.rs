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
    parallel: bool,
    #[cfg(feature = "feature-regime-v2")]
    finite_moments: bool,
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

    /// Enable internal parallel work (the default), or disable it when the
    /// caller parallelizes independent comparisons.
    #[must_use]
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Opt into finite L2/L4/L8 moment removal for rectangle refinement.
    ///
    /// Disabled by default. Scalar scores and additive density are unchanged.
    /// Retains a binned integral per active root feature, using the requested
    /// attribution bin; fine bins increase memory and preparation cost.
    /// Predictions freeze the current signals and model sensitivities: actual
    /// repairs can change neighboring windows, gates and nonlinear heads.
    #[cfg(feature = "feature-regime-v2")]
    #[must_use]
    pub fn with_finite_moment_refinement(mut self, enabled: bool) -> Self {
        self.finite_moments = enabled;
        self
    }

    /// Bind a source, its cache and reusable scratch for repeated SDR steering.
    ///
    /// Accepts bakes that read basic and peak features (f0-f227), v2 features (f372-f719) and append/append2
    /// features (f720-f943), SDR only beyond f227; the maps (including exact BLOCKINESS terms for aligned
    /// rectangle repairs) come from the same owner as [`Self::compute_with_ref_and_attribution`]. Masked/IW
    /// (f228-f371), f944 and above, and any read without a complete integrand are refused up front, as is a bake whose
    /// corruption companion reads such an ID; the error names the family's ID range. No scored term is
    /// dropped to make a partial map. Coverage is not a quality certification: models still need finite-edit
    /// and codec-level validation.
    /// `bin` is the grid spacing in source pixels; rectangle semantics match
    /// [`Self::compute_with_ref_and_attribution`]. Negative scores are preserved.
    ///
    /// # Example
    /// ```no_run
    /// # fn example(model: &zenpredict::Model) -> Result<(), zensim::ZensimError> {
    /// let source_pixels = vec![[128u8; 3]; 96 * 96];
    /// let decoded_pixels = vec![[120u8; 3]; 96 * 96];
    /// let source = zensim::RgbSlice::new(&source_pixels, 96, 96);
    /// let decoded = zensim::RgbSlice::new(&decoded_pixels, 96, 96);
    /// let mut scorer = zensim::BakeScorer::new(model)?.with_parallel(false);
    /// let mut worker = scorer.prepare_steering(&source, 8)?;
    /// let comparison = worker.compute(&decoded, Some("jxl"))?;
    /// let score = comparison.result().score();
    /// let expected_gain = comparison.refinement_gain(0, 0, 32, 32);
    /// # let _ = (score, expected_gain);
    /// # Ok(()) }
    /// ```
    ///
    /// # Errors
    /// Refuses zero bins, invalid sources, unsupported feature families (see above) or
    /// incompatible arithmetic contracts, or unsupported companion features.
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    pub fn prepare_steering<'s, S: ImageSource>(
        &'s mut self,
        source: &'s S,
        bin: usize,
    ) -> Result<SteeringSession<'s, 'a, S>, ZensimError> {
        self.prepare_steering_input(source, bin, None)
    }

    /// Bind native HDR input and viewing parameters for repeated spatial steering.
    /// Uses the same basic/peak features, inference and map composition as SDR; bakes reading v2 features are
    /// refused here (SDR only).
    /// The encoding applies to both images; decoded primaries remain authoritative.
    ///
    /// # Errors
    /// Refuses invalid HDR input/display parameters, unsupported sampling or feature
    /// families, and incompatible model/companion arithmetic. No SDR conversion occurs.
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    pub fn prepare_steering_hdr<'s, S: ImageSource>(
        &'s mut self,
        source: &'s S,
        encoding: crate::feature_v2::HdrEncoding,
        bin: usize,
    ) -> Result<SteeringSession<'s, 'a, S>, ZensimError> {
        self.prepare_steering_input(source, bin, Some(encoding))
    }

    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn prepare_steering_input<'s, S: ImageSource>(
        &'s mut self,
        source: &'s S,
        bin: usize,
        encoding: Option<crate::feature_v2::HdrEncoding>,
    ) -> Result<SteeringSession<'s, 'a, S>, ZensimError> {
        if bin == 0 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "steering bin must be nonzero",
            });
        }
        // The complete served read set (active ensemble members and corruption companions included): a map
        // that dropped any scored term would be fabricated, so every read must have a complete integrand.
        let reads = self.consumed_feature_ids()?;
        steering_support(&reads, encoding.is_some())?;
        let reference = if let Some(encoding) = encoding {
            self.check_pixel_revision()?;
            if self.plan()?.compute.sampling.is_some() {
                return Err(ZensimError::HdrInputRequiresPuPath);
            }
            crate::feature_v2::validate_hdr_pair(source, source, encoding, Some(120_000_000))?;
            crate::PrecomputedReference::for_candidate(
                source,
                self.parallel,
                Some(encoding),
                self.plan()?.compute.formula_revision,
            )
        } else {
            self.precompute_reference(source)?
        };
        Ok(SteeringSession {
            encoding,
            scorer: self,
            source,
            reference,
            scratch: crate::Fused944Session::new(),
            bin,
        })
    }

    pub(super) fn with_metadata(
        model: &'a crate::mlp::Model,
        metadata: Arc<ScoreMetadata>,
    ) -> Result<Self, ZensimError> {
        crate::feature_layout::formula_revision(model)?;
        crate::sampling::Sampling::from_model(model)?;
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
            parallel: true,
            #[cfg(feature = "feature-regime-v2")]
            finite_moments: false,
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

    /// Canonical feature IDs structurally consumed by this complete candidate.
    ///
    /// Uses the serving planner and its read-set owner, including active
    /// ensemble members and corruption companions. Zero local sensitivity is
    /// not evidence of an unread input. IDs are sorted and unique; this is an
    /// audit surface, not a model-quality or spatial-coverage certification.
    ///
    /// # Errors
    /// Refuses an unreadable or unservable extraction/composition contract.
    #[doc(hidden)]
    #[cfg(feature = "feature-regime-v2")]
    pub fn consumed_feature_ids(&self) -> Result<Vec<u16>, ZensimError> {
        let plan = self.plan()?;
        let mut reads = vec![false; plan.walk_width()];
        self.mark_feature_reads(&mut reads);
        Ok(reads
            .into_iter()
            .enumerate()
            .filter_map(|(id, read)| read.then_some(id as u16))
            .collect())
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
    /// includes the gate on every scoring surface: below the head's score
    /// threshold, take the minimum of perceptual and head scores; otherwise
    /// preserve the perceptual score. Equality at the threshold is inactive.
    ///
    /// # Errors
    /// The deadband must be finite and within 0–100 score units. The head
    /// must match the base's arithmetic revision and available native features;
    /// legacy ZCTH v1/v2 require revision 1.
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
    /// The gate uses the same thresholded minimum as [`Self::with_corruption_head`],
    /// preserving negative scores from either model.
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
        if let Some((value, threshold)) =
            self.companion_score(features, width, height, codec_hint)?
        {
            return Ok(crate::corruption_head::gate_score(score, value, threshold));
        }
        Ok(score)
    }

    #[cfg(feature = "corruption-head")]
    fn companion_score(
        &mut self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Result<Option<(f64, f64)>, ZensimError> {
        if let Some(head) = self.corruption.as_mut() {
            let value = match head {
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
            Ok(Some(value))
        } else {
            Ok(None)
        }
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
        let parallel = if self.parallel && indices.len() >= 128 && rayon::current_num_threads() > 1
        {
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

    // Reuse the extraction planner's structural read proof, including input
    // transforms and dense IDs. An observed zero gradient is never a proof.
    // Unknown contracts conservatively retain every declared input.
    fn mark_feature_reads(&self, reads: &mut [bool]) {
        if self.weights.as_ref().is_none_or(|w| w[0] != 0.0) {
            #[cfg(feature = "feature-regime-v2")]
            let known = crate::feature_plan::bake_read_slots(self.model)
                .map(|slots| {
                    for id in slots.iter_slots() {
                        if let Some(read) = reads.get_mut(id) {
                            *read = true;
                        }
                    }
                })
                .is_some();
            #[cfg(not(feature = "feature-regime-v2"))]
            let known = false;
            if !known {
                for pos in 0..self.layout.width() {
                    if let Some(id) = self.layout.slot_at(pos)
                        && let Some(read) = reads.get_mut(usize::from(id))
                    {
                        *read = true;
                    }
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
                    if !plan.revisions_agree(&other)
                        || plan.compute.sampling != other.compute.sampling
                    {
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
            if plan.compute.sampling.is_some() {
                return Err(ZensimError::ModelLoadFailed {
                    reason: "sampling variant requires a matching corruption feature contract",
                });
            }
            // What the companion reads, and the plan that computes it. The
            // extraction is the UNION of the base's and the companion's plans
            // (one walk, as for ensemble members): a narrow base plan — a
            // local-only basic subset — does not populate a slot the head
            // reads, and refusing that would make every basic-only bake
            // unable to carry a head that reads the peaks or a later block.
            let (needed, companion_plan) = match companion {
                Companion::Tree(h, _) => {
                    if h.formula_revision() != plan.formula_revision() {
                        return Err(ZensimError::ModelLoadFailed {
                            reason: "corruption head requires another feature revision",
                        });
                    }
                    let needed = crate::feature_set_id::SlotSet::from_slots(
                        h.declared_feature_ids().iter().map(|&id| usize::from(id)),
                    );
                    let own = Plan::derive(&needed, h.caller_input_width().max(372)).map_err(
                        |_| ZensimError::ModelLoadFailed {
                            reason: "corruption head reads features not computed by the model's extraction plan",
                        },
                    )?;
                    (needed, own)
                }
                Companion::Linear(h, _) => {
                    let p = h.plan()?;
                    if !plan.revisions_agree(&p) {
                        return Err(ZensimError::ModelLoadFailed {
                            reason: "corruption head requires another feature revision",
                        });
                    }
                    let needed = crate::feature_plan::bake_read_slots(h.model).ok_or(
                        ZensimError::ModelLoadFailed {
                            reason: "corruption head has no readable feature declaration",
                        },
                    )?;
                    (needed, p)
                }
            };
            if companion_plan.compute.sampling.is_some() {
                return Err(ZensimError::ModelLoadFailed {
                    reason: "sampling variant requires a matching corruption feature contract",
                });
            }
            if !plan.covers(&needed) {
                plan = plan.union(&companion_plan);
            }
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
            if crate::sampling::Sampling::from_model(self.model)?.is_some()
                || self.layout.walk_width() > 372
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
                    Companion::Tree(h, _)
                        if h.formula_revision()
                            != crate::feature_layout::formula_revision(self.model)? =>
                    {
                        return Err(ZensimError::ModelLoadFailed {
                            reason: "corruption head requires another feature revision",
                        });
                    }
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

    // Basic/peak arithmetic is per-model; remaining wide-family kernels
    // require matching process arithmetic. Research bypasses remain explicit.
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
        // featcanon D1, before any exemption: no bake crosses the Rev4
        // boundary (a Rev1–Rev3 bake in a Rev4 process would run the
        // process's canonical Rev4 arithmetic under the bake's declared
        // revision — measured: the D bake, declared Rev1, scored 12/12
        // pairs with Rev4 features; the reverse mix is equally mislabelled).
        // REV4SERVE: a Rev4 bake IS served in a pure Rev4 process — the
        // served leaves are canonical there. No diagnostic bypass.
        crate::ssim_form::refuse_rev4_mix(revision)?;
        // Basic/peak plans carry their arithmetic explicitly through both
        // SIMD passes and the cached spatial owner. Wide-family kernels still
        // use process defaults and must retain the mismatch refusal below.
        #[cfg(feature = "feature-regime-v2")]
        {
            let plan = self.plan()?;
            #[cfg(feature = "corruption-head")]
            let no_companion = self.corruption.is_none();
            #[cfg(not(feature = "corruption-head"))]
            let no_companion = true;
            if no_companion
                && crate::ssim_form::effective_revision(revision) == revision
                && !plan.compute.v2_blocks
                && matches!(
                    plan.compute.v1_pools,
                    crate::feature_v2::V1PoolsMode::Off | crate::feature_v2::V1PoolsMode::Peaks
                )
                && plan.compute.free_extras == crate::feature_v2::V1FreeExtras::Off
                && crate::ssim_form::active_luma_form()
                    == crate::ssim_form::SsimLumaForm::for_revision(
                        crate::ssim_form::active_revision(),
                    )
            {
                return Ok(());
            }
            // REV4SERVE: the process is pinned Rev4 (the mix guard above
            // already refused every cross-boundary case), so every feature
            // the plan computes runs the fold walk's canonical arithmetic.
            // Two things remain outside that envelope: the SAMPLING front
            // end (a different, unproven subset walk — the bake keeps its
            // refusal) and a CORRUPTION COMPANION, whose reads are not part
            // of the bake's declared-id plan coverage.
            if revision >= crate::feature_defs::FormulaRevision::Rev4 {
                if plan.compute.sampling.is_some() {
                    return Err(ZensimError::ModelLoadFailed {
                        reason: "formula revisions 4 and later do not serve sampled bake plans: the subset-extraction front end is not proven tier-canonical",
                    });
                }
                if !no_companion {
                    return Err(ZensimError::ModelLoadFailed {
                        reason: "formula revisions 4 and later do not serve bakes with a corruption companion: companion read coverage is not part of the canonical fold plan",
                    });
                }
                return Ok(());
            }
        }
        if crate::ssim_form::active_revision() != revision
            || crate::ssim_form::active_luma_form()
                != crate::ssim_form::SsimLumaForm::for_revision(revision)
        {
            // The ONLY way past this, and it does not exist in a product
            // build: the `cross-revision-diagnostic` feature must be compiled
            // in AND the environment must ask for it. It exists so an
            // already-fit candidate can be replayed on a corrected extraction
            // before any refit — a measurement, never a score.
            #[cfg(feature = "cross-revision-diagnostic")]
            if crate::ssim_form::cross_revision_diagnostic() {
                crate::ssim_form::warn_cross_revision_once(revision);
                return Ok(());
            }
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
        if plan.compute.sampling.is_some() {
            return Err(ZensimError::ModelLoadFailed {
                reason: "sampling v1 is SDR only; HDR needs a separately validated PU sampling contract",
            });
        }
        let mut features = crate::feature_v2::compute_folded720_hdr_streaming_impl(
            source,
            distorted,
            encoding,
            Some(120_000_000),
            self.parallel,
            plan.toggles(),
            &mut self.pixel_scratch,
            Some(plan.compute),
        )?
        .into_features();
        // Same emit-width rule as the SDR arm: an identity-declared bake can
        // be wider than the emitted regime — the uncomputed tail is zeros.
        // REV4SERVE review F1: every id the plan claims to emit must already
        // be materialized — a slot it promised but the walk skipped must not
        // be zero-filled into a read.
        debug_assert!(
            plan.emit_covered(features.len()),
            "plan emit claims ids past the emitted vector (bound {} > {})",
            plan.emit_bound(),
            features.len()
        );
        plan.check_emit_covered(features.len())
            .map_err(|_| ZensimError::ModelLoadFailed {
                reason: "bake plan emits features the HDR fold walk did not materialize",
            })?;
        features.resize(plan.walk_width(), 0.0);
        self.score_features_with_identity(
            &features,
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
    /// Basic/peak kernels honor the bake revision directly. Wide feature families
    /// still require matching process arithmetic and refuse a mismatch.
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
        let config = config_from_params(params, self.parallel);
        #[cfg(feature = "feature-regime-v2")]
        let plan = self.plan()?;
        #[cfg(feature = "feature-regime-v2")]
        if plan.compute.formula_revision >= crate::feature_defs::FormulaRevision::Rev5 {
            let mut result = crate::fold_engine::compute_fold_backed(
                source,
                distorted,
                &config,
                params.weights,
                &mut self.pixel_scratch,
                Some(&plan),
            )?;
            if images_byte_identical(source, distorted) {
                result.score = 100.0;
                result.raw_distance = 0.0;
                return Ok(result.mark_identical());
            }
            result.score = self.score_features(
                result.features(),
                source.width() as u32,
                source.height() as u32,
                codec_hint,
            )?;
            return Ok(result);
        }
        #[cfg(not(feature = "feature-regime-v2"))]
        if self.layout.walk_width() > 372 {
            return Err(ZensimError::ModelLoadFailed {
                reason: "this bake requires feature-regime-v2 for image extraction",
            });
        }
        #[cfg(feature = "feature-regime-v2")]
        if let Some(sampling) = plan.compute.sampling {
            check_within_max_pixels(
                source.width().max(sampling.min_dim()),
                source.height().max(sampling.min_dim()),
                Some(120_000_000),
            )?;
            if images_byte_identical(source, distorted) {
                return Ok(identical_result_at(&config, plan.walk_width()));
            }
            let (mut features, mean_offset) =
                crate::feature_v2::compute_folded_v1_372_streaming_impl(
                    source,
                    distorted,
                    Some(120_000_000),
                    self.parallel,
                    &mut self.pixel_scratch,
                    Some(&plan),
                    #[cfg(feature = "custom-profiles")]
                    None,
                )?;
            // Wide identity plans declare inputs past the fold's emitted
            // regime; the bake's live reads stop below it, so extend with
            // the same structural zeros `research::extract` reports for
            // unpopulated slots (`truncate` cannot extend).
            // REV4SERVE review F1: the emit claim must already be
            // materialized — a promised slot the walk skipped must error,
            // not zero-fill.
            debug_assert!(
                plan.emit_covered(features.len()),
                "plan emit claims ids past the emitted vector (bound {} > {})",
                plan.emit_bound(),
                features.len()
            );
            plan.check_emit_covered(features.len())
                .map_err(|_| ZensimError::ModelLoadFailed {
                    reason: "bake plan emits features the fold walk did not materialize",
                })?;
            features.resize(plan.walk_width(), 0.0);
            let (_, raw_distance) =
                score_v1_layout_features(&mut features, params.weights, &config, config.num_scales);
            let score = self.score_features(
                &features,
                source.width() as u32,
                source.height() as u32,
                codec_hint,
            )?;
            return Ok(ZensimResult::new(
                score,
                raw_distance,
                features,
                ZensimProfile::B,
                mean_offset,
            ));
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
        if let Some(sampling) = self.plan()?.compute.sampling {
            validate_pair(source, source)?;
            check_within_max_pixels(source.width(), source.height(), Some(120_000_000))?;
            check_within_max_pixels(
                source.width().max(sampling.min_dim()),
                source.height().max(sampling.min_dim()),
                Some(120_000_000),
            )?;
            return Ok(sampling.reference(
                source,
                self.parallel,
                self.plan()?.compute.formula_revision,
            ));
        }
        let plan = self.plan()?;
        if plan.toggles().v1_only
            && plan.compute.free_extras == crate::feature_v2::V1FreeExtras::Off
            && matches!(
                plan.compute.v1_pools,
                crate::feature_v2::V1PoolsMode::Off | crate::feature_v2::V1PoolsMode::Peaks
            )
        {
            validate_pair(source, source)?;
            check_within_max_pixels(source.width(), source.height(), Some(120_000_000))?;
            return Ok(crate::PrecomputedReference::for_candidate(
                source,
                self.parallel,
                None,
                plan.compute.formula_revision,
            ));
        }
        Zensim::new(ZensimProfile::B)
            .with_parallel(self.parallel)
            .precompute_reference(source)
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
    /// removal-based moment linearization as L2/L4. Hard maxima are available
    /// through [`ScoredAttribution::refinement_gain`](crate::ScoredAttribution::refinement_gain),
    /// separately from density. Masked/IW pools remain unsupported; large
    /// removals retain root-curvature and blur-neighborhood approximation errors.
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
        self.compute_attribution_input(
            source,
            precomputed,
            distorted,
            codec_hint,
            session,
            bin,
            None,
        )
    }

    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    #[allow(clippy::too_many_arguments)]
    fn compute_attribution_input(
        &mut self,
        source: &impl ImageSource,
        precomputed: &crate::PrecomputedReference,
        distorted: &impl ImageSource,
        codec_hint: Option<&str>,
        session: &mut crate::Fused944Session,
        bin: usize,
        encoding: Option<crate::feature_v2::HdrEncoding>,
    ) -> Result<crate::ScoredAttribution, ZensimError> {
        if bin == 0 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "attribution bin must be nonzero",
            });
        }
        self.check_pixel_revision()?;
        if let Some(encoding) = encoding {
            crate::feature_v2::validate_hdr_pair(source, distorted, encoding, Some(120_000_000))?;
        } else {
            validate_pair(source, distorted)?;
        }
        validate_ref_dimensions(precomputed, distorted)?;
        check_within_max_pixels(source.width(), source.height(), Some(120_000_000))?;
        let plan = self.plan()?;
        if precomputed.sampling != plan.compute.sampling {
            return Err(ZensimError::ModelLoadFailed {
                reason: "reference cache sampling contract differs from model",
            });
        }
        let params = ZensimProfile::B.params();
        let config = config_from_params(params, self.parallel);
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
                max_removals: Vec::new(),
                moment_removals: Vec::new(),
                unsupported_refinement_feature_ids: Vec::new(),
                has_corruption_gate,
                neighbour_exact: None,
            });
        }
        let (mut features, mean_offset) = session.planned_features(
            source,
            precomputed,
            distorted,
            &plan,
            self.parallel,
            encoding,
        )?;
        // REV4SERVE: resize, not truncate — an identity-declared bake can be
        // wider than the plan's emitted regime (1853-input v2+basic cells),
        // and the uncomputed tail slots are provably unread zeros.
        // REV4SERVE review F1: the emit claim must already be materialized —
        // a promised slot the walk skipped must error, not zero-fill.
        debug_assert!(
            plan.emit_covered(features.len()),
            "plan emit claims ids past the emitted vector (bound {} > {})",
            plan.emit_bound(),
            features.len()
        );
        plan.check_emit_covered(features.len())
            .map_err(|_| ZensimError::ModelLoadFailed {
                reason: "bake plan emits features the fold walk did not materialize",
            })?;
        features.resize(
            plan.walk_width()
                .max(crate::fold_engine::v1_feature_width(&config)),
            0.0,
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
        let (mut spatial, unsupported_feature_ids) =
            crate::attribution::candidate_map_sensitivities(&plan, &sensitivities);
        // NEIGHSTEER (`ZENSIM_NEIGHBOUR_EXACT=1`, SDR v2 sessions only):
        // retain the coarse-scale walk state the local-refinement engine
        // needs, and zero the frozen density it replaces — the v2 pooled
        // features at scales 1–3 (f459..f719) are added back as exact
        // finite deltas inside `ScoredAttribution::refinement_gain`.
        // The switch is EXACTLY `"1"`: presence alone (`=0`, empty) is
        // off, matching the other `ZENSIM_*` gates in this module.
        // Capture refuses (and the density stays whole) for sampling
        // plans, v2-off plans, reflect-padded pairs and foreign dims.
        let neighbour_exact = if std::env::var("ZENSIM_NEIGHBOUR_EXACT").as_deref() == Ok("1")
            && encoding.is_none()
        {
            let snap = crate::local_refine::LocalRefineSnapshot::capture(
                session.retention(),
                &plan,
                (source.width(), source.height()),
                source,
                distorted,
            );
            if snap.is_some() {
                let lo = (372 + 87).min(spatial.len());
                let hi = (372 + 348).min(spatial.len());
                spatial[lo..hi].fill(0.0);
            }
            snap.map(Box::new)
        } else {
            None
        };
        let mut max_removals = Vec::new();
        let mut moment_removals = Vec::new();
        let (_, attribution) = Zensim::new(ZensimProfile::B)
            .with_parallel(self.parallel)
            .attribution_from_retention_binned(
                precomputed,
                distorted,
                &spatial,
                sensitivities
                    .get(156..sensitivities.len().min(228))
                    .unwrap_or(&[]),
                Some(&mut max_removals),
                self.finite_moments.then_some(&mut moment_removals),
                session,
                bin,
                Some(plan.compute.formula_revision),
            )?;
        let unsupported_refinement_feature_ids = crate::attribution::bind_max_removals(
            &mut max_removals,
            &features,
            &unsupported_feature_ids,
        );
        Ok(crate::ScoredAttribution {
            max_removals,
            moment_removals,
            unsupported_refinement_feature_ids,
            result: ZensimResult::new(score, raw_distance, features, ZensimProfile::B, mean_offset),
            attribution,
            sensitivities,
            unsupported_feature_ids,
            has_corruption_gate,
            neighbour_exact,
        })
    }
}

/// Feature IDs a steering session serves with a complete score and map: basic and peaks (f0-227), v2
/// (f372-719; its reference-only PJND_FRAGILITY slots have an exactly-zero integrand), and append/append2
/// (f720-943; reference-only and SDR-structural-zero slots are exact zeros). v2 and later families are SDR
/// only. Everything else is refused up front, naming the family's ID range (the error type carries a static
/// string): masked/IW (f228-371) and f944 and above have no session integrand yet.
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
fn steering_support(reads: &[u16], hdr: bool) -> Result<(), ZensimError> {
    let mut masked_iw = false;
    let mut beyond = false;
    let mut wide = false;
    for &id in reads {
        match id {
            0..=227 => {}
            228..=371 => masked_iw = true,
            372..=943 => wide = true,
            _ => beyond = true,
        }
    }
    if hdr && wide {
        return Err(ZensimError::ModelLoadFailed {
            reason: "HDR steering session supports basic/peak feature IDs f0-f227 only; the bake reads v2/append IDs f372-f943",
        });
    }
    match (masked_iw, beyond) {
        (false, false) => Ok(()),
        (true, false) => Err(ZensimError::ModelLoadFailed {
            reason: "steering session refuses masked/IW feature IDs f228-f371 (no spatial refinement)",
        }),
        (false, true) => Err(ZensimError::ModelLoadFailed {
            reason: "steering session refuses feature IDs f944 and above (no integrand)",
        }),
        (true, true) => Err(ZensimError::ModelLoadFailed {
            reason: "steering session refuses masked/IW feature IDs f228-f371 and feature IDs f944 and above",
        }),
    }
}

/// A source-bound worker created by [`BakeScorer::prepare_steering`].
/// Reuse it for reconstructions of that source. Source, scorer and model
/// borrows keep lifetimes explicit; no image or model is installed globally.
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
pub struct SteeringSession<'s, 'a, S: ImageSource> {
    encoding: Option<crate::feature_v2::HdrEncoding>,
    scorer: &'s mut BakeScorer<'a>,
    source: &'s S,
    reference: crate::PrecomputedReference,
    scratch: crate::Fused944Session,
    bin: usize,
}

#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
impl<S: ImageSource> SteeringSession<'_, '_, S> {
    /// Score a reconstruction and return complete local refinement predictions.
    ///
    /// # Errors
    /// Returns the underlying pixel/scoring error or refuses incomplete
    /// refinement coverage. An activated integrity head returns
    /// [`ZensimError::CorruptionDetected`], even if it would not lower the scalar
    /// score. The returned map is the inactive perceptual branch; each new
    /// reconstruction must be checked again. A failed call preserves the session.
    pub fn compute(
        &mut self,
        distorted: &impl ImageSource,
        codec_hint: Option<&str>,
    ) -> Result<crate::ScoredAttribution, ZensimError> {
        // The head is evaluated on the exact extracted row, once per actual
        // reconstruction. Local probes must never differentiate its threshold.
        #[cfg(feature = "corruption-head")]
        let companion = self.scorer.corruption.take();
        let attempted = self.scorer.compute_attribution_input(
            self.source,
            &self.reference,
            distorted,
            codec_hint,
            &mut self.scratch,
            self.bin,
            self.encoding,
        );
        #[cfg(feature = "corruption-head")]
        {
            self.scorer.corruption = companion;
        }
        let result = attempted?;
        #[cfg(feature = "corruption-head")]
        if !result.result().is_identical()
            && let Some((value, threshold)) = self.scorer.companion_score(
                result.result().features(),
                distorted.width() as u32,
                distorted.height() as u32,
                codec_hint,
            )?
            && value < threshold
        {
            return Err(ZensimError::CorruptionDetected);
        }
        if !result.unsupported_refinement_feature_ids().is_empty() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "steering comparison has incomplete refinement coverage",
            });
        }
        Ok(result)
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

#[cfg(test)]
mod revision_contract_tests {
    use crate::feature_defs::FormulaRevision;
    use crate::ssim_form::SsimLumaForm;
    // Everything that scores PIXELS through a bake needs `feature-regime-v2`:
    // `bake_declaring` fixes a v2 extraction requirement, so `BakeScorer::new`
    // refuses it without the feature and the pixel-path tests are gated.
    #[cfg(feature = "feature-regime-v2")]
    use crate::ssim_form::run_at_revision;
    #[cfg(feature = "feature-regime-v2")]
    use crate::{RgbSlice, ZensimError};

    #[test]
    #[cfg(all(feature = "feature-regime-v2", feature = "corruption-head"))]
    fn consumed_ids_include_inactive_linear_companion() {
        let base = zenpredict::Model::from_bytes(&bake_declaring(None, 13)).unwrap();
        let head = zenpredict::Model::from_bytes(&bake_declaring(None, 91)).unwrap();
        let mut scorer = crate::BakeScorer::new(&base)
            .unwrap()
            .with_linear_corruption_head(&head, 10.0)
            .unwrap();
        let mut row = vec![0.0; 372];
        row[13] = 40.0;
        row[91] = 80.0;
        assert_eq!(scorer.score_features(&row, 96, 96, None).unwrap(), 40.0);
        assert_eq!(scorer.consumed_feature_ids().unwrap(), [13, 91]);
        // Even an inactive gate must have accurate inputs: a feature error
        // can change its activation state on the next image.
        row[91] = 5.0;
        assert_eq!(scorer.score_features(&row, 96, 96, None).unwrap(), 5.0);
        assert_eq!(scorer.consumed_feature_ids().unwrap(), [13, 91]);
    }

    /// A bake over the given feature IDs with small deterministic negative weights (declared revision = the
    /// process default, so the wide-family arithmetic contract matches).
    #[cfg(feature = "feature-regime-v2")]
    fn bake_over(ids: &[usize]) -> zenpredict::Model {
        let weights: Vec<f64> = ids
            .iter()
            .map(|&id| -(0.002 + 0.0003 * ((id * 7) % 13) as f64))
            .collect();
        let recipe = serde_json::json!({
            "schema_hash":1,"scaler_mean":vec![0.0;ids.len()],"scaler_scale":vec![1.0;ids.len()],
            "metadata":[{"key":"zentrain.feature_ids","type":"utf8",
                "text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")}],
            "layers":[{"in_dim":ids.len(),"out_dim":1,"activation":"identity","dtype":"f32",
                "weights":weights,"biases":[100.0]}]
        });
        let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
        zenpredict::Model::from_bytes(&bytes).unwrap()
    }

    /// COSTSET3: the prepared session reuses the basic walk and runs the v2 walk with its v1 block off.
    /// Features, scores and maps must be bit-identical to the legacy route (folded walk with its own v1
    /// fold, then a second v1 walk), under every forced token permutation, for plans with and without the
    /// channel mask, peaks, and append reads. Negative control: the flag really selects a different route
    /// (checked through the retained basic result being present only after the new route's first walk).
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn prepared_session_second_v1_walk_removal_is_bit_identical() {
        use archmage::testing::{
            CompileTimePolicy, for_each_token_permutation, lock_token_testing,
        };
        use core::sync::atomic::Ordering;
        let _guard = lock_token_testing();
        let v2 = |s: usize, c: usize, k: usize| 372 + s * 87 + c * 29 + k;
        let basic_y: Vec<usize> = (0..156).filter(|id| (id / 13) % 3 == 1).collect();
        let v2_y_all: Vec<usize> = (0..4)
            .flat_map(|s| (0..29).map(move |k| v2(s, 1, k)))
            .collect();
        let sets: Vec<(&str, Vec<usize>)> = vec![
            ("v2basic", (0..156).chain(372..720).collect()),
            ("v2basic+append", (0..156).chain(372..944).collect()),
            ("basic+peaks+v2", (0..228).chain(372..720).collect()),
            (
                "basicY+v2Y (mask)",
                basic_y
                    .iter()
                    .copied()
                    .chain(v2_y_all.iter().copied())
                    .collect(),
            ),
            (
                "basic228 + v2 scales 1-3",
                (0..228)
                    .chain((1..4).flat_map(|s| (0..87).map(move |k| v2(s, 0, 0) + k)))
                    .collect(),
            ),
        ];
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            for (w, h) in [(64usize, 64usize), (97, 65), (128, 96)] {
                let (src, dst) = pair(w, h);
                let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
                for (name, ids) in &sets {
                    for bin in [1usize, 8] {
                        let model = bake_over(ids);
                        let mut scorer =
                            crate::BakeScorer::new(&model).unwrap().with_parallel(false);
                        let mut run = |legacy: bool| {
                            crate::attribution::LEGACY_V2_ROUTE.store(legacy, Ordering::Relaxed);
                            let mut worker = scorer.prepare_steering(&rs, bin).unwrap();
                            let a = worker.compute(&ds, None).unwrap();
                            let b = worker.compute(&ds, None).unwrap();
                            crate::attribution::LEGACY_V2_ROUTE.store(false, Ordering::Relaxed);
                            (a, b)
                        };
                        let hits = || crate::attribution::REUSE_ROUTE_HITS.load(Ordering::Relaxed);
                        let (old, _) = run(true);
                        let before = hits();
                        let (new, again) = run(false);
                        // Other tests may serve through the same counter concurrently: a lower bound only.
                        assert!(
                            hits() >= before + 2,
                            "{name}: reuse route must serve both computes"
                        );
                        for scored in [&new, &again] {
                            assert_eq!(
                                scored.result().score().to_bits(),
                                old.result().score().to_bits(),
                                "{name} {w}x{h} bin {bin}"
                            );
                            let (a, b) = (scored.result().features(), old.result().features());
                            assert_eq!(a.len(), b.len());
                            assert!(
                                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                                "{name} {w}x{h} bin {bin}: features differ"
                            );
                            for y0 in (0..h).step_by(8) {
                                for x0 in (0..w).step_by(8) {
                                    let (x1, y1) = ((x0 + 16).min(w), (y0 + 16).min(h));
                                    assert_eq!(
                                        scored.refinement_gain(x0, y0, x1, y1).to_bits(),
                                        old.refinement_gain(x0, y0, x1, y1).to_bits(),
                                        "{name} {w}x{h} bin {bin} rect {:?}",
                                        (x0, y0, x1, y1)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// COSTSET2: v2-bearing plans over basic/peaks/v2 skip X/B work at scales whose chroma slots they never read
    /// (the channel mask), and every consumed feature stays bit-identical to the unrestricted extraction.
    /// The mask must engage for the Y-only and coarse-chroma sets (otherwise the saving is not real), must NOT
    /// engage when an append (cross-channel) block is read, and `edge_width_change` of a chroma channel must
    /// keep the next scale's chroma gradients alive.
    #[test]
    #[cfg(feature = "feature-regime-v2")]
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

    /// `prepare_steering` serves v2 + basic bakes (STEERAPI): the prepared session's score and features equal the
    /// scalar path's bit for bit, and its map equals the older `compute_with_ref_and_attribution` path (they are
    /// the same owner) on fresh and reused sessions, over the whole image grid; also with an ensemble and a
    /// bin of 1.
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn prepared_session_serves_v2_plus_basic_bakes() {
        let (w, h) = (128usize, 96usize);
        let (src, dst) = pair(w, h);
        let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        let sets: [Vec<usize>; 2] = [
            (0..156).chain(372..720).collect(),
            (0..156).chain(372..944).collect(),
        ];
        let model = bake_over(&sets[0]);
        for (ids, bin) in sets.iter().flat_map(|s| [(s, 1usize), (s, 8)]) {
            let model = bake_over(ids);
            let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
            let scalar = scorer.compute(&rs, &ds, None).unwrap();
            let pre = scorer.precompute_reference(&rs).unwrap();
            let old = scorer
                .compute_with_ref_and_attribution(
                    &rs,
                    &pre,
                    &ds,
                    None,
                    &mut crate::Fused944Session::new(),
                    bin,
                )
                .unwrap();
            let mut worker = scorer.prepare_steering(&rs, bin).unwrap();
            let first = worker.compute(&ds, None).unwrap();
            let again = worker.compute(&ds, None).unwrap();
            assert_eq!(scalar.score().to_bits(), first.result().score().to_bits());
            assert_eq!(scalar.features(), first.result().features());
            for scored in [&first, &again] {
                assert_eq!(
                    scored.result().score().to_bits(),
                    old.result().score().to_bits()
                );
                for (y0, x0) in (0..h)
                    .step_by(16)
                    .flat_map(|y| (0..w).step_by(16).map(move |x| (y, x)))
                {
                    let (x1, y1) = ((x0 + 32).min(w), (y0 + 32).min(h));
                    assert_eq!(
                        scored.refinement_gain(x0, y0, x1, y1).to_bits(),
                        old.refinement_gain(x0, y0, x1, y1).to_bits(),
                        "rect {:?} bin {bin}",
                        (x0, y0, x1, y1)
                    );
                }
            }
            assert!(first.refinement_gain(0, 0, 64, 64).is_finite());
            assert!(first.refinement_gain(0, 0, 64, 64) != 0.0);
        }
        // An equal-weight ensemble of two v2 + basic members serves too.
        let other = bake_over(&(0..156).chain(400..700).collect::<Vec<_>>());
        let models = vec![model, other];
        let mut ensemble = crate::BakeScorer::ensemble(&models, None)
            .unwrap()
            .with_parallel(false);
        let scalar = ensemble.compute(&rs, &ds, None).unwrap();
        let scored = ensemble
            .prepare_steering(&rs, 8)
            .unwrap()
            .compute(&ds, None)
            .unwrap();
        assert_eq!(scalar.score().to_bits(), scored.result().score().to_bits());
    }

    /// Refusals: every family without a complete session integrand is refused up front with an error naming
    /// its ID range, even when mixed with supported reads; HDR refuses v2.
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn prepared_session_refuses_unsupported_families_by_name() {
        let (w, h) = (96usize, 96usize);
        let (src, _) = pair(w, h);
        let rs = RgbSlice::new(&src, w, h);
        let refuse = |ids: &[usize]| -> String {
            let model = bake_over(ids);
            let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
            match scorer.prepare_steering(&rs, 8) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("{ids:?} must be refused"),
            }
        };
        assert!(
            refuse(&[13, 300]).contains("masked/IW") && refuse(&[13, 300]).contains("f228-f371")
        );
        assert!(refuse(&[13, 330, 372]).contains("masked/IW"));
        // Mixed with supported append/append2 reads, the masked/IW refusal still names its family.
        assert!(refuse(&[13, 300, 800, 930]).contains("masked/IW feature IDs f228-f371"));
        // f944 and above: refused whether the plan or the session notices first.
        let model = bake_over(&[13, 950]);
        assert!(
            crate::BakeScorer::new(&model)
                .map_or(true, |mut s| s.prepare_steering(&rs, 8).is_err())
        );
        // The same bake served by a plain (non-steering) compute is unaffected by the session's refusal.
        let model = bake_over(&[13, 300]);
        let (s2, d2) = pair(96, 96);
        let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
        assert!(
            scorer
                .compute(
                    &RgbSlice::new(&s2, 96, 96),
                    &RgbSlice::new(&d2, 96, 96),
                    None
                )
                .is_ok()
        );
        // HDR: v2 refused, basic/peak accepted by the contract check (the error is about v2, not the encoding).
        let model = bake_over(&[13, 400]);
        let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
        let err = scorer
            .prepare_steering_hdr(&rs, crate::feature_v2::HdrEncoding::Linear, 8)
            .err()
            .expect("HDR + v2 refused")
            .to_string();
        assert!(
            err.contains("HDR steering") && err.contains("f372-f943"),
            "{err}"
        );
    }

    /// A corruption companion that reads an unsupported ID refuses the session too (its inputs need accurate
    /// maps like every other read).
    #[test]
    #[cfg(all(
        feature = "custom-profiles",
        feature = "feature-regime-v2",
        feature = "corruption-head"
    ))]
    fn prepared_session_refuses_a_companion_reading_unsupported_ids() {
        let (w, h) = (96usize, 96usize);
        let (src, _) = pair(w, h);
        let rs = RgbSlice::new(&src, w, h);
        let base = zenpredict::Model::from_bytes(&bake_declaring(None, 13)).unwrap();
        let ok = zenpredict::Model::from_bytes(&bake_declaring(None, 91)).unwrap();
        let bad = zenpredict::Model::from_bytes(&bake_declaring(None, 300)).unwrap();
        let mut fine = crate::BakeScorer::new(&base)
            .unwrap()
            .with_linear_corruption_head(&ok, 10.0)
            .unwrap();
        assert!(fine.prepare_steering(&rs, 8).is_ok());
        let mut refused = crate::BakeScorer::new(&base)
            .unwrap()
            .with_linear_corruption_head(&bad, 10.0)
            .unwrap();
        let err = refused
            .prepare_steering(&rs, 8)
            .err()
            .expect("companion read refused")
            .to_string();
        assert!(err.contains("masked/IW"), "{err}");
    }

    #[test]
    #[cfg(feature = "feature-regime-v2")]
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
    #[cfg(feature = "feature-regime-v2")]
    fn structurally_skipped_gradient_matches_exhaustive_probes() {
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
            let ids: Vec<_> = (0..944).collect();
            let mut exhaustive = vec![0.0; 944];
            scorer
                .fd_gradient_into(&row, &ids, &mut exhaustive, (96, 96), Some("jxl"))
                .unwrap();
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

    #[test]
    #[cfg(all(
        feature = "corruption-head",
        feature = "feature-regime-v2",
        feature = "custom-profiles"
    ))]
    fn prepared_integrity_gate_preserves_map_and_survives_failures() {
        fn constant(value: f32) -> zenpredict::Model {
            let recipe = serde_json::json!({
                "schema_hash":1,"scaler_mean":[0.0],"scaler_scale":[1.0],
                "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"22"}],
                "layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-0.01],"biases":[value]}]
            });
            let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
            zenpredict::Model::from_bytes(&bytes).unwrap()
        }
        let base = constant(-50.0);
        let src: Vec<_> = (0..96 * 96)
            .map(|i| [(i % 251) as u8, (i % 199) as u8, (i % 127) as u8])
            .collect();
        let mut dst = src.clone();
        dst[100] = [255, 0, 255];
        let rs = crate::RgbSlice::new(&src, 96, 96);
        let ds = crate::RgbSlice::new(&dst, 96, 96);
        let mut plain = crate::BakeScorer::new(&base).unwrap().with_parallel(false);
        let expected = plain
            .prepare_steering(&rs, 8)
            .unwrap()
            .compute(&ds, None)
            .unwrap();
        for (bias, active) in [(100.0, false), (5.0, true)] {
            let head = constant(bias);
            let mut scorer = crate::BakeScorer::new(&base)
                .unwrap()
                .with_parallel(false)
                .with_linear_corruption_head(&head, 10.0)
                .unwrap();
            // Already-poor perceptual scores hide activation in the minimum.
            let scalar = scorer.compute(&rs, &ds, None).unwrap().score();
            assert_eq!(scalar, expected.result().score());
            let mut worker = scorer.prepare_steering(&rs, 8).unwrap();
            let invalid = crate::RgbSlice::new(&dst[..64], 8, 8);
            assert!(worker.compute(&invalid, None).is_err());
            for _ in 0..2 {
                match worker.compute(&ds, None) {
                    Err(crate::ZensimError::CorruptionDetected) => assert!(active),
                    Ok(value) => {
                        assert!(!active);
                        assert_eq!(value.result().score(), scalar);
                        assert_eq!(
                            value.refinement_gain(0, 0, 32, 32),
                            expected.refinement_gain(0, 0, 32, 32)
                        );
                    }
                    Err(e) => panic!("unexpected error: {e}"),
                }
                assert_eq!(worker.compute(&rs, None).unwrap().result().score(), 100.0);
            }
        }
    }

    #[test]
    #[cfg(feature = "corruption-head")]
    fn znpr_companion_applies_threshold_before_minimum_through_surface() {
        let recipe = serde_json::json!({
            "schema_hash":1, "scaler_mean":[0.0], "scaler_scale":[1.0],
            "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"13"}],
            "layers":[{"in_dim":1,"out_dim":1,"activation":"identity",
                "dtype":"f32","weights":[-1.0],"biases":[83.0]}]
        });
        let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let catcher = zenpredict::Model::from_bytes(&bake_declaring(None, 13)).unwrap();
        let mut scorer = crate::BakeScorer::new(&model)
            .unwrap()
            .with_linear_corruption_head(&catcher, 10.0)
            .unwrap();
        let mut features = vec![0.0; 372];
        for (head_score, expected) in [(55.0, 28.0), (2.0, 2.0), (10.0, 73.0), (-60.0, -60.0)] {
            features[13] = head_score;
            assert_eq!(
                scorer.score_features(&features, 96, 96, None).unwrap(),
                expected
            );
        }
    }

    #[cfg(feature = "feature-regime-v2")]
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

    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn finite_moments_preserve_scalar_density_and_session_reuse() {
        if !run_at_revision(
            "3",
            "metric::bake::revision_contract_tests::finite_moments_preserve_scalar_density_and_session_reuse",
            "FINITE-MOMENTS-REUSE-RAN",
        ) {
            return;
        }
        let ids = [14, 15, 53, 54, 165, 183];
        let recipe = serde_json::json!({
            "schema_hash":1,"scaler_mean":vec![0.0;ids.len()],"scaler_scale":vec![1.0;ids.len()],
            "metadata":[
                {"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")},
                {"key":"zentrain.formula_revision","type":"utf8","text":"3"}],
            "layers":[{"in_dim":ids.len(),"out_dim":1,"activation":"identity","dtype":"f32",
                "weights":vec![-1.0;ids.len()],"biases":[100.0]}]
        });
        let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        for (w, h, bin) in [(128, 128, 8), (97, 131, 8), (17, 9, 1)] {
            let (src, dst) = pair(w, h);
            let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
            for parallel in [false, true] {
                let mut old = crate::BakeScorer::new(&model)
                    .unwrap()
                    .with_parallel(parallel);
                let scalar = old.compute(&rs, &ds, None).unwrap();
                let mut finite = crate::BakeScorer::new(&model)
                    .unwrap()
                    .with_parallel(parallel)
                    .with_finite_moment_refinement(true);
                let mut old = old.prepare_steering(&rs, bin).unwrap();
                let mut finite = finite.prepare_steering(&rs, bin).unwrap();
                for image in [&ds, &rs, &ds] {
                    let a = old.compute(image, None).unwrap();
                    let b = finite.compute(image, None).unwrap();
                    assert_eq!(a.result().features(), b.result().features());
                    assert_eq!(a.result().score().to_bits(), b.result().score().to_bits());
                    assert_eq!(a.sensitivities(), b.sensitivities());
                    for rect in [(0, 0, w, h), (1, 1, w / 2, h / 2)] {
                        let (x0, y0, x1, y1) = rect;
                        assert_eq!(
                            a.attribution().query_rect(x0, y0, x1, y1).to_bits(),
                            b.attribution().query_rect(x0, y0, x1, y1).to_bits()
                        );
                        assert!(b.refinement_gain(x0, y0, x1, y1).is_finite());
                    }
                    if !b.result().is_identical() {
                        assert_eq!(scalar.features(), b.result().features());
                        assert!(b.refinement_gain(0, 0, w, h) > a.refinement_gain(0, 0, w, h));
                    } else {
                        assert_eq!(b.refinement_gain(0, 0, w, h), 0.0);
                    }
                }
            }
        }
        println!("FINITE-MOMENTS-REUSE-RAN");
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
    #[cfg(feature = "feature-regime-v2")] // `bake_declaring` fixes a v2 extraction requirement
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

    /// **REV5 scope: a Rev5 bake reading inside `basic + peaks + v2` serves;
    /// one reading any other family is refused at plan time.**
    ///
    /// The refusal is `Plan::for_bake`'s `UnsupportedAtRev5`, surfaced as a
    /// load refusal — the bake's declared revision (5) is what the plan's
    /// scope check reads, so this also pins that a Rev5-declared bake does
    /// not silently compute at a lower revision. The cross-mix arm asserts
    /// the symmetric `refuse_rev4_mix` property: a Rev4 bake in a Rev5
    /// process cannot be served.
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn rev5_bake_serves_supported_reads_and_refuses_unsupported() {
        if !run_at_revision(
            "5",
            "metric::bake::revision_contract_tests::rev5_bake_serves_supported_reads_and_refuses_unsupported",
            "REV5-BAKE-RAN",
        ) {
            return;
        }
        let (w, h) = (96usize, 96usize);
        let (src, dst) = pair(w, h);
        let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));

        // One reader from each supported family: basic, peaks, v2 (two
        // scales). All serve at the declared Rev5.
        for id in [22usize, 200, 400, 700] {
            let model =
                zenpredict::Model::from_bytes(&bake_declaring(Some("5"), id)).expect("parse bake");
            let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
            scorer
                .compute(&rs, &ds, None)
                .unwrap_or_else(|e| panic!("a Rev5 bake reading f{id} must serve, got {e:?}"));
        }
        // Same computed vector as research, including the nonzero reference-only slot.
        let model = zenpredict::Model::from_bytes(&bake_declaring(Some("5"), 393)).unwrap();
        let mut scorer = crate::BakeScorer::new(&model).unwrap();
        let result = scorer.compute(&rs, &rs, None).unwrap();
        let plan = scorer.plan().unwrap();
        let req = crate::research::Request::for_slots(plan.emit.clone(), plan.walk_width());
        let expected = crate::research::extract(&req, &rs, &rs).unwrap();
        assert_eq!(result.score(), 100.0);
        assert!(result.is_identical());
        assert_eq!(result.features(), expected.values());
        assert!(result.features()[393] > 0.0);
        // masked, iw, append, append2, csfw and a Rev4 bank slot: Rev5 does
        // not compute them, so the plan — and the serve — refuses. The
        // refusal lands wherever the plan is first demanded (`new`'s
        // `check_servable` is eager), never silently at a lower revision.
        for id in [228usize, 300, 720, 924, 944, 1100] {
            let model =
                zenpredict::Model::from_bytes(&bake_declaring(Some("5"), id)).expect("parse bake");
            let served = crate::BakeScorer::new(&model)
                .and_then(|mut scorer| scorer.compute(&rs, &ds, None).map(|_| ()));
            assert!(served.is_err(), "a Rev5 bake reading f{id} must refuse");
        }
        // A Rev4-declaring bake in a Rev5 process is the cross-boundary mix.
        let model =
            zenpredict::Model::from_bytes(&bake_declaring(Some("4"), 22)).expect("parse bake");
        let served = crate::BakeScorer::new(&model)
            .and_then(|mut scorer| scorer.compute(&rs, &ds, None).map(|_| ()));
        assert!(
            served.is_err(),
            "a Rev4 bake must not be served by a Rev5 process"
        );
        println!("REV5-BAKE-RAN");
    }

    /// **A revision-3 bake serves a complete scalar score AND a spatial
    /// attribution map through `BakeScorer`.**
    ///
    /// The scalar path alone would not establish map correctness: the
    /// candidate map entry re-runs basic extraction, so it has its own copy
    /// of the corrected signal to get right. This pins that the two agree
    /// bit-for-bit on score and features, and that nothing silently becomes
    /// an unsupported refinement term at this revision.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn rev3_bake_serves_scalar_and_spatial_attribution() {
        if !run_at_revision(
            "3",
            "metric::bake::revision_contract_tests::rev3_bake_serves_scalar_and_spatial_attribution",
            "REV3-BAKE-ATTR-RAN",
        ) {
            return;
        }
        let (w, h) = (128usize, 128usize);
        let (src, dst) = pair(w, h);
        let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        let bytes = bake_declaring(Some("3"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
        let pre = scorer
            .precompute_reference(&rs)
            .expect("precompute reference");
        let mut session = crate::Fused944Session::new();
        let scalar = scorer.compute(&rs, &ds, None).expect("scalar score");
        let scored = scorer
            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
            .expect("scored attribution");
        assert_eq!(
            scalar.score().to_bits(),
            scored.result().score().to_bits(),
            "the attribution entry re-derives the score and must not change it"
        );
        assert_eq!(scalar.features(), scored.result().features());
        assert!(
            scored.unsupported_refinement_feature_ids().is_empty(),
            "revision 3 left refinement terms unsupported: {:?}",
            scored.unsupported_refinement_feature_ids()
        );
        println!("REV3-BAKE-ATTR-RAN score {}", scalar.score());
    }

    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn coarse_v2_attribution_skips_unrequested_scales() {
        if !run_at_revision(
            "3",
            "metric::bake::revision_contract_tests::coarse_v2_attribution_skips_unrequested_scales",
            "COARSE-V2-ATTR-RAN",
        ) {
            return;
        }
        for (w, h) in [(128, 128), (97, 131), (17, 9)] {
            let (src, dst) = pair(w, h);
            let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
            for scale in 0..4 {
                let id = 372 + scale * 87 + 29 + crate::feature_v2::idx::IW_MSE;
                let bytes = bake_declaring(Some("3"), id);
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                for parallel in [false, true] {
                    let mut scorer = crate::BakeScorer::new(&model)
                        .unwrap()
                        .with_parallel(parallel);
                    let pre = scorer.precompute_reference(&rs).unwrap();
                    let mut session = crate::Fused944Session::new();
                    let scalar = scorer.compute(&rs, &ds, None).unwrap();
                    for _ in 0..2 {
                        let scored = scorer
                            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
                            .unwrap();
                        assert_eq!(scalar.features(), scored.result().features());
                        assert_eq!(scalar.score().to_bits(), scored.result().score().to_bits());
                        assert!(
                            scored.unsupported_refinement_feature_ids().is_empty(),
                            "id {id}: {:?}",
                            scored.unsupported_refinement_feature_ids()
                        );
                        let gain = scored.refinement_gain(0, 0, w, h);
                        assert!(gain.is_finite(), "{w}x{h} scale {scale}: {gain}");
                        // A one-input positive weighted-MSE model has negative repair gain.
                        // On unpadded geometry, its weighted-pool density integrates
                        // to minus that feature (up to f32 map combination).
                        assert!(gain < 0.0, "active MSE must not be zeroed");
                        if (w, h) == (128, 128) {
                            assert!(
                                (gain + scalar.features()[id]).abs()
                                    <= 2e-5 * scalar.features()[id].abs().max(1e-12)
                            );
                        }
                    }
                }
            }
        }
        println!("COARSE-V2-ATTR-RAN");
    }

    /// The same contract from the other side: the SHIPPED process refuses a
    /// revision-3 bake. An old bake relabelled `3` therefore cannot be served
    /// as if it had been refit, and a genuine revision-3 bake cannot be
    /// served against revision-1 pixels.
    #[test]
    #[cfg(feature = "feature-regime-v2")] // `bake_declaring` fixes a v2 extraction requirement
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

    /// **Compiling the diagnostic bypass in is not arming it.** With the
    /// `cross-revision-diagnostic` feature ON but the environment switch
    /// unset, revision disagreement is still refused. Two independent
    /// switches is the whole design: `--all-features` builds — including
    /// CI's — must behave exactly like a product build here.
    #[cfg(all(feature = "cross-revision-diagnostic", feature = "feature-regime-v2"))]
    #[test]
    fn the_diagnostic_bypass_is_inert_without_its_environment_switch() {
        assert!(
            std::env::var("ZENSIM_CROSS_REVISION_DIAGNOSTIC").is_err(),
            "this test asserts the DEFAULT; do not run it with the switch set"
        );
        let (w, h) = (96usize, 96usize);
        let (src, dst) = pair(w, h);
        let other = if crate::ssim_form::active_revision() == FormulaRevision::Rev3 {
            "1"
        } else {
            "3"
        };
        let bytes = bake_declaring(Some(other), 400);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
        assert!(
            scorer
                .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h), None)
                .is_err(),
            "the bypass armed itself from the feature flag alone"
        );
    }

    /// And when BOTH switches are on it does bypass, loudly. This is the
    /// capability issue #61 needs to replay an already-fit candidate on a
    /// corrected extraction before any refit; the stderr line is what keeps
    /// such a number attributable in a log.
    #[cfg(all(feature = "cross-revision-diagnostic", feature = "feature-regime-v2"))]
    #[test]
    fn the_diagnostic_bypass_serves_a_mismatched_bake_and_says_so() {
        const SENTINEL: &str = "REV3-CROSS-DIAG-RAN";
        let path = "metric::bake::revision_contract_tests::the_diagnostic_bypass_serves_a_mismatched_bake_and_says_so";
        if std::env::var("ZENSIM_CROSS_REVISION_DIAGNOSTIC").as_deref() != Ok("1") {
            let exe = std::env::current_exe().expect("test binary path");
            let out = std::process::Command::new(exe)
                .args([path, "--exact", "--nocapture", "--test-threads=1"])
                .env("ZENSIM_FORMULA_REV", "3")
                .env("ZENSIM_CROSS_REVISION_DIAGNOSTIC", "1")
                .output()
                .expect("re-exec the test binary");
            let (so, se) = (
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            assert!(out.status.success(), "child failed\n{so}\n{se}");
            assert!(so.contains(SENTINEL), "the control did not run\n{so}");
            assert!(
                se.contains("CROSS-REVISION DIAGNOSTIC"),
                "the bypass served a mismatched bake SILENTLY\n{se}"
            );
            return;
        }
        assert_eq!(crate::ssim_form::active_revision(), FormulaRevision::Rev3);
        let (w, h) = (96usize, 96usize);
        let (src, dst) = pair(w, h);
        let bytes = bake_declaring(Some("1"), 5);
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let mut scorer = crate::BakeScorer::new(&model).expect("load bake");
        let r = scorer
            .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h), None)
            .expect("the armed bypass serves a revision-1 bake on revision-3 pixels");
        assert!(r.score().is_finite());
        println!("{SENTINEL}");
    }

    /// Each subprocess has a different research default, but serves all three
    /// explicit model revisions. The matching process supplies an independent
    /// unrestricted-extractor oracle; equality across processes proves that a
    /// worker neither inherits nor changes global arithmetic.
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn prepared_workers_honor_model_revision_and_local_subset() {
        const MARK: &str = "STEERING-MODEL-BITS ";
        let path = "metric::bake::revision_contract_tests::prepared_workers_honor_model_revision_and_local_subset";
        if std::env::var("ZENSIM_STEERING_CONTRACT_CHILD").is_err() {
            let mut expected = None;
            for process in ["1", "2", "3"] {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([path, "--exact", "--nocapture", "--test-threads=1"])
                    .env("ZENSIM_FORMULA_REV", process)
                    .env("ZENSIM_STEERING_CONTRACT_CHILD", "1")
                    .env_remove("ZENSIM_CROSS_REVISION_DIAGNOSTIC")
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&out.stdout);
                assert!(
                    out.status.success(),
                    "{stdout}\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let bits: Vec<_> = stdout
                    .lines()
                    .filter_map(|l| l.find(MARK).map(|i| l[i..].to_owned()))
                    .collect();
                assert_eq!(bits.len(), 6);
                if let Some(ref expected) = expected {
                    assert_eq!(&bits, expected, "process {process}");
                } else {
                    expected = Some(bits);
                }
            }
            return;
        }
        let (w, h) = (127, 97);
        let (src, mut dst) = pair(w, h);
        for (i, px) in dst.iter_mut().enumerate() {
            *px = [
                ((i * 71 + 19) % 256) as u8,
                ((i * 31 + 87) % 256) as u8,
                ((i * 113 + 3) % 256) as u8,
            ];
        }
        let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        for local_only in [true, false] {
            let ids: Vec<_> = (0..228)
                .filter(|id| !local_only || (*id < 156 && id % 13 < 10))
                .collect();
            let full = crate::Zensim::new(crate::ZensimProfile::B)
                .with_parallel(false)
                .compute_folded720_features_streaming(
                    &rs,
                    &ds,
                    crate::feature_v2::V2NewFeatureToggles {
                        v1_pools: crate::feature_v2::V1PoolsMode::Peaks,
                        ..Default::default()
                    },
                    &mut crate::feature_v2::V2Scratch::new(),
                )
                .unwrap();
            let mut model_rows = Vec::new();
            for revision in ["1", "2", "3"] {
                let recipe = serde_json::json!({
                    "schema_hash":1,"scaler_mean":vec![0.0;ids.len()],"scaler_scale":vec![1.0;ids.len()],
                    "metadata":[
                        {"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")},
                        {"key":"zentrain.formula_revision","type":"utf8","text":revision}],
                    "layers":[{"in_dim":ids.len(),"out_dim":1,"activation":"identity","dtype":"f32",
                        "weights":vec![-1.0;ids.len()],"biases":[100.0]}]
                });
                let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
                assert_eq!(scorer.plan().unwrap().compute.local_only, local_only);
                let scalar = scorer.compute(&rs, &ds, None).unwrap();
                if revision.parse::<u8>().unwrap() == crate::ssim_form::active_revision() as u8 + 1
                {
                    for &id in &ids {
                        assert_eq!(
                            scalar.features()[id].to_bits(),
                            full.features()[id].to_bits(),
                            "f{id}"
                        );
                    }
                }
                let pre = scorer.precompute_reference(&rs).unwrap();
                let old = scorer
                    .compute_with_ref_and_attribution(
                        &rs,
                        &pre,
                        &ds,
                        None,
                        &mut crate::Fused944Session::new(),
                        8,
                    )
                    .unwrap();
                let mut worker = scorer.prepare_steering(&rs, 8).unwrap();
                let invalid = RgbSlice::new(&dst[..64], 8, 8);
                assert!(worker.compute(&invalid, None).is_err());
                let scored = worker.compute(&ds, None).unwrap();
                assert_eq!(scalar.score().to_bits(), scored.result().score().to_bits());
                assert_eq!(scalar.features(), scored.result().features());
                let gain = scored.refinement_gain(8, 8, 40, 40);
                assert_eq!(gain.to_bits(), old.refinement_gain(8, 8, 40, 40).to_bits());
                assert!(gain.is_finite());
                assert_eq!(
                    gain.to_bits(),
                    worker
                        .compute(&ds, None)
                        .unwrap()
                        .refinement_gain(8, 8, 40, 40)
                        .to_bits()
                );
                drop(worker);
                let mut threaded = crate::BakeScorer::new(&model).unwrap().with_parallel(true);
                let scored_mt = threaded
                    .prepare_steering(&rs, 8)
                    .unwrap()
                    .compute(&ds, None)
                    .unwrap();
                assert_eq!(scalar.features(), scored_mt.result().features());
                assert_eq!(
                    gain.to_bits(),
                    scored_mt.refinement_gain(8, 8, 40, 40).to_bits()
                );
                if revision == "3" {
                    assert_eq!(scorer.compute(&rs, &rs, None).unwrap().score(), 100.0);
                }
                let bits: Vec<_> = ids
                    .iter()
                    .map(|&id| scalar.features()[id].to_bits())
                    .collect();
                println!("{MARK}{revision} {bits:?} {}", gain.to_bits());
                model_rows.push(bits);
            }
            assert_ne!(
                model_rows[0], model_rows[2],
                "fixture must distinguish arithmetic eras"
            );
        }
    }

    /// **The exact configuration the 2026-09-18 speed-matrix report accused of
    /// silent mis-serving: the frozen R915 narrow plans.**
    ///
    /// `R915_y60_h32_*.bin` declares `zentrain.formula_revision = 3` over the
    /// 60 ids `13..23, 52..62, 91..101, 117..127, 130..140, 143..153`, and
    /// `R915_basic228_h128_*.bin` declares revision 3 over `0..228`. The report
    /// claimed that with `ZENSIM_FORMULA_REV` unset (a revision-1 process) the
    /// narrow route neither refused nor selected, and scored at the process
    /// arithmetic. Measured on 2026-09-18 against the real bakes and against
    /// these reconstructions, it does neither of those things: it SELECTS. The
    /// two properties that have to hold together, and that this test pins:
    ///
    /// 1. every served number is bit-identical across revision-1, -2 and -3
    ///    processes — the environment is not a serving requirement; and
    /// 2. a revision-1 and a revision-3 bake over the SAME ids score
    ///    differently in one process — so property 1 is selection, not a
    ///    declaration that never reaches the kernels.
    ///
    /// The sibling
    /// [`prepared_workers_honor_model_revision_and_local_subset`] covers the
    /// 228-id and 120-id local sets; this adds the 60-id Y plan the report
    /// named and the HDR entry, which no cross-process test reached.
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn narrow_plans_serve_the_declared_revision_in_every_process() {
        use crate::feature_v2::HdrEncoding;
        use crate::source::{AlphaMode, PixelFormat, StridedBytes};
        const MARK: &str = "NARROW-DECLARED-BITS ";
        let path = "metric::bake::revision_contract_tests::narrow_plans_serve_the_declared_revision_in_every_process";
        if std::env::var("ZENSIM_NARROW_REVISION_CHILD").is_err() {
            let mut expected: Option<Vec<String>> = None;
            for process in ["1", "2", "3"] {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([path, "--exact", "--nocapture", "--test-threads=1"])
                    .env("ZENSIM_FORMULA_REV", process)
                    .env("ZENSIM_NARROW_REVISION_CHILD", "1")
                    .env_remove("ZENSIM_CROSS_REVISION_DIAGNOSTIC")
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&out.stdout);
                assert!(
                    out.status.success(),
                    "{stdout}\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let lines: Vec<_> = stdout
                    .lines()
                    .filter_map(|l| l.find(MARK).map(|i| l[i..].to_owned()))
                    .collect();
                assert_eq!(lines.len(), 4, "child at revision {process}: {stdout}");
                if let Some(expected) = &expected {
                    assert_eq!(
                        &lines, expected,
                        "ZENSIM_FORMULA_REV={process} changed a served number; the \
                         process revision must not reach a declared-revision bake"
                    );
                } else {
                    expected = Some(lines);
                }
            }
            return;
        }
        let (w, h) = (127usize, 97usize);
        let (src, mut dst) = pair(w, h);
        for (i, px) in dst.iter_mut().enumerate() {
            *px = [
                ((i * 71 + 19) % 256) as u8,
                ((i * 31 + 87) % 256) as u8,
                ((i * 113 + 3) % 256) as u8,
            ];
        }
        let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        // HDR-declared copies of the same pair, in absolute linear light.
        let to_hdr = |px: &[[u8; 3]]| -> Vec<[f32; 4]> {
            px.iter()
                .map(|p| {
                    [
                        f32::from(p[0]) * 4.0 + 1.0,
                        f32::from(p[1]) * 4.0 + 1.0,
                        f32::from(p[2]) * 4.0 + 1.0,
                        1.0,
                    ]
                })
                .collect()
        };
        let (hs, hd) = (to_hdr(&src), to_hdr(&dst));
        fn hdr_source(px: &[[f32; 4]], w: usize, h: usize) -> StridedBytes<'_> {
            StridedBytes::with_alpha_mode(
                bytemuck::cast_slice(px),
                w,
                h,
                w * 16,
                PixelFormat::LinearF32Rgba,
                AlphaMode::Opaque,
            )
        }
        let y60: Vec<usize> = [13usize, 52, 91, 117, 130, 143]
            .iter()
            .flat_map(|&base| base..base + 10)
            .collect();
        let basic228: Vec<usize> = (0..228).collect();
        for (name, ids) in [("y60", &y60), ("basic228", &basic228)] {
            let mut per_revision = Vec::new();
            for revision in ["1", "3"] {
                let recipe = serde_json::json!({
                    "schema_hash":1,
                    "scaler_mean": vec![0.0; ids.len()],
                    "scaler_scale": vec![1.0; ids.len()],
                    "metadata":[
                        {"key":"zentrain.feature_ids","type":"utf8",
                         "text": ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")},
                        {"key":"zentrain.formula_revision","type":"utf8","text":revision}],
                    "layers":[{"in_dim":ids.len(),"out_dim":1,"activation":"identity",
                        "dtype":"f32","weights": vec![-1.0; ids.len()],"biases":[100.0]}]
                });
                let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
                let scalar = scorer.compute(&rs, &ds, None).unwrap();
                let hdr_score = scorer
                    .compute_hdr(
                        &hdr_source(&hs, w, h),
                        &hdr_source(&hd, w, h),
                        HdrEncoding::Linear,
                        None,
                    )
                    .unwrap();
                let steering = scorer
                    .prepare_steering(&rs, 8)
                    .unwrap()
                    .compute(&ds, None)
                    .unwrap()
                    .result()
                    .score()
                    .to_bits();
                assert_eq!(
                    scalar.score().to_bits(),
                    steering,
                    "{name} rev{revision}: the steering session and the scalar entry \
                     must serve one arithmetic"
                );
                println!(
                    "{MARK}{name} {revision} {} {}",
                    scalar.score().to_bits(),
                    hdr_score.to_bits()
                );
                per_revision.push((scalar.score().to_bits(), hdr_score.to_bits()));
            }
            // Both entries, separately: a tuple comparison would pass while one
            // of the two quietly stopped selecting.
            let sdr_differs = per_revision[0].0 != per_revision[1].0;
            let hdr_differs = per_revision[0].1 != per_revision[1].1;
            // i686 CARVE-OUT, verified from source and measurement (not
            // assumed): on `target_arch = "x86"` the raw f64 features DO
            // differ between the declared-rev1 and declared-rev3 bakes here
            // (confirmed by probing individual slots, e.g. y60's id 13:
            // 0.9823222077553783 vs 0.9823224186529558) — revision selection
            // reaches the SDR kernels exactly like every other target. What
            // differs is the *magnitude* of that divergence: i686's
            // pixel-accumulation order (32-bit codegen, independent of
            // revision — the same non-associative-float variation this
            // module's own `det_math` doc documents across libc/build
            // configurations) yields an f64 feature sum whose rev1-vs-rev3
            // gap is small enough that `BakeScorer`'s f32-precision model
            // score (`zenpredict`'s `dtype: f32`) rounds `100.0 - sum` to the
            // identical f32 result for both revisions on i686, while the
            // same pair's wider x86_64 gap survives that cast. The HDR entry
            // does not go through this f32 narrow-model score path and does
            // distinguish revisions on every target measured (i686 included),
            // so require at least one of the two here instead of weakening
            // the general (non-x86) contract that both must move.
            if cfg!(target_arch = "x86") {
                assert!(
                    sdr_differs || hdr_differs,
                    "{name}: neither the SDR nor the HDR entry distinguished \
                     revision-1 from revision-3 on this target, so this \
                     fixture cannot tell selection from a declaration that \
                     never reaches the kernels"
                );
            } else {
                assert!(
                    sdr_differs,
                    "{name}: the revision-1 and revision-3 bakes scored the same SDR \
                     number, so this fixture cannot tell selection from a declaration \
                     that never reaches the kernels"
                );
                assert!(hdr_differs, "{name}: same, for the HDR entry");
            }
        }
    }

    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn prepared_worker_refuses_incomplete_contracts_before_use() {
        let (src, _) = pair(96, 96);
        let rs = RgbSlice::new(&src, 96, 96);
        // Masked/IW (f228-371) have no complete session integrand.
        for id in [228, 300, 371] {
            let bytes = bake_declaring(None, id);
            let model = zenpredict::Model::from_bytes(&bytes).unwrap();
            let mut scorer = crate::BakeScorer::new(&model).unwrap();
            assert!(scorer.prepare_steering(&rs, 8).is_err(), "f{id}");
        }
        // v2 (f372-719) and append/append2 (f720-943) are served since STEERAPI (this test used to refuse f400
        // with the old 228 limit).
        for id in [372, 400, 719, 720, 800, 923, 930, 943] {
            let bytes = bake_declaring(None, id);
            let model = zenpredict::Model::from_bytes(&bytes).unwrap();
            let mut scorer = crate::BakeScorer::new(&model).unwrap();
            assert!(scorer.prepare_steering(&rs, 8).is_ok(), "f{id}");
        }
        let bytes = bake_declaring(None, 5);
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = crate::BakeScorer::new(&model).unwrap();
        assert!(scorer.prepare_steering(&rs, 0).is_err());
        assert!(scorer.prepare_steering(&rs, 8).is_ok());
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
        // `4` is registered since featcanon (Rev4, the `tiercanon` era) and
        // `5` since rev5 (the `localwin` era); the first unregistered value
        // is `6`.
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
        assert!(
            crate::feature_layout::formula_revision(&model).is_err(),
            "an unregistered revision must be refused, not defaulted"
        );
        #[cfg(feature = "feature-regime-v2")]
        assert!(
            crate::feature_v2::bake_formula_revision(&model).is_err(),
            "the feature layer must refuse it too, not default to the shipped era"
        );
    }
}
