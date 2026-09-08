//! Legacy CLI policy and diagnostic metadata/gather adapters.
//!
//! September 7: candidate scoring moved to `zensim::BakeScorer`. This module
//! carries no model forward or score composition. Byte decoding is owned by
//! `zensim::bake_metadata`; short feature rows are refused by the surface.

use zenpredict::Model;

/// Resolve the legacy CLI spelling into the product's disposition contract.
/// Unknown or malformed policies refuse before evaluation starts.
pub fn post_mode_params(mode: &str) -> Result<zensim::profile::ProfileParams, String> {
    let builder = zensim::profile::ProfileParams::builder();
    Ok(match mode {
        "raw" | "extrapolate" => builder
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .build(),
        "clamp" => builder
            .skip_score_mapping(true)
            .extrapolate_score(false)
            .build(),
        "mapped" => builder.score_mapping(18.0, 0.7).build(),
        m if m.starts_with("mapped:") => {
            let (a, b) = m[7..].split_once(',').ok_or("expected mapped:A,B")?;
            let a: f64 = a.parse().map_err(|_| "invalid mapping A")?;
            let b: f64 = b.parse().map_err(|_| "invalid mapping B")?;
            if !a.is_finite() || a <= 0.0 || !b.is_finite() || b <= 0.0 {
                return Err("mapping A and B must be finite and positive".into());
            }
            builder.score_mapping(a, b).build()
        }
        _ => return Err(format!("unknown bake-post policy {mode:?}")),
    })
}

/// Per-sample-α head dispatch payload — parsed from the bake's
/// `zentrain.per_sample_alpha_head` metadata. Layout matches
/// `zensim-train-core::per_sample_alpha_head::bake_per_sample_alpha_head_v3_with_tanh`.
///
/// `(W_α, b_α, rank_w, rank_b, reducer_w, reducer_b, p_norm)`.
pub type PerSampleAlphaHeadDispatch = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);

/// Hybrid-head dispatch payload — parsed from the bake's
/// `zentrain.hybrid_head` metadata. Layout matches
/// `zensim-train-core::hybrid_head::bake_hybrid_head_v3`.
///
/// `(rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)`.
pub type HybridHeadDispatch = (Vec<f32>, f32, f32, [f32; 4], f32, f32);

/// Optional pin scale. Malformed present metadata panics with a named error.
pub fn extract_tanh_output_head_scale(model: &Model) -> Option<f64> {
    zensim::bake_metadata::parse_bake_metadata(model)
        .expect("invalid score metadata")
        .tanh_pin_scale
}

/// Read the `zentrain.per_sample_alpha_head` metadata payload, if any.
/// Returns `Some((W_α, b_α, rank_w, rank_b, reducer_w, reducer_b, p_norm))`.
pub fn extract_per_sample_alpha_head(model: &Model) -> Option<PerSampleAlphaHeadDispatch> {
    let md = zensim::bake_metadata::parse_bake_metadata(model).expect("invalid score metadata");
    md.per_sample_alpha.map(|p| {
        (
            p.w_alpha.clone(),
            p.b_alpha,
            p.rank_w.clone(),
            p.rank_b,
            p.reducer_w,
            p.reducer_b,
            p.p_norm,
        )
    })
}

/// Read the `zentrain.hybrid_head` metadata payload, if any.
/// Returns `Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm))`.
pub fn extract_hybrid_head(model: &Model) -> Option<HybridHeadDispatch> {
    let md = zensim::bake_metadata::parse_bake_metadata(model).expect("invalid score metadata");
    md.hybrid_head.map(|p| {
        (
            p.rank_w.clone(),
            p.rank_b,
            p.alpha_logit,
            p.reducer_w,
            p.reducer_b,
            p.p_norm,
        )
    })
}

/// Legacy diagnostic name for the runtime-owned min-max parameter view.
pub use zensim::bake_metadata::MinMaxHeadMeta as MinMaxHeadDispatch;

/// Read the `zentrain.minmax_monotone_head` payload, if any. Layout:
/// `[u32 k, u32 j, u32 n, w f32×(k·j·n), b f32×(k·j)]`.
pub fn extract_minmax_head(model: &Model) -> Option<MinMaxHeadDispatch> {
    zensim::bake_metadata::parse_bake_metadata(model)
        .expect("invalid score metadata")
        .minmax_head
        .map(|p| (*p).clone())
}

/// Diagnostic gather policy, resolved from the runtime's feature declaration.
/// Candidate scoring uses `BakeScorer` directly; this adapter is only for
/// network-level diagnostics and table admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallerGather {
    /// Copy the declared prefix; a short row refuses.
    Positional,
    /// Take `row[ids[j]]` into slot `j`; every requested ID must exist.
    ByFeatureId(Vec<u16>),
}

impl CallerGather {
    /// The gather this bake declares. Resolves through
    /// `zensim::declared_feature_ids` — the ONE owner of the declaration — so
    /// the evaluation tools and the `zensim` runtime cannot disagree about
    /// which ids a bake reads.
    pub fn for_model(model: &Model) -> CallerGather {
        match zensim::declared_feature_ids(model) {
            Some(ids) => CallerGather::ByFeatureId(ids),
            None => CallerGather::Positional,
        }
    }

    /// Is this bake dense? Used by callers that want to REPORT the fact.
    pub fn is_dense(&self) -> bool {
        matches!(self, CallerGather::ByFeatureId(_))
    }

    /// Can a grid or corpus of `n_features` columns feed this bake?
    ///
    /// **Positional keeps the EXACT rule every grid gate used before this type
    /// existed — `n_features == n_inputs`** — so no identity bake's grid is
    /// admitted or skipped differently. A DENSE bake reads ids out of a wider
    /// caller-laid-out row, so its requirement is that the row REACHES its
    /// highest declared id; a `==` test against its packed width would skip
    /// every real grid it can actually score, which is a silent coverage loss
    /// rather than a wrong number, and just as bad in a published verdict.
    pub fn accepts_row_width(&self, n_features: usize, n_inputs: usize) -> bool {
        match self {
            CallerGather::Positional => n_features == n_inputs,
            CallerGather::ByFeatureId(ids) => ids
                .iter()
                .max()
                .is_some_and(|&hi| n_features > usize::from(hi)),
        }
    }

    /// Can a PREFIX-tolerant source of `n_features` columns feed this bake?
    ///
    /// Two admission rules exist because two DIFFERENT historical rules exist,
    /// and collapsing them would move numbers in both directions.
    /// [`accepts_row_width`](Self::accepts_row_width) is the GRID rule: an
    /// exact `==`, because a dial/corruption grid that is not the bake's own
    /// width is a different instrument and admitting it silently would publish
    /// a panel about the wrong thing. This is the rule for a source whose
    /// contract has always been *"take the leading `n_inputs` columns"* — the
    /// kadis multi-metric per-pair table, which is 720 columns wide and feeds
    /// 372-input bakes by prefix. Using the grid rule there dropped the whole
    /// per-pair block from every identity bake's verdict (measured while
    /// wiring the dense gather, hence this split).
    ///
    /// The DENSE arm is the same for both: a declared bake needs the row to
    /// REACH its highest declared id, because it indexes by id and its own
    /// packed width says nothing about where those ids live.
    pub fn accepts_prefix_row_width(&self, n_features: usize, n_inputs: usize) -> bool {
        match self {
            CallerGather::Positional => n_features >= n_inputs,
            CallerGather::ByFeatureId(_) => self.accepts_row_width(n_features, n_inputs),
        }
    }

    /// Fill `dst` (already sized to the bake's caller width) from `row`.
    pub fn fill(&self, dst: &mut [f32], row: &[f64]) {
        assert!(
            self.accepts_prefix_row_width(row.len(), dst.len()),
            "diagnostic row does not contain the bake's declared IDs"
        );
        match self {
            Self::Positional => {
                for (out, &value) in dst.iter_mut().zip(row) {
                    *out = value as f32;
                }
            }
            Self::ByFeatureId(ids) => {
                assert_eq!(
                    dst.len(),
                    ids.len(),
                    "diagnostic scratch does not match the declaration"
                );
                for (out, &id) in dst.iter_mut().zip(ids) {
                    *out = row[usize::from(id)] as f32;
                }
            }
        }
    }
}
