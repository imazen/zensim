// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Layer 2 — the LAYOUT**, a declared mapping from feature ids to positions
//! in an emitted vector.
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §4. Phase 4 of
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`.
//!
//! ## What this makes expressible that was not
//!
//! Phase 1 landed `layout_width` + `LayoutBlocks::for_width` and recorded WHY
//! (`PLAN_FEATURE_SYSTEM` §1b): every layout registered at the time was the
//! IDENTITY mapping over its range, so a full `id -> position` map would have
//! been a type with no consumer and no test that could distinguish it from the
//! width it wrapped. That is no longer true — a **dense** layout has a
//! different mapping, and phase 4 is where it gets its first real consumer.
//!
//! Three facts follow, none of which held before:
//!
//! 1. **The legacy widths are DECLARATIONS, not code paths.** `w156`, `w372`,
//!    `w504`, `w720`, `w924`, `w944`, `w956` are `Layout`s whose `slot_at[i]
//!    == Some(i)`. Every stored table, every shipped bake and every board row
//!    is already in one of them, so **no stored byte moves**.
//! 2. **A structural zero is a LAYOUT property, not a data property.** A
//!    position whose id the plan does not compute is written `0.0` *because
//!    the layout declares that id lives there and the plan does not populate
//!    it* — the distinction `FEATURE_SET_IDS.md` §3.1 says is missing when
//!    seven different feature sets are all called "944".
//! 3. **A dense subset is expressible.** [`Layout::dense`] packs a read set
//!    with no gaps, which is what retires "944-with-structural-zeros as the
//!    wire format" for NEW artifacts without touching the 944 that exists.
//!
//! ## Why the gather is a real safety fix, not a tidiness one
//!
//! Before this, `metric::prep_bake_input_f32` resolved a width disagreement
//! POSITIONALLY: `n_inputs < features.len()` took the first `n_inputs`
//! positions. For every bake that exists today that is correct, because every
//! declared width is a PREFIX of the walk's identity layout. The moment a
//! dense table exists it stops being correct — a 265-input bake over
//! `basic+peaks+moments` would silently be served `f0..265`, which is
//! `basic + peaks + 37 slots of masked` — plausible numbers, wrong features,
//! no error. A declared layout turns that into a gather, and an *undeclarable*
//! width into a loud refusal.

use crate::feature_set_id::{FeatureSetId, SlotSet};

/// Resolve a bake's arithmetic era. Absence is the registered pre-stamp
/// legacy era; a present unknown value is never a request for that default.
pub(crate) fn formula_revision(
    model: &crate::mlp::Model,
) -> Result<crate::feature_defs::FormulaRevision, crate::ZensimError> {
    use crate::feature_defs::FormulaRevision;
    if model.metadata().get("zentrain.formula_revision").is_none() {
        return Ok(crate::ssim_form::SHIPPED_REVISION);
    }
    match model
        .metadata()
        .get_utf8("zentrain.formula_revision")
        .ok()
        .map(str::trim)
    {
        Some("1" | "rev1" | "Rev1") => Ok(FormulaRevision::Rev1),
        Some("7" | "rev7" | "Rev7") => Ok(FormulaRevision::Rev7),
        _ => Err(crate::ZensimError::ModelLoadFailed {
            reason: "unknown zentrain.formula_revision",
        }),
    }
}

/// A declared mapping from feature ids to positions in an emitted vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Position -> feature id. `None` is a DECLARED GAP: a position the
    /// layout reserves that carries no registered feature (only reachable
    /// past the registry's full width).
    slot_at: Vec<Option<u16>>,
    /// `(id, position)` sorted by id — the reverse index. A `Vec` + binary
    /// search rather than a `HashMap`: a layout is built once per plan, never
    /// per pixel, and the ids are already sorted for every layout anyone can
    /// construct here.
    by_id: Vec<(u16, u32)>,
    /// A stable name for manifests and error messages.
    name: String,
}

impl Layout {
    /// The identity layout of `width` — `slot_at[i] == Some(i)`.
    ///
    /// Every layout registered before phase 4 is one of these, which is why
    /// declaring them changes no stored byte.
    pub(crate) fn identity(width: usize) -> Layout {
        let slot_at: Vec<Option<u16>> = (0..width).map(|i| u16::try_from(i).ok()).collect();
        let by_id = slot_at
            .iter()
            .enumerate()
            .filter_map(|(pos, id)| id.map(|id| (id, pos as u32)))
            .collect();
        Layout {
            slot_at,
            by_id,
            name: format!("w{width}"),
        }
    }

    /// A DENSE layout over `slots`: the requested ids packed in ascending
    /// order with no gaps, so the emitted width is `slots.len()`.
    pub(crate) fn dense(slots: &SlotSet) -> Layout {
        let slot_at: Vec<Option<u16>> = slots.iter_slots().map(|s| u16::try_from(s).ok()).collect();
        let by_id = slot_at
            .iter()
            .enumerate()
            .filter_map(|(pos, id)| id.map(|id| (id, pos as u32)))
            .collect();
        Layout {
            name: format!("dense{}", slot_at.len()),
            slot_at,
            by_id,
        }
    }

    /// The emitted width.
    pub(crate) fn width(&self) -> usize {
        self.slot_at.len()
    }

    /// The feature id at `pos`, or `None` for a declared gap.
    pub(crate) fn slot_at(&self, pos: usize) -> Option<u16> {
        self.slot_at.get(pos).copied().flatten()
    }

    /// Every id this layout carries.
    // Consumers (`feature_plan`, `research`) are `feature-regime-v2`-gated; the
    // LAYOUT itself is not, because the shipped dense bakes it serves are not
    // (2026-09-06 ungate). Scoped so a genuinely-dead accessor still shows up
    // on a default build.
    pub(crate) fn ids(&self) -> SlotSet {
        SlotSet::from_slots(self.slot_at.iter().flatten().map(|&id| usize::from(id)))
    }

    /// Is this the identity mapping over `0..width`?
    ///
    /// The fast path the runtime takes for every artifact that exists today.
    pub(crate) fn is_identity(&self) -> bool {
        self.slot_at
            .iter()
            .enumerate()
            .all(|(pos, id)| *id == u16::try_from(pos).ok())
    }

    /// The identity width the WALK must emit for this layout to be fillable:
    /// one past the highest id it carries.
    ///
    /// Distinct from [`Self::width`] the moment a layout is dense — a
    /// `dense265` layout over `basic+peaks+moments` still needs a walk that
    /// reaches slot 943.
    pub(crate) fn walk_width(&self) -> usize {
        self.slot_at
            .iter()
            .flatten()
            .map(|&id| usize::from(id) + 1)
            .max()
            .unwrap_or(0)
    }

    /// Gather a walk's identity-laid-out vector into this layout.
    ///
    /// A position whose id the walk did not reach, and a declared gap, both
    /// become the structural `0.0` — which is what the layout DECLARES, not a
    /// silently-dropped value: `Plan::emit` is what says whether a position
    /// carries a computed number.
    pub(crate) fn gather(&self, walk: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.reserve(self.slot_at.len());
        for id in &self.slot_at {
            out.push(match id {
                Some(id) => walk.get(usize::from(*id)).copied().unwrap_or(0.0),
                None => 0.0,
            });
        }
    }
}

/// The layout a loaded bake DECLARES.
///
/// Conservative by construction, and that is the safety property: a dense
/// layout is used **only** when the bake carries a `zentrain.feature_set_id`
/// that this build can reproduce AND whose slot count equals the bake's
/// caller-facing width exactly. Anything else — no id, an unparseable id, an
/// id whose hash disagrees, or an id whose slot count differs from the width
/// (which is the NORMAL case: the metadata records the id of the tables the
/// bake was TRAINED on, and a `944`-with-structural-zeros producer names 265
/// slots at width 944) — resolves to the IDENTITY layout at
/// `caller_input_width()`, which is exactly what every bake in existence gets
/// today.
///
/// So this function returns a non-identity layout for **zero** currently
/// shipped or board bakes, by design; it is the hook a dense-table bake needs
/// in order to be servable at all, and
/// [`tests::every_shipped_bake_resolves_to_an_identity_layout`] pins that.
pub(crate) fn declared_layout(model: &crate::mlp::Model) -> Layout {
    let width = model.caller_input_width();
    // THE declaration, preferred over every inference: an explicit ascending
    // id list. A family-token id can only name a FAMILY UNION, so it cannot
    // express a read set like Profile D's 28 basic slots; `feature_ids` can,
    // and it is what the cruft purge stamps onto every densified bake.
    if let Some(slots) = declared_ids(model)
        && slots.len() == width
    {
        return Layout::dense(&slots);
    }
    // E33 `fx1`: a derived-input bake's declared ids are its READ SET and its
    // gather layout; the model's caller width counts derived entries instead.
    // Validity (every id read, entry count == caller width) is checked where
    // the bake is loaded (`bake_metadata::parse_bake_metadata`), which refuses
    // a malformed declaration before any forward.
    if model
        .metadata()
        .get(crate::derived_inputs::DERIVED_INPUTS_KEY)
        .is_some()
        && let Some(slots) = declared_ids(model)
    {
        return Layout::dense(&slots);
    }
    let dense = model
        .metadata()
        .get_utf8("zentrain.feature_set_id")
        .ok()
        .and_then(FeatureSetId::parse)
        .as_ref()
        .and_then(dense_slots_of)
        .filter(|slots| slots.len() == width);
    match dense {
        Some(slots) => Layout::dense(&slots),
        None => Layout::identity(width),
    }
}

/// The bake's EXPLICIT declared read set, from `zentrain.feature_ids`.
///
/// Format: ascending decimal feature ids, one per line (whitespace-separated
/// is accepted so the payload can be written either way). **Strict**: a
/// duplicate, a descending pair, an unparseable token or an id past the
/// registry's full width returns `None` and the caller falls back to the
/// identity layout — a declaration this build cannot reproduce must never be
/// half-believed, because a wrong id list permutes the vector the bake is
/// served.
///
/// This is a METADATA convention, not a format change: ZNPR v3 metadata is
/// free-form utf8 key/value, and `zenpredict` is untouched.
pub(crate) fn declared_ids(model: &crate::mlp::Model) -> Option<SlotSet> {
    let text = model.metadata().get_utf8(FEATURE_IDS_KEY).ok()?;
    parse_feature_ids(text, crate::feature_defs::full_width(crate::NUM_SCALES))
}

/// The metadata key carrying a bake's explicit declared read set.
pub const FEATURE_IDS_KEY: &str = "zentrain.feature_ids";

/// Parse + VALIDATE an explicit id list. Separated from [`declared_ids`] so
/// the strictness is testable without building a `Model`.
pub(crate) fn parse_feature_ids(text: &str, full_width: usize) -> Option<SlotSet> {
    let mut ids: Vec<usize> = Vec::new();
    for tok in text.split_ascii_whitespace() {
        let v: usize = tok.parse().ok()?;
        if v >= full_width {
            return None;
        }
        if let Some(&last) = ids.last() {
            // Strictly ascending: rejects both duplicates and disorder, which
            // is what makes position `j` provably the `j`-th smallest id and
            // so makes `Layout::dense`'s packing the same mapping the writer
            // intended.
            if v <= last {
                return None;
            }
        }
        ids.push(v);
    }
    (!ids.is_empty()).then(|| SlotSet::from_slots(ids))
}

/// The DENSE reading of an id, and only the dense one.
///
/// [`slots_of`] tries both readings because a caller reproducing a registered
/// set wants whichever one the id names. [`declared_layout`] must not: a
/// SPARSE id whose reconstruction happens to have as many members as the
/// bake's width would otherwise be read as a dense layout and permute the
/// vector the bake is served. So the layout decision takes the strict form —
/// clip to the registry's full width, require the hash, require the packed
/// count — and anything else stays the identity layout.
fn dense_slots_of(id: &FeatureSetId) -> Option<SlotSet> {
    let ns = crate::NUM_SCALES;
    let mut union = SlotSet::from_slots([]);
    for t in id.compute().iter() {
        union = union.union(&crate::feature_defs::family_slots(t, ns));
    }
    let dense = union.clipped_to(crate::feature_defs::full_width(ns));
    // A set that IS the identity range is not a dense layout, it is the
    // identity one under another name — `Layout::dense(0..372)` and
    // `Layout::identity(372)` are the same mapping, and returning the former
    // would only make `is_identity` say something different about a vector
    // nothing did to.
    let is_identity_range = dense.iter_slots().last() == Some(dense.len().saturating_sub(1));
    // The recorded width is not consulted: it is not part of the identity, and
    // `declared_layout` — the only caller — filters on the BAKE's caller width,
    // which is the number a mis-sized layout would actually break.
    (dense.hash8() == id.slots_hash() && !is_identity_range).then_some(dense)
}

/// The feature ids a bake declares, in ascending order, or `None` when it
/// declares the identity layout.
pub(crate) fn declared_feature_ids(model: &crate::mlp::Model) -> Option<Vec<u16>> {
    let layout = declared_layout(model);
    if layout.is_identity() {
        return None;
    }
    Some(
        (0..layout.width())
            .map(|pos| layout.slot_at(pos).unwrap_or(u16::MAX))
            .collect(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The gather moves values to their declared positions and fills the rest
    /// with the structural zero — never a stale or shifted value.
    #[test]
    fn the_gather_places_ids_at_their_declared_positions() {
        let want = SlotSet::from_ranges([(0, 4), (10, 12)]);
        let l = Layout::dense(&want);
        assert_eq!(l.width(), 6);
        let walk: Vec<f64> = (0..12).map(|i| i as f64 * 100.0).collect();
        let mut out = Vec::new();
        l.gather(&walk, &mut out);
        assert_eq!(out, vec![0.0, 100.0, 200.0, 300.0, 1000.0, 1100.0]);
        // An id past the walk's reach becomes the structural zero, not a
        // panic and not a wrapped read.
        let short = vec![1.0, 2.0];
        l.gather(&short, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
        // The identity layout's gather is a plain copy.
        let id = Layout::identity(12);
        id.gather(&walk, &mut out);
        assert_eq!(out, walk);
    }

    /// The parse is STRICT, and each rejection is its own reason. A
    /// half-believed id list permutes the vector a bake is served, so every
    /// one of these must fall back to identity rather than guess.
    #[test]
    fn the_feature_id_list_parse_refuses_anything_it_cannot_prove() {
        let full = crate::feature_defs::full_width(crate::NUM_SCALES);
        assert_eq!(
            parse_feature_ids("0 1 2 155", full),
            Some(SlotSet::from_slots([0usize, 1, 2, 155]))
        );
        // Newline-separated is the written form; whitespace-separated parses
        // the same, so the payload can be either.
        assert_eq!(
            parse_feature_ids("0\n1\n2\n155\n", full),
            Some(SlotSet::from_slots([0usize, 1, 2, 155]))
        );
        assert_eq!(parse_feature_ids("0 1 1 2", full), None, "duplicate");
        assert_eq!(parse_feature_ids("0 5 3", full), None, "descending");
        assert_eq!(parse_feature_ids("0 x 3", full), None, "unparseable");
        assert_eq!(parse_feature_ids("", full), None, "empty");
        assert_eq!(parse_feature_ids("-1 3", full), None, "negative");
        assert_eq!(
            parse_feature_ids(&format!("0 {full}"), full),
            None,
            "past the registry's full width"
        );
        assert_eq!(
            parse_feature_ids(&format!("0 {}", full - 1), full),
            Some(SlotSet::from_slots([0usize, full - 1])),
            "the last registered id is in range"
        );
    }

    /// A 1-layer bake of `ids.len()` inputs declaring those ids explicitly.
    pub(crate) fn ids_bake_bytes(ids: &[usize]) -> Vec<u8> {
        let list = ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        bake_bytes_with_metadata(
            ids.len(),
            &format!(
                r#"{{"key": "{FEATURE_IDS_KEY}", "type": "utf8", "text": "{}"}}"#,
                list.replace('\n', "\\n")
            ),
        )
    }

    fn bake_bytes_with_metadata(n: usize, md: &str) -> Vec<u8> {
        let arr = |v: &[f32]| -> String {
            let mut s = String::from("[");
            for (k, x) in v.iter().enumerate() {
                if k > 0 {
                    s.push(',');
                }
                s.push_str(&x.to_string());
            }
            s.push(']');
            s
        };
        let json = format!(
            r#"{{
                "schema_hash": 1,
                "scaler_mean": {mean},
                "scaler_scale": {scale},
                "metadata": [{md}],
                "layers": [
                    {{"in_dim": {n}, "out_dim": 1, "activation": "identity",
                      "dtype": "f32", "weights": {w}, "biases": [0.0]}}
                ]
            }}"#,
            mean = arr(&vec![0.0f32; n]),
            scale = arr(&vec![1.0f32; n]),
            w = arr(&vec![1.0f32; n]),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic bake must build")
    }
}
