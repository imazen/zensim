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

    /// A stable name for manifests and messages (`w944`, `dense265`).
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The feature id at `pos`, or `None` for a declared gap.
    pub(crate) fn slot_at(&self, pos: usize) -> Option<u16> {
        self.slot_at.get(pos).copied().flatten()
    }

    /// The position of feature `id`, or `None` when this layout does not
    /// carry it.
    pub(crate) fn pos_of(&self, id: u16) -> Option<usize> {
        self.by_id
            .binary_search_by_key(&id, |&(i, _)| i)
            .ok()
            .map(|k| self.by_id[k].1 as usize)
    }

    /// Every id this layout carries.
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

/// The slot set a registered [`FeatureSetId`] names, or `None` when this build
/// cannot reproduce it.
///
/// The id carries COMPUTE tokens, a WIDTH, an era and a slot hash; the slots
/// follow from the registry. **The width means two different things depending
/// on the layout, and the HASH is what decides which** — which is exactly the
/// job the identity layer exists to do:
///
/// * for a SPARSE (identity) layout the width is also the clip: a
///   `basic+peaks+moments@w944` id names the 265 slots of those families that
///   fall below 944, laid out at their own indices with 679 structural fills;
/// * for a DENSE layout the width is the PACKED count: the same 265 slots,
///   at consecutive positions, so the id reads `…@w265` — and clipping the
///   family union to 265 would reconstruct a completely different set.
///
/// So both readings are tried and the recorded `slots_hash8` picks the one
/// the id actually names. An id neither reading reproduces describes a set
/// this build does not have, and saying so is the point.
pub(crate) fn slots_of(id: &FeatureSetId) -> Option<SlotSet> {
    let ns = crate::NUM_SCALES;
    let mut union = SlotSet::from_slots([]);
    for t in id.compute().iter() {
        union = union.union(&crate::feature_defs::family_slots(t, ns));
    }
    let sparse = union.clipped_to(id.layout_width());
    if sparse.hash8() == id.slots_hash() {
        return Some(sparse);
    }
    let dense = union.clipped_to(crate::feature_defs::full_width(ns));
    (dense.hash8() == id.slots_hash() && dense.len() == id.layout_width()).then_some(dense)
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
    (dense.hash8() == id.slots_hash() && dense.len() == id.layout_width() && !is_identity_range)
        .then_some(dense)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_set_id::ComputeToken;

    /// **G4.2, the legacy half** — every registered legacy width is the
    /// IDENTITY mapping over its range, so declaring it moves no stored byte.
    ///
    /// Generalizes phase 1's G1.7 from a width to a `Layout`.
    #[test]
    fn every_legacy_width_is_the_identity_mapping() {
        for w in [156usize, 372, 504, 720, 924, 944, 956] {
            let l = Layout::identity(w);
            assert_eq!(l.width(), w);
            assert_eq!(l.walk_width(), w, "identity: walk width IS the width");
            assert!(l.is_identity(), "w{w} must be identity");
            for pos in 0..w {
                assert_eq!(l.slot_at(pos), u16::try_from(pos).ok());
                assert_eq!(l.pos_of(pos as u16), Some(pos));
            }
        }
    }

    /// A dense layout packs its ids with no gaps, and round-trips both ways.
    #[test]
    fn a_dense_layout_packs_and_round_trips() {
        let ns = crate::NUM_SCALES;
        let want = SlotSet::from_ranges([(0, 228)])
            .union(&crate::feature_defs::family_slots(
                ComputeToken::Moments,
                ns,
            ))
            .clipped_to(944);
        let l = Layout::dense(&want);
        assert_eq!(l.width(), 265, "the free-set arm packs to 265");
        // 942, not 944: the highest raw-moment slot is f941, so the WALK
        // must reach 942 — a fact worth having a number for, because
        // `LayoutBlocks::for_width(942)` still reaches append2 (base 924) and
        // still does NOT reach CSFW (base 944), which is exactly right for
        // this set.
        assert_eq!(
            l.walk_width(),
            942,
            "the walk must reach one past the highest id the layout carries"
        );
        assert!(!l.is_identity());
        assert_eq!(l.ids(), want);
        // Every id round-trips through its position, in both directions.
        for (pos, id) in want.iter_slots().enumerate() {
            assert_eq!(l.slot_at(pos), u16::try_from(id).ok());
            assert_eq!(l.pos_of(id as u16), Some(pos));
        }
        // And an id the layout does not carry has no position.
        assert_eq!(l.pos_of(300), None, "f300 is IW, not in this set");
    }

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

    /// A bake DECLARING a dense feature set resolves to that dense layout —
    /// the positive control for
    /// [`tests::every_shipped_bake_resolves_to_an_identity_layout`], without
    /// which that test would pass just as happily if `declared_layout` always
    /// returned identity.
    #[test]
    fn a_bake_declaring_a_dense_set_resolves_to_that_layout() {
        let ns = crate::NUM_SCALES;
        let slots = SlotSet::from_ranges([(0, 228)])
            .union(&crate::feature_defs::family_slots(
                ComputeToken::Moments,
                ns,
            ))
            .clipped_to(944);
        assert_eq!(slots.len(), 265);
        let id = crate::feature_set_id::FeatureSetId::from_slots(
            crate::feature_set_id::ComputeParts::EMPTY
                .with(ComputeToken::Basic)
                .with(ComputeToken::Peaks)
                .with(ComputeToken::Moments),
            265,
            "era2r4",
            &slots,
        )
        .expect("a dense id");
        // The id must round-trip through its own grammar, and `slots_of` must
        // read the DENSE interpretation out of it.
        let reparsed = crate::feature_set_id::FeatureSetId::parse(&id.to_string())
            .expect("the dense id must reparse");
        assert_eq!(
            slots_of(&reparsed).as_ref(),
            Some(&slots),
            "slots_of must read the dense interpretation"
        );

        let bytes = dense_bake_bytes(&id.to_string(), 265);
        let m = crate::mlp::Model::from_bytes(&bytes).expect("bake parses");
        assert_eq!(m.caller_input_width(), 265);
        assert_eq!(
            m.metadata().get_utf8("zentrain.feature_set_id").ok(),
            Some(id.to_string().as_str()),
            "the metadata entry must survive the bake"
        );
        let l = declared_layout(&m);
        assert!(!l.is_identity(), "must resolve to a DENSE layout");
        assert_eq!(l.width(), 265);
        assert_eq!(l.walk_width(), 942);
        assert_eq!(l.ids(), slots);
    }

    /// A 1-layer identity ZNPR v3 bake of `n` inputs carrying a declared
    /// feature-set id. Built through the mandated JSON pipeline.
    fn dense_bake_bytes(set_id: &str, n: usize) -> Vec<u8> {
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
                "metadata": [{{"key": "zentrain.feature_set_id", "type": "utf8", "text": "{set_id}"}}],
                "layers": [
                    {{"in_dim": {n}, "out_dim": 1, "activation": "identity",
                      "dtype": "f32", "weights": {w}, "biases": [0.0]}}
                ]
            }}"#,
            mean = arr(&vec![0.0f32; n]),
            scale = arr(&vec![1.0f32; n]),
            w = arr(&vec![1.0f32; n]),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic dense bake must build")
    }

    /// **G4.2, the bake half** — every bake that ships today resolves to an
    /// IDENTITY layout, so the declaration path cannot move a shipped byte.
    ///
    /// The normal case is a bake whose `zentrain.feature_set_id` names FEWER
    /// slots than its declared width (a `944`-with-structural-zeros producer
    /// names 265 at width 944); the filter refuses to read that as a dense
    /// layout, which is the whole safety argument.
    #[test]
    fn every_shipped_bake_resolves_to_an_identity_layout() {
        let mut n = 0usize;
        for (name, p) in crate::feature_plan::servability_census::shipped_profiles() {
            for bytes in p.params().scoring_bake_bytes() {
                let Ok(m) = crate::mlp::Model::from_bytes(bytes) else {
                    continue;
                };
                let l = declared_layout(&m);
                assert!(
                    l.is_identity(),
                    "{name}: a shipped bake resolved to a NON-identity layout \
                     ({}), which would change the vector it is served",
                    l.name()
                );
                assert_eq!(l.width(), m.caller_input_width());
                n += 1;
            }
        }
        assert!(n >= 5, "the census must actually see bakes, saw {n}");
    }
}
