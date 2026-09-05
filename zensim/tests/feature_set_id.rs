//! `zensim::feature_set_id` — the feature-set identifier grammar, its ONE hash
//! owner, and the `ComputeSet` → COMPUTE-token mapping.
//!
//! Design: `docs/FEATURE_SET_IDS.md`. These are the failing-first gates for
//! the properties the design promises:
//!
//! * same slots / different era ⇒ DIFFERENT id (the ext944-vs-era2r4 pair of
//!   §3.1, which are bit-identical everywhere except `f720..923`);
//! * same era / different slots ⇒ DIFFERENT id (the `foldapp2pools`-vs-free-set
//!   pair, same width, same era family, zero shared non-basic slots);
//! * the canonical short form round-trips through `parse`;
//! * the hash is a SET hash — order and duplicates cannot move it;
//! * `SlotSet::covers` is the subset test the compatibility check rests on.

use zensim::feature_set_id::{ComputeParts, ComputeToken as T, FeatureSetId, SlotSet, slots_hash8};

/// The `ext944` / `era2r4` pair: identical compute, identical width, identical
/// populated slots — DIFFERENT extractor era. Measured difference between them
/// is `f720..923`, max abs 7.2e-9 (`~/tmp/gaddrinst_RESUME.md` finding 1), i.e.
/// a real numeric divergence a count cannot express.
#[test]
fn same_slots_different_era_is_a_different_id() {
    let compute = ComputeParts::EMPTY
        .with(T::Basic)
        .with(T::V2)
        .with(T::Append)
        .with(T::Append2);
    let slots = SlotSet::parse("0-155,372-943").unwrap();
    let a = FeatureSetId::from_slots(compute, 944, "ext944", &slots).unwrap();
    let b = FeatureSetId::from_slots(compute, 944, "era2r4", &slots).unwrap();
    assert_eq!(a.slots_hash(), b.slots_hash(), "same slots ⇒ same slot hash");
    assert_ne!(a, b, "a different era MUST be a different id");
    assert_ne!(a.to_string(), b.to_string());
    assert!(a.to_string().starts_with("basic+v2+append+append2@w944/ext944#"));
    assert!(b.to_string().starts_with("basic+v2+append+append2@w944/era2r4#"));
}

/// GOLDEN pin on the hash algorithm itself (design §2.4: FNV-1a/64 over the
/// canonical sorted slot rendering, folded by `NamedFeature::fold_hash`'s rule).
/// Filled from the implementation once; a change to the hash moves every
/// registered id, so it must never move silently.
#[test]
fn slot_hash_algorithm_is_pinned() {
    assert_eq!(zensim::feature_set_id::hex8(slots_hash8([0usize, 1, 2])), "d404bc88");
    assert_eq!(
        zensim::feature_set_id::hex8(SlotSet::parse("0-155,372-943").unwrap().hash8()),
        "7ed470b4"
    );
}

/// `foldapp2pools` vs the free-set arm: same width, same era, and their
/// non-basic slots do not intersect at all.
#[test]
fn same_era_different_slots_is_a_different_id() {
    let pools = FeatureSetId::from_slots(
        ComputeParts::EMPTY
            .with(T::Basic)
            .with(T::Peaks)
            .with(T::Masked)
            .with(T::Iw)
            .with(T::V2)
            .with(T::Append)
            .with(T::Append2),
        944,
        "pools",
        &SlotSet::parse("0-943").unwrap(),
    )
    .unwrap();
    let free = FeatureSetId::from_slots(
        ComputeParts::EMPTY.with(T::Basic).with(T::Peaks).with(T::Moments),
        944,
        "pools",
        &SlotSet::parse("0-227,733-735").unwrap(),
    )
    .unwrap();
    assert_eq!(pools.era(), free.era());
    assert_eq!(pools.layout_width(), free.layout_width());
    assert_ne!(pools.slots_hash(), free.slots_hash());
    assert_ne!(pools, free);
}

/// Two sets with the SAME canonical name and different slot lists must not
/// collide — the property that makes the name a handle and the hash the
/// identity (design §2.4).
#[test]
fn same_name_different_slots_cannot_collide() {
    let compute = ComputeParts::EMPTY.with(T::Basic).with(T::Peaks).with(T::Moments);
    let with_all_free = FeatureSetId::from_slots(
        compute,
        944,
        "era2r4",
        &SlotSet::parse("0-227,733-735,750-752").unwrap(),
    )
    .unwrap();
    let with_some_free =
        FeatureSetId::from_slots(compute, 944, "era2r4", &SlotSet::parse("0-227,733-735").unwrap())
            .unwrap();
    assert_eq!(with_all_free.compute(), with_some_free.compute());
    assert_ne!(with_all_free, with_some_free);
}

#[test]
fn canonical_form_round_trips() {
    for s in [
        "basic@w372/v1cur#00000000",
        "basic+peaks+masked+iw@w372/v1pre#deadbeef",
        "basic+peaks+moments+classc@w944/era2r4#0123abcd",
        "basic+carriers+v2+append+append2@w944/pools#ffffffff",
    ] {
        let id = FeatureSetId::parse(s).unwrap_or_else(|| panic!("parse {s}"));
        assert_eq!(id.to_string(), s);
    }
    // Strict, exactly like `zenanalyze_api::NamedFeature::parse`.
    assert!(FeatureSetId::parse("basic@w944/era2r4#ABCDEF01").is_none(), "uppercase hex");
    assert!(FeatureSetId::parse("basic@w944/era2r4#abcdef0").is_none(), "7 digits");
    assert!(FeatureSetId::parse("basic@w944/era2r4#abcdef012").is_none(), "9 digits");
    assert!(FeatureSetId::parse("basic@w944/era2r4").is_none(), "no hash");
    assert!(FeatureSetId::parse("basic@944/era2r4#abcdef01").is_none(), "no w prefix");
    assert!(FeatureSetId::parse("basic@w944/Era2R4#abcdef01").is_none(), "era charset");
    assert!(FeatureSetId::parse("basic+nope@w944/era2r4#abcdef01").is_none(), "unknown token");
}

/// The hash is over a SET: sorted + de-duplicated before hashing, so two
/// producers that walk their slots in different internal orders agree.
#[test]
fn slot_hash_is_order_and_duplicate_insensitive() {
    let a = slots_hash8([5usize, 1, 9, 1, 5]);
    let b = slots_hash8([1usize, 5, 9]);
    assert_eq!(a, b);
    assert_ne!(slots_hash8([1usize, 5, 9]), slots_hash8([1usize, 5, 10]));
    // The empty set has a defined, stable hash (FNV-1a/64 offset basis, folded).
    assert_eq!(slots_hash8(core::iter::empty()), slots_hash8(core::iter::empty()));
}

#[test]
fn slot_set_normalizes_and_renders_compactly() {
    let s = SlotSet::from_slots([3usize, 1, 2, 0, 7, 6]);
    assert_eq!(s.to_string(), "0-3,6-7");
    assert_eq!(s.len(), 6);
    assert_eq!(SlotSet::parse("0-3,6-7").unwrap(), s);
    // A single-slot run renders bare, and round-trips.
    let one = SlotSet::from_slots([42usize]);
    assert_eq!(one.to_string(), "42");
    assert_eq!(SlotSet::parse("42").unwrap(), one);
    assert_eq!(SlotSet::default().to_string(), "");
    assert_eq!(SlotSet::parse("").unwrap(), SlotSet::default());
    assert!(SlotSet::parse("7-3").is_none(), "descending range");
}

/// The subset test the §4 compatibility check rests on: shipped B reads 49
/// lines in `f156..371`; the `ext944` root populates NONE of them.
#[test]
fn covers_is_the_subset_test_that_catches_the_regime_944_bug() {
    let ext944 = SlotSet::parse("0-155,372-943").unwrap();
    let b_reads = SlotSet::parse("0-155,300-310").unwrap();
    assert!(!ext944.covers(&b_reads), "the block B reads is NOT populated at ext944");
    let missing = ext944.missing_from(&b_reads);
    assert_eq!(missing.to_string(), "300-310");
    let add156_reads = SlotSet::parse("0-27").unwrap();
    assert!(ext944.covers(&add156_reads), "a basic-only reader is safe at any root");
    assert!(ext944.covers(&ext944));
    assert!(SlotSet::default().is_empty());
    assert!(ext944.covers(&SlotSet::default()));
}

#[test]
fn compute_parts_render_in_registry_order_regardless_of_insertion_order() {
    let a = ComputeParts::EMPTY.with(T::Append2).with(T::Basic).with(T::Peaks);
    let b = ComputeParts::EMPTY.with(T::Peaks).with(T::Append2).with(T::Basic);
    assert_eq!(a, b);
    assert_eq!(a.to_string(), "basic+peaks+append2");
    assert_eq!(ComputeParts::parse("basic+peaks+append2"), Some(a));
    assert_eq!(ComputeParts::parse("append2+peaks+basic"), Some(a), "parse is order-tolerant");
    assert_eq!(ComputeParts::EMPTY.to_string(), "none");
    assert_eq!(ComputeParts::parse("none"), Some(ComputeParts::EMPTY));
    assert!(a.contains(T::Basic) && !a.contains(T::Iw));
}
