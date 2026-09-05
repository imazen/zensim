//! `zensim_validate::feature_set` — deriving a feature-set id for a BAKE and
//! for a FEATURES ROOT, and the compatibility check that closes the
//! `--regime 944` silent-mis-scoring bug at the root.
//!
//! Design: `docs/FEATURE_SET_IDS.md`. Registry:
//! `benchmarks/feature_sets_registry.json`.
//!
//! The class under test has two published instances: `ebothg_m504` (a
//! wrong-root board row) and appendix U.R0 — shipped **B**, a 372-input bake
//! with 49 structurally-used lines in `f156..371`, scored at `--regime 944`
//! where that block is STRUCTURAL ZEROS, reading CID22 **0.3862** against its
//! true **0.8764**. No error, no warning, a plausible number half a point low.

use std::path::Path;
use zenpredict::{Activation, Model, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};
use zensim::feature_set_id::{ComputeToken as T, SlotSet};
use zensim_validate::feature_set;

const OUT: usize = 2;
const W1: [f32; OUT] = [0.7, -0.3];
const B1: [f32; 1] = [0.1];

/// Single hidden layer + identity head over `n_raw` caller lines; layer-0
/// weight on caller line `k` iff `live(k)`.
fn build_model(n_raw: usize, live: impl Fn(usize) -> bool) -> Model {
    let mut w0 = vec![0.0f32; n_raw * OUT];
    for k in 0..n_raw {
        if live(k) {
            for o in 0..OUT {
                w0[k * OUT + o] = 0.05 + ((k * 31 + o * 7) % 97) as f32 / 100.0;
            }
        }
    }
    let b0 = vec![0.0f32; OUT];
    let mean = vec![0.0f32; n_raw];
    let scale = vec![1.0f32; n_raw];
    let layers = [
        BakeLayer {
            in_dim: n_raw,
            out_dim: OUT,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: OUT,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &W1,
            biases: &B1,
        },
    ];
    let bytes = bake(&BakeRequest {
        schema_hash: 0x5eed_0000_c105_0001,
        flags: 0,
        scaler_mean: &mean,
        scaler_scale: &scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake");
    Model::from_bytes(&bytes).expect("parse")
}

/// **THE bug.** A 372-wide bake that structurally reads `f156..371` (shipped
/// B's shape: 49 live lines in that block), checked against the `ext944` root
/// whose f156..371 is structural zeros.
#[test]
fn a_372_bake_reading_f156_371_mismatches_the_944_root() {
    let m = build_model(372, |k| k < 156 || (300..349).contains(&k));
    let bake_ref = feature_set::bake_feature_set_ref(&m, "v1cur").expect("derive");
    assert!(
        bake_ref.id.compute().contains(T::Iw),
        "the bake reads the IW block"
    );
    let root = feature_set::registry()
        .set("basic+v2+append+append2@w944/ext944")
        .expect("registered")
        .as_ref()
        .expect("a pinned set has slots");

    let ms = feature_set::check(&bake_ref, root);
    assert!(!ms.is_empty(), "reading a zeroed block MUST be reported");
    let joined = ms
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        joined.contains("300-348"),
        "must name the unpopulated slots: {joined}"
    );
    assert!(
        ms.iter()
            .any(|m| m.kind == feature_set::MismatchKind::SlotsNotPopulated),
        "the slot-coverage mismatch is the load-bearing one: {joined}"
    );
}

/// A basic-only reader (ADD156's shape) is safe at every root that populates
/// `f0..155` — the T.R4 bridge cell measured ≤0.0008 cross-root drift for
/// exactly this shape.
#[test]
fn a_basic_only_bake_is_compatible_with_every_registered_producer() {
    let m = build_model(372, |k| k < 28);
    let bake_ref = feature_set::bake_feature_set_ref(&m, "v1cur").expect("derive");
    assert_eq!(bake_ref.id.compute().to_string(), "basic");
    for (id, set) in feature_set::registry().sets() {
        let Some(p) = set.as_ref() else { continue };
        if p.id.layout_width() < 372 {
            continue;
        }
        let slot_fails = feature_set::check(&bake_ref, p)
            .into_iter()
            .filter(|m| m.kind == feature_set::MismatchKind::SlotsNotPopulated)
            .count();
        assert_eq!(
            slot_fails, 0,
            "basic-only must be slot-compatible with {id}"
        );
    }
}

/// Same slots, different era ⇒ still reported, because the values differ even
/// though the layout and the populated set do not.
#[test]
fn an_era_difference_is_reported_even_when_the_slots_match() {
    let m = build_model(372, |k| k < 156);
    let a = feature_set::bake_feature_set_ref(&m, "v1cur").expect("derive");
    let root = feature_set::registry()
        .set("basic+peaks+masked+iw@w372/v1pre")
        .expect("registered")
        .as_ref()
        .expect("pinned");
    let ms = feature_set::check(&a, root);
    assert!(
        ms.iter()
            .any(|m| m.kind == feature_set::MismatchKind::EraDiffers)
    );
    assert!(
        !ms.iter()
            .any(|m| m.kind == feature_set::MismatchKind::SlotsNotPopulated)
    );
}

/// An UNKNOWN era is never silently a match.
#[test]
fn an_unknown_era_is_reported_not_waved_through() {
    let m = build_model(372, |k| k < 156);
    let a =
        feature_set::bake_feature_set_ref(&m, zensim::feature_set_id::ERA_UNKNOWN).expect("derive");
    let root = feature_set::registry()
        .set("basic+peaks+masked+iw@w372/v1cur")
        .expect("registered")
        .as_ref()
        .expect("pinned");
    assert!(
        feature_set::check(&a, root)
            .iter()
            .any(|m| m.kind == feature_set::MismatchKind::EraUnknown)
    );
}

/// The committed registry is checked against the OWNER: every pinned set's
/// stored `slots_hash8` must equal `zensim::feature_set_id::slots_hash8` over
/// its own `slots`, and its canonical id string must render back to its key.
#[test]
fn every_registered_set_agrees_with_the_hash_owner() {
    let reg = feature_set::registry();
    let mut pinned = 0usize;
    for (key, set) in reg.sets() {
        assert_eq!(
            set.id.to_string_class(),
            key.split('#').next().unwrap(),
            "the registry key must be the canonical id (modulo the hash)"
        );
        if let Some(p) = set.as_ref() {
            pinned += 1;
            assert_eq!(
                p.slots.hash8(),
                p.id.slots_hash(),
                "{key}: stored hash disagrees with the owner"
            );
            assert_eq!(
                p.slots.len(),
                set.n_slots.expect("pinned sets carry n_slots")
            );
            assert!(
                p.id.layout_width() >= p.slots.ranges().last().map(|r| r.1).unwrap_or(0),
                "{key}: slots overflow the declared layout"
            );
        }
    }
    assert!(
        pinned >= 12,
        "the registry must pin every set of the doc's §3 table"
    );
    // Every alias target resolves.
    for (name, ids) in reg.aliases() {
        for id in ids {
            assert!(reg.set(id).is_some(), "alias {name} -> unregistered {id}");
        }
    }
    // "944" is the ambiguous one, by construction.
    assert!(
        reg.aliases_for("944").len() >= 6,
        "944 has had at least six meanings"
    );
    assert_eq!(reg.aliases_for("720").len(), 1);
}

/// A features root resolves through its `_MANIFEST.json` `feature_set_id` key
/// when it has one, and through the registry's root/regime tables when it does
/// not — the latter marked `inferred`.
#[test]
fn a_registered_root_path_resolves_and_is_marked_inferred() {
    let p = Path::new("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01");
    let Some(r) = feature_set::root_feature_set_ref(p) else {
        // The root is on block storage; skip only when the REGISTRY itself
        // cannot answer, which would be a registry defect, not a missing disk.
        panic!("the registry alone must resolve a registered root path");
    };
    assert_eq!(
        r.id.to_string(),
        "basic+v2+append+append2@w944/ext944#7ed470b4"
    );
    assert!(
        r.inferred,
        "a root with no stored id is INFERRED, never asserted"
    );
    assert!(
        !r.slots.contains(300),
        "f156..371 is not populated at ext944"
    );
    assert!(r.slots.covers(&SlotSet::parse("0-155").unwrap()));
}
