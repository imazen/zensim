//! `zensim_validate::feature_set::resolve_features_root` — the EVAL features
//! root DERIVED FROM THE BAKE.
//!
//! **The defect.** `scripts/run_full_eval.sh` hard-coded one features root per
//! regime, so a bake trained at a non-default root had two outcomes and no
//! third: a wrong-regime read (which `bake_verdict` correctly REFUSES) or no
//! board cell at all. The casualty of record is **A3b**, the replication
//! wave's one genuinely-k=1 recipe, trained on `ext944-era2r4-2026-09-01`: it
//! scores 0.88–0.89 CID22 at its native root and had NO board row
//! (`benchmarks/replication_wave_2026-09-05.md` §4c.3, filed there as a named
//! gap because closing it needed a change at this owner).
//!
//! Every test below failed before `resolve_features_root` existed — the
//! function is the fix, and these pin the four resolution steps plus the two
//! refusals.

use std::path::{Path, PathBuf};

use zenpredict::MetadataType;
use zenpredict::{Activation, Model, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zensim_validate::feature_set::{self, RootSource};

const OUT: usize = 2;
const W1: [f32; OUT] = [0.7, -0.3];
const B1: [f32; 1] = [0.1];

/// A minimal 944-wide bake carrying `metadata` verbatim.
fn model_with(metadata: &[BakeMetadataEntry<'_>]) -> Model {
    let n_raw = 944usize;
    let mut w0 = vec![0.0f32; n_raw * OUT];
    for k in 0..156 {
        for o in 0..OUT {
            w0[k * OUT + o] = 0.05 + ((k * 31 + o * 7) % 97) as f32 / 100.0;
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
        schema_hash: 0x5eed_0000_c105_0002,
        flags: 0,
        scaler_mean: &mean,
        scaler_scale: &scale,
        layers: &layers,
        feature_bounds: &[],
        metadata,
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

fn repro_meta(json: &str) -> BakeMetadataEntry<'_> {
    BakeMetadataEntry {
        key: "zentrain.repro",
        kind: MetadataType::Utf8,
        value: json.as_bytes(),
    }
}

const REGIME_944_DEFAULT: &str = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01";
const ERA2R4: &str = "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01";

/// **The A3b case, verbatim.** A `--group name:path:w:w:mode` argv naming the
/// era-2×radius-4 root must resolve to that root, NOT to the regime default.
#[test]
fn a3b_shaped_repro_resolves_to_its_native_root() {
    // Trimmed from the real bake's embedded argv (A3b__S__i4004_p5001).
    let repro = format!(
        r#"{{"argv":["/x/target/release/zensim_mlp_train",
           "--group","safesyn:{ERA2R4}/recipe_views/safesyn_pure.parquet:1.0:0.5:both",
           "--group","cid22_train:{ERA2R4}/ext_cid22_train201.parquet:1.0:2.0:both",
           "--max-features","944"],"seed":4004}}"#
    );
    let m = model_with(&[repro_meta(&repro)]);
    let r = feature_set::resolve_features_root(&m, Path::new(REGIME_944_DEFAULT), None)
        .expect("A3b's root is determinable from its own repro");
    assert_eq!(r.root, PathBuf::from(ERA2R4));
    assert_eq!(r.source, RootSource::ReproTrainingPaths(ERA2R4.into()));
    assert_ne!(
        r.root,
        PathBuf::from(REGIME_944_DEFAULT),
        "resolving to the regime default is the defect, not the fix"
    );
}

/// A canonical-root bake resolves to the SAME path the regime default would
/// have used — so every existing harvest is behaviorally unchanged. Without
/// this the fix would be a silent re-rooting of the whole board.
#[test]
fn a_canonical_bake_resolves_to_the_regime_default_path() {
    let repro = format!(
        r#"{{"argv":["zensim_mlp_train",
           "--group","safesyn:{REGIME_944_DEFAULT}/recipe_views/safesyn_pure.parquet:1.0:0.5:both"],
           "seed":4021}}"#
    );
    let m = model_with(&[repro_meta(&repro)]);
    let r = feature_set::resolve_features_root(&m, Path::new(REGIME_944_DEFAULT), None).unwrap();
    assert_eq!(r.root, PathBuf::from(REGIME_944_DEFAULT));
}

/// A bake trained on a corpus that is NOT a registered features root (every
/// 372/720-era bake: `canonical-2026-05-21` is a training store, not an eval
/// root) resolves to the regime default — as a DETERMINATION carrying its
/// reason, never as an unexplained fallback.
#[test]
fn a_non_root_training_corpus_determines_the_regime_default() {
    let repro = r#"{"argv":["zensim_mlp_train","--group",
        "safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet:1.0:0.5"],
        "seed":7}"#;
    let m = model_with(&[repro_meta(repro)]);
    let r = feature_set::resolve_features_root(&m, Path::new(REGIME_944_DEFAULT), None).unwrap();
    assert_eq!(r.root, PathBuf::from(REGIME_944_DEFAULT));
    match &r.source {
        RootSource::RegimeDefault(why) => assert!(
            why.contains("no registered features root"),
            "the reason must be stated: {why}"
        ),
        other => panic!("expected a stated regime-default determination, got {other:?}"),
    }
    assert!(r.source.describe().contains("regime default"));
}

/// **The refusal.** A bake with no `zentrain.repro` cannot have its root
/// determined, and must be an ERROR — never a silent default. This is the
/// half of the rule that keeps the fix honest: a wrong-root read returns
/// plausible-looking numbers with no error (shipped B: CID22 0.3862 against a
/// true 0.8764), so guessing is worse than refusing.
#[test]
fn a_bake_with_no_repro_is_refused_not_defaulted() {
    let m = model_with(&[]);
    let e = feature_set::resolve_features_root(&m, Path::new(REGIME_944_DEFAULT), None)
        .expect_err("no repro ⇒ undeterminable");
    assert!(e.contains("zentrain.repro"), "the reason must name it: {e}");
    assert!(e.contains("--features-root"), "and name the escape: {e}");
    // ...and the explicit override still works on exactly that bake.
    let r = feature_set::resolve_features_root(
        &m,
        Path::new(REGIME_944_DEFAULT),
        Some(Path::new(ERA2R4)),
    )
    .expect("an explicit root is always honored");
    assert_eq!(r.root, PathBuf::from(ERA2R4));
    assert_eq!(r.source, RootSource::Explicit);
}

/// A repro spanning TWO registered roots is ambiguous. Picking one would be a
/// guess, so it refuses and names both.
#[test]
fn a_repro_spanning_two_registered_roots_is_refused() {
    let repro = format!(
        r#"{{"argv":["zensim_mlp_train",
           "--group","a:{ERA2R4}/ext_cid22_train201.parquet:1.0:1.0",
           "--group","b:{REGIME_944_DEFAULT}/ext_kadid.parquet:1.0:1.0"],"seed":1}}"#
    );
    let m = model_with(&[repro_meta(&repro)]);
    let e = feature_set::resolve_features_root(&m, Path::new(REGIME_944_DEFAULT), None)
        .expect_err("two roots ⇒ ambiguous");
    assert!(e.contains(ERA2R4) && e.contains(REGIME_944_DEFAULT), "{e}");
    assert!(e.contains("guess"), "it must say why it refuses: {e}");
}

/// The registry's `roots` table is the ONE owner of "which paths are features
/// roots", and the nested entry must win over its parent — otherwise a bake
/// trained on `…/era2r4/foldapp2_views` would be scored on the parent's
/// different slot set.
#[test]
fn a_nested_registered_root_wins_over_its_parent() {
    let nested = format!("{ERA2R4}/foldapp2_views");
    assert!(
        feature_set::registry().roots.contains_key(nested.as_str()),
        "precondition: the nested root is registered"
    );
    let repro =
        format!(r#"{{"argv":["zensim_mlp_train","--group","a:{nested}/ext_kadid.parquet:1:1"]}}"#);
    let m = model_with(&[repro_meta(&repro)]);
    let r = feature_set::resolve_features_root(&m, Path::new(REGIME_944_DEFAULT), None).unwrap();
    assert_eq!(
        r.root,
        PathBuf::from(&nested),
        "the longest registered prefix wins"
    );
}
