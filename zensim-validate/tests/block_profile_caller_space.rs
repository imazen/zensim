//! Block-profile family classification happens in CALLER space — the
//! regression tests for the caller-width bug, instance #4 of the class.
//!
//! The bug (found 2026-08-05 by a read-only verification session): the
//! original `bake_block_profile` sliced the model's INTERNAL layer-0
//! columns (`n_inputs()` = 667 on a pruned bake) at CALLER-family
//! boundaries (f0-155 / f156-371 / f372-719 / f720-943). On the first real
//! pruned candidate (`W10L9_s4003_packed`, 944 → 667 via `Drop`
//! transforms) that reported:
//!
//! ```text
//! f0_155     156  0  0  156          ← right by luck (no drops below 156)
//! f156_371   216  0  0  216          ← FALSE: these are v2 columns shifted down
//! f372_719   295  0  0  295          ← 295-wide phantom family
//! (no f720_943 at all)
//! uses_f156_371: true                ← FALSE (parent: 216/216 exact-zero)
//! ```
//!
//! while the unpruned parent's ground truth is f156-371 = 216/216
//! exact-zero, uses = false, f720-943 = 61 exact-zero / 163 used. The
//! `Drop` entries in the bake's dense `feature_transforms` define the
//! internal→caller mapping; `zensim_validate::block_profile` now walks
//! them (the same arity walk as `Model::caller_input_width`), which is
//! what these tests pin.
//!
//! Instance #3 of the same class is documented in `prune_classes.rs`
//! (`pruned_bake_stays_routable_by_caller_feature_width` — the coherence
//! harness dispatching regimes on `n_inputs()`).
//!
//! Skip semantics: the `real_sota944_*` test is `#[ignore]`d because it
//! needs the `/mnt/v` sota944 bake artifacts; the skip is caller-controlled
//! (run with `-- --ignored`). Everything else is hermetic.

use zenpredict::{Activation, MetadataType, Model, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zensim_validate::block_profile::{self, NEAR_ZERO_REL};
use zensim_validate::prune;

const N_RAW: usize = 944;
const OUT: usize = 2;

/// Caller lines that are exactly zero in the parent (class-1 dead): the
/// whole f156-371 block plus a 61-line run in the append block — mirroring
/// what the 944-regime trainers actually emit (216 + 61) — plus one line
/// each in v1-basic and v2 so every family exercises the mapping.
fn is_dead(k: usize) -> bool {
    (156..372).contains(&k) || (800..861).contains(&k) || k == 10 || k == 500
}

/// One near-zero line (nonzero but ≤ `NEAR_ZERO_REL` × max caller norm) in
/// v2. NOT prunable — class 1 requires exact zeros — so it must survive
/// pruning and count as `near_zero` in BOTH profiles.
const NEAR_LINE: usize = 600;

fn parent_weights() -> Vec<f32> {
    let mut w = vec![0.0f32; N_RAW * OUT];
    for k in 0..N_RAW {
        if is_dead(k) {
            continue;
        }
        for o in 0..OUT {
            w[k * OUT + o] = if k == NEAR_LINE {
                1e-9
            } else {
                0.05 + ((k * 31 + o * 7) % 97) as f32 / 100.0
            };
        }
    }
    w
}

const W1: [f32; OUT] = [0.7, -0.3];
const B1: [f32; 1] = [0.1];

fn build(mean: &[f32], scale: &[f32], w0: &[f32], b0: &[f32], md: &[BakeMetadataEntry]) -> Vec<u8> {
    let layers = [
        BakeLayer {
            in_dim: mean.len(),
            out_dim: OUT,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: w0,
            biases: b0,
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
    bake(&BakeRequest {
        schema_hash: 0x5eed_0000_b10c_0000,
        flags: 0,
        scaler_mean: mean,
        scaler_scale: scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake fixture")
}

/// (parent bytes, pruned-twin bytes) — the twin is produced by the same
/// `prune` machinery `bake_dial_refit pack` uses (class-1 only, so the
/// retained weights are bit-identical to the parent's).
fn fixture_pair() -> (Vec<u8>, Vec<u8>) {
    let w0 = parent_weights();
    let b0 = [0.05f32, -0.15];
    let mean = vec![0.0f32; N_RAW];
    let scale = vec![1.0f32; N_RAW];
    let parent = build(&mean, &scale, &w0, &b0, &[]);

    let model = Model::from_bytes(&parent).expect("load parent");
    let l0 = prune::Layer0View {
        in_dim: N_RAW,
        out_dim: OUT,
        weights: &w0,
        biases: &b0,
        is_i8: false,
    };
    let plan = prune::plan(&model, &l0, false).expect("plan");
    assert_eq!(
        plan.drop.len(),
        216 + 61 + 2,
        "fixture must prune 279 lines"
    );
    let pw = prune::prune_layer0_weights(&plan, &w0, OUT);
    let pb = prune::prune_layer0_biases(&plan, &b0);
    let pmean = prune::prune_input_array(&plan, model.scaler_mean());
    let pscale = prune::prune_input_array(&plan, model.scaler_scale());
    let (t_txt, p_txt) = prune::transform_metadata(&plan);
    let md = [
        BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: t_txt.as_bytes(),
        },
        BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: p_txt.as_bytes(),
        },
    ];
    let pruned = build(&pmean, &pscale, &pw, &pb, &md);
    (parent, pruned)
}

#[test]
fn pruned_fixture_profiles_identically_to_its_unpruned_parent() {
    let (parent, pruned) = fixture_pair();
    let parent_model = Model::from_bytes(&parent).expect("load parent");
    let pruned_model = Model::from_bytes(&pruned).expect("load pruned");

    // Vacuity guard: the twin must actually be pruned.
    assert_eq!(pruned_model.caller_input_width(), N_RAW);
    assert_eq!(pruned_model.n_inputs(), N_RAW - 279);

    let pp = block_profile::profile(&parent_model).expect("profile parent");
    let qp = block_profile::profile(&pruned_model).expect("profile pruned");

    // The ground truth both must report, family by family:
    // (label, cols, exact_zero, near_zero, used)
    let expect: &[(&str, usize, usize, usize, usize)] = &[
        ("f0_155", 156, 1, 0, 155),
        ("f156_371", 216, 216, 0, 0),
        ("f372_719", 348, 1, 1, 346),
        ("f720_943", 224, 61, 0, 163),
    ];
    let pc: Vec<_> = pp.families.iter().map(|f| f.counts()).collect();
    let qc: Vec<_> = qp.families.iter().map(|f| f.counts()).collect();
    assert_eq!(pc, expect, "parent profile");
    assert_eq!(qc, expect, "pruned profile — must equal the parent's");
    assert!(!pp.uses_f156_371);
    assert!(
        !qp.uses_f156_371,
        "the exact false-positive the bug produced"
    );

    // Class-1 pruning keeps retained weights bit-identical and dropped
    // lines fold to an exactly-0.0 norm, so the FULL family stats —
    // including per-family max norms — are equal, not merely the counts.
    assert_eq!(pp.families, qp.families);

    // The shape of the bug, asserted away: internal-space slicing yielded
    // a 295-wide "f372_719" and no f720_943 family at all.
    assert_eq!(qp.families.len(), 4, "all four families must be present");
    assert_eq!(
        qp.families[2].cols, 348,
        "f372_719 is 348 caller lines, not 295"
    );
    assert_eq!(qp.n_dropped, 279);
    assert_eq!(qp.caller_input_width, N_RAW);
    assert_ne!(qp.n_inputs, qp.caller_input_width);
    assert_eq!(pp.n_dropped, 0);
    assert_eq!(qp.beyond_f943_cols, 0);
}

#[test]
fn json_carries_the_board_consumed_fields() {
    // promote_fulleval.py stores this JSON verbatim as `block_profile`;
    // gauntlet.py reads `uses_f156_371` + `families.*.{used,cols}` and
    // freeze_check reads `families.*.used` — those names are load-bearing.
    let (_, pruned) = fixture_pair();
    let model = Model::from_bytes(&pruned).expect("load pruned");
    let j = block_profile::profile(&model).expect("profile").to_json();
    for key in [
        "\"uses_f156_371\":false",
        "\"caller_input_width\":944",
        "\"n_dropped\":279",
        "\"n_inputs\":665",
        "\"f156_371\":{\"cols\":216,\"exact_zero\":216,\"near_zero\":0,\"used\":0,",
        "\"f720_943\":{\"cols\":224,\"exact_zero\":61,\"near_zero\":0,\"used\":163,",
    ] {
        assert!(j.contains(key), "JSON missing `{key}`:\n{j}");
    }
    // sanity on the shared constant the near-zero band derives from
    assert!(j.contains(&format!("\"near_zero_rel\":{NEAR_ZERO_REL:e}")));
}

#[test]
fn sinusoidal_expansion_folds_into_one_caller_line() {
    // The other variable-arity transform: one caller line expanding to 2·N
    // internal columns must still count as ONE caller line in its family.
    let transforms = "identity\nsinusoidal\nidentity";
    let params = "\n1,2\n";
    let md = [
        BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: transforms.as_bytes(),
        },
        BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params.as_bytes(),
        },
    ];
    // in_dim = 1 + 4 + 1 = 6
    let w0: [f32; 6 * OUT] = [
        0.5, 0.5, // caller 0
        0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, // caller 1 (sin/cos × 2 freqs)
        0.25, 0.25, // caller 2
    ];
    let b0 = [0.0f32, 0.0];
    let mean = vec![0.0f32; 6];
    let scale = vec![1.0f32; 6];
    let bytes = build(&mean, &scale, &w0, &b0, &md);
    let model = Model::from_bytes(&bytes).expect("load");
    assert_eq!(model.n_inputs(), 6);
    assert_eq!(model.caller_input_width(), 3);

    let p = block_profile::profile(&model).expect("profile");
    assert_eq!(p.families.len(), 1, "3 caller lines all live in f0_155");
    assert_eq!(p.families[0].counts(), ("f0_155", 3, 0, 0, 3));
    // Same arithmetic as the profiler: f32 weights widened to f64, then
    // squared/summed (an f64-literal 0.1² differs from (0.1f32 as f64)² at
    // ~1.5e-9 relative — enough to matter at a tight tolerance).
    let expected: f64 = w0[2..10]
        .iter()
        .map(|&w| (w as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        (p.families[0].max_col_norm - expected).abs() < 1e-12,
        "expanded line folds ALL its internal columns into one caller norm: \
         got {}, want {expected}",
        p.families[0].max_col_norm
    );
    assert_eq!(p.n_dropped, 0);
    assert_eq!(p.beyond_f943_cols, 0, "expansion is not caller width");
}

/// The three real sota944 packed candidates against their unpruned `_dial`
/// parents. `W10L9_s4003_packed` is genuinely pruned (277 drops, packed
/// after prune-by-default landed 2026-08-04); the other two shipped packed
/// bakes predate it (`pack --no-prune` reproduces them byte-for-byte per
/// `benchmarks/dead_column_pruning_2026-08-04.md`) — either way the packed
/// twin's STRUCTURAL profile must equal its parent's (norms are excluded:
/// the twins are f16, the parents f32).
#[test]
#[ignore = "needs the /mnt/v sota944 bake artifacts; run with -- --ignored"]
fn real_sota944_packed_candidates_profile_like_their_unpruned_parents() {
    const DIR: &str = "/mnt/v/output/zensim/bakes/sota944/bakes";
    let pairs = [
        ("W10L9_s4003_packed.bin", "W10L9_s4003_dial.bin"),
        ("H_co3abpg_s2507_packed.bin", "H_co3abpg_s2507_dial.bin"),
        ("C_em944_s31_packed.bin", "C_em944_s31_dial.bin"),
    ];
    let mut saw_pruned = false;
    for (packed, parent) in pairs {
        let pb = std::fs::read(format!("{DIR}/{packed}")).expect(packed);
        let db = std::fs::read(format!("{DIR}/{parent}")).expect(parent);
        let pm = Model::from_bytes(&pb).expect(packed);
        let dm = Model::from_bytes(&db).expect(parent);
        let pp = block_profile::profile(&pm).unwrap_or_else(|e| panic!("{packed}: {e}"));
        let dp = block_profile::profile(&dm).unwrap_or_else(|e| panic!("{parent}: {e}"));

        assert_eq!(pp.caller_input_width, 944, "{packed}");
        assert_eq!(dp.caller_input_width, 944, "{parent}");
        if pp.n_dropped > 0 {
            saw_pruned = true;
            assert_ne!(pp.n_inputs, pp.caller_input_width, "{packed}");
        }

        let pc: Vec<_> = pp.families.iter().map(|f| f.counts()).collect();
        let dc: Vec<_> = dp.families.iter().map(|f| f.counts()).collect();
        assert_eq!(pc, dc, "{packed} structural profile must match {parent}");
        assert_eq!(pp.uses_f156_371, dp.uses_f156_371, "{packed}");

        // The measured ground truth on every 944-regime candidate: the
        // whole zeroed block is exact-zero — the exact fact the buggy
        // internal-space slicing inverted on the pruned bake.
        let z = pp
            .families
            .iter()
            .find(|f| f.label == "f156_371")
            .expect("f156_371 present");
        assert_eq!(z.counts(), ("f156_371", 216, 216, 0, 0), "{packed}");
        assert!(!pp.uses_f156_371, "{packed}");
    }
    assert!(
        saw_pruned,
        "no candidate carries Drop transforms — the pair list no longer \
         exercises the caller-width fix (W10L9_s4003_packed carried 277 \
         drops as of 2026-08-05); point it at a pruned artifact"
    );
}
