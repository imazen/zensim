//! GATE: `zentrain.repro` must record the SPLIT seed streams.
//!
//! `--init-seed` / `--sample-seed` (2026-09-04) exist so a study can hold the
//! drawn pair subset fixed while varying the weight init, or the reverse.
//! The consumers of that split — `gauntlet.seed_identity` and its Rust mirror
//! `freeze_check::seed_identity` — decide how many DISTINCT draws a seed group
//! holds by reading `repro.init_seed` / `repro.sample_seed`, falling back to
//! `repro.seed`.
//!
//! The READER shipped before the WRITER. Until 2026-09-05 the trainer's repro
//! block emitted only `"seed": args.seed`, and `--seed` defaults to **1**, so
//! every split-seed run recorded the same `seed: 1` and no split keys at all.
//! `seed_identity` therefore returned `"1"` for every arm of a k-arm study and
//! `build_seed_groups` collapsed the whole study into ONE identity — k=1, the
//! exact quantity such a study exists to measure. Caught while registering
//! `benchmarks/replication_wave_2026-09-05.md`, before any fit landed.
//!
//! This is a source gate rather than a runtime one because the repro JSON is a
//! literal inside one long function in a binary; the cheap, honest check is
//! that the two keys are emitted from the args at all. Runtime behaviour is
//! covered by `freeze_check`'s `seed_identity_reports_the_pair`.
//!
//! Mutation-verified: deleting either `"init_seed": args.init_seed` or
//! `"sample_seed": args.sample_seed` from the repro block fails this test.

use std::path::PathBuf;

fn trainer_source() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/zensim_mlp_train.rs");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

#[test]
fn repro_block_emits_both_split_seed_streams() {
    let src = trainer_source();
    for key in ["init_seed", "sample_seed"] {
        let emit = format!("\"{key}\": args.{key},");
        assert!(
            src.contains(&emit),
            "zensim_mlp_train's zentrain.repro block must emit `{emit}` — without it \
             every --{} run records only `seed` (default 1) and an entire k-arm seed \
             study collapses to k=1 under seed_identity. See \
             benchmarks/replication_wave_2026-09-05.md.",
            key.replace('_', "-")
        );
    }
}

#[test]
fn repro_block_still_emits_the_legacy_seed_for_fallback() {
    // The fallback is what keeps every pre-2026-09-04 bake groupable; the split
    // keys are ADDITIVE, and removing `seed` would silently orphan them.
    let src = trainer_source();
    assert!(
        src.contains("\"seed\": args.seed,"),
        "the legacy `seed` key must stay: seed_identity falls back to it for every \
         bake trained with --seed alone"
    );
}
