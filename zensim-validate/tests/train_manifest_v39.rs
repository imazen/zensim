//! Integration test for the `--manifest` reproduce-this input mode of
//! `zensim_mlp_train` (see `src/train_manifest.rs`).
//!
//! Asserts the manifest→config mapping against the REAL shipped V39
//! manifest (`zensim/weights/manifests/v39_v32plus_spline_seed17_2026-05-25.toml`)
//! and exercises the sha-verification logic with a synthetic manifest +
//! a temp file whose bytes we control. Does NOT run a full training —
//! that is far too slow for a unit test; the mapping + verification are
//! the load-bearing pieces this test guards.

use std::path::{Path, PathBuf};

use zensim_validate::train_manifest::{
    self, ManifestError, ManifestInput, parse_manifest, parse_manifest_str, sha256_file,
    verify_inputs,
};

/// Path to the real shipped V39 manifest, relative to this crate dir.
fn v39_manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../zensim/weights/manifests/v39_v32plus_spline_seed17_2026-05-25.toml")
}

#[test]
fn v39_manifest_maps_recorded_training_fields() {
    let path = v39_manifest_path();
    assert!(
        path.exists(),
        "real V39 manifest missing at {} — test fixture moved?",
        path.display()
    );
    let cfg = parse_manifest(&path).expect("parse V39 manifest");

    // --- structured [training] hyperparameters, verbatim from the manifest ---
    assert_eq!(cfg.seed, Some(17), "seed");
    assert_eq!(cfg.hidden, Some(128), "hidden");
    assert_eq!(cfg.n_hidden_layers, Some(2), "n_hidden_layers");
    assert_eq!(cfg.epochs, Some(200), "epochs");
    assert_eq!(cfg.pairs_per_epoch, Some(50_000), "pairs_per_epoch");
    assert_eq!(cfg.lr, Some(0.001), "lr");
    assert_eq!(cfg.l2, Some(0.0001), "l2");
    assert_eq!(cfg.leaky_alpha, Some(0.01), "leaky_alpha");
    assert_eq!(cfg.max_features, Some(372), "max_features");
    assert_eq!(cfg.minibatch_size, Some(32), "minibatch_size");
    assert_eq!(cfg.out_dtype.as_deref(), Some("f32"), "out_dtype");
    assert_eq!(
        cfg.val_aggregate.as_deref(),
        Some("geomean3"),
        "val_aggregate"
    );
    assert_eq!(
        cfg.target_column.as_deref(),
        Some("human_score"),
        "target_column"
    );
    assert_eq!(cfg.target_scale, Some(1.0), "target_scale");
    assert_eq!(
        cfg.per_sample_alpha_head,
        Some(true),
        "per_sample_alpha_head"
    );
    assert_eq!(cfg.mse_weight, Some(0.6), "mse_weight");
    assert_eq!(cfg.ranknet_weight, Some(0.6), "ranknet_weight");
    assert_eq!(cfg.monotonicity_reg, Some(1.0), "monotonicity_reg");
    assert_eq!(cfg.monotonicity_margin, Some(0.0), "monotonicity_margin");
    assert_eq!(
        cfg.tanh_output_head_scale,
        Some(30.0),
        "tanh_output_head_scale"
    );
    assert_eq!(cfg.anchor_loss_weight, Some(0.01), "anchor_loss_weight");
    assert_eq!(cfg.anchor_step_p, Some(0.05), "anchor_step_p");
    // anchor_target_score = "PER-ROW (multi-band)" (string) → not mapped
    // to a scalar (the anchor is per-row, driven by anchor_parquet).
    assert_eq!(
        cfg.anchor_target_score, None,
        "anchor_target_score (per-row → None)"
    );

    // --- groups: the 5-group V32 structure, name/weights verbatim ---
    let names: Vec<&str> = cfg.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["safesyn", "cid22_train", "kadid", "tid", "konjnd_dense"],
        "group names + order"
    );
    let weights: Vec<(f64, f64)> = cfg.groups.iter().map(|g| (g.train_w, g.val_w)).collect();
    assert_eq!(
        weights,
        vec![(1.0, 0.5), (1.5, 2.0), (0.5, 1.0), (0.5, 1.0), (1.2, 1.5)],
        "group train_w:val_w"
    );

    // {canonical} resolves against [inputs.canonical_root].local.
    let canonical_root = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train";
    assert_eq!(
        cfg.groups[0].path,
        PathBuf::from(format!("{canonical_root}/safesyn.parquet")),
        "safesyn path resolves {{canonical}}"
    );
    assert_eq!(
        cfg.groups[1].path,
        PathBuf::from(format!("{canonical_root}/cid22_train_norm.parquet")),
        "cid22_train path"
    );

    // The --group spec round-trips into the trainer's flag form.
    assert_eq!(
        cfg.groups[0].to_group_spec(),
        format!("safesyn:{canonical_root}/safesyn.parquet:1:0.5"),
        "group spec render"
    );

    // anchor_parquet resolves {canonical} too.
    assert_eq!(
        cfg.anchor_parquet,
        Some(PathBuf::from(format!(
            "{canonical_root}/multiband_anchor_dial100.parquet"
        ))),
        "anchor_parquet"
    );

    // auto_transforms is repo-relative (../../../benchmarks/...) →
    // resolved against the manifest's parent dir.
    let at = cfg
        .auto_transforms
        .as_ref()
        .expect("auto_transforms present");
    assert!(
        at.ends_with(
            "benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
        ),
        "auto_transforms resolved path tail, got {}",
        at.display()
    );

    // --out from [bake].file resolves relative to the manifest dir.
    let out = cfg.out.as_ref().expect("out from [bake].file");
    assert!(
        out.ends_with("v39_v32plus_spline_seed17_2026-05-25.bin"),
        "out path tail, got {}",
        out.display()
    );

    // --- post-training steps surfaced (spline injection is step 2) ---
    assert_eq!(cfg.post_training_steps.len(), 2, "two post-training steps");
    assert!(
        cfg.post_training_steps[1].contains("inject_spline"),
        "step 2 is the spline injection, got {:?}",
        cfg.post_training_steps[1]
    );

    // --- inputs: every [inputs.<name>] carrying a sha256 collected,
    // with canonical_root's R2/Tower mirror inherited for {canonical}
    // paths. Sorted by key. ---
    let keys: Vec<&str> = cfg.inputs.iter().map(|i| i.key.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "auto_transforms",
            "cid22_train",
            "kadid",
            "konjnd_dense",
            "multiband_anchor",
            "safesyn",
            "tid",
        ],
        "input keys carrying sha256 (sorted)"
    );

    let safesyn = cfg.inputs.iter().find(|i| i.key == "safesyn").unwrap();
    assert_eq!(
        safesyn.sha256, "ad15cc79cde156109d920bfade8ed465908e256fcd1cd8556b430791dcaf1b18",
        "safesyn sha256"
    );
    assert_eq!(safesyn.rows, Some(196_086), "safesyn rows");
    assert_eq!(
        safesyn.r2.as_deref(),
        Some("s3://zentrain/canonical-2026-05-21/train"),
        "safesyn inherits canonical_root R2 mirror"
    );
    assert_eq!(
        safesyn.tower.as_deref(),
        Some("/mnt/tower/output/zensim-archive-2026-05-20/canonical-2026-05-21/train"),
        "safesyn inherits canonical_root Tower mirror"
    );
}

#[test]
fn v39_phone_manifest_maps_two_groups_with_dial_dir() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../zensim/weights/manifests/zensim_b_phone_oled_2026-05-26.toml");
    assert!(
        path.exists(),
        "phone manifest missing at {}",
        path.display()
    );
    let cfg = parse_manifest(&path).expect("parse phone manifest");

    assert_eq!(cfg.seed, Some(17));
    assert_eq!(cfg.target_scale, Some(100.0), "phone bake uses scale 100");
    let names: Vec<&str> = cfg.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, vec!["kadid_ph", "tid_ph"], "phone group names");
    // {dial_dir} resolves against [inputs.dial_dir].local.
    let dial = "/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25";
    assert_eq!(
        cfg.groups[0].path,
        PathBuf::from(format!("{dial}/kadid_phone_cvvdptgt_train.parquet")),
        "kadid_ph resolves {{dial_dir}}"
    );
    // Phone manifest's per-input blocks carry NO sha256 → none collected
    // for verification (only auto_transforms has one).
    let keys: Vec<&str> = cfg.inputs.iter().map(|i| i.key.as_str()).collect();
    assert_eq!(keys, vec!["auto_transforms"], "only sha-bearing input");
}

// --- sha verification with a synthetic manifest + controlled temp file ---

fn tmp_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("zensim_manifest_it_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn synthetic_manifest_sha_verify_passes_then_fails_on_drift() {
    let dir = tmp_dir();
    // A real input file with known bytes → known sha.
    let data_path = dir.join("data.parquet");
    std::fs::write(&data_path, b"reproducible-bytes").unwrap();
    let real_sha = sha256_file(&data_path).unwrap();

    // Build a synthetic manifest referencing it via an absolute path.
    let manifest_toml = format!(
        r#"
[bake]
file = "out.bin"

[training]
seed = 7
hidden = 64
target_column = "human_score"
groups = [
    {{ name = "g0", path = "{}", train_w = 1.0, val_w = 0.5 }},
]

[inputs.g0]
path = "{}"
sha256 = "{real_sha}"
rows = 1
"#,
        data_path.display(),
        data_path.display()
    );
    let manifest_path = dir.join("synthetic.toml");
    std::fs::write(&manifest_path, &manifest_toml).unwrap();

    let cfg = parse_manifest(&manifest_path).expect("parse synthetic manifest");
    assert_eq!(cfg.seed, Some(7));
    assert_eq!(cfg.hidden, Some(64));
    assert_eq!(cfg.inputs.len(), 1);
    assert_eq!(cfg.out, Some(dir.join("out.bin")), "out from [bake].file");

    // Matching sha → passes, no warnings.
    let warns = verify_inputs(&cfg.inputs, false).expect("sha match should pass");
    assert!(warns.is_empty(), "no warnings on a clean match");

    // Mutate the file → sha drifts → hard error (NOT relaxed).
    std::fs::write(&data_path, b"DRIFTED-bytes").unwrap();
    let err = verify_inputs(&cfg.inputs, false).expect_err("drift must fail loud");
    assert!(
        matches!(err, ManifestError::ShaMismatch { .. }),
        "drift is a ShaMismatch, got {err:?}"
    );

    // The escape hatch downgrades drift to a warning (still trains).
    let warns = verify_inputs(&cfg.inputs, true).expect("allow-drift proceeds");
    assert_eq!(warns.len(), 1, "one drift warning under the escape hatch");
}

#[test]
fn synthetic_manifest_missing_input_points_at_mirror() {
    let inputs = vec![ManifestInput {
        key: "safesyn".into(),
        path: PathBuf::from("/definitely/not/here/safesyn.parquet"),
        sha256: "0".repeat(64),
        rows: None,
        r2: Some("s3://zentrain/canonical-2026-05-21/train/safesyn.parquet".into()),
        tower: None,
    }];
    // Missing files error EVEN with the drift escape hatch on.
    let err = verify_inputs(&inputs, true).expect_err("missing input must error");
    match err {
        ManifestError::MissingInput { mirror, .. } => assert_eq!(
            mirror.as_deref(),
            Some("s3://zentrain/canonical-2026-05-21/train/safesyn.parquet")
        ),
        other => panic!("expected MissingInput, got {other:?}"),
    }
}

#[test]
fn parse_manifest_str_round_trips_explicit_overridable_fields() {
    // Guards the parse_manifest_str entry point used by unit tests +
    // the binary, with a minimal manifest carrying just a few fields.
    let toml = r#"
[training]
seed = 42
epochs = 5
"#;
    let cfg = parse_manifest_str(toml, Path::new("/x/manifest.toml")).unwrap();
    assert_eq!(cfg.seed, Some(42));
    assert_eq!(cfg.epochs, Some(5));
    // Unset fields stay None so the binary keeps clap defaults / explicit flags.
    assert_eq!(cfg.hidden, None);
    assert!(cfg.groups.is_empty());
    let _ = train_manifest::ManifestConfig::default();
}
