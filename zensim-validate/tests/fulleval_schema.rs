//! Golden schema test for `bake_verdict --fulleval` (consolidation item 2).
//!
//! The fulleval JSON used to be ASSEMBLED: `bake_verdict --full-json` emitted
//! most fields and `scripts/run_full_eval.sh` jq-injected the five M3 slots,
//! so a file's key set depended on which wrapper produced it (and a file that
//! never went through the wrapper silently lacked `m3_n`/`m3a_*`). `--fulleval`
//! emits the schema-complete file directly; this test pins its key set against
//! the established `*.fulleval.json` schema (e.g. `C_co3a_s1301.fulleval.json`)
//! so the jq step in `run_full_eval.sh` only ever injects INTO existing keys.
//!
//! Also pins the ensemble model-identity fields (`model.kind` / `members` /
//! `member_names` / `anchor`) that fix the dashboard's Model-details
//! misattribution at the source (schema = `scripts/promote_ensemble_fulleval.py`).
//!
//! Runs the real binary (CARGO_BIN_EXE) on a COMMITTED bake over a synthetic
//! 372-col parquet fixture written into CARGO_TARGET_TMPDIR — no /mnt/v, no
//! network, CI-safe.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

/// The committed shipped-B bake (linear-372 + winsor transforms + dial
/// spline) — a real production artifact, 7.3 KB, in-repo.
fn fixture_bake() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin")
}

/// Write a small deterministic 372-feature corpus parquet under the cid22
/// slot name so `--corpora cid22` resolves it from a temp features root.
fn write_fixture_corpus(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    let n_rows = 12usize;
    let n_feat = 372usize;
    let mut fields: Vec<Field> = vec![
        Field::new("ref_basename", DataType::Utf8, false),
        Field::new("human_score", DataType::Float64, false),
    ];
    for i in 0..n_feat {
        fields.push(Field::new(format!("f{i}"), DataType::Float64, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let refs: Vec<String> = (0..n_rows).map(|r| format!("ref{}", r / 3)).collect();
    let humans: Vec<f64> = (0..n_rows).map(|r| 0.05 + 0.08 * r as f64).collect();
    // Deterministic LCG features in [0, 1) — varied so predictions are not tied.
    let mut state = 0x2545F491_u64;
    let mut lcg = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (u32::MAX as f64 + 1.0)
    };
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(refs)),
        Arc::new(Float64Array::from(humans)),
    ];
    for _ in 0..n_feat {
        let v: Vec<f64> = (0..n_rows).map(|_| lcg()).collect();
        cols.push(Arc::new(Float64Array::from(v)));
    }
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let f = std::fs::File::create(root.join("cid22_features_372col_2026-05-15.parquet")).unwrap();
    let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Run bake_verdict with the fixture corpus; `extra` appends mode flags.
fn run_verdict(out_json: &Path, root: &Path, extra: &[&str]) -> serde_json::Value {
    let bake = fixture_bake();
    assert!(bake.exists(), "committed fixture bake missing: {bake:?}");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bake_verdict"));
    cmd.arg("--bake")
        .arg(&bake)
        .args(["--corpora", "cid22"])
        .arg("--features-root")
        .arg(root)
        // Non-existent grids/per-pair source: those sections degrade to
        // null/skip notes, which is exactly the shape the schema must hold.
        .args(["--dial-grid", "/nonexistent/dial.parquet"])
        .args(["--corruption-grid", "/nonexistent/corr.parquet"])
        .args(["--perpair-metrics", "/nonexistent/perpair.parquet"])
        .args(["--name", "fulleval_schema_fixture"])
        .arg("--fulleval")
        .arg(out_json)
        .args(extra);
    let out = cmd.output().expect("run bake_verdict");
    assert!(
        out.status.success(),
        "bake_verdict failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let s = std::fs::read_to_string(out_json).expect("read emitted fulleval json");
    serde_json::from_str(&s).expect("emitted fulleval json parses")
}

fn tmpdir(sub: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(sub);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The established fulleval top-level schema: every key the dashboard and
/// `run_full_eval.sh` read from a `*.fulleval.json` (measured against
/// `C_co3a_s1301.fulleval.json`, the wrapper-assembled reference).
const FULLEVAL_KEYS: &[&str] = &[
    "bake",
    "bake_sha256",
    "composite",
    "corruption",
    "corruption_head",
    "dial",
    "gates",
    "m3_coherence",
    "m3_dropped_mass_pct",
    "m3_n",
    "m3a_coherence",
    "m3a_n",
    "model",
    "n_inputs",
    "name",
    "per_pair",
    "rank",
    "regime",
    "repro",
];

#[test]
fn fulleval_emits_schema_complete_json_with_null_m3_slots() {
    let dir = tmpdir("fulleval_single");
    let root = dir.join("features");
    write_fixture_corpus(&root);
    let out = dir.join("fixture.fulleval.json");
    let v = run_verdict(&out, &root, &[]);

    let obj = v.as_object().expect("fulleval root is an object");
    for key in FULLEVAL_KEYS {
        assert!(
            obj.contains_key(*key),
            "fulleval JSON is missing schema key {key:?} — run_full_eval.sh's jq \
             step and the gauntlet expect it. Keys present: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }
    // The five M3 slots are the wrapper-measured fields — emitted as explicit
    // nulls so the jq splice only ever assigns into existing keys.
    for key in [
        "m3_coherence",
        "m3_n",
        "m3_dropped_mass_pct",
        "m3a_coherence",
        "m3a_n",
    ] {
        assert!(
            v[key].is_null(),
            "{key} must be emitted as null (measured later by diffmap_block_coherence), got {:?}",
            v[key]
        );
    }
    // Content sanity: the rank block actually scored the fixture corpus.
    assert!(
        v["rank"]["cid22"]["n"].as_u64() == Some(12),
        "rank.cid22.n: {:?}",
        v["rank"]["cid22"]
    );
    assert_eq!(v["model"]["kind"], "single");
    assert_eq!(v["name"], "fulleval_schema_fixture");
}

#[test]
fn fulleval_marks_ensembles_in_the_model_block() {
    let dir = tmpdir("fulleval_ensemble");
    let root = dir.join("features");
    write_fixture_corpus(&root);
    let out = dir.join("ens.fulleval.json");
    let bake = fixture_bake();
    let pair = format!("{},{}", bake.display(), bake.display());
    let v = run_verdict(&out, &root, &["--ensemble", &pair]);

    // Schema matches scripts/promote_ensemble_fulleval.py + the gauntlet's
    // isEns/ensK readers — the fix for the Model-details misattribution,
    // now applied at the source.
    assert_eq!(v["model"]["kind"], "ensemble");
    assert_eq!(v["model"]["members"], 2);
    let names = v["model"]["member_names"]
        .as_array()
        .expect("member_names is a list");
    assert_eq!(names.len(), 2);
    assert_eq!(
        names[0], "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07",
        "member_names carries file stems (the campaign's bake-stem convention)"
    );
    assert_eq!(
        v["model"]["anchor"], "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin",
        "anchor carries the provenance-anchor basename"
    );
    // The m3 nulls hold in ensemble mode too.
    assert!(v["m3a_coherence"].is_null());
}
