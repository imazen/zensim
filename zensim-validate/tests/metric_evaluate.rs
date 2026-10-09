//! The real CLI on synthetic data only; no canonical corpus defaults.
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let base = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap()).join("tmp"));
        let root = base.join(format!(
            "metric-evaluate-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }
    fn path(&self, s: &str) -> PathBuf {
        self.root.join(s)
    }
    fn write(&self, s: &str, data: &str) {
        fs::write(self.path(s), data).unwrap();
    }
    fn manifest(&self) -> Value {
        self.write("data.csv","key,human,score,ref,codec,q\na,1,1,r1,jpeg,1\nb,2,2,r1,jpeg,2\nc,3,3,r1,jpeg,3\nd,4,4,r2,jpeg,1\ne,5,5,r2,jpeg,2\nf,6,6,r2,jpeg,3\n");
        json!({"schema":"metric-evaluation-v1","datasets":[{"id":"test","path":"data.csv","sha256":hash(&self.path("data.csv")),"role":"synthetic","target_kind":"synthetic","quality_direction":"higher","columns":{"id":"key","target":"human","score":"score","reference_id":"ref","codec":"codec","quality":"q"}}],"metrics":[{"id":"good","direction":"higher","implementation":"synthetic-v1","input_contract":"synthetic score table","ladder_epsilon":0.5,"source":{"kind":"column","column":"score"}}]})
    }
    fn run(&self, v: &Value, out: &str) -> Output {
        self.write("suite.json", &v.to_string());
        Command::new(env!("CARGO_BIN_EXE_panel"))
            .args(["evaluate", "--manifest"])
            .arg(self.path("suite.json"))
            .arg("--output")
            .arg(self.path(out))
            .output()
            .unwrap()
    }
    fn report(&self, out: &str) -> Value {
        serde_json::from_slice(&fs::read(self.path(out).join("report.json")).unwrap()).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn hash(p: &Path) -> String {
    zensim_validate::train_manifest::sha256_file(p).unwrap()
}

#[test]
fn signed_rank_matches_canonical_and_reports_missing_capabilities() {
    let f = Fixture::new();
    let mut v = f.manifest();
    v["metrics"].as_array_mut().unwrap().push(json!({"id":"backwards","direction":"lower","implementation":"synthetic-v1","input_contract":"synthetic score table","source":{"kind":"column","column":"score"}}));
    let o = f.run(&v, "out");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r = f.report("out");
    let good = &r["runs"][0]["criteria"];
    let bad = &r["runs"][1]["criteria"];
    assert_eq!(good["rank"]["measurements"]["srocc_signed"], 1.0);
    assert_eq!(bad["rank"]["measurements"]["srocc_signed"], -1.0);
    assert_eq!(good["within_reference"]["measurements"]["mean"], 1.0);
    assert_eq!(bad["within_reference"]["measurements"]["mean"], -1.0);
    assert_eq!(good["dial"]["measurements"]["pairs"], 4);
    assert_eq!(good["runtime"]["state"], "not_measured");
    assert_eq!(good["spatial"]["state"], "not_measured");
    let p = [1., 2., 3., 4., 5., 6.];
    assert_eq!(
        good["rank"]["measurements"]["z_rmse"],
        json!(zensim_validate::panel::compute_panel(&p, &p).z_rmse)
    );
    assert!(f.path("out/report.html").is_file());
}

#[test]
fn requirements_fail_on_direction_and_missing_evidence() {
    let f = Fixture::new();
    let mut v = f.manifest();
    v["requirements"] = json!([{"metric":"good","dataset":"test","criterion":"targeting","pointer":"/p95","max":1}]);
    let o = f.run(&v, "missing");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(f.report("missing")["status"], "incomplete");
    v["metrics"][0]["direction"] = json!("lower");
    v["requirements"] = json!([{"metric":"good","dataset":"test","criterion":"rank","pointer":"/srocc_signed","min":0.9}]);
    assert_eq!(f.run(&v, "bad").status.code(), Some(1));
    assert_eq!(f.report("bad")["checks"][0]["state"], "fail");
}

#[test]
fn bad_scores_are_retained_and_never_silently_dropped() {
    let f = Fixture::new();
    let mut v = f.manifest();
    let path = f.path("data.csv");
    let text = fs::read_to_string(&path)
        .unwrap()
        .replace("b,2,2,", "b,2,NaN,");
    fs::write(&path, text).unwrap();
    v["datasets"][0]["sha256"] = json!(hash(&path));
    assert_eq!(f.run(&v, "out").status.code(), Some(1));
    let r = f.report("out");
    assert_eq!(r["runs"][0]["rows"], 6);
    assert_eq!(r["runs"][0]["score_failures"], 1);
    assert_eq!(r["runs"][0]["criteria"]["rank"]["state"], "failed");
    assert!(
        fs::read_to_string(f.path("out/test/good/scores.tsv"))
            .unwrap()
            .contains("nonfinite")
    );
}

#[test]
fn verifies_input_hash_and_refuses_overwrite() {
    let f = Fixture::new();
    let v = f.manifest();
    assert!(f.run(&v, "out").status.success());
    let before = hash(&f.path("out/report.json"));
    assert_eq!(f.run(&v, "out").status.code(), Some(2));
    assert_eq!(hash(&f.path("out/report.json")), before);
    f.write("data.csv", "not the pinned dataset\n");
    let o = f.run(&v, "changed");
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("SHA256 mismatch"));
}

#[test]
fn quoted_csv_and_parquet_use_exact_column_mapping() {
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    let f = Fixture::new();
    let mut v = f.manifest();
    f.write(
        "quoted.csv",
        "key,human,score\n\"one,quoted\",1,1\nb,2,2\nc,3,3\n",
    );
    v["datasets"][0]["path"] = json!("quoted.csv");
    v["datasets"][0]["sha256"] = json!(hash(&f.path("quoted.csv")));
    v["datasets"][0]["columns"] = json!({"id":"key","target":"human","score":"score"});
    assert!(f.run(&v, "csv").status.success());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("human", DataType::Float64, false),
        Field::new("score", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["one,quoted", "b", "c"])),
            Arc::new(Float64Array::from(vec![1., 2., 3.])),
            Arc::new(Float64Array::from(vec![1., 2., 3.])),
        ],
    )
    .unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(
        fs::File::create(f.path("data.parquet")).unwrap(),
        schema,
        None,
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    v["datasets"][0]["path"] = json!("data.parquet");
    v["datasets"][0]["sha256"] = json!(hash(&f.path("data.parquet")));
    let o = f.run(&v, "parquet");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        f.report("csv")["runs"][0]["criteria"],
        f.report("parquet")["runs"][0]["criteria"]
    );
}

#[test]
fn stale_instrument_is_failed_and_valid_bound_measurements_are_checked() {
    let f = Fixture::new();
    let mut v = f.manifest();
    assert!(f.run(&v, "initial").status.success());
    let binding = f.report("initial")["runs"][0]["binding"].clone();
    let mut artifact = json!({"schema":"metric-instrument-v1","criterion":"targeting","metric_sha256":binding["metric_sha256"],"dataset_sha256":binding["dataset_sha256"],"n":6,"measurements":{"p95":2.0}});
    f.write("instrument.json", &artifact.to_string());
    v["instruments"] = json!([{"metric":"good","dataset":"test","criterion":"targeting","kind":"artifact","path":"instrument.json","sha256":hash(&f.path("instrument.json"))}]);
    v["requirements"] = json!([{"metric":"good","dataset":"test","criterion":"targeting","pointer":"/p95","max":3}]);
    let o = f.run(&v, "valid");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(f.report("valid")["checks"][0]["state"], "pass");
    artifact["metric_sha256"] = json!("wrong-model");
    f.write("instrument.json", &artifact.to_string());
    v["instruments"][0]["sha256"] = json!(hash(&f.path("instrument.json")));
    assert_eq!(f.run(&v, "stale").status.code(), Some(1));
    assert_eq!(
        f.report("stale")["runs"][0]["criteria"]["targeting"]["state"],
        "failed"
    );
}

#[test]
fn no_implicit_sampling_of_large_or_constant_panels() {
    let f = Fixture::new();
    let mut v = f.manifest();
    v["max_panel_rows"] = json!(3);
    assert!(f.run(&v, "limited").status.success());
    assert_eq!(
        f.report("limited")["runs"][0]["criteria"]["rank"]["state"],
        "not_measured"
    );
    assert_eq!(f.report("limited")["runs"][0]["rows"], 6);
}

#[cfg(unix)]
fn executable(f: &Fixture, code: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = f.path("adapter.sh");
    fs::write(&path, code).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

#[cfg(unix)]
#[test]
fn external_adapter_uses_literal_argv_and_retains_output() {
    let f = Fixture::new();
    let mut v = f.manifest();
    let path = executable(
        &f,
        "#!/bin/sh\n[ \"$1\" = '$(touch SHOULD_NOT_EXIST)' ] || exit 9\nprintf '{\"value\":2.5}\\n'\n",
    );
    v["metrics"][0]["source"] = json!({"kind":"command","program":path,"args":["$(touch SHOULD_NOT_EXIST)"],"json_pointer":"/value","timeout_seconds":2});
    let o = f.run(&v, "out");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!f.path("SHOULD_NOT_EXIST").exists());
    assert_eq!(f.report("out")["runs"][0]["score_failures"], 0);
    assert!(f.path("out/test/good/row-0.stdout").exists());
}

#[cfg(unix)]
#[test]
fn zenmetrics_adapter_selects_declared_score_column_and_rejects_ambiguity() {
    let f = Fixture::new();
    let mut v = f.manifest();
    let path = executable(
        &f,
        "#!/bin/sh\n[ \"$1\" = score ] || exit 9\nprintf '{\"scores\":{\"max\":2,\"pnorm\":3}}\\n'\n",
    );
    f.write("image.dat", "synthetic");
    let mut data = fs::read_to_string(f.path("data.csv")).unwrap();
    data = data
        .lines()
        .enumerate()
        .map(|(i, l)| {
            if i == 0 {
                format!("{l},reference,distorted\n")
            } else {
                format!("{l},image.dat,image.dat\n")
            }
        })
        .collect();
    f.write("data.csv", &data);
    v["datasets"][0]["sha256"] = json!(hash(&f.path("data.csv")));
    v["datasets"][0]["columns"]["reference"] = json!("reference");
    v["datasets"][0]["columns"]["distorted"] = json!("distorted");
    v["metrics"][0]["source"] =
        json!({"kind":"zenmetrics","program":path,"metric":"butteraugli","score_column":"max"});
    let o = f.run(&v, "selected");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    v["metrics"][0]["source"]
        .as_object_mut()
        .unwrap()
        .remove("score_column");
    assert_eq!(f.run(&v, "ambiguous").status.code(), Some(1));
    assert_eq!(f.report("ambiguous")["runs"][0]["score_failures"], 6);
}

#[cfg(unix)]
#[test]
fn process_failure_is_not_a_numeric_score() {
    let f = Fixture::new();
    let mut v = f.manifest();
    let path = executable(&f, "#!/bin/sh\nprintf '42\\n'\nexit 7\n");
    v["metrics"][0]["source"] = json!({"kind":"command","program":path});
    assert_eq!(f.run(&v, "out").status.code(), Some(1));
    assert_eq!(f.report("out")["runs"][0]["score_failures"], 6);
}

#[test]
fn keyed_scores_join_by_id_and_reject_missing_or_extra_rows() {
    let f = Fixture::new();
    let mut v = f.manifest();
    f.write("external.csv", "id,value\nf,6\ne,5\nd,4\nc,3\nb,2\na,1\n");
    v["metrics"][0]["source"] = json!({"kind":"table","path":"external.csv","sha256":hash(&f.path("external.csv")),"id_column":"id","score_column":"value"});
    let o = f.run(&v, "joined");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        f.report("joined")["runs"][0]["criteria"]["rank"]["measurements"]["srocc_signed"],
        1.0
    );
    f.write("external.csv", "id,value\na,1\nb,2\n");
    v["metrics"][0]["source"]["sha256"] = json!(hash(&f.path("external.csv")));
    assert_eq!(f.run(&v, "missing").status.code(), Some(1));
    assert_eq!(f.report("missing")["runs"][0]["score_failures"], 4);
    f.write("external.csv", "id,value\nunknown,1\n");
    v["metrics"][0]["source"]["sha256"] = json!(hash(&f.path("external.csv")));
    assert_eq!(f.run(&v, "extra").status.code(), Some(2));
}

#[cfg(unix)]
#[test]
fn batch_adapter_joins_shuffled_ids_and_rejects_duplicates() {
    let f = Fixture::new();
    let mut v = f.manifest();
    let path = executable(
        &f,
        "#!/bin/sh\nprintf '%s' '{\"scores\":[{\"id\":\"f\",\"score\":6},{\"id\":\"e\",\"score\":5},{\"id\":\"d\",\"score\":4},{\"id\":\"c\",\"score\":3},{\"id\":\"b\",\"score\":2},{\"id\":\"a\",\"score\":1}]}'\n",
    );
    v["metrics"][0]["source"] = json!({"kind":"batch_command","command":{"program":path,"args":["{dataset}"],"timeout_seconds":2}});
    let o = f.run(&v, "batch");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        f.report("batch")["runs"][0]["criteria"]["rank"]["measurements"]["srocc_signed"],
        1.0
    );
    assert!(f.path("batch/test/good/batch.stdout").is_file());
    executable(
        &f,
        "#!/bin/sh\nprintf '%s' '{\"scores\":[{\"id\":\"a\",\"score\":1},{\"id\":\"a\",\"score\":2}]}'\n",
    );
    let o = f.run(&v, "duplicate");
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("duplicate score row id"));
}

#[test]
fn preference_direction_and_weights_match_the_owner() {
    let f = Fixture::new();
    let mut v = f.manifest();
    f.write(
        "data.csv",
        "id,score,right,choice,group,weight\na,1,b,left,g1,3\nb,2,,,,\nc,1,d,right,g1,1\nd,2,,,,\n",
    );
    v["datasets"][0]["sha256"] = json!(hash(&f.path("data.csv")));
    v["datasets"][0]["columns"] = json!({"id":"id","score":"score","right_id":"right","choice":"choice","preference_group":"group","weight":"weight"});
    let o = f.run(&v, "pairwise");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r = f.report("pairwise");
    let m = &r["runs"][0]["criteria"]["pairwise"]["measurements"];
    assert_eq!(m["accuracy"], 0.75);
    assert_eq!(m["ceiling"], 0.75);
    assert_eq!(m["normalized_accuracy"], 1.0);
    assert_eq!(r["runs"][0]["criteria"]["rank"]["state"], "not_measured");
}

#[test]
fn paired_comparison_keeps_declared_direction() {
    let f = Fixture::new();
    let mut v = f.manifest();
    let mut other = v["metrics"][0].clone();
    other["id"] = json!("other");
    v["metrics"].as_array_mut().unwrap().push(other);
    v["comparisons"] =
        json!([{"dataset":"test","a":"good","b":"other","bootstrap_resamples":8,"seed":42}]);
    let o = f.run(&v, "same");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        f.report("same")["comparisons"][0]["report"]["state"],
        "measured"
    );
    v["metrics"][1]["direction"] = json!("lower");
    assert!(f.run(&v, "inverted").status.success());
    assert_eq!(
        f.report("inverted")["comparisons"][0]["report"]["state"],
        "not_measured"
    );
}
