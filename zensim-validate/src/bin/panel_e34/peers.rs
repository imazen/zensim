//! Exploratory matched reference comparison using E34's numerical owner.
//! Only the explicitly authorised, previously captured E33 panel is accepted.
use super::*;

const E33_CAPTURE: &str = "45e5f49dd329eee4bfb7751fe43b5dd70e548094a89b0af5dfdfd31828dd2eda";
const E34_CAPTURE: &str = "a1511bab58833e202fe89d3c90ac663cd1e9dbbb88cc71057b5a7d99385caa9c";
const MODELS: [&str; 3] = ["diff410", "diff_frag820", "w_c"];
const COUNTS: [usize; 4] = [7869, 3000, 756, 2192];
const PIXEL_KEYS: [&str; 4] = [
    "e3c360c61b90f0a8cba2440631c1fa6df466758cdb22b45d24eab010f241a739",
    "a33dc5c9c1c52b765b9c3e910311b3834f948249a7e9973dd45217d71836b054",
    "60beee802856e4f3260a433a15d4aa4762d4757d65d75bed13c296452cb608d4",
    "d9182fa790dae08a3f9d3b336ee854d012878afcab3ed624460b5ad6620def77",
];
const REFERENCES: [usize; 4] = [65, 25, 7, 25];
const FIELD: &str = "cvvdp_cpu_imazen_v0_1_0_standard_fhd";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    schema: String,
    paired_input: PathBuf,
    paired_sha256: String,
    peer_scores: BTreeMap<String, Peer>,
    current_input: PathBuf,
    current_sha256: String,
    progress_markdown: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Peer {
    tsv: PathBuf,
    sha256: String,
    pixel_keys: PathBuf,
    pixel_keys_sha256: String,
}

fn predictions(peer: &Peer, count: usize) -> Result<Vec<f64>> {
    let payload = fs::read(&peer.tsv)?;
    ensure!(digest(&payload) == peer.sha256, "peer score bytes changed");
    let keys = fs::read(&peer.pixel_keys)?;
    ensure!(
        digest(&keys) == peer.pixel_keys_sha256,
        "pixel keys changed"
    );
    let keys: Vec<Value> = serde_json::from_slice(&keys)?;
    ensure!(keys.len() == count, "pixel-key coverage");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(payload.as_slice());
    let header = reader.headers()?.clone();
    let column = |name: &str| -> Result<usize> {
        header
            .iter()
            .position(|v| v == name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))
    };
    let (index, score, reference, distorted) = (
        column("row_index")?,
        column(FIELD)?,
        column("ref_path")?,
        column("dist_path")?,
    );
    let mut result = vec![f64::NAN; count];
    let mut seen = BTreeSet::new();
    for record in reader.records() {
        let r = record?;
        let i: usize = r[index].parse()?;
        ensure!(
            i < count && seen.insert(i),
            "duplicate/foreign physical row"
        );
        ensure!(keys[i]["row_index"] == i, "metadata physical order");
        ensure!(
            keys[i]["ref_path"].as_str() == Some(&r[reference])
                && keys[i]["dist_path"].as_str() == Some(&r[distorted]),
            "peer pixel pair differs"
        );
        result[i] = r[score].parse()?;
        ensure!(result[i].is_finite(), "nonfinite peer score");
    }
    ensure!(seen.len() == count, "incomplete peer panel");
    Ok(result)
}

fn progress(path: &Path, message: &str) -> Result<()> {
    let mut file = fs::OpenOptions::new().append(true).open(path)?;
    writeln!(file, "\n- {}: {}", chrono::Utc::now().to_rfc3339(), message)?;
    eprintln!("{message}");
    Ok(())
}

pub(super) fn run(bytes: &[u8]) -> Result<Value> {
    let job: Job = serde_json::from_slice(bytes)?;
    ensure!(
        job.schema == "addrfix-matched-cvvdp-v1" && job.paired_sha256 == E33_CAPTURE,
        "only the admitted E33 exploratory capture is supported"
    );
    ensure!(
        job.peer_scores
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == DATASETS.into_iter().collect(),
        "four exact source populations required"
    );
    let paired = fs::read(&job.paired_input)?;
    ensure!(digest(&paired) == E33_CAPTURE, "paired capture changed");
    let mut input: Input = serde_json::from_slice(&paired)?;
    input.validate(true)?;
    ensure!(
        job.current_sha256 == E34_CAPTURE,
        "only admitted E34 capture supported"
    );
    let current = fs::read(&job.current_input)?;
    ensure!(digest(&current) == E34_CAPTURE, "current capture changed");
    let current: Input = serde_json::from_slice(&current)?;
    current.validate(false)?;
    for (old, new) in input.datasets.iter_mut().zip(&current.datasets) {
        ensure!(
            old.name == new.name
                && old.target == new.target
                && old.reference == new.reference
                && old.orientation == new.orientation,
            "paired capture population differs"
        );
        for (label, source) in MODELS.into_iter().zip(["k-a", "k-c", "w-c"]) {
            old.predictions
                .insert(label.into(), new.predictions[source].clone());
        }
    }
    let mut point_scores = BTreeMap::new();
    for (di, dataset) in input.datasets.iter_mut().enumerate() {
        ensure!(
            dataset.target.len() == COUNTS[di]
                && dataset.reference.iter().collect::<BTreeSet<_>>().len() == REFERENCES[di],
            "source population differs"
        );
        ensure!(
            job.peer_scores[&dataset.name].pixel_keys_sha256 == PIXEL_KEYS[di],
            "only frozen admitted pixel order supported"
        );
        let pred = predictions(&job.peer_scores[&dataset.name], COUNTS[di])?;
        dataset
            .predictions
            .insert("cvvdp_fhd".into(), vec![pred; 10]);
        let mut models = BTreeMap::new();
        for model in ["a", "c", "diff410", "diff_frag820", "w_c", "cvvdp_fhd"] {
            let scores: Vec<_> = dataset.predictions[model]
                .iter()
                .map(|p| dataset.orientation * zenstats::spearman(p, &dataset.target))
                .collect();
            ensure!(
                scores.iter().all(|v| v.is_finite()),
                "undefined matched SROCC"
            );
            models.insert(
                model,
                json!({"mean_signed_srocc":scores.iter().sum::<f64>()/10.,
                "seeds":scores}),
            );
        }
        progress(
            &job.progress_markdown,
            &format!(
                "Matched owner point SROCC {}: {}",
                dataset.name,
                serde_json::to_string(&models)?
            ),
        )?;
        point_scores.insert(
            dataset.name.clone(),
            json!({"rows":COUNTS[di],
            "references":REFERENCES[di],"models":models}),
        );
    }
    let mut overall = BTreeMap::new();
    for model in ["a", "c", "diff410", "diff_frag820", "w_c", "cvvdp_fhd"] {
        overall.insert(
            model,
            DATASETS
                .iter()
                .map(|d| {
                    point_scores[*d]["models"][model]["mean_signed_srocc"]
                        .as_f64()
                        .unwrap()
                })
                .sum::<f64>()
                / 4.,
        );
    }
    let mut comparisons = Vec::new();
    for model in MODELS {
        let test = Test {
            index: 1,
            arm: model.into(),
            control: "cvvdp_fhd".into(),
            eligibility: false,
        };
        for subset in [vec![0, 1, 2, 3], vec![0], vec![1], vec![2], vec![3]] {
            // Reuse the exact paired source x seed statistic. Offset5 binds
            // the predeclared seed20261016; the reference has identical copies
            // at all ten model-seed levels, not ten independent reference fits.
            let scope = if subset.len() == 4 {
                "overall"
            } else {
                DATASETS[subset[0]]
            };
            eprintln!("ADDRFIX paired interval started: {model}/{scope}");
            let iv = input.interval(&test, &subset, 5, false, false, 0.)?;
            ensure!(
                iv.n_finite > 0
                    && iv.estimate.is_finite()
                    && iv.lo.is_finite()
                    && iv.hi.is_finite(),
                "undefined matched peer interval"
            );
            let groups = subset
                .iter()
                .map(|&di| REFERENCES[di])
                .sum::<usize>()
                .min(10);
            progress(
                &job.progress_markdown,
                &format!(
                    "Matched exploratory {model}/{scope}: delta SROCC={} 95% CI [{},{}], finite {}/10000; no qualification.",
                    iv.estimate, iv.lo, iv.hi, iv.n_finite
                ),
            )?;
            comparisons.push(json!({"candidate":model,
                "scope":if subset.len()==4 {"overall"} else {DATASETS[subset[0]]},
                "delta":interval_json(&iv,groups),
                "attempted":10000,"dropped":10000-iv.n_finite,"p_zero":iv.p_zero}));
        }
    }
    Ok(
        json!({"schema":"addrfix-matched-cvvdp-result-v1","exploratory":true,
        "qualification_allowed":false,"paired_capture_sha256":E33_CAPTURE,"current_capture_sha256":E34_CAPTURE,
        "point_scores":point_scores,"four_source_mean":overall,"comparisons":comparisons,
        "bootstrap":{"unit":"source-reference x shared training seed","resamples":10000,
            "seed":20261016,"level":0.95,"reference_prediction":"one fixed score vector repeated across ten candidate-seed slots"},
        "limits":"Previously exposed exploratory sources. Small reference counts; intervals are descriptive. No protected holdout or confirmatory qualification."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let home = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .expect("test needs a user scratch root");
            let root = PathBuf::from(home).join("tmp").join(format!(
                "addrfix-peer-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&root).unwrap();
            Self(root)
        }
        fn peer(&self, body: &str) -> Peer {
            let score = format!("row_index\tref_path\tdist_path\t{FIELD}\n{body}");
            let keys = br#"[{"row_index":0,"ref_path":"r0","dist_path":"d0"},{"row_index":1,"ref_path":"r1","dist_path":"d1"}]"#;
            fs::write(self.0.join("scores.tsv"), &score).unwrap();
            fs::write(self.0.join("keys.json"), keys).unwrap();
            Peer {
                tsv: self.0.join("scores.tsv"),
                sha256: digest(score.as_bytes()),
                pixel_keys: self.0.join("keys.json"),
                pixel_keys_sha256: digest(keys),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn physical_order_survives_grouped_batch_output() {
        let f = Fixture::new();
        let peer = f.peer("1\tr1\td1\t7.5\n0\tr0\td0\t9.0\n");
        assert_eq!(predictions(&peer, 2).unwrap(), vec![9.0, 7.5]);
    }

    #[test]
    fn wrong_pairs_duplicates_missing_and_nonfinite_refuse() {
        let f = Fixture::new();
        for body in [
            "0\tr0\td0\t9\n0\tr0\td0\t8\n",
            "0\tr0\td0\t9\n1\tr0\td1\t8\n",
            "0\tr0\td0\t9\n",
            "0\tr0\td0\t9\n1\tr1\td1\tNaN\n",
        ] {
            assert!(predictions(&f.peer(body), 2).is_err());
        }
    }
}
