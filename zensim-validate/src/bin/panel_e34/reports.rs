//! Registered report-only source-clustered intervals; no adoption decisions.
use super::*;

pub(super) fn packs(bytes: &[u8]) -> Result<Value> {
    pack_inventory(bytes, 21, "e34-pack-checks-v1")
}

pub(super) fn pack_inventory(bytes: &[u8], count: usize, schema: &str) -> Result<Value> {
    let models: BTreeMap<String, Value> = serde_json::from_slice(bytes)?;
    ensure!(
        models.len() == count,
        "{count} frozen full-data models required"
    );
    let mut output = BTreeMap::new();
    for (name, binding) in models {
        let model_bytes = fs::read(binding["path"].as_str().unwrap())?;
        ensure!(
            digest(&model_bytes) == binding["sha256"].as_str().unwrap(),
            "model changed: {name}"
        );
        let model =
            zenpredict::Model::from_bytes(&model_bytes).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let metadata = model.metadata();
        let payload = metadata
            .get("zentrain.output_calibration_spline")
            .ok_or_else(|| anyhow::anyhow!("missing output spline: {name}"))?;
        let check = zensim_validate::dial_spline::check_identity_pinned(payload.value, 100.);
        output.insert(
            name,
            match check {
                Ok(c) => json!({"model_sha256":binding["sha256"],"K1_K2_K3":true,
                "x_floor":c.x_floor,"floor_score":c.floor,"x0":c.x0,
                "identity_slope":c.identity_slope,"K2_points":c.k2_points}),
                Err(e) => json!({"model_sha256":binding["sha256"],"K1_K2_K3":false,"reason":e}),
            },
        );
    }
    Ok(json!({"schema":schema,"models":output}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reports {
    schema: String,
    paired_input: PathBuf,
    paired_sha256: String,
    /// Explicit E33 top-20% thresholds, computed before filtering any model.
    quality_thresholds: Vec<f64>,
    /// Fixed order: G-STEER rates, dev rates, then W/K log Rel.L2 ratios.
    series: Vec<Series>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Series {
    name: String,
    reference: Vec<u32>,
    values: Vec<f64>,
    /// Log ratios are averaged on the log scale and exponentiated afterwards.
    exp: bool,
}

fn cfg(offset: u64) -> BootstrapConfig {
    BootstrapConfig {
        resamples: 10_000,
        seed: 20261011 + offset,
        level: 0.95,
    }
}

fn class_intervals(input: &Input, t: &Test, offset: &mut u64) -> Result<Value> {
    let result = pair_classes_at(input, t, |_| {
        let current = *offset;
        *offset += 1;
        current
    })?;
    ensure!(
        result
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["interval"]["estimate"].as_f64().is_some()
                && r["interval"]["n_finite"].as_u64().is_some_and(|n| n > 0)),
        "undefined registered high-quality pair class for {}",
        t.arm
    );
    Ok(result)
}

pub(super) fn run(bytes: &[u8]) -> Result<Value> {
    let job: Reports = serde_json::from_slice(bytes)?;
    ensure!(job.schema == "e34-report-input-v1", "report schema");
    let paired = fs::read(&job.paired_input)?;
    ensure!(
        digest(&paired) == job.paired_sha256,
        "paired report input changed"
    );
    let mut input: Input = serde_json::from_slice(&paired)?;
    input.validate(false)?;
    ensure!(
        job.quality_thresholds.len() == 4,
        "four E33 quality thresholds required"
    );
    let models: Vec<_> = ["w-a", "k-a", "l-a", "w-c", "k-c", "base-a", "base-c"]
        .into_iter()
        .flat_map(|arm| (0..3).map(move |seed| format!("{arm}-s{seed}")))
        .collect();
    let expected: Vec<_> = ["gsteer-pass-rate", "dev-pass-rate"]
        .into_iter()
        .flat_map(|kind| models.iter().map(move |model| format!("{kind}/{model}")))
        .chain(
            ["w-a", "k-a", "w-c", "k-c"]
                .into_iter()
                .flat_map(|arm| (0..3).map(move |seed| format!("rel-l2-ratio/{arm}-s{seed}"))),
        )
        .collect();
    ensure!(
        job.series.iter().map(|s| &s.name).eq(expected.iter()),
        "registered report series order"
    );
    let mut offset = 2000;
    let mut series = Vec::new();
    for s in job.series {
        ensure!(
            s.values.len() == s.reference.len()
                && !s.values.is_empty()
                && s.values.iter().all(|v| v.is_finite()),
            "finite aligned report series {}",
            s.name
        );
        let groups = s.reference.iter().collect::<BTreeSet<_>>().len();
        ensure!(groups >= 3, "three source levels required for {}", s.name);
        let config = cfg(offset);
        offset += 1;
        let iv = cluster_bootstrap(
            s.values.len(),
            &Design::OneWay(&s.reference),
            &config,
            |draw| {
                vec![zenstats::resample::weighted_mean(
                    &s.values,
                    draw.row_weights(),
                )]
            },
        )?
        .remove(0);
        ensure!(
            iv.estimate.is_finite() && iv.n_finite > 0,
            "undefined report series {}",
            s.name
        );
        let mut measured = interval_json(&iv, groups);
        if s.exp {
            let student = iv
                .student(groups, 0.95)
                .map(|(lo, hi)| (lo.exp(), hi.exp()));
            measured = json!({"estimate":iv.estimate.exp(),"lo":iv.lo.exp(),"hi":iv.hi.exp(),
                "student_interval":student,"log_interval":measured,"n_finite":iv.n_finite});
        }
        series.push(
            json!({"name":s.name,"references":groups,"rows":s.values.len(),
            "seed":config.seed,"exp":s.exp,"interval":measured}),
        );
    }
    let mut population = Vec::new();
    for (di, d) in input.datasets.iter_mut().enumerate() {
        let mask: Vec<_> = d
            .target
            .iter()
            .enumerate()
            .filter_map(|(i, &y)| (y >= job.quality_thresholds[di]).then_some(i))
            .collect();
        let pick_f = |v: &Vec<f64>| mask.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let pick_u = |v: &Vec<u32>| mask.iter().map(|&i| v[i]).collect::<Vec<_>>();
        d.target = pick_f(&d.target);
        d.reference = pick_u(&d.reference);
        d.codec = pick_u(&d.codec);
        for seeds in d.predictions.values_mut() {
            for p in seeds {
                *p = pick_f(p);
            }
        }
        population.push(
            json!({"dataset":d.name,"threshold":job.quality_thresholds[di],
            "rows":mask.len(),"references":d.reference.iter().collect::<BTreeSet<_>>().len()}),
        );
    }
    let mut high_quality = Vec::new();
    for t in input.tests.iter().filter(|t| t.eligibility) {
        let mut datasets = Vec::new();
        for di in 0..4 {
            let iv = input.interval(t, &[di], offset, false, false, 0.)?;
            let seed = 20261011 + offset;
            offset += 1;
            ensure!(
                iv.estimate.is_finite() && iv.n_finite > 0,
                "undefined registered high-quality SROCC {} {}",
                t.arm,
                input.datasets[di].name
            );
            let groups = input.datasets[di]
                .reference
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                .min(10);
            datasets.push(json!({"dataset":input.datasets[di].name,"seed":seed,
                "interval":interval_json(&iv,groups)}));
        }
        high_quality.push(json!({"arm":t.arm,"control":t.control,"srocc":datasets,
            "pair_classes":class_intervals(&input,t,&mut offset)?}));
    }
    Ok(
        json!({"schema":"e34-reports-v1","report_only":true,"resamples":10000,
        "level":0.95,"percentile":"Hyndman-Fan type 7","offset_start":2000,
        "offset_next":offset,"series":series,"high_quality_population":population,
        "high_quality":high_quality}),
    )
}
