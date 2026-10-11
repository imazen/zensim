//! E34 draft revision 3: source-clustered intervals for the comparative claims in the characterization (§1), computed
//! by zenmetrics-stats (`resample::cluster_bootstrap`, `Design::OneWay`, `weighted_mean`), percentile rule Hyndman-Fan
//! type 7, B = 10,000, seed 20261011, level 0.95.
//!
//! Input: `per_case.json` from `characterize.py` (stored, label-free G-STEER rows). Rows: the seven served models
//! (A, C seeds 0-2; production seed 0) x 135 cases. Cluster: the underlying source picture (the eight broad origins;
//! the three M3 fixtures, whose 256/384/576 renditions are one picture; the four KADID references).
//!
//! usage: e34-clustered <per_case.json> <out.json>
use serde_json::{Value, json};
use std::collections::BTreeMap;
use zenmetrics_stats::resample::{BootstrapConfig, Design, Draw, cluster_bootstrap, weighted_mean};

const SERVED: [&str; 7] = ["A-s0", "A-s1", "A-s2", "C-s0", "C-s1", "C-s2", "P-s0"];

struct Row {
    cluster: String,
    broad: bool,
    block: u64,
    level: Option<String>,
    fail: f64,
    floor: bool,
    log_rel_l2: f64,
}

fn picture(r: &str) -> String {
    let stem = r.split(".png").next().unwrap_or(r);
    stem.split('_')
        .next()
        .map(|s| {
            if stem.starts_with("o_") {
                stem.to_string()
            } else {
                s.to_string()
            }
        })
        .unwrap()
}

fn masked_mean(d: &Draw<'_>, values: &[f64], keep: &[bool]) -> f64 {
    let w: Vec<f64> = d
        .row_weights()
        .iter()
        .zip(keep)
        .map(|(&w, &k)| if k { w } else { 0.0 })
        .collect();
    weighted_mean(values, &w)
}

fn run(rows: &[Row], name: &str, stat: impl Fn(&Draw<'_>) -> Vec<f64> + Sync) -> Value {
    let mut ids: BTreeMap<&str, u32> = BTreeMap::new();
    let labels: Vec<u32> = rows
        .iter()
        .map(|r| {
            let n = ids.len() as u32;
            *ids.entry(r.cluster.as_str()).or_insert(n)
        })
        .collect();
    let cfg = BootstrapConfig {
        resamples: 10_000,
        seed: 20_261_011,
        level: 0.95,
    };
    let out =
        cluster_bootstrap(rows.len(), &Design::OneWay(&labels), &cfg, stat).expect("bootstrap");
    let i = out[0];
    json!({"statistic": name, "estimate": i.estimate, "lo": i.lo, "hi": i.hi, "se": i.se, "n_finite": i.n_finite,
           "p_zero": i.p_zero, "clusters": ids.len(), "rows": rows.len(), "resamples": cfg.resamples, "seed": cfg.seed,
           "level": cfg.level, "percentile_rule": "Hyndman-Fan type 7 (zenmetrics-stats quantile7)",
           "design": "OneWay(source picture)"})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // characterize.py writes Python's `NaN` for undefined drop-one M2 values (floor cases); JSON has no NaN.
    let text = std::fs::read_to_string(&args[1])
        .unwrap()
        .replace("NaN", "null");
    let recs: Vec<Value> = serde_json::from_str(&text).unwrap();
    let rows: Vec<Row> = recs
        .iter()
        .filter(|r| SERVED.contains(&r["model"].as_str().unwrap()))
        .map(|r| {
            let floor = r["panel"] == "jpeg8" && r["m2"].as_f64() == Some(0.0);
            Row {
                cluster: picture(r["ref"].as_str().unwrap()),
                broad: r["panel"] == "broad",
                block: r["block"].as_u64().unwrap(),
                level: r["level"].as_str().map(str::to_string),
                fail: if !r["passes"].as_bool().unwrap() && !floor {
                    1.0
                } else {
                    0.0
                },
                floor,
                log_rel_l2: r["rel_l2_lin_vs_true"].as_f64().unwrap().ln(),
            }
        })
        .collect();
    let mut results = Vec::new();
    // Floor ties are a code/output-stage class (fixed for A and C), excluded from every rate's denominator.
    let rate_diff = |rows: &[Row], a: &dyn Fn(&Row) -> bool, b: &dyn Fn(&Row) -> bool| {
        let fail: Vec<f64> = rows.iter().map(|r| r.fail).collect();
        let ka: Vec<bool> = rows.iter().map(|r| !r.floor && a(r)).collect();
        let kb: Vec<bool> = rows.iter().map(|r| !r.floor && b(r)).collect();
        move |d: &Draw<'_>| vec![masked_mean(d, &fail, &ka) - masked_mean(d, &fail, &kb)]
    };
    let small = |r: &Row| r.block <= 32;
    results.push(run(
        &rows,
        "failure rate b64 minus b8-b32, all panels",
        rate_diff(&rows, &|r| r.block == 64, &small),
    ));
    let broad: Vec<Row> = rows
        .iter()
        .filter(|r| r.broad)
        .map(|r| Row {
            cluster: r.cluster.clone(),
            level: r.level.clone(),
            ..*r
        })
        .collect();
    results.push(run(
        &broad,
        "failure rate b64 minus b8-b32, broad panel",
        rate_diff(&broad, &|r| r.block == 64, &small),
    ));
    results.push(run(
        &broad,
        "failure rate level 10 minus level 17, broad panel",
        rate_diff(&broad, &|r| r.level.as_deref() == Some("10"), &|r| {
            r.level.as_deref() == Some("17")
        }),
    ));
    for b in [8u64, 16, 32, 64] {
        let logv: Vec<f64> = rows.iter().map(|r| r.log_rel_l2).collect();
        let kf: Vec<bool> = rows.iter().map(|r| r.block == b && r.fail == 1.0).collect();
        let kp: Vec<bool> = rows
            .iter()
            .map(|r| r.block == b && r.fail == 0.0 && !r.floor)
            .collect();
        results.push(run(
            &rows,
            &format!("log Rel. L2 geometric-mean ratio failing/passing, b{b}, all panels"),
            move |d: &Draw<'_>| vec![masked_mean(d, &logv, &kf) - masked_mean(d, &logv, &kp)],
        ));
    }
    std::fs::write(&args[2], serde_json::to_string_pretty(&json!({"zenmetrics_stats_rev": "1e48c1eafcaa8afaa8b632c4f12d8181a971a248", "results": results})).unwrap()).unwrap();
}
