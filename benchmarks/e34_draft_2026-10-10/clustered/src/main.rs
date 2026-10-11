//! E34 draft revisions 3-4: source-clustered intervals for the comparative claims in the characterization (§1), computed
//! by zenmetrics-stats (`resample::cluster_bootstrap`, `Design::OneWay`, `weighted_mean`), percentile rule Hyndman-Fan
//! type 7, B = 10,000, seed 20261011, level 0.95.
//!
//! Input: `per_case.json` from `characterize.py` (stored, label-free G-STEER rows). Rows: the seven served models
//! (A, C seeds 0-2; production seed 0) x 135 cases. Cluster: the underlying source picture (the eight broad origins;
//! the three M3 fixtures, whose 256/384/576 renditions are one picture; the four KADID references).
//!
//! Revision 4 adds, per review round 3 (P2-2): the pictures contributing to each side of every contrast, and the
//! within-case form of the Rel. L2 claim (cases that fail in some served models and pass in others; per case, mean log
//! Rel. L2 of the failing models minus the passing models; clustered over the pictures that carry such cases).
//!
//! usage: e34-clustered <per_case.json> <out.json>
use serde_json::{Value, json};
use std::collections::BTreeMap;
use zenmetrics_stats::resample::{BootstrapConfig, Design, Draw, cluster_bootstrap, weighted_mean};

const SERVED: [&str; 7] = ["A-s0", "A-s1", "A-s2", "C-s0", "C-s1", "C-s2", "P-s0"];

struct Row {
    case: String,
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

fn run(
    clusters: &[&str],
    name: &str,
    sides: [usize; 2],
    stat: impl Fn(&Draw<'_>) -> Vec<f64> + Sync,
) -> Value {
    let mut ids: BTreeMap<&str, u32> = BTreeMap::new();
    let labels: Vec<u32> = clusters
        .iter()
        .map(|&c| {
            let n = ids.len() as u32;
            *ids.entry(c).or_insert(n)
        })
        .collect();
    let cfg = BootstrapConfig {
        resamples: 10_000,
        seed: 20_261_011,
        level: 0.95,
    };
    let out =
        cluster_bootstrap(clusters.len(), &Design::OneWay(&labels), &cfg, stat).expect("bootstrap");
    let i = out[0];
    json!({"statistic": name, "estimate": i.estimate, "lo": i.lo, "hi": i.hi, "se": i.se, "n_finite": i.n_finite,
           "p_zero": i.p_zero, "clusters": ids.len(), "rows": clusters.len(), "pictures_side_a": sides[0],
           "pictures_side_b": sides[1], "resamples": cfg.resamples, "seed": cfg.seed,
           "level": cfg.level, "percentile_rule": "Hyndman-Fan type 7 (zenmetrics-stats quantile7)",
           "design": "OneWay(source picture)"})
}

fn refs(l: &[String]) -> Vec<&str> {
    l.iter().map(String::as_str).collect()
}

/// Distinct pictures with at least one kept row.
fn pictures(rows: &[Row], keep: &[bool]) -> usize {
    let mut s = std::collections::BTreeSet::new();
    for (r, &k) in rows.iter().zip(keep) {
        if k {
            s.insert(r.cluster.as_str());
        }
    }
    s.len()
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
                case: r["case"].as_str().unwrap().to_string(),
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
    let labels = |rows: &[Row]| -> Vec<String> { rows.iter().map(|r| r.cluster.clone()).collect() };
    // Floor ties are a code/output-stage class (fixed for A and C), excluded from every rate's denominator.
    // Returns the statistic and the pictures contributing rows to each side.
    let rate_diff = |rows: &[Row], a: &dyn Fn(&Row) -> bool, b: &dyn Fn(&Row) -> bool| {
        let fail: Vec<f64> = rows.iter().map(|r| r.fail).collect();
        let ka: Vec<bool> = rows.iter().map(|r| !r.floor && a(r)).collect();
        let kb: Vec<bool> = rows.iter().map(|r| !r.floor && b(r)).collect();
        let sides = [pictures(rows, &ka), pictures(rows, &kb)];
        (sides, move |d: &Draw<'_>| {
            vec![masked_mean(d, &fail, &ka) - masked_mean(d, &fail, &kb)]
        })
    };
    let small = |r: &Row| r.block <= 32;
    let all_l = labels(&rows);
    let (sides, stat) = rate_diff(&rows, &|r| r.block == 64, &small);
    results.push(run(
        &refs(&all_l),
        "failure rate b64 minus b8-b32, all panels",
        sides,
        stat,
    ));
    let broad: Vec<Row> = rows
        .iter()
        .filter(|r| r.broad)
        .map(|r| Row {
            case: r.case.clone(),
            cluster: r.cluster.clone(),
            level: r.level.clone(),
            ..*r
        })
        .collect();
    let broad_l = labels(&broad);
    let (sides, stat) = rate_diff(&broad, &|r| r.block == 64, &small);
    results.push(run(
        &refs(&broad_l),
        "failure rate b64 minus b8-b32, broad panel",
        sides,
        stat,
    ));
    let (sides, stat) = rate_diff(&broad, &|r| r.level.as_deref() == Some("10"), &|r| {
        r.level.as_deref() == Some("17")
    });
    results.push(run(
        &refs(&broad_l),
        "failure rate level 10 minus level 17, broad panel",
        sides,
        stat,
    ));
    for b in [8u64, 16, 32, 64] {
        let logv: Vec<f64> = rows.iter().map(|r| r.log_rel_l2).collect();
        let kf: Vec<bool> = rows.iter().map(|r| r.block == b && r.fail == 1.0).collect();
        let kp: Vec<bool> = rows
            .iter()
            .map(|r| r.block == b && r.fail == 0.0 && !r.floor)
            .collect();
        let sides = [pictures(&rows, &kf), pictures(&rows, &kp)];
        results.push(run(
            &refs(&all_l),
            &format!("log Rel. L2 geometric-mean ratio failing/passing, b{b}, all panels"),
            sides,
            move |d: &Draw<'_>| vec![masked_mean(d, &logv, &kf) - masked_mean(d, &logv, &kp)],
        ));
    }
    // Within case (revision 4): a (case, block) is "mixed" when some served models fail it and others pass it (floor
    // ties excluded). Its value is the mean log Rel. L2 of the failing models minus that of the passing models, so the
    // picture, distortion and block are held fixed and only the model differs.
    struct Mixed {
        cluster: String,
        block: u64,
        fail: Vec<f64>,
        pass: Vec<f64>,
    }
    let mut mixed: BTreeMap<&str, Mixed> = BTreeMap::new();
    for r in rows.iter().filter(|r| !r.floor) {
        let m = mixed.entry(r.case.as_str()).or_insert_with(|| Mixed {
            cluster: r.cluster.clone(),
            block: r.block,
            fail: Vec::new(),
            pass: Vec::new(),
        });
        if r.fail == 1.0 {
            m.fail.push(r.log_rel_l2);
        } else {
            m.pass.push(r.log_rel_l2);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let items: Vec<(String, u64, f64)> = mixed
        .values()
        .filter(|m| !m.fail.is_empty() && !m.pass.is_empty())
        .map(|m| (m.cluster.clone(), m.block, mean(&m.fail) - mean(&m.pass)))
        .collect();
    for b in [0u64, 8, 16, 32, 64] {
        let sel: Vec<&(String, u64, f64)> = items.iter().filter(|it| b == 0 || it.1 == b).collect();
        let l: Vec<&str> = sel.iter().map(|it| it.0.as_str()).collect();
        let v: Vec<f64> = sel.iter().map(|it| it.2).collect();
        let g = l.iter().collect::<std::collections::BTreeSet<_>>().len();
        let not_larger = v.iter().filter(|&&x| x <= 0.0).count();
        let what = if b == 0 {
            "all blocks".to_string()
        } else {
            format!("b{b}")
        };
        let mut res = run(
            &l,
            &format!("log Rel. L2 within-case ratio failing/passing models, {what}"),
            [g, g],
            move |d: &Draw<'_>| vec![weighted_mean(&v, d.row_weights())],
        );
        res["mixed_cases"] = json!(sel.len());
        res["cases_failing_not_larger"] = json!(not_larger);
        results.push(res);
    }
    // Per broad picture: failures and non-floor rows at b64 and at b8-b32 (counts only; the rate contrast's like-for-like
    // reading, revision 4).
    let mut per_picture: BTreeMap<&str, [u64; 4]> = BTreeMap::new();
    for r in broad.iter().filter(|r| !r.floor) {
        let e = per_picture.entry(r.cluster.as_str()).or_default();
        let o = if r.block == 64 { 2 } else { 0 };
        e[o] += r.fail as u64;
        e[o + 1] += 1;
    }
    let per_picture: Vec<Value> = per_picture
        .iter()
        .map(|(p, c)| json!({"picture": p, "small_fail": c[0], "small_rows": c[1], "b64_fail": c[2], "b64_rows": c[3]}))
        .collect();
    std::fs::write(&args[2], serde_json::to_string_pretty(&json!({"zenmetrics_stats_rev": "1e48c1eafcaa8afaa8b632c4f12d8181a971a248", "results": results, "broad_per_picture_b64_vs_small": per_picture})).unwrap()).unwrap();
}
