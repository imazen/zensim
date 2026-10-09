use super::adapter::Score;
use super::dataset::Table;
use super::*;
use zensim_validate::panel::{self, Orientation};

fn rank(p: &[f64], t: &[f64], limit: usize) -> Value {
    if p.len() < 3 {
        return missing("at least three paired scores needed");
    }
    if p.len() > limit {
        return missing(format!(
            "full PWRC is quadratic: {} rows exceed max_panel_rows={limit}; explicitly raise the limit with an adequate memory budget",
            p.len()
        ));
    }
    if !p.iter().any(|v| *v != p[0]) || !t.iter().any(|v| *v != t[0]) {
        return missing("constant predictions or targets: correlations undefined");
    }
    let stats = panel::compute_panel(p, t);
    let mut values = super::super::signed_quality_correlations(p, t);
    values["srocc_magnitude"] = json!(stats.srocc);
    values["krocc_magnitude"] = json!(stats.krocc);
    values["or"] = json!(stats.or_ratio);
    values["pwrc"] = json!(stats.pwrc);
    values["z_rmse"] = json!(stats.z_rmse);
    measured(values)
}

fn grouped(table: &Table, p: &[f64], t: &[f64]) -> Result<Value> {
    let keys = table.strings("reference_id")?;
    let groups: Vec<&str> = keys.iter().map(String::as_str).collect();
    let total = groups.iter().collect::<BTreeSet<_>>().len();
    Ok(
        match panel::per_group_srocc(p, t, &groups, 3, Orientation::HigherIsBetter) {
            Some(v) => measured(
                json!({"mean":v.mean,"median":v.median,"frac_negative":v.frac_negative,"frac_perfect":v.frac_perfect,"n_groups":v.n_groups,"total_groups":total,"unrankable_groups":total-v.n_groups,"min_rows":3}),
            ),
            None => missing("no reference has three rows and nonconstant predictions and targets"),
        },
    )
}

fn bands(d: &Dataset, p: &[f64], t: &[f64], limit: usize) -> Value {
    let Some([lo, hi]) = d.target_range else {
        return missing("declare target_range for merged quality bands");
    };
    let raw: Vec<f64> = t
        .iter()
        .map(|v| match d.target_direction {
            Direction::Higher => (v - lo) / (hi - lo),
            Direction::Lower => (-v - lo) / (hi - lo),
        })
        .collect();
    if raw.iter().any(|x| !(0.0..=1.0).contains(x)) {
        return failed("target outside declared target_range");
    }
    let normalized: Vec<f64> = raw
        .iter()
        .map(|x| match d.target_direction {
            Direction::Higher => *x,
            Direction::Lower => 1.0 - x,
        })
        .collect();
    let mut rows = Vec::new();
    for band in zensim_validate::bands::merged_bands(&normalized) {
        let idx = band.members(&normalized);
        let bp: Vec<f64> = idx.iter().map(|&i| p[i]).collect();
        let bt: Vec<f64> = idx.iter().map(|&i| t[i]).collect();
        let span = idx
            .iter()
            .map(|&i| normalized[i])
            .fold(f64::NEG_INFINITY, f64::max)
            - idx
                .iter()
                .map(|&i| normalized[i])
                .fold(f64::INFINITY, f64::min);
        let report = match zensim_validate::bands::not_measured_reason(idx.len(), span) {
            Some(reason) => missing(reason),
            None => rank(&bp, &bt, limit),
        };
        rows.push(json!({"band":band.label,"range":band.range_label(),"n":idx.len(),"normalized_target_span":span,"report":report}));
    }
    let count = rows
        .iter()
        .filter(|r| r["report"]["state"] == "measured")
        .count();
    let mut result = if count == 0 {
        missing("no quality band meets the canonical sample/span/variation gates")
    } else {
        measured(
            json!({"panels":rows,"target_range":[lo,hi],"measured_bands":count,"total_bands":rows.len()}),
        )
    };
    if count == 0 {
        result["panels"] = json!(rows);
    }
    result
}

fn pairwise(table: &Table, p: &[f64]) -> Result<Value> {
    use zensim_validate::pairwise::{Choice, PairwiseRow, agreement};
    let lookup: BTreeMap<&str, usize> = table
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r["id"].as_str(), i))
        .collect();
    let mut rows = Vec::new();
    let mut groups = BTreeMap::new();
    let mut group_scores = BTreeMap::new();
    for (i, row) in table.rows.iter().enumerate() {
        if row["right_id"].is_empty() && row["choice"].is_empty() {
            continue;
        }
        let right = lookup
            .get(row["right_id"].as_str())
            .context("unknown right_id in preference")?;
        let choice = match row["choice"].as_str() {
            "left" => Choice::Left,
            "right" => Choice::Right,
            other => bail!("choice must be left or right, got {other}"),
        };
        let key = row
            .get("preference_group")
            .context("pairwise needs preference_group column")?;
        ensure!(!key.is_empty(), "empty preference_group");
        ensure!(i != *right, "preference cannot compare a row with itself");
        if table.has("reference_id") {
            ensure!(
                row["reference_id"] == table.rows[*right]["reference_id"],
                "preference images must share a reference"
            );
        }
        if let Some(previous) = group_scores.insert(key, (p[i], p[*right])) {
            ensure!(
                previous == (p[i], p[*right]),
                "preference_group must have consistent left/right scores"
            );
        }
        let next = groups.len();
        let group = *groups.entry(key).or_insert(next);
        let weight = row
            .get("weight")
            .map(|s| s.parse::<f64>())
            .transpose()?
            .unwrap_or(1.0);
        ensure!(
            weight.is_finite() && weight > 0.0,
            "preference weight must be positive"
        );
        rows.push(PairwiseRow {
            group,
            s_left: p[i],
            s_right: p[*right],
            choice,
            weight,
        });
    }
    if rows.is_empty() {
        return Ok(missing("no preference rows"));
    }
    let s = agreement(&rows, groups.len());
    Ok(measured(
        json!({"n_groups":s.n_groups,"n_responses":s.n_responses,"accuracy":s.acc_response,"tie_rate":s.tie_rate,"ceiling":s.ceiling_response,"normalized_accuracy":s.acc_norm,"group_majority_accuracy":s.acc_group_majority,"n_majority_groups":s.n_groups_majority}),
    ))
}

fn dial(table: &Table, p: &[f64], metric: &Metric, dataset: &Dataset) -> Result<Value> {
    let Some(eps) = metric.ladder_epsilon else {
        return Ok(missing(
            "declare ladder_epsilon in this metric's score units",
        ));
    };
    let quality: Vec<f64> = table
        .numbers("quality")?
        .into_iter()
        .map(|q| dataset.quality_direction.unwrap().orient(q))
        .collect();
    let refs = table.strings("reference_id")?;
    let codecs = table.strings("codec")?;
    let mut curves: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for i in 0..p.len() {
        curves.entry((&refs[i], &codecs[i])).or_default().push(i);
    }
    let mut reports = Vec::new();
    let mut total = [0usize; 5];
    for ((reference, codec), mut indices) in curves {
        indices.sort_by(|&a, &b| quality[a].total_cmp(&quality[b]));
        let mut counts = [0usize; 5];
        for pair in indices.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            ensure!(
                quality[a] != quality[b],
                "duplicate quality setting in {reference}/{codec}; use separate configuration keys"
            );
            let same_pixels = table.rows[a]
                .get("pixel_sha256")
                .zip(table.rows[b].get("pixel_sha256"))
                .is_some_and(|(a, b)| !a.is_empty() && a == b);
            let bucket = super::super::ladder_step::bucket(p[b] - p[a], eps, same_pixels);
            counts[usize::from(bucket)] += 1;
        }
        for (sum, n) in total.iter_mut().zip(counts) {
            *sum += n;
        }
        reports.push(
            json!({"reference_id":reference,"codec":codec,"rungs":indices.len(),"counts":counts}),
        );
    }
    let pairs: usize = total.iter().sum();
    if pairs == 0 {
        return Ok(missing("no adjacent quality settings"));
    }
    let mono = 1.0 - total[1] as f64 / pairs as f64;
    let tied = total[3] as f64 / pairs as f64;
    let raw: Vec<f64> = p.iter().map(|x| metric.direction.orient(*x)).collect();
    let range = zensim_validate::dial_addressability::GridMeasure::from_pooled(&raw, mono, tied);
    Ok(measured(
        json!({"pairs":pairs,"monotonicity":mono,"dead_zone_rate":tied,
        "counts":total,"count_order":["forward","inversion","identical_pixels","dead_zone","sub_resolution"],"curves":reports,
        "raw_min":range.min,"raw_max":range.max,"raw_p5":range.p5,"raw_p95":range.p95,"raw_range":range.reach,
        "interpretation":"single-reference diagnostic; codec-inversion attribution and mentor floor qualification require a registered instrument",
        "ladder_epsilon":eps,"pixel_identity_available":table.has("pixel_sha256")}),
    ))
}

pub(super) fn evaluate(
    d: &Dataset,
    m: &Metric,
    table: &Table,
    scores: &[Score],
    limit: usize,
) -> Result<BTreeMap<String, Value>> {
    let mut out: BTreeMap<String, Value> = CRITERIA
        .iter()
        .map(|k| {
            (
                (*k).into(),
                missing("required dataset columns or specialized instrument not supplied"),
            )
        })
        .collect();
    if scores.iter().any(|s| s.value.is_none()) {
        for key in CRITERIA.iter().filter(|k| !INSTRUMENT_CRITERIA.contains(k)) {
            out.insert(
                (*key).into(),
                failed("scoring failed for one or more rows; no survivor-only assessment"),
            );
        }
        return Ok(out);
    }
    let p: Vec<f64> = scores
        .iter()
        .map(|s| m.direction.orient(s.value.unwrap()))
        .collect();
    if table.has("target") {
        let t: Vec<f64> = table
            .numbers("target")?
            .into_iter()
            .map(|v| d.target_direction.orient(v))
            .collect();
        out.insert("rank".into(), rank(&p, &t, limit));
        if out["rank"]["state"] == "measured" {
            out.get_mut("rank").unwrap()["measurements"]["target_kind"] = json!(d.target_kind);
            if let Some(threshold) = d.difference_threshold {
                out.get_mut("rank").unwrap()["measurements"]["difference_auc"] =
                    json!(super::super::difference_auc::ds_auc(&p, &t, threshold));
                out.get_mut("rank").unwrap()["measurements"]["difference_threshold"] =
                    json!(threshold);
            }
            if table.has("sigma") {
                let sigma = table.numbers("sigma")?;
                ensure!(sigma.iter().all(|s| *s > 0.0), "sigma must be positive");
                let mapped = panel::rescale_logistic(&p, &t);
                out.get_mut("rank").unwrap()["measurements"]["z_rmse_per_sample"] =
                    json!(panel::z_rmse_per_sample(&mapped, &t, &sigma));
            }
        }
        let mut scatter = super::super::scatter_json::assess(&p, &t);
        scatter.as_object_mut().unwrap().remove("normalized_pred");
        out.insert(
            "scatter".into(),
            if scatter["status"] == "MEASURED" {
                measured(scatter)
            } else {
                missing(scatter["reason"].as_str().unwrap_or("scatter undefined"))
            },
        );
        if table.has("reference_id") {
            out.insert("within_reference".into(), grouped(table, &p, &t)?);
        }
        out.insert("bands".into(), bands(d, &p, &t, limit));
        for key in ["content", "codec", "band"] {
            if table.has(key) {
                let labels = table.strings(key)?;
                let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
                for (i, label) in labels.iter().enumerate() {
                    groups.entry(label).or_default().push(i);
                }
                let panels: Vec<Value> = groups
                    .into_iter()
                    .map(|(label, idx)| {
                        let gp: Vec<f64> = idx.iter().map(|&i| p[i]).collect();
                        let gt: Vec<f64> = idx.iter().map(|&i| t[i]).collect();
                        json!({"label":label,"n":idx.len(),"report":rank(&gp,&gt,limit)})
                    })
                    .collect();
                out.get_mut("rank").unwrap()[format!("by_{key}")] = json!(panels);
            }
        }
    }
    if table.has("right_id") && table.has("choice") {
        out.insert("pairwise".into(), pairwise(table, &p)?);
    }
    if table.has("quality") && table.has("reference_id") && table.has("codec") {
        out.insert("dial".into(), dial(table, &p, m, d)?);
    }
    if table.has("severity_code") && table.has("reference_id") {
        let q = table.numbers("severity_code")?;
        let refs = table.strings("reference_id")?;
        let stats = zensim_validate::eval_report::severity_ramp(
            &refs,
            &q,
            &p,
            m.ladder_epsilon.unwrap_or(0.0),
        );
        out.insert("severity_ramp".into(),if stats.n_ramps+stats.n_signed_arms==0 {missing("no complete registered five-level severity ramps")} else {measured(json!({"n_ramps":stats.n_ramps,"n_signed":stats.n_signed,"monotone_fraction":stats.pct_monotone,"strict_fraction":stats.pct_strict,"mean_worst_inversion":stats.mean_worst_inv,"signed_monotone_fraction":stats.pct_signed_monotone,"n_signed_arms":stats.n_signed_arms}))});
    }
    if table.has("corruption_label") {
        let labels = table.strings("corruption_label")?;
        let stats = zensim_validate::eval_report::corruption_gate(&labels, &p);
        out.insert("corruption_ordering".into(),if stats.n_triples==0 {missing("no source-matched corruption/q20/q10 triples")} else {measured(json!({"n_triples":stats.n_triples,"pass_q20":stats.pass_q20,"pass_q10":stats.pass_q10,"scope":"legacy ordering diagnostic; not severity-aware integrity qualification"}))});
    }
    Ok(out)
}
