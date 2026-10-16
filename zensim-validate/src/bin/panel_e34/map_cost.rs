//! Paired-round p95 differences; experimental reporting only.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    baseline_ns: Vec<f64>,
    candidates_ns: BTreeMap<String, Vec<f64>>,
}
fn p95(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    x[(0.95 * x.len() as f64).ceil() as usize - 1]
}
pub(super) fn run(bytes: &[u8]) -> Result<Value> {
    let packet: Packet = serde_json::from_slice(bytes)?;
    ensure!(packet.schema == "addrfix-map-cost-v1", "cost packet schema");
    let n = packet.baseline_ns.len();
    ensure!(
        n == 32 && !packet.candidates_ns.is_empty(),
        "32 paired rounds required"
    );
    let finite = |x: &[f64]| x.len() == n && x.iter().all(|v| v.is_finite() && *v > 0.);
    ensure!(finite(&packet.baseline_ns), "finite positive baseline");
    let groups: Vec<u32> = (0..n as u32).collect();
    let cfg = BootstrapConfig {
        resamples: 10000,
        seed: 20261016,
        level: 0.95,
    };
    let mut out = BTreeMap::new();
    for (name, values) in packet.candidates_ns {
        ensure!(finite(&values), "finite aligned candidate {name}");
        let iv = cluster_bootstrap(n, &Design::OneWay(&groups), &cfg, |draw| {
            let indices = draw.expand().rows;
            vec![
                p95(indices.iter().map(|&i| values[i]).collect())
                    - p95(indices.iter().map(|&i| packet.baseline_ns[i]).collect()),
            ]
        })?
        .remove(0);
        ensure!(
            iv.n_finite == 10000 && iv.lo.is_finite() && iv.hi.is_finite(),
            "incomplete cost interval"
        );
        out.insert(name,json!({"p95_ns":p95(values),"delta_ns":interval_json(&iv,n),"upper_ci_le_zero":iv.hi<=0.}));
    }
    Ok(
        json!({"schema":"addrfix-map-cost-result-v1","baseline_p95_ns":p95(packet.baseline_ns),
        "comparisons":out,"bootstrap":{"unit":"paired round","resamples":10000,"seed":20261016,
        "level":0.95,"point_p95":"nearest rank","interval_percentiles":"Hyndman-Fan type7"},
        "exploratory":true,"qualification_allowed":false}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_rank_p95_preserves_paired_shifts() {
        assert_eq!(p95((1..=32).map(f64::from).collect()), 31.);
        let input = json!({"schema":"addrfix-map-cost-v1","baseline_ns":vec![100.;32],
            "candidates_ns":{"shift":vec![90.;32]}});
        let result = run(&serde_json::to_vec(&input).unwrap()).unwrap();
        let iv = &result["comparisons"]["shift"]["delta_ns"];
        assert_eq!(iv["estimate"], -10.);
        assert_eq!(iv["lo"], -10.);
        assert_eq!(iv["hi"], -10.);
    }
}
