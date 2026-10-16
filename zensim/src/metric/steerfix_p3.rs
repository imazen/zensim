//! Registered P3, report-only: reuse the existing f64 loaded-network reference.
//! This module is reachable only through the opt-in steering instrument.
use super::*;

#[test]
fn e34_registered_switch_diagnostic() {
    let path = std::env::var("E34_P3_JOB").expect("explicit registered P3 job");
    let bytes = std::fs::read(path).unwrap();
    let job: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut cases = Vec::new();
    for case in job["cases"].as_array().unwrap() {
        let blob = std::fs::read(case["model"].as_str().unwrap()).unwrap();
        assert_eq!(sha(&blob), case["model_sha256"].as_str().unwrap());
        let model = crate::mlp::Model::from_bytes(&blob).unwrap();
        let reference = Reference::new(&model);
        let feature_bytes = std::fs::read(case["feature_file"].as_str().unwrap()).unwrap();
        assert_eq!(
            sha(&feature_bytes),
            case["feature_sha256"].as_str().unwrap()
        );
        let width = case["feature_width"].as_u64().unwrap() as usize;
        assert!(width > 0 && feature_bytes.len().is_multiple_of(width * 8));
        let rows: Vec<Vec<f64>> = feature_bytes
            .chunks_exact(width * 8)
            .map(|row| {
                row.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b))
                    .collect()
            })
            .collect();
        assert_eq!(rows.len(), case["blocks"].as_u64().unwrap() as usize + 1);
        let base = &rows[0];
        let h0 = reference.hidden(base);
        let value = |input: &Input, row: &[f64]| match *input {
            Input::Direct(a) => row[a],
            Input::Product(a, b) => row[a] * row[b],
        };
        let mut switches = Vec::new();
        let mut unit_counts = std::collections::BTreeMap::new();
        for (block, repair) in rows.iter().skip(1).enumerate() {
            let h1 = reference.hidden(repair);
            for (unit, (&a, &b)) in h0.iter().zip(&h1).enumerate() {
                if (a > 0.) == (b > 0.) {
                    continue;
                }
                *unit_counts.entry(unit).or_insert(0usize) += 1;
                let mut terms: Vec<_> = reference
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, input)| {
                        let delta =
                            (value(input, repair) - value(input, base)) / reference.scales[i];
                        let contribution = delta * reference.w1[i * h0.len() + unit];
                        let ids = match *input {
                            Input::Direct(id) => vec![id],
                            Input::Product(a, b) => vec![a, b],
                        };
                        (i, ids, contribution)
                    })
                    .collect();
                let sum = terms.iter().map(|x| x.2).sum::<f64>();
                terms.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
                switches.push(json!({"block_index":block,"unit":unit,"base":a,"repair":b,
                    "preactivation_delta":b-a,"decomposition_residual":sum-(b-a),
                    "top_inputs":terms.iter().take(10).map(|(slot,ids,c)|json!({
                        "model_input":slot,"feature_ids":ids,"contribution":c})).collect::<Vec<_>>()}));
            }
        }
        cases.push(
            json!({"model":case["name"],"model_sha256":case["model_sha256"],
            "key":case["key"],"block":case["block"],"repairs":rows.len()-1,
            "units_with_switches":unit_counts,"switches":switches}),
        );
    }
    let output = job["output"].as_str().unwrap();
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .unwrap();
    writeln!(
        f,
        "{}",
        serde_json::to_string_pretty(&json!({
        "schema":"e34-p3-switches-v1","report_only":true,"job_sha256":sha(&bytes),
        "method":"existing f64 loaded-network precision reference; endpoint ReLU activity; no gate",
        "cases":cases}))
        .unwrap()
    )
    .unwrap();
}
