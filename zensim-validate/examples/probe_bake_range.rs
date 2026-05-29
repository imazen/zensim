//! Diagnostic: load a bake, score the CID22 holdout, dump pred percentiles + range.
//! Used to verify the #40 hidden=1 bake-emit fix is producing genuinely varying output.
use std::fs;

use parquet::file::reader::{FileReader, SerializedFileReader};
use zenpredict::{Model, Predictor};

fn main() {
    let bake = std::env::args()
        .nth(1)
        .expect("usage: probe_bake_range BAKE [PARQUET]");
    let pq_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet".to_string()
    });
    let bytes = fs::read(&bake).expect("read bake");
    let model = Model::from_bytes(&bytes).expect("decode bake");
    let mut predictor = Predictor::new(&model);
    let n_feat: usize = 372;
    let file = fs::File::open(&pq_path).expect("open pq");
    let reader = SerializedFileReader::new(file).expect("reader");
    let row_iter = reader.get_row_iter(None).expect("rows");
    let mut preds: Vec<f32> = Vec::with_capacity(4292);
    let mut feats: Vec<f32> = vec![0.0; n_feat];
    let mut humans: Vec<f64> = Vec::with_capacity(4292);
    for rec in row_iter {
        let rec = rec.expect("rec");
        let cols = rec.into_columns();
        let mut human: f64 = 0.0;
        for (i, (name, field)) in cols.iter().enumerate() {
            if i < n_feat {
                let f = match field {
                    parquet::record::Field::Double(v) => *v as f32,
                    parquet::record::Field::Float(v) => *v,
                    _ => 0.0,
                };
                feats[i] = f;
            }
            if name == "human_score" {
                human = match field {
                    parquet::record::Field::Double(v) => *v,
                    parquet::record::Field::Float(v) => *v as f64,
                    _ => f64::NAN,
                };
            }
        }
        humans.push(human);
        let out = predictor.predict_transformed(&feats).expect("predict");
        preds.push(out[0]);
    }
    let mut sorted = preds.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let p = |frac: f64| sorted[((n as f64 - 1.0) * frac).floor() as usize];
    println!(
        "n={} pred range: p1={:.4} p25={:.4} p50={:.4} p75={:.4} p99={:.4} span={:.4}",
        n,
        p(0.01),
        p(0.25),
        p(0.5),
        p(0.75),
        p(0.99),
        p(0.99) - p(0.01)
    );
    println!(
        "human_score range: min={:.4} max={:.4}",
        humans.iter().cloned().fold(f64::INFINITY, f64::min),
        humans.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!("First 10: pred,human");
    for i in 0..10.min(n) {
        println!("  {:.4}  {:.4}", preds[i], humans[i]);
    }
    println!("Last 5: pred,human");
    for i in (n.saturating_sub(5))..n {
        println!("  {:.4}  {:.4}", preds[i], humans[i]);
    }
}
