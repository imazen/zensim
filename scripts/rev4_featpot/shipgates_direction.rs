//! Private adapter: frozen A7r agreement flags to canonical paired cluster stats.
#![forbid(unsafe_code)]
use std::io::{self, Read};
use zenstats::resample::{BootstrapConfig, Design, cluster_bootstrap, weighted_mean};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut labels = Vec::new();
    let mut values = vec![Vec::new(); 5];
    for line in input.lines() {
        let cols: Vec<_> = line.split('\t').collect();
        assert_eq!(cols.len(), 6);
        labels.push(cols[0].parse::<u32>()?);
        for (col, v) in values.iter_mut().enumerate() {
            let flag = cols[col + 1].parse::<u8>()?;
            assert!(flag <= 1);
            v.push(f64::from(flag));
        }
    }
    // One row per edge. Carry identical ladder multiplicities through every
    // scorer and difference: unequal edge counts retain their edge weights.
    for col in 0..3 {
        values.push(
            values[col]
                .iter()
                .zip(&values[3])
                .map(|(a, b)| a - b)
                .collect(),
        );
    }
    let calc = |level| {
        cluster_bootstrap(
            labels.len(),
            &Design::OneWay(&labels),
            &BootstrapConfig {
                resamples: 10_000,
                seed: 20261017,
                level,
            },
            |draw| {
                values
                    .iter()
                    .map(|v| weighted_mean(v, draw.row_weights()))
                    .collect()
            },
        )
    };
    let descriptive = calc(0.95)?;
    let directional = calc(0.90)?; // Lower endpoint is the registered 5th percentile.
    println!("column\testimate\tlo95\thi95\tlower5\tn_finite");
    for (i, (a, b)) in descriptive.iter().zip(&directional).enumerate() {
        assert_eq!(a.n_finite, 10_000);
        assert_eq!(b.n_finite, 10_000);
        assert_eq!(a.estimate, b.estimate);
        println!(
            "{i}\t{}\t{}\t{}\t{}\t{}",
            a.estimate, a.lo, a.hi, b.lo, a.n_finite
        );
    }
    Ok(())
}
