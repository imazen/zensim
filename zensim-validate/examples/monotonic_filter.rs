//! Filter synthetic training CSV to pairs where dssim, butteraugli, and ssimulacra2
//! all agree on monotonic quality ordering within each (source, codec) group.
//!
//! "Monotonic" means: for every pair of rows with quality q_a < q_b within the
//! same (source, codec) group, all three metrics agree that q_b is better.
//!
//! Usage:
//!   cargo run --release --example monotonic_filter -- <input.csv> [output.csv]
//!
//! If output.csv is omitted, prints stats only (dry run).

use std::collections::HashMap;
use std::env;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: monotonic_filter <input.csv> [output.csv]");
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str());

    // Parse CSV into groups keyed by (source, codec)
    let mut rdr = csv::Reader::from_path(input_path).expect("Failed to open input CSV");
    let headers = rdr.headers().expect("No headers").clone();

    let col = |name: &str| -> usize {
        headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("Missing column: {name}"))
    };

    let i_source = col("source_path");
    let i_codec = col("codec");
    let i_quality = col("quality");
    let i_ssim2 = col("gpu_ssimulacra2");
    let i_butt = col("gpu_butteraugli");
    let i_dssim = col("dssim");

    struct Row {
        quality: f64,
        ssim2: f64, // higher = better
        butt: f64,  // lower = better (distance)
        dssim: f64, // lower = better (distance)
        raw: Vec<String>,
    }

    let mut groups: HashMap<(String, String), Vec<Row>> = HashMap::new();
    let mut total_rows = 0u64;
    let mut skipped_nan = 0u64;

    for result in rdr.records() {
        let record = result.expect("CSV parse error");
        total_rows += 1;

        let source = record[i_source].to_string();
        let codec = record[i_codec].to_string();
        let quality: f64 = match record[i_quality].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped_nan += 1;
                continue;
            }
        };
        let ssim2: f64 = match record[i_ssim2].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped_nan += 1;
                continue;
            }
        };
        let butt: f64 = match record[i_butt].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped_nan += 1;
                continue;
            }
        };
        let dssim: f64 = match record[i_dssim].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped_nan += 1;
                continue;
            }
        };

        let raw: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        groups.entry((source, codec)).or_default().push(Row {
            quality,
            ssim2,
            butt,
            dssim,
            raw,
        });
    }

    // Sort each group by quality ascending
    for rows in groups.values_mut() {
        rows.sort_by(|a, b| a.quality.partial_cmp(&b.quality).unwrap());
    }

    // For each group, find the longest monotonic-in-all-three subsequence.
    // A row can be included if all three metrics are strictly monotonic with
    // respect to every previously included row.
    //
    // Greedy forward scan: include a row if it's consistent with all prior included rows.
    // ssim2 must increase (higher = better), butt must decrease, dssim must decrease.

    let mut kept_rows: Vec<&[String]> = Vec::new();
    let mut total_kept = 0u64;
    let mut total_dropped = 0u64;
    let mut groups_all_kept = 0u64;
    let mut groups_partial = 0u64;
    let mut groups_empty = 0u64;

    // Per-metric violation counters
    let mut violations_ssim2_only = 0u64;
    let mut violations_butt_only = 0u64;
    let mut violations_dssim_only = 0u64;
    let mut violations_multi = 0u64;

    // Per-codec stats
    let mut codec_total: HashMap<String, u64> = HashMap::new();
    let mut codec_kept: HashMap<String, u64> = HashMap::new();

    let n_groups = groups.len();

    for ((source, codec), rows) in &groups {
        let n = rows.len();
        *codec_total.entry(codec.clone()).or_default() += n as u64;

        // Greedy: walk forward, keep rows that are monotonic with the last kept row
        let mut included: Vec<usize> = Vec::new();

        for i in 0..n {
            let dominated = if let Some(&last) = included.last() {
                let prev = &rows[last];
                let cur = &rows[i];
                // All three must agree this row is better (higher quality)
                cur.ssim2 > prev.ssim2 && cur.butt < prev.butt && cur.dssim < prev.dssim
            } else {
                true // first row always included
            };

            if dominated {
                included.push(i);
            } else if let Some(&last) = included.last() {
                // Count which metric(s) violated
                let prev = &rows[last];
                let cur = &rows[i];
                let s_bad = cur.ssim2 <= prev.ssim2;
                let b_bad = cur.butt >= prev.butt;
                let d_bad = cur.dssim >= prev.dssim;
                match (s_bad, b_bad, d_bad) {
                    (true, false, false) => violations_ssim2_only += 1,
                    (false, true, false) => violations_butt_only += 1,
                    (false, false, true) => violations_dssim_only += 1,
                    _ => violations_multi += 1,
                }
            }
        }

        let kept = included.len();
        let dropped = n - kept;
        total_kept += kept as u64;
        total_dropped += dropped as u64;
        *codec_kept.entry(codec.clone()).or_default() += kept as u64;

        if kept == n {
            groups_all_kept += 1;
        } else if kept == 0 {
            groups_empty += 1;
        } else {
            groups_partial += 1;
        }

        // Report groups with heavy drops
        if n >= 5 && kept < n / 2 {
            let source_short = Path::new(source)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| source.clone());
            eprintln!("  heavy drop: {source_short} × {codec}: {kept}/{n} kept");
        }

        for &idx in &included {
            kept_rows.push(&rows[idx].raw);
        }
    }

    // Print summary
    println!("=== Monotonic Filter Summary ===");
    println!("Input:  {total_rows} rows ({n_groups} source×codec groups)");
    if skipped_nan > 0 {
        println!("Skipped (unparseable): {skipped_nan}");
    }
    println!(
        "Kept:   {total_kept} ({:.1}%)",
        100.0 * total_kept as f64 / (total_kept + total_dropped) as f64
    );
    println!(
        "Dropped: {total_dropped} ({:.1}%)",
        100.0 * total_dropped as f64 / (total_kept + total_dropped) as f64
    );
    println!();
    println!(
        "Groups: {groups_all_kept} fully monotonic, {groups_partial} partial, {groups_empty} empty"
    );
    println!();
    println!("--- Violation breakdown (dropped rows) ---");
    println!("  ssim2 only:      {violations_ssim2_only}");
    println!("  butteraugli only: {violations_butt_only}");
    println!("  dssim only:      {violations_dssim_only}");
    println!("  multiple metrics: {violations_multi}");
    println!();

    // Per-codec table
    let mut codecs: Vec<&String> = codec_total.keys().collect();
    codecs.sort();
    println!("--- Per-codec retention ---");
    println!("{:<40} {:>7} {:>7} {:>6}", "Codec", "Total", "Kept", "%");
    for codec in &codecs {
        let t = codec_total[*codec];
        let k = codec_kept.get(*codec).copied().unwrap_or(0);
        println!(
            "{:<40} {:>7} {:>7} {:>5.1}%",
            codec,
            t,
            k,
            100.0 * k as f64 / t as f64
        );
    }

    // Write output if requested
    if let Some(out_path) = output_path {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(out_path).expect("Failed to create output CSV"),
        );
        // Write header
        let header_line: Vec<&str> = headers.iter().collect();
        writeln!(f, "{}", header_line.join(",")).unwrap();
        for row in &kept_rows {
            writeln!(f, "{}", row.join(",")).unwrap();
        }
        f.flush().unwrap();
        println!("\nWrote {total_kept} rows to {out_path}");
    } else {
        println!("\nDry run — pass an output path to write filtered CSV.");
    }
}
