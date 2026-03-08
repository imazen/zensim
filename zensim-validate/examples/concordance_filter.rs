//! Filter synthetic training CSV by pairwise rank concordance per source image.
//!
//! For each source image, all rows (across all codecs and quality levels) are
//! collected. Every pair of rows is checked: do ssim2, butteraugli, and dssim
//! all agree on which image is better? Rows that participate in too many
//! disagreements are dropped.
//!
//! Algorithm:
//!   1. Group rows by source image
//!   2. For each pair (i, j) within a source group, check if all three metrics
//!      agree on relative ordering (concordant) or if any disagree (discordant)
//!   3. For each row, count its discordant pairs
//!   4. Iteratively remove the row with the most discordant pairs until all
//!      remaining pairs are concordant (or below a tolerance)
//!
//! Usage:
//!   cargo run --release --example concordance_filter -- <input.csv> [output.csv]

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

struct Row {
    ssim2: f64, // higher = better
    butt: f64,  // lower = better (distance)
    dssim: f64, // lower = better (distance)
    raw: Vec<String>,
}

/// Returns true if ssim2 and dssim agree on the ordering of a and b,
/// or if either metric is tied.
fn concordant(a: &Row, b: &Row) -> bool {
    let s = (a.ssim2 - b.ssim2).signum() as i8; //  1 = a better, -1 = b better
    let d = (b.dssim - a.dssim).signum() as i8; //  1 = a better, -1 = b better

    // If either metric sees zero difference, it's not discordant
    if s == 0 || d == 0 {
        return true;
    }

    s == d
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: concordance_filter <input.csv> [output.csv]");
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str());

    let mut rdr = csv::Reader::from_path(input_path).expect("Failed to open input CSV");
    let headers = rdr.headers().expect("No headers").clone();

    let col = |name: &str| -> usize {
        headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("Missing column: {name}"))
    };

    let i_source = col("source_path");
    let i_ssim2 = col("gpu_ssimulacra2");
    let i_butt = col("gpu_butteraugli");
    let i_dssim = col("dssim");

    let mut groups: HashMap<String, Vec<Row>> = HashMap::new();
    let mut total_rows = 0u64;
    let mut skipped = 0u64;

    for result in rdr.records() {
        let record = result.expect("CSV parse error");
        total_rows += 1;

        let source = record[i_source].to_string();
        let ssim2: f64 = match record[i_ssim2].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let butt: f64 = match record[i_butt].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let dssim: f64 = match record[i_dssim].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let raw: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        groups.entry(source).or_default().push(Row {
            ssim2,
            butt,
            dssim,
            raw,
        });
    }

    let n_sources = groups.len();
    let mut total_kept = 0u64;
    let mut total_dropped = 0u64;
    let mut total_pairs_checked = 0u64;
    let mut total_discordant_pairs = 0u64;
    let mut kept_rows: Vec<*const Vec<String>> = Vec::new();

    // Per-metric disagreement counters (which metric was the odd one out)
    let mut odd_ssim2 = 0u64;
    let mut odd_dssim = 0u64;
    let mut odd_ambiguous = 0u64;

    // Sources sorted for deterministic output
    let mut sources: Vec<String> = groups.keys().cloned().collect();
    sources.sort();

    for source in &sources {
        let rows = groups.get(source).unwrap();
        let n = rows.len();
        if n <= 1 {
            // Single row — always keep
            total_kept += n as u64;
            for r in rows {
                kept_rows.push(&r.raw as *const Vec<String>);
            }
            continue;
        }

        // alive[i] = true means row i is still in the candidate set
        let mut alive = vec![true; n];
        let mut remaining = n;

        loop {
            // Count discordant pairs per alive row
            let mut discord_count = vec![0u32; n];
            let mut any_discord = false;

            for i in 0..n {
                if !alive[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !alive[j] {
                        continue;
                    }
                    total_pairs_checked += 1;
                    if !concordant(&rows[i], &rows[j]) {
                        discord_count[i] += 1;
                        discord_count[j] += 1;
                        any_discord = true;
                        total_discordant_pairs += 1;

                        // ssim2 and dssim disagree — count which way
                        let s = (rows[i].ssim2 - rows[j].ssim2).signum() as i8;
                        let d = (rows[j].dssim - rows[i].dssim).signum() as i8;
                        // Also check what butteraugli thinks (informational)
                        let bu = (rows[j].butt - rows[i].butt).signum() as i8;
                        if bu != 0 {
                            if bu == s {
                                odd_dssim += 1;
                            } else if bu == d {
                                odd_ssim2 += 1;
                            } else {
                                odd_ambiguous += 1;
                            }
                        } else {
                            odd_ambiguous += 1;
                        }
                    }
                }
            }

            if !any_discord {
                break;
            }

            // Remove the row with the most discordant pairs
            let worst = (0..n)
                .filter(|&i| alive[i])
                .max_by_key(|&i| discord_count[i])
                .unwrap();
            alive[worst] = false;
            remaining -= 1;
            total_dropped += 1;
        }

        total_kept += remaining as u64;

        let source_short = Path::new(source)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| source.clone());

        if remaining < n {
            let dropped = n - remaining;
            if dropped > n / 3 {
                eprintln!("  heavy drop: {source_short}: {remaining}/{n} kept ({dropped} dropped)");
            }
        }

        for (i, r) in rows.iter().enumerate() {
            if alive[i] {
                kept_rows.push(&r.raw as *const Vec<String>);
            }
        }
    }

    println!("=== Rank Concordance Filter Summary ===");
    println!("Input:  {total_rows} rows ({n_sources} source groups)");
    if skipped > 0 {
        println!("Skipped (unparseable): {skipped}");
    }
    let parseable = total_kept + total_dropped;
    println!(
        "Kept:    {total_kept} ({:.1}%)",
        100.0 * total_kept as f64 / parseable as f64
    );
    println!(
        "Dropped: {total_dropped} ({:.1}%)",
        100.0 * total_dropped as f64 / parseable as f64
    );
    println!();
    println!("Pairwise stats:");
    println!("  Total pairs checked:    {total_pairs_checked}");
    println!("  Discordant pairs found: {total_discordant_pairs}");
    if total_pairs_checked > 0 {
        println!(
            "  Concordance rate:       {:.2}%",
            100.0 * (total_pairs_checked - total_discordant_pairs) as f64
                / total_pairs_checked as f64
        );
    }
    println!();
    println!("--- Butteraugli sides with (in discordant pairs) ---");
    println!("  butt agrees with ssim2 (dssim odd): {odd_dssim}");
    println!("  butt agrees with dssim (ssim2 odd): {odd_ssim2}");
    println!("  butt neutral/ambiguous:             {odd_ambiguous}");

    // Write output
    if let Some(out_path) = output_path {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(out_path).expect("Failed to create output CSV"),
        );
        let header_line: Vec<&str> = headers.iter().collect();
        writeln!(f, "{}", header_line.join(",")).unwrap();
        for ptr in &kept_rows {
            // SAFETY: all pointers point into `groups` which is still alive
            let row = unsafe { &**ptr };
            writeln!(f, "{}", row.join(",")).unwrap();
        }
        f.flush().unwrap();
        println!("\nWrote {total_kept} rows to {out_path}");
    } else {
        println!("\nDry run — pass an output path to write filtered CSV.");
    }
}
