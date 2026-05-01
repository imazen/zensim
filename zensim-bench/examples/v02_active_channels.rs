//! Print V0_2's active (channel × scale) compute mask, mirroring
//! zensim's internal `active_channels` logic. Quick audit tool to
//! see which (channel, scale) slots V0_2 already skips.

use zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2;

fn main() {
    let weights: &[f64] = &LINEAR_WEIGHTS_PREVIEW_V0_2;
    let basic_fpc = 13;
    let n_scales = 4;
    let basic_total = n_scales * basic_fpc * 3;
    let chan_names = ["X", "Y", "B"];

    let has_weight = |base: usize, count: usize| -> bool {
        (base..base + count).all(|i| i < weights.len())
            && (base..base + count).any(|i| weights[i].abs() > 0.001)
    };

    println!("V0_2 channel × scale compute mask (active_channels equivalent):\n");
    println!("scale | X            | Y            | B            ");
    println!("------|--------------|--------------|--------------");
    let mut total_active = 0;
    let mut total_skipped = 0;
    for s in 0..n_scales {
        let mut row = format!("  s{} ", s);
        for c in 0..3 {
            let base = s * (basic_fpc * 3) + c * basic_fpc;
            let need_ssim = has_weight(base, 3);
            let need_hf = has_weight(base + 10, 3);
            let need_edge = has_weight(base + 3, 6) || need_hf;
            let need_mse = has_weight(base + 9, 1);
            let peak_base = basic_total + s * 18 + c * 6;
            let need_peak_ssim = has_weight(peak_base, 1) || has_weight(peak_base + 3, 1);
            let need_peak_edge = has_weight(peak_base + 1, 2) || has_weight(peak_base + 4, 2);
            let any = need_ssim || need_edge || need_mse || need_peak_ssim || need_peak_edge;
            if any {
                total_active += 1;
                let mut tags = Vec::new();
                if need_ssim {
                    tags.push("S");
                }
                if need_edge {
                    tags.push("E");
                }
                if need_mse {
                    tags.push("M");
                }
                if need_peak_ssim || need_peak_edge {
                    tags.push("P");
                }
                row.push_str(&format!(
                    "| {:<13}",
                    format!("{} ({})", chan_names[c], tags.join(""))
                ));
            } else {
                total_skipped += 1;
                row.push_str(&format!("| {:<13}", "SKIP"));
            }
        }
        println!("{}", row);
    }
    println!("\nlegend: S=ssim E=edge M=mse P=peak (max/p95)");
    println!(
        "\n{} active / {} skipped of {} (channel, scale) slots",
        total_active,
        total_skipped,
        total_active + total_skipped
    );

    // Per-feature weight-zero analysis
    let n_zero = weights.iter().filter(|w| w.abs() < 0.001).count();
    let n_total = weights.len();
    println!(
        "\nPer-feature: {}/{} weights below 0.001 threshold ({:.0}%)",
        n_zero,
        n_total,
        n_zero as f64 / n_total as f64 * 100.0
    );
}
