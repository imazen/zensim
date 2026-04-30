//! Brute-force the error of `cbrt-first` XYB vs the standard
//! `matrix-then-cbrt` XYB.
//!
//! Standard XYB pipeline:
//!   1. linear = srgb_u8_to_linear(u8)
//!   2. mixed_LMS = M · linear + K_B0     (linear-space matrix + bias)
//!   3. t_LMS = cbrt(mixed_LMS)
//!   4. opponent: X = (t_L - t_M)/2 · 14 + 0.42
//!                Y = (t_L + t_M)/2 + 0.01
//!                B = t_S - Y + 0.55
//!
//! cbrt-first variant (lets us fold linearize+cbrt into a single u8→f32 LUT
//! for matlut-style perf):
//!   1. cbrt_linear_RGB = cbrt(srgb_u8_to_linear(u8) + K_B0_per_channel)
//!   2. t'_LMS = M' · cbrt_linear_RGB
//!   3. same opponent assembly
//!
//! M' choices tested:
//!   (a) M' = M (no calibration — pure reordering)
//!   (b) M' = least-squares fit minimising ||t_LMS - M'·cbrt_linear_RGB||
//!         over a 32³ sampled grid
//!
//! For each variant, walks all 256³ = 16,777,216 (R, G, B) and reports
//! per-channel max / mean / p99 error in the final positive-XYB outputs.

use linear_srgb::default::srgb_u8_to_linear;

const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4;

const M: [[f32; 3]; 3] = [
    [K_M00, K_M01, K_M02],
    [K_M10, K_M11, K_M12],
    [K_M20, K_M21, K_M22],
];

#[inline]
fn cbrtf(x: f32) -> f32 {
    x.cbrt()
}

/// Standard XYB: linear → matrix+bias → cbrt → opponent → positive shift.
fn standard_xyb(r: u8, g: u8, b: u8) -> [f32; 3] {
    let lr = srgb_u8_to_linear(r);
    let lg = srgb_u8_to_linear(g);
    let lb = srgb_u8_to_linear(b);
    let m_l = (M[0][0] * lr + M[0][1] * lg + M[0][2] * lb + K_B0).max(0.0);
    let m_m = (M[1][0] * lr + M[1][1] * lg + M[1][2] * lb + K_B0).max(0.0);
    let m_s = (M[2][0] * lr + M[2][1] * lg + M[2][2] * lb + K_B0).max(0.0);
    let t_l = cbrtf(m_l);
    let t_m = cbrtf(m_m);
    let t_s = cbrtf(m_s);
    let ab = -cbrtf(K_B0);
    let c0 = t_l + ab;
    let c1 = t_m + ab;
    let x = (c0 - c1) * 0.5 * 14.0 + 0.42;
    let y = (c0 + c1) * 0.5 + 0.01;
    let b_out = t_s - (c0 + c1) * 0.5 + 0.55;
    [x, y, b_out]
}

/// cbrt-first XYB: cbrt(linear) → matrix M' → opponent → positive shift.
fn cbrt_first_xyb(r: u8, g: u8, b: u8, m_prime: &[[f32; 3]; 3]) -> [f32; 3] {
    let lr = srgb_u8_to_linear(r);
    let lg = srgb_u8_to_linear(g);
    let lb = srgb_u8_to_linear(b);
    // Pre-add K_B0 to linear inputs to keep the cbrt argument away from 0
    // (matches the standard path's bias semantics).
    let cr = cbrtf(lr + K_B0);
    let cg = cbrtf(lg + K_B0);
    let cb = cbrtf(lb + K_B0);
    let t_l = m_prime[0][0] * cr + m_prime[0][1] * cg + m_prime[0][2] * cb;
    let t_m = m_prime[1][0] * cr + m_prime[1][1] * cg + m_prime[1][2] * cb;
    let t_s = m_prime[2][0] * cr + m_prime[2][1] * cg + m_prime[2][2] * cb;
    let ab = -cbrtf(K_B0);
    let c0 = t_l + ab;
    let c1 = t_m + ab;
    let x = (c0 - c1) * 0.5 * 14.0 + 0.42;
    let y = (c0 + c1) * 0.5 + 0.01;
    let b_out = t_s - (c0 + c1) * 0.5 + 0.55;
    [x, y, b_out]
}

/// Least-squares fit M' so that M' · cbrt(linear_RGB) ≈ standard t_LMS,
/// over a 32³ grid of u8 inputs (32_768 samples). Fits each output row
/// independently (3 separate 3-parameter fits).
fn calibrate_m_prime() -> [[f32; 3]; 3] {
    // Build sample set
    let step = 8u32;
    let mut samples_x: Vec<[f64; 3]> = Vec::with_capacity(33 * 33 * 33);
    let mut targets: Vec<[f64; 3]> = Vec::with_capacity(33 * 33 * 33);
    for r in (0..=255u32).step_by(step as usize) {
        for g in (0..=255u32).step_by(step as usize) {
            for b in (0..=255u32).step_by(step as usize) {
                let lr = srgb_u8_to_linear(r as u8) as f64;
                let lg = srgb_u8_to_linear(g as u8) as f64;
                let lb = srgb_u8_to_linear(b as u8) as f64;
                let cr = (lr + K_B0 as f64).cbrt();
                let cg = (lg + K_B0 as f64).cbrt();
                let cb_val = (lb + K_B0 as f64).cbrt();
                samples_x.push([cr, cg, cb_val]);

                let m_l = (M[0][0] as f64 * lr + M[0][1] as f64 * lg + M[0][2] as f64 * lb
                    + K_B0 as f64)
                    .max(0.0);
                let m_m = (M[1][0] as f64 * lr + M[1][1] as f64 * lg + M[1][2] as f64 * lb
                    + K_B0 as f64)
                    .max(0.0);
                let m_s = (M[2][0] as f64 * lr + M[2][1] as f64 * lg + M[2][2] as f64 * lb
                    + K_B0 as f64)
                    .max(0.0);
                targets.push([m_l.cbrt(), m_m.cbrt(), m_s.cbrt()]);
            }
        }
    }

    // Normal-equations solver for 3-param linear regression per output row.
    // For each output channel k: solve A^T A · w = A^T y, where A is N×3
    // (columns = cr, cg, cb), y is N×1 (target_k).
    let mut ata = [[0.0f64; 3]; 3];
    for sx in &samples_x {
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += sx[i] * sx[j];
            }
        }
    }
    let mut m_prime = [[0.0f32; 3]; 3];
    for k in 0..3 {
        let mut atb = [0.0f64; 3];
        for (sx, t) in samples_x.iter().zip(targets.iter()) {
            for i in 0..3 {
                atb[i] += sx[i] * t[k];
            }
        }
        // Solve 3x3 system via gaussian elimination
        let mut a = ata;
        let mut b = atb;
        for i in 0..3 {
            let mut piv = i;
            for j in (i + 1)..3 {
                if a[j][i].abs() > a[piv][i].abs() {
                    piv = j;
                }
            }
            a.swap(i, piv);
            b.swap(i, piv);
            for j in (i + 1)..3 {
                let f = a[j][i] / a[i][i];
                for kk in i..3 {
                    a[j][kk] -= f * a[i][kk];
                }
                b[j] -= f * b[i];
            }
        }
        let mut w = [0.0f64; 3];
        for i in (0..3).rev() {
            let mut s = b[i];
            for j in (i + 1)..3 {
                s -= a[i][j] * w[j];
            }
            w[i] = s / a[i][i];
        }
        for i in 0..3 {
            m_prime[k][i] = w[i] as f32;
        }
    }
    m_prime
}

struct ErrStats {
    max: f32,
    sum: f64,
    sum_sq: f64,
    n: usize,
    samples: Vec<f32>, // for percentile
}

impl ErrStats {
    fn new() -> Self {
        Self {
            max: 0.0,
            sum: 0.0,
            sum_sq: 0.0,
            n: 0,
            samples: Vec::new(),
        }
    }
    fn push(&mut self, e: f32) {
        let abs = e.abs();
        if abs > self.max {
            self.max = abs;
        }
        self.sum += abs as f64;
        self.sum_sq += (abs * abs) as f64;
        self.n += 1;
        // Reservoir-sample to keep memory bounded for percentile estimation.
        if self.samples.len() < 1_000_000 {
            self.samples.push(abs);
        }
    }
    fn report(&mut self) -> String {
        self.samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = self.samples[self.samples.len() / 2];
        let p99 = self.samples[self.samples.len() * 99 / 100];
        let p999 = self.samples[self.samples.len() * 999 / 1000];
        let mean = self.sum / self.n as f64;
        let rms = (self.sum_sq / self.n as f64).sqrt();
        format!(
            "max={:.6}  rms={:.6}  mean={:.6}  p50={:.6}  p99={:.6}  p99.9={:.6}",
            self.max, rms, mean, p50, p99, p999,
        )
    }
}

fn run(label: &str, m_prime: &[[f32; 3]; 3]) {
    println!(
        "\n=== {label} (M' = {m_prime:?}) ===",
        label = label,
        m_prime = m_prime
    );
    let mut x_err = ErrStats::new();
    let mut y_err = ErrStats::new();
    let mut b_err = ErrStats::new();

    for r in 0..=255u32 {
        for g in 0..=255u32 {
            for b in 0..=255u32 {
                let std_xyb = standard_xyb(r as u8, g as u8, b as u8);
                let cbrt_xyb = cbrt_first_xyb(r as u8, g as u8, b as u8, m_prime);
                x_err.push(cbrt_xyb[0] - std_xyb[0]);
                y_err.push(cbrt_xyb[1] - std_xyb[1]);
                b_err.push(cbrt_xyb[2] - std_xyb[2]);
            }
        }
    }
    println!("X channel:  {}", x_err.report());
    println!("Y channel:  {}", y_err.report());
    println!("B channel:  {}", b_err.report());

    // Reference scale of XYB output values: at gray midpoint, X≈0.42, Y≈0.5,
    // B≈0.55; values span roughly [0.0, 1.0] for all 3 channels. So a max
    // error of 0.001 is ~0.1% of full scale, 0.01 is 1%.
}

fn main() {
    println!("# Brute-force XYB error test: cbrt-first vs standard");
    println!("# Walks all 256^3 = 16,777,216 RGB inputs.");
    println!("# Standard XYB output range is ~[0.0, 1.0] per channel.");

    // Variant (a): no calibration — just reorder cbrt and matrix.
    run("(a) no calibration: M' = M", &M);

    // Variant (b): calibrated M' via least squares.
    let m_prime = calibrate_m_prime();
    run("(b) calibrated: M' = least-squares fit", &m_prime);

    println!("\nM' calibrated values for reference:");
    for row in m_prime.iter() {
        println!("  {:?}", row);
    }
}
