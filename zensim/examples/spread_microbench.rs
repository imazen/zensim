//! #70 lever-1 diagnostic: isolate `box_spread_merge_f32` serial vs
//! parallel cost at the production plane sizes (padded-width scale planes
//! of a 576²/1152² compare). Not a shipping benchmark — a lever probe.
//!
//! ```sh
//! cargo run --release -p zensim --features custom-profiles \
//!   --example spread_microbench
//! ```

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(51);
    // (w, h) per scale for 576² and 1152² compares (padded widths).
    let sizes: &[(usize, usize)] = &[
        (592, 576),
        (296, 288),
        (148, 144),
        (1168, 1152),
        (584, 576),
        (2320, 2304),
        (4112, 4096),
    ];
    for &(w, h) in sizes {
        let n = w * h;
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32 - 2.4e-4
        };
        let base: Vec<f32> = (0..n).map(|_| next() * 3.0).collect();
        let tgt0: Vec<f32> = (0..n).map(|_| next()).collect();
        let mut win = base.clone();
        let mut tgt = tgt0.clone();
        let mut tmp = Vec::new();
        let mut scratch = Vec::new();
        let mut t_ser = Vec::new();
        let mut t_par = Vec::new();
        for _ in 0..iters {
            win.copy_from_slice(&base);
            tgt.copy_from_slice(&tgt0);
            let t = std::time::Instant::now();
            zensim::__bench_stages::box_spread_merge_f32(
                &mut win,
                &mut tgt,
                w,
                h,
                5,
                &mut tmp,
                &mut scratch,
                false,
            );
            t_ser.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(tgt[0]);

            win.copy_from_slice(&base);
            tgt.copy_from_slice(&tgt0);
            let t = std::time::Instant::now();
            zensim::__bench_stages::box_spread_merge_f32(
                &mut win,
                &mut tgt,
                w,
                h,
                5,
                &mut tmp,
                &mut scratch,
                true,
            );
            t_par.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(tgt[0]);
        }
        println!(
            "SPREAD {w}x{h} (n={n}): serial {:.3} ms | parallel {:.3} ms | speedup {:.2}x",
            median(t_ser.clone()),
            median(t_par.clone()),
            median(t_ser) / median(t_par).max(1e-9)
        );
    }
}
