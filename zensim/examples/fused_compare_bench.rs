//! C3a perf harness: score-only vs fold-diffmap vs FUSED score+attribution
//! on one (ref, dist) pair with the reference precomputed once (the codec
//! loop shape). Reports medians over N iterations.
//!
//! ```sh
//! cargo run --release -p zensim --features custom-profiles \
//!   --example fused_compare_bench -- <ref> <dist> [iters]
//! ```

use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: fused_compare_bench <ref> <dist> [iters]");
        std::process::exit(2);
    }
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let r = image::open(&args[0]).expect("open ref").to_rgb8();
    let d = image::open(&args[1]).expect("open dist").to_rgb8();
    let (w, h) = (r.width() as usize, r.height() as usize);
    assert_eq!((d.width() as usize, d.height() as usize), (w, h));
    let rpx: Vec<[u8; 3]> = r.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dpx: Vec<[u8; 3]> = d.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let rs = RgbSlice::new(&rpx, w, h);
    let ds = RgbSlice::new(&dpx, w, h);

    let z = Zensim::new(ZensimProfile::codec_target());
    let pre = z.precompute_reference(&rs).expect("precompute");
    let s = vec![-1.0f64; 156];

    // Warm-up (page-faults, rayon pool, dispatch).
    let _ = z.compute_with_ref(&pre, &ds).unwrap();
    let _ = z
        .compute_with_ref_and_diffmap(&pre, &ds, DiffmapWeighting::Trained)
        .unwrap();
    let _ = z
        .compute_with_ref_score_and_attribution(&pre, &ds, &s)
        .unwrap();

    // #70 stale arm setup: prime the session during warm-up so every
    // measured call below is the single-pass path (the codec-loop shape).
    let mut sess = zensim::AttributionSession::new();
    let _ = z
        .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sess)
        .unwrap();

    let mut t_scalar = Vec::new();
    let mut t_fold = Vec::new();
    let mut t_fused = Vec::new();
    let mut t_stale = Vec::new();
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let r1 = z.compute_with_ref(&pre, &ds).unwrap();
        t_scalar.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(r1.score());

        let t = std::time::Instant::now();
        let r2 = z
            .compute_with_ref_and_diffmap(&pre, &ds, DiffmapWeighting::Trained)
            .unwrap();
        t_fold.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(r2.diffmap()[0]);

        let t = std::time::Instant::now();
        let (r3, a3) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        t_fused.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((r3.score(), a3.query_rect(0, 0, 32, 32)));

        // #70 stale-scalar single-pass arm (session primed in warm-up, so
        // every measured call is the single-pass path — the codec-loop
        // shape). Interleaved with the other arms per iteration so load
        // shifts hit all arms equally.
        let t = std::time::Instant::now();
        let (r4, a4) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sess)
            .unwrap();
        t_stale.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((r4.score(), a4.query_rect(0, 0, 32, 32)));
    }

    let (ms_s, ms_fo, ms_fu, ms_st) = (
        median_ms(t_scalar),
        median_ms(t_fold),
        median_ms(t_fused),
        median_ms(t_stale),
    );
    let marg_attr = ms_fu - ms_s;
    let marg_fold = ms_fo - ms_s;
    let marg_stale = ms_st - ms_s;
    println!(
        "FUSEDPERF {w}x{h} (iters {iters}): scalar-only {ms_s:.1} ms | fold {ms_fo:.1} ms | fused(score+map) {ms_fu:.1} ms | marginal map {marg_attr:.1} ms | marginal fold {marg_fold:.1} ms | ratio {:.2}x (target <=1.10x)",
        marg_attr / marg_fold.max(1e-9)
    );
    println!(
        "STALEPERF {w}x{h} (iters {iters}): stale(score+map) {ms_st:.1} ms | marginal stale map {marg_stale:.1} ms | stale ratio {:.2}x vs fold marginal {marg_fold:.1} ms (target <=1.10x)",
        marg_stale / marg_fold.max(1e-9)
    );
}
