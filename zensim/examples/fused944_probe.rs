//! Fused-944 design probe (campaign appendix N groundwork): per-component
//! costs of everything a folded-944 score+attribution compare touches, at
//! the jxl-loop shape (serial, reference precomputed once).
//!
//! Components timed (medians over N iters):
//!   1. v1 walk score-only            (`compute_with_ref`)
//!   2. v1 walk + Trained diffmap     (`compute_with_ref_and_diffmap`) — the
//!      call the loop pays today for the redistribution map
//!   3. folded-944 extraction         (`compute_folded720_features_streaming`
//!      with append+append2, reused scratch) — the loop's score pass
//!   4. standalone full density       (`compute_attribution_density_full`,
//!      944-wide gradient) — the map the 944 class cannot currently afford
//!   5. C3a fused v1 score+map        (`compute_with_ref_score_and_attribution`)
//!   6. C1 f64 basic density          (`compute_attribution_density`)
//!
//! ```sh
//! ZENSIM_ATTR_PERF=1 cargo run --release -p zensim \
//!   --features custom-profiles,feature-regime-v2 \
//!   --example fused944_probe -- [size] [iters]
//! ```

use zensim::profile::ProfileParams;
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Deterministic textured pair (the attribution tests' content family).
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let size: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(576);
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9);
    let (w, h) = (size, size);
    let (src, dst) = test_pair(w, h);
    let rs = RgbSlice::new(&src, w, h);
    let ds = RgbSlice::new(&dst, w, h);

    // The jxl loop's shape: serial, extended-features custom profile (the
    // walk config every bake mount uses; extended so the fused attribution
    // has all basic stats).
    let params = ProfileParams::builder()
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    let z = Zensim::new(ZensimProfile::Custom {
        params,
        name: "fused944-probe",
    })
    .with_parallel(false);

    let pre = z.precompute_reference(&rs).expect("precompute");
    let s156 = vec![-1.0f64; 156];
    let mut s944 = vec![0.0f64; 944];
    for (k, v) in s944.iter_mut().enumerate() {
        *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
    }
    let mut scratch = zensim::feature_v2::V2Scratch::new();
    let toggles944 = {
        let mut t = zensim::feature_v2::V2NewFeatureToggles::default();
        t.append_block = true;
        t.append2_block = true;
        t
    };

    // Warm-up every arm once.
    let _ = z.compute_with_ref(&pre, &ds).unwrap();
    let _ = z
        .compute_with_ref_and_diffmap(&pre, &ds, DiffmapWeighting::Trained)
        .unwrap();
    let _ = z
        .compute_folded720_features_streaming(&rs, &ds, toggles944, &mut scratch)
        .unwrap();
    let _ = z.compute_attribution_density_full(&rs, &ds, &s944).unwrap();
    let _ = z
        .compute_with_ref_score_and_attribution(&pre, &ds, &s156)
        .unwrap();
    let _ = z.compute_attribution_density(&rs, &ds, &s156).unwrap();

    let mut t1 = Vec::new();
    let mut t2 = Vec::new();
    let mut t3 = Vec::new();
    let mut t4 = Vec::new();
    let mut t5 = Vec::new();
    let mut t6 = Vec::new();
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let r = z.compute_with_ref(&pre, &ds).unwrap();
        t1.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(r.score());

        let t = std::time::Instant::now();
        let r = z
            .compute_with_ref_and_diffmap(&pre, &ds, DiffmapWeighting::Trained)
            .unwrap();
        t2.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(r.diffmap()[0]);

        let t = std::time::Instant::now();
        let v2 = z
            .compute_folded720_features_streaming(&rs, &ds, toggles944, &mut scratch)
            .unwrap();
        t3.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(v2.features()[943]);

        let t = std::time::Instant::now();
        let a = z.compute_attribution_density_full(&rs, &ds, &s944).unwrap();
        t4.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(a.query_rect(0, 0, 32, 32));

        let t = std::time::Instant::now();
        let (r, a) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s156)
            .unwrap();
        t5.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((r.score(), a.query_rect(0, 0, 32, 32)));

        let t = std::time::Instant::now();
        let a = z.compute_attribution_density(&rs, &ds, &s156).unwrap();
        t6.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(a.query_rect(0, 0, 32, 32));
    }

    println!(
        "F944PROBE {w}x{h} serial (iters {iters}) medians:\n\
         \x20 1. v1 score-only                 {:.1} ms\n\
         \x20 2. v1 + Trained diffmap          {:.1} ms   <- loop pays today (map+discarded score)\n\
         \x20 3. folded-944 extraction         {:.1} ms   <- loop pays today (score)\n\
         \x20 4. standalone full density (944) {:.1} ms   <- the unaffordable map\n\
         \x20 5. C3a fused v1 score+map        {:.1} ms\n\
         \x20 6. C1 f64 basic density          {:.1} ms\n\
         \x20 loop today (2+3):                {:.1} ms\n\
         \x20 naive fused floor (3+4):         {:.1} ms",
        median_ms(t1.clone()),
        median_ms(t2.clone()),
        median_ms(t3.clone()),
        median_ms(t4.clone()),
        median_ms(t5.clone()),
        median_ms(t6),
        median_ms(t2) + median_ms(t3.clone()),
        median_ms(t3) + median_ms(t4),
    );
}
