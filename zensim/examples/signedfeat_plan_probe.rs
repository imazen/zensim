//! What does a family-only request compute? Times `research::extract` for the SIGNEDFEAT slots requested
//! (a) at the full identity layout (what `extract_features_372col --restore-cuts` does), (b) as a dense
//! layout, (c) as `everything()`, on one synthetic 1024² pair, single thread. Reports the plan's emitted slot count.
//! `ZENSIM_FORMULA_REV=4 cargo run --release -p zensim --features training --example signedfeat_plan_probe`
use std::time::Instant;
use zensim::RgbSlice;
use zensim::feature_set_id::ComputeToken;
use zensim::research::{self, Request, family_slots};

fn main() {
    let n = 1024usize;
    let mut s = 0x1234_5678u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 24) as u8
    };
    let src: Vec<[u8; 3]> = (0..n * n)
        .map(|i| {
            let v = ((i % n) * 3 / 4 + (i / n) / 3) as u8;
            [v, v.wrapping_add(rnd() % 20), 255 - v / 2]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|p| [p[0].saturating_add(rnd() % 6), p[1] / 2 * 2, p[2]])
        .collect();
    let (a, b) = (RgbSlice::new(&src, n, n), RgbSlice::new(&dst, n, n));
    let want = family_slots(ComputeToken::Texgain).union(&family_slots(ComputeToken::Satsign));
    let w = research::full_width();
    let reqs = [
        (
            "identity layout (extractor)",
            Request::for_slots(want.clone(), w),
        ),
        ("dense layout", Request::for_slots(want.clone(), w).dense()),
        ("everything", Request::everything()),
    ];
    for (name, req) in reqs {
        let t = Instant::now();
        match research::extract(&req, &a, &b) {
            Ok(e) => println!(
                "{name}: {:.1} ms, values {}, emitted slots {}, id {}",
                t.elapsed().as_secs_f64() * 1e3,
                e.values().len(),
                e.emitted().len(),
                e.feature_set_id()
                    .map(|i| i.to_string())
                    .unwrap_or_default()
            ),
            Err(e) => println!("{name}: ERR {e}"),
        }
    }
}
