//! zenbench harness for zensim wall-time perf experiments.
//!
//! Reports min, mean, median per benchmark — the user asked for fastest run
//! since the dev box is loaded with other work.
//!
//! Build the binary multiple times with different feature flags to measure
//! each optimization in isolation, then ping-pong via separate binaries.
//!
//! Run with: cargo run --release -p zensim-bench --example zwe_zenbench

use zensim::{RgbSlice, Zensim, ZensimProfile};

fn make_test_images(width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = width * height;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = ((i % width) * 255 / width) as u8;
            let y = ((i / width) * 255 / height) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(8), g.saturating_add(4), b])
        .collect();
    (src, dst)
}

fn main() {
    // Pre-build everything outside the closure so &'static refs stay alive
    // for the entire suite.
    struct Prep {
        label: &'static str,
        w: usize,
        h: usize,
        z_mt: &'static Zensim,
        pre_mt: &'static zensim::PrecomputedReference,
        z_st: &'static Zensim,
        pre_st: &'static zensim::PrecomputedReference,
        dst: &'static [[u8; 3]],
    }
    let preps: Vec<Prep> = [
        ("256x256", 256, 256),
        ("512x512", 512, 512),
        ("1280x720", 1280, 720),
        ("1920x1080", 1920, 1080),
    ]
    .into_iter()
    .map(|(label, w, h)| {
        let (src, dst) = make_test_images(w, h);
        let z_mt = Box::leak(Box::new(
            Zensim::new(ZensimProfile::latest()).with_parallel(true),
        ));
        let z_st = Box::leak(Box::new(
            Zensim::new(ZensimProfile::latest()).with_parallel(false),
        ));
        let s_view = RgbSlice::new(&src, w, h);
        let pre_mt = Box::leak(Box::new(z_mt.precompute_reference(&s_view).unwrap()));
        let pre_st = Box::leak(Box::new(z_st.precompute_reference(&s_view).unwrap()));
        let dst_static: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
        Prep {
            label,
            w,
            h,
            z_mt,
            pre_mt,
            z_st,
            pre_st,
            dst: dst_static,
        }
    })
    .collect();

    let preps: &'static [Prep] = Box::leak(preps.into_boxed_slice());
    let result = zenbench::run(|suite| {
        for p in preps {
            let label = p.label;
            let w = p.w;
            let h = p.h;
            let z_mt: &'static Zensim = p.z_mt;
            let pre_mt: &'static zensim::PrecomputedReference = p.pre_mt;
            let z_st: &'static Zensim = p.z_st;
            let pre_st: &'static zensim::PrecomputedReference = p.pre_st;
            let dst_static = p.dst;
            suite.compare(format!("compute_with_ref_{label}"), |group| {
                // Heavy busy-system mode: take many rounds, report min.
                group.config().max_rounds(400);

                group.bench("multithread", move |b| {
                    b.iter(move || {
                        let d = RgbSlice::new(dst_static, w, h);
                        let r = z_mt.compute_with_ref(pre_mt, &d).unwrap();
                        zenbench::black_box(r);
                    })
                });

                group.bench("singlethread", move |b| {
                    b.iter(move || {
                        let d = RgbSlice::new(dst_static, w, h);
                        let r = z_st.compute_with_ref(pre_st, &d).unwrap();
                        zenbench::black_box(r);
                    })
                });
            });
        }
    });

    let _ = result;
}
