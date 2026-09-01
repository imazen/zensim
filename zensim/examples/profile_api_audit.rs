//! PRODUCT-API audit for a candidate bake: can a caller actually SHIP this model?
//!
//! `bake_verdict` scores a bake over STORED feature parquets — it never runs the
//! extractor, never runs the product entry points, and never sees the score
//! mapping a caller gets. This instrument closes that gap: it loads a bake the
//! way a product profile would (`ProfileParams::builder` → `ZensimProfile::
//! Custom`) and exercises the properties the metric's own contract states:
//!
//!   1. **identity** — `compute(ref, ref)` must be the top of the dial,
//!   2. **monotone in codec quality** — a q-ladder must not rank backwards,
//!   3. **bounded above / negative-capable below** (`zensim/CLAUDE.md`:
//!      "NEGATIVE zensim values MUST work"),
//!   4. **path agreement** — the buffered `compute` and the streaming-strip
//!      entry point must agree, since a codec loop uses whichever is cheaper.
//!
//! Usage:
//!   profile_api_audit <ref.png> <dist1.png> [dist2.png ...]   # --bake via env
//!   ZENSIM_AUDIT_BAKE=/path/to/bake.bin profile_api_audit ref.png q*.jpg.png
//!
//! With no `ZENSIM_AUDIT_BAKE` it audits the shipped `codec_target()` profile,
//! which is the control every candidate should be read against.

use std::env;
use zensim::{RgbSlice, Zensim, ZensimProfile, profile::ProfileParams};

fn load(path: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (img.pixels().map(|p| p.0).collect(), w, h)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <ref.png> <dist.png>...", args[0]);
        std::process::exit(2);
    }

    // A candidate bake is loaded exactly as a product profile would load it.
    // NOTE: `ProfileParams::builder().mlp()` takes a bare `fn() -> &'static [u8]`,
    // NOT a closure — a runtime-chosen bake therefore has to be parked in a
    // process-global `OnceLock` first. That is a real ergonomic cost for any
    // caller who wants to A/B a candidate model, and it is why this instrument
    // exists rather than a two-line snippet.
    static BAKE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    fn bake_bytes() -> &'static [u8] {
        BAKE.get().expect("bake set").as_slice()
    }

    let (profile, label) = match env::var("ZENSIM_AUDIT_BAKE") {
        Ok(p) => {
            let raw = std::fs::read(&p).unwrap_or_else(|e| panic!("read {p}: {e}"));
            let _ = BAKE.set(raw);
            let n_in = zenpredict::Model::from_bytes(bake_bytes())
                .map(|m| m.caller_input_width())
                .unwrap_or(372);
            // These four MUST match the shipped `PROFILE_B` literal
            // (`zensim/src/profile.rs`). `skip_score_mapping` and
            // `extrapolate_score` both default to FALSE in the builder, and
            // omitting them applies the legacy distance->score mapping ON TOP
            // of the bake's own output spline. That does not error: it
            // silently returns 0.0 for every distortion above the very worst,
            // i.e. a dead dial that looks like a catastrophically bad model.
            let params = ProfileParams::builder()
                .mlp(bake_bytes)
                .skip_score_mapping(true)
                .extrapolate_score(true)
                .extended_features(true)
                .compute_iw_features(n_in > 300)
                .build();
            let params: &'static ProfileParams = Box::leak(Box::new(params));
            eprintln!("# bake {p} — caller_input_width {n_in}");
            (
                ZensimProfile::Custom {
                    params,
                    name: "audit-candidate",
                },
                p,
            )
        }
        Err(_) => (ZensimProfile::codec_target(), "codec_target()".into()),
    };

    let z = Zensim::new(profile);
    let (rpx, w, h) = load(&args[1]);
    let rs = RgbSlice::new(&rpx, w, h);

    println!("# profile: {label}   ref: {} ({w}x{h})", args[1]);

    // (1) IDENTITY — the reference against itself.
    let ident = z.compute(&rs, &rs).expect("identity compute").score();
    println!("identity\t{ident:.6}");

    // (2)-(4) the ladder, in the order given (expected worst -> best).
    let mut prev = f64::NEG_INFINITY;
    let mut inversions = 0usize;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &args[2..] {
        let (dpx, dw, dh) = load(p);
        assert_eq!((dw, dh), (w, h), "{p}: size differs from reference");
        let ds = RgbSlice::new(&dpx, dw, dh);
        let buffered = z.compute(&rs, &ds).expect("buffered compute").score();
        // Path agreement: the streaming-strip entry point a loop may prefer.
        let streamed = z
            .compute_streaming_strips_default(&rs, &ds)
            .map(|r| r.score())
            .unwrap_or(f64::NAN);
        let dpath = (buffered - streamed).abs();
        if buffered < prev - 1e-9 {
            inversions += 1;
        }
        prev = buffered;
        lo = lo.min(buffered);
        hi = hi.max(buffered);
        println!(
            "{}\t{buffered:.6}\tstreaming={streamed:.6}\t|Δpath|={dpath:.3e}",
            p.rsplit('/').next().unwrap_or(p)
        );
    }
    println!("ladder_inversions\t{inversions}");
    println!("ladder_min\t{lo:.6}");
    println!("ladder_max\t{hi:.6}");
    println!("identity_is_max\t{}", ident >= hi - 1e-9);
    println!("negative_reachable\t{}", lo < 0.0);
}
