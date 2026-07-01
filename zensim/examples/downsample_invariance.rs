//! Downsample-invariance prototype.
//!
//! Criterion (user, 2026-06-06): take a (ref, dist) pair, score it, then
//! downsample BOTH and rescore — the score must NOT fluctuate with size
//! (≤ ~2pt drift is fine). Must score all the way to 1×1 with NO errors,
//! bake-agnostic, simple.
//!
//! Candidate under test: **mirror (reflect) pad the image up to the 4-scale
//! minimum (64px)** before scoring. The bake always sees 4 genuinely-computed
//! scales (no synthetic feature fill → works for any bake); a 1×1 reflects to
//! a constant 64×64 so it scores with no special fallback.
//!
//! Run: cargo run --release --example downsample_invariance -p zensim

use zensim::{RgbSlice, Zensim, ZensimProfile};

const MIN_DIM: usize = 64; // 4 scales × min coarsest 8px = 8·2^3

/// Deterministic multi-scale synthetic RGB at n×n (content at every scale).
fn make_ref(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n * n * 3];
    for y in 0..n {
        for x in 0..n {
            let fx = x as f32 / n.max(1) as f32;
            let fy = y as f32 / n.max(1) as f32;
            let base = 0.5
                + 0.25 * (fx + fy)
                + 0.15 * (fx * core::f32::consts::TAU).sin()
                + 0.10 * (fy * 18.8496).cos()
                + 0.08 * ((fx + fy) * 50.265).sin();
            let h = ((x.wrapping_mul(2654435761)) ^ (y.wrapping_mul(40503))) as f32;
            let tex = ((h % 17.0) / 17.0 - 0.5) * 0.12;
            let i = (y * n + x) * 3;
            v[i] = ((base + tex) * 255.0).clamp(0.0, 255.0) as u8;
            v[i + 1] = ((base * 0.9 + 0.05 + tex) * 255.0).clamp(0.0, 255.0) as u8;
            v[i + 2] = ((base * 1.1 - 0.05 - tex) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    v
}

/// Scale-stable distortion: contrast reduction toward mid-gray.
fn distort(src: &[u8], factor: f32) -> Vec<u8> {
    src.iter()
        .map(|&c| (128.0 + (c as f32 - 128.0) * factor).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Area-average downscale of an RGB buffer from `src_n×src_n` to `n×n` (n ≤ src_n).
fn area_resize(src: &[u8], src_n: usize, n: usize) -> Vec<u8> {
    if n == src_n {
        return src.to_vec();
    }
    let mut out = vec![0u8; n * n * 3];
    for oy in 0..n {
        let y0 = oy * src_n / n;
        let y1 = ((oy + 1) * src_n / n).max(y0 + 1);
        for ox in 0..n {
            let x0 = ox * src_n / n;
            let x1 = ((ox + 1) * src_n / n).max(x0 + 1);
            for c in 0..3 {
                let mut s = 0u32;
                let mut cnt = 0u32;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        s += src[(sy * src_n + sx) * 3 + c] as u32;
                        cnt += 1;
                    }
                }
                out[(oy * n + ox) * 3 + c] = (s / cnt.max(1)) as u8;
            }
        }
    }
    out
}

/// reflect-101 index map (no edge repeat); n==1 → 0.
fn refl(i: i32, n: i32) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = ((i % period) + period) % period;
    if k >= n {
        k = period - k;
    }
    k as usize
}

/// Reflect-pad an `w×h` RGB image up to `W×H` (W≥w, H≥h), top-left aligned.
fn reflect_pad(src: &[u8], w: usize, h: usize, big_w: usize, big_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; big_w * big_h * 3];
    for y in 0..big_h {
        let sy = refl(y as i32, h as i32);
        for x in 0..big_w {
            let sx = refl(x as i32, w as i32);
            for c in 0..3 {
                out[(y * big_w + x) * 3 + c] = src[(sy * w + sx) * 3 + c];
            }
        }
    }
    out
}

/// Tile (wrap) pad an `w×h` RGB image up to `W×H` by repetition.
fn tile_pad(src: &[u8], w: usize, h: usize, big_w: usize, big_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; big_w * big_h * 3];
    for y in 0..big_h {
        let sy = y % h;
        for x in 0..big_w {
            let sx = x % w;
            for c in 0..3 {
                out[(y * big_w + x) * 3 + c] = src[(sy * w + sx) * 3 + c];
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Pad {
    Raw,
    Mirror,
    Tile,
}

fn pad(strategy: Pad, src: &[u8], n: usize) -> (Vec<u8>, usize) {
    if strategy == Pad::Raw || n >= MIN_DIM {
        return (src.to_vec(), n);
    }
    let big = MIN_DIM;
    let out = match strategy {
        Pad::Mirror => reflect_pad(src, n, n, big, big),
        Pad::Tile => tile_pad(src, n, n, big, big),
        Pad::Raw => unreachable!(),
    };
    (out, big)
}

fn to_px(flat: &[u8]) -> Vec<[u8; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn score(z: &Zensim, r: &[u8], d: &[u8], n: usize, strategy: Pad) -> Result<f64, String> {
    let (rr, w) = pad(strategy, r, n);
    let (dd, _) = pad(strategy, d, n);
    let rp = to_px(&rr);
    let dp = to_px(&dd);
    let s = RgbSlice::new(&rp, w, w);
    let ds = RgbSlice::new(&dp, w, w);
    z.compute(&s, &ds)
        .map(|res| res.score())
        .map_err(|e| format!("{e:?}"))
}

fn fluct(v: &[(usize, f64)]) -> (f64, f64, f64) {
    let (mn, mx) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &(_, s)| (a.min(s), b.max(s)));
    (mx - mn, mn, mx)
}

fn main() {
    let base = 512usize;
    let mut sizes: Vec<usize> = (1..=96).collect();
    for s in [128, 192, 256, 384, 512] {
        sizes.push(s);
    }

    // Test patterns: (name, ref@base, dist@base). Solid pairs are constant
    // colors that MUST score identically at every size (hard invariant).
    let textured_ref = make_ref(base);
    let textured_dist = distort(&textured_ref, 0.85);
    let solid_ref = vec_solid(base, [100, 120, 140]);
    let solid_dist = vec_solid(base, [112, 116, 150]); // constant per-channel delta
    let patterns: [(&str, &[u8], &[u8]); 2] = [
        ("textured (15% contrast)", &textured_ref, &textured_dist),
        ("SOLID (Δ=[12,-4,10])", &solid_ref, &solid_dist),
    ];

    for (pname, pref, pdist) in patterns {
        println!("######## pattern: {pname} ########");
        for (mname, profile) in [("A (MLP bake)", ZensimProfile::A)] {
            let z = Zensim::new(profile);
            for strat in [Pad::Raw, Pad::Mirror, Pad::Tile] {
                let label = match strat {
                    Pad::Raw => "raw   ",
                    Pad::Mirror => "mirror",
                    Pad::Tile => "tile  ",
                };
                let mut ok: Vec<(usize, f64)> = Vec::new();
                let mut errs = 0usize;
                for &n in &sizes {
                    let r = area_resize(pref, base, n);
                    let d = area_resize(pdist, base, n);
                    match score(&z, &r, &d, n, strat) {
                        Ok(s) => ok.push((n, s)),
                        Err(_) => errs += 1,
                    }
                }
                let (f, mn, mx) = fluct(&ok);
                // small-regime (8px+) fluctuation: the part we care about
                let small: Vec<(usize, f64)> =
                    ok.iter().cloned().filter(|(n, _)| *n >= 8).collect();
                let (f8, _, _) = fluct(&small);
                print!(
                    "  {mname:22} {label}: {:>3} ok, {errs} err; fluct(all) {f:6.3} fluct(≥8px) {f8:6.3} [{mn:.2}..{mx:.2}]",
                    ok.len()
                );
                if pname.starts_with("SOLID") {
                    // solid invariant: every size must score identically
                    print!("  | solid Δrange = {f:.4} (must be ~0)");
                }
                println!();
            }
        }
        println!();
    }
}

fn vec_solid(n: usize, c: [u8; 3]) -> Vec<u8> {
    let mut v = vec![0u8; n * n * 3];
    for px in v.chunks_exact_mut(3) {
        px.copy_from_slice(&c);
    }
    v
}
