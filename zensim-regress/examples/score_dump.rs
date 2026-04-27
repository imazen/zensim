//! Dump zensim scores + raw distances + diffmap stats for a fixed set of
//! synthetic image pairs. Used to A/B-compare metric output across commits.

use zensim::{DiffmapOptions, DiffmapWeighting, RgbaSlice, Zensim, ZensimProfile};

type ImageGen = fn(u32, u32) -> Vec<[u8; 4]>;

fn gradient(w: u32, h: u32, seed: u32) -> Vec<[u8; 4]> {
    let mut s = seed;
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let r = ((x * 251 + (s.wrapping_mul(7) >> 8)) % 256) as u8;
            let g = ((y * 173 + (s.wrapping_mul(13) >> 8)) % 256) as u8;
            let b = (((x ^ y) * 47 + (s.wrapping_mul(19) >> 8)) % 256) as u8;
            out.push([r, g, b, 255]);
        }
    }
    out
}

fn checker(w: u32, h: u32, sz: u32) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / sz) + (y / sz)).is_multiple_of(2);
            let v = if on { 200 } else { 50 };
            out.push([v, v, v, 255]);
        }
    }
    out
}

fn jpeg_like(w: u32, h: u32, q: u8) -> Vec<[u8; 4]> {
    // Simulate JPEG-style block quantization: round each 8x8 block's mean.
    let mut out = vec![[0u8; 4]; (w * h) as usize];
    for by in 0..(h / 8) {
        for bx in 0..(w / 8) {
            let mut sums = [0u32; 3];
            for y in 0..8 {
                for x in 0..8 {
                    let xx = bx * 8 + x;
                    let yy = by * 8 + y;
                    let mut s = (xx ^ yy).wrapping_mul(0x9E37);
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    sums[0] += (xx * 251 + (s.wrapping_mul(7) >> 8)) % 256;
                    sums[1] += (yy * 173 + (s.wrapping_mul(13) >> 8)) % 256;
                    sums[2] += ((xx ^ yy) * 47 + (s.wrapping_mul(19) >> 8)) % 256;
                }
            }
            let mean = [sums[0] / 64, sums[1] / 64, sums[2] / 64];
            let step = (255 - q as u32).max(1);
            let q_mean = [
                (mean[0] / step * step) as u8,
                (mean[1] / step * step) as u8,
                (mean[2] / step * step) as u8,
            ];
            for y in 0..8 {
                for x in 0..8 {
                    let i = ((by * 8 + y) * w + (bx * 8 + x)) as usize;
                    out[i] = [q_mean[0], q_mean[1], q_mean[2], 255];
                }
            }
        }
    }
    out
}

fn main() {
    let z = Zensim::new(ZensimProfile::latest());

    let cases: &[(&str, u32, u32, ImageGen, ImageGen)] = &[
        (
            "identical_512",
            512,
            512,
            |w, h| gradient(w, h, 1),
            |w, h| gradient(w, h, 1),
        ),
        (
            "noise_512",
            512,
            512,
            |w, h| gradient(w, h, 1),
            |w, h| gradient(w, h, 7),
        ),
        (
            "checker_512",
            512,
            512,
            |w, h| checker(w, h, 8),
            |w, h| checker(w, h, 9),
        ),
        (
            "jpeg_q90",
            1024,
            768,
            |w, h| jpeg_like(w, h, 250),
            |w, h| jpeg_like(w, h, 200),
        ),
        (
            "jpeg_q50",
            1024,
            768,
            |w, h| jpeg_like(w, h, 250),
            |w, h| jpeg_like(w, h, 100),
        ),
        (
            "identical_1080",
            1920,
            1080,
            |w, h| gradient(w, h, 42),
            |w, h| gradient(w, h, 42),
        ),
    ];

    println!("# zensim score dump (commit-stable A/B comparison fixture)");
    println!("# format: name,score,raw_distance,diffmap_max,diffmap_mean,diffmap_sum");

    for &(name, w, h, src_fn, dst_fn) in cases {
        let src = src_fn(w, h);
        let dst = dst_fn(w, h);
        let s = RgbaSlice::new(&src, w as usize, h as usize);
        let d = RgbaSlice::new(&dst, w as usize, h as usize);

        let result = z.compute(&s, &d).unwrap();
        let opts = DiffmapOptions {
            weighting: DiffmapWeighting::Balanced,
            masking_strength: None,
            sqrt: false,
            include_edge_mse: false,
            include_hf: false,
        };
        let dm = z.compute_with_diffmap(&s, &d, opts).unwrap();
        let dm_data = dm.diffmap();
        let dm_max = dm_data.iter().copied().fold(0.0f32, f32::max);
        let dm_sum: f64 = dm_data.iter().map(|&v| v as f64).sum();
        let dm_mean = dm_sum / dm_data.len() as f64;

        println!(
            "{name},{:.10},{:.12e},{:.10e},{:.10e},{:.10e}",
            result.score(),
            result.raw_distance(),
            dm_max,
            dm_mean,
            dm_sum,
        );
    }
}
