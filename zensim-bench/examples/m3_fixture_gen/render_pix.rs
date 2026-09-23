//! Pixel helpers for the E5A rendering-corruption generator.
//!
//! Buffer-level pixel ops that are not imaging algorithms: swizzles,
//! quantization, edge-fill translation, parity decimation, alpha-mask
//! synthesis, dithering, and the linear-light composite wrapper around
//! `zenblend`/`linear-srgb`. Resampling itself lives in `zenresize` calls in
//! `render_families.rs` — nothing here convolves.
use super::*;

/// Packed RGBA8 image.
pub(super) struct Rgba8 {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

/// Emitted output image — always composited opaque RGB by the time it wraps
/// a twin (RGBA items are composited over the declared backgrounds inside the
/// family code, so the scorer never sees alpha).
pub(super) enum OutImg {
    Rgb(Rgb8),
}

impl OutImg {
    pub fn as_rgb(&self) -> &Rgb8 {
        match self {
            OutImg::Rgb(i) => i,
        }
    }

    #[cfg(test)]
    pub fn dims(&self) -> (u32, u32) {
        match self {
            OutImg::Rgb(i) => (i.w, i.h),
        }
    }
}

/// One correct/broken rendering-implementation pair.
pub(super) struct Twin {
    pub family: &'static str,
    pub variant: &'static str,
    /// Declared severity rank inside the family (bigger = worse). The analysis
    /// re-derives the anchor from pixels; this is the design intent.
    pub severity: u32,
    pub params: serde_json::Value,
    pub correct: OutImg,
    pub broken: OutImg,
}

pub(super) fn rgb_to_rgba_opaque(img: &Rgb8) -> Rgba8 {
    let mut px = Vec::with_capacity(img.px.len() / 3 * 4);
    for p in img.px.as_chunks::<3>().0 {
        px.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    Rgba8 {
        w: img.w,
        h: img.h,
        px,
    }
}

/// Deterministic per-item seed: first 8 bytes (LE) of
/// sha256("e5a-render" | origin | family | variant | extra).
pub(super) fn item_seed(origin: &str, family: &str, variant: &str, extra: &str) -> u64 {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for part in ["e5a-render", origin, family, variant, extra] {
        h.update(part.as_bytes());
        h.update([0u8]);
    }
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().unwrap())
}

/// Small deterministic PRNG (mulberry64) for mask noise/jitter.
pub(super) struct Mulberry(u64);
impl Mulberry {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Inclusive integer range.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

// ---- linear-light conversions (linear-srgb owner) ----

/// sRGB u8 -> linear f32 (0..1), `linear-srgb` default (SIMD) path.
pub(super) fn srgb8_to_linear(px: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; px.len()];
    linear_srgb::default::srgb_u8_to_linear_slice(px, &mut out);
    out
}

/// linear f32 -> sRGB u8, `linear-srgb` default path.
pub(super) fn linear_to_srgb8(lin: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; lin.len()];
    linear_srgb::default::linear_to_srgb_u8_slice(lin, &mut out);
    out
}

/// sRGB u8 -> linear f32 via the `precise` (powf/f64) path — the alternate
/// correct implementation used for benign-drift pairs.
pub(super) fn srgb8_to_linear_precise(px: &[u8]) -> Vec<f32> {
    px.iter()
        .map(|&v| linear_srgb::precise::srgb_to_linear_f64(v as f64 / 255.0) as f32)
        .collect()
}

pub(super) fn linear_to_srgb8_precise(lin: &[f32]) -> Vec<u8> {
    lin.iter()
        .map(|&v| {
            (linear_srgb::precise::linear_to_srgb_f64(v as f64) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

// ---- composite ----

/// Composite straight-alpha RGBA8 over an opaque RGB8 background in linear
/// light (`zenblend` SrcOver on premultiplied f32 rows; `linear-srgb` owns
/// the transfer functions). This is the CORRECT implementation every alpha
/// twin's scored output goes through.
pub(super) fn composite_over(fg: &Rgba8, bg: &Rgb8) -> Res<Rgb8> {
    if (fg.w, fg.h) != (bg.w, bg.h) {
        return Err(format!(
            "composite dims differ: fg {}x{} bg {}x{}",
            fg.w, fg.h, bg.w, bg.h
        )
        .into());
    }
    // fg: sRGB u8 straight -> linear premultiplied f32 (linear-srgb owner).
    let mut fgf = vec![0f32; fg.px.len()];
    linear_srgb::default::srgb_u8_to_linear_premultiply_rgba_slice(&fg.px, &mut fgf);
    // bg: opaque straight -> premul is itself; alpha = 1.
    let mut bgf = vec![0f32; fg.px.len()];
    for (i, p) in bg.px.as_chunks::<3>().0.iter().enumerate() {
        bgf[4 * i] = linear_srgb::default::srgb_u8_to_linear(p[0]);
        bgf[4 * i + 1] = linear_srgb::default::srgb_u8_to_linear(p[1]);
        bgf[4 * i + 2] = linear_srgb::default::srgb_u8_to_linear(p[2]);
        bgf[4 * i + 3] = 1.0;
    }
    zenblend::blend_row(&mut fgf, &bgf, zenblend::BlendMode::SrcOver);
    // Drop alpha, encode RGB.
    let mut lin = Vec::with_capacity(fg.px.len() / 4 * 3);
    for p in fgf.as_chunks::<4>().0 {
        lin.extend_from_slice(&[p[0], p[1], p[2]]);
    }
    Ok(Rgb8 {
        w: fg.w,
        h: fg.h,
        px: linear_to_srgb8(&lin),
    })
}

/// Same composite as `composite_over` — the identical LINEAR-light SrcOver
/// operation — but routed through a u16 premultiplied intermediate instead of
/// f32. Fixed-point vs float rounding of the same op: the benign-drift twin.
pub(super) fn composite_over_u16(fg: &Rgba8, bg: &Rgb8) -> Res<Rgb8> {
    if (fg.w, fg.h) != (bg.w, bg.h) {
        return Err("composite dims differ".into());
    }
    let mut out = Vec::with_capacity(bg.px.len());
    for (fp, bp) in fg
        .px
        .as_chunks::<4>()
        .0
        .iter()
        .zip(bg.px.as_chunks::<3>().0)
    {
        let a16 = fp[3] as u32 * 257; // u8 alpha -> u16, exact
        let ia16 = 65535 - a16;
        let af = fp[3] as f64 / 255.0;
        for c in 0..3 {
            // premultiplied linear u16 channels.
            let fg16 = (linear_srgb::precise::srgb_to_linear_f64(fp[c] as f64 / 255.0)
                * af
                * 65535.0)
                .round() as u32;
            let bg16 = (linear_srgb::precise::srgb_to_linear_f64(bp[c] as f64 / 255.0)
                * 65535.0)
                .round() as u32;
            // SrcOver on premultiplied u16: fg + bg*(1-a), +0.5 round.
            let o16 = (fg16 * 65535 + bg16 * ia16 + 32767) / 65535;
            let s = linear_srgb::precise::linear_to_srgb_f64(o16.min(65535) as f64 / 65535.0);
            out.push((s * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(Rgb8 {
        w: fg.w,
        h: fg.h,
        px: out,
    })
}

/// BUGGY composite: foreground premultiplied twice (darkens semi-transparent
/// pixels — the "premultiplying twice" / "unpremultiplied twice" state bug).
pub(super) fn composite_premul_twice(fg: &Rgba8, bg: &Rgb8) -> Res<Rgb8> {
    if (fg.w, fg.h) != (bg.w, bg.h) {
        return Err("composite dims differ".into());
    }
    let mut fgf = vec![0f32; fg.px.len()];
    linear_srgb::default::srgb_u8_to_linear_premultiply_rgba_slice(&fg.px, &mut fgf);
    // Apply premultiply AGAIN: the second application multiplies rgb by a.
    for p in fgf.as_chunks_mut::<4>().0 {
        let a = p[3];
        p[0] *= a;
        p[1] *= a;
        p[2] *= a;
    }
    let mut bgf = vec![0f32; fg.px.len()];
    for (i, p) in bg.px.as_chunks::<3>().0.iter().enumerate() {
        bgf[4 * i] = linear_srgb::default::srgb_u8_to_linear(p[0]);
        bgf[4 * i + 1] = linear_srgb::default::srgb_u8_to_linear(p[1]);
        bgf[4 * i + 2] = linear_srgb::default::srgb_u8_to_linear(p[2]);
        bgf[4 * i + 3] = 1.0;
    }
    zenblend::blend_row(&mut fgf, &bgf, zenblend::BlendMode::SrcOver);
    let mut lin = Vec::with_capacity(fg.px.len() / 4 * 3);
    for p in fgf.as_chunks::<4>().0 {
        lin.extend_from_slice(&[p[0], p[1], p[2]]);
    }
    Ok(Rgb8 {
        w: fg.w,
        h: fg.h,
        px: linear_to_srgb8(&lin),
    })
}

/// BUGGY composite: straight-alpha data consumed as premultiplied — the
/// "forgot to premultiply" bug. out.rgb = fg.rgb + bg.rgb*(1-a): colour
/// bleeds at full strength into transparent regions (halo/fringe).
pub(super) fn composite_forgot_premul(fg: &Rgba8, bg: &Rgb8) -> Res<Rgb8> {
    if (fg.w, fg.h) != (bg.w, bg.h) {
        return Err("composite dims differ".into());
    }
    // fg stays STRAIGHT rgb in linear space (bug: never premultiplied).
    let mut fgf = vec![0f32; fg.px.len()];
    for (i, p) in fg.px.as_chunks::<4>().0.iter().enumerate() {
        fgf[4 * i] = linear_srgb::default::srgb_u8_to_linear(p[0]);
        fgf[4 * i + 1] = linear_srgb::default::srgb_u8_to_linear(p[1]);
        fgf[4 * i + 2] = linear_srgb::default::srgb_u8_to_linear(p[2]);
        fgf[4 * i + 3] = p[3] as f32 / 255.0;
    }
    let mut bgf = vec![0f32; fg.px.len()];
    for (i, p) in bg.px.as_chunks::<3>().0.iter().enumerate() {
        bgf[4 * i] = linear_srgb::default::srgb_u8_to_linear(p[0]);
        bgf[4 * i + 1] = linear_srgb::default::srgb_u8_to_linear(p[1]);
        bgf[4 * i + 2] = linear_srgb::default::srgb_u8_to_linear(p[2]);
        bgf[4 * i + 3] = 1.0;
    }
    zenblend::blend_row(&mut fgf, &bgf, zenblend::BlendMode::SrcOver);
    let mut lin = Vec::with_capacity(fg.px.len() / 4 * 3);
    for p in fgf.as_chunks::<4>().0 {
        lin.extend_from_slice(&[p[0], p[1], p[2]]);
    }
    Ok(Rgb8 {
        w: fg.w,
        h: fg.h,
        px: linear_to_srgb8(&lin),
    })
}

// ---- backgrounds ----

pub(super) fn bg_black(w: u32, h: u32) -> Rgb8 {
    Rgb8 {
        w,
        h,
        px: vec![0; (w * h * 3) as usize],
    }
}

pub(super) fn bg_white(w: u32, h: u32) -> Rgb8 {
    Rgb8 {
        w,
        h,
        px: vec![255; (w * h * 3) as usize],
    }
}

/// 8px checkerboard, sRGB 204/51 (declared in the prereg).
pub(super) fn bg_checker(w: u32, h: u32) -> Rgb8 {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = if ((x / 8) + (y / 8)) % 2 == 0 {
                204
            } else {
                51
            };
            px.extend_from_slice(&[v, v, v]);
        }
    }
    Rgb8 { w, h, px }
}

pub(super) const BACKGROUNDS: [(&str, fn(u32, u32) -> Rgb8); 3] = [
    ("black", bg_black),
    ("white", bg_white),
    ("checker", bg_checker),
];

// ---- alpha masks (deterministic per origin/variant seed) ----

/// Smooth radial vignette: opaque centre disc fading to transparent at the
/// corners — exercises premultiply correctness across the full alpha ramp.
pub(super) fn mask_vignette(w: u32, h: u32) -> Vec<u8> {
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let maxd = (cx * cx + cy * cy).sqrt();
    let mut m = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt() / maxd;
            // opaque out to ~0.55 of the corner distance, cosine falloff after.
            let a = if d < 0.55 {
                1.0
            } else {
                let t = ((d - 0.55) / 0.45).min(1.0);
                (1.0 + (t * std::f64::consts::PI).cos()) / 2.0
            };
            m.push((a * 255.0).round() as u8);
        }
    }
    m
}

/// Hard-edged shapes (filled circle + two rects) with a ~2px feathered rim —
/// sharp alpha edges where fringe bugs live. Positions are seed-jittered.
pub(super) fn mask_shapes(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut rng = Mulberry::new(seed);
    let mut m = vec![0u8; (w * h) as usize];
    let cx = (w as i64 / 4) + rng.below((w / 8).max(2) as u64) as i64;
    let cy = (h as i64 / 4) + rng.below((h / 8).max(2) as u64) as i64;
    let r = (w.min(h) / 4) as i64;
    let (rx0, ry0) = (
        w as i64 / 2 + rng.below((w / 6).max(2) as u64) as i64,
        h as i64 / 8 + rng.below((h / 8).max(2) as u64) as i64,
    );
    let (rw, rh) = (w as i64 / 3, h as i64 / 3);
    let (sx0, sy0) = (
        w as i64 / 8 + rng.below((w / 8).max(2) as u64) as i64,
        h as i64 * 5 / 8 + rng.below((h / 8).max(2) as u64) as i64,
    );
    let (sw, sh) = (w as i64 / 4, h as i64 / 5);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            // signed distance to each shape; negative = inside.
            let dc = (((x - cx).pow(2) + (y - cy).pow(2)) as f64).sqrt() - r as f64;
            let dr = [rx0 - x, x - (rx0 + rw), ry0 - y, y - (ry0 + rh)]
                .into_iter()
                .map(|v| v as f64)
                .fold(f64::MIN, f64::max);
            let ds = [sx0 - x, x - (sx0 + sw), sy0 - y, y - (sy0 + sh)]
                .into_iter()
                .map(|v| v as f64)
                .fold(f64::MIN, f64::max);
            let d = dc.min(dr).min(ds);
            let a = if d <= 0.0 {
                255.0
            } else if d < 2.0 {
                (2.0 - d) / 2.0 * 255.0 // ~2px linear feather
            } else {
                0.0
            };
            m[(y * w as i64 + x) as usize] = a.round() as u8;
        }
    }
    m
}

pub(super) fn rgba_from(img: &Rgb8, alpha: &[u8]) -> Rgba8 {
    let mut px = Vec::with_capacity(img.px.len() / 3 * 4);
    for (i, p) in img.px.as_chunks::<3>().0.iter().enumerate() {
        px.extend_from_slice(&[p[0], p[1], p[2], alpha[i]]);
    }
    Rgba8 {
        w: img.w,
        h: img.h,
        px,
    }
}

// ---- geometry ----

#[derive(Clone, Copy)]
pub(super) enum EdgeFill {
    Mirror,
    Wrap,
    Clamp,
}

fn edge_idx(i: i32, n: u32, mode: EdgeFill) -> u32 {
    let n = n as i32;
    match mode {
        EdgeFill::Clamp => i.clamp(0, n - 1) as u32,
        EdgeFill::Wrap => i.rem_euclid(n) as u32,
        EdgeFill::Mirror => {
            // reflect101 (edge pixel not repeated): -1 -> 1, n -> n-2
            if n <= 1 {
                return 0;
            }
            let period = 2 * (n - 1);
            let mut j = i.rem_euclid(period);
            if j >= n {
                j = period - j;
            }
            j as u32
        }
    }
}

/// Translate content by (dx, dy); vacated pixels filled by edge extension
/// with `mode` semantics.
pub(super) fn translate(img: &Rgb8, dx: i32, dy: i32, mode: EdgeFill) -> Rgb8 {
    let (w, h) = (img.w as i32, img.h as i32);
    let mut out = vec![0u8; img.px.len()];
    for y in 0..h {
        for x in 0..w {
            let sx = edge_idx(x - dx, img.w, mode);
            let sy = edge_idx(y - dy, img.h, mode);
            let si = (sy * img.w + sx) as usize * 3;
            let di = (y as u32 * img.w + x as u32) as usize * 3;
            out[di..di + 3].copy_from_slice(&img.px[si..si + 3]);
        }
    }
    Rgb8 {
        w: img.w,
        h: img.h,
        px: out,
    }
}

/// Axis-aligned integer crop (pure buffer op).
pub(super) fn crop(img: &Rgb8, x: u32, y: u32, w: u32, h: u32) -> Res<Rgb8> {
    if x + w > img.w || y + h > img.h || w == 0 || h == 0 {
        return Err(format!("crop {x},{y} {w}x{h} outside {}x{}", img.w, img.h).into());
    }
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for row in y..y + h {
        let s = (row * img.w + x) as usize * 3;
        px.extend_from_slice(&img.px[s..s + w as usize * 3]);
    }
    Ok(Rgb8 { w, h, px })
}

/// Every-other-pixel decimate at (xpar, ypar) parity on an even-dim image —
/// output w/2 x h/2.
pub(super) fn decimate_parity(img: &Rgb8, xpar: u32, ypar: u32) -> Res<Rgb8> {
    if img.w % 2 != 0 || img.h % 2 != 0 || xpar > 1 || ypar > 1 {
        return Err(format!("decimate parity on {}x{} bad parity", img.w, img.h).into());
    }
    let (ow, oh) = (img.w / 2, img.h / 2);
    let mut px = Vec::with_capacity((ow * oh * 3) as usize);
    for y in 0..oh {
        for x in 0..ow {
            let s = ((2 * y + ypar) * img.w + (2 * x + xpar)) as usize * 3;
            px.extend_from_slice(&img.px[s..s + 3]);
        }
    }
    Ok(Rgb8 { w: ow, h: oh, px })
}

/// Center square crop.
pub(super) fn center_square(img: &Rgb8) -> Rgb8 {
    let s = img.w.min(img.h);
    let x = (img.w - s) / 2;
    let y = (img.h - s) / 2;
    crop(img, x, y, s, s).expect("center square inside image")
}

// ---- channels / quantize ----

pub(super) fn swizzle_rb(img: &Rgb8) -> Rgb8 {
    let mut px = img.px.clone();
    for p in px.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    Rgb8 {
        w: img.w,
        h: img.h,
        px,
    }
}

/// Expand u8 -> u16 by *257 (exact, reversible).
pub(super) fn expand_u16(img: &Rgb8) -> Vec<u16> {
    img.px.iter().map(|&v| v as u16 * 257).collect()
}

/// u16 -> u8, correct rounding (round v/257 = (v*255 + 32767)/65535).
pub(super) fn quantize_u16_round(v: u16) -> u8 {
    ((v as u32 * 255 + 32767) / 65535) as u8
}

/// u16 -> u8, truncation bug (v >> 8 — loses up to ~0.5 LSB + bias).
pub(super) fn quantize_u16_trunc(v: u16) -> u8 {
    (v >> 8) as u8
}

/// Alternate CORRECT u16->u8 rounding (round-half-away vs the canonical
/// integer form) for the ±1-LSB benign pair.
pub(super) fn quantize_u16_round_away(v: u16) -> u8 {
    ((v as u32 + 128) / 257) as u8
}

/// 4x4 Bayer ordered-dither matrix entry, 0..15.
pub(super) fn bayer4(x: u32, y: u32) -> u32 {
    const M: [[u32; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    M[(y % 4) as usize][(x % 4) as usize]
}

/// Quantize u8 -> 5-bit value domain (0..31) with ordered dither, then
/// expand back to u8 (v*255/31, rounded). `dither` toggles the bug twin;
/// `phase` shifts the matrix for the benign pair.
pub(super) fn quantize5(img: &Rgb8, dither: bool, phase: u32) -> Rgb8 {
    let mut px = vec![0u8; img.px.len()];
    for y in 0..img.h {
        for x in 0..img.w {
            let i = (y * img.w + x) as usize * 3;
            for c in 0..3 {
                let v = img.px[i + c] as u32;
                let d = if dither {
                    bayer4(x + phase, y + phase) as i32 - 8 // -8..7
                } else {
                    0
                };
                // quantize to 31 levels (step = 255/31 ≈ 8.23)
                let scaled = v as i32 * 31 + d * 8; // dither amp ≈ 1 step
                let q = ((scaled + 127) / 255).clamp(0, 31);
                px[i + c] = ((q * 255 + 15) / 31) as u8;
            }
        }
    }
    Rgb8 {
        w: img.w,
        h: img.h,
        px,
    }
}

// ---- BT.601 full-range YCbCr (documented matrix; siting lives in the
// resample step, which is zenresize's) ----

pub(super) fn rgb_to_ycbcr601(img: &Rgb8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = (img.w * img.h) as usize;
    let (mut y, mut cb, mut cr) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for p in img.px.as_chunks::<3>().0 {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let yy = 0.299 * r + 0.587 * g + 0.114 * b;
        let cbb = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
        let crr = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
        y.push(yy.round().clamp(0.0, 255.0) as u8);
        cb.push(cbb.round().clamp(0.0, 255.0) as u8);
        cr.push(crr.round().clamp(0.0, 255.0) as u8);
    }
    (y, cb, cr)
}

pub(super) fn ycbcr601_to_rgb(w: u32, h: u32, y: &[u8], cb: &[u8], cr: &[u8]) -> Rgb8 {
    let mut px = Vec::with_capacity(y.len() * 3);
    for i in 0..y.len() {
        let (yy, cbb, crr) = (y[i] as f32, cb[i] as f32 - 128.0, cr[i] as f32 - 128.0);
        let r = yy + 1.402 * crr;
        let g = yy - 0.344_136 * cbb - 0.714_136 * crr;
        let b = yy + 1.772 * cbb;
        px.push(r.round().clamp(0.0, 255.0) as u8);
        px.push(g.round().clamp(0.0, 255.0) as u8);
        px.push(b.round().clamp(0.0, 255.0) as u8);
    }
    Rgb8 { w, h, px }
}

/// Translate a single-channel plane by (dx, dy) with mirror edge — used to
/// mis-site chroma relative to luma.
pub(super) fn translate_plane(plane: &[u8], w: u32, h: u32, dx: i32, dy: i32) -> Vec<u8> {
    let mut out = vec![0u8; plane.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = edge_idx(x - dx, w, EdgeFill::Mirror);
            let sy = edge_idx(y - dy, h, EdgeFill::Mirror);
            out[(y as u32 * w + x as u32) as usize] = plane[(sy * w + sx) as usize];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> Rgb8 {
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                px.extend_from_slice(&[
                    (x * 255 / w.max(1)) as u8,
                    (y * 255 / h.max(1)) as u8,
                    ((x + y) * 255 / (w + h).max(1)) as u8,
                ]);
            }
        }
        Rgb8 { w, h, px }
    }

    #[test]
    fn mirror_wrap_clamp_indices() {
        assert_eq!(edge_idx(-1, 8, EdgeFill::Mirror), 1);
        assert_eq!(edge_idx(8, 8, EdgeFill::Mirror), 6);
        assert_eq!(edge_idx(-1, 8, EdgeFill::Wrap), 7);
        assert_eq!(edge_idx(9, 8, EdgeFill::Wrap), 1);
        assert_eq!(edge_idx(-3, 8, EdgeFill::Clamp), 0);
        assert_eq!(edge_idx(11, 8, EdgeFill::Clamp), 7);
    }

    #[test]
    fn composite_matches_analytic_over() {
        // f64 reference: out = round_srgb( fg_lin*a + bg_lin*(1-a) )
        let fg = Rgba8 {
            w: 2,
            h: 1,
            px: vec![200, 50, 30, 128, 10, 220, 90, 64],
        };
        let bg = Rgb8 {
            w: 2,
            h: 1,
            px: vec![255, 255, 255, 0, 0, 0],
        };
        let out = composite_over(&fg, &bg).unwrap();
        for i in 0..2 {
            let a = fg.px[4 * i + 3] as f64 / 255.0;
            for c in 0..3 {
                let f = linear_srgb::precise::srgb_to_linear_f64(fg.px[4 * i + c] as f64 / 255.0);
                let b = linear_srgb::precise::srgb_to_linear_f64(bg.px[3 * i + c] as f64 / 255.0);
                let want = (linear_srgb::precise::linear_to_srgb_f64(f * a + b * (1.0 - a)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as i32;
                let got = out.px[3 * i + c] as i32;
                assert!(
                    (got - want).abs() <= 1,
                    "px{i} ch{c}: got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn buggy_composites_diverge_from_correct() {
        let fg = Rgba8 {
            w: 1,
            h: 1,
            px: vec![200, 50, 30, 128],
        };
        let bg = bg_white(1, 1);
        let ok = composite_over(&fg, &bg).unwrap();
        let dark = composite_premul_twice(&fg, &bg).unwrap();
        let halo = composite_forgot_premul(&fg, &bg).unwrap();
        assert_ne!(ok.px, dark.px);
        assert_ne!(ok.px, halo.px);
        // premul-twice darkens, forgot-premul adds halo
        assert!(dark.px[0] < ok.px[0]);
        assert!(halo.px[0] > ok.px[0] || halo.px[1] > ok.px[1] || halo.px[2] > ok.px[2]);
    }

    #[test]
    fn quantize_paths() {
        assert_eq!(quantize_u16_round(65535), 255);
        assert_eq!(quantize_u16_round(32768), 128);
        assert_eq!(quantize_u16_trunc(65535), 255);
        assert_eq!(quantize_u16_trunc(32768), 128);
        assert_eq!(quantize_u16_trunc(32896), 128); // trunc loses the +128
        // ±1 LSB bound between the two correct roundings
        for v in (0u32..65536).step_by(97) {
            let a = quantize_u16_round(v as u16) as i32;
            let b = quantize_u16_round_away(v as u16) as i32;
            assert!((a - b).abs() <= 1, "v={v}: {a} vs {b}");
        }
    }

    #[test]
    fn dither_and_no_dither_differ() {
        let img = ramp(31, 17); // odd dims control
        let a = quantize5(&img, true, 0);
        let b = quantize5(&img, false, 0);
        assert_ne!(a.px, b.px);
    }

    #[test]
    fn decimate_and_crop() {
        let img = ramp(8, 6);
        let d = decimate_parity(&img, 1, 1).unwrap();
        assert_eq!((d.w, d.h), (4, 3));
        assert_eq!(&d.px[0..3], &img.px[(1 * 8 + 1) * 3..(1 * 8 + 1) * 3 + 3]);
        let c = crop(&img, 2, 1, 4, 3).unwrap();
        assert_eq!(&c.px[0..3], &img.px[(1 * 8 + 2) * 3..(1 * 8 + 2) * 3 + 3]);
    }

    #[test]
    fn ycbcr_roundtrip_close() {
        let img = ramp(37, 23);
        let (y, cb, cr) = rgb_to_ycbcr601(&img);
        let back = ycbcr601_to_rgb(img.w, img.h, &y, &cb, &cr);
        let maxd = img
            .px
            .iter()
            .zip(&back.px)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(maxd <= 2, "max |roundtrip| = {maxd}");
    }
}
