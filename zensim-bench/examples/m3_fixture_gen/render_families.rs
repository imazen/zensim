//! E5A rendering-corruption families (prereg §3): each family pairs a CORRECT
//! implementation with BROKEN variants of the same rendering operation.
//! Resample/composite/convert work goes through the imazen owners
//! (zenresize, zenpixels, zenpixels-convert, zenblend, linear-srgb); per-pixel
//! buffer ops live in `render_pix.rs`.
use super::render_pix::*;
use super::*;
use zenpixels::{
    ColorPrimaries, Orientation, PixelDescriptor as ZPD, PixelSlice, TransferFunction,
};
use zenpixels_convert::RowConverter;
use zenpixels_convert::orient::apply_orientation;
use zenresize::{Filter, ResizeConfig, Resizer, StreamingResize};

const MITCHELL: Filter = Filter::Mitchell;
const LANCZOS: Filter = Filter::Lanczos;

/// Opaque-RGBA resize of an Rgb8 source. `linear` selects the pipeline:
/// true = linear-light (correct), false = gamma-encoded sRGB (the bug).
fn resize_rgb8(src: &Rgb8, out_w: u32, out_h: u32, filter: Filter, linear: bool) -> Res<Rgb8> {
    if out_w == 0 || out_h == 0 {
        return Err(format!("bad resize {}x{} -> {out_w}x{out_h}", src.w, src.h).into());
    }
    let rgba = rgb_to_rgba_opaque(src);
    let mut b = ResizeConfig::builder(src.w, src.h, out_w, out_h)
        .filter(filter)
        .format(ZPD::RGBA8_SRGB);
    b = if linear { b.linear() } else { b.srgb() };
    let out = Resizer::new(&b.build()).resize(&rgba.px);
    let mut px = Vec::with_capacity((out_w * out_h * 3) as usize);
    for p in out.as_chunks::<4>().0 {
        px.extend_from_slice(&[p[0], p[1], p[2]]);
    }
    Ok(Rgb8 {
        w: out_w,
        h: out_h,
        px,
    })
}

/// Resize an Rgba8 through a chosen descriptor — `RGBA8_SRGB` premultiplies
/// (correct); `RGBX8_SRGB` treats all four channels independently (the
/// no-premultiply bug).
fn resize_rgba(
    src: &Rgba8,
    out_w: u32,
    out_h: u32,
    filter: Filter,
    desc: ZPD,
    linear: bool,
) -> Res<Rgba8> {
    if out_w == 0 || out_h == 0 {
        return Err("bad rgba resize".into());
    }
    let mut b = ResizeConfig::builder(src.w, src.h, out_w, out_h)
        .filter(filter)
        .format(desc);
    b = if linear { b.linear() } else { b.srgb() };
    let out = Resizer::new(&b.build()).resize(&src.px);
    if out.len() != (out_w * out_h * 4) as usize {
        return Err("rgba resize output size mismatch".into());
    }
    Ok(Rgba8 {
        w: out_w,
        h: out_h,
        px: out,
    })
}

fn dims_div(w: u32, h: u32, r: u32) -> (u32, u32) {
    ((w + r - 1) / r, (h + r - 1) / r)
}

/// Composite an RGBA result over every declared background into one RGB twin
/// per background — the prereg alpha protocol (§3, "Alpha protocol").
fn alpha_twins(
    family: &'static str,
    stem: &str,
    severity: u32,
    params: serde_json::Value,
    correct: &Rgba8,
    broken: &Rgba8,
    out: &mut Vec<Twin>,
) -> Res<()> {
    for (bg_name, bg_fn) in BACKGROUNDS {
        let bg = bg_fn(correct.w, correct.h);
        let c = composite_over(correct, &bg)?;
        let b = composite_over(broken, &bg)?;
        out.push(Twin {
            family,
            variant: match (stem, bg_name) {
                ("nopremul_r2_vignette", "black") => "nopremul_r2_vignette_black",
                ("nopremul_r2_vignette", "white") => "nopremul_r2_vignette_white",
                ("nopremul_r2_vignette", _) => "nopremul_r2_vignette_checker",
                ("nopremul_r4_vignette", "black") => "nopremul_r4_vignette_black",
                ("nopremul_r4_vignette", "white") => "nopremul_r4_vignette_white",
                ("nopremul_r4_vignette", _) => "nopremul_r4_vignette_checker",
                ("nopremul_r2_shapes", "black") => "nopremul_r2_shapes_black",
                ("nopremul_r2_shapes", "white") => "nopremul_r2_shapes_white",
                ("nopremul_r2_shapes", _) => "nopremul_r2_shapes_checker",
                ("nopremul_r4_shapes", "black") => "nopremul_r4_shapes_black",
                ("nopremul_r4_shapes", "white") => "nopremul_r4_shapes_white",
                ("nopremul_r4_shapes", _) => "nopremul_r4_shapes_checker",
                _ => "alpha_item",
            },
            severity,
            params: {
                let mut p = params.clone();
                p["bg"] = serde_json::json!(bg_name);
                p
            },
            correct: OutImg::Rgb(c),
            broken: OutImg::Rgb(b),
        });
    }
    Ok(())
}

// ===========================================================================
// F01 gamma_downsample — downscale in gamma-encoded sRGB instead of linear.
// ===========================================================================
pub(super) fn f01_gamma_downsample(src: &Rgb8) -> Res<Vec<Twin>> {
    let mut out = Vec::new();
    for (r, sev) in [(2u32, 1u32), (3, 2), (4, 3), (8, 4)] {
        for k in [MITCHELL, LANCZOS] {
            let (ow, oh) = dims_div(src.w, src.h, r);
            let correct = resize_rgb8(src, ow, oh, k, true)?;
            let broken = resize_rgb8(src, ow, oh, k, false)?;
            out.push(Twin {
                family: "gamma_downsample",
                variant: match (r, k) {
                    (2, Filter::Mitchell) => "srgb_space_r2_mitchell",
                    (3, Filter::Mitchell) => "srgb_space_r3_mitchell",
                    (4, Filter::Mitchell) => "srgb_space_r4_mitchell",
                    (8, Filter::Mitchell) => "srgb_space_r8_mitchell",
                    (2, _) => "srgb_space_r2_lanczos",
                    (3, _) => "srgb_space_r3_lanczos",
                    (4, _) => "srgb_space_r4_lanczos",
                    (_, _) => "srgb_space_r8_lanczos",
                },
                severity: sev,
                params: serde_json::json!({"op":"downscale","ratio":r,"kernel":format!("{k:?}"),
                    "correct":"zenresize .linear()","broken":"zenresize .srgb()"}),
                correct: OutImg::Rgb(correct),
                broken: OutImg::Rgb(broken),
            });
        }
    }
    Ok(out)
}

// ===========================================================================
// F02 alpha_nopremul — RGBA scaling without premultiplication.
// 4 defs (ratio {2,4} x mask {vignette,shapes}) x 3 backgrounds = 12 items.
// ===========================================================================
pub(super) fn f02_alpha_nopremul(src: &Rgb8, origin: &str) -> Res<Vec<Twin>> {
    let mut out = Vec::new();
    for (mask_name, alpha) in [
        ("vignette", mask_vignette(src.w, src.h)),
        (
            "shapes",
            mask_shapes(
                src.w,
                src.h,
                item_seed(origin, "alpha_nopremul", "shapes", ""),
            ),
        ),
    ] {
        let rgba = rgba_from(src, &alpha);
        for (r, sev) in [(2u32, 1u32), (4, 2)] {
            let (ow, oh) = dims_div(src.w, src.h, r);
            let correct = resize_rgba(&rgba, ow, oh, MITCHELL, ZPD::RGBA8_SRGB, true)?;
            let broken = resize_rgba(&rgba, ow, oh, MITCHELL, ZPD::RGBX8_SRGB, true)?;
            let stem = match (r, mask_name) {
                (2, "vignette") => "nopremul_r2_vignette",
                (4, "vignette") => "nopremul_r4_vignette",
                (2, _) => "nopremul_r2_shapes",
                (_, _) => "nopremul_r4_shapes",
            };
            alpha_twins(
                "alpha_nopremul",
                stem,
                sev,
                serde_json::json!({"op":"rgba downscale + composite","ratio":r,
                    "kernel":"Mitchell","mask":mask_name,
                    "correct":"RGBA8_SRGB .linear() (zenresize premultiplies)",
                    "broken":"RGBX8_SRGB .linear() (4 independent channels)"}),
                &correct,
                &broken,
                &mut out,
            )?;
        }
    }
    Ok(out)
}

// ===========================================================================
// F03 alpha_premul_state — premultiplied-alpha state bugs at composite time.
// 2 bugs x 3 backgrounds = 6 items, all emitted pre-composited.
// ===========================================================================
pub(super) fn f03_alpha_premul_state(src: &Rgb8) -> Res<Vec<Twin>> {
    let alpha = mask_vignette(src.w, src.h);
    let rgba = rgba_from(src, &alpha);
    let mut out = Vec::new();
    for (bg_name, bg_fn) in BACKGROUNDS {
        let bg = bg_fn(src.w, src.h);
        let correct = composite_over(&rgba, &bg)?;
        for (bug_name, bug_fn, sev) in [
            (
                "premul_twice",
                composite_premul_twice as fn(&Rgba8, &Rgb8) -> Res<Rgb8>,
                1u32,
            ),
            (
                "forgot_premul",
                composite_forgot_premul as fn(&Rgba8, &Rgb8) -> Res<Rgb8>,
                2,
            ),
        ] {
            let broken = bug_fn(&rgba, &bg)?;
            out.push(Twin {
                family: "alpha_premul_state",
                variant: match (bug_name, bg_name) {
                    ("premul_twice", "black") => "premul_twice_black",
                    ("premul_twice", "white") => "premul_twice_white",
                    ("premul_twice", _) => "premul_twice_checker",
                    (_, "black") => "forgot_premul_black",
                    (_, "white") => "forgot_premul_white",
                    _ => "forgot_premul_checker",
                },
                severity: sev,
                params: serde_json::json!({"op":"composite rgba over bg","bg":bg_name,
                    "mask":"vignette","correct":"premul once -> SrcOver (zenblend)",
                    "broken":bug_name}),
                correct: OutImg::Rgb(correct.clone_shallow()),
                broken: OutImg::Rgb(broken),
            });
        }
    }
    Ok(out)
}

// ===========================================================================
// F04 gamma_apply — sRGB transfer applied twice, or omitted.
// ===========================================================================
pub(super) fn f04_gamma_apply(src: &Rgb8) -> Res<Vec<Twin>> {
    // Op: x1.5 linear-light gain (clamped) then sRGB encode.
    let lin = srgb8_to_linear(&src.px);
    let gained: Vec<f32> = lin.iter().map(|v| (v * 1.5).min(1.0)).collect();
    let correct = Rgb8 {
        w: src.w,
        h: src.h,
        px: linear_to_srgb8(&gained),
    };
    // (a) gamma applied twice: encode the ENCODED output again.
    let enc_f: Vec<f32> = correct.px.iter().map(|&v| v as f32 / 255.0).collect();
    let twice = Rgb8 {
        w: src.w,
        h: src.h,
        px: linear_to_srgb8(&enc_f),
    };
    // (b) gamma omitted: linear buffer emitted as bytes.
    let omitted = Rgb8 {
        w: src.w,
        h: src.h,
        px: gained
            .iter()
            .map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect(),
    };
    Ok(vec![
        Twin {
            family: "gamma_apply",
            variant: "encode_twice",
            severity: 1,
            params: serde_json::json!({"op":"linear gain 1.5 + sRGB encode",
                "correct":"encode once","broken":"sRGB encode applied to encoded data"}),
            correct: OutImg::Rgb(correct.clone_shallow()),
            broken: OutImg::Rgb(twice),
        },
        Twin {
            family: "gamma_apply",
            variant: "encode_omitted",
            severity: 2,
            params: serde_json::json!({"op":"linear gain 1.5 + sRGB encode",
                "correct":"encode once","broken":"linear emitted unencoded"}),
            correct: OutImg::Rgb(correct),
            broken: OutImg::Rgb(omitted),
        },
    ])
}

// ===========================================================================
// F05 geometry_shift — half/one-pixel shifts and wrong edge handling.
// ===========================================================================
pub(super) fn f05_geometry(src: &Rgb8) -> Res<Vec<Twin>> {
    let mut out = Vec::new();
    // (a) 1px crop-origin bug: same-size tile via zenresize .crop, origin off
    // by one. The crop is a source_region inside the resize; a 1:1 Triangle
    // resample of an integer-aligned region is identity, so the twins differ
    // only by which region is read.
    let (tw, th) = (src.w - 16, src.h - 16);
    let crop_cfg = |x: u32, y: u32| {
        ResizeConfig::builder(src.w, src.h, tw, th)
            .filter(Filter::Triangle)
            .format(ZPD::RGBA8_SRGB)
            .srgb()
            .crop(x, y, tw, th)
            .build()
    };
    let rgba = rgb_to_rgba_opaque(src);
    let strip = |cfg: &zenresize::ResizeConfig| -> Rgb8 {
        let out = Resizer::new(cfg).resize(&rgba.px);
        let mut px = Vec::with_capacity((tw * th * 3) as usize);
        for p in out.as_chunks::<4>().0 {
            px.extend_from_slice(&[p[0], p[1], p[2]]);
        }
        Rgb8 { w: tw, h: th, px }
    };
    let correct_crop = strip(&crop_cfg(8, 8));
    let broken_crop = strip(&crop_cfg(9, 9));
    out.push(Twin {
        family: "geometry_shift",
        variant: "crop_offby1",
        severity: 2,
        params: serde_json::json!({"op":format!("crop {tw}x{th} tile via zenresize .crop"),
            "correct":"origin (8,8)","broken":"origin (9,9)"}),
        correct: OutImg::Rgb(correct_crop),
        broken: OutImg::Rgb(broken_crop),
    });
    // (b) half-pixel phase: 2x Triangle upsample then parity decimate.
    //     even parity = original grid phase (correct); odd = +0.5px (bug).
    if src.w % 2 == 0 && src.h % 2 == 0 {
        let up = resize_rgb8(src, src.w * 2, src.h * 2, Filter::Triangle, true)?;
        let even = decimate_parity(&up, 0, 0)?;
        let odd = decimate_parity(&up, 1, 1)?;
        out.push(Twin {
            family: "geometry_shift",
            variant: "halfpx_phase",
            severity: 1,
            params: serde_json::json!({"op":"resample at same geometry",
                "correct":"even parity decimate (original phase)",
                "broken":"odd parity decimate (+0.5px)","note":
                "both twins share the 2x Triangle upsample kernel; only the sampling phase differs"}),
            correct: OutImg::Rgb(even),
            broken: OutImg::Rgb(odd),
        });
    }
    // (c,d) 8px translate; correct = mirror edge, broken = wrap / clamp.
    for (name, mode, sev) in [
        ("edge_wrap", EdgeFill::Wrap, 3u32),
        ("edge_clamp", EdgeFill::Clamp, 3),
    ] {
        let correct = translate(src, 8, 8, EdgeFill::Mirror);
        let broken = translate(src, 8, 8, mode);
        out.push(Twin {
            family: "geometry_shift",
            variant: name,
            severity: sev,
            params: serde_json::json!({"op":"translate +8,+8 with edge extension",
                "correct":"mirror edge","broken":name}),
            correct: OutImg::Rgb(correct),
            broken: OutImg::Rgb(broken),
        });
    }
    Ok(out)
}

// ===========================================================================
// F06 wrong_kernel — wrong resampling kernel at 4x downscale.
// ===========================================================================
pub(super) fn f06_wrong_kernel(src: &Rgb8) -> Res<Vec<Twin>> {
    let (ow, oh) = dims_div(src.w, src.h, 4);
    let correct = resize_rgb8(src, ow, oh, LANCZOS, true)?;
    let mut out = Vec::new();
    for (k, name, sev) in [
        (Filter::Triangle, "triangle", 1u32),
        (Filter::Box, "box", 2),
    ] {
        let broken = resize_rgb8(src, ow, oh, k, true)?;
        out.push(Twin {
            family: "wrong_kernel",
            variant: name,
            severity: sev,
            params: serde_json::json!({"op":"4x downscale linear-light",
                "correct":"Lanczos","broken":name}),
            correct: OutImg::Rgb(correct.clone_shallow()),
            broken: OutImg::Rgb(broken),
        });
    }
    Ok(out)
}

// ===========================================================================
// F07 channel_chroma — RGB<->BGR swap; wrong chroma siting.
// ===========================================================================
pub(super) fn f07_channel_chroma(src: &Rgb8) -> Res<Vec<Twin>> {
    let mut out = vec![Twin {
        family: "channel_chroma",
        variant: "rgb_bgr_swap",
        severity: 2,
        params: serde_json::json!({"op":"channel plumbing","correct":"RGB","broken":"BGR"}),
        correct: OutImg::Rgb(src.clone_shallow()),
        broken: OutImg::Rgb(swizzle_rb(src)),
    }];
    // Chroma siting: 4:2:0 roundtrip. Chroma planes resampled 2x down + 2x up
    // through zenresize gray (Box down = centered siting; Triangle up).
    // Broken = chroma planes shifted +1,+1 luma px first — a 0.5-chroma-px
    // siting/phase error, i.e. co-sited sampling of the wrong convention.
    let (y, cb, cr) = rgb_to_ycbcr601(src);
    let site = |plane: &[u8], shift: bool| -> Res<Vec<u8>> {
        let p = if shift {
            translate_plane(plane, src.w, src.h, 1, 1)
        } else {
            plane.to_vec()
        };
        let (ow, oh) = dims_div(src.w, src.h, 2);
        let down_cfg = ResizeConfig::builder(src.w, src.h, ow, oh)
            .filter(Filter::Box)
            .format(ZPD::GRAY8_SRGB)
            .srgb() // chroma planes stay in encoded space
            .build();
        let down = Resizer::new(&down_cfg).resize(&p);
        let up_cfg = ResizeConfig::builder(ow, oh, src.w, src.h)
            .filter(Filter::Triangle)
            .format(ZPD::GRAY8_SRGB)
            .srgb()
            .build();
        Ok(Resizer::new(&up_cfg).resize(&down))
    };
    let up_correct_cb = site(&cb, false)?;
    let up_correct_cr = site(&cr, false)?;
    let up_broken_cb = site(&cb, true)?;
    let up_broken_cr = site(&cr, true)?;
    let correct_rgb = ycbcr601_to_rgb(src.w, src.h, &y, &up_correct_cb, &up_correct_cr);
    let broken_rgb = ycbcr601_to_rgb(src.w, src.h, &y, &up_broken_cb, &up_broken_cr);
    out.push(Twin {
        family: "channel_chroma",
        variant: "chroma_siting_cosited",
        severity: 1,
        params: serde_json::json!({"op":"YCbCr 4:2:0 roundtrip (BT.601 full-range)",
            "correct":"chroma centered siting (Box 2x down, Triangle 2x up)",
            "broken":"chroma co-sited at (0,0): planes shifted +1,+1 before the same resample"}),
        correct: OutImg::Rgb(correct_rgb),
        broken: OutImg::Rgb(broken_rgb),
    });
    Ok(out)
}

// ===========================================================================
// F08 primaries_dropped — Display-P3 content treated as sRGB.
// ===========================================================================
pub(super) fn f08_primaries_dropped(src: &Rgb8) -> Res<Vec<Twin>> {
    let srgb = ZPD::RGB8_SRGB;
    let p3 = srgb.with_primaries(ColorPrimaries::DisplayP3);
    // Build the P3 asset: convert sRGB -> Display-P3 (same colours, P3
    // primaries encoding). correct = P3 -> sRGB conversion back; broken =
    // the P3 bytes read as sRGB.
    let mut to_p3 = RowConverter::new(srgb, p3).map_err(|e| format!("P3 converter: {e:?}"))?;
    let mut p3px = vec![0u8; src.px.len()];
    to_p3
        .convert_rows(
            &src.px,
            src.w as usize * 3,
            &mut p3px,
            src.w as usize * 3,
            src.w,
            src.h,
        )
        .map_err(|e| format!("P3 convert_rows: {e:?}"))?;
    let mut to_srgb = RowConverter::new(p3, srgb).map_err(|e| format!("sRGB converter: {e:?}"))?;
    let mut correct = vec![0u8; src.px.len()];
    to_srgb
        .convert_rows(
            &p3px,
            src.w as usize * 3,
            &mut correct,
            src.w as usize * 3,
            src.w,
            src.h,
        )
        .map_err(|e| format!("sRGB convert_rows: {e:?}"))?;
    Ok(vec![Twin {
        family: "primaries_dropped",
        variant: "p3_as_srgb",
        severity: 1,
        params: serde_json::json!({"op":"Display-P3 -> sRGB delivery",
            "correct":"zenpixels-convert P3->sRGB","broken":"P3 bytes read as sRGB"}),
        correct: OutImg::Rgb(Rgb8 {
            w: src.w,
            h: src.h,
            px: correct,
        }),
        broken: OutImg::Rgb(Rgb8 {
            w: src.w,
            h: src.h,
            px: p3px,
        }),
    }])
}

// ===========================================================================
// F09 exif_orientation — orientation ignored.
// ===========================================================================
fn orient_rgb8(img: &Rgb8, o: Orientation) -> Res<Rgb8> {
    let slice = PixelSlice::new(&img.px, img.w, img.h, img.w as usize * 3, ZPD::RGB8_SRGB)
        .map_err(|e| format!("PixelSlice: {e:?}"))?;
    let out = apply_orientation(slice, o);
    let view = out.as_slice();
    let mut px = Vec::with_capacity((view.width() * view.rows() * 3) as usize);
    for y in 0..view.rows() {
        px.extend_from_slice(view.row(y));
    }
    Ok(Rgb8 {
        w: out.width(),
        h: out.height(),
        px,
    })
}

pub(super) fn f09_exif(src: &Rgb8) -> Res<Vec<Twin>> {
    let mut out = Vec::new();
    for (o, name, sev) in [
        (Orientation::FlipH, "exif2_fliph_ignored", 1u32),
        (Orientation::Rotate180, "exif3_rot180_ignored", 1),
        (Orientation::FlipV, "exif4_flipv_ignored", 1),
    ] {
        let correct = orient_rgb8(src, o)?;
        out.push(Twin {
            family: "exif_orientation",
            variant: name,
            severity: sev,
            params: serde_json::json!({"op":format!("apply orientation {o:?}"),
                "correct":"zenpixels-convert apply_orientation","broken":"identity"}),
            correct: OutImg::Rgb(correct),
            broken: OutImg::Rgb(src.clone_shallow()),
        });
    }
    // rot90 on the center square (dims preserved for scoring).
    let sq = center_square(src);
    let correct = orient_rgb8(&sq, Orientation::Rotate90)?;
    out.push(Twin {
        family: "exif_orientation",
        variant: "exif6_rot90_ignored",
        severity: 2,
        params: serde_json::json!({"op":"apply orientation Rotate90 on center square",
            "correct":"apply_orientation","broken":"identity"}),
        correct: OutImg::Rgb(correct),
        broken: OutImg::Rgb(sq),
    });
    Ok(out)
}

// ===========================================================================
// F10 bitdepth — 16->8 truncation, overflow wrap, changed dither.
// ===========================================================================
pub(super) fn f10_bitdepth(src: &Rgb8) -> Res<Vec<Twin>> {
    let v16 = expand_u16(src);
    let mut out = Vec::new();
    // (a) truncation vs rounding. A u16 gain first is REQUIRED: raw *257
    // expansion lands every sample on an exact u8 boundary, so trunc and
    // round agree byte-for-byte (the twin would be inert).
    let g16: Vec<u16> = v16
        .iter()
        .map(|&v| ((v as u32 * 3 / 2).min(65535)) as u16)
        .collect();
    let correct: Vec<u8> = g16.iter().map(|&v| quantize_u16_round(v)).collect();
    let broken: Vec<u8> = g16.iter().map(|&v| quantize_u16_trunc(v)).collect();
    out.push(Twin {
        family: "bitdepth",
        variant: "trunc_vs_round",
        severity: 1,
        params: serde_json::json!({"op":"u16 gain x1.5 -> u8 quantize",
            "correct":"round v*255/65535","broken":"v >> 8"}),
        correct: OutImg::Rgb(Rgb8 {
            w: src.w,
            h: src.h,
            px: correct,
        }),
        broken: OutImg::Rgb(Rgb8 {
            w: src.w,
            h: src.h,
            px: broken,
        }),
    });
    // (b) overflow: +4096 u16 gain, saturate vs wrap.
    let correct: Vec<u8> = v16
        .iter()
        .map(|&v| quantize_u16_round(v.saturating_add(4096)))
        .collect();
    let broken: Vec<u8> = v16
        .iter()
        .map(|&v| quantize_u16_round(v.wrapping_add(4096)))
        .collect();
    out.push(Twin {
        family: "bitdepth",
        variant: "overflow_wrap",
        severity: 2,
        params: serde_json::json!({"op":"u16 gain +4096 then quantize",
            "correct":"saturating_add","broken":"wrapping_add"}),
        correct: OutImg::Rgb(Rgb8 {
            w: src.w,
            h: src.h,
            px: correct,
        }),
        broken: OutImg::Rgb(Rgb8 {
            w: src.w,
            h: src.h,
            px: broken,
        }),
    });
    // (c) changed dither: 5-bit quantize with vs without ordered dither.
    let correct = quantize5(src, true, 0);
    let broken = quantize5(src, false, 0);
    out.push(Twin {
        family: "bitdepth",
        variant: "dither_removed",
        severity: 1,
        params: serde_json::json!({"op":"quantize to 5bpc and back",
            "correct":"Bayer 4x4 ordered dither","broken":"no dither"}),
        correct: OutImg::Rgb(correct),
        broken: OutImg::Rgb(broken),
    });
    Ok(out)
}

// ===========================================================================
// Benign drift — correct-but-different implementation pairs (prereg §4).
// Same Twin shape; the driver records kind=benign and treats a/b as
// unordered implementations of the same op.
// ===========================================================================

/// sRGB-space resize in the f32 domain: u8 -> f32 (encoded) -> resize_f32 ->
/// encode. The float-vs-fixed-point side of the benign pair.
fn resize_srgb_f32(src: &Rgb8, ow: u32, oh: u32, k: Filter) -> Res<Rgb8> {
    let rgba = rgb_to_rgba_opaque(src);
    let f: Vec<f32> = rgba.px.iter().map(|&v| v as f32 / 255.0).collect();
    let desc = ZPD::RGBAF32_LINEAR.with_transfer(TransferFunction::Srgb);
    let cfg = ResizeConfig::builder(src.w, src.h, ow, oh)
        .filter(k)
        .format(desc)
        .srgb()
        .build();
    let out = Resizer::new(&cfg).resize_f32(&f);
    let mut px = Vec::with_capacity((ow * oh * 3) as usize);
    for p in out.as_chunks::<4>().0 {
        for c in 0..3 {
            px.push((p[c] * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(Rgb8 { w: ow, h: oh, px })
}

/// Same sRGB-space resize via the crate's own u8 -> f32 route
/// (`resize_u8_to_f32`) — the tier-orthogonal route pair.
fn resize_u8_f32_route(src: &Rgb8, ow: u32, oh: u32, k: Filter) -> Res<Rgb8> {
    let rgba = rgb_to_rgba_opaque(src);
    let cfg = ResizeConfig::builder(src.w, src.h, ow, oh)
        .filter(k)
        .input(ZPD::RGBA8_SRGB)
        .output(ZPD::RGBAF32_LINEAR.with_transfer(TransferFunction::Srgb))
        .srgb()
        .build();
    let out = Resizer::new(&cfg).resize_u8_to_f32(&rgba.px);
    let mut px = Vec::with_capacity((ow * oh * 3) as usize);
    for p in out.as_chunks::<4>().0 {
        for c in 0..3 {
            px.push((p[c] * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(Rgb8 { w: ow, h: oh, px })
}

/// Linear-light resize through the u16 working path (`resize_u8_to_u16`),
/// quantized u16 -> u8 by rounding — the fixed-point side of the
/// linear-light benign pair.
fn resize_lin_u16(src: &Rgb8, ow: u32, oh: u32, k: Filter) -> Res<Rgb8> {
    let rgba = rgb_to_rgba_opaque(src);
    let cfg = ResizeConfig::builder(src.w, src.h, ow, oh)
        .filter(k)
        .input(ZPD::RGBA8_SRGB)
        .output(ZPD::RGBA16_SRGB)
        .linear()
        .build();
    let out16 = Resizer::new(&cfg).resize_u8_to_u16(&rgba.px);
    let mut px = Vec::with_capacity((ow * oh * 3) as usize);
    for p in out16.as_chunks::<4>().0 {
        for c in 0..3 {
            px.push(quantize_u16_round(p[c]));
        }
    }
    Ok(Rgb8 { w: ow, h: oh, px })
}

/// StreamingResize, same config as `resize_rgb8` — second correct resampler
/// implementation for `resize_streaming_vs_fullframe`.
fn resize_streaming(src: &Rgb8, ow: u32, oh: u32, k: Filter, linear: bool) -> Res<Rgb8> {
    let rgba = rgb_to_rgba_opaque(src);
    let mut b = ResizeConfig::builder(src.w, src.h, ow, oh)
        .filter(k)
        .format(ZPD::RGBA8_SRGB);
    b = if linear { b.linear() } else { b.srgb() };
    let mut r = StreamingResize::new(&b.build());
    let mut out = Vec::with_capacity((ow * oh * 4) as usize);
    let rowlen = src.w as usize * 4;
    for y in 0..src.h as usize {
        r.push_row(&rgba.px[y * rowlen..(y + 1) * rowlen])
            .map_err(|e| format!("{e:?}"))?;
        while let Some(row) = r.next_output_row() {
            out.extend_from_slice(row);
        }
    }
    r.finish();
    let mut guard = 0usize;
    while !r.is_complete() {
        match r.next_output_row() {
            Some(row) => out.extend_from_slice(row),
            None => break,
        }
        guard += 1;
        if guard > oh as usize + 4 {
            return Err("streaming resize did not complete".into());
        }
    }
    if out.len() != (ow * oh * 4) as usize {
        return Err(format!(
            "streaming resize produced {} bytes, expected {}",
            out.len(),
            ow * oh * 4
        )
        .into());
    }
    let mut px = Vec::with_capacity((ow * oh * 3) as usize);
    for p in out.as_chunks::<4>().0 {
        px.extend_from_slice(&[p[0], p[1], p[2]]);
    }
    Ok(Rgb8 { w: ow, h: oh, px })
}

/// u16 -> u8 quantize with a Bayer-ordered ±1-LSB dither — a second correct
/// quantization policy for the `quantize_dithered` benign items.
fn quantize_u16_dithered(w: u32, h: u32, v16: &[u16], phase: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(v16.len());
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) as usize * 3;
            for c in 0..3usize {
                let v = v16[base + c] as u32;
                // dither in u16 units: +-~128 u16 = +-0.5 u8-LSB
                let d = (bayer4(x + phase, y + phase) as i32 - 8) * 16;
                let q = ((v as i32 + d).clamp(0, 65535) as u32 * 255 + 32767) / 65535;
                out.push(q.min(255) as u8);
            }
        }
    }
    out
}

pub(super) fn benign_items(src256: &Rgb8, src512: &Rgb8) -> Res<Vec<Twin>> {
    let mut out: Vec<Twin> = Vec::new();
    let mut push = |family: &'static str, params: serde_json::Value, a: Rgb8, b: Rgb8| {
        out.push(Twin {
            family,
            variant: family,
            severity: 0,
            params,
            correct: OutImg::Rgb(a),
            broken: OutImg::Rgb(b),
        });
    };
    // resample families run on the 512 rendition, ratios {2,4} x {M,L} (4 each)
    for (r, k) in [(2u32, MITCHELL), (2, LANCZOS), (4, MITCHELL), (4, LANCZOS)] {
        let (ow, oh) = dims_div(src512.w, src512.h, r);
        let kn = format!("{k:?}");
        push(
            "resize_streaming_vs_fullframe",
            serde_json::json!({"op":"srgb-space resize","ratio":r,"kernel":kn,
                "a":"Resizer","b":"StreamingResize"}),
            resize_rgb8(src512, ow, oh, k, false)?,
            resize_streaming(src512, ow, oh, k, false)?,
        );
        push(
            "resize_f32_vs_i16",
            serde_json::json!({"op":"srgb-space resize","ratio":r,"kernel":kn,
                "a":"u8 i16 path (.srgb())","b":"f32 encoded-space path"}),
            resize_rgb8(src512, ow, oh, k, false)?,
            resize_srgb_f32(src512, ow, oh, k)?,
        );
        push(
            "resize_u16_vs_f32_lin",
            serde_json::json!({"op":"linear-light resize","ratio":r,"kernel":kn,
                "a":"u8 linear f32 path","b":"u16 linear path"}),
            resize_rgb8(src512, ow, oh, k, true)?,
            resize_lin_u16(src512, ow, oh, k)?,
        );
    }
    // sRGB<->linear implementations: default LUT/poly vs precise powf,
    // through gain {1.0, 0.85} on the source AND its Mitchell-2x downscale.
    let src256_half = resize_rgb8(src256, src256.w / 2, src256.h / 2, MITCHELL, false)?;
    for (gain, gname) in [(1.0f32, "g100"), (0.85, "g085")] {
        for (img, iname) in [(src256, "full"), (&src256_half, "half")] {
            let lin_a = srgb8_to_linear(&img.px);
            let lin_b = srgb8_to_linear_precise(&img.px);
            let a = Rgb8 {
                w: img.w,
                h: img.h,
                px: linear_to_srgb8(&lin_a.iter().map(|v| v * gain).collect::<Vec<_>>()),
            };
            let b = Rgb8 {
                w: img.w,
                h: img.h,
                px: linear_to_srgb8_precise(&lin_b.iter().map(|v| v * gain).collect::<Vec<_>>()),
            };
            push(
                "srgb_lut_vs_poly",
                serde_json::json!({"op":"srgb->linear->gain->srgb","gain":gname,
                    "input":iname,"a":"default LUT/poly","b":"precise powf"}),
                a,
                b,
            );
        }
    }
    // u16 -> u8: two correct roundings (canonical vs half-away). For u16->u8
    // these are provably identical when they agree (65535 = 255*257 admits no
    // exact ties) — items that come out byte-identical are counted inert per
    // the prereg drop rule. The dithered-quantize items below carry the real
    // ±1-LSB drift.
    let q_contexts: Vec<(String, Vec<u16>, u32, u32)> = vec![
        ("expand_full".into(), expand_u16(src256), src256.w, src256.h),
        (
            "expand_quad".into(),
            expand_u16(&crop(src256, 0, 0, src256.w / 2, src256.h / 2)?),
            src256.w / 2,
            src256.h / 2,
        ),
        (
            "expand_half".into(),
            expand_u16(&src256_half),
            src256_half.w,
            src256_half.h,
        ),
        (
            "expand_half_quad".into(),
            expand_u16(&crop(
                &src256_half,
                0,
                0,
                src256_half.w / 2,
                src256_half.h / 2,
            )?),
            src256_half.w / 2,
            src256_half.h / 2,
        ),
    ];
    for (name, v16, w, h) in &q_contexts {
        // u16 gain first (REQUIRED): raw *257-expanded inputs sit exactly on
        // u8 quantization boundaries, so both roundings AND the sub-LSB
        // dither agree byte-for-byte — every pair would be inert.
        let g16: Vec<u16> = v16
            .iter()
            .map(|&v| ((v as u32 * 3 / 2).min(65535)) as u16)
            .collect();
        let a: Vec<u8> = g16.iter().map(|&v| quantize_u16_round(v)).collect();
        let b: Vec<u8> = g16.iter().map(|&v| quantize_u16_round_away(v)).collect();
        push(
            "quantize_round_half_even_vs_away",
            serde_json::json!({"op":"u16 gain x1.5 -> u8 quantize","context":name,
                "a":"round v*255/65535","b":"(v+128)/257"}),
            Rgb8 {
                w: *w,
                h: *h,
                px: a,
            },
            Rgb8 {
                w: *w,
                h: *h,
                px: b,
            },
        );
        // plain-round vs ordered-dither quantize — the real ±1-LSB pair.
        let c = quantize_u16_dithered(*w, *h, &g16, 0);
        push(
            "quantize_dithered_vs_plain",
            serde_json::json!({"op":"u16 gain x1.5 -> u8 quantize","context":name,
                "a":"round v*255/65535","b":"Bayer-dithered round"}),
            Rgb8 {
                w: *w,
                h: *h,
                px: g16.iter().map(|&v| quantize_u16_round(v)).collect(),
            },
            Rgb8 {
                w: *w,
                h: *h,
                px: c,
            },
        );
    }
    // Dither phase — two correct dithered quantizations.
    for phase in [0u32, 1] {
        push(
            "dither_phase",
            serde_json::json!({"op":"5bpc dithered quantize","phase":phase,
                "a":"Bayer at phase p","b":"Bayer at phase p+1"}),
            quantize5(src256, true, phase),
            quantize5(src256, true, phase + 1),
        );
    }
    // Composite precision — f32 vs u16 premultiplied intermediate.
    let alpha = mask_vignette(src256.w, src256.h);
    let rgba = rgba_from(src256, &alpha);
    for (bg_name, bg_fn) in BACKGROUNDS {
        let bg = bg_fn(src256.w, src256.h);
        push(
            "composite_f32_vs_u16",
            serde_json::json!({"op":"composite vignette rgba","bg":bg_name,
                "a":"linear-srgb f32 + zenblend","b":"u16 premul arithmetic"}),
            composite_over(&rgba, &bg)?,
            composite_over_u16(&rgba, &bg)?,
        );
    }
    // Tier-orthogonal whole-route pairs: the crate's own conversion routes
    // vs the direct u8 route (u8->f32 sRGB route; u8->u16 linear route).
    for (r, k) in [
        (2u32, MITCHELL),
        (2, LANCZOS),
        (4, MITCHELL),
        (4, LANCZOS),
        (3, MITCHELL),
        (8, MITCHELL),
    ] {
        let (ow, oh) = dims_div(src512.w, src512.h, r);
        let kn = format!("{k:?}");
        push(
            "route_u8_vs_u8f32",
            serde_json::json!({"op":"srgb-space resize (whole route)","ratio":r,
                "kernel":kn,"a":"resize u8->u8","b":"resize_u8_to_f32 + encode"}),
            resize_rgb8(src512, ow, oh, k, false)?,
            resize_u8_f32_route(src512, ow, oh, k)?,
        );
        if r != 3 && r != 8 {
            push(
                "route_u8_vs_u8u16_lin",
                serde_json::json!({"op":"linear resize (whole route)","ratio":r,
                    "kernel":kn,"a":"resize u8->u8 linear","b":"resize_u8_to_u16 + quantize"}),
                resize_rgb8(src512, ow, oh, k, true)?,
                resize_lin_u16(src512, ow, oh, k)?,
            );
        }
    }
    Ok(out)
}

// ===========================================================================
// Family table
// ===========================================================================

/// All corruption twins for one source. `src512` feeds the resample families
/// (F01/F02/F06); the pixel-state families take `src256`.
pub(super) fn all_twins(src256: &Rgb8, src512: &Rgb8, origin: &str) -> Res<Vec<Twin>> {
    let mut v = Vec::new();
    v.extend(f01_gamma_downsample(src512)?);
    v.extend(f02_alpha_nopremul(src512, origin)?);
    v.extend(f03_alpha_premul_state(src256)?);
    v.extend(f04_gamma_apply(src256)?);
    v.extend(f05_geometry(src256)?);
    v.extend(f06_wrong_kernel(src512)?);
    v.extend(f07_channel_chroma(src256)?);
    v.extend(f08_primaries_dropped(src256)?);
    v.extend(f09_exif(src256)?);
    v.extend(f10_bitdepth(src256)?);
    Ok(v)
}

impl Rgb8 {
    /// Clone pixel data (kept off the struct's derive to match the owner's
    /// non-Clone Rgb8).
    pub fn clone_shallow(&self) -> Rgb8 {
        Rgb8 {
            w: self.w,
            h: self.h,
            px: self.px.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn busy(w: u32, h: u32) -> Rgb8 {
        // deterministic high-detail image: edges + gradients + saturated cols
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        let mut rng = Mulberry::new(0xE5A);
        for y in 0..h {
            for x in 0..w {
                let edge = if (x / 4 + y / 4) % 2 == 0 { 230u32 } else { 25 };
                let n = (rng.next_u64() % 37) as u32;
                px.extend_from_slice(&[
                    ((x * 3 + n) % 256) as u8,
                    ((y * 5 + edge) % 256) as u8,
                    ((x + y * 2 + edge) % 256) as u8,
                ]);
            }
        }
        Rgb8 { w, h, px }
    }

    #[test]
    fn every_family_produces_different_twins() {
        // even dims: halfpx_phase emits -> full 44-item prereg count
        let s256 = busy(254, 300);
        let s512 = busy(512, 384);
        let twins = all_twins(&s256, &s512, "testsrc").unwrap();
        assert_eq!(twins.len(), 44, "twins = {}", twins.len());
        // odd dims: halfpx_phase is skipped (parity decimate needs even dims)
        let s256o = busy(255, 301);
        let s512o = busy(513, 387);
        let twins_o = all_twins(&s256o, &s512o, "testsrc").unwrap();
        assert_eq!(twins_o.len(), 43, "odd-dim twins = {}", twins_o.len());
        for t in twins.iter().chain(&twins_o) {
            let same = t.correct.as_rgb().px == t.broken.as_rgb().px;
            assert!(!same, "inert twin {}/{}", t.family, t.variant);
            assert_eq!(
                t.correct.dims(),
                t.broken.dims(),
                "dims {}/{}",
                t.family,
                t.variant
            );
        }
    }

    #[test]
    fn gamma_downsample_correct_is_zenresize_linear() {
        let img = busy(64, 48);
        let twins = f01_gamma_downsample(&img).unwrap();
        let c = twins[0].correct.as_rgb();
        let direct = resize_rgb8(&img, 32, 24, MITCHELL, true).unwrap();
        assert_eq!(c.px, direct.px);
    }

    #[test]
    fn p3_roundtrip_close() {
        let img = busy(31, 17);
        let twins = f08_primaries_dropped(&img).unwrap();
        let c = twins[0].correct.as_rgb();
        let maxd = img
            .px
            .iter()
            .zip(&c.px)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        // sRGB→P3→sRGB is near-lossless (P3 gamut covers sRGB) but each u8
        // quantization near the gamut edge can cost a couple of LSB.
        assert!(maxd <= 8, "P3 roundtrip max |Δ| = {maxd}");
    }

    #[test]
    fn geometry_variants_move_content() {
        let img = busy(64, 64);
        let twins = f05_geometry(&img).unwrap();
        for t in &twins {
            let (a, b) = (t.correct.as_rgb(), t.broken.as_rgb());
            assert_eq!(a.w, b.w);
            let diff: usize =
                a.px.iter()
                    .zip(&b.px)
                    .map(|(x, y)| (*x as i32 - *y as i32).abs() as usize)
                    .sum();
            assert!(diff > 0, "{} produced no diff", t.variant);
        }
    }

    #[test]
    fn exif_dims_preserved() {
        let img = busy(50, 34);
        let twins = f09_exif(&img).unwrap();
        for t in &twins {
            assert_eq!(t.correct.dims(), t.broken.dims(), "{}", t.variant);
        }
    }

    #[test]
    fn benign_pairs_drift_bounded() {
        let s256 = busy(255, 301);
        let s512 = busy(513, 387);
        let items = benign_items(&s256, &s512).unwrap();
        assert!(items.len() >= 30, "benign items = {}", items.len());
        for t in &items {
            let (a, b) = (t.correct.as_rgb(), t.broken.as_rgb());
            assert_eq!((a.w, a.h), (b.w, b.h), "{} dims", t.family);
            // correct implementations of the same op must be CLOSE:
            // max |diff| stays small (bounding drift, not bugs)
            let maxd =
                a.px.iter()
                    .zip(&b.px)
                    .map(|(x, y)| (*x as i32 - *y as i32).abs())
                    .max()
                    .unwrap_or(0);
            assert!(maxd <= 16, "{}: benign drift max |Δ| = {maxd}", t.family);
        }
    }
}
