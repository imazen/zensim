//! E5A render-mode driver: emits correct/broken rendering-implementation
//! twins plus the benign-drift set for one TRAIN source, with a hashed
//! manifest. Reached through `m3_fixture_gen corruption render ...` so the
//! work stays inside the existing corruption owner.
//!
//! Layout per source dir:
//!   ref256.png / ref512.png          the two input renditions, byte-exact
//!   item-NNNN-{correct,broken}.png   corruption twins (a scored pair)
//!   item-NNNN-{a,b}.png              benign drift pairs (unordered)
//!   _MANIFEST.json                   full record set + input/output hashes
//!   COMPLETE.json                    counts + manifest sha
use super::corruption::{sha, write_verified_png};
use super::render_families::{all_twins, benign_items};
use super::render_pix::{OutImg, Twin};
use super::*;
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::time::Instant;

fn emit_item(
    out: &PathBuf,
    index: usize,
    kind: &str,
    t: &Twin,
    records: &mut Vec<serde_json::Value>,
) -> Res<bool> {
    let (a_name, b_name) = if kind == "benign" {
        (
            format!("item-{index:04}-a.png"),
            format!("item-{index:04}-b.png"),
        )
    } else {
        (
            format!("item-{index:04}-correct.png"),
            format!("item-{index:04}-broken.png"),
        )
    };
    let write = |name: &str, img: &OutImg| -> Res<(String, String, u32, u32)> {
        let i = img.as_rgb();
        let png_sha = write_verified_png(&out.join(name), i)?;
        Ok((png_sha, sha(&i.px), i.w, i.h))
    };
    let (a_png, a_px, aw, ah) = write(&a_name, &t.correct)?;
    let (b_png, b_px, bw, bh) = write(&b_name, &t.broken)?;
    if (aw, ah) != (bw, bh) {
        return Err(format!("{}: twin dims differ ({aw}x{ah} vs {bw}x{bh})", t.variant).into());
    }
    let inert = a_px == b_px;
    records.push(json!({
        "kind": kind,
        "index": index,
        "family": t.family,
        "variant": t.variant,
        "severity": t.severity,
        "params": t.params,
        "a_file": a_name, "b_file": b_name,
        "a_pixels_sha256": a_px, "b_pixels_sha256": b_px,
        "a_png_sha256": a_png, "b_png_sha256": b_png,
        "width": aw, "height": ah,
        "inert": inert,
    }));
    Ok(inert)
}

pub(super) fn run(args: &[String]) -> Res<()> {
    // args[0] == "render"; strict key/value pairs after it.
    let mut flags = BTreeSet::new();
    for pair in args[1..].chunks(2) {
        if pair.len() != 2
            || !["--in256", "--in512", "--out", "--ref-id", "--seed", "--set"]
                .contains(&pair[0].as_str())
            || pair[1].starts_with("--")
            || !flags.insert(&pair[0])
        {
            return Err("invalid, missing, or duplicate render argument.\n\
                 usage: m3_fixture_gen corruption render --in256 <p.png> --in512 <p.png> \\\n\
                 --out <dir> --ref-id <origin> [--seed N] [--set corruption|benign|both]"
                .into());
        }
    }
    let in256 = PathBuf::from(need(args, "--in256")?);
    let in512 = PathBuf::from(need(args, "--in512")?);
    let out = PathBuf::from(need(args, "--out")?);
    let origin = need(args, "--ref-id")?;
    let seed: u64 = arg(args, "--seed").unwrap_or_else(|| "1".into()).parse()?;
    let set = arg(args, "--set").unwrap_or_else(|| "both".into());
    if !["corruption", "benign", "both"].contains(&set.as_str()) {
        return Err("--set must be corruption|benign|both".into());
    }
    if origin.is_empty()
        || !origin
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("--ref-id must be a nonempty simple origin identifier".into());
    }
    if out.exists() {
        return Err("output must be fresh".into());
    }
    let src256_bytes = fs::read(&in256)?;
    let src512_bytes = fs::read(&in512)?;
    let src256 = read_png_rgb8(&in256)?;
    let src512 = read_png_rgb8(&in512)?;
    if src256.w < 32 || src256.h < 32 || src512.w < 64 || src512.h < 64 {
        return Err(format!(
            "render sources too small: 256={}x{} 512={}x{}",
            src256.w, src256.h, src512.w, src512.h
        )
        .into());
    }
    fs::create_dir_all(&out)?;
    let started = Instant::now();
    let ref256_sha = write_verified_png(&out.join("ref256.png"), &src256)?;
    let ref512_sha = write_verified_png(&out.join("ref512.png"), &src512)?;
    let mut records = Vec::new();
    let (mut n_corr, mut n_corr_inert, mut n_benign, mut n_benign_inert) = (0, 0, 0, 0);
    let mut index = 0usize;
    if set != "benign" {
        for t in all_twins(&src256, &src512, &origin)? {
            n_corr += 1;
            n_corr_inert += usize::from(emit_item(&out, index, "corruption", &t, &mut records)?);
            index += 1;
        }
    }
    if set != "corruption" {
        for t in benign_items(&src256, &src512)? {
            n_benign += 1;
            n_benign_inert += usize::from(emit_item(&out, index, "benign", &t, &mut records)?);
            index += 1;
        }
    }
    let manifest = json!({
        "schema": "e5a-render-fixtures-v1",
        "generator": "m3_fixture_gen corruption render (quarantine/devin/e5a-render)",
        "generator_revision": std::env::var("E5A_REV").unwrap_or_else(|_| "workspace".into()),
        "origin": origin,
        "base_seed": seed,
        "in256": in256, "in256_sha256": sha(&src256_bytes),
        "in256_pixels_sha256": sha(&src256.px),
        "in256_dims": [src256.w, src256.h],
        "in512": in512, "in512_sha256": sha(&src512_bytes),
        "in512_pixels_sha256": sha(&src512.px),
        "in512_dims": [src512.w, src512.h],
        "ref256_png_sha256": ref256_sha, "ref512_png_sha256": ref512_sha,
        "pixel_contract": "packed opaque RGB8 sRGB; PNGs written+verified by zenpng roundtrip",
        "alpha_protocol": "every RGBA item composited in linear light (zenblend SrcOver on premultiplied f32) over black/white/8px-checker(204/51); one scored item per background",
        "n_corruption": n_corr, "n_corruption_inert": n_corr_inert,
        "n_benign": n_benign, "n_benign_inert": n_benign_inert,
        "records": records,
    });
    fs::write(
        out.join("_MANIFEST.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(
        out.join("COMPLETE.json"),
        serde_json::to_vec_pretty(&json!({
            "in256_sha256": sha(&src256_bytes), "in512_sha256": sha(&src512_bytes),
            "n_corruption": n_corr, "n_corruption_inert": n_corr_inert,
            "n_benign": n_benign, "n_benign_inert": n_benign_inert,
            "manifest_sha256": sha(&fs::read(out.join("_MANIFEST.json"))?),
            "elapsed_seconds": started.elapsed().as_secs_f64()
        }))?,
    )?;
    eprintln!(
        "{origin}: {n_corr} corruption ({n_corr_inert} inert), {n_benign} benign ({n_benign_inert} inert)"
    );
    Ok(())
}
