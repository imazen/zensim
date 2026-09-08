//! Serve an arbitrary ZNPR bake through the PRODUCTION scoring path
//! (`BakeScorer::compute`), so "is this candidate servable?" is a MEASUREMENT
//! rather than an inference from reading `profile.rs`.
//!
//! Originally written for `benchmarks/fastclass2_campaign_2026-09-05.md`
//! gate G7. Since September 7, `BakeScorer` serves declared feature IDs and
//! complete candidate composition through the shared extraction plan,
//! including supported wide bakes. No input-count guess selects a layout.
//! `diffmap_block_coherence --bake` also exercises the bound score/map surface.
//!
//! ```sh
//! cargo run --release --example serve_custom_bake \
//!   --features custom-profiles,candidate-profiles \
//!   -- <bake.bin> <ref.png> <dist.png>
//! ```
//!
//! Prints the bake's declared caller width, the served score, and the
//! reference's identity score (`ref` vs itself) — the C5 quantity — or the
//! exact error if the production path refuses the bake.
//!
//! ## Census mode — the SERVABILITY CENSUS driver
//!
//! ```sh
//! cargo run --release --example serve_custom_bake \
//!   --features custom-profiles,candidate-profiles \
//!   -- --census <ref.png> <dist.png> [--fulleval-dir DIR] [PATH|DIR]...
//! ```
//!
//! Walks many bakes through the SAME `BakeScorer::compute` entry the single-bake
//! mode uses — one implementation, not two — and prints a TSV row per bake
//! plus a SERVED/REFUSED summary. This is the filesystem tier of the
//! servability contract (user directive 2026-09-05: *"also make sure
//! everything can be served"*); the no-filesystem tier is the in-lib census
//! (`serving::tests::every_shipped_profile_is_servable`), which gates every SHIPPED profile and
//! every registered producer set on a build with no `/mnt/v`.
//!
//! A REFUSED row is the contract failing, not the tool: every bake whose read
//! set is registered feature ids at a supported revision must serve.

use zensim::{BakeScorer, RgbSlice};

fn load_rgb(path: &str) -> (Vec<[u8; 3]>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let raw = img.into_raw();
    (raw.as_chunks::<3>().0.to_vec(), w, h)
}

/// One census row: the bake, its declared width, and what the production
/// entry did with it.
fn census_one(path: &str, rs: &RgbSlice<'_>, ds: &RgbSlice<'_>) -> (bool, String) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return (false, format!("{path}\t-\t-\tUNREADABLE\t{e}")),
    };
    let model = match zenpredict::Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => return (false, format!("{path}\t-\t-\tNOT_A_ZNPR\t{e:?}")),
    };
    let mut z = match BakeScorer::new(&model) {
        Ok(z) => z,
        Err(e) => return (false, format!("{path}\t-\t-\tREFUSED\t{e}")),
    };
    let declared = model.caller_input_width();
    match z.compute(rs, ds, None) {
        Ok(res) => (
            true,
            format!(
                "{path}\t{declared}\t{}\tSERVED\t{:.17e}",
                res.features().len(),
                res.score()
            ),
        ),
        Err(e) => (false, format!("{path}\t{declared}\t-\tREFUSED\t{e:?}")),
    }
}

/// Every `*.bin` under `root`, recursively.
fn collect_bins(root: &std::path::Path, out: &mut Vec<String>) {
    if root.is_file() {
        out.push(root.display().to_string());
        return;
    }
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_bins(&p, out);
        } else if p.extension().is_some_and(|x| x == "bin") {
            out.push(p.display().to_string());
        }
    }
}

/// Bake paths named by the `"bake"` field of every `*.fulleval.json` in `dir`.
/// A narrow scan rather than a serde dep, matching the rest of this repo's
/// registry readers.
fn bakes_from_fullevals(dir: &str) -> Vec<String> {
    use sha2::{Digest, Sha256};
    let mut out = Vec::new();
    let mut local = Vec::new();
    collect_bins(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights"),
        &mut local,
    );
    let by_sha: std::collections::HashMap<String, String> = local
        .into_iter()
        .filter_map(|p| {
            std::fs::read(&p).ok().map(|b| {
                (
                    Sha256::digest(b)
                        .iter()
                        .map(|x| format!("{x:02x}"))
                        .collect::<String>(),
                    p,
                )
            })
        })
        .collect();
    let rd = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("census root {dir}: {e}"));
    for e in rd {
        let p = e.expect("read census entry").path();
        if !p.to_string_lossy().ends_with(".fulleval.json") {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&p).expect("read verdict"))
                .unwrap_or_else(|e| panic!("malformed verdict {}: {e}", p.display()));
        let Some(path) = value.get("bake").and_then(|v| v.as_str()) else {
            continue;
        };
        if !std::path::Path::new(path).is_file()
            && let Some(resolved) = value
                .get("bake_sha256")
                .and_then(|v| v.as_str())
                .and_then(|sha| by_sha.get(sha))
        {
            eprintln!(
                "# resolved missing historical bake by recorded SHA256: {path} -> {resolved}"
            );
            out.push(resolved.clone());
        } else {
            out.push(path.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn census(args: &[String]) {
    let mut it = args.iter();
    let ref_path = it.next().expect("--census <ref> <dist> [paths...]").clone();
    let dist_path = it.next().expect("dist image required").clone();
    let mut paths: Vec<String> = Vec::new();
    let rest: Vec<&String> = it.collect();
    let mut k = 0;
    while k < rest.len() {
        if rest[k] == "--fulleval-dir" {
            paths.extend(bakes_from_fullevals(rest[k + 1]));
            k += 2;
        } else {
            collect_bins(std::path::Path::new(rest[k]), &mut paths);
            k += 1;
        }
    }
    for p in &mut paths {
        if let Ok(c) = std::fs::canonicalize(&*p) {
            *p = c.display().to_string();
        }
    }
    paths.sort();
    paths.dedup();

    let (r, w, h) = load_rgb(&ref_path);
    let (d, dw, dh) = load_rgb(&dist_path);
    assert_eq!((w, h), (dw, dh), "ref and dist must share dimensions");
    let rs = RgbSlice::new(&r, w as usize, h as usize);
    let ds = RgbSlice::new(&d, w as usize, h as usize);

    println!("bake\tdeclared\temitted\toutcome\tdetail");
    let (mut served, mut refused) = (0usize, 0usize);
    let mut refusals: Vec<String> = Vec::new();
    for p in &paths {
        let (ok, row) = census_one(p, &rs, &ds);
        println!("{row}");
        if ok {
            served += 1;
        } else {
            refused += 1;
            refusals.push(row);
        }
    }
    eprintln!(
        "\n# SERVABILITY CENSUS: {served} SERVED, {refused} REFUSED, of {} bakes",
        paths.len()
    );
    if refused > 0 {
        eprintln!("# refusals (the contract failing, not the tool):");
        for r in refusals.iter().take(20) {
            eprintln!("#   {r}");
        }
    }
    if refused != 0 {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--census") {
        census(&args[1..]);
        return;
    }
    let mut a = args.into_iter();
    let bake_path = a.next().expect(
        "usage: serve_custom_bake <bake.bin> <ref> <dist> | --census <ref> <dist> [paths...]",
    );
    let ref_path = a.next().expect("ref image required");
    let dist_path = a.next().expect("dist image required");

    let bytes = std::fs::read(&bake_path).unwrap_or_else(|e| panic!("read {bake_path}: {e}"));
    println!("bake: {bake_path} ({} bytes)", bytes.len());
    match zenpredict::Model::from_bytes(&bytes) {
        Ok(m) => println!(
            "  declared: n_inputs={} caller_input_width={}",
            m.n_inputs(),
            m.caller_input_width()
        ),
        Err(e) => println!("  NOT a loadable ZNPR: {e:?}"),
    }
    let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
    let mut z = BakeScorer::new(&model).expect("invalid score metadata");

    let (r, w, h) = load_rgb(&ref_path);
    let (d, dw, dh) = load_rgb(&dist_path);
    assert_eq!((w, h), (dw, dh), "ref and dist must share dimensions");

    let rs = RgbSlice::new(&r, w as usize, h as usize);
    let ds = RgbSlice::new(&d, w as usize, h as usize);
    // The whole point: this is the PRODUCTION entry point, not a training one.
    match z.compute(&rs, &ds, None) {
        Ok(res) => println!(
            "SERVED  score={:.6}  raw_distance={:.6}  emitted={}",
            res.score(),
            res.raw_distance(),
            res.features().len()
        ),
        Err(e) => println!("REFUSED by Zensim::compute: {e:?}"),
    }
    match z.compute(&rs, &rs, None) {
        Ok(res) => println!("IDENTITY (ref vs ref) score={:.6}", res.score()),
        Err(e) => println!("IDENTITY REFUSED: {e:?}"),
    }
}
