//! Serve an arbitrary ZNPR bake through the PRODUCTION scoring path
//! (`Zensim::compute`), so "is this candidate servable?" is a MEASUREMENT
//! rather than an inference from reading `profile.rs`.
//!
//! Written for `benchmarks/fastclass2_campaign_2026-09-05.md` gate G7. The
//! kernel lane (`benchmarks/kernel_fastclass_2026-09-05.md` §4 and commit
//! `8817f379`) established that `Zensim::compute` emits a **372-layout**
//! vector with `free_extras: Off`, so a 944-declared bake is refused and a
//! 156/228-slice bake at the v1-372 layout should serve. This example checks
//! the second half on real pixels instead of taking it on trust — the
//! `d_ship_flip` lane found the 944 refusal exactly this way.
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
//! Walks many bakes through the SAME `Zensim::compute` entry the single-bake
//! mode uses — one implementation, not two — and prints a TSV row per bake
//! plus a SERVED/REFUSED summary. This is the filesystem tier of the
//! servability contract (user directive 2026-09-05: *"also make sure
//! everything can be served"*); the no-filesystem tier is the in-lib census
//! (`serving::tests::every_shipped_profile_is_servable`), which gates every SHIPPED profile and
//! every registered producer set on a build with no `/mnt/v`.
//!
//! A REFUSED row is the contract failing, not the tool: every bake whose read
//! set is registered feature ids at a supported revision must serve.

use std::sync::RwLock;
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

// A `ProfileParams` needs an `fn() -> &'static [u8]`, which cannot capture —
// so the census parks the current bake here and scores sequentially. A
// `OnceLock` (the single-bake original) can only ever hold one.
static CUR: RwLock<Option<&'static [u8]>> = RwLock::new(None);
fn bake_bytes() -> &'static [u8] {
    CUR.read().expect("poisoned").expect("bake set before use")
}
fn set_bake(bytes: Vec<u8>) {
    // Leaked deliberately: the profile's `fn` pointer hands out `'static`.
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    *CUR.write().expect("poisoned") = Some(leaked);
}

/// The custom profile that wraps whatever bake is currently parked in [`CUR`].
/// One construction, shared by both modes.
fn custom_profile() -> ZensimProfile {
    let params: &'static ProfileParams = Box::leak(Box::new(
        ProfileParams::builder()
            .mlp(bake_bytes)
            .extended_features(true)
            .compute_iw_features(true)
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .build(),
    ));
    ZensimProfile::Custom {
        name: "servability-census",
        params,
    }
}

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
    let declared = match zenpredict::Model::from_bytes(&bytes) {
        Ok(m) => m.caller_input_width(),
        Err(e) => return (false, format!("{path}\t-\t-\tNOT_A_ZNPR\t{e:?}")),
    };
    set_bake(bytes);
    let z = Zensim::new(custom_profile());
    match z.compute(rs, ds) {
        Ok(res) => (
            true,
            format!(
                "{path}\t{declared}\t{}\tSERVED\t{:.6}",
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
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        eprintln!("# fulleval dir unreadable: {dir}");
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if !p.to_string_lossy().ends_with(".fulleval.json") {
            continue;
        }
        let Ok(txt) = std::fs::read_to_string(&p) else {
            continue;
        };
        // Narrow scan: take the first `"bake":` whose value is a PATH. A
        // fulleval can carry a nested `"bake": { ... }` object (the board
        // census found one), and a naive first-match reads its first key as
        // the path — so require a `/`.
        for rest in txt.split("\"bake\":").skip(1) {
            let Some(a) = rest.find('"') else { continue };
            let Some(b) = rest[a + 1..].find('"') else {
                continue;
            };
            let v = &rest[a + 1..a + 1 + b];
            if v.contains('/') {
                out.push(v.to_string());
                break;
            }
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
    set_bake(bytes);
    let z = Zensim::new(custom_profile());

    let (r, w, h) = load_rgb(&ref_path);
    let (d, dw, dh) = load_rgb(&dist_path);
    assert_eq!((w, h), (dw, dh), "ref and dist must share dimensions");

    let rs = RgbSlice::new(&r, w as usize, h as usize);
    let ds = RgbSlice::new(&d, w as usize, h as usize);
    // The whole point: this is the PRODUCTION entry point, not a training one.
    match z.compute(&rs, &ds) {
        Ok(res) => println!(
            "SERVED  score={:.6}  raw_distance={:.6}  emitted={}",
            res.score(),
            res.raw_distance(),
            res.features().len()
        ),
        Err(e) => println!("REFUSED by Zensim::compute: {e:?}"),
    }
    match z.compute(&rs, &rs) {
        Ok(res) => println!("IDENTITY (ref vs ref) score={:.6}", res.score()),
        Err(e) => println!("IDENTITY REFUSED: {e:?}"),
    }
}
