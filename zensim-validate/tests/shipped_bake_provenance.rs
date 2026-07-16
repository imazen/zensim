//! Every bake we actually ship must have a manifest that identifies it.
//!
//! # Why this test exists
//!
//! `zensim/weights/manifests/README.md` already states the rule and names
//! the incident it exists to prevent:
//!
//! > *"Every shipped `.bin` bake under `zensim/weights/` should have a
//! > sibling `.toml` manifest … every shipped bake is bit-exactly
//! > reproducible from the manifest."*
//! >
//! > *"**Why:** the V32 recipe-archaeology incident — V32's commit message
//! > documented hyperparameters but not the exact CLI … A future agent
//! > reconstructing it from documentation alone got `CID22 0.295`
//! > (held-out, ~0.59 below V32's documented 0.8879)."*
//!
//! The rule was written, the schema was designed, the parser was built —
//! and nothing ever checked it. As of 2026-07-15, measured:
//!
//! - **`[bake]` was never parsed.** `RawBake` carried only `file`; serde
//!   silently dropped `sha256` and `file_bytes`. `verify_inputs()` checked
//!   every *input* hash; no code read the *output* hash. The field was
//!   inert prose in a TOML costume.
//! - **128 of 142 manifests recorded the same `sha256`** (`d0ef7a30…` — the
//!   shipped Profile A bake) **and the same `[eval]`** (`cid22_srocc =
//!   0.8657` — Profile A's number). Every experiment was forked from
//!   `v47_strict_qat.toml` and kept the outcome fields verbatim. A
//!   manifest's whole job is to identify a bake; 128 identified someone
//!   else's.
//! - **Neither shipped SDR/HDR bake had a manifest at all**, though the
//!   README mandates one per shipped bake.
//!
//! This test is the missing enforcement. It is deliberately scoped to
//! **in-repo bakes only** — no `/mnt/v`, no network — so it runs on every
//! CI platform.
//!
//! # Scope: the bakes the crate actually ships
//!
//! "Shipped" means reachable at runtime, i.e. named by an `include_bytes!`
//! in `zensim/src/profile.rs`. That list is parsed from the source rather
//! than hardcoded here, so adding a profile without a manifest fails this
//! test instead of silently widening the gap. Probe bakes on `/mnt/v` are
//! out of scope: they are not shipped, and CI does not mount that volume.

use std::path::{Path, PathBuf};
use zensim_validate::train_manifest;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/zensim-validate
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("zensim-validate has a parent dir")
        .to_path_buf()
}

fn weights_dir() -> PathBuf {
    repo_root().join("zensim/weights")
}

/// The bake filenames `profile.rs` actually `include_bytes!`s.
///
/// Parsed from source, not hardcoded: a new profile whose bake lacks a
/// manifest must FAIL this test, and a hardcoded list would let it pass.
/// Doc-comment examples are excluded — they show the pattern to users and
/// name a bake without shipping it.
fn shipped_bake_files() -> Vec<String> {
    let src = repo_root().join("zensim/src/profile.rs");
    let text =
        std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("read {}: {e}", src.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        // Skip doc examples (`/// fn my_bake() -> … include_bytes!(…)`).
        if trimmed.starts_with("//") {
            continue;
        }
        let Some(rest) = line.split_once("include_bytes!(\"") else {
            continue;
        };
        let Some((path, _)) = rest.1.split_once('"') else {
            continue;
        };
        if let Some(name) = path.rsplit('/').next() {
            out.push(name.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Index every manifest by the bake path its `[bake].file` resolves to.
///
/// Keyed on the resolved path rather than the filename stem: the README's
/// naming convention (`<bake>.bin` → `<bake>.toml`) is not universally
/// followed — `v47_strict_qat.toml` describes
/// `v47_strict_qat_native_2026-05-27.bin` — and what matters is which bake
/// a manifest *points at*, not what it is called.
fn manifests_by_bake() -> Vec<(PathBuf, PathBuf, Option<String>, Option<u64>)> {
    let dir = weights_dir().join("manifests");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let p = entry.expect("dir entry").path();
        if p.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let cfg = match train_manifest::parse_manifest(&p) {
            Ok(c) => c,
            Err(e) => panic!(
                "manifest {} failed to parse: {e}\nEvery .toml under \
                 zensim/weights/manifests/ must parse with the canonical Rust parser \
                 (train_manifest::parse_manifest). If this file is not a manifest, move it out.",
                p.display()
            ),
        };
        if let Some(out_path) = cfg.out.clone() {
            out.push((p, out_path, cfg.bake_sha256.clone(), cfg.bake_file_bytes));
        }
    }
    out
}

#[test]
fn every_shipped_bake_has_a_manifest_that_identifies_it() {
    let shipped = shipped_bake_files();
    assert!(
        !shipped.is_empty(),
        "parsed zero include_bytes! bakes out of zensim/src/profile.rs — the parser in this \
         test has drifted from the source. Fix the parser; do NOT weaken the assertion."
    );

    let manifests = manifests_by_bake();
    let mut failures: Vec<String> = Vec::new();

    for bake_name in &shipped {
        let bake_path = weights_dir().join(bake_name);
        assert!(
            bake_path.exists(),
            "profile.rs include_bytes!s {bake_name} but {} does not exist",
            bake_path.display()
        );

        // A manifest "describes" this bake when its [bake].file resolves to it.
        let matches: Vec<_> = manifests
            .iter()
            .filter(|(_, out, _, _)| out.canonicalize().ok() == bake_path.canonicalize().ok())
            .collect();

        if matches.is_empty() {
            failures.push(format!(
                "  {bake_name}\n      NO MANIFEST. zensim/weights/manifests/README.md requires one \
                 per shipped bake so the bake is bit-exactly reproducible. Add a .toml whose \
                 [bake].file points at ../{bake_name} with its true sha256 + file_bytes."
            ));
            continue;
        }

        for (mpath, _, sha, bytes) in matches {
            let mname = mpath.file_name().unwrap().to_string_lossy();
            let Some(sha) = sha else {
                failures.push(format!(
                    "  {bake_name}\n      manifest {mname} has no [bake].sha256 — it cannot \
                     identify the bake it describes."
                ));
                continue;
            };
            let actual = train_manifest::sha256_file(&bake_path).expect("hash shipped bake");
            if &actual != sha {
                failures.push(format!(
                    "  {bake_name}\n      manifest {mname} records sha256 {} but the bake is {}.\n\
                     \x20     A manifest that names the wrong bytes is worse than none: it reads as \
                     provenance. This is the fork signature — recompute the outcome fields from the \
                     bake this manifest actually produces.",
                    &sha[..16.min(sha.len())],
                    &actual[..16]
                ));
            }
            if let Some(b) = bytes {
                let real = std::fs::metadata(&bake_path).expect("stat bake").len();
                if *b != real {
                    failures.push(format!(
                        "  {bake_name}\n      manifest {mname} records file_bytes {b} but the bake \
                         is {real} bytes."
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "shipped bakes without correct provenance ({} problem(s)):\n{}\n\n\
         Shipped bakes parsed from profile.rs: {:?}\n\
         Do NOT fix this by deleting the assertion. The README explains what it buys: without a \
         verified manifest, reconstructing a recipe from prose alone cost CID22 0.295 vs a \
         documented 0.8879 (the V32 recipe-archaeology incident).",
        failures.len(),
        failures.join("\n"),
        shipped
    );
}

/// No manifest anywhere in the directory may misdescribe a bake that is
/// present on disk.
///
/// Absence is skipped, not failed: probe manifests point at `/mnt/v`, which
/// CI does not mount. So on CI this checks the in-repo bakes; run locally it
/// also covers every probe bake that happens to be present — which is how
/// the 128 copy-pasted `d0ef7a30…` hashes surface.
#[test]
fn no_manifest_misdescribes_a_bake_that_exists() {
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (mpath, out, _, _) in manifests_by_bake() {
        if !out.exists() {
            continue; // /mnt/v probe bake, or a bake that was never kept
        }
        checked += 1;
        let cfg = train_manifest::parse_manifest(&mpath).expect("reparse");
        if let Err(e) = train_manifest::verify_bake(&cfg, false) {
            bad.push(format!(
                "  {}: {e}",
                mpath.file_name().unwrap().to_string_lossy()
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} on-disk bakes are misdescribed by their manifest:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
