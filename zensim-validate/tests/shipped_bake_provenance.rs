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

/// A seed is not a recipe: no two manifests may carry the same `[training]`
/// recipe differing only in `seed`.
///
/// **This is the "prohibited" half of "kill duplication and prohibit it".**
/// The trainer merges a manifest as DEFAULTS and lets explicit CLI flags win
/// (`--manifest foo.toml --seed 99` reproduces foo at seed 99 —
/// `zensim_mlp_train.rs:1013`), so a `<recipe>_s<N>.toml` family is pure
/// duplication of a thing the CLI already does.
///
/// It was not hypothetical. On 2026-07-15: **144 manifests carried 58 distinct
/// recipes.** `w7_guard` alone was 16 files for one recipe across 16 seeds;
/// `w3ctl_t1dro51` was 11. 76 were collapsed away. The forks were generated by
/// `scripts/v_next/make_manifest.py` — a tool written to PREVENT copy-errors,
/// which recomputed every `[inputs.*]` hash while carrying the base's
/// `[bake].sha256` / `[eval]` through untouched. 128 manifests ended up
/// claiming the shipped Profile A bake and Profile A's scores.
///
/// Two legitimate exemptions, both measured rather than assumed:
///   * **Different `trainer_commit`** — `v52`/`v53` ran their seeds on
///     different trainer builds (v52: 4022296c vs 5a89534b). Reproduce-exactly
///     requires building the trainer at the recorded commit, and one manifest
///     cannot record two builds. `trainer_commit` is part of the recipe key
///     below, so these do not collide.
///   * **Different `[inputs]`** — `v51box` shares v51's recipe but points at
///     box-local `/data` paths. A deployment fork. Inputs are in the key too.
///
/// If this fails: you almost certainly want `--seed <N>` on the existing
/// manifest, not a new file.
///
/// # Why a NAMING rule and not a recipe hash
///
/// Hashing the parsed recipe and flagging collisions was tried first and is
/// wrong — it fires on four things that are all legitimate:
///
///   * `v47_mainfix_repro` ≡ `v47_strict_qat` — recipe-identical **on
///     purpose**: it is the reproduction that proved v47 rebuilds
///     byte-identically. Under a recipe hash, the artifact that PROVES
///     reproducibility reads as duplication.
///   * `v51` ≡ `v51box_s17` — box-local `/data` input paths (deployment fork).
///   * `ab_hardpair` ≡ `w2_hponly` — one recipe re-run under a new experiment
///     label; merging would break the identity benchmark docs cite.
///   * Every manifest with no `[training]` collides with every other, because
///     they all key to the empty recipe — which would have "deduplicated" one
///     of the two shipped-bake manifests (B / BHdr).
///
/// The thing that actually proliferated is a *naming* pattern —
/// `<stem>_s<N>.toml` — and 76 of 144 manifests were it. So that is what this
/// prohibits: unambiguous, no false positives, and it fails exactly when
/// someone reaches for a new file instead of `--seed`.
#[test]
fn no_seed_fork_manifest_families() {
    use std::collections::HashMap;

    // Families whose per-seed split is REAL and must not collapse. Each entry
    // needs a measured reason, not a preference.
    //
    //   v52 / v53: their seeds were trained by DIFFERENT trainer builds
    //   (v52: 4022296c vs 5a89534b; v53: c567cb69 vs 5a89534b). Reproduce-
    //   exactly requires building the trainer at the recorded commit, and one
    //   manifest cannot record two. Collapsing these would silently claim a
    //   build made a bake it never made.
    const EXEMPT: &[&str] = &["v52", "v53"];

    let dir = weights_dir().join("manifests");
    let mut fams: HashMap<String, Vec<String>> = HashMap::new();
    for entry in std::fs::read_dir(&dir).expect("read manifests dir") {
        let p = entry.expect("dir entry").path();
        if p.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
        // `<anything>_s<digits>` at the end of the stem.
        let Some(idx) = stem.rfind("_s") else {
            continue;
        };
        let suffix = &stem[idx + 2..];
        if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let base = stem[..idx].to_string();
        if EXEMPT.contains(&base.as_str()) {
            continue;
        }
        fams.entry(base).or_default().push(format!("{stem}.toml"));
    }

    // A LONE `_sN` file is not a fork — nothing was duplicated. Only a family
    // (or one that shadows an existing base recipe) is.
    let dupes: Vec<_> = fams
        .iter()
        .filter(|(base, v)| v.len() > 1 || dir.join(format!("{base}.toml")).exists())
        .map(|(base, v)| {
            let mut s = v.clone();
            s.sort();
            format!("  {base}: {} file(s) — {}", s.len(), s.join(", "))
        })
        .collect();

    assert!(
        dupes.is_empty(),
        "seed-fork manifest families are back ({}):\n{}\n\n\
         A seed is not a recipe. `--manifest <recipe>.toml --seed <N> --out <path>.bin` \
         already does exactly this — the manifest supplies DEFAULTS and explicit CLI flags \
         win (zensim_mlp_train.rs:1013). Delete the fork and pass --seed.\n\n\
         This is not hypothetical: 144 manifests carried 58 recipes until 2026-07-15. \
         w7_guard alone was 16 files for one recipe; 76 were collapsed away. They were \
         generated by scripts/v_next/make_manifest.py — a tool built to PREVENT \
         copy-errors, which carried the base's [bake].sha256 and [eval] through untouched \
         until 128 manifests all claimed the shipped Profile A bake.",
        dupes.len(),
        dupes.join("\n")
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
