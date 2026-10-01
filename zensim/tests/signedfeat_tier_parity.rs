// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Tier-parity gate on REAL TRAIN-role pixels** for the families that run on the v1 band side pass
//! (`mapdev`, `z1max`) and, as they land, the signedfeat families.
//!
//! Inputs are raw RGB8 files written by `scripts/signedfeat/prep_parity_pairs.py` into the directory
//! named by `SIGNEDFEAT_PARITY_DIR` (6 non-identical pairs from KADID-train, TID2013, KonFiG-train at
//! 64², 256², 1024² and 2112²). The caller names the directory; an unset variable is a hard failure,
//! not a skip. `SIGNEDFEAT_PARITY_FAMILIES` (comma list of compute tokens, default `mapdev,z1max`)
//! selects the slots compared.
//!
//! Compiled only with `RUSTFLAGS='--cfg signedfeat_real_pairs'` (the caller's explicit decision; `just signedfeat-tier-parity`
//! wires it), because the inputs live outside the repository.
//!
//! Own test executable (`for_each_token_permutation` mutates process-wide dispatch); the body runs in a
//! child at `ZENSIM_FORMULA_REV=4`, the same protocol as `featcanon_tier_parity.rs`.

#![cfg(all(
    feature = "training",
    feature = "feature-regime-v2",
    signedfeat_real_pairs
))]
#![allow(deprecated)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::RgbSlice;
use zensim::feature_set_id::{ComputeToken, SlotSet};
use zensim::research::{self, Request, family_slots};

const SENTINEL: &str = "SIGNEDFEAT_TIER_PARITY_OK";
const TEST: &str = "rev4_family_slots_bit_identical_on_real_pairs";

fn families() -> Vec<ComputeToken> {
    std::env::var("SIGNEDFEAT_PARITY_FAMILIES")
        .unwrap_or_else(|_| "mapdev,z1max".into())
        .split(',')
        .map(|t| ComputeToken::parse(t).unwrap_or_else(|| panic!("unknown token {t:?}")))
        .collect()
}

#[test]
fn rev4_family_slots_bit_identical_on_real_pairs() {
    // `SIGNEDFEAT_PARITY_REV` (default 4) exists for the negative control: at 3 the same bodies must
    // diverge across tiers, which proves the Rev4 pass is not vacuous.
    let rev = std::env::var("SIGNEDFEAT_PARITY_REV").unwrap_or_else(|_| "4".into());
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok(rev.as_str()) {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([TEST, "--exact", "--nocapture", "--test-threads=1"])
            .env("ZENSIM_FORMULA_REV", &rev)
            .output()
            .expect("re-exec");
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success(), "child failed");
        assert!(String::from_utf8_lossy(&out.stdout).contains(SENTINEL));
        return;
    }
    let dir = std::env::var("SIGNEDFEAT_PARITY_DIR")
        .expect("SIGNEDFEAT_PARITY_DIR must name the prep directory");
    let index = std::fs::read_to_string(format!("{dir}/index.tsv")).expect("index.tsv");
    let toks = families();
    let want = toks
        .iter()
        .fold(SlotSet::default(), |a, &t| a.union(&family_slots(t)));
    let width = research::full_width();
    let ids: Vec<usize> = (0..width).filter(|&i| want.contains(i)).collect();
    assert!(!ids.is_empty());
    let mut total_cells = 0usize;
    let mut tiers = 0usize;
    for line in index.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (name, n): (&str, usize) = (f[0], f[3].parse().unwrap());
        let rd = |suffix: &str| -> Vec<[u8; 3]> {
            let b = std::fs::read(format!("{dir}/{name}_{suffix}.rgb")).expect("rgb");
            assert_eq!(b.len(), n * n * 3);
            b.as_chunks::<3>().0.to_vec()
        };
        let (src, dst) = (rd("ref"), rd("dst"));
        assert_ne!(src, dst, "{name}: identical pair");
        let (s, d) = (RgbSlice::new(&src, n, n), RgbSlice::new(&dst, n, n));
        let mut runs: Vec<(String, Vec<u64>)> = Vec::new();
        let req = Request::for_slots(want.clone(), width);
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let e = research::extract(&req, &s, &d).expect("extract");
            let v = e.values();
            runs.push((
                perm.label.clone(),
                ids.iter().map(|&i| v[i].to_bits()).collect(),
            ));
        });
        assert!(runs.len() >= 3, "{name}: only {} permutations", runs.len());
        tiers = tiers.max(runs.len());
        let mut diff_slots = std::collections::BTreeSet::new();
        for (label, v) in &runs[1..] {
            for (k, (a, b)) in runs[0].1.iter().zip(v).enumerate() {
                if a != b {
                    diff_slots.insert(ids[k]);
                    let _ = label;
                }
            }
        }
        total_cells += ids.len() * runs.len();
        eprintln!(
            "{name} {n}²: {} permutations, {} slots, {} differing slots{}",
            runs.len(),
            ids.len(),
            diff_slots.len(),
            if diff_slots.is_empty() {
                String::new()
            } else {
                format!(" (first f{})", diff_slots.iter().next().unwrap())
            }
        );
        assert!(
            diff_slots.is_empty(),
            "{name}: tier divergence: {diff_slots:?}"
        );
    }
    eprintln!("families {toks:?}: {total_cells} cells compared, up to {tiers} permutations");
    println!("{SENTINEL}");
}
