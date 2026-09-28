//! Tier-parity audit for `zensim::research::extract` — MEASUREMENT ONLY.
//!
//! Runs the full-width research extraction (all families, current revision)
//! on real TRAIN image pairs under each forcible SIMD tier by disabling
//! archmage tokens process-wide:
//!
//!   v4x    — nothing disabled (best token wins; V4x where consulted)
//!   v4     — X64V4xToken disabled
//!   v3     — X64V4xToken + X64V4Token disabled (canonical AVX2 arithmetic)
//!   scalar — every x86 vector token disabled (incant falls to scalar)
//!
//! Compares each tier's `Extraction::values()` against the v3 baseline
//! bit-for-bit and by max |relative difference|, then rolls up per feature
//! block (`provenance.n`) so the report can name the kernel responsible.
//!
//! Usage: `tier_audit_features <pairs.tsv>` where the TSV has a header
//! `ref_path<TAB>dist_path...` (the Rev4-featbank pair-list shape), or
//! `tier_audit_features --pair <label> <ref> <dist>` repeated.
//!
//! Prints a per-(pair, tier, block) table plus a per-(tier, block) rollup
//! across pairs to stdout; write `TIER_AUDIT_OUT=<path>` to also emit TSV.

use std::fmt::Write as _;
use std::process::exit;

use zensim::RgbSlice;
use zensim::research::{self, Request};

fn decode_rgb(path: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let img = image::open(path).unwrap_or_else(|e| panic!("decode {path}: {e}"));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    let (chunks, rem) = raw.as_chunks::<3>();
    assert!(rem.is_empty(), "{path}: RGB pixel count not divisible by 3");
    (chunks.to_vec(), w, h)
}

struct Tier {
    label: &'static str,
    /// Tokens to force OFF for this measurement.
    disable: &'static [&'static str],
}

const TIERS: &[Tier] = &[
    Tier {
        label: "v4x",
        disable: &[],
    },
    Tier {
        label: "v4",
        disable: &["x86-64-v4x"],
    },
    Tier {
        label: "v3",
        disable: &["x86-64-v4x", "x86-64-v4"],
    },
    Tier {
        label: "scalar",
        disable: &[
            "x86-64-v4x",
            "x86-64-v4",
            "x86-64-v3 Crypto",
            "x86-64-v3",
            "x86-64 Crypto",
            "x86-64-v2",
        ],
    },
];

/// Toggle one named token-disable flag. Names match
/// `archmage::testing::for_each_token_permutation`'s labels.
fn set_token_disabled(name: &str, off: bool) {
    use archmage::{
        X64CryptoToken, X64V2Token, X64V3CryptoToken, X64V3Token, X64V4Token, X64V4xToken,
    };
    let r = match name {
        "x86-64-v4x" => X64V4xToken::dangerously_disable_token_process_wide(off),
        "x86-64-v4" => X64V4Token::dangerously_disable_token_process_wide(off),
        "x86-64-v3 Crypto" => X64V3CryptoToken::dangerously_disable_token_process_wide(off),
        "x86-64-v3" => X64V3Token::dangerously_disable_token_process_wide(off),
        "x86-64 Crypto" => X64CryptoToken::dangerously_disable_token_process_wide(off),
        "x86-64-v2" => X64V2Token::dangerously_disable_token_process_wide(off),
        _ => panic!("unknown token group {name}"),
    };
    if let Err(e) = r {
        // A compile-time-guaranteed token cannot be disabled — on this build
        // (no target-cpu pinning) that never happens; surface it loudly anyway.
        panic!("cannot set {name} disabled={off}: {e}");
    }
}

fn apply_tier(tier: &Tier) {
    // Re-enable everything first, then apply this tier's disable set —
    // keeps each measurement independent of ordering.
    for name in [
        "x86-64-v4x",
        "x86-64-v4",
        "x86-64-v3 Crypto",
        "x86-64-v3",
        "x86-64 Crypto",
        "x86-64-v2",
    ] {
        set_token_disabled(name, false);
    }
    for &name in tier.disable {
        set_token_disabled(name, true);
    }
}

fn main() {
    let mut pairs: Vec<(String, String, String)> = Vec::new(); // (label, ref, dist)
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pair" => {
                pairs.push((
                    args[i + 1].clone(),
                    args[i + 2].clone(),
                    args[i + 3].clone(),
                ));
                i += 4;
            }
            tsv => {
                let text =
                    std::fs::read_to_string(tsv).unwrap_or_else(|e| panic!("read {tsv}: {e}"));
                for (n, line) in text.lines().enumerate().skip(1) {
                    let c: Vec<&str> = line.split('\t').collect();
                    if c.len() >= 2 && !c[0].is_empty() {
                        let set = tsv
                            .rsplit('/')
                            .next()
                            .unwrap_or(tsv)
                            .trim_end_matches(".tsv");
                        let label = c
                            .get(2)
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("{set}#{n}"));
                        pairs.push((label, c[0].to_string(), c[1].to_string()));
                    }
                }
                i += 1;
            }
        }
    }
    if pairs.is_empty() {
        eprintln!("no pairs given");
        exit(2);
    }

    // Sanity: confirm the host can actually summon v4/v3 — else the audit
    // silently measures degenerate tiers.
    {
        use archmage::SimdToken;
        eprintln!(
            "tokens: v4x={} v4={} v3={}",
            archmage::X64V4xToken::summon().is_some(),
            archmage::X64V4Token::summon().is_some(),
            archmage::X64V3Token::summon().is_some(),
        );
    }

    let mut out = String::new();
    writeln!(
        out,
        "pair\ttier\tn_diff\tmax_abs_diff\tmax_rel_diff\tblock\tslot\tslot_name"
    )
    .unwrap();

    for (label, refp, distp) in &pairs {
        let (rp, rw, rh) = decode_rgb(refp);
        let (dp, dw, dh) = decode_rgb(distp);
        assert_eq!((rw, rh), (dw, dh), "{label}: ref/dist size mismatch");
        let (rs, rd) = (RgbSlice::new(&rp, rw, rh), RgbSlice::new(&dp, dw, dh));
        eprintln!("== {label} {rw}x{rh} == {refp}");

        // v3 baseline first — canonical arithmetic.
        let mut per_tier: Vec<(
            &'static str,
            Vec<f64>,
            Vec<zensim::research::FeatureProvenance>,
        )> = Vec::new();
        for tier in TIERS {
            apply_tier(tier);
            let e = research::extract(&Request::everything(), &rs, &rd)
                .unwrap_or_else(|e| panic!("{label} tier {}: {e}", tier.label));
            per_tier.push((tier.label, e.values().to_vec(), e.provenance().to_vec()));
        }
        let base = &per_tier.iter().find(|t| t.0 == "v3").unwrap().1;
        let prov = &per_tier[0].2;

        for (tlabel, vals, _) in &per_tier {
            if *tlabel == "v3" {
                continue;
            }
            let mut n_diff = 0usize;
            let mut worst_rel = 0.0f64;
            let mut worst_slot = usize::MAX;
            for (idx, (a, b)) in vals.iter().zip(base.iter()).enumerate() {
                if a.to_bits() == b.to_bits() {
                    continue;
                }
                let abs = (a - b).abs();
                let rel = abs / b.abs().max(1e-300);
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_slot = idx;
                }
                n_diff += 1;
                writeln!(
                    out,
                    "{label}\t{tlabel}\t1\t{abs:.6e}\t{rel:.6e}\t{}\t{}\t{}",
                    prov[idx].family, prov[idx].id, prov[idx].name
                )
                .unwrap();
            }
            eprintln!(
                "   {tlabel}: {n_diff}/{} slots differ vs v3; worst rel {worst_rel:.3e} {}",
                vals.len(),
                if worst_slot == usize::MAX {
                    String::new()
                } else {
                    format!("at f{} {}", prov[worst_slot].id, prov[worst_slot].name)
                }
            );
        }
    }

    print!("{out}");
}
