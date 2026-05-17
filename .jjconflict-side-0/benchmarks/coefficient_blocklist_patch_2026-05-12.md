# Coefficient repo blocklist patch (for user to apply)

Per the 2026-05-12 purge, 361 hex-hashed source files were identified as perceptual near-duplicates of CID22 holdout references (d ≤ 16) and permanently deleted. To prevent regeneration during future training-data builds, the synth generator's CID22 blocklist needs to be extended.

**Target file** (outside this repo, in the `coefficient` workspace):
`coefficient/examples/generate_zensim_training.rs`

**Current state** (CID22_VALIDATION_49 constant): 41 numeric IDs + 8 non-numeric names. This catches filename matches for the 49 CID22 reference *stems*, but the contaminated training sources have HEX-HASHED names like `00b13be94a4867dd_1022x818.png` that don't match any stem.

**Required change**: extend the blocklist (or add a parallel constant) with the 361 hex stems from `benchmarks/contaminated_sources_purged_2026-05-12.txt`. Suggested diff:

```rust
// After CID22_VALIDATION_49, add:

/// Hex-hashed perceptual near-duplicates of CID22 validation references
/// identified by the 2026-05-12 dHash-64 purge audit (zensim repo,
/// `benchmarks/contaminated_sources_purged_2026-05-12.txt`). These
/// stems must NEVER appear as training sources — re-introducing them
/// would re-inflate V_X bake CID22 scores via training-set leakage.
const PURGED_HEX_STEMS_2026_05_12: &[&str] = &[
    "00b13be94a4867dd_1022x818",
    "00b13be94a4867dd_1024sq",
    "00b13be94a4867dd_512sq",
    "00b13be94a4867dd_513x769",
    "00b13be94a4867dd_769x513",
    // ... (full 361 stems from the purge manifest)
];

fn is_banned(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    for pattern in BANNED_PATH_PATTERNS {
        if path_str.contains(pattern) { return true; }
    }
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        // Existing 49-ref filename check
        static INIT_49: std::sync::OnceLock<HashSet<&'static str>> = std::sync::OnceLock::new();
        let banned_49 = INIT_49.get_or_init(|| CID22_VALIDATION_49.iter().copied().collect());
        if banned_49.contains(stem) { return true; }

        // NEW: 361 hex stems from 2026-05-12 purge
        static INIT_HEX: std::sync::OnceLock<HashSet<&'static str>> = std::sync::OnceLock::new();
        let banned_hex = INIT_HEX.get_or_init(|| PURGED_HEX_STEMS_2026_05_12.iter().copied().collect());
        if banned_hex.contains(stem) { return true; }

        // ... rest of existing checks
    }
    false
}
```

**Action needed**: the 361 stems are pre-formatted as a Rust constant array in `benchmarks/purged_hex_stems_const_2026-05-12.rs` — just `include!()` it or paste-in. Then apply the `is_banned()` diff above to `coefficient/examples/generate_zensim_training.rs`, run `cargo build -p coefficient --examples` to verify, commit.

**Drop-in commands**:
```bash
# From coefficient repo root:
cp ../zensim/benchmarks/purged_hex_stems_const_2026-05-12.rs examples/purged_hex_stems_const_2026-05-12.rs
# Add `mod purged_hex_stems_const_2026_05_12;` and `use purged_hex_stems_const_2026_05_12::PURGED_HEX_STEMS_2026_05_12;` to generate_zensim_training.rs, then apply the is_banned() diff.
```

**Why this matters**: the purge removed the source files but the generator still has the BANNED_PATH_PATTERNS list that ONLY catches the original 49 CID22 stems (`1418519`, `pexels-photo-1933873`, etc.). If anyone re-runs the generator against a new source corpus that includes the same hex-hashed crops (or new ones), they'd re-contaminate. The hex-stem blocklist is the durable fix.

This patch is staged here in the zensim repo for the user to apply when they next work in the coefficient repo. Per zensim's strict cross-repo rule, the patch isn't applied autonomously.
