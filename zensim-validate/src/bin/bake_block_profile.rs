//! bake_block_profile — static per-bake FEATURE-BLOCK usage fingerprint from
//! the bake bytes alone (no corpus, no scoring).
//!
//! Thin CLI over [`zensim_validate::block_profile::profile`], which owns the
//! semantics: per-family (f0-155 / f156-371 / f372-719 / f720-943) counts of
//! exactly-zero, near-zero and structurally-used **caller lines**. Families
//! are classified in CALLER feature space — a pruned bake's `Drop` transforms
//! map its internal layer-0 columns back to the caller numbering, so the
//! profile of a packed twin equals its unpruned parent's (see the module
//! docs for the caller-width bug this guards against, instance #4 of the
//! class).
//!
//! Usage: bake_block_profile --bake <bake.bin> [--json]
//! Exit: 0 ok; 2 usage/load/malformed-bake error.

use std::path::PathBuf;
use zenpredict::Model;
use zensim_validate::block_profile;

fn usage() -> ! {
    eprintln!("usage: bake_block_profile --bake <bake.bin> [--json]");
    std::process::exit(2);
}

fn main() {
    let mut bake: Option<PathBuf> = None;
    let mut json = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = args.next().map(PathBuf::from),
            "--json" => json = true,
            _ => usage(),
        }
    }
    let path = match bake {
        Some(p) => p,
        None => usage(),
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake_block_profile: read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let model = match Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bake_block_profile: load {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let prof = match block_profile::profile(&model) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bake_block_profile: profile {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    if json {
        println!("{}", prof.to_json());
    } else {
        print!("{}", prof.render_text(&path.display().to_string()));
    }
}
