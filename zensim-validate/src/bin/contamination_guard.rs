//! CLI wrapper around `crate::contamination_guard::scrub_csv_or_die`.
//!
//! Probes one or more CSVs; exits 2 on detection, 1 on read error,
//! 0 on all-clean. Use as a standalone smoke test before kicking off
//! a long training run.

#[path = "../contamination_guard.rs"]
mod contamination_guard;

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    if args.len() == 0 {
        eprintln!("usage: contamination_guard <CSV> [CSV...]");
        return ExitCode::from(1);
    }
    let mut had_err = false;
    for p in args.by_ref() {
        if let Err(e) = contamination_guard::scrub_csv_or_die(&p) {
            eprintln!("read {p}: {e}");
            had_err = true;
        }
    }
    if had_err {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}
