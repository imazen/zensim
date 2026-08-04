//! Shared rayon thread-pool policy for the eval binaries.
//!
//! This box is a 28-core WSL2 VM frequently shared by concurrent agents
//! (workspace CLAUDE.md, "MACHINE SAFETY"), so an eval that grabs every
//! core makes the machine unresponsive for whoever else is training. The
//! policy here is the same one `~/work/zen/scripts/run-heavy` applies:
//! leave a few cores free unless told otherwise.
//!
//! Precedence, highest first:
//! 1. `RAYON_NUM_THREADS` — rayon's own env var. If set we do nothing and
//!    let rayon configure itself, so `run-heavy` (which exports it) stays
//!    authoritative.
//! 2. `ZENSIM_THREADS` — this repo's override, same meaning.
//! 3. Default: `available_parallelism() − 4`, floored at 1.
//!
//! Parallelism NEVER changes a number. Every use in these binaries splits
//! work whose units are independent (one row's forward, one corpus's
//! panel, one file's sha256) and re-assembles them in the original index
//! order, so the output is bit-identical to the sequential run — which is
//! gated by `scripts/verify_verdict_identity.sh`.

use std::sync::OnceLock;

/// Thread count this process should use for its own rayon pool, or `None`
/// when `RAYON_NUM_THREADS` is set (rayon configures itself).
fn requested_threads() -> Option<usize> {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return None;
    }
    if let Ok(v) = std::env::var("ZENSIM_THREADS")
        && let Ok(n) = v.trim().parse::<usize>()
        && n > 0
    {
        return Some(n);
    }
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    Some(avail.saturating_sub(4).max(1))
}

/// Install the global rayon pool once, honoring the policy above.
///
/// Safe to call repeatedly and from anywhere; the second and later calls
/// are no-ops. Failure to install (e.g. another crate got there first) is
/// deliberately ignored — the work still runs, just on rayon's default
/// pool.
pub fn init() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if let Some(n) = requested_threads() {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
    });
}

/// Effective worker count (for logs / benchmark headers).
pub fn threads() -> usize {
    init();
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default must leave headroom, never claim the whole box.
    #[test]
    fn default_policy_leaves_cores_free() {
        // Only meaningful when neither env var is set; the test runner may
        // set RAYON_NUM_THREADS, in which case the policy defers to it.
        if std::env::var_os("RAYON_NUM_THREADS").is_some()
            || std::env::var_os("ZENSIM_THREADS").is_some()
        {
            return;
        }
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let want = requested_threads().expect("no RAYON_NUM_THREADS in this branch");
        assert!(want >= 1);
        assert!(
            want <= avail.saturating_sub(4).max(1),
            "policy must leave >= 4 cores free on a multicore box (avail={avail}, want={want})"
        );
    }
}
