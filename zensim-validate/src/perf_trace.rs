//! `ZENSIM_PERF_TRACE=1` phase timing — the coarse instrumentation that
//! answers "where did the wall clock go?" without a profiler.
//!
//! Why it is committed rather than a throwaway `eprintln!`: the 2026-08-04
//! eval-perf pass found that `bake_verdict`'s own "complete in N s" line
//! measured only the part of `main` up to the markdown assembly — the
//! `--full-json` per-pair block (a 2.5 GB parquet open) ran *after* the
//! timer stopped, so the binary under-reported its own wall time by ~2×.
//! A self-timer that lies about its own cost is exactly the kind of thing
//! that has to be measurable on demand, in-tree, on the next run.
//!
//! It is OFF unless `ZENSIM_PERF_TRACE=1`, and when off costs one relaxed
//! atomic load per mark, so instrumented paths stay byte-identical in
//! output and effectively identical in time.
//!
//! Usage:
//! ```ignore
//! use zensim_validate::perf_trace::PerfTrace;
//! let t = PerfTrace::new("bake_verdict");
//! t.mark("args parsed");
//! // ...
//! t.mark("corpora scored");
//! t.finish();
//! ```

use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ZENSIM_PERF_TRACE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// A phase timer. Cheap to construct; marks print only when the env flag
/// is set. Interior mutability via `std::sync::Mutex` keeps `mark` callable
/// through a shared reference (so it can live next to `&`-borrowed state in
/// a long `main`) without threading a `&mut` through every call site.
pub struct PerfTrace {
    label: &'static str,
    start: Instant,
    last: std::sync::Mutex<Instant>,
}

impl PerfTrace {
    pub fn new(label: &'static str) -> Self {
        let now = Instant::now();
        if enabled() {
            eprintln!("[perf {label}] +0.000s  (trace start)");
        }
        Self {
            label,
            start: now,
            last: std::sync::Mutex::new(now),
        }
    }

    /// Print `delta since previous mark` and `total since start`.
    pub fn mark(&self, what: &str) {
        if !enabled() {
            return;
        }
        let now = Instant::now();
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let d = now.duration_since(*last).as_secs_f64();
        let t = now.duration_since(self.start).as_secs_f64();
        *last = now;
        eprintln!("[perf {}] +{d:7.3}s  (t={t:7.3}s)  {what}", self.label);
    }

    /// Total elapsed since construction, whether or not tracing is on.
    pub fn total(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn finish(&self) {
        if !enabled() {
            return;
        }
        eprintln!(
            "[perf {}] TOTAL {:.3}s",
            self.label,
            self.start.elapsed().as_secs_f64()
        );
    }
}

/// Time one expression and mark it. Returns the expression's value.
#[macro_export]
macro_rules! perf_phase {
    ($trace:expr, $what:expr, $body:expr) => {{
        let __v = $body;
        $trace.mark($what);
        __v
    }};
}
