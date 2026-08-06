//! The band scheme has exactly ONE owner (`zensim_validate::bands`) and one
//! gated mirror (`scripts/band_reliability.py::scheme_merge`, which needs it to
//! compare candidate schemes). This test pins both to a committed fixture so
//! they cannot drift apart silently — the failure mode the no-duplication rule
//! exists to prevent, and the one that let `bake_verdict`'s inlined stat copy
//! report a wrong OR/PWRC for months while the `panel` binary reported the
//! right one on the same data.
//!
//! The fixture (`benchmarks/appendixV/band_scheme_parity.tsv`) is the contract:
//! decile occupancy in, band labels out. `band_reliability.py selfcheck` asserts
//! the same file from the Python side.

use zensim_validate::bands;

/// Rebuild a corpus with the given per-decile occupancy, filling each decile
/// across (almost) its full width so realised spans match production.
fn corpus(counts: &[usize], top_span: Option<f64>) -> Vec<f64> {
    let mut v = Vec::new();
    for (d, &n) in counts.iter().enumerate() {
        let lo = d as f64 / 10.0;
        // A short top band (CID22's MOS stops at 0.9194) is the whole reason
        // this scheme exists, so the fixture must be able to express it.
        let width = if d == counts.len() - 1 {
            top_span.unwrap_or(0.0999)
        } else {
            0.0999
        };
        for i in 0..n {
            let f = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            v.push(lo + width * f);
        }
    }
    v
}

fn fixture_path() -> String {
    format!(
        "{}/../benchmarks/appendixV/band_scheme_parity.tsv",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn owner_reproduces_the_committed_parity_fixture() {
    let text = std::fs::read_to_string(fixture_path())
        .unwrap_or_else(|e| panic!("parity fixture must exist in-repo: {e}"));
    let mut checked = 0usize;
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            f.len(),
            4,
            "fixture row must be `name<TAB>decile-counts<TAB>top-span<TAB>expected-labels`: {line}"
        );
        let (name, counts, top_span, expect) = (f[0], f[1], f[2], f[3]);
        let counts: Vec<usize> = counts.split(',').map(|x| x.parse().unwrap()).collect();
        assert_eq!(counts.len(), bands::BASE_BANDS, "{name}: need 10 deciles");
        let top: Option<f64> = if top_span == "-" {
            None
        } else {
            Some(top_span.parse().unwrap())
        };
        let t = corpus(&counts, top);
        let got: Vec<String> = bands::merged_bands(&t)
            .into_iter()
            .map(|b| b.label)
            .collect();
        assert_eq!(got.join(","), expect, "{name}: band labels changed");

        // Every emitted band must be usable, or be the single-band collapse
        // the caller reports as NOT-MEASURED.
        let defs = bands::merged_bands(&t);
        for b in &defs {
            let m = b.members(&t);
            let span = if m.is_empty() {
                0.0
            } else {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for &i in &m {
                    lo = lo.min(t[i]);
                    hi = hi.max(t[i]);
                }
                hi - lo
            };
            if defs.len() > 1 {
                assert!(
                    bands::not_measured_reason(m.len(), span).is_none(),
                    "{name}: band {} survived unusable (n={}, span={span})",
                    b.label,
                    m.len()
                );
            }
        }
        checked += 1;
    }
    assert!(
        checked >= 5,
        "fixture must cover every banded corpus, got {checked}"
    );
}

/// The floors the fixture was generated against. If someone retunes them, the
/// fixture is stale and every published band moves — that must break loudly
/// here rather than quietly change the board.
#[test]
fn parity_fixture_floors_match_the_owners_constants() {
    let text = std::fs::read_to_string(fixture_path()).unwrap();
    let header = text
        .lines()
        .find(|l| l.starts_with("# floors:"))
        .expect("fixture must record the floors it was generated against");
    assert!(
        header.contains(&format!("n_min={}", bands::N_MIN))
            && header.contains(&format!("span_min={}", bands::SPAN_MIN)),
        "fixture floors are stale vs the owner ({} / {}): {header}",
        bands::N_MIN,
        bands::SPAN_MIN
    );
}
