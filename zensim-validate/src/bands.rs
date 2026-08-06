//! Band edges for the per-band panel — THE owner of "where do the cuts go".
//!
//! # Why this module exists
//!
//! Per-band statistics were cut at fixed `[0, 1]` deciles. That is degenerate at
//! the ends of any corpus whose target distribution stops short of the nominal
//! edge, and two of the five banded corpora do:
//!
//! * **CID22** — MOS tops out at 0.9194, so `[0.90, 1.00]` held **43 pairs from
//!   11 of 49 references spanning 0.0194 MOS**, a target sd 4.4× tighter than
//!   every other band. `B0`/`B1` were structurally EMPTY and `B2` held one pair.
//! * **TID** — `B9` empty, `B8` one pair.
//!
//! A band that narrow cannot be measured: range restriction attenuates the
//! correlation toward zero while doing nothing to the noise, so the statistic is
//! mostly rater noise against model noise. Campaign appendix U measured the
//! consequence — the models that scored BEST on CID22's `B9` were the ones
//! ordering that region most backwards — and appendix V measured the fix.
//!
//! # The rule (campaign appendix V)
//!
//! A band is USABLE only if it satisfies BOTH a count floor and a span floor.
//! Both are load-bearing and neither substitutes for the other: [`N_MIN`] alone
//! admits CID22's quantile bands (n = 429 each, spans 0.024–0.066, none
//! discriminating), and [`SPAN_MIN`] alone admits CID22's `B3` (span 0.096,
//! n = 57). The measured discrimination curve at roughly constant span 0.10 is
//! what sets [`N_MIN`]:
//!
//! | n | 57 | 266 | 615 | 836 | 1092 | 1382 |
//! |---|--:|--:|--:|--:|--:|--:|
//! | split-half `r_SB` | 0.31 | 0.35 | 0.67 | 0.76 | **0.90** | 0.96 |
//!
//! [`SPAN_MIN`] is set from the pure-span curve at fixed n, where the CI
//! half-width crosses the registered 0.20 estimability bar.
//!
//! [`merged_bands`] starts from the fixed deciles — so a band still names a
//! fixed quality region and stays comparable across corpora — and merges
//! adjacent bands inward until every survivor clears both floors. The edges are
//! a function of the TARGET column only, never of any model's predictions, so
//! **every model evaluated on a corpus gets identical bands** and the cross-bake
//! band table stays meaningful. That is a hard requirement of the design, and
//! [`merged_bands`] takes no prediction argument so it cannot be violated.
//!
//! Equal-population (quantile) banding was measured and REJECTED: it guarantees
//! `n` by construction but collapses span exactly where the target distribution
//! is dense, which is the failure mode being fixed.

/// Minimum pairs for a band to be usable (appendix V).
///
/// Set from the split-half discrimination curve at ~constant span 0.10, where
/// `r_SB` crosses the 0.90 threshold registered in appendix O for the HF-NL
/// axis. Below this a band may still be *estimable* — its confidence interval
/// can be narrow enough to report — while being unable to rank models
/// consistently, which is what a gate needs it to do.
pub const N_MIN: usize = 1000;

/// Minimum target span for a band to be usable (appendix V).
///
/// Set from the pure-span curve at fixed n: below this the marginal 95 % CI
/// half-width crosses the registered 0.20 estimability bar no matter how many
/// pairs the band holds, because range restriction attenuates the signal while
/// leaving the noise alone.
pub const SPAN_MIN: f64 = 0.08;

/// Number of fixed deciles the merge starts from (the historical grid).
pub const BASE_BANDS: usize = 10;

/// One band: a label and a half-open `[lo, hi)` target interval. The top band
/// carries `hi = f64::INFINITY` so a target above the nominal 1.0 (LIVE's DMOS
/// reaches 1.026) is never silently dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct BandDef {
    pub label: String,
    pub lo: f64,
    pub hi: f64,
}

impl BandDef {
    /// Row indices of `targets` falling in this band.
    pub fn members(&self, targets: &[f64]) -> Vec<usize> {
        targets
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t >= self.lo && t < self.hi).then_some(i))
            .collect()
    }

    /// Human-readable interval, matching the historical markdown rendering.
    pub fn range_label(&self) -> String {
        if self.hi.is_infinite() {
            format!("[{:.2}, →)", self.lo)
        } else {
            format!("[{:.2}, {:.2})", self.lo, self.hi)
        }
    }
}

/// The fixed-decile grid — the pre-appendix-V scheme, kept because it is what
/// every published board number before 2026-08-06 was cut on and because the
/// merge starts from it.
pub fn fixed_bands() -> Vec<BandDef> {
    (0..BASE_BANDS)
        .map(|i| BandDef {
            label: format!("B{i}"),
            lo: i as f64 / BASE_BANDS as f64,
            hi: if i == BASE_BANDS - 1 {
                f64::INFINITY
            } else {
                (i + 1) as f64 / BASE_BANDS as f64
            },
        })
        .collect()
}

/// Span (max − min) of the targets inside a decile range, and their count.
fn occupancy(targets: &[f64], lo: f64, hi: f64) -> (usize, f64) {
    let mut n = 0usize;
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for &t in targets {
        if t >= lo && t < hi {
            n += 1;
            mn = mn.min(t);
            mx = mx.max(t);
        }
    }
    if n == 0 { (0, 0.0) } else { (n, mx - mn) }
}

/// Why a band cannot be measured, or `None` when it is usable.
///
/// The three states are deliberately distinct in the emitted JSON: a band with
/// too few pairs, a band too narrow to resolve, and a band that is fine. An
/// unusable band is reported as NOT-MEASURED — never as a measured zero, and
/// never silently dropped.
pub fn not_measured_reason(n: usize, span: f64) -> Option<String> {
    if n == 0 {
        return Some("empty: no pairs in this target range".into());
    }
    if n < N_MIN && span < SPAN_MIN {
        return Some(format!(
            "n={n} < {N_MIN} and span={span:.4} < {SPAN_MIN}: too few pairs AND too narrow to resolve"
        ));
    }
    if n < N_MIN {
        return Some(format!("n={n} < {N_MIN}: too few pairs to rank models"));
    }
    if span < SPAN_MIN {
        return Some(format!(
            "span={span:.4} < {SPAN_MIN}: range-restricted, correlation attenuated toward 0"
        ));
    }
    None
}

/// The appendix-V scheme: fixed deciles merged inward until every surviving
/// band clears both floors.
///
/// Deterministic in `targets` alone. The merge repeatedly takes the worst
/// offending band — smallest `n`, ties broken by smallest span — and folds it
/// into whichever neighbour has the smaller `n`, so a sparse tail is absorbed by
/// the adjacent tail rather than by the dense middle. It stops when every band
/// is usable or only one band remains (a corpus too small to band at all, which
/// the caller then reports as a single NOT-MEASURED row).
pub fn merged_bands(targets: &[f64]) -> Vec<BandDef> {
    let k = BASE_BANDS;
    // Inclusive decile index ranges, e.g. [3, 5] == deciles B3..=B5.
    let mut spans: Vec<(usize, usize)> = (0..k).map(|i| (i, i)).collect();

    let edge = |i: usize| i as f64 / k as f64;
    let top = |j: usize| {
        if j == k - 1 {
            f64::INFINITY
        } else {
            (j + 1) as f64 / k as f64
        }
    };

    while spans.len() > 1 {
        let stat = |s: &(usize, usize)| occupancy(targets, edge(s.0), top(s.1));
        let worst = spans
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                let (n, sp) = stat(s);
                n < N_MIN || sp < SPAN_MIN
            })
            // smallest n, then smallest span; index breaks remaining ties so the
            // result never depends on iteration order
            .min_by(|(ia, a), (ib, b)| {
                let (na, sa) = stat(a);
                let (nb, sb) = stat(b);
                na.cmp(&nb)
                    .then(sa.total_cmp(&sb))
                    .then(ia.cmp(ib))
            })
            .map(|(i, _)| i);
        let Some(i) = worst else { break };

        let j = if i == 0 {
            1
        } else if i == spans.len() - 1 {
            spans.len() - 2
        } else if stat(&spans[i - 1]).0 <= stat(&spans[i + 1]).0 {
            i - 1
        } else {
            i + 1
        };
        let (a, b) = (i.min(j), i.max(j));
        spans[a] = (spans[a].0, spans[b].1);
        spans.remove(b);
    }

    spans
        .iter()
        .map(|&(a, b)| BandDef {
            label: if a == b {
                format!("B{a}")
            } else {
                format!("B{a}-B{b}")
            },
            lo: edge(a),
            hi: top(b),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CID22's actual MOS shape: nothing below 0.277, nothing above 0.9194,
    /// mass concentrated in 0.5..0.9.
    fn cid22_like() -> Vec<f64> {
        let mut v = Vec::new();
        let mut push = |lo: f64, hi: f64, n: usize| {
            for i in 0..n {
                v.push(lo + (hi - lo) * (i as f64) / (n as f64));
            }
        };
        push(0.30, 0.399, 57);
        push(0.40, 0.499, 266);
        push(0.50, 0.599, 615);
        push(0.60, 0.699, 836);
        push(0.70, 0.799, 1092);
        push(0.80, 0.899, 1382);
        push(0.900, 0.9194, 43);
        v
    }

    #[test]
    fn fixed_bands_reproduce_the_historical_grid() {
        let b = fixed_bands();
        assert_eq!(b.len(), 10);
        assert_eq!(b[0].lo, 0.0);
        assert!((b[3].lo - 0.30).abs() < 1e-12 && (b[3].hi - 0.40).abs() < 1e-12);
        assert_eq!(b[9].lo, 0.9);
        assert!(b[9].hi.is_infinite(), "top band must be open above");
    }

    /// The whole point: CID22's 43-pair / 0.019-span top band must not survive
    /// as its own band, and what replaces it must clear both floors.
    #[test]
    fn cid22_top_band_is_merged_and_the_survivor_clears_both_floors() {
        let t = cid22_like();
        let bands = merged_bands(&t);
        let top = bands.last().unwrap();
        assert_eq!(top.label, "B8-B9", "B9 must be absorbed by B8, got {bands:?}");
        assert!(top.hi.is_infinite());
        let m = top.members(&t);
        let span = t[*m.last().unwrap()] - t[m[0]];
        assert!(m.len() >= N_MIN, "merged top band n={} < {N_MIN}", m.len());
        assert!(span >= SPAN_MIN, "merged top band span={span} < {SPAN_MIN}");
        assert!(not_measured_reason(m.len(), span).is_none());
    }

    #[test]
    fn every_surviving_band_is_usable_or_the_scheme_collapsed_to_one() {
        for t in [cid22_like(), (0..3000).map(|i| i as f64 / 3000.0).collect()] {
            let bands = merged_bands(&t);
            if bands.len() == 1 {
                continue;
            }
            for b in &bands {
                let m = b.members(&t);
                let span = if m.is_empty() {
                    0.0
                } else {
                    t[*m.iter().max_by(|a, c| t[**a].total_cmp(&t[**c])).unwrap()]
                        - t[*m.iter().min_by(|a, c| t[**a].total_cmp(&t[**c])).unwrap()]
                };
                assert!(
                    not_measured_reason(m.len(), span).is_none(),
                    "band {} survived the merge unusable: n={} span={span}",
                    b.label,
                    m.len()
                );
            }
        }
    }

    /// Edges must not depend on any model — the merge takes only targets, and
    /// the same targets must always give the same bands.
    #[test]
    fn merge_is_deterministic_in_the_target_column_alone() {
        let t = cid22_like();
        let a = merged_bands(&t);
        let mut shuffled = t.clone();
        shuffled.reverse();
        let b = merged_bands(&shuffled);
        assert_eq!(a, b, "band edges must not depend on row order");
    }

    /// Bands must partition: every row lands in exactly one band.
    #[test]
    fn bands_partition_the_corpus() {
        for t in [cid22_like(), (0..500).map(|i| i as f64 / 499.0).collect::<Vec<_>>()] {
            for bands in [fixed_bands(), merged_bands(&t)] {
                let mut seen = vec![0usize; t.len()];
                for b in &bands {
                    for i in b.members(&t) {
                        seen[i] += 1;
                    }
                }
                assert!(
                    seen.iter().all(|&c| c == 1),
                    "every row must fall in exactly one band"
                );
            }
        }
    }

    /// A target above 1.0 (LIVE's DMOS reaches 1.026) must land in the top band,
    /// not vanish.
    #[test]
    fn targets_above_one_land_in_the_top_band() {
        let t: Vec<f64> = (0..1200).map(|i| i as f64 / 1000.0).collect();
        for bands in [fixed_bands(), merged_bands(&t)] {
            let top = bands.last().unwrap();
            assert!(top.hi.is_infinite());
            assert!(top.members(&t).iter().any(|&i| t[i] > 1.0));
        }
    }

    #[test]
    fn not_measured_reasons_are_distinct_and_specific() {
        assert!(not_measured_reason(0, 0.0).unwrap().starts_with("empty"));
        assert!(not_measured_reason(43, 0.019).unwrap().contains("AND"));
        assert!(not_measured_reason(57, 0.10).unwrap().contains("too few pairs"));
        assert!(
            not_measured_reason(2000, 0.02)
                .unwrap()
                .contains("range-restricted")
        );
        assert!(not_measured_reason(N_MIN, SPAN_MIN).is_none());
    }
}
