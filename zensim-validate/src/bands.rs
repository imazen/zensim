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
//! [`SPAN_MIN`] is set from the same surface, at the span below which the bar
//! is unreachable at ANY n the corpus can supply. Note what span does NOT do:
//! it barely moves the confidence interval. At a fixed n of 200 the marginal
//! 95 % CI half-width runs 0.140 → 0.086 across spans 0.02 → 0.20 while the
//! correlation itself runs 0.056 → 0.632. **Span binds through the SIGNAL, not
//! the noise** — which is why an estimability bar alone would have passed
//! CID22's degenerate top band, and why both floors are needed.
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
/// axis: measured 0.877 at n=768 and 0.918 at n=1024 on centred slices, and
/// corroborated by CID22's REAL bands at the same span — B7 (n=1092) 0.897,
/// B8 (n=1382) 0.955.
///
/// Discrimination, not estimability, is what binds. A band is *estimable* far
/// sooner — at span 0.10 the marginal 95 % CI half-width crosses 0.20 between
/// n=64 (0.209) and n=96 (0.168) — but an estimable band that ranks models
/// inconsistently cannot gate anything, which is the job. CID22's old `B9` was
/// exactly that: half-width 0.178 (inside the estimability bar) with
/// `r_SB` 0.753 and a model population running −0.263 … −0.015.
pub const N_MIN: usize = 1000;

/// Minimum target span for a band to be usable (appendix V).
///
/// Set from the measured discrimination surface, as the span below which
/// `r_SB ≥ 0.90` is unreachable at ANY n CID22 can supply — at span 0.06 the
/// best observed is 0.659 (n=512, the largest that fits), at 0.04 it is 0.407,
/// at 0.02 it is 0.298. The bar IS reached at 0.10 / 0.15 / 0.20 (n ≈ 1024 /
/// 1024 / 768). 0.08 is the boundary between the two regimes: the trend there
/// (0.762 @384, 0.812 @512, 0.854 @768) is heading for the bar at n ≈ 1200 but
/// was not directly observed clearing it, so this constant sits one grid step
/// BELOW the lowest span where the bar was actually seen — stated plainly
/// because it is the one number here that is not a direct observation.
///
/// The value also has a hard structural upper limit: a fixed decile's realised
/// span is always slightly under 0.10 (CID22's are 0.0956–0.0999), so any floor
/// at 0.10 or above would merge away every single decile on every corpus and
/// the scheme would degenerate to a handful of very wide bands.
///
/// The registered rationale for this constant was WRONG in mechanism and is
/// corrected here: V.3 said span would bind through the estimability bar. It
/// does not — at fixed n the CI half-width is nearly span-independent
/// (0.140 → 0.086 across spans 0.02 → 0.20 at n=200, i.e. every span "passes"
/// `H ≤ 0.20`). Span binds through the attenuated SIGNAL instead, exactly as
/// Thorndike case-II predicts (predicted 0.081 / 0.346 / 0.612 at spans
/// 0.02 / 0.10 / 0.20 against measured 0.056 / 0.370 / 0.632).
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

/// The appendix-V scheme: fixed deciles accumulated into the FINEST partition
/// whose every band clears both floors.
///
/// Deterministic in `targets` alone (it takes no predictions, so band edges
/// cannot depend on the model being evaluated). Sweeping low→high and closing a
/// band the moment it becomes usable is optimal for maximising the number of
/// bands: both floors are monotone under adding another decile, so closing as
/// early as possible leaves the most material for the bands that follow. A
/// deficient remainder at the top is folded into the band before it — the only
/// repair the sweep needs.
///
/// A pairwise "merge the worst band into its smaller neighbour" greedy was
/// tried first and REJECTED: it is myopic and strands bands. On TID it spent
/// B4 (677) on the already-satisfied B5 (705), which left B6-B9 (877) with no
/// deficient neighbour, and the corpus collapsed to a single band — where this
/// sweep finds two clean ones (1418 / 1582). Sweeping high→low instead gives
/// identical bands on all five banded corpora, so the direction is not
/// load-bearing.
///
/// When even the whole corpus cannot clear the floors the result is a single
/// band, which the caller reports as one NOT-MEASURED row. That is the honest
/// answer for CSIQ (866 pairs) and LIVE (779): they are too small to band at
/// the discrimination bar at all.
pub fn merged_bands(targets: &[f64]) -> Vec<BandDef> {
    let k = BASE_BANDS;
    let edge = |i: usize| i as f64 / k as f64;
    let top = |j: usize| {
        if j == k - 1 {
            f64::INFINITY
        } else {
            (j + 1) as f64 / k as f64
        }
    };

    // Inclusive decile index ranges, e.g. (3, 5) == deciles B3..=B5.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for i in 0..k {
        let (n, span) = occupancy(targets, edge(start), top(i));
        if n >= N_MIN && span >= SPAN_MIN {
            groups.push((start, i));
            start = i + 1;
        }
    }
    if start < k {
        // Deficient remainder: fold it into the previous band, or — if nothing
        // ever closed — the corpus is one band.
        match groups.last_mut() {
            Some(last) => last.1 = k - 1,
            None => groups.push((0, k - 1)),
        }
    }

    groups
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

    /// Build a corpus with a given per-decile occupancy, each decile filled
    /// across (almost) its whole width so realised spans match the real ones.
    fn corpus(counts: [usize; 10]) -> Vec<f64> {
        let mut v = Vec::new();
        for (d, &n) in counts.iter().enumerate() {
            let lo = d as f64 / 10.0;
            for i in 0..n {
                let f = if n == 1 {
                    0.0
                } else {
                    i as f64 / (n - 1) as f64
                };
                v.push(lo + 0.0999 * f);
            }
        }
        v
    }

    /// CID22's measured decile occupancy (n=4292): B0/B1 empty, B2 = 1 pair,
    /// top band 43 pairs spanning 0.0194.
    fn cid22_like() -> Vec<f64> {
        let mut v = corpus([0, 0, 1, 57, 266, 615, 836, 1092, 1382, 0]);
        for i in 0..43 {
            v.push(0.9000 + 0.0194 * (i as f64) / 42.0);
        }
        v
    }

    /// TID's measured decile occupancy (n=3000): B8 = 1 pair, B9 empty.
    fn tid_like() -> Vec<f64> {
        corpus([29, 34, 185, 493, 677, 705, 809, 67, 1, 0])
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
        assert_eq!(
            top.label, "B8-B9",
            "B9 must be absorbed by B8, got {bands:?}"
        );
        assert!(top.hi.is_infinite());
        let m = top.members(&t);
        let span = t[*m.last().unwrap()] - t[m[0]];
        assert!(m.len() >= N_MIN, "merged top band n={} < {N_MIN}", m.len());
        assert!(span >= SPAN_MIN, "merged top band span={span} < {SPAN_MIN}");
        assert!(not_measured_reason(m.len(), span).is_none());
    }

    /// TID is the case the rejected pairwise greedy collapsed to one band: it
    /// spent B4 on the already-satisfied B5 and stranded B6-B9 (877). The
    /// sweep must find two clean bands.
    #[test]
    fn tid_does_not_collapse_and_its_empty_top_is_absorbed() {
        let t = tid_like();
        let bands = merged_bands(&t);
        assert!(
            bands.len() >= 2,
            "TID must support >= 2 bands, got {bands:?}"
        );
        assert_eq!(bands.last().unwrap().label, "B5-B9");
        assert!(bands.last().unwrap().members(&t).len() >= N_MIN);
    }

    /// A corpus smaller than the count floor cannot be banded at all, and must
    /// say so as ONE band rather than pretending to ten. CSIQ (866) / LIVE
    /// (779) are the real instances.
    #[test]
    fn a_corpus_below_the_count_floor_collapses_to_one_band() {
        let t = corpus([19, 36, 56, 66, 79, 92, 108, 94, 103, 213]); // CSIQ shape
        let bands = merged_bands(&t);
        assert_eq!(bands.len(), 1, "got {bands:?}");
        assert_eq!(bands[0].label, "B0-B9");
        let m = bands[0].members(&t);
        assert_eq!(m.len(), t.len(), "the single band must hold every pair");
        assert!(
            not_measured_reason(m.len(), 1.0).is_some(),
            "and be NOT-MEASURED"
        );
    }

    /// Sweeping high→low must give the same bands as low→high on every real
    /// corpus shape — if it did not, the direction would be a free parameter
    /// and the scheme would need a justification it does not have.
    #[test]
    fn sweep_direction_is_not_load_bearing() {
        fn rtl(targets: &[f64]) -> Vec<(usize, usize)> {
            let k = BASE_BANDS;
            let edge = |i: usize| i as f64 / k as f64;
            let top = |j: usize| {
                if j == k - 1 {
                    f64::INFINITY
                } else {
                    (j + 1) as f64 / k as f64
                }
            };
            let mut groups: Vec<(usize, usize)> = Vec::new();
            let mut end = k as isize - 1;
            for i in (0..k).rev() {
                if end < i as isize {
                    continue;
                }
                let (n, span) = occupancy(targets, edge(i), top(end as usize));
                if n >= N_MIN && span >= SPAN_MIN {
                    groups.push((i, end as usize));
                    end = i as isize - 1;
                }
            }
            if end >= 0 {
                match groups.last_mut() {
                    Some(last) => last.0 = 0,
                    None => groups.push((0, k - 1)),
                }
            }
            groups.reverse();
            groups
        }
        for t in [
            cid22_like(),
            tid_like(),
            corpus([19, 36, 56, 66, 79, 92, 108, 94, 103, 213]),
        ] {
            let ltr: Vec<(usize, usize)> = merged_bands(&t)
                .iter()
                .map(|b| {
                    let a = (b.lo * 10.0).round() as usize;
                    let z = if b.hi.is_infinite() {
                        BASE_BANDS - 1
                    } else {
                        (b.hi * 10.0).round() as usize - 1
                    };
                    (a, z)
                })
                .collect();
            assert_eq!(ltr, rtl(&t), "sweep direction changed the bands");
        }
    }

    #[test]
    fn every_surviving_band_is_usable_or_the_scheme_collapsed_to_one() {
        for t in [
            cid22_like(),
            tid_like(),
            (0..30000).map(|i| i as f64 / 30000.0).collect(),
        ] {
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
        for t in [
            cid22_like(),
            (0..500).map(|i| i as f64 / 499.0).collect::<Vec<_>>(),
        ] {
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
        assert!(
            not_measured_reason(57, 0.10)
                .unwrap()
                .contains("too few pairs")
        );
        assert!(
            not_measured_reason(2000, 0.02)
                .unwrap()
                .contains("range-restricted")
        );
        assert!(not_measured_reason(N_MIN, SPAN_MIN).is_none());
    }
}
