//! THE owner of the RankNet **pair-draw step** and of subset-coverage
//! measurement.
//!
//! # Why this module exists
//!
//! The draw step used to be copy-pasted inline in four training loops
//! (`train_mlp_strategy`, `train_mlp_pool_head_with_tv`,
//! `train_mlp_hybrid_head_with_tv`, `train_mlp_per_sample_alpha_head`).
//! Three of the four were byte-identical; the fourth added the STRATEGY
//! stratified-row variant. Re-simulating a drawn subset for the
//! subset-quality study (`benchmarks/subset_quality_study_2026-09-04.md`)
//! would have made a *fifth* copy — exactly the duplication zensim's
//! `CLAUDE.md` "one owner per task" rule forbids — so the step was
//! extracted here instead and every caller, training and simulating,
//! routes through [`draw_pair`].
//!
//! # The contract that makes re-simulation exact
//!
//! The trainer runs **two independent `SplitMix64` streams**, both derived
//! from one `--seed`:
//!
//! - init: `SplitMix64::new(seed)` — He-normal weights.
//! - sample: [`sample_stream_seed`] — every pair draw.
//!
//! The separation is deliberate and predates this module: it exists so a
//! 228-vs-372 feature A/B sees the *same* pair sequence even though init
//! consumes a different number of normals. The consequence exploited here
//! is that **the drawn multiset is a pure function of**
//! `(seed, [train_weight], [row_count], epochs, pairs_per_epoch, boosts,
//! within_ref)` **and of nothing else** — in particular not of the feature
//! matrix, the architecture, or the loss. A subset can therefore be
//! reconstructed from a bake's embedded `zentrain.repro` block without
//! reading a single feature column.
//!
//! [`SampleSequenceDigest`] is what proves the reconstruction is faithful:
//! the same rolling hash is computed by the training loop (under
//! `ZENSIM_SAMPLE_DIGEST=1`) and by [`simulate`], and they must match.

use super::RefBuckets;
use super::SplitMix64;

/// Sample-stream seed for every entry point except
/// `train_mlp_per_sample_alpha_head`.
///
/// Mirrors `mod.rs`'s in-loop construction exactly; changing either
/// without the other silently re-rolls every model's training subset.
#[inline]
pub fn sample_stream_seed(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_BABE)
}

/// Sample-stream seed for `train_mlp_per_sample_alpha_head`, which uses a
/// different additive constant.
#[inline]
pub fn sample_stream_seed_per_sample_alpha(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x0123_4567_89AB_CDEF)
}

/// Everything the draw step reads. All slices are indexed by
/// *position within `train_indices`*, not by group index.
pub(crate) struct PairDrawCtx<'a> {
    /// Cumulative normalised `train_weight`, one entry per train group.
    pub cdf: &'a [f64],
    /// Row count per train group (`groups[gi].features.len()`).
    pub row_counts: &'a [usize],
    /// Per-row sampling CDF when any q-boost is non-trivial, else `None`.
    pub per_row_cdfs: &'a [Option<Vec<f64>>],
    /// Within-ref buckets for groups that opted in, else `None`.
    pub ref_buckets: &'a [Option<RefBuckets>],
    /// STRATEGY stratified row-A bands; empty slice = off.
    pub strat_bands: &'a [Vec<Vec<usize>>],
}

/// One draw's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Draw {
    /// The group had fewer than 2 rows. Consumed 1 RNG value.
    GroupTooSmall,
    /// Row indices collided (`ia == ib == row`). The RNG values were
    /// consumed exactly as for a `Pair`. Most callers skip; the STRATEGY
    /// site instead runs its hard-pair miner first, which is why the
    /// collided index is carried rather than discarded.
    SameRow { train_pos: usize, row: usize },
    /// A usable pair.
    Pair {
        train_pos: usize,
        ia: usize,
        ib: usize,
    },
}

/// Draw one RankNet pair.
///
/// **RNG consumption is part of the contract**, not an implementation
/// detail: 1 value for the group, then 2 (uniform / per-row-CDF) or 3
/// (within-ref) for the rows, with an early return after the group value
/// when the group is unusable. Any change re-rolls every subsequent draw
/// of every model ever trained, so it must be treated as an era break.
pub(crate) fn draw_pair(ctx: &PairDrawCtx<'_>, rng: &mut SplitMix64) -> Draw {
    let u = rng.next_f64_unit();
    let train_pos = ctx.cdf.partition_point(|&c| c < u).min(ctx.cdf.len() - 1);
    let n = ctx.row_counts[train_pos];
    if n < 2 {
        return Draw::GroupTooSmall;
    }
    let (ia, ib) = if let Some(rb) = &ctx.ref_buckets[train_pos] {
        // Within-ref: pick a ref, then two rows inside it.
        rb.draw(rng.next_u64(), rng.next_u64(), rng.next_u64())
    } else {
        match &ctx.per_row_cdfs[train_pos] {
            Some(row_cdf) => {
                let ua = rng.next_f64_unit();
                let ub = rng.next_f64_unit();
                (
                    row_cdf.partition_point(|&c| c < ua).min(n - 1),
                    row_cdf.partition_point(|&c| c < ub).min(n - 1),
                )
            }
            None => {
                // STRATEGY: stratified row A — band-uniform then row-uniform.
                let ia = if !ctx.strat_bands.is_empty() {
                    let bands = &ctx.strat_bands[train_pos];
                    let b = &bands[(rng.next_u64() as usize) % bands.len()];
                    b[(rng.next_u64() as usize) % b.len()]
                } else {
                    (rng.next_u64() as usize) % n
                };
                (ia, (rng.next_u64() as usize) % n)
            }
        }
    };
    if ia == ib {
        return Draw::SameRow { train_pos, row: ia };
    }
    Draw::Pair { train_pos, ia, ib }
}

/// Order-sensitive rolling hash over the drawn `(train_pos, ia, ib)`
/// stream — the faithfulness proof for [`simulate`].
///
/// FNV-1a over little-endian `u32` triples. Skipped draws are folded in
/// too (as a distinct tag) so a divergence in *skip* behaviour, which
/// would silently shift every later draw, cannot hash equal.
#[derive(Clone, Copy, Debug)]
pub struct SampleSequenceDigest(u64);

impl Default for SampleSequenceDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl SampleSequenceDigest {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    #[inline]
    fn byte(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
    }
    #[inline]
    fn word(&mut self, w: u32) {
        for b in w.to_le_bytes() {
            self.byte(b);
        }
    }
    /// Fold one draw outcome in.
    #[inline]
    pub(crate) fn push(&mut self, d: Draw) {
        match d {
            Draw::GroupTooSmall => self.byte(0x01),
            Draw::SameRow { train_pos, row } => {
                self.byte(0x02);
                self.word(train_pos as u32);
                self.word(row as u32);
            }
            Draw::Pair { train_pos, ia, ib } => {
                self.byte(0x03);
                self.word(train_pos as u32);
                self.word(ia as u32);
                self.word(ib as u32);
            }
        }
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

// ---------------------------------------------------------------------
// Coverage simulation
// ---------------------------------------------------------------------

/// One training group as the simulator needs it: counts, scores, refs.
pub struct SimGroup {
    pub name: String,
    pub train_weight: f64,
    pub n_rows: usize,
    pub human_scores: Vec<f64>,
    pub ref_ids: Option<Vec<u32>>,
    pub within_ref: bool,
}

/// The knobs that change RNG consumption or row weighting.
pub struct SimParams {
    pub seed: u64,
    pub epochs: usize,
    pub pairs_per_epoch: usize,
    pub low_q_boost: f64,
    pub mid_q_boost: f64,
    pub high_q_boost: f64,
    pub stratified_bands: usize,
    /// Draw count defining the "early window" descriptors. `0` = one epoch.
    pub early_window: usize,
    /// Use the `train_mlp_per_sample_alpha_head` stream constant.
    pub per_sample_alpha_head: bool,
}

/// Per-group coverage descriptors over one window.
#[derive(Clone, Debug, Default)]
pub struct GroupCoverage {
    pub name: String,
    pub n_rows: usize,
    pub train_weight: f64,
    /// Pairs drawn from this group.
    pub n_pairs: u64,
    /// Rows touched at least once.
    pub rows_touched: usize,
    /// `rows_touched / n_rows`.
    pub row_coverage: f64,
    /// Distinct reference images touched (groups with `ref_ids`).
    pub refs_touched: usize,
    pub n_refs: usize,
    pub ref_coverage: f64,
    /// Distinct (ref, native-band) cells touched / total non-empty cells.
    pub cell_coverage: f64,
    pub cells_touched: usize,
    pub n_cells: usize,
    /// Shannon entropy of per-row multiplicity, normalised by `ln(n_rows)`.
    pub row_entropy_norm: f64,
    /// Coefficient of variation of per-row multiplicity.
    pub row_multiplicity_cv: f64,
    /// Share of this group's pairs whose higher endpoint scores >= 90 —
    /// the near-threshold / near-lossless zone.
    pub near_threshold_share: f64,
    /// Share of drawn pairs whose two rows share a reference image.
    pub within_image_share: f64,
    /// Pairs drawn from rows in each of the trainer's four NATIVE quality
    /// bands (B0 <50, B1 [50,65), B2 [65,90), B3 >=90), endpoint-counted.
    pub band_pair_counts: [u64; 4],
}

/// Whole-run (or whole-window) coverage descriptors.
#[derive(Clone, Debug, Default)]
pub struct SampleCoverage {
    pub window_draws: u64,
    pub n_pairs: u64,
    /// Draws lost to `ia == ib` — a pure RNG-luck control.
    pub same_row_skips: u64,
    pub group_too_small_skips: u64,
    /// Pooled `rows_touched / total_rows`.
    pub pooled_row_coverage: f64,
    /// L1 deviation between realised per-group pair share and declared
    /// normalised `train_weight`.
    pub group_share_l1: f64,
    /// Pearson chi-square of realised vs declared group shares.
    pub group_share_chisq: f64,
    /// Pooled near-threshold (>=90) endpoint share.
    pub near_threshold_share: f64,
    /// Pooled within-image pair share.
    pub within_image_share: f64,
    /// Fraction of drawn pairs repeating an already-drawn `(g, ia, ib)`.
    /// Only computed when the window is bounded (see `SimParams`).
    pub duplicate_pair_rate: f64,
    pub per_group: Vec<GroupCoverage>,
}

/// Full simulation output: the whole-run window, the early window, and the
/// digest proving it matches a real run.
pub struct SimResult {
    pub full: SampleCoverage,
    pub early: SampleCoverage,
    pub digest: SampleSequenceDigest,
    /// Digest over just the early window, so a short faithfulness check
    /// does not need a full 6M-draw run.
    pub early_digest: SampleSequenceDigest,
}

/// The trainer's own native quality bands, as used by its q-boost logic
/// (`mod.rs`: B0 <50, B1 [50,65), B2 [65,90), B3 >=90). Reused rather than
/// inventing a banding so descriptors speak the sampler's own language.
#[inline]
pub fn native_band(score: f64) -> usize {
    if score < 50.0 {
        0
    } else if score < 65.0 {
        1
    } else if score < 90.0 {
        2
    } else {
        3
    }
}

/// Build the per-row sampling CDF exactly as `mod.rs` does.
fn build_per_row_cdf(scores: &[f64], lo: f64, mid: f64, hi: f64) -> Option<Vec<f64>> {
    if lo == 1.0 && mid == 1.0 && hi == 1.0 {
        return None;
    }
    let mut cum = 0.0;
    let raw: Vec<f64> = scores
        .iter()
        .map(|&s| {
            let mut w = 1.0;
            if lo != 1.0 {
                if s < 50.0 {
                    w *= lo;
                } else if s < 65.0 {
                    w *= lo.sqrt();
                }
            }
            if mid != 1.0 && (50.0..90.0).contains(&s) {
                w *= mid;
            }
            if hi != 1.0 && s >= 90.0 {
                w *= hi;
            }
            cum += w;
            cum
        })
        .collect();
    let total = *raw.last().unwrap_or(&1.0);
    Some(raw.into_iter().map(|c| c / total).collect())
}

/// Quantile bands over a group's scores, matching
/// `strategy::build_bands`'s contract closely enough for RNG-consumption
/// parity (band COUNT is what the draw reads).
fn build_bands(scores: &[f64], n_bands: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut out = vec![Vec::new(); n_bands.max(1)];
    for (rank, &row) in order.iter().enumerate() {
        let b = (rank * n_bands) / scores.len().max(1);
        out[b.min(n_bands - 1)].push(row);
    }
    out.retain(|b| !b.is_empty());
    out
}

/// Accumulates coverage over one window.
struct Acc {
    hits: Vec<Vec<u32>>,
    ref_seen: Vec<Vec<bool>>,
    cell_seen: Vec<std::collections::HashSet<(u32, u8)>>,
    pairs: Vec<u64>,
    near: Vec<u64>,
    within: Vec<u64>,
    bands: Vec<[u64; 4]>,
    draws: u64,
    same_row: u64,
    too_small: u64,
    dup: Option<(std::collections::HashSet<u64>, u64)>,
}

impl Acc {
    fn new(groups: &[SimGroup], track_dup: bool) -> Self {
        Self {
            hits: groups.iter().map(|g| vec![0u32; g.n_rows]).collect(),
            ref_seen: groups
                .iter()
                .map(|g| {
                    let n = g
                        .ref_ids
                        .as_ref()
                        .and_then(|r| r.iter().copied().max())
                        .map(|m| m as usize + 1)
                        .unwrap_or(0);
                    vec![false; n]
                })
                .collect(),
            cell_seen: groups
                .iter()
                .map(|_| std::collections::HashSet::new())
                .collect(),
            pairs: vec![0; groups.len()],
            near: vec![0; groups.len()],
            within: vec![0; groups.len()],
            bands: vec![[0u64; 4]; groups.len()],
            draws: 0,
            same_row: 0,
            too_small: 0,
            dup: track_dup.then(|| (std::collections::HashSet::new(), 0)),
        }
    }

    fn record(&mut self, groups: &[SimGroup], d: Draw) {
        self.draws += 1;
        match d {
            Draw::GroupTooSmall => self.too_small += 1,
            Draw::SameRow { .. } => self.same_row += 1,
            Draw::Pair { train_pos, ia, ib } => {
                let g = &groups[train_pos];
                self.pairs[train_pos] += 1;
                self.hits[train_pos][ia] += 1;
                self.hits[train_pos][ib] += 1;
                let sa = g.human_scores[ia];
                let sb = g.human_scores[ib];
                self.bands[train_pos][native_band(sa)] += 1;
                self.bands[train_pos][native_band(sb)] += 1;
                if sa.max(sb) >= 90.0 {
                    self.near[train_pos] += 1;
                }
                if let Some(rids) = &g.ref_ids {
                    let (ra, rb) = (rids[ia], rids[ib]);
                    self.ref_seen[train_pos][ra as usize] = true;
                    self.ref_seen[train_pos][rb as usize] = true;
                    self.cell_seen[train_pos].insert((ra, native_band(sa) as u8));
                    self.cell_seen[train_pos].insert((rb, native_band(sb) as u8));
                    if ra == rb {
                        self.within[train_pos] += 1;
                    }
                }
                if let Some((set, dups)) = &mut self.dup {
                    let key = ((train_pos as u64) << 56)
                        ^ ((ia.min(ib) as u64) << 28)
                        ^ (ia.max(ib) as u64);
                    if !set.insert(key) {
                        *dups += 1;
                    }
                }
            }
        }
    }

    fn finish(self, groups: &[SimGroup], cdf: &[f64]) -> SampleCoverage {
        let total_pairs: u64 = self.pairs.iter().sum();
        let mut per_group = Vec::with_capacity(groups.len());
        let mut pooled_touched = 0usize;
        let mut pooled_rows = 0usize;
        let mut pooled_near = 0u64;
        let mut pooled_within = 0u64;
        let mut pooled_within_denom = 0u64;
        for (i, g) in groups.iter().enumerate() {
            let hits = &self.hits[i];
            let touched = hits.iter().filter(|&&h| h > 0).count();
            pooled_touched += touched;
            pooled_rows += g.n_rows;
            let total_hits: f64 = hits.iter().map(|&h| h as f64).sum();
            let (mut ent, mut sumsq) = (0.0f64, 0.0f64);
            for &h in hits {
                if h > 0 && total_hits > 0.0 {
                    let p = h as f64 / total_hits;
                    ent -= p * p.ln();
                }
                sumsq += (h as f64) * (h as f64);
            }
            let n = g.n_rows.max(1) as f64;
            let mean = total_hits / n;
            let var = (sumsq / n) - mean * mean;
            let cv = if mean > 0.0 {
                var.max(0.0).sqrt() / mean
            } else {
                0.0
            };
            let n_refs = self.ref_seen[i].len();
            let refs_touched = self.ref_seen[i].iter().filter(|&&b| b).count();
            // Total non-empty (ref, band) cells present in the group.
            let n_cells = g
                .ref_ids
                .as_ref()
                .map(|r| {
                    let mut s = std::collections::HashSet::new();
                    for (row, &rid) in r.iter().enumerate() {
                        s.insert((rid, native_band(g.human_scores[row]) as u8));
                    }
                    s.len()
                })
                .unwrap_or(0);
            let cells_touched = self.cell_seen[i].len();
            pooled_near += self.near[i];
            if g.ref_ids.is_some() {
                pooled_within += self.within[i];
                pooled_within_denom += self.pairs[i];
            }
            per_group.push(GroupCoverage {
                name: g.name.clone(),
                n_rows: g.n_rows,
                train_weight: g.train_weight,
                n_pairs: self.pairs[i],
                rows_touched: touched,
                row_coverage: touched as f64 / n,
                refs_touched,
                n_refs,
                ref_coverage: if n_refs > 0 {
                    refs_touched as f64 / n_refs as f64
                } else {
                    f64::NAN
                },
                cell_coverage: if n_cells > 0 {
                    cells_touched as f64 / n_cells as f64
                } else {
                    f64::NAN
                },
                cells_touched,
                n_cells,
                row_entropy_norm: if n > 1.0 { ent / n.ln() } else { 0.0 },
                row_multiplicity_cv: cv,
                near_threshold_share: if self.pairs[i] > 0 {
                    self.near[i] as f64 / self.pairs[i] as f64
                } else {
                    f64::NAN
                },
                within_image_share: if g.ref_ids.is_some() && self.pairs[i] > 0 {
                    self.within[i] as f64 / self.pairs[i] as f64
                } else {
                    f64::NAN
                },
                band_pair_counts: self.bands[i],
            });
        }
        // Declared shares from the CDF (differences of the cumulative).
        let mut declared = Vec::with_capacity(cdf.len());
        let mut prev = 0.0;
        for &c in cdf {
            declared.push(c - prev);
            prev = c;
        }
        let (mut l1, mut chisq) = (0.0f64, 0.0f64);
        if total_pairs > 0 {
            for (i, d) in declared.iter().enumerate() {
                let obs = self.pairs[i] as f64 / total_pairs as f64;
                l1 += (obs - d).abs();
                if *d > 0.0 {
                    let e = d * total_pairs as f64;
                    let o = self.pairs[i] as f64;
                    chisq += (o - e) * (o - e) / e;
                }
            }
        }
        SampleCoverage {
            window_draws: self.draws,
            n_pairs: total_pairs,
            same_row_skips: self.same_row,
            group_too_small_skips: self.too_small,
            pooled_row_coverage: pooled_touched as f64 / pooled_rows.max(1) as f64,
            group_share_l1: l1,
            group_share_chisq: chisq,
            near_threshold_share: if total_pairs > 0 {
                pooled_near as f64 / total_pairs as f64
            } else {
                f64::NAN
            },
            within_image_share: if pooled_within_denom > 0 {
                pooled_within as f64 / pooled_within_denom as f64
            } else {
                f64::NAN
            },
            duplicate_pair_rate: match self.dup {
                Some((_, dups)) if total_pairs > 0 => dups as f64 / total_pairs as f64,
                _ => f64::NAN,
            },
            per_group,
        }
    }
}

/// Replay a training run's pair sampler without training.
///
/// Uses the same [`draw_pair`] the training loops use, so the sequence is
/// the run's sequence by construction rather than by a re-implementation
/// that has to be kept in sync.
pub fn simulate(groups: &[SimGroup], params: &SimParams) -> SimResult {
    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    let mut cum = 0.0;
    let cdf: Vec<f64> = groups
        .iter()
        .map(|g| {
            cum += g.train_weight;
            cum / train_total
        })
        .collect();
    let row_counts: Vec<usize> = groups.iter().map(|g| g.n_rows).collect();
    let per_row_cdfs: Vec<Option<Vec<f64>>> = groups
        .iter()
        .map(|g| {
            build_per_row_cdf(
                &g.human_scores,
                params.low_q_boost,
                params.mid_q_boost,
                params.high_q_boost,
            )
        })
        .collect();
    let ref_buckets: Vec<Option<RefBuckets>> = groups
        .iter()
        .map(|g| {
            if g.within_ref {
                g.ref_ids.as_deref().and_then(RefBuckets::build)
            } else {
                None
            }
        })
        .collect();
    let strat_bands: Vec<Vec<Vec<usize>>> = if params.stratified_bands > 0 {
        groups
            .iter()
            .map(|g| build_bands(&g.human_scores, params.stratified_bands))
            .collect()
    } else {
        Vec::new()
    };

    let ctx = PairDrawCtx {
        cdf: &cdf,
        row_counts: &row_counts,
        per_row_cdfs: &per_row_cdfs,
        ref_buckets: &ref_buckets,
        strat_bands: &strat_bands,
    };

    let stream = if params.per_sample_alpha_head {
        sample_stream_seed_per_sample_alpha(params.seed)
    } else {
        sample_stream_seed(params.seed)
    };
    let mut rng = SplitMix64::new(stream);

    let early_n = if params.early_window == 0 {
        params.pairs_per_epoch
    } else {
        params.early_window
    };
    let total = params.epochs.saturating_mul(params.pairs_per_epoch);

    let mut full = Acc::new(groups, false);
    let mut early = Acc::new(groups, true);
    let mut digest = SampleSequenceDigest::new();
    let mut early_digest = SampleSequenceDigest::new();

    for step in 0..total {
        let d = draw_pair(&ctx, &mut rng);
        digest.push(d);
        full.record(groups, d);
        if step < early_n {
            early_digest.push(d);
            early.record(groups, d);
        }
    }

    SimResult {
        full: full.finish(groups, &cdf),
        early: early.finish(groups, &cdf),
        digest,
        early_digest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(name: &str, w: f64, scores: Vec<f64>, refs: Option<Vec<u32>>, wr: bool) -> SimGroup {
        SimGroup {
            name: name.into(),
            train_weight: w,
            n_rows: scores.len(),
            human_scores: scores,
            ref_ids: refs,
            within_ref: wr,
        }
    }

    fn params(seed: u64) -> SimParams {
        SimParams {
            seed,
            epochs: 3,
            pairs_per_epoch: 500,
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            stratified_bands: 0,
            early_window: 100,
            per_sample_alpha_head: false,
        }
    }

    fn two_groups() -> Vec<SimGroup> {
        let a: Vec<f64> = (0..40).map(|i| (i as f64) * 2.5).collect();
        let b: Vec<f64> = (0..25).map(|i| 40.0 + (i as f64) * 2.0).collect();
        vec![
            g("a", 1.0, a, Some((0..40).map(|i| i / 4).collect()), false),
            g("b", 0.5, b, Some((0..25).map(|i| i / 5).collect()), false),
        ]
    }

    /// The stream constants are a wire contract: if either changes, every
    /// model ever trained had a different training subset than its repro
    /// block claims. Pinned, not derived.
    #[test]
    fn sample_stream_seeds_are_pinned() {
        assert_eq!(sample_stream_seed(0), 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(
            sample_stream_seed_per_sample_alpha(0),
            0x0123_4567_89AB_CDEF
        );
        // A concrete production seed, so a refactor of the arithmetic
        // (not just the constants) is caught too.
        assert_eq!(
            sample_stream_seed(4004),
            4004u64
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0xDEAD_BEEF_CAFE_BABE)
        );
        assert_ne!(sample_stream_seed(4004), sample_stream_seed(4005));
        // Init and sample streams must not coincide for any seed we ship.
        for s in [0u64, 1, 42, 4004, 4005, 1301] {
            assert_ne!(s, sample_stream_seed(s));
        }
    }

    /// A PINNED digest over a fixed synthetic fixture.
    ///
    /// `simulate_is_deterministic_for_a_seed` only proves the replay agrees
    /// with itself; this pins the actual sequence, so any future change to
    /// the draw order or the RNG consumption pattern fails here instead of
    /// silently re-rolling every model's training subset. The value was
    /// taken from a run whose digest was independently verified to match a
    /// REAL `zensim_mlp_train` run under `ZENSIM_SAMPLE_DIGEST=1` (see
    /// `benchmarks/subset_quality_study_2026-09-04.md` §Phase 1).
    #[test]
    fn simulate_digest_is_pinned() {
        let gs = two_groups();
        let r = simulate(&gs, &params(4004));
        assert_eq!(
            r.digest.hex(),
            "39dd23c68ee3f018",
            "the pair-draw sequence changed — this is an ERA BREAK for every \
             model trained through this sampler, not a test to update casually"
        );
    }

    #[test]
    fn simulate_is_deterministic_for_a_seed() {
        let gs = two_groups();
        let a = simulate(&gs, &params(4004));
        let b = simulate(&gs, &params(4004));
        assert_eq!(a.digest.finish(), b.digest.finish());
        assert_eq!(a.full.n_pairs, b.full.n_pairs);
        assert_eq!(a.early.pooled_row_coverage, b.early.pooled_row_coverage);
    }

    #[test]
    fn different_seeds_give_different_subsets() {
        let gs = two_groups();
        let a = simulate(&gs, &params(4004));
        let b = simulate(&gs, &params(4005));
        assert_ne!(a.digest.finish(), b.digest.finish());
    }

    /// The digest must be sensitive to skip behaviour, not just to the
    /// pairs that survive — a skip shifts every later draw.
    #[test]
    fn digest_distinguishes_skips_from_pairs() {
        let mut d1 = SampleSequenceDigest::new();
        let mut d2 = SampleSequenceDigest::new();
        d1.push(Draw::Pair {
            train_pos: 0,
            ia: 1,
            ib: 2,
        });
        d2.push(Draw::SameRow {
            train_pos: 0,
            row: 5,
        });
        d2.push(Draw::Pair {
            train_pos: 0,
            ia: 1,
            ib: 2,
        });
        assert_ne!(d1.finish(), d2.finish());
        let mut d3 = SampleSequenceDigest::new();
        d3.push(Draw::Pair {
            train_pos: 0,
            ia: 2,
            ib: 1,
        });
        assert_ne!(
            d1.finish(),
            d3.finish(),
            "pair ORDER is part of the sequence"
        );
    }

    /// A group with fewer than 2 rows consumes exactly ONE RNG value and
    /// returns early. Getting this wrong desynchronises the whole stream.
    #[test]
    fn tiny_group_consumes_one_draw_only() {
        let gs = vec![g("solo", 1.0, vec![50.0], None, false)];
        let cdf = [1.0];
        let rows = [1usize];
        let prc = [None];
        let rb = [None];
        let ctx = PairDrawCtx {
            cdf: &cdf,
            row_counts: &rows,
            per_row_cdfs: &prc,
            ref_buckets: &rb,
            strat_bands: &[],
        };
        let _ = gs;
        let mut r1 = SplitMix64::new(7);
        assert_eq!(draw_pair(&ctx, &mut r1), Draw::GroupTooSmall);
        // One value consumed: a fresh RNG advanced once must match.
        let mut r2 = SplitMix64::new(7);
        let _ = r2.next_f64_unit();
        assert_eq!(r1.next_u64(), r2.next_u64());
    }

    /// Within-ref draws must never cross reference images.
    #[test]
    fn within_ref_pairs_share_a_reference() {
        let refs: Vec<u32> = (0..40).map(|i| i / 4).collect();
        let scores: Vec<f64> = (0..40).map(|i| (i as f64) * 2.5).collect();
        let gs = vec![g("a", 1.0, scores, Some(refs.clone()), true)];
        let mut p = params(11);
        p.epochs = 2;
        let res = simulate(&gs, &p);
        assert!(res.full.n_pairs > 100);
        let wi = res.full.per_group[0].within_image_share;
        assert!(
            (wi - 1.0).abs() < 1e-12,
            "within-ref group drew cross-image pairs: share={wi}"
        );
    }

    /// Cross-image is the default: the same group WITHOUT the opt-in must
    /// draw mostly across references (the negative control for the test
    /// above).
    #[test]
    fn cross_image_is_the_default() {
        let refs: Vec<u32> = (0..40).map(|i| i / 4).collect();
        let scores: Vec<f64> = (0..40).map(|i| (i as f64) * 2.5).collect();
        let gs = vec![g("a", 1.0, scores, Some(refs), false)];
        let res = simulate(&gs, &params(11));
        let wi = res.full.per_group[0].within_image_share;
        assert!(wi < 0.25, "expected mostly cross-image draws, got {wi}");
    }

    /// A q-boost must move the realised band mix, and must do so in the
    /// declared direction — otherwise the boost silently does nothing.
    #[test]
    fn high_q_boost_shifts_the_band_mix() {
        let gs = two_groups();
        let flat = simulate(&gs, &params(3));
        let mut p = params(3);
        p.high_q_boost = 3.0;
        let boosted = simulate(&gs, &p);
        let b3 = |r: &SimResult| -> f64 {
            let (mut hi, mut tot) = (0u64, 0u64);
            for g in &r.full.per_group {
                hi += g.band_pair_counts[3];
                tot += g.band_pair_counts.iter().sum::<u64>();
            }
            hi as f64 / tot as f64
        };
        assert!(
            b3(&boosted) > b3(&flat) * 1.2,
            "high_q_boost=3 did not raise the >=90 share: {} vs {}",
            b3(&boosted),
            b3(&flat)
        );
    }

    /// Group shares must track the declared weights (2:1 here).
    #[test]
    fn realised_group_share_tracks_declared_weight() {
        let gs = two_groups();
        let mut p = params(5);
        p.epochs = 20;
        let r = simulate(&gs, &p);
        let a = r.full.per_group[0].n_pairs as f64;
        let b = r.full.per_group[1].n_pairs as f64;
        assert!(
            (a / b - 2.0).abs() < 0.15,
            "declared 1.0:0.5 but drew {a}:{b}"
        );
        assert!(r.full.group_share_l1 < 0.02, "{}", r.full.group_share_l1);
    }

    /// The early window is a strict prefix of the full run.
    #[test]
    fn early_window_is_a_prefix_of_the_full_run() {
        let gs = two_groups();
        let r = simulate(&gs, &params(9));
        assert_eq!(r.early.window_draws, 100);
        assert_eq!(r.full.window_draws, 1500);
        assert!(r.early.n_pairs <= r.full.n_pairs);
        for (e, f) in r.early.per_group.iter().zip(r.full.per_group.iter()) {
            assert!(
                e.rows_touched <= f.rows_touched,
                "early window touched more rows than the whole run"
            );
        }
        // And a simulation whose TOTAL equals the early window must have
        // the same digest as that window.
        let mut p = params(9);
        p.epochs = 1;
        p.pairs_per_epoch = 100;
        p.early_window = 100;
        let short = simulate(&gs, &p);
        assert_eq!(short.digest.finish(), r.early_digest.finish());
    }
}
