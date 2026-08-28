//! Within-corpus near-duplicate CONTENT clustering (issue #33).
//!
//! dHash-64 was retired as a cross-corpus *contamination* detector
//! (`benchmarks/dhash_threshold_revert_2026-05-14.md`: at d ≤ 10..16 it
//! false-positives on flat UI screens and similar-composition photos).
//! It is still the right tool for a different, much stricter job: finding
//! the deliberate resample variants of ONE source inside a training
//! corpus (`<hex>_512sq.png`, `<hex>_769x513.png`, `<hex>_1024sq.png`, …),
//! which hash to within a few bits of each other. A source with five size
//! variants currently contributes ~5× the pair weight of a source with
//! one — per-content overfit risk.
//!
//! This module is the ONE owner of the dHash primitive and of the
//! clustering / weighting / culling / split math. The
//! `corpus_content_clusters` binary drives it over a corpus directory or
//! a training CSV; `check_holdout_overlap{,_stage2}` reuse [`dhash_64`].
//!
//! The three curation strategies the issue proposes map onto:
//!
//! * option 2, per-content reweighting → [`content_weights`] (`1 /
//!   cluster_size` per row) and, because the trainer weights at GROUP
//!   granularity (`--group NAME:PATH:TRAIN_W:VAL_W`, sampling ∝
//!   `train_w`), [`reweight_groups`]: split the CSV into one group per
//!   cluster size `k` and give it `train_w ∝ n_rows / k` — exactly the
//!   per-row `1/k` weighting with zero trainer changes;
//! * option 3, one-variant-per-cluster culling → [`canonical_members`]
//!   (largest pixel count wins, first-seen breaks ties);
//! * option 4, content-stratified validation split → [`stratified_split`]
//!   (whole clusters go to one side; deterministic in `seed` and the
//!   clusters' content hashes, not in file order).
//!
//! NOT done here (needs the corpus + a training box): the V0_18-recipe
//! retrain comparison the issue's validation step 3 asks for.

use image::DynamicImage;

/// dHash-64 of a decoded image: resize to 9×8 luma (Lanczos3), one bit
/// per horizontally adjacent pair, set if `left > right`. Robust to
/// resampling and mild recompression; blind to crops.
pub fn dhash_64(img: &DynamicImage) -> u64 {
    let small =
        image::imageops::resize(&img.to_luma8(), 9, 8, image::imageops::FilterType::Lanczos3);
    let mut hash = 0u64;
    let mut bit = 0u32;
    for y in 0..8 {
        for x in 0..8 {
            let left = small.get_pixel(x, y).0[0];
            let right = small.get_pixel(x + 1, y).0[0];
            if left > right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// Hamming distance between two 64-bit hashes.
#[inline]
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Single-linkage clustering: `i` and `j` share a cluster when a chain of
/// hashes at pairwise distance ≤ `max_dist` connects them. Returns one
/// dense cluster id per input, numbered by first appearance (so the
/// output is a deterministic function of input ORDER; use the hashes,
/// not the ids, for order-independent keys).
///
/// O(n²) hash compares — ~150 M `popcnt` for the 17 k-source synthetic
/// corpus, well under a second.
pub fn cluster_by_hamming(hashes: &[u64], max_dist: u32) -> Vec<usize> {
    let n = hashes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if hamming(hashes[i], hashes[j]) <= max_dist {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    // Attach the later root under the earlier one so the
                    // root of every set is its lowest index.
                    let (lo, hi) = if ri < rj { (ri, rj) } else { (rj, ri) };
                    parent[hi] = lo;
                }
            }
        }
    }
    let mut id_of_root: Vec<Option<usize>> = vec![None; n];
    let mut next = 0usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = find(&mut parent, i);
        let id = match id_of_root[r] {
            Some(id) => id,
            None => {
                id_of_root[r] = Some(next);
                next += 1;
                next - 1
            }
        };
        out.push(id);
    }
    out
}

/// Number of members per cluster id (index = cluster id).
pub fn cluster_sizes(cluster_ids: &[usize]) -> Vec<usize> {
    let n_clusters = cluster_ids.iter().copied().max().map_or(0, |m| m + 1);
    let mut sizes = vec![0usize; n_clusters];
    for &c in cluster_ids {
        sizes[c] += 1;
    }
    sizes
}

/// Option 2: per-member weight `1 / cluster_size`, so every content
/// cluster contributes the same total weight regardless of how many
/// resample variants it has.
pub fn content_weights(cluster_ids: &[usize], sizes: &[usize]) -> Vec<f64> {
    cluster_ids.iter().map(|&c| 1.0 / sizes[c] as f64).collect()
}

/// Option 3: exactly one canonical member per cluster — the one with the
/// most pixels (highest resolution); the first-seen member wins a tie.
pub fn canonical_members(cluster_ids: &[usize], pixels: &[u64]) -> Vec<bool> {
    assert_eq!(cluster_ids.len(), pixels.len());
    let n_clusters = cluster_ids.iter().copied().max().map_or(0, |m| m + 1);
    let mut best: Vec<Option<(u64, usize)>> = vec![None; n_clusters];
    for (i, (&c, &px)) in cluster_ids.iter().zip(pixels).enumerate() {
        match best[c] {
            Some((bpx, _)) if bpx >= px => {}
            _ => best[c] = Some((px, i)),
        }
    }
    let mut out = vec![false; cluster_ids.len()];
    for b in best.into_iter().flatten() {
        out[b.1] = true;
    }
    out
}

/// Which side of a content-stratified split a member lands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Split {
    Train,
    Val,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
        }
    }
}

/// splitmix64 finaliser — a cheap, well-mixed 64-bit hash for split keys.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Option 4: content-stratified split. Whole clusters are assigned to
/// one side, so validation never sees a resample variant of training
/// content. Clusters are ordered by a key derived from `seed` and the
/// cluster's MINIMUM member hash (content-addressed: re-listing the
/// corpus in a different order gives the same split), and assigned to
/// `Val` greedily until at least `val_frac` of the MEMBERS are on the
/// validation side.
pub fn stratified_split(
    cluster_ids: &[usize],
    hashes: &[u64],
    val_frac: f64,
    seed: u64,
) -> Vec<Split> {
    assert_eq!(cluster_ids.len(), hashes.len());
    assert!(
        (0.0..=1.0).contains(&val_frac),
        "val_frac must be in [0,1], got {val_frac}"
    );
    let sizes = cluster_sizes(cluster_ids);
    let mut min_hash: Vec<u64> = vec![u64::MAX; sizes.len()];
    for (&c, &h) in cluster_ids.iter().zip(hashes) {
        min_hash[c] = min_hash[c].min(h);
    }
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    // Key on (mixed hash, cluster id) so two clusters with the same min
    // hash (possible: the same dHash under two different max_dist
    // components cannot happen, but identical hashes in one cluster can)
    // still order deterministically.
    order.sort_by_key(|&c| (splitmix64(seed ^ min_hash[c]), c));
    let target = (val_frac * cluster_ids.len() as f64).ceil() as usize;
    let mut val_members = 0usize;
    let mut side = vec![Split::Train; sizes.len()];
    for c in order {
        if val_members >= target {
            break;
        }
        side[c] = Split::Val;
        val_members += sizes[c];
    }
    cluster_ids.iter().map(|&c| side[c]).collect()
}

/// One trainer group per cluster size, for option 2 without trainer
/// changes.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupSpec {
    /// Cluster size `k` shared by every row in this group.
    pub cluster_size: usize,
    /// Rows in the group.
    pub n_rows: usize,
    /// `--group` train weight: `n_rows / k`, normalised so all groups sum
    /// to 1. Because the trainer samples pairs from a group in proportion
    /// to its train weight, a row in a size-`k` group is sampled at
    /// `1/k` the rate of a row in a size-1 group.
    pub train_weight: f64,
}

/// Group the rows of a training table by the cluster size of their
/// source and compute the per-group train weights that realise per-row
/// `1 / cluster_size` sampling. `row_cluster_sizes[i]` is the size of
/// the cluster row `i`'s source belongs to (1 for an unclustered source).
/// Output is sorted by `cluster_size`.
pub fn reweight_groups(row_cluster_sizes: &[usize]) -> Vec<GroupSpec> {
    let mut counts: std::collections::BTreeMap<usize, usize> = Default::default();
    for &k in row_cluster_sizes {
        assert!(k >= 1, "cluster size must be ≥ 1");
        *counts.entry(k).or_default() += 1;
    }
    let raw: Vec<(usize, usize, f64)> = counts
        .iter()
        .map(|(&k, &n)| (k, n, n as f64 / k as f64))
        .collect();
    let total: f64 = raw.iter().map(|r| r.2).sum();
    raw.into_iter()
        .map(|(k, n, w)| GroupSpec {
            cluster_size: k,
            n_rows: n,
            train_weight: if total > 0.0 { w / total } else { 0.0 },
        })
        .collect()
}

/// The name-derived content hint for a corpus file: the stem up to the
/// first `_` (`00b13be94a4867dd_769x513.png` → `00b13be94a4867dd`). Used
/// only as a diagnostic against the hash clusters — it is what the
/// corpus naming *claims*; the hash is what the pixels say.
pub fn base_hint(file_name: &str) -> &str {
    let stem = file_name.rsplit_once('.').map_or(file_name, |(s, _)| s);
    stem.split_once('_').map_or(stem, |(b, _)| b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_counts_differing_bits() {
        assert_eq!(hamming(0, 0), 0);
        assert_eq!(hamming(0, u64::MAX), 64);
        assert_eq!(hamming(0b1011, 0b0001), 2);
    }

    #[test]
    fn clustering_is_single_linkage_and_first_appearance_numbered() {
        // 0 ~ 1 (d=1), 1 ~ 2 (d=1) but 0 vs 2 is d=2: chain joins all three
        // at max_dist=1. 3 is far from everything. 4 ~ 0 exactly.
        let h = [0b0000u64, 0b0001, 0b0011, 0xFFFF_0000_0000_0000, 0b0000];
        let ids = cluster_by_hamming(&h, 1);
        assert_eq!(ids, vec![0, 0, 0, 1, 0]);
        // At max_dist=0 only exact duplicates cluster.
        let ids0 = cluster_by_hamming(&h, 0);
        assert_eq!(ids0, vec![0, 1, 2, 3, 0]);
        assert_eq!(cluster_sizes(&ids), vec![4, 1]);
    }

    #[test]
    fn clustering_empty_input() {
        assert!(cluster_by_hamming(&[], 3).is_empty());
        assert!(cluster_sizes(&[]).is_empty());
    }

    #[test]
    fn content_weights_are_inverse_cluster_size() {
        let ids = [0usize, 0, 0, 1, 2, 2];
        let sizes = cluster_sizes(&ids);
        let w = content_weights(&ids, &sizes);
        assert_eq!(w, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 0.5, 0.5]);
        // Every cluster contributes total weight 1.
        let per_cluster: Vec<f64> = (0..sizes.len())
            .map(|c| {
                ids.iter()
                    .zip(&w)
                    .filter(|&(&i, _)| i == c)
                    .map(|(_, &w)| w)
                    .sum()
            })
            .collect();
        for t in per_cluster {
            assert!((t - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn canonical_member_is_largest_first_seen_on_tie() {
        let ids = [0usize, 0, 0, 1, 1];
        let px = [512 * 512u64, 1024 * 1024, 1024 * 1024, 64, 64];
        let canon = canonical_members(&ids, &px);
        // Cluster 0: index 1 (first 1024²); cluster 1: index 3 (first 64).
        assert_eq!(canon, vec![false, true, false, true, false]);
        assert_eq!(canon.iter().filter(|&&c| c).count(), 2);
    }

    #[test]
    fn stratified_split_never_straddles_a_cluster_and_hits_the_fraction() {
        // 10 clusters of varying size, 40 members.
        let mut ids = Vec::new();
        let mut hashes = Vec::new();
        for c in 0..10usize {
            for m in 0..(c % 4 + 1) * 2 {
                ids.push(c);
                hashes.push(splitmix64((c as u64) << 8 | m as u64));
            }
        }
        let n = ids.len();
        let split = stratified_split(&ids, &hashes, 0.25, 7);
        // No straddle.
        for c in 0..10 {
            let sides: std::collections::BTreeSet<_> = ids
                .iter()
                .zip(&split)
                .filter(|&(&i, _)| i == c)
                .map(|(_, s)| *s)
                .collect();
            assert_eq!(sides.len(), 1, "cluster {c} straddles the split");
        }
        let n_val = split.iter().filter(|&&s| s == Split::Val).count();
        let target = (0.25 * n as f64).ceil() as usize;
        assert!(n_val >= target, "val {n_val} < target {target}");
        // Greedy overshoot is bounded by one cluster (max size 8).
        assert!(n_val < target + 8, "val {n_val} overshoots target {target}");
        // Deterministic in seed; different seed usually differs.
        assert_eq!(split, stratified_split(&ids, &hashes, 0.25, 7));
        // Content-addressed: permuting the member order permutes the
        // output identically (same cluster → same side).
        let perm: Vec<usize> = (0..n).rev().collect();
        let ids_p: Vec<usize> = perm.iter().map(|&i| ids[i]).collect();
        let hashes_p: Vec<u64> = perm.iter().map(|&i| hashes[i]).collect();
        let split_p = stratified_split(&ids_p, &hashes_p, 0.25, 7);
        for (k, &i) in perm.iter().enumerate() {
            assert_eq!(split_p[k], split[i]);
        }
    }

    #[test]
    fn stratified_split_extremes() {
        let ids = [0usize, 0, 1, 2];
        let hashes = [1u64, 2, 3, 4];
        assert!(
            stratified_split(&ids, &hashes, 0.0, 1)
                .iter()
                .all(|&s| s == Split::Train)
        );
        assert!(
            stratified_split(&ids, &hashes, 1.0, 1)
                .iter()
                .all(|&s| s == Split::Val)
        );
        assert!(stratified_split(&[], &[], 0.5, 1).is_empty());
    }

    #[test]
    fn reweight_groups_realise_per_row_inverse_k() {
        // 18 rows from size-3 clusters, 2 rows from size-1 sources.
        let mut rows = vec![3usize; 18];
        rows.extend([1usize; 2]);
        let g = reweight_groups(&rows);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].cluster_size, g[0].n_rows), (1, 2));
        assert_eq!((g[1].cluster_size, g[1].n_rows), (3, 18));
        // raw: 2/1 = 2, 18/3 = 6 → 0.25 : 0.75
        assert!((g[0].train_weight - 0.25).abs() < 1e-12);
        assert!((g[1].train_weight - 0.75).abs() < 1e-12);
        // Per-row sampling rate = train_w / n_rows ∝ 1/k.
        let r1 = g[0].train_weight / g[0].n_rows as f64;
        let r3 = g[1].train_weight / g[1].n_rows as f64;
        assert!((r1 / r3 - 3.0).abs() < 1e-9, "r1/r3 = {}", r1 / r3);
        assert!(reweight_groups(&[]).is_empty());
    }

    #[test]
    fn base_hint_strips_variant_suffix_and_extension() {
        assert_eq!(
            base_hint("00b13be94a4867dd_769x513.png"),
            "00b13be94a4867dd"
        );
        assert_eq!(base_hint("00b13be94a4867dd_512sq.png"), "00b13be94a4867dd");
        assert_eq!(base_hint("kodim01.png"), "kodim01");
        assert_eq!(base_hint("noext_512sq"), "noext");
        assert_eq!(base_hint("plain"), "plain");
    }

    /// The hash primitive on synthetic pixels: a resample variant hashes
    /// within a few bits, a different image does not.
    #[test]
    fn dhash_is_resample_stable_and_content_sensitive() {
        use image::{ImageBuffer, Luma};
        // A seeded 9×9 random lattice, bilinearly upsampled: smooth
        // (resample-stable) but with independent coin-flip structure per
        // seed, so different seeds sit ~32 bits apart.
        let render = |seed: u64, w: u32, h: u32| -> DynamicImage {
            let lattice: Vec<f64> = (0..81u64)
                .map(|i| (splitmix64(seed * 1000 + i) & 0xFFFF) as f64 / 65535.0)
                .collect();
            let img = ImageBuffer::from_fn(w, h, |x, y| {
                let fx = x as f64 / w as f64 * 8.0;
                let fy = y as f64 / h as f64 * 8.0;
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
                let at = |i: usize, j: usize| lattice[j.min(8) * 9 + i.min(8)];
                let v = at(x0, y0) * (1.0 - tx) * (1.0 - ty)
                    + at(x0 + 1, y0) * tx * (1.0 - ty)
                    + at(x0, y0 + 1) * (1.0 - tx) * ty
                    + at(x0 + 1, y0 + 1) * tx * ty;
                Luma([(v.clamp(0.0, 1.0) * 255.0) as u8])
            });
            DynamicImage::ImageLuma8(img)
        };
        let base = render(1, 256, 256);
        let h_base = dhash_64(&base);
        let small = base.resize_exact(128, 128, image::imageops::FilterType::Lanczos3);
        let wide = base.resize_exact(192, 128, image::imageops::FilterType::Lanczos3);
        let (d_small, d_wide) = (
            hamming(h_base, dhash_64(&small)),
            hamming(h_base, dhash_64(&wide)),
        );
        assert!(d_small <= 3, "128sq variant d={d_small}");
        assert!(d_wide <= 3, "192x128 variant d={d_wide}");
        for seed in 2..12u64 {
            let d = hamming(h_base, dhash_64(&render(seed, 256, 256)));
            assert!(d > 10, "seed {seed}: different content only d={d} away");
        }
    }
}
