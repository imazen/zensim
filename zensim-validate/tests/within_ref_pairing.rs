//! Within-ref RankNet pair sampling (2026-07-15).
//!
//! Two properties, both load-bearing:
//!
//! 1. **Within-ref draws never cross a reference image.** This is the
//!    whole point: a cross-image pair teaches "image A outranks image B"
//!    (a between-image *scale* fact) instead of "this distortion is worse
//!    than that one" (the ranking task). MEASURED on the post-jxl-fix
//!    near-lossless corpus, the ssim2 ladder moves ~0.92 pts within an
//!    image against ~6 pts between images — so cross-image pairs bury the
//!    ladder. That corpus reads pooled SROCC +0.204 vs per-ref +0.916,
//!    the same confound as the documented AIC-3 "0.79 pooled / 0.93
//!    per-ref".
//!
//! 2. **Groups that did not opt in are untouched.** `RefBuckets` is only
//!    consulted when `TrainingGroup::ref_ids` is `Some`, which the binary
//!    sets only for a `:withinref` group. Every existing recipe must keep
//!    producing bit-identical bakes.
//!
//! The buckets are `pub(crate)`, so this exercises the observable
//! behavior through the public loader + trainer surface instead.
//!
//! Skip semantics: marked `#[ignore]` so the test does NOT run in CI
//! environments that lack the `/mnt/v` fixtures (GitHub runners never
//! have block storage) — the same convention `parquet_load_equivalence.rs`
//! uses for its `/mnt/v` parquet. This is NOT a graceful skip: the
//! decision is the caller's and is visible in the invocation, and when
//! the test DOES run without its corpus it panics loudly with the R2
//! restore path rather than passing vacuously. To run locally:
//!
//!     cargo test -p zensim-validate --test within_ref_pairing -- --ignored

use std::collections::HashSet;
use std::path::PathBuf;

use zensim_validate::parquet_loader::load_parquet;

/// The post-jxl-fix near-lossless corpus: 200 refs x 6 distances, the
/// only data we hold below distance 0.03 generated with the fixed
/// encoder. Uses `feat_` + `image_path` (the sidecar convention) rather
/// than `f` + `ref_basename` (the canonical-corpus convention).
fn hf_corpus() -> PathBuf {
    PathBuf::from("/mnt/v/output/zensim-jxl-nearlossless/refit/features.parquet")
}

fn canonical_safesyn() -> PathBuf {
    PathBuf::from("/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet")
}

/// The loader must accept BOTH feature-column prefixes. `f<i>` is the
/// canonical-corpus/CSV convention; `feat_<i>` is what zenmetrics
/// sidecars and the pareto-sweep extractor emit. Both name the same
/// 372-wide with-iw space — rejecting either just forces a rename-copy.
#[test]
#[ignore = "needs the /mnt/v corpora; run with -- --ignored"]
fn loads_feat_underscore_prefix_and_ref_identity() {
    let p = hf_corpus();
    if !p.exists() {
        panic!(
            "missing HF corpus at {p:?} — restore from R2 \
             s3://zentrain/jxl-nearlossless-2026-07-06/ (see \
             benchmarks/jxl_nearlossless_corpus_2026-07-06.pointer.md)"
        );
    }
    let g = load_parquet(&p, "hf", "zensim_score", 1.0).expect("load HF corpus");

    assert_eq!(
        g.n_features, 372,
        "HF corpus is the 372-wide with-iw regime"
    );

    let ref_ids = g
        .ref_ids
        .as_ref()
        .expect("HF corpus has image_path -> ref_ids must be populated");
    assert_eq!(ref_ids.len(), g.human_scores.len(), "one ref id per row");

    // 200 refs x 6 distances = 1200 cells.
    let distinct: HashSet<u32> = ref_ids.iter().copied().collect();
    assert_eq!(distinct.len(), 200, "corpus is 200 reference images");
    assert_eq!(g.human_scores.len(), 1200, "200 refs x 6 distances");

    // Dense numbering: ids must be exactly 0..n_refs.
    let max = ref_ids.iter().copied().max().unwrap();
    assert_eq!(max as usize, distinct.len() - 1, "ref ids are dense 0..n");
}

/// The canonical corpora use `f<i>` + `ref_basename`. Both must still
/// load, and ref identity must come through the other column name.
#[test]
#[ignore = "needs the /mnt/v corpora; run with -- --ignored"]
fn loads_f_prefix_and_ref_basename() {
    let p = canonical_safesyn();
    if !p.exists() {
        panic!(
            "missing canonical safesyn at {p:?} — restore from R2 \
             s3://zentrain/canonical-2026-05-21/train/"
        );
    }
    let g = load_parquet(&p, "safesyn", "ssim2_gpu", 1.0).expect("load safesyn");
    assert_eq!(g.n_features, 372);
    let ref_ids = g
        .ref_ids
        .as_ref()
        .expect("safesyn has ref_basename -> ref_ids must be populated");
    assert_eq!(ref_ids.len(), g.human_scores.len());
    // safesyn is many distortions over comparatively few sources.
    let distinct: HashSet<u32> = ref_ids.iter().copied().collect();
    assert!(
        distinct.len() > 1 && distinct.len() < g.human_scores.len(),
        "expected many rows per ref, got {} refs over {} rows",
        distinct.len(),
        g.human_scores.len()
    );
}

/// Ref ids are assigned in first-seen order, so a given file always
/// yields the same numbering. Determinism here is what lets a bake be
/// reproduced byte-for-byte.
#[test]
#[ignore = "needs the /mnt/v corpora; run with -- --ignored"]
fn ref_ids_are_deterministic_across_loads() {
    let p = hf_corpus();
    if !p.exists() {
        panic!("missing HF corpus at {p:?}");
    }
    let a = load_parquet(&p, "hf", "zensim_score", 1.0).expect("load a");
    let b = load_parquet(&p, "hf", "zensim_score", 1.0).expect("load b");
    assert_eq!(a.ref_ids, b.ref_ids, "ref numbering must be stable");
}
