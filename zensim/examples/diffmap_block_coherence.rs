//! SPATIAL diffmap↔scalar coherence — the test the pooled diagnostic can't do.
//!
//! The closed loop spends bits per-BLOCK guided by the diffmap, to raise the
//! SCALAR toward a target. So the diffmap must predict WHERE refining raises the
//! scalar. This measures that directly: for each block, copy the reference pixels
//! into the distorted image there (the limit of "spend unlimited bits on this
//! block"), rescore, and record ΔS = score_refined − score_base. A coherent
//! diffmap has `diffmap_block_sum` rank-agreeing with ΔS — it IS the scalar's
//! spatial gradient. We also report SSE-per-block (the codec's PSNR default) as
//! the bar the diffmap must beat.
//!
//! `SROCC(diffmap_block, ΔS)` ≈ 1 → the diffmap points exactly where the scalar
//! rewards bits. If SSE correlates with ΔS as well as the diffmap does, the
//! diffmap adds nothing over the codec default. If the diffmap's SROCC is LOW,
//! it points at the wrong blocks — the incoherence the closed loop can't tolerate.
//!
//! ```sh
//! cargo run --release -p zensim --example diffmap_block_coherence -- <ref> <dist> [--block 32]
//! ```
//!
//! ## `--bake <path>` mode (requires the `custom-profiles` feature)
//!
//! Measures coherence for an ARBITRARY bake (e.g. the 156→128→1 MLP "winner" or
//! an additive basic-156) instead of the shipped profile. The bake is mounted as
//! a `ZensimProfile::Custom` and scored through the production features→score
//! runtime (`score_features_with_profile`: bake forward + spline). Reports three
//! numbers per pair (2026-07-18, the additive-vs-MLP decision instrumentation):
//!
//! - **M1** `SROCC(current_diffmap_block, ΔS_bake)` — how well the SHIPPED
//!   per-pixel diffmap predicts where refining raises THIS bake's scalar.
//! - **M2** `SROCC(Σ_k s_k·Δf_k(block), ΔS_bake)` — the GRADIENT/LINEARIZATION
//!   ceiling: `s_k = ∂score/∂f_k` via central differences at the base image,
//!   applied to the true per-block feature deltas. For an additive bake this is
//!   exact (≈1 up to spline ties); for an MLP it caps ANY gradient-based
//!   diffmap — if M2 is low, no per-pixel map can serve that bake's closed loop.
//! - **SSE** `SROCC(sse_block, ΔS_bake)` — the codec PSNR default bar.
//!
//! ```sh
//! cargo run --release -p zensim --features custom-profiles \
//!   --example diffmap_block_coherence -- <ref> <dist> --bake winner.bin [--block 32]
//! ```

use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

#[cfg(feature = "custom-profiles")]
static BAKE_BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
#[cfg(feature = "custom-profiles")]
fn bake_bytes_static() -> &'static [u8] {
    BAKE_BYTES.get().expect("bake bytes set before profile use")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: diffmap_block_coherence <ref> <dist> [--block N] [--weighting trained|balanced] [--bake <path>]"
        );
        std::process::exit(2);
    }
    let mut block = 32usize;
    let mut weighting = DiffmapWeighting::default(); // Trained (V0_2 weights)
    let mut bake_path: Option<String> = None;
    let mut i = 2;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--block" => block = args[i + 1].parse().unwrap(),
            "--bake" => bake_path = Some(args[i + 1].clone()),
            "--weighting" => {
                weighting = match args[i + 1].as_str() {
                    "balanced" => DiffmapWeighting::Balanced,
                    "trained" => DiffmapWeighting::Trained,
                    other => {
                        eprintln!("unknown weighting {other}");
                        std::process::exit(2);
                    }
                }
            }
            _ => {}
        }
        i += 2;
    }
    let r = image::open(&args[0]).expect("open ref").to_rgb8();
    let d = image::open(&args[1]).expect("open dist").to_rgb8();
    let (w, h) = (r.width() as usize, r.height() as usize);
    assert_eq!(
        (d.width() as usize, d.height() as usize),
        (w, h),
        "size mismatch"
    );
    let rpx: Vec<[u8; 3]> = r.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dpx: Vec<[u8; 3]> = d.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    if let Some(bp) = bake_path {
        #[cfg(feature = "custom-profiles")]
        {
            run_bake_mode(&bp, &rpx, &dpx, w, h, block, weighting);
            return;
        }
        #[cfg(not(feature = "custom-profiles"))]
        {
            let _ = bp;
            eprintln!("--bake requires building with --features custom-profiles");
            std::process::exit(2);
        }
    }

    let z = Zensim::new(ZensimProfile::latest_preview());
    let base = z
        .compute_with_diffmap(
            &RgbSlice::new(&rpx, w, h),
            &RgbSlice::new(&dpx, w, h),
            weighting,
        )
        .expect("diffmap");
    let base_score = base.score();
    let diff = base.diffmap().to_vec();

    let bx = w.div_ceil(block);
    let by = h.div_ceil(block);
    let nblocks = bx * by;
    let (dmap_block, sse_block) = block_sums(&diff, &rpx, &dpx, w, h, block, bx);

    // Ground truth per block = how much refining THIS block raises each candidate scalar.
    //   delta_s      : the FULL (non-additive) zensim scalar — the current metric.
    //   delta_pooled : the ADDITIVE scalar = pooled diffmap (Σ per-pixel weighted signal).
    //                  Its spatial gradient IS the diffmap by construction, so this measures
    //                  the exact-diffmap ceiling the design's additive core targets.
    // The gap between the two = the cost of the current scalar's NON-additivity.
    let pooled_before: f64 = diff.iter().map(|&x| x as f64).sum();
    let mut delta_s = vec![0f64; nblocks];
    let mut delta_pooled = vec![0f64; nblocks];
    let mut scratch = dpx.clone();
    for by_i in 0..by {
        for bx_i in 0..bx {
            let b = by_i * bx + bx_i;
            let (x0, y0) = (bx_i * block, by_i * block);
            let (x1, y1) = ((x0 + block).min(w), (y0 + block).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = rpx[y * w + x];
                }
            }
            let refined = z
                .compute_with_diffmap(
                    &RgbSlice::new(&rpx, w, h),
                    &RgbSlice::new(&scratch, w, h),
                    weighting,
                )
                .expect("diffmap");
            delta_s[b] = refined.score() - base_score;
            let pooled_after: f64 = refined.diffmap().iter().map(|&x| x as f64).sum();
            // refining a block REDUCES pooled error; negate so higher = more-improved,
            // matching dmap_block's polarity (high = high error = high improvement potential).
            delta_pooled[b] = -(pooled_after - pooled_before);
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = dpx[y * w + x];
                }
            }
        }
    }

    let srocc_full = spearman(&dmap_block, &delta_s);
    let srocc_add = spearman(&dmap_block, &delta_pooled);
    let srocc_sse = spearman(&sse_block, &delta_s);
    println!(
        "spatial coherence ({} blocks, {block}px)  base_score={base_score:.2}",
        nblocks
    );
    println!("  additive-scalar target:");
    println!(
        "    SROCC(diffmap_block, Δ additive-scalar) = {srocc_add:+.4}   (exact-gradient ceiling)"
    );
    println!("  current (non-additive) scalar:");
    println!(
        "    SROCC(diffmap_block, ΔS_full)           = {srocc_full:+.4}   PLCC = {:+.4}",
        pearson(&dmap_block, &delta_s)
    );
    println!(
        "    SROCC(SSE_block,     ΔS_full)           = {srocc_sse:+.4}   (codec PSNR default — the bar)"
    );
    println!(
        "  => additive core buys +{:.4} spatial coherence ({:.4} → {:.4}); non-additivity is the {:.0}% gap the design removes",
        srocc_add - srocc_full,
        srocc_full,
        srocc_add,
        (srocc_add - srocc_full) * 100.0
    );
}

/// Per-block sums of the diffmap and of pixel SSE (the codec default selector).
fn block_sums(
    diff: &[f32],
    rpx: &[[u8; 3]],
    dpx: &[[u8; 3]],
    w: usize,
    h: usize,
    block: usize,
    bx: usize,
) -> (Vec<f64>, Vec<f64>) {
    let by = h.div_ceil(block);
    let nblocks = bx * by;
    let mut dmap_block = vec![0f64; nblocks];
    let mut sse_block = vec![0f64; nblocks];
    for y in 0..h {
        for x in 0..w {
            let b = (y / block) * bx + (x / block);
            let p = y * w + x;
            dmap_block[b] += diff[p] as f64;
            let e: f64 = (0..3)
                .map(|c| {
                    let dv = rpx[p][c] as f64 - dpx[p][c] as f64;
                    dv * dv
                })
                .sum();
            sse_block[b] += e;
        }
    }
    (dmap_block, sse_block)
}

/// `--bake` mode: coherence of an arbitrary bake's scalar (2026-07-18).
///
/// ΔS comes from the production features→score runtime on per-block-refined
/// feature vectors; M2's `s_k` is a numerical central-difference gradient of the
/// same runtime at the base features (spline included — it is monotone, so rank
/// is preserved; flat-spline pairs are degenerate and should be mid-dial).
#[cfg(feature = "custom-profiles")]
fn run_bake_mode(
    bake_path: &str,
    rpx: &[[u8; 3]],
    dpx: &[[u8; 3]],
    w: usize,
    h: usize,
    block: usize,
    weighting: DiffmapWeighting,
) {
    use zensim::profile::ProfileParams;
    use zensim::score_features_with_profile;

    let bytes = std::fs::read(bake_path).expect("read bake");
    let n_in = zenpredict::Model::from_bytes(&bytes)
        .expect("parse bake header")
        .n_inputs();
    BAKE_BYTES.set(bytes).expect("bake bytes set once");

    // Feature pipeline sized to the bake: basic-only bakes (n_in ≤ 156) skip the
    // IW pyramid (cheaper per-block loop); 372-input bakes need the full set.
    let params = ProfileParams::builder()
        .mlp(bake_bytes_static)
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .compute_iw_features(n_in > 300)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    let profile = ZensimProfile::Custom {
        params,
        name: "bake-under-test",
    };
    let z = Zensim::new(profile);

    let rs = RgbSlice::new(rpx, w, h);
    let base = z
        .compute_extended_features(&rs, &RgbSlice::new(dpx, w, h))
        .expect("base features");
    // `mut` used only by the v2 concat below (feature-regime-v2 / >372 bake).
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(unused_mut))]
    let mut base_feats = base.features().to_vec();
    // Combined append-only bakes (n_in > 372 = frozen v1-372 ++ v2) need the
    // v2 block concatenated — same dual-compute the extended extractor does.
    // Requires the `feature-regime-v2` build feature (2026-07-19).
    #[cfg(feature = "feature-regime-v2")]
    if n_in > base_feats.len() {
        let v2 = z
            .compute_v2_features(&rs, &RgbSlice::new(dpx, w, h))
            .expect("v2 features (build with --features feature-regime-v2 for combined bakes)");
        base_feats.extend_from_slice(v2.features());
    }
    assert!(
        base_feats.len() >= n_in,
        "bake wants {n_in} inputs, extractor produced {} — for a >372 combined \
         bake, build the example with --features feature-regime-v2",
        base_feats.len()
    );
    let score = |feats: &[f64]| -> f64 {
        score_features_with_profile(profile, &feats[..n_in], w as u32, h as u32)
            .expect("bake forward")
    };
    let base_score = score(&base_feats);

    // s_k = ∂score/∂f_k at the base image (central differences through the full
    // runtime: transforms + MLP/linear forward + spline).
    let mut s = vec![0f64; n_in];
    let mut probe = base_feats.clone();
    for k in 0..n_in {
        let eps = (base_feats[k].abs() * 1e-3).max(1e-5);
        probe[k] = base_feats[k] + eps;
        let up = score(&probe);
        probe[k] = base_feats[k] - eps;
        let dn = score(&probe);
        probe[k] = base_feats[k];
        s[k] = (up - dn) / (2.0 * eps);
    }
    let grad_zero = s.iter().filter(|v| v.abs() < 1e-12).count();

    // Combined append-only bakes (n_in > 372): the scalar path (score + s_k)
    // works, but the runtime diffmap generator (`compute_with_diffmap`, used
    // for M1/M1b/M3 below) is hardwired to the ≤372 feature space — it cannot
    // fold the v2 block (f372+) into the per-pixel map yet. Rather than panic,
    // report the scalar-side coherence diagnostics and point at the corpus-
    // level foldable-mass proxy. M2 (linearization ceiling) is ≈1.0 for any
    // LeakyReLU MLP (piecewise-linear ⇒ exact local gradient), so the open
    // question is purely M3-deployed = whether the fold, once extended to read
    // v2, reaches that ceiling. See benchmarks/v2_trainability_ab_2026-07-19.md
    // (v2_combined_steer_mass.py: 100% foldable for the coherence-maxed model).
    // Combined append-only bake (n_in > 372 = v1-372 ++ v2): the v2 block's
    // gradient `s[372..]` now folds into a per-pixel map via
    // `Zensim::compute_v2_diffmap` (task #48), so M3 below reads v2. The v1
    // masked/iw/peak (f156-371) remain non-spatializable in the v1 fold, so
    // that share of the gradient still can't be deployed — reported here.
    let v2_grad: Option<Vec<f64>> = if n_in > 372 {
        let total: f64 = s.iter().map(|v| v.abs()).sum::<f64>().max(1e-30);
        let nonspat: f64 = s[156..372].iter().map(|v| v.abs()).sum();
        println!(
            "combined bake (n_inputs={n_in})  base_score={base_score:.2}  grad_zero={grad_zero}/{n_in}"
        );
        println!(
            "  non-deployable v1 block (f156-371) raw-|s_k| mass: {:.1}%  (the v1 masked/iw/peak the v1 fold can't spatialize; the v2 versions at f372+ DO fold)",
            100.0 * nonspat / total
        );
        // NEGATE the gradient to match the v1 `model_sensitivity_weights`
        // convention (`w = -s`): with `s_k<0` for a quality metric, `-s_k>0`
        // makes the map high where refining raises the score (the "refine-here"
        // polarity M1/M3 share). compute_v2_diffmap folds Σ w·M with whatever
        // weights it's given; the steering weight is `-∂score/∂f`.
        Some(s[372..n_in].iter().map(|&x| -x).collect::<Vec<f64>>())
    } else {
        None
    };

    // Three per-pixel diffmaps for the same pair:
    //   M1  — the CURRENT shipped default (ssim-only signals, profile weighting):
    //         what a codec consumes today.
    //   M1b — same weight source, ALL per-pixel signals (edge/mse/hf on):
    //         isolates the signal-set effect from the weight-source effect.
    //   M3  — ModelSensitivity: signals weighted by THIS bake's own s_k
    //         (the deployable approximation of the exact gradient).
    let dist_slice = RgbSlice::new(dpx, w, h);
    // The per-pixel MAPS are the deployed v1 error signals (weighted); they do
    // not depend on the bake's scalar. `compute_with_diffmap` on the custom
    // bake would try to SCORE it (needs all n_in features; the streaming path
    // only extracts ≤372 → fails for a 720 bake), so the maps are built with a
    // v1 profile. The bake (`z`) is used only for the ground-truth `delta_s` /
    // `s_k` below. The v2 contribution is added separately via `compute_v2_diffmap`.
    let z_map = Zensim::new(ZensimProfile::latest_preview());
    let diff = z_map
        .compute_with_diffmap(&rs, &dist_slice, weighting)
        .expect("diffmap")
        .diffmap()
        .to_vec();
    let all_signals = |wt: DiffmapWeighting| zensim::DiffmapOptions {
        weighting: wt,
        include_edge_mse: true,
        include_hf: true,
        ..Default::default()
    };
    let diff_all = z_map
        .compute_with_diffmap(&rs, &dist_slice, all_signals(weighting))
        .expect("diffmap all-signals")
        .diffmap()
        .to_vec();
    // s_k over the basic block only — the per-pixel machinery spatializes f0..155.
    let s_basic: &'static [f64] = Box::leak(s[..n_in.min(156)].to_vec().into_boxed_slice());
    // `mut` is used only by the v2 fold below (a >372 combined bake). Without
    // `feature-regime-v2` that fold is compiled out and the binding stays immutable.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(unused_mut))]
    let mut diff_model = z_map
        .compute_with_diffmap(
            &rs,
            &dist_slice,
            all_signals(DiffmapWeighting::ModelSensitivity(s_basic)),
        )
        .expect("diffmap model-sensitivity")
        .diffmap()
        .to_vec();

    // task #48: for a v1++v2 bake, add the v2 block's per-pixel contribution
    // (its gradient s[372..] folded through the additive v2 families) to the
    // v1 model-sensitivity map — the deployed map now reads v2.
    //
    // `compute_v2_diffmap` (and `compute_v2_features` above) are gated behind
    // `feature-regime-v2` — the impl reads the v2 family bank that only exists in
    // that regime. The sibling `compute_v2_features` calls are already cfg-gated;
    // this consumer was NOT, so `--features custom-profiles` alone failed to build
    // (E0599: method not found). It is only reachable for a >372 bake, which cannot
    // occur without the regime anyway (the `base_feats.len() >= n_in` assert above
    // fires first when the v2 block is absent), so gating it is sound.
    #[cfg(feature = "feature-regime-v2")]
    if let Some(v2g) = &v2_grad {
        let v2map = z_map
            .compute_v2_diffmap(&rs, &dist_slice, v2g)
            .expect("v2 diffmap");
        assert_eq!(v2map.len(), diff_model.len());
        for (m, v) in diff_model.iter_mut().zip(v2map.iter()) {
            *m += *v;
        }
    }
    // Without `feature-regime-v2` the fold is compiled out; `v2_grad` is then only
    // read for its diagnostic prints (built above), so acknowledge it here.
    #[cfg(not(feature = "feature-regime-v2"))]
    let _ = &v2_grad;

    let bx = w.div_ceil(block);
    let by = h.div_ceil(block);
    let nblocks = bx * by;
    let (dmap_block, sse_block) = block_sums(&diff, rpx, dpx, w, h, block, bx);
    let (dmap_all_block, _) = block_sums(&diff_all, rpx, dpx, w, h, block, bx);
    let (dmap_model_block, _) = block_sums(&diff_model, rpx, dpx, w, h, block, bx);

    let mut delta_s = vec![0f64; nblocks];
    let mut lin_pred = vec![0f64; nblocks];
    let mut scratch = dpx.to_vec();
    for by_i in 0..by {
        for bx_i in 0..bx {
            let b = by_i * bx + bx_i;
            let (x0, y0) = (bx_i * block, by_i * block);
            let (x1, y1) = ((x0 + block).min(w), (y0 + block).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = rpx[y * w + x];
                }
            }
            let rr = z
                .compute_extended_features(&rs, &RgbSlice::new(&scratch, w, h))
                .expect("refined features");
            // `mut` used only by the v2 concat below (feature-regime-v2).
            #[cfg_attr(not(feature = "feature-regime-v2"), allow(unused_mut))]
            let mut rfeats = rr.features().to_vec();
            // Concat the refined v2 block for a >372 combined bake — same dual-
            // compute as base_feats, so `score`/`s_k` see the full n_in vector.
            #[cfg(feature = "feature-regime-v2")]
            if n_in > rfeats.len() {
                let v2 = z
                    .compute_v2_features(&rs, &RgbSlice::new(&scratch, w, h))
                    .expect("refined v2 features");
                rfeats.extend_from_slice(v2.features());
            }
            delta_s[b] = score(&rfeats) - base_score;
            lin_pred[b] = (0..n_in).map(|k| s[k] * (rfeats[k] - base_feats[k])).sum();
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = dpx[y * w + x];
                }
            }
        }
    }

    let m1 = spearman(&dmap_block, &delta_s);
    let m1b = spearman(&dmap_all_block, &delta_s);
    let m3 = spearman(&dmap_model_block, &delta_s);
    let m2 = spearman(&lin_pred, &delta_s);
    let sse = spearman(&sse_block, &delta_s);
    println!(
        "bake spatial coherence ({nblocks} blocks, {block}px)  bake={bake_path}  n_inputs={n_in}  base_score={base_score:.2}  grad_zero={grad_zero}/{n_in}"
    );
    println!(
        "  M1  SROCC(shipped_default_diffmap, ΔS_bake) = {m1:+.4}   PLCC {:+.4}   (ssim-only, profile weights)",
        pearson(&dmap_block, &delta_s)
    );
    println!(
        "  M1b SROCC(all_signals_diffmap,     ΔS_bake) = {m1b:+.4}   PLCC {:+.4}   (edge/mse/hf on, profile weights)",
        pearson(&dmap_all_block, &delta_s)
    );
    println!(
        "  M3  SROCC(model_sensitivity_map,   ΔS_bake) = {m3:+.4}   PLCC {:+.4}   (bake's own s_k — deployable)",
        pearson(&dmap_model_block, &delta_s)
    );
    println!(
        "  M2  SROCC(grad_lin_pred,           ΔS_bake) = {m2:+.4}   PLCC {:+.4}   (gradient/linearization ceiling)",
        pearson(&lin_pred, &delta_s)
    );
    println!(
        "      SROCC(SSE_block,               ΔS_bake) = {sse:+.4}   (codec PSNR default — the bar)"
    );
}

fn rank(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    for (k, &ix) in idx.iter().enumerate() {
        r[ix] = k as f64;
    }
    r
}
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&rank(a), &rank(b))
}
