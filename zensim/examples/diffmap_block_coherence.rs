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
    // M3 supports the v1 layouts (n_in ≤ 372), the combined v1+v2 layout
    // (720 = 372 v1 ++ 348 v2), and the folded-append 924 regime (f0-155
    // folded basic, f156-371 STRUCTURAL ZEROS, f372-719 v2, f720-923 append).
    // Any other width — e.g. an ext504 bake (156 basic ++ 348 v2) — puts the
    // v2 block at a different offset, so the fold and the dropped-mass are
    // undefined for it. Skip cleanly rather than panic.
    if n_in > 372 && n_in != 720 && n_in != 924 {
        println!(
            "  M3 skipped: unsupported bake layout (n_inputs={n_in}; the diffmap fold supports n_inputs ≤ 372, 720, or 924)"
        );
        return;
    }
    let folded924 = n_in == 924;
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
    // Feature extraction closure, regime-matched to the bake:
    //  - 924 (folded+append): the CANONICAL streaming extractor — bit-identical
    //    to the ext924 parquets, INCLUDING the f156-371 structural zeros. Using
    //    the extended path here would feed real iw/masked values into weights
    //    that only ever saw zeros in training (noise injection, wrong regime).
    //  - otherwise: extended (+v2 concat for a >372 combined bake).
    #[cfg(feature = "feature-regime-v2")]
    let mut v2_scratch = zensim::feature_v2::V2Scratch::new();
    let mut feats_of = |dist: &[[u8; 3]]| -> Vec<f64> {
        let ds = RgbSlice::new(dist, w, h);
        #[cfg(feature = "feature-regime-v2")]
        if folded924 {
            return z
                .compute_folded720_append_features_streaming(
                    &rs,
                    &ds,
                    zensim::feature_v2::V2NewFeatureToggles::default(),
                    &mut v2_scratch,
                )
                .expect("folded-append 924 features")
                .features()
                .to_vec();
        }
        let base = z
            .compute_extended_features(&rs, &ds)
            .expect("base features");
        #[cfg_attr(not(feature = "feature-regime-v2"), allow(unused_mut))]
        let mut feats = base.features().to_vec();
        #[cfg(feature = "feature-regime-v2")]
        if n_in > feats.len() {
            let v2 = z
                .compute_v2_features(&rs, &ds)
                .expect("v2 features (build with --features feature-regime-v2 for combined bakes)");
            feats.extend_from_slice(v2.features());
        }
        feats
    };
    let base_feats = feats_of(dpx);
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
        // Folded-924: f156-371 are STRUCTURAL ZEROS in this regime — never
        // probed (the deployed runtime cannot vary them; probing them measures
        // untrained-weight noise, not the model).
        if folded924 && (156..372).contains(&k) {
            continue;
        }
        let eps = (base_feats[k].abs() * 1e-3).max(1e-5);
        probe[k] = base_feats[k] + eps;
        let up = score(&probe);
        probe[k] = base_feats[k] - eps;
        let dn = score(&probe);
        probe[k] = base_feats[k];
        s[k] = (up - dn) / (2.0 * eps);
    }
    let grad_zero = s.iter().filter(|v| v.abs() < 1e-12).count();

    // ── gradient-mass diagnostic (ZENSIM_GRAD_MASS=1) ────────────────────
    // Where does THIS bake's |s_k| live? The M2≈0.99-vs-M3≈0.2 gap at 924
    // means the scalar is steerable but the per-pixel FOLD misses the mass —
    // this print locates it (region / v2-slot / top indices) so the fold gap
    // is attributable to named families instead of guessed at.
    if std::env::var("ZENSIM_GRAD_MASS").as_deref() == Ok("1") {
        let total: f64 = s.iter().map(|v| v.abs()).sum::<f64>().max(1e-30);
        let mass = |r: core::ops::Range<usize>| -> f64 {
            if r.end <= n_in {
                100.0 * s[r].iter().map(|v| v.abs()).sum::<f64>() / total
            } else {
                0.0
            }
        };
        println!(
            "  GRADMASS regions: basic {:.1}% | v1-pool {:.1}% | v2 {:.1}% | append {:.1}%",
            mass(0..156.min(n_in)),
            if n_in > 156 { mass(156..372.min(n_in)) } else { 0.0 },
            if n_in > 372 { mass(372..720.min(n_in)) } else { 0.0 },
            if n_in > 720 { mass(720..924.min(n_in)) } else { 0.0 },
        );
        if n_in >= 720 {
            const V2_NAMES: [&str; 29] = [
                "SSIM_MEAN", "SSIM_DEV2", "SSIM_DEV4", "ART", "DET", "MSE", "HF_GAIN", "HF_LOSS",
                "HF_MAG_LOSS", "SSIM_SOFT_PEAK", "ART_SOFT_PEAK", "DET_SOFT_PEAK", "MASKED_SSIM",
                "MASKED_ART", "MASKED_DET", "MASKED_MSE", "IW_SSIM", "IW_ART", "IW_DET", "IW_MSE",
                "PJND_TRANSDUCER", "PJND_FRAGILITY", "GMS", "s23", "s24", "s25", "s26", "s27",
                "s28",
            ];
            let mut per_slot = [0f64; 29];
            for sc in 0..4 {
                for ch in 0..3 {
                    for slot in 0..29 {
                        let i = 372 + sc * 87 + ch * 29 + slot;
                        if i < n_in {
                            per_slot[slot] += s[i].abs();
                        }
                    }
                }
            }
            let mut ranked: Vec<(usize, f64)> =
                per_slot.iter().cloned().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let line: Vec<String> = ranked
                .iter()
                .take(8)
                .map(|&(i, m)| format!("{}={:.1}%", V2_NAMES[i], 100.0 * m / total))
                .collect();
            println!("  GRADMASS v2-slots(top8): {}", line.join(" "));
        }
        let mut top: Vec<(usize, f64)> = s.iter().map(|v| v.abs()).enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let line: Vec<String> = top
            .iter()
            .take(12)
            .map(|&(i, m)| format!("f{}={:.1}%", i, 100.0 * m / total))
            .collect();
        println!("  GRADMASS top-idx: {}", line.join(" "));
        // Basic mass per scale — the fold's spatial resolution is scale-blended
        // by mass, so coarse-scale concentration = a blurry map (M3 mechanism).
        let mut per_scale = [0f64; 4];
        for sc in 0..4 {
            let base = sc * 39;
            if base + 39 <= n_in.min(156) {
                per_scale[sc] = s[base..base + 39].iter().map(|v| v.abs()).sum();
            }
        }
        println!(
            "  GRADMASS basic-scales: s0={:.1}% s1={:.1}% s2={:.1}% s3={:.1}%",
            100.0 * per_scale[0] / total, 100.0 * per_scale[1] / total,
            100.0 * per_scale[2] / total, 100.0 * per_scale[3] / total
        );
    }

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
    // Non-spatializable v1 mass — the share of THIS bake's gradient on the
    // f156-371 masked/iw/peak block, which the v1 fold cannot spatialize into the
    // per-pixel map, so M3 is STRUCTURALLY BLIND to it. Reported for EVERY bake
    // (0.0% for a basic-156 bake; the real fraction for a 372 bake that leans on
    // iw/masked; the v1 share for a 720 bake whose v2 f372+ versions DO fold).
    // This is the number that lets a LOW M3 be read correctly: a bake with high
    // dropped-mass has a structurally-capped M3 (it uses pooled features the map
    // can't carry), which is a DIFFERENT thing from an incoherent map. Widened +
    // always-emitted 2026-07-26 (stats review §Rec-8); previously only n_in>372.
    {
        let total: f64 = s.iter().map(|v| v.abs()).sum::<f64>().max(1e-30);
        let hi = n_in.min(372);
        let nonspat: f64 = if hi > 156 {
            s[156..hi].iter().map(|v| v.abs()).sum()
        } else {
            0.0
        };
        println!(
            "  non-deployable v1 block (f156-371) raw-|s_k| mass: {:.1}%  (masked/iw/peak the v1 fold can't spatialize; M3 is blind to it{})",
            100.0 * nonspat / total,
            if n_in > 372 {
                "; the v2 versions at f372+ DO fold"
            } else {
                ""
            }
        );
        // Folded-924: the append block (f720-923) has no per-pixel fold yet —
        // its gradient share is a SECOND blind spot for M3, reported so a low
        // M3 on a 924 bake is read against append reliance, not miscoherence.
        if folded924 {
            let app: f64 = s[720..924].iter().map(|v| v.abs()).sum();
            println!(
                "  not-yet-foldable append block (f720-923) raw-|s_k| mass: {:.1}%  (no per-pixel fold for the append kernels yet; M3 is blind to it)",
                100.0 * app / total
            );
        }
    }
    let v2_grad: Option<Vec<f64>> = if n_in > 372 {
        println!(
            "combined bake (n_inputs={n_in})  base_score={base_score:.2}  grad_zero={grad_zero}/{n_in}"
        );
        // NEGATE the gradient to match the v1 `model_sensitivity_weights`
        // convention (`w = -s`): with `s_k<0` for a quality metric, `-s_k>0`
        // makes the map high where refining raises the score (the "refine-here"
        // polarity M1/M3 share). compute_v2_diffmap folds Σ w·M with whatever
        // weights it's given; the steering weight is `-∂score/∂f`.
        Some(s[372..720].iter().map(|&x| -x).collect::<Vec<f64>>())
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
    let t_model = std::time::Instant::now();
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
    let ms_model = t_model.elapsed().as_secs_f64() * 1e3;

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
    // The v2 map is computed ONCE and shared: added into the M3 signal fold
    // below (the pre-existing behavior) AND into the M3a attribution density
    // (same fold-in, per task #67 C1).
    #[cfg(feature = "feature-regime-v2")]
    let v2map: Option<Vec<f32>> = v2_grad.as_ref().map(|v2g| {
        let m = z_map
            .compute_v2_diffmap(&rs, &dist_slice, v2g)
            .expect("v2 diffmap");
        assert_eq!(m.len(), diff_model.len());
        m
    });
    #[cfg(feature = "feature-regime-v2")]
    if let Some(v2m) = &v2map {
        for (m, v) in diff_model.iter_mut().zip(v2m.iter()) {
            *m += *v;
        }
    }
    // Without `feature-regime-v2` the fold is compiled out; `v2_grad` is then only
    // read for its diagnostic prints (built above), so acknowledge it here.
    #[cfg(not(feature = "feature-regime-v2"))]
    let _ = &v2_grad;

    // M3a (task #67 C1): attribution density with TRUE per-feature integrands
    // + summed-area table — block sums via the O(1) rectangle-query API the
    // codec loop would use. Same s_k, same v2 fold-in as M3; the difference
    // is the fold itself (absolute integrand attribution vs the normalized
    // mass-blended signal fold).
    let t_attr = std::time::Instant::now();
    let attr = z_map
        .compute_attribution_density(&rs, &dist_slice, s_basic)
        .expect("attribution density");
    let ms_attr = t_attr.elapsed().as_secs_f64() * 1e3;
    let attr_block_basic = attr.block_sums(block);
    // v2 fold-in, UNIT-CORRECT (unlike M3's raw add, which is fine for the
    // normalized signal fold but would swamp the score-unit density by
    // orders of magnitude): the v2 channel-scale fold is linear in the
    // per-(scale,ch,slot) weights, replicate-upsamples coarse scales, and
    // its family maps pool to the features by (weighted) MEAN over the
    // scale plane (`v2_diffmap_block_pool_matches_features`). Scaling each
    // weight by 1/(w·h) = 1/(N_sc·4^sc) (exact for even pyramid dims)
    // therefore turns the fold into a v2 attribution density in score
    // units — a mean-integrand approximation; the fold's non-additive v2
    // families (dev/soft-peak/fragility/edge-width) stay excluded, and the
    // append block f720-923 stays blind.
    #[cfg(feature = "feature-regime-v2")]
    let mut v2attr_block: Option<Vec<f64>> = None;
    #[cfg(feature = "feature-regime-v2")]
    let attr_block: Vec<f64> = if let Some(v2g) = &v2_grad {
        let inv_n0 = 1.0 / (w as f64 * h as f64);
        let v2g_attr: Vec<f64> = v2g.iter().map(|&x| x * inv_n0).collect();
        let v2attr = z_map
            .compute_v2_diffmap(&rs, &dist_slice, &v2g_attr)
            .expect("v2 attribution diffmap");
        v2attr_block =
            Some(zensim::AttributionResult::from_density(v2attr.clone(), w, h).block_sums(block));
        let mut d = attr.density().to_vec();
        for (a, v) in d.iter_mut().zip(v2attr.iter()) {
            *a += *v;
        }
        zensim::AttributionResult::from_density(d, w, h).block_sums(block)
    } else {
        attr_block_basic.clone()
    };
    #[cfg(not(feature = "feature-regime-v2"))]
    let attr_block: Vec<f64> = attr_block_basic.clone();
    #[cfg(feature = "feature-regime-v2")]
    let attr_has_v2 = v2_grad.is_some();
    #[cfg(not(feature = "feature-regime-v2"))]
    let attr_has_v2 = false;

    let bx = w.div_ceil(block);
    let by = h.div_ceil(block);
    let nblocks = bx * by;
    let (dmap_block, sse_block) = block_sums(&diff, rpx, dpx, w, h, block, bx);
    let (dmap_all_block, _) = block_sums(&diff_all, rpx, dpx, w, h, block, bx);
    let (dmap_model_block, _) = block_sums(&diff_model, rpx, dpx, w, h, block, bx);

    let mut delta_s = vec![0f64; nblocks];
    let mut lin_pred = vec![0f64; nblocks];
    // ZENSIM_ATTR_DIAG=1: class-restricted TRUE linearizations, to decompose
    // an M3a gap into (mass outside basic) vs (density approximation error).
    let attr_diag = std::env::var("ZENSIM_ATTR_DIAG").as_deref() == Ok("1");
    let mut lin_basic = vec![0f64; nblocks];
    let mut lin_mse = vec![0f64; nblocks];
    let mut lin_ssim = vec![0f64; nblocks];
    let mut lin_edge = vec![0f64; nblocks];
    let mut lin_hf = vec![0f64; nblocks];
    let mut lin_v2 = vec![0f64; nblocks];
    let mut lin_append = vec![0f64; nblocks];
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
            // Regime-matched refined features (extended+v2 concat, or the
            // canonical folded-append 924 path) — same closure as base_feats.
            let rfeats = feats_of(&scratch);
            delta_s[b] = score(&rfeats) - base_score;
            lin_pred[b] = (0..n_in).map(|k| s[k] * (rfeats[k] - base_feats[k])).sum();
            if attr_diag {
                for k in 0..n_in.min(156) {
                    let d = s[k] * (rfeats[k] - base_feats[k]);
                    lin_basic[b] += d;
                    match k % 13 {
                        0..=2 => lin_ssim[b] += d,
                        3..=8 => lin_edge[b] += d,
                        9 => lin_mse[b] += d,
                        _ => lin_hf[b] += d,
                    }
                }
                for k in 372..n_in.min(720) {
                    lin_v2[b] += s[k] * (rfeats[k] - base_feats[k]);
                }
                for k in 720..n_in.min(924) {
                    lin_append[b] += s[k] * (rfeats[k] - base_feats[k]);
                }
            }
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
    let m3a = spearman(&attr_block, &delta_s);
    let m3a_basic = spearman(&attr_block_basic, &delta_s);
    let m2 = spearman(&lin_pred, &delta_s);
    let sse = spearman(&sse_block, &delta_s);
    if attr_diag {
        println!(
            "  ATTRDIAG true-lin restrictions vs ΔS: basic {:+.4} | mse {:+.4} | ssim {:+.4} | edge {:+.4} | hf {:+.4}",
            spearman(&lin_basic, &delta_s),
            spearman(&lin_mse, &delta_s),
            spearman(&lin_ssim, &delta_s),
            spearman(&lin_edge, &delta_s),
            spearman(&lin_hf, &delta_s),
        );
        println!(
            "  ATTRDIAG attr_basic vs true-lin_basic SROCC {:+.4}  (density approximation quality; 1.0 = density == true basic linearization)",
            spearman(&attr_block_basic, &lin_basic)
        );
        let lin_nonbasic: Vec<f64> = lin_pred
            .iter()
            .zip(lin_basic.iter())
            .map(|(p, b)| p - b)
            .collect();
        println!(
            "  ATTRDIAG true-lin non-basic (v2+append) vs ΔS SROCC {:+.4}; attr_block(+v2) vs true-lin_full SROCC {:+.4}",
            spearman(&lin_nonbasic, &delta_s),
            spearman(&attr_block, &lin_pred)
        );
        let lin_no_append: Vec<f64> = lin_basic
            .iter()
            .zip(lin_v2.iter())
            .map(|(a, b)| a + b)
            .collect();
        println!(
            "  ATTRDIAG true-lin v2-only vs ΔS {:+.4} | append-only vs ΔS {:+.4} | basic+v2 (append-blind ceiling) vs ΔS {:+.4}",
            spearman(&lin_v2, &delta_s),
            spearman(&lin_append, &delta_s),
            spearman(&lin_no_append, &delta_s),
        );
        #[cfg(feature = "feature-regime-v2")]
        if let Some(v2b) = &v2attr_block {
            println!(
                "  ATTRDIAG v2attr_block vs true-lin_v2 SROCC {:+.4}  (v2 fold-in approximation quality)",
                spearman(v2b, &lin_v2)
            );
        }
    }
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
        "  M3a SROCC(attribution_density,     ΔS_bake) = {m3a:+.4}   PLCC {:+.4}   (true-integrand density + SAT{})",
        pearson(&attr_block, &delta_s),
        if attr_has_v2 {
            format!("; basic-only {m3a_basic:+.4}")
        } else {
            String::new()
        }
    );
    println!(
        "      perf: attribution build {ms_attr:.1} ms vs ModelSensitivity diffmap {ms_model:.1} ms  (C1 reports; C2 bar is <=1.1x)"
    );
    println!(
        "  M2  SROCC(grad_lin_pred,           ΔS_bake) = {m2:+.4}   PLCC {:+.4}   (gradient/linearization ceiling)",
        pearson(&lin_pred, &delta_s)
    );
    println!(
        "      SROCC(SSE_block,               ΔS_bake) = {sse:+.4}   (codec PSNR default — the bar)"
    );
}

/// Tie-correct ranks (midrank averaging over equal values) — a verified-equivalent
/// mirror of `zenstats::panel::ranks`. This example lives in the `zensim` crate,
/// which does not depend on `zenmetrics`/`zenstats`, so the stat is mirrored here
/// rather than adding a cross-repo dev-dep for one diagnostic binary. The previous
/// body assigned the raw sort position, which is WRONG for ties: block sums tie
/// often (flat/clamped blocks share a diffmap value), and distinct ranks on tied
/// inputs bias the Spearman that M3 reports. Midrank matches the canonical panel.
fn rank(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0; // midrank over the tie block [i, j)
        for &ix in &idx[i..j] {
            r[ix] = avg;
        }
        i = j;
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
