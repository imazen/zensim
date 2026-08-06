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
    // E-JBU perf mode (protocol: ms/MP of the redistribution pass): `--perf WxH`
    // synthesizes a pair and times the diffmap render with the guided option
    // OFF vs ON, interleaved, medians of 4 per arm.
    #[cfg(feature = "custom-profiles")]
    if args.first().map(String::as_str) == Some("--perf") {
        run_jbu_perf(args.get(1).map(String::as_str).unwrap_or("1024x1024"));
        return;
    }
    if args.len() < 2 {
        eprintln!(
            "usage: diffmap_block_coherence <ref> <dist> [--block N] [--weighting trained|balanced] [--bake <path>] | --perf WxH"
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
    let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake header");
    // CALLER width, not `n_inputs()`. Every use of `n_in` below is
    // caller-space: the vector handed to `score_features_with_profile`, the
    // finite-difference gradient length, the extraction width, and the
    // f0-155 / f156-371 / f372+ block-mass ranges. Since dead-column pruning
    // (`ae852b1b`) a packed 944 bake is a 667-INPUT MODEL THAT STILL ACCEPTS
    // 944 FEATURES, so `n_inputs()` (667) is not a regime and this function
    // would take the "unsupported bake layout" path below — emitting NO M3
    // and NO M3a, silently. That is a selection-visible failure now that a
    // missing M3a means UNMEASURED ⇒ NOT SELECTABLE (campaign appendix E.4),
    // so it must not be reachable for a pruned bake. Identical to
    // `n_inputs()` on every unpruned bake (the transform array is dense).
    // Hazard class: campaign appendix E.9.
    let n_in = model.caller_input_width();
    if model.n_inputs() != n_in {
        println!(
            "  bake is PRUNED: layer0_in_dim={}, caller feature width={n_in} (routing on the latter)",
            model.n_inputs()
        );
    }
    // M3 supports the v1 layouts (n_in ≤ 372), the combined v1+v2 layout
    // (720 = 372 v1 ++ 348 v2), the folded-append 924 regime (f0-155
    // folded basic, f156-371 STRUCTURAL ZEROS, f372-719 v2, f720-923 append),
    // and the folded-append2 944 regime (924 ++ f924-943 append2, SOTA-944).
    // Any other width — e.g. an ext504 bake (156 basic ++ 348 v2) — puts the
    // v2 block at a different offset, so the fold and the dropped-mass are
    // undefined for it. Skip cleanly rather than panic.
    if n_in > 372 && n_in != 720 && n_in != 924 && n_in != 944 {
        println!(
            "  M3 skipped: unsupported bake layout (n_inputs={n_in}; the diffmap fold supports n_inputs ≤ 372, 720, 924, or 944)"
        );
        return;
    }
    let folded924 = n_in == 924 || n_in == 944;
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
    // `mut` is load-bearing only when the v2 branch (below) captures
    // `v2_scratch` mutably; without the feature the closure is Fn and
    // clippy flags the mut — cfg the allow, not the mut.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(unused_mut))]
    let mut feats_of = |dist: &[[u8; 3]]| -> Vec<f64> {
        let ds = RgbSlice::new(dist, w, h);
        #[cfg(feature = "feature-regime-v2")]
        if folded924 {
            // 944 = the same canonical streaming extractor with the append2
            // block toggled on (bit-identical f0..f923; the bf944 executor's
            // exact recipe). Toggle-dependent width matches n_in below.
            // ZENSIM_APPEND2_DSTACT=1 (appendix X, X-I1): honor the BANDVIS
            // dst-activity toggle exactly as `v2_ab_extract` does, so an
            // ON-definition arm's M3a is measured on ON-definition features.
            // Default (env unset) is byte-stable OFF — identical toggles to
            // before this change.
            let dstact_on = n_in == 944
                && std::env::var("ZENSIM_APPEND2_DSTACT")
                    .map(|v| v == "1")
                    .unwrap_or(false);
            let toggles = zensim::feature_v2::V2NewFeatureToggles {
                append2_block: n_in == 944,
                append2_dst_activity: dstact_on,
                ..Default::default()
            };
            return z
                .compute_folded720_append_features_streaming(&rs, &ds, toggles, &mut v2_scratch)
                .expect("folded-append 924/944 features")
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
            "  GRADMASS regions: basic {:.1}% | v1-pool {:.1}% | v2 {:.1}% | append {:.1}% | append2 {:.1}%",
            mass(0..156.min(n_in)),
            if n_in > 156 {
                mass(156..372.min(n_in))
            } else {
                0.0
            },
            if n_in > 372 {
                mass(372..720.min(n_in))
            } else {
                0.0
            },
            if n_in > 720 {
                mass(720..924.min(n_in))
            } else {
                0.0
            },
            if n_in > 924 {
                mass(924..944.min(n_in))
            } else {
                0.0
            },
        );
        if n_in >= 720 {
            const V2_NAMES: [&str; 29] = [
                "SSIM_MEAN",
                "SSIM_DEV2",
                "SSIM_DEV4",
                "ART",
                "DET",
                "MSE",
                "HF_GAIN",
                "HF_LOSS",
                "HF_MAG_LOSS",
                "SSIM_SOFT_PEAK",
                "ART_SOFT_PEAK",
                "DET_SOFT_PEAK",
                "MASKED_SSIM",
                "MASKED_ART",
                "MASKED_DET",
                "MASKED_MSE",
                "IW_SSIM",
                "IW_ART",
                "IW_DET",
                "IW_MSE",
                "PJND_TRANSDUCER",
                "PJND_FRAGILITY",
                "GMS",
                "s23",
                "s24",
                "s25",
                "s26",
                "s27",
                "s28",
            ];
            let mut per_slot = [0f64; 29];
            for sc in 0..4 {
                for ch in 0..3 {
                    #[allow(clippy::needless_range_loop)] // slot builds i AND indexes per_slot
                    for slot in 0..29 {
                        let i = 372 + sc * 87 + ch * 29 + slot;
                        if i < n_in {
                            per_slot[slot] += s[i].abs();
                        }
                    }
                }
            }
            let mut ranked: Vec<(usize, f64)> = per_slot.iter().cloned().enumerate().collect();
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
        #[allow(clippy::needless_range_loop)] // sc builds base AND indexes per_scale
        for sc in 0..4 {
            let base = sc * 39;
            if base + 39 <= n_in.min(156) {
                per_scale[sc] = s[base..base + 39].iter().map(|v| v.abs()).sum();
            }
        }
        println!(
            "  GRADMASS basic-scales: s0={:.1}% s1={:.1}% s2={:.1}% s3={:.1}%",
            100.0 * per_scale[0] / total,
            100.0 * per_scale[1] / total,
            100.0 * per_scale[2] / total,
            100.0 * per_scale[3] / total
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
            let app: f64 = s[720..924.min(n_in)].iter().map(|v| v.abs()).sum();
            println!(
                "  not-yet-foldable append block (f720-923) raw-|s_k| mass: {:.1}%  (no per-pixel fold for the append kernels yet; M3 is blind to it)",
                100.0 * app / total
            );
            if n_in > 924 {
                let app2: f64 = s[924..n_in].iter().map(|v| v.abs()).sum();
                println!(
                    "  not-yet-foldable append2 block (f924-943) raw-|s_k| mass: {:.1}%  (same blind spot, BANDVIS/luma lanes)",
                    100.0 * app2 / total
                );
            }
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
    let model_res = z_map
        .compute_with_diffmap(
            &rs,
            &dist_slice,
            all_signals(DiffmapWeighting::ModelSensitivity(s_basic)),
        )
        .expect("diffmap model-sensitivity");
    let ms_model = t_model.elapsed().as_secs_f64() * 1e3;
    let map_score_off = model_res.score();
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(unused_mut))]
    let mut diff_model = model_res.diffmap().to_vec();

    // ── E-JBU A/B (ZENSIM_JBU_AB=1, protocol 2026-07-30) ────────────────────
    // Second ModelSensitivity map with `guided_coarse_redistribution: true`,
    // computed in the SAME process so s_k / ΔS / M2 are shared exactly between
    // arms. Per-cell mass conservation predicts aligned block sums ≥ 8px
    // unchanged (f32 order only); the drift lines verify it, the per-pixel
    // lines locate where the map actually moved.
    let jbu_ab = std::env::var("ZENSIM_JBU_AB").as_deref() == Ok("1");
    let mut diff_model_jbu: Option<Vec<f32>> = None;
    let mut ms_jbu = 0.0f64;
    if jbu_ab {
        let mut o = all_signals(DiffmapWeighting::ModelSensitivity(s_basic));
        o.guided_coarse_redistribution = true;
        let t_jbu = std::time::Instant::now();
        let jbu_res = z_map
            .compute_with_diffmap(&rs, &dist_slice, o)
            .expect("diffmap model-sensitivity (guided)");
        ms_jbu = t_jbu.elapsed().as_secs_f64() * 1e3;
        // Scalar-drift gate: the render option must not perturb scoring.
        let sd = (jbu_res.score() - map_score_off).abs() / map_score_off.abs().max(1e-12);
        println!(
            "  JBU scalar drift (map-profile score, ON vs OFF): {:.3e} rel ({})",
            sd,
            if jbu_res.score() == map_score_off {
                "bit-identical"
            } else {
                "NONZERO — gate is <=1e-6"
            }
        );
        diff_model_jbu = Some(jbu_res.diffmap().to_vec());
    }

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
    // E-JBU: v1-fold-only copies for the A/B per-pixel report. The combined
    // map below adds the raw v2 fold, which is orders of magnitude larger in
    // VALUE than the v1 signal fold (the C1 "raw add swamps" note, seen from
    // the other side) — per-pixel stats on the combined map would portray the
    // redistribution (which acts inside the v1 fold only) as ~0.01% noise.
    let jbu_v1_pair: Option<(Vec<f32>, Vec<f32>)> = diff_model_jbu
        .as_ref()
        .map(|jm| (diff_model.clone(), jm.clone()));
    #[cfg(feature = "feature-regime-v2")]
    if let Some(v2m) = &v2map {
        for (m, v) in diff_model.iter_mut().zip(v2m.iter()) {
            *m += *v;
        }
        // E-JBU: the IDENTICAL v2 fold-in goes into the guided arm — the A/B
        // isolates the v1 fold's coarse upsample; the v2 fold is out of scope
        // (pre-registered) and shared bit-for-bit between arms.
        if let Some(jm) = &mut diff_model_jbu {
            for (m, v) in jm.iter_mut().zip(v2m.iter()) {
                *m += *v;
            }
        }
    }
    // Without `feature-regime-v2` the fold is compiled out; `v2_grad` is then only
    // read for its diagnostic prints (built above), so acknowledge it here.
    #[cfg(not(feature = "feature-regime-v2"))]
    let _ = &v2_grad;

    // M3a (task #67 C1+C2a): attribution density with TRUE per-feature
    // integrands + summed-area table — block sums via the O(1)
    // rectangle-query API the codec loop would use. C1 covered the basic
    // block (f0-155); C2a extends coverage with exact-integrand densities
    // for the v2 (f372-719) and append (f720-923) blocks
    // (`compute_attribution_density_full`), replacing C1's unit-scaled
    // mean-integrand `compute_v2_diffmap` fold-in.
    let t_attr = std::time::Instant::now();
    let attr = z_map
        .compute_attribution_density(&rs, &dist_slice, s_basic)
        .expect("attribution density");
    let ms_attr = t_attr.elapsed().as_secs_f64() * 1e3;
    let attr_block_basic = attr.block_sums(block);
    #[cfg(feature = "feature-regime-v2")]
    let mut ms_attr_full = 0.0f64;
    #[cfg(feature = "feature-regime-v2")]
    let attr_block: Vec<f64> = if n_in > 372 {
        let t_full = std::time::Instant::now();
        let full = z_map
            .compute_attribution_density_full(&rs, &dist_slice, &s[..n_in])
            .expect("full attribution density");
        ms_attr_full = t_full.elapsed().as_secs_f64() * 1e3;
        full.block_sums(block)
    } else {
        attr_block_basic.clone()
    };
    #[cfg(not(feature = "feature-regime-v2"))]
    let attr_block: Vec<f64> = attr_block_basic.clone();
    #[cfg(feature = "feature-regime-v2")]
    let attr_has_v2 = n_in > 372;
    #[cfg(not(feature = "feature-regime-v2"))]
    let attr_has_v2 = false;
    // v2+append-only block sums (full − basic): the diag's approximation-
    // quality probe against the non-basic true linearization.
    let v2attr_block: Option<Vec<f64>> = if attr_has_v2 {
        Some(
            attr_block
                .iter()
                .zip(attr_block_basic.iter())
                .map(|(f, b)| f - b)
                .collect(),
        )
    } else {
        None
    };

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
        if let Some(v2b) = &v2attr_block {
            let lin_v2app: Vec<f64> = lin_v2
                .iter()
                .zip(lin_append.iter())
                .map(|(a, b)| a + b)
                .collect();
            println!(
                "  ATTRDIAG v2+append attr vs true-lin (v2 {:+.4} | v2+append {:+.4})  (non-basic density approximation quality)",
                spearman(v2b, &lin_v2),
                spearman(v2b, &lin_v2app)
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
    // ── E-JBU A/B report (protocol 2026-07-30) ──────────────────────────────
    if let Some(jm) = &diff_model_jbu {
        let (jbu_block, _) = block_sums(jm, rpx, dpx, w, h, block, bx);
        let m3r = spearman(&jbu_block, &delta_s);
        // Drift: totals + per-block (predicted f32-order-only at aligned ≥8px).
        let tot_off: f64 = diff_model.iter().map(|&v| v as f64).sum();
        let tot_on: f64 = jm.iter().map(|&v| v as f64).sum();
        let tot_rel = (tot_on - tot_off).abs() / tot_off.abs().max(1e-12);
        let mut max_blk_rel = 0.0f64;
        let blk_scale = dmap_model_block
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max)
            .max(1e-12);
        for (a, b) in dmap_model_block.iter().zip(jbu_block.iter()) {
            max_blk_rel = max_blk_rel.max((a - b).abs() / blk_scale);
        }
        // Per-pixel movement: where the render actually changed.
        let mut max_px = 0.0f32;
        let mut changed = 0usize;
        for (a, b) in diff_model.iter().zip(jm.iter()) {
            let d = (a - b).abs();
            max_px = max_px.max(d);
            if d > 1e-9 {
                changed += 1;
            }
        }
        let off64: Vec<f64> = diff_model.iter().map(|&v| v as f64).collect();
        let on64: Vec<f64> = jm.iter().map(|&v| v as f64).collect();
        let px_srocc = spearman(&off64, &on64);
        // Map scale for reading the per-pixel deltas: p50/p99.5 of |OFF map|.
        let mut mag: Vec<f32> = diff_model.iter().map(|v| v.abs()).collect();
        mag.sort_by(f32::total_cmp);
        let p50 = mag[mag.len() / 2];
        let p995 = mag[((mag.len() - 1) as f64 * 0.995) as usize].max(1e-30);
        println!(
            "  M3r SROCC(guided_redistrib_map,    ΔS_bake) = {m3r:+.4}   PLCC {:+.4}   (E-JBU: per-cell mass-conserving; ΔM3 {:+.4})",
            pearson(&jbu_block, &delta_s),
            m3r - m3
        );
        println!(
            "      JBU drift: total {tot_rel:.3e} rel | max block Δ {max_blk_rel:.3e} of max|block| | px changed {:.1}% | max px Δ {max_px:.3e} ({:.2}% of map p99.5 {p995:.3e}; p50 {p50:.3e}) | px SROCC(off,on) {px_srocc:+.6}",
            100.0 * changed as f64 / diff_model.len() as f64,
            100.0 * max_px as f64 / p995 as f64
        );
        // v2-only M3: if the combined map's ranks are set by the raw v2 add
        // (v1 value share ~1e-6..1e-4), M3(v2-only) must ≈ M3(combined). A
        // direct measurement, not an inference from value shares.
        #[cfg(feature = "feature-regime-v2")]
        if let Some(v2m) = &v2map {
            let (v2b, _) = block_sums(v2m, rpx, dpx, w, h, block, bx);
            println!(
                "      JBU v2-only: M3(v2map alone) {:+.4} vs combined {m3:+.4}",
                spearman(&v2b, &delta_s)
            );
        }
        // v1-fold-only A/B: the redistribution acts inside the v1 fold; report
        // its effect at the scale where it lives (the combined map above is
        // value-dominated by the shared raw v2 add for >372 bakes).
        if let Some((v1_off, v1_on)) = &jbu_v1_pair {
            let (v1b_off, _) = block_sums(v1_off, rpx, dpx, w, h, block, bx);
            let (v1b_on, _) = block_sums(v1_on, rpx, dpx, w, h, block, bx);
            let m3_v1_off = spearman(&v1b_off, &delta_s);
            let m3_v1_on = spearman(&v1b_on, &delta_s);
            let mut mag: Vec<f32> = v1_off.iter().map(|v| v.abs()).collect();
            mag.sort_by(f32::total_cmp);
            let v1_p995 = mag[((mag.len() - 1) as f64 * 0.995) as usize].max(1e-30);
            let mut v1_max_px = 0.0f32;
            for (a, b) in v1_off.iter().zip(v1_on.iter()) {
                v1_max_px = v1_max_px.max((a - b).abs());
            }
            let o64: Vec<f64> = v1_off.iter().map(|&v| v as f64).collect();
            let n64: Vec<f64> = v1_on.iter().map(|&v| v as f64).collect();
            println!(
                "      JBU v1-fold-only: M3(off) {m3_v1_off:+.4} -> M3(on) {m3_v1_on:+.4} (Δ {:+.4}) | max px Δ {v1_max_px:.3e} = {:.1}% of v1 p99.5 {v1_p995:.3e} | px SROCC(off,on) {:+.6} | v1 share of combined p99.5: {:.4}%",
                m3_v1_on - m3_v1_off,
                100.0 * v1_max_px as f64 / v1_p995 as f64,
                spearman(&o64, &n64),
                100.0 * v1_p995 as f64 / p995 as f64
            );
        }
        println!(
            "      JBU perf (incl. ref precompute + score): guided {ms_jbu:.1} ms vs base {ms_model:.1} ms"
        );
        // Optional visual A/B: ZENSIM_JBU_DUMP=<prefix> writes p99.5-normalized
        // grayscale PNGs (shared scale so brightness is comparable).
        if let Ok(prefix) = std::env::var("ZENSIM_JBU_DUMP") {
            let mut sorted: Vec<f32> = diff_model.iter().map(|v| v.abs()).collect();
            sorted.sort_by(f32::total_cmp);
            let p995 = sorted[((sorted.len() - 1) as f64 * 0.995) as usize].max(1e-12);
            let dump = |m: &[f32], path: String| {
                let px: Vec<u8> = m
                    .iter()
                    .map(|&v| ((v.abs() / p995) * 255.0).clamp(0.0, 255.0) as u8)
                    .collect();
                image::GrayImage::from_raw(w as u32, h as u32, px)
                    .expect("gray image")
                    .save(&path)
                    .expect("save png");
                eprintln!("      wrote {path}");
            };
            dump(&diff_model, format!("{prefix}_b{block}_off.png"));
            dump(jm, format!("{prefix}_b{block}_on.png"));
            let delta: Vec<f32> = diff_model
                .iter()
                .zip(jm.iter())
                .map(|(a, b)| (a - b).abs() * 4.0) // ×4 so structure is visible
                .collect();
            dump(&delta, format!("{prefix}_b{block}_absdelta_x4.png"));
            // v1-fold-only visuals — where the redistribution actually acts
            // (normalized to the v1 fold's own p99.5, not the v2-swamped
            // combined scale).
            if let Some((v1_off, v1_on)) = &jbu_v1_pair {
                let mut m: Vec<f32> = v1_off.iter().map(|v| v.abs()).collect();
                m.sort_by(f32::total_cmp);
                let v1s = m[((m.len() - 1) as f64 * 0.995) as usize].max(1e-30);
                let dump1 = |mp: &[f32], path: String| {
                    let px: Vec<u8> = mp
                        .iter()
                        .map(|&v| ((v.abs() / v1s) * 255.0).clamp(0.0, 255.0) as u8)
                        .collect();
                    image::GrayImage::from_raw(w as u32, h as u32, px)
                        .expect("gray image")
                        .save(&path)
                        .expect("save png");
                    eprintln!("      wrote {path}");
                };
                dump1(v1_off, format!("{prefix}_b{block}_v1_off.png"));
                dump1(v1_on, format!("{prefix}_b{block}_v1_on.png"));
                let d1: Vec<f32> = v1_off
                    .iter()
                    .zip(v1_on.iter())
                    .map(|(a, b)| (a - b).abs() * 2.0)
                    .collect();
                dump1(&d1, format!("{prefix}_b{block}_v1_absdelta_x2.png"));
            }
        }
    }
    println!(
        "  M3a SROCC(attribution_density,     ΔS_bake) = {m3a:+.4}   PLCC {:+.4}   (true-integrand density + SAT{})",
        pearson(&attr_block, &delta_s),
        if attr_has_v2 {
            format!("; basic-only {m3a_basic:+.4}")
        } else {
            String::new()
        }
    );
    #[cfg(feature = "feature-regime-v2")]
    let full_note = if attr_has_v2 {
        format!(" | full (basic+v2+append) {ms_attr_full:.1} ms")
    } else {
        String::new()
    };
    #[cfg(not(feature = "feature-regime-v2"))]
    let full_note = String::new();
    println!(
        "      perf: attribution basic {ms_attr:.1} ms{full_note} vs ModelSensitivity diffmap {ms_model:.1} ms  (measure-only; C2b optimizes)"
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

/// E-JBU perf protocol (2026-07-30): time the diffmap render with the guided
/// coarse redistribution OFF vs ON on a synthesized pair, interleaved
/// (off,on)×4, report per-arm medians-of-4 and the ON−OFF delta in ms and
/// ms/MP. Reference precompute is done ONCE outside the timed region — the
/// timed call is `compute_with_ref_and_diffmap` (the encoder-loop shape).
#[cfg(feature = "custom-profiles")]
fn run_jbu_perf(spec: &str) {
    let (ws, hs) = spec.split_once('x').expect("--perf WxH");
    let (w, h): (usize, usize) = (ws.parse().expect("W"), hs.parse().expect("H"));
    let mut rpx = vec![[0u8; 3]; w * h];
    let mut dpx = vec![[0u8; 3]; w * h];
    let mut lcg = 0x9E3779B97F4A7C15u64;
    for y in 0..h {
        for x in 0..w {
            let g = (x * 255 / w.max(1)) as f32;
            let t = 26.0 * ((x as f32 * 0.61).sin() * (y as f32 * 0.43).cos());
            let v = (g + t).clamp(0.0, 255.0) as u8;
            rpx[y * w + x] = [v, v.saturating_add(6), v / 2 + 30];
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let n = ((lcg >> 33) & 0xB) as i16 - 5;
            let q = if (x / 8 + y / 8) % 7 == 0 { 12i16 } else { 0 };
            let mut p = rpx[y * w + x];
            for c in &mut p {
                *c = (*c as i16 + n + q).clamp(0, 255) as u8;
            }
            dpx[y * w + x] = p;
        }
    }
    let rs = RgbSlice::new(&rpx, w, h);
    let ds = RgbSlice::new(&dpx, w, h);
    let z = Zensim::new(ZensimProfile::latest_preview());
    let pre = z.precompute_reference(&rs).expect("precompute");
    let opts_off = zensim::DiffmapOptions {
        include_edge_mse: true,
        include_hf: true,
        ..Default::default()
    };
    let opts_on = zensim::DiffmapOptions {
        guided_coarse_redistribution: true,
        ..opts_off
    };
    let mp = (w * h) as f64 / 1e6;
    let mut t_off = Vec::new();
    let mut t_on = Vec::new();
    // Warmup (untimed) then interleaved (off,on) × 4.
    let _ = z.compute_with_ref_and_diffmap(&pre, &ds, opts_off).unwrap();
    let _ = z.compute_with_ref_and_diffmap(&pre, &ds, opts_on).unwrap();
    for _ in 0..4 {
        let t = std::time::Instant::now();
        let a = z.compute_with_ref_and_diffmap(&pre, &ds, opts_off).unwrap();
        t_off.push(t.elapsed().as_secs_f64() * 1e3);
        let t = std::time::Instant::now();
        let b = z.compute_with_ref_and_diffmap(&pre, &ds, opts_on).unwrap();
        t_on.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(
            a.score(),
            b.score(),
            "scalar must be untouched by the render option"
        );
        std::hint::black_box((a.diffmap()[0], b.diffmap()[0]));
    }
    let med = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        (v[1] + v[2]) / 2.0
    };
    let (mo, mn) = (med(&mut t_off), med(&mut t_on));
    println!(
        "JBU perf {w}x{h} ({mp:.2} MP): OFF median {mo:.2} ms ({:.2} ms/MP) | ON median {mn:.2} ms ({:.2} ms/MP) | redistribution pass {:+.2} ms = {:+.3} ms/MP",
        mo / mp,
        mn / mp,
        mn - mo,
        (mn - mo) / mp
    );
    println!("      raw OFF {t_off:.2?}  ON {t_on:.2?}");
}
