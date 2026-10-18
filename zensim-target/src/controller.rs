//! Shared controller implementation; unit conversion does not change its decisions.
use crate::{CodecKind, ProbeRecord, TargetResult, TargetSpec, codec};
use anyhow::{Context, Result, bail};
use zensim::{RgbSlice, ZensimProfile};

#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    rgb: &[u8],
    width: u32,
    height: u32,
    codec: CodecKind,
    spec: TargetSpec,
    profile: Option<ZensimProfile>,
    backend: &dyn codec::CodecBackend,
    mut score: impl FnMut(&RgbSlice<'_>, &RgbSlice<'_>) -> Result<f32>,
) -> Result<TargetResult> {
    if width == 0
        || height == 0
        || !spec.target.is_finite()
        || !spec.tolerance.is_finite()
        || spec.tolerance < 0.0
        || spec.max_iterations == 0
    {
        bail!(
            "nonzero dimensions/pass budget, finite target, and finite nonnegative tolerance required"
        );
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .context("image dimensions overflow")?;
    if rgb.len() != expected {
        bail!(
            "rgb buffer length {} != width*height*3 = {}",
            rgb.len(),
            expected
        );
    }

    let (q_lo_native, q_hi_native) = backend.quality_range();
    if !q_lo_native.is_finite() || !q_hi_native.is_finite() || q_lo_native > q_hi_native {
        bail!("codec knob range must be finite and ordered");
    }
    let inverted = backend.lower_quality_means_higher_score();
    if let Some(seed) = spec.seed
        && (!seed.knob.is_finite()
            || seed.knob < q_lo_native
            || seed.knob > q_hi_native
            || !seed.score_per_knob.is_finite()
            || seed.score_per_knob.abs() < 1e-6
            || (seed.score_per_knob < 0.0) != inverted)
    {
        bail!("seed must be in the codec range with a finite, correctly oriented nonzero slope");
    }

    // Initial probe at the midpoint of the range.
    let src_pixels: &[[u8; 3]] = bytemuck::cast_slice(rgb);
    let scratch_src = RgbSlice::try_new(src_pixels, width as usize, height as usize)
        .map_err(|e| anyhow::anyhow!("rgb slice for reference image: {e:?}"))?;

    let mut probes: Vec<ProbeRecord> = Vec::new();
    let mut q_lo = q_lo_native;
    let mut q_hi = q_hi_native;
    let mut best_idx: Option<usize> = None;
    let mut best_encoded: Vec<u8> = Vec::new();

    // A seeded run uses the training slope for its second probe and measured
    // secants thereafter. Unseeded runs retain the historical environment
    // switch. Safeguarding keeps proposals inside the current interval; it
    // does not prove convergence or monotonicity of a real codec/scorer pair.
    let use_secant = spec.seed.is_some()
        || std::env::var("ZENSIM_TARGET_SECANT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let mut sec_a: Option<(f32, f32)> = None; // (q, f=achieved−target), older
    let mut sec_b: Option<(f32, f32)> = None; // newer

    let budget = if codec.is_lossless_only() {
        1
    } else {
        spec.max_iterations
    };
    for iter in 0..budget {
        let q_bisect = (q_lo + q_hi) * 0.5;
        let q_mid = if let Some(seed) = spec.seed
            && iter == 0
        {
            seed.knob
        } else if let Some(seed) = spec.seed
            && iter == 1
        {
            let previous = &probes[0];
            let proposed =
                previous.knob + (spec.target - previous.achieved_score) / seed.score_per_knob;
            guarded_knob(proposed, q_lo, q_hi, q_bisect)
        } else if use_secant {
            if let (Some((qa, fa)), Some((qb, fb))) = (sec_a, sec_b) {
                let denom = fb - fa;
                let slope = denom / (qb - qa);
                if denom.abs() > 1e-6 && slope.is_finite() && (slope < 0.0) == inverted {
                    let qs = qb - fb * (qb - qa) / denom;
                    guarded_knob(qs, q_lo, q_hi, q_bisect)
                } else {
                    q_bisect
                }
            } else {
                q_bisect
            }
        } else {
            q_bisect
        };
        let (encoded, decoded_rgb) = backend
            .encode_decode(rgb, width, height, q_mid)
            .with_context(|| format!("codec {codec:?} encode/decode at knob {q_mid:.3}"))?;
        if decoded_rgb.len() != expected {
            bail!(
                "codec {codec:?} decode returned {} bytes; expected {}",
                decoded_rgb.len(),
                expected
            );
        }
        let dst_pixels: &[[u8; 3]] = bytemuck::cast_slice(&decoded_rgb);
        let dst = RgbSlice::try_new(dst_pixels, width as usize, height as usize)
            .map_err(|e| anyhow::anyhow!("rgb slice for decoded image: {e:?}"))?;
        let achieved =
            score(&scratch_src, &dst).with_context(|| format!("zensim compute on iter {iter}"))?;
        if !achieved.is_finite() {
            bail!("nonfinite score on iteration {iter}");
        }
        if use_secant {
            sec_a = sec_b;
            sec_b = Some((q_mid, achieved - spec.target));
        }

        probes.push(ProbeRecord {
            iteration: iter,
            knob: q_mid,
            achieved_score: achieved,
            byte_count: encoded.len(),
        });

        // Track best-so-far by |achieved - target|.
        let is_best = match best_idx {
            None => true,
            Some(i) => {
                (achieved - spec.target).abs() < (probes[i].achieved_score - spec.target).abs()
            }
        };
        if is_best {
            best_idx = Some(probes.len() - 1);
            best_encoded = encoded.clone();
        }

        if (achieved - spec.target).abs() <= spec.tolerance {
            // Converged.
            return Ok(finalize(
                codec,
                spec,
                profile,
                encoded,
                achieved,
                q_mid,
                iter + 1,
                probes,
                width,
                height,
                true,
            ));
        }

        // Move the search window. With inverted direction (jxl distance):
        // achieved > target means quality is too high → distance too low → increase q_lo.
        let too_high = achieved > spec.target;
        if inverted {
            if too_high {
                q_lo = q_mid;
            } else {
                q_hi = q_mid;
            }
        } else if too_high {
            q_hi = q_mid;
        } else {
            q_lo = q_mid;
        }
    }

    // Out of budget — return best-so-far.
    let best = best_idx.expect("at least one probe ran");
    let best_probe = probes[best].clone();
    Ok(finalize(
        codec,
        spec,
        profile,
        best_encoded,
        best_probe.achieved_score,
        best_probe.knob,
        budget,
        probes,
        width,
        height,
        false,
    ))
}

fn guarded_knob(proposed: f32, low: f32, high: f32, fallback: f32) -> f32 {
    if proposed.is_finite() && proposed > low && proposed < high {
        proposed
    } else {
        fallback
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    codec: CodecKind,
    spec: TargetSpec,
    profile: Option<ZensimProfile>,
    encoded: Vec<u8>,
    achieved: f32,
    knob: f32,
    iterations: u32,
    probes: Vec<ProbeRecord>,
    width: u32,
    height: u32,
    converged: bool,
) -> TargetResult {
    TargetResult {
        codec,
        target: spec.target,
        tolerance: spec.tolerance,
        profile,
        encoded,
        achieved_score: achieved,
        final_knob: knob,
        iterations,
        probes,
        width,
        height,
        converged,
    }
}
