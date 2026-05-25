use super::TrainingGroup;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Weighted mean of per-group aggregate scores.
    Mean,
    /// Worst per-group aggregate score (conservative).
    Min,
    /// Goal-weighted scoring per CODEC_TARGET_GOALS.md. Each goal
    /// (G1-G11) is a 0.0-1.0 soft gate computed from per-group panel
    /// stats + anchor predictions + optional sweep data. Goals whose
    /// input data wasn't loaded get zero weight (skipped). The
    /// checkpoint score is the weighted sum of goal scores.
    Goals,
}

// ---------------------------------------------------------------------------
// Goal-based validation (CODEC_TARGET_GOALS.md §"Validation policy")
// ---------------------------------------------------------------------------

/// Per-epoch goal check result. Each field is 0.0-1.0 (soft gate:
/// linear ramp from floor to aspiration threshold).
#[derive(Clone, Debug, Default)]
pub struct GoalScores {
    /// G2: JND semantic anchor (|mean_pred - 60| ≤ 5).
    pub g2_jnd_anchor: f64,
    /// G5: HF rank fidelity (konjnd SROCC ≥ 0.70).
    pub g5_hf_rank: f64,
    /// G6: MF band coverage (max SROCC gap vs ssim2 ≤ 0.10).
    pub g6_mf_band_coverage: f64,
    /// G7: CID22 compression-corpus rank (advisory SROCC ≥ 0.85).
    pub g7_cid22_rank: f64,
    /// G8: Z-RMSE quality (lower is better; ≤ 30 floor).
    pub g8_zrmse: f64,
    /// Active goal weights (goals with missing data get 0.0).
    pub weights: [f64; 5],
}

impl GoalScores {
    /// Weighted sum of active goals, normalized by sum of active weights.
    pub fn aggregate(&self) -> f64 {
        let scores = [
            self.g2_jnd_anchor,
            self.g5_hf_rank,
            self.g6_mf_band_coverage,
            self.g7_cid22_rank,
            self.g8_zrmse,
        ];
        let wsum: f64 = self.weights.iter().sum();
        if wsum < 1e-12 {
            return 0.0;
        }
        scores
            .iter()
            .zip(self.weights.iter())
            .map(|(&s, &w)| s * w)
            .sum::<f64>()
            / wsum
    }
}

/// Soft gate: linear ramp from 0 at `floor` to 1 at `target`.
/// Values above target clamp to 1; below floor clamp to 0.
/// `higher_is_better = true` for SROCC; `false` for Z-RMSE (lower is better).
pub(super) fn soft_gate(value: f64, floor: f64, target: f64, higher_is_better: bool) -> f64 {
    if higher_is_better {
        if value >= target {
            1.0
        } else if value <= floor {
            0.0
        } else {
            (value - floor) / (target - floor)
        }
    } else {
        // lower is better: floor is the BAD end, target is the GOOD end
        if value <= target {
            1.0
        } else if value >= floor {
            0.0
        } else {
            (floor - value) / (floor - target)
        }
    }
}

/// Compute goal scores from per-group light panels + anchor predictions.
///
/// `group_panels` is indexed by group. `anchor_mean_pred` is the mean
/// prediction at KonJND PJND pairs (None if no anchor loaded).
/// `konjnd_group_idx` identifies which group (if any) is the KonJND
/// holdout for G5 HF rank checking. `cid22_group_idx` similarly for G7.
pub fn compute_goal_scores(
    group_panels: &[crate::panel::LightPanel],
    groups: &[TrainingGroup<'_>],
    anchor_mean_pred: Option<f64>,
    anchor_zrmse: Option<f64>,
) -> GoalScores {
    let mut gs = GoalScores::default();

    // G2: JND anchor — mean prediction at PJND pairs should be ~60.
    if let Some(mean_pred) = anchor_mean_pred {
        let error = (mean_pred - 60.0).abs();
        // floor: error=10 → 0.0; target: error=0 → 1.0
        gs.g2_jnd_anchor = soft_gate(error, 10.0, 0.0, false);
        gs.weights[0] = 2.0;
    }

    // G5: HF rank — look for groups named konjnd* with SROCC check.
    for (gi, g) in groups.iter().enumerate() {
        let name_lower = g.name.to_lowercase();
        if name_lower.contains("konjnd") && gi < group_panels.len() {
            let srocc = group_panels[gi].srocc;
            gs.g5_hf_rank = soft_gate(srocc, 0.30, 0.85, true);
            gs.weights[1] = 1.5;
            break;
        }
    }

    // G6: MF band coverage — no per-band check yet (needs per-band panel),
    // use worst-group SROCC as a proxy.
    if group_panels.len() >= 2 {
        let min_srocc = group_panels
            .iter()
            .map(|p| p.srocc)
            .fold(f64::INFINITY, f64::min);
        gs.g6_mf_band_coverage = soft_gate(min_srocc, 0.70, 0.95, true);
        gs.weights[2] = 0.5;
    }

    // G7: CID22 rank — look for groups named cid22*.
    for (gi, g) in groups.iter().enumerate() {
        let name_lower = g.name.to_lowercase();
        if name_lower.contains("cid22") && gi < group_panels.len() {
            let srocc = group_panels[gi].srocc;
            gs.g7_cid22_rank = soft_gate(srocc, 0.80, 0.92, true);
            gs.weights[3] = 0.5;
            break;
        }
    }

    // G8: Z-RMSE from anchor predictions (if available).
    if let Some(zrmse) = anchor_zrmse {
        gs.g8_zrmse = soft_gate(zrmse, 1.5, 0.5, false);
        gs.weights[4] = 2.5;
    }

    gs
}

