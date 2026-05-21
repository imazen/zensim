"""Independent castleCSF validator.

Ports the relevant achromatic + chromatic spatial-CSF math from
/tmp/castleCSF/matlab/CSF_stelaCSF_lum_peak.m and CSF_castleCSF_chrom.m,
generates sensitivity values at cvvdp-gpu's 32x32 grid points, and
byte-compares against cvvdp-gpu's vendored LUT.

If discrepancies are large → cvvdp-gpu's LUT has cvvdp-specific
calibration baked in (e.g., the -0.279 dB sensitivity_correction).
If small → the LUT is a faithful sampling of castleCSF.
"""
import math
import re
import struct
import sys
from pathlib import Path


# --- Parameters from /tmp/castleCSF/matlab/CSF_stelaCSF_lum_peak.m -----
# Achromatic SUSTAINED channel (CSF_stelaCSF_lum_peak.m:305-310)
ACH_SUST_S_MAX = [56.4947, 7.54726, 0.144532, 5.58341e-07, 9.66862e9]
ACH_SUST_F_MAX = [1.78119, 91.5718, 0.256682]
ACH_SUST_BW = 0.000213047
ACH_SUST_A = 0.100207
ACH_SUST_A_0 = 157.103
ACH_SUST_F_0 = 0.702338

# Achromatic TRANSIENT channel (lines 311-316)
ACH_TRANS_S_MAX = [0.193434, 2748.09]  # 2-param model
ACH_TRANS_F_MAX = 0.000316696  # constant
ACH_TRANS_BW = 2.6761
ACH_TRANS_A = 0.000241177
ACH_TRANS_A_0 = 3.81611
ACH_TRANS_F_0 = 3.01389

# Temporal response (lines 317-320)
SIGMA_TRANS = 0.0844836
SIGMA_SUST = 10.5795
OMEGA_TRANS_SL = 2.41482
OMEGA_TRANS_C = 4.7036
BETA_SUST = 1.3314
BETA_TRANS = 0.1898

# Eccentricity (lines 321-324)
ECC_DROP = 0.0239853
ECC_DROP_F = 0.0189038

# Red-Green (CSF_castleCSF_chrom.m:299-309)
RG_S_MAX = [681.434, 38.0038, 0.480386]
RG_F_MAX = 0.0178364
RG_BW = 2.42104
RG_A_0 = 2816.44
RG_F_0 = 0.0711058
RG_SIGMA_SUST = 16.4325
RG_BETA_SUST = 1.15591

# Yellow-Violet (CSF_castleCSF_chrom.m:312-322)
YV_S_MAX = [166.683, 62.8974, 0.41193]
YV_F_MAX = 0.00425753
YV_BW = 2.68197
YV_A_0 = 2.82789e7
YV_F_0 = 0.000635093
YV_SIGMA_SUST = 7.15012
YV_BETA_SUST = 0.969123

# cvvdp's LUT uses ge_sigma = 1.5 → area = pi * 1.5^2 = 7.069
GE_SIGMA = 1.5
AREA = math.pi * GE_SIGMA * GE_SIGMA  # 7.0686

# cvvdp's calibration scalar
SENS_CORRECTION_DB = -0.279742


def get_lum_dep(pars, L):
    """Family of luminance-dependent functions (CSF_base.m:375-395)."""
    if len(pars) == 1:
        return pars[0]
    if len(pars) == 2:
        return pars[1] * (L ** pars[0])  # power-of-L (paper Eq. 22 form for transient)
    if len(pars) == 3:
        return pars[0] * (1 + pars[1] / L) ** (-pars[2])
    if len(pars) == 5:
        return (
            pars[0]
            * (1 + pars[1] / L) ** (-pars[2])
            * (1 - (1 + pars[3] / L) ** (-pars[4]))
        )
    raise ValueError(f"unsupported pars length {len(pars)}")


def csf_achrom(rho, area, L, s_max_pars, f_max_pars, bw, a, A_0, f_0):
    """Achromatic sustained or transient (csf_achrom in MATLAB:150)."""
    S_max = get_lum_dep(s_max_pars, L)
    f_max = get_lum_dep(f_max_pars if isinstance(f_max_pars, list) else [f_max_pars], L)
    S_LP = 10.0 ** (-((math.log10(rho) - math.log10(f_max)) ** 2) / (2 ** bw))
    if rho < f_max and S_LP < (1 - a):
        S_LP = 1 - a
    S_peak = S_max * S_LP
    Ac = A_0 / (1 + (rho / f_0) ** 2)
    return S_peak * math.sqrt(Ac / (1 + Ac / area)) * (rho ** 1)


def csf_chrom(rho, area, L, s_max_pars, f_max, bw, A_0, f_0):
    """Chromatic (RG, YV) — CSF_castleCSF_chrom.m:135ish.
    Truncated log-parabola with S_LP=1 if rho<f_max (no 1-a floor).
    """
    S_max = get_lum_dep(s_max_pars, L)
    S_LP = 10.0 ** (-((math.log10(rho) - math.log10(f_max)) ** 2) / (2 ** bw))
    if rho < f_max:
        S_LP = 1.0
    S_peak = S_max * S_LP
    Ac = A_0 / (1 + (rho / f_0) ** 2)
    return S_peak * math.sqrt(Ac / (1 + Ac / area)) * (rho ** 1)


def temporal_responses(omega, L):
    """R_sust, R_trans (CSF_stelaCSF_lum_peak.m:135-146)."""
    omega_0 = math.log10(L) * OMEGA_TRANS_SL + OMEGA_TRANS_C
    R_sust = math.exp(-(omega ** BETA_SUST) / SIGMA_SUST) if omega > 0 else 1.0
    diff = abs(omega ** BETA_TRANS - omega_0 ** BETA_TRANS)
    R_trans = math.exp(-(diff ** 2) / SIGMA_TRANS)
    return R_sust, R_trans


def sensitivity_achromatic(rho, L, ecc=0.0, omega=0.0):
    """Full achromatic sensitivity at static foveal viewing (ps_beta=1)."""
    R_sust, R_trans = temporal_responses(omega, L)
    S_sust = csf_achrom(
        rho, AREA, L,
        ACH_SUST_S_MAX, ACH_SUST_F_MAX,
        ACH_SUST_BW, ACH_SUST_A, ACH_SUST_A_0, ACH_SUST_F_0,
    )
    S_trans = csf_achrom(
        rho, AREA, L,
        ACH_TRANS_S_MAX, ACH_TRANS_F_MAX,
        ACH_TRANS_BW, ACH_TRANS_A, ACH_TRANS_A_0, ACH_TRANS_F_0,
    )
    S = R_sust * S_sust + R_trans * S_trans
    # Foveal: ecc=0 → 10^(-0) = 1.0, no drop.
    return S


def sensitivity_rg(rho, L, ecc=0.0, omega=0.0):
    """RG sensitivity (single sustained channel; no transient)."""
    # No transient channel for chromatic in castleCSF (paper line 494).
    # Temporal sustained: exp(-omega^beta_sust / sigma).
    R_sust = math.exp(-(omega ** RG_BETA_SUST) / RG_SIGMA_SUST) if omega > 0 else 1.0
    S = csf_chrom(rho, AREA, L, RG_S_MAX, RG_F_MAX, RG_BW, RG_A_0, RG_F_0)
    return R_sust * S


def sensitivity_yv(rho, L, ecc=0.0, omega=0.0):
    R_sust = math.exp(-(omega ** YV_BETA_SUST) / YV_SIGMA_SUST) if omega > 0 else 1.0
    S = csf_chrom(rho, AREA, L, YV_S_MAX, YV_F_MAX, YV_BW, YV_A_0, YV_F_0)
    return R_sust * S


# --- cvvdp-gpu LUT extraction --------------------------------------------
def parse_cvvdp_lut(path):
    """Pull LOG_S_O0_C{1,2,3} const arrays + log-axes out of v0_5_4.rs."""
    text = Path(path).read_text()
    def grab(name):
        # match `pub const NAME ... = [ ... ];`
        m = re.search(rf"pub const {name}\s*:\s*\[f32;\s*\d+\]\s*=\s*\[(.*?)\];", text, re.DOTALL)
        if not m:
            return None
        # find every float literal (handles _f32 suffix)
        nums = re.findall(r"[-]?\d+\.\d+e?[+-]?\d*", m.group(1))
        return [float(x) for x in nums]
    return {
        "log_l_bkg": grab("LOG_L_BKG_AXIS"),
        "log_rho":   grab("LOG_RHO_AXIS"),
        "log_s_a":   grab("LOG_S_O0_C1"),
        "log_s_rg":  grab("LOG_S_O0_C2"),
        "log_s_yv":  grab("LOG_S_O0_C3"),
    }


def main():
    lut_path = (
        "/home/lilith/work/zen/zenmetrics/crates/cvvdp-gpu/src/kernels/csf_lut/v0_5_4.rs"
    )
    lut = parse_cvvdp_lut(lut_path)
    n_l = len(lut["log_l_bkg"])
    n_r = len(lut["log_rho"])
    print(f"Parsed cvvdp-gpu LUT: {n_l} L_bkg axis points, {n_r} rho axis points")
    print(f"L range: [{10**lut['log_l_bkg'][0]:.4f}, {10**lut['log_l_bkg'][-1]:.0f}] cd/m²")
    print(f"ρ range: [{10**lut['log_rho'][0]:.4f}, {10**lut['log_rho'][-1]:.1f}] cy/deg")
    print()

    channels = [
        ("Achromatic (A)", lut["log_s_a"], sensitivity_achromatic),
        ("Red-Green (RG)", lut["log_s_rg"], sensitivity_rg),
        ("Yellow-Violet (YV)", lut["log_s_yv"], sensitivity_yv),
    ]

    sens_corr_factor = 10.0 ** (SENS_CORRECTION_DB / 20.0)
    print(f"cvvdp sensitivity_correction = {SENS_CORRECTION_DB:+.4f} dB "
          f"(× {sens_corr_factor:.6f} = {1/sens_corr_factor:.6f}⁻¹)")
    print()

    for name, log_s_lut, sens_fn in channels:
        max_dev_raw = 0.0
        max_dev_corr = 0.0
        sum_dev_raw = 0.0
        sum_dev_corr = 0.0
        worst_pt = (0, 0, 0.0, 0.0)
        for l_idx in range(n_l):
            for r_idx in range(n_r):
                log_l = lut["log_l_bkg"][l_idx]
                log_r = lut["log_rho"][r_idx]
                L = 10 ** log_l
                rho = 10 ** log_r
                lut_log_s = log_s_lut[l_idx * n_r + r_idx]
                ana_S = sens_fn(rho, L)
                if ana_S <= 0:
                    continue
                ana_log_s = math.log10(ana_S)
                ana_log_s_corr = math.log10(ana_S * sens_corr_factor)
                dev_raw = lut_log_s - ana_log_s          # log10 units
                dev_corr = lut_log_s - ana_log_s_corr
                if abs(dev_raw) > abs(max_dev_raw):
                    max_dev_raw = dev_raw
                    worst_pt = (l_idx, r_idx, L, rho)
                if abs(dev_corr) > abs(max_dev_corr):
                    max_dev_corr = dev_corr
                sum_dev_raw += dev_raw
                sum_dev_corr += dev_corr
        n = n_l * n_r
        mean_dev_raw = sum_dev_raw / n
        mean_dev_corr = sum_dev_corr / n
        print(f"=== {name} ===")
        print(f"  raw analytical vs LUT:")
        print(f"    mean log10(S) dev: {mean_dev_raw:+.5f} ({mean_dev_raw*20:+.3f} dB)")
        print(f"    max  log10(S) dev: {max_dev_raw:+.5f} ({max_dev_raw*20:+.3f} dB) "
              f"@ L={worst_pt[2]:.3f}, ρ={worst_pt[3]:.3f}")
        print(f"  WITH sensitivity_correction backed out:")
        print(f"    mean log10(S) dev: {mean_dev_corr:+.5f} ({mean_dev_corr*20:+.3f} dB)")
        print(f"    max  log10(S) dev: {max_dev_corr:+.5f} ({max_dev_corr*20:+.3f} dB)")
        print()

    # Spot-check some canonical anchor points
    print("=== Anchor spot checks (raw analytical, no correction) ===")
    for L in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        for rho in [0.5, 4.0, 16.0]:
            S = sensitivity_achromatic(rho, L)
            print(f"  L={L:7.2f} cd/m²  ρ={rho:5.1f} cy/deg  →  S={S:8.2f}  log10(S)={math.log10(S):+.4f}")


if __name__ == "__main__":
    main()
