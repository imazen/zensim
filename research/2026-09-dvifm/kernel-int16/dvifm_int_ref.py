#!/usr/bin/env python3
"""Pure-integer DVIFM reference — bit-exact mirror of `zensim/src/dvifm_int.rs`.

Every value the kernel computes is reproduced here with Python ints
(arbitrary precision, so the i16/u16/i32/i64 wraps and ranges are
asserted, never simulated). The ONLY floating arithmetic is the LUT
bake at param-load — `visibility`/`hat_memberships`/`phi`/`mpow` are
evaluated in f64 on the index grid exactly like `IntLuts::bake` — and
the one `sum / (n * 2**29)` division per feature on the output side.

The gate is `int16_matches_python_reference` in `dvifm.rs` tests: this
script writes a fixture (`tests/fixtures/dvifm_int_parity_*.txt`) the
Rust test replays bit-exactly.

    python3 scripts/dvifm_int_ref.py > tests/fixtures/dvifm_int_parity_<date>.txt
"""

import math
import struct
import sys


def f32(v):
    """Round an f64 to the f32 the Rust quantiser actually sees."""
    return struct.unpack("f", struct.pack("f", v))[0]

Q14 = 16384          # G/band quanta per unit (2^14)
Q15 = 32768          # vis/hat quanta per unit (2^15)
POOL_SHIFT = 29      # Q15 * Q14

LOG_K = 6            # mantissa bits below the leading one
LOG_R = 4            # interpolation fraction bits
LOG_LIN = 1 << LOG_K # 64: identity index below this
LOG_MASK_M = LOG_LIN - 1
LOG_MASK_R = (1 << LOG_R) - 1
INT_LOG_N = LOG_LIN + 10 * LOG_LIN + 1  # 705

LEVELS = 5
BLOCK = 5
BINS = 5
PER_LEVEL = 6
FEATURES = 30

# ---------------------------------------------------------------------------
# Params — mirrors DVIFM_SCREEN_FITTED (screen-4 fitted constants) with the
# per-level `vis` selector: 'curve' | 'gate' | 'off'.
# ---------------------------------------------------------------------------

def fitted_levels(vis="gate", band="local"):
    """DVIFM_SCREEN_FITTED with a uniform per-level visibility/band form."""
    base = [
        dict(g=0.8229549277499173, p=0.839164839021145,
             c0=0.001492783539634792, beta=0.604101903215656,
             sharp=3.5450968244265835, c_hi=math.inf,
             f2_centers=[-5.197234115174323, -4.12354037297495,
                         -3.258800367501043, -2.466767628390845,
                         -1.615985826127243],
             band="local", edge=True),
        dict(g=0.8740482654245578, p=0.9267231301577087,
             c0=0.0017371984815094774, beta=0.6582619541066784,
             sharp=4.096324180352423, c_hi=math.inf,
             f2_centers=[-5.5115575568537585, -4.270301029595456,
                         -3.2847288406948953, -2.5392262555092624,
                         -1.7942537854116223],
             band="local", edge=True),
        dict(g=1.00567258718167, p=1.1373112213843213,
             c0=0.0019382839980098213, beta=0.6210656868986264,
             sharp=3.7036452355361607, c_hi=math.inf,
             f2_centers=[-6.263872914866205, -4.425306891513245,
                         -3.447744257638887, -2.78159112990764,
                         -2.0907573283398624],
             band="local", edge=True),
        dict(g=0.9923898490930416, p=1.0358388140325796,
             c0=0.003823534343442909, beta=0.6487053444624307,
             sharp=3.835264422374912, c_hi=math.inf,
             f2_centers=[-5.558327229743577, -3.7586419475139277,
                         -3.000786933353879, -2.479081379781561,
                         -1.9242737207426723],
             band="local", edge=True),
        dict(g=0.5240864965287544, p=0.8345457349642225,
             c0=0.022227412394756348, beta=0.6173566191341664,
             sharp=4.943042371601268, c_hi=math.inf,
             f2_centers=[-4.327125794285995, -3.274545098232736,
                         -2.710117736533956, -2.2103097055026,
                         -1.6975422772791406],
             band="local", edge=True),
    ]
    for lp in base:
        lp["vis"] = vis
        lp["band"] = band
    return base


# ---------------------------------------------------------------------------
# f64 oracle functions — identical formulas to dvifm.rs (`visibility`,
# `hat_memberships`, `log1pexp`). Used ONLY at table bake.
# ---------------------------------------------------------------------------

def log1pexp(x):
    return x + math.log1p(math.exp(-x)) if x > 0.0 else math.log1p(math.exp(x))


def visibility(c, lp):
    if lp["vis"] == "off":
        return 1.0
    if lp["vis"] == "gate":
        return 0.0 if c > lp["c0"] else 1.0
    # curve — `not (c > 0)` keeps the NaN -> 1 convention.
    if not (c > 0.0):
        return 1.0
    k = lp["beta"] * lp["sharp"]
    ell = math.log(c)
    lower = log1pexp(k * (ell - math.log(lp["c0"])))
    upper = log1pexp(k * (ell - math.log(lp["c_hi"]))) if math.isfinite(lp["c_hi"]) else 0.0
    return math.exp(-(lower - upper) / lp["sharp"])


def hat_memberships(ell, centers):
    ell = min(max(ell, centers[0]), centers[4])
    i = 0
    while i < 5 and centers[i] < ell:
        i += 1
    j = min(max(i - 1, 0), 3)
    t = (ell - centers[j]) / (centers[j + 1] - centers[j])
    h = [0.0] * 5
    h[j] = 1.0 - t
    h[j + 1] = t
    return h


# ---------------------------------------------------------------------------
# Log-indexed tables — identical index map and bake to dvifm_int.rs.
# ---------------------------------------------------------------------------

def log_idx(c):
    if c < LOG_LIN:
        return c, 0
    e = c.bit_length() - 1            # floor(log2 c), >= K
    sh = e - LOG_K
    m = (c >> sh) & LOG_MASK_M
    frac = ((c >> (sh - LOG_R)) & LOG_MASK_R) if sh >= LOG_R \
        else ((c << (LOG_R - sh)) & LOG_MASK_R)
    idx = LOG_LIN + (e - LOG_K) * LOG_LIN + m
    return (INT_LOG_N - 1, 0) if idx >= INT_LOG_N else (idx, frac)


def idx_to_c(idx):
    if idx < LOG_LIN:
        return idx
    o = idx - LOG_LIN
    e = LOG_K + o // LOG_LIN
    m = o % LOG_LIN
    return (1 << e) + (m << (e - LOG_K))


def q15(x):
    return int(math.floor(x * Q15 + 0.5))


def q14(x):
    return int(math.floor(x * Q14 + 0.5))


def bake_luts(lp):
    vis_hat = []
    for i in range(INT_LOG_N):
        c = idx_to_c(i) / Q14
        v = visibility(c, lp)
        h = hat_memberships(math.log(c + 1e-6), lp["f2_centers"])
        vis_hat.append([q15(v)] + [q15(hj) for hj in h])
    phi = None
    if lp["g"] != 1.0:
        phi = [q14((idx_to_c(i) / Q14) ** lp["g"]) for i in range(INT_LOG_N)]
    mpow = None
    if lp["p"] != 1.0:
        mpow = [q14((idx_to_c(i) / Q14) ** lp["p"]) for i in range(INT_LOG_N)]
    return dict(vis_hat=vis_hat, phi=phi, mpow=mpow)


def lut_at(tab, idx, frac):
    a = tab[idx]
    if frac == 0:
        return a
    return a + (((tab[idx + 1] - a) * frac) >> LOG_R)


def vis_hat_at(luts, idx, frac):
    a = luts["vis_hat"][idx]
    if frac == 0:
        return a
    b = luts["vis_hat"][idx + 1]
    return [a[j] + (((b[j] - a[j]) * frac) >> LOG_R) for j in range(6)]


def phi_i(luts, x):
    if luts["phi"] is None:
        return x
    ax = abs(x)
    i, f = log_idx(ax)
    v = lut_at(luts["phi"], i, f)
    return -v if x < 0 else v


def mpow_i(luts, m):
    if luts["mpow"] is None:
        return m
    i, f = log_idx(m)
    return lut_at(luts["mpow"], i, f)


# ---------------------------------------------------------------------------
# Integer row kernels — the two-step rounding average everywhere.
# ---------------------------------------------------------------------------

def _avg121(a, b, c):
    # avg121(a,b,c) = ceil( (ceil((a+c)/2) + b) / 2 ) — the two-step
    # rounding average, one rule for every [1 2 1]/4 tap.
    return (((a + c + 1) // 2) + b + 1) // 2


def reflect_101(i, n):
    if n < 2:
        return 0
    k = i
    while k < 0 or k >= n:
        k = -k if k < 0 else 2 * (n - 1) - k
    return k


def hblur_row(row, w):
    if w < 2:
        return list(row)
    out = [0] * w
    out[0] = _avg121(row[1], row[0], row[1])
    out[w - 1] = _avg121(row[w - 2], row[w - 1], row[w - 2])
    for c in range(1, w - 1):
        out[c] = _avg121(row[c - 1], row[c], row[c + 1])
    return out


def downsample(plane, w, h):
    w2 = (w + 1) // 2
    h2 = (h + 1) // 2
    hb = [hblur_row(plane[r * w:(r + 1) * w], w) for r in range(h)]
    out = [0] * (w2 * h2)
    for j in range(h2):
        ra = reflect_101(2 * j - 1, h)
        rc = reflect_101(2 * j + 1, h)
        for c in range(w2):
            out[j * w2 + c] = _avg121(hb[ra][2 * c], hb[2 * j][2 * c], hb[rc][2 * c])
    return out, w2, h2


def expand(down, w2, h2, w, h):
    # z lattice -> hblur (avg121) -> vertical a + 2b + c (the un-normalised
    # sum; exactly one of the three taps is a zero row per band row).
    z = [0] * (w * h)
    for r in range(h2):
        if 2 * r >= h:
            break
        for c in range(w2):
            if 2 * c < w:
                z[2 * r * w + 2 * c] = down[r * w2 + c]
    zb = [hblur_row(z[r * w:(r + 1) * w], w) for r in range(h)]
    out = [0] * (w * h)
    for r in range(h):
        ra = reflect_101(r - 1, h)
        rc = reflect_101(r + 1, h)
        for c in range(w):
            out[r * w + c] = zb[ra][c] + 2 * zb[r][c] + zb[rc][c]
    return out


def local_band(g, w, h):
    hb = [hblur_row(g[r * w:(r + 1) * w], w) for r in range(h)]
    b1 = [0] * (w * h)
    for r in range(h):
        ra, rc = reflect_101(r - 1, h), reflect_101(r + 1, h)
        for c in range(w):
            b1[r * w + c] = _avg121(hb[ra][c], hb[r][c], hb[rc][c])
    hb2 = [hblur_row(b1[r * w:(r + 1) * w], w) for r in range(h)]
    b2 = [0] * (w * h)
    for r in range(h):
        ra, rc = reflect_101(r - 1, h), reflect_101(r + 1, h)
        for c in range(w):
            b2[r * w + c] = _avg121(hb2[ra][c], hb2[r][c], hb2[rc][c])
    return [g[r * w + c] - b2[r * w + c] for r in range(h) for c in range(w)]


# ---------------------------------------------------------------------------
# Quantiser, pyramid, block stage, pooling.
# ---------------------------------------------------------------------------

def quantize(plane):
    # floor(clamp(v,0,1)·16384 + 0.5); NaN maps to 0 (the Rust `is_nan`
    # arm). Inputs are f32 — callers round through `f32` first so the
    # boundary cases are the same values Rust quantises.
    return [
        0 if math.isnan(v) else int(math.floor(min(max(v, 0.0), 1.0) * Q14 + 0.5))
        for v in plane
    ]


def pyramid(gs, w, h, levels):
    """G_0..G_{L-1}: repeated integer downsample."""
    gs = [gs]
    ws, hs = [w], [h]
    for _ in range(1, levels):
        d, w2, h2 = downsample(gs[-1], ws[-1], hs[-1])
        gs.append(d)
        ws.append(w2)
        hs.append(h2)
    return gs, ws, hs


def band_planes(gs, ws, hs, l, mode):
    """The level-l band plane for one side; last level = G itself."""
    if l == len(gs) - 1:
        return gs[l]
    w, h = ws[l], hs[l]
    if mode == "laplacian":
        e = expand(gs[l + 1], ws[l + 1], hs[l + 1], w, h)
        return [gs[l][r * w + c] - e[r * w + c] for r in range(h) for c in range(w)]
    return local_band(gs[l], w, h)


def scan_blocks(bs, bd, w, h):
    """Emit BlockRec dicts in block-row-major order over the full grid."""
    n = BLOCK
    q = (n + 1) // 2
    recs = []
    for by in range(h // n):
        for bx in range(w // n):
            c0 = bx * n
            m = 0
            cmax_s = [-2**15] * 4
            cmin_s = [2**15 - 1] * 4
            cmax_d = [-2**15] * 4
            cmin_d = [2**15 - 1] * 4
            for r in range(n):
                base = (by * n + r) * w
                for c in range(c0, c0 + n):
                    sv, dv = bs[base + c], bd[base + c]
                    d = max(sv, dv) - min(sv, dv)   # u16: exact incl. 32768
                    if d > m:
                        m = d
            for qi, (r0, cq) in enumerate([(0, 0), (0, n - q), (n - q, 0), (n - q, n - q)]):
                for r in range(r0, r0 + q):
                    base = (by * n + r) * w
                    for c in range(c0 + cq, c0 + cq + q):
                        sv, dv = bs[base + c], bd[base + c]
                        cmax_s[qi] = max(cmax_s[qi], sv)
                        cmin_s[qi] = min(cmin_s[qi], sv)
                        cmax_d[qi] = max(cmax_d[qi], dv)
                        cmin_d[qi] = min(cmin_d[qi], dv)
            recs.append(dict(m=m, cmax_s=cmax_s, cmin_s=cmin_s,
                             cmax_d=cmax_d, cmin_d=cmin_d))
    return recs


def contrast_i(luts, rec, side, edge):
    cmax = rec["cmax_s"] if side == 0 else rec["cmax_d"]
    cmin = rec["cmin_s"] if side == 0 else rec["cmin_d"]
    if edge:
        c = 2**32 - 1
        for qi in range(4):
            cq = max(phi_i(luts, cmax[qi]) - phi_i(luts, cmin[qi]), 0)
            if cq < c:
                c = cq
        return c
    return max(phi_i(luts, max(cmax)) - phi_i(luts, min(cmin)), 0)


def pool_level(luts, lp, recs):
    n = 0
    f1 = 0
    f2 = [0] * BINS
    for rec in recs:
        cs = contrast_i(luts, rec, 0, lp["edge"])
        cd = contrast_i(luts, rec, 1, lp["edge"])
        idx, frac = log_idx(min(cs, cd))
        e = mpow_i(luts, rec["m"])
        vh = vis_hat_at(luts, idx, frac)
        f1 += vh[0] * e
        for j in range(BINS):
            f2[j] += vh[j + 1] * e
        n += 1
    if n == 0:
        return [0.0] * PER_LEVEL
    d = n * (1 << POOL_SHIFT)
    return [f1 / d] + [fj / d for fj in f2]


def dvifm_features_int(src_norm, dst_norm, w, h, levels):
    """The 30 features for normalised planes — mirrors the Rust int path."""
    ps, pd = quantize(src_norm), quantize(dst_norm)
    gs_s, ws, hs = pyramid(ps, w, h, LEVELS)
    gs_d, _, _ = pyramid(pd, w, h, LEVELS)
    out = []
    for l, lp in enumerate(levels):
        luts = bake_luts(lp)
        bs = band_planes(gs_s, ws, hs, l, lp["band"])
        bd = band_planes(gs_d, ws, hs, l, lp["band"])
        recs = scan_blocks(bs, bd, ws[l], hs[l])
        out.extend(pool_level(luts, lp, recs))
    return out


# ---------------------------------------------------------------------------
# Fixture: the same closed-form cases as the f64 parity test.
# ---------------------------------------------------------------------------

def parity_inputs(name, h, w):
    ref = [0.0] * (w * h)
    dist = [0.0] * (w * h)
    for y in range(h):
        for x in range(w):
            ry, cx = float(y), float(x)
            if name == "a":
                v = (0.5 + 0.28 * math.sin(0.21 * ry + 0.13 * cx)
                     * math.cos(0.17 * ry - 0.11 * cx)
                     + 0.07 * math.sin(0.53 * (ry + cx)))
                dv = v + 0.03 * math.sin(1.3 * ry - 0.7 * cx) * math.cos(0.31 * ry + 0.9 * cx)
            elif name == "b":
                v = 0.45 + 0.25 * math.sin(0.35 * ry) * math.cos(0.28 * cx) \
                    + (0.12 if cx >= 18.0 else 0.0)
                dv = v + 0.02 * math.cos(0.9 * ry - 0.4 * cx)
            elif name == "c":
                v = 0.5 + 0.2 * math.sin(0.9 * ry) * math.cos(1.1 * cx)
                dv = v + 0.03 * math.sin(3.0 * ry + 2.0 * cx)
            elif name == "d":
                v = 0.5 + 0.3 * math.sin(0.8 * ry + 0.5 * cx)
                dv = v + 0.04 * math.cos(1.7 * ry - 0.6 * cx)
            else:
                raise ValueError(name)
            ref[y * w + x] = f32(v)
            dist[y * w + x] = f32(dv)
    return ref, dist


def main():
    out = sys.stdout
    out.write("# int16 DVIFM parity fixture — scripts/dvifm_int_ref.py\n")
    out.write("# f64 repr() values round-trip bit-exactly; Rust asserts ==, not ~\n")
    for (name, h, w) in [("a", 61, 67), ("b", 55, 63), ("c", 48, 40), ("d", 30, 26)]:
        ref, dist = parity_inputs(name, h, w)
        for vis in ("gate", "curve"):
            for band in ("local", "laplacian"):
                got = dvifm_features_int(ref, dist, w, h, fitted_levels(vis, band))
                out.write(f"case {name} {h} {w} vis={vis} band={band}\n")
                for l in range(LEVELS):
                    out.write(" ".join(repr(v) for v in got[l * 6:(l + 1) * 6]) + "\n")
                out.write("end\n")


if __name__ == "__main__":
    main()
