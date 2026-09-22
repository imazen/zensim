#!/usr/bin/env python3
"""joint-core-v1 DVIFM histogram/per-block equivalence check.

For each (plane, level) the pooled per-block term sum
    S(c0, beta) = sum_b  v(C~_b; c0,beta,sharp) * m_b**P
is computed three ways over one domain:

  HIST        -- from the 256x256 (C~, m) histogram, evaluating v/m^P at bin
                 centres (bin 0 arms: v=1 for C~<=0, term=0 for m=0).
  BLOCK_BIN   -- per stored (f16) block record, with (C~, m) snapped to the
                 same bin centres the histogram uses.  |HIST - BLOCK_BIN| is
                 pure summation-order noise: the exactness claim.
  BLOCK_TRUE  -- per stored (f16) block record at its true value.
                 |HIST - BLOCK_TRUE| is the histogram's binning error.
  (optional) BLOCK_F32 -- with a parallel uncapped f32 cache: adds the
                 cap+f16 storage error.

Max abs and rel deviation per (plane, level) over the full C0 x beta grid
is reported to stdout and written as JSON.
"""
import json, math, sys
import numpy as np

HIST_BINS = 256
HIST_LN_LO = -16.11809565095832   # ln(1e-7) — Rust side constant
HIST_LN_HI = 2.772588722239781    # ln(16)
F32 = np.dtype("<f4")
F16 = np.dtype("<f2")

# the fitter's grid (fit_standalone.py constants)
GRID_C0 = np.exp(np.linspace(math.log(1e-4), math.log(3.0), 18))
GRID_BETA = np.exp(np.linspace(math.log(0.05), math.log(3.0), 18))


def load_hist(path):
    with open(path, "rb") as f:
        header = json.loads(f.readline())
        counts = np.frombuffer(f.read(), dtype="<u4")
    n = HIST_BINS * HIST_BINS
    levels = [counts[l * n:(l + 1) * n].reshape(HIST_BINS, HIST_BINS)
              for l in range(5)]
    return header, levels


def bin_centres(header):
    lo, hi = header["ln_lo"], header["ln_hi"]
    edges = np.exp(np.linspace(lo, hi, HIST_BINS))      # 256 edges -> 255 cells + bin0
    ctr = np.empty(HIST_BINS)
    ctr[0] = 0.0                                        # nonpositive arm
    ctr[1:] = np.exp(lo + (np.arange(1, HIST_BINS) - 0.5) / (HIST_BINS - 1)
                     * (hi - lo))
    return ctr, edges


def hist_bin_idx(v):
    """vectorised mirror of the Rust hist_bin"""
    out = np.zeros(v.shape, dtype=np.int64)
    pos = v > 0.0
    t = (np.log(np.where(pos, v, 1.0)) - HIST_LN_LO) / (HIST_LN_HI - HIST_LN_LO)
    k = 1 + np.floor(t * (HIST_BINS - 1)).astype(np.int64)
    out[pos] = np.clip(k[pos], 1, HIST_BINS - 1)
    return out


def load_index(bin_path):
    return [json.loads(x) for x in open(bin_path + ".index.jsonl")]


def rec_width(bin_path, index, dtype):
    """v1 = 18 f32/record, v2 = 20 (adds block means) — infer from the
    file size vs the index's record total."""
    import os
    esz = np.dtype(dtype).itemsize
    total = sum(sum(e["level_records"]) for e in index)
    w = os.path.getsize(bin_path) / (esz * max(total, 1))
    assert w in (18.0, 20.0), f"{bin_path}: record width {w}"
    return int(w)


def iter_level_records(bin_path, index, level, dtype):
    data = np.memmap(bin_path, dtype=dtype, mode="r")
    esz = np.dtype(dtype).itemsize
    rec = rec_width(bin_path, index, dtype)
    for e in index:
        n = e["level_records"][level]
        if n == 0:
            continue
        base = e["offset"] // esz
        start = base + int(np.sum(e["level_records"][:level])) * rec
        yield data[start:start + n * rec].reshape(n, rec).astype(np.float64)


def contrast(recs, side, g, edge):
    mx = recs[:, 2 + side * 8:6 + side * 8]
    mn = recs[:, 6 + side * 8:10 + side * 8]
    phi = lambda x: np.sign(x) * np.abs(x) ** g
    if edge:
        return (phi(mx) - phi(mn)).min(axis=1)
    return phi(mx.max(axis=1)) - phi(mn.min(axis=1))


def visibility(c, c0, beta, sharp):
    v = np.ones(np.shape(c))
    pos = np.asarray(c) > 0.0
    if np.any(pos):
        cp = np.asarray(c)[pos]
        u = beta * sharp * (np.log(cp) - math.log(c0))
        v[pos] = np.exp(-np.logaddexp(0.0, u) / sharp)
    return v


def cell_sums(terms_per_bin_fn):
    return np.array([[terms_per_bin_fn(c0, b) for b in GRID_BETA]
                     for c0 in GRID_C0])


def main():
    spec_path, bin_path, hist_path, out_json = sys.argv[1:5]
    f32_bin = sys.argv[5] if len(sys.argv) > 5 else None
    spec = json.load(open(spec_path))
    header, hists = load_hist(hist_path)
    index = load_index(bin_path)
    ctr, _ = bin_centres(header)
    quant = index[0]["quant"]
    dtype = F16 if quant == "f16" else F32
    index32 = load_index(f32_bin) if f32_bin else None

    report = {}
    for l in range(5):
        lv = spec["levels"][l]
        g, P, sharp, edge = lv["g"], lv["p"], lv["sharp"], lv["edge"]

        # pooled per-block C~/m arrays for this level (stored recs)
        ctilde, m = [], []
        for recs in iter_level_records(bin_path, index, l, dtype):
            cs = contrast(recs, 0, g, edge)
            cd = contrast(recs, 1, g, edge)
            ctilde.append(np.minimum(cs, cd))
            m.append(recs[:, 0])
        ctilde = np.concatenate(ctilde) if ctilde else np.zeros(0)
        m = np.concatenate(m) if m else np.zeros(0)

        # f32 reference (full-precision uncapped cache), if given
        c32 = m32 = None
        if index32:
            c32, m32 = [], []
            for recs in iter_level_records(f32_bin, index32, l, F32):
                c32.append(np.minimum(contrast(recs, 0, g, edge),
                                      contrast(recs, 1, g, edge)))
                m32.append(recs[:, 0])
            c32 = np.concatenate(c32) if c32 else np.zeros(0)
            m32 = np.concatenate(m32) if m32 else np.zeros(0)

        # block -> bin centres
        cb = ctr[hist_bin_idx(ctilde)]
        mb = ctr[hist_bin_idx(m)]

        def hist_sum(c0, beta):
            v = visibility(ctr, c0, beta, sharp)          # bin0 -> v(0)=1
            mp = np.where(np.arange(HIST_BINS) > 0,
                          ctr ** P, 0.0)                  # bin0 -> 0
            return float((hists[l] * v[:, None] * mp[None, :]).sum())

        def block_bin_sum(c0, beta):
            v = visibility(cb, c0, beta, sharp)
            e = np.where(hist_bin_idx(m) > 0, mb ** P, 0.0)
            return float((v * e).sum())

        def block_true_sum(c0, beta):
            v = visibility(ctilde, c0, beta, sharp)
            e = np.where(m > 0, m ** P, 0.0)
            return float((v * e).sum())

        def block_f32_sum(c0, beta):
            v = visibility(c32, c0, beta, sharp)
            e = np.where(m32 > 0, m32 ** P, 0.0)
            return float((v * e).sum())

        H = cell_sums(hist_sum)
        B = cell_sums(block_bin_sum)
        T = cell_sums(block_true_sum)
        F = cell_sums(block_f32_sum) if index32 else None
        denom = np.maximum(np.abs(T), 1e-30)
        rep = {
            "n_blocks_hist": int(hists[l].sum()),
            "n_blocks_kept": int(ctilde.size),
            "n_blocks_full": int(c32.size) if index32 else None,
            "max_abs_hist_vs_binned": float(np.abs(H - B).max()),
            "max_rel_hist_vs_binned": float((np.abs(H - B) / denom).max()),
            "max_abs_hist_vs_true": float(np.abs(H - T).max()),
            "max_rel_hist_vs_true": float((np.abs(H - T) / denom).max()),
        }
        if F is not None:
            rep["max_abs_true_vs_f32full"] = float(np.abs(T - F).max())
            rep["max_rel_true_vs_f32full"] = float((np.abs(T - F)
                                                  / np.maximum(np.abs(F), 1e-30)).max())
        report[f"level{l}"] = rep
        print(f"L{l}: hist-vs-binned max_rel={rep['max_rel_hist_vs_binned']:.3e} "
              f"hist-vs-true max_rel={rep['max_rel_hist_vs_true']:.3e} "
              f"blocks={rep['n_blocks_kept']}/{rep['n_blocks_hist']}")
    json.dump({"schema": "dvifm-equiv-v1", "bin": bin_path, "hist": hist_path,
                   "spec": spec_path, "levels": report}, open(out_json, "w"),
              indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
