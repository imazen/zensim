#!/usr/bin/env python3
"""Build the ACTUALLY-additive basic-156 core the design doc thought it had (2026-07-18).

Context / the error this closes: the whole `benchmarks/final_metric_experiments_2026-07-18.md`
campaign labelled its `L1_linear_mf156` / `Eboth_basic156` / `cl_linear` bakes "additive/linear
basic-156". They are NOT — `zenpredict inspect` shows every one is a 156→128→1 **LeakyReLU MLP**
(the `zensim_mlp_train` binary has no linear mode; `--n-hidden-layers 0` is not consumed by
train-core, so it always emits a 128-unit hidden MLP). So the design doc's central number —
"additive core beats full at CID22 0.8978" — was measured on an MLP, and the only genuinely
additive bakes in the program are the B family (b_sdr_linear_*, n_layers=1 identity, 372-input).
NO truly-additive basic-156 has ever been built. This builds it.

Why additive basic-156 is the closed-loop ideal (both properties, unlike winner/B which each
have only one):
  • additive (single identity layer, score = Σ wᵢ·featureᵢ) → the diffmap is the EXACT spatial
    gradient (measured 0.987 for additive vs ~0.87 non-additive, benchmarks/diffmap_coherence_2026-07-16.md)
  • basic-input (weight mass only on f0..155) → that exact diffmap lands only on the spatializable
    basic block, never on the murky peak/max/masked/IW features (f156..371)

Method (NO new solver — imports the linear-projection owner's MixGram / fit_spline_knots /
bake_candidate). Restricting the fit to f0..155 is exactly slicing the standardized Gram to its
leading 156×156 block: minimizing ‖Xw−y‖² s.t. w[156:]=0 has normal equations G[:156,:156]·w = c[:156].
Weights are then padded back to length 372 (zeros on 156..371) and baked as one identity layer.

Usage:  python3 scripts/v_next/additive_basic156_probe.py
Writes bakes to /mnt/v/output/zensim/corr-lq/ADD156_*.bin (+ prints train-side val_metrics).
Evaluate with the full panel via bake_verdict afterwards (this script does NOT call it, to avoid
concurrent CPU with a running re-emit)."""
import importlib.util
import struct
import numpy as np
from pathlib import Path
from scipy.optimize import lsq_linear

# Import the linear-projection owner by path (same pattern it uses for v02bvls).
HERE = Path(__file__).resolve().parent  # scripts/v_next
_spec = importlib.util.spec_from_file_location(
    "linprobe", HERE / "linear_projections_2026-07-03.py")
LP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LP)

SCRATCH = LP.SCRATCH
N_FEAT = LP.N_FEAT           # 372
N_BASIC = 156                # f0..155 = the spatializable basic block
OUT = Path("/mnt/v/output/zensim/corr-lq")

# Mixes to try — the CID-favoring intent of the winner recipe (safesyn + cid22_train heavy +
# kadid), in both the raw and winsor-"shaped" feature spaces. tid gram is not precomputed; its
# 0.5 weight in the winner recipe is minor, so we proceed without it (documented, not silent).
MIXES = {
    "cidmix": [("safesyn", 1.0, "human_score"), ("cid22_train", 2.0, "human_score"),
               ("kadid", 0.5, "human_score")],
    "safesyn_only": [("safesyn", 1.0, "human_score")],
}


def solve_basic156(mg, method: str, lam: float = 0.0):
    """Solve the additive linear map restricted to the leading 156 features.
    Slicing the standardized Gram to [:156,:156] IS the w[156:]=0-constrained least squares."""
    G = mg.G[:N_BASIC, :N_BASIC]
    c = mg.c[:N_BASIC]
    W = mg.W
    if method == "ridge":
        A = G + lam * W * np.eye(N_BASIC)
        w156 = np.linalg.solve(A, c)
    elif method == "bvls":  # non-negativity sign mask like B's canonical solve
        jit = 1e-10 * float(np.trace(G)) / N_BASIC
        L = None
        for _ in range(8):
            try:
                L = np.linalg.cholesky(G + jit * np.eye(N_BASIC)); break
            except np.linalg.LinAlgError:
                jit *= 10.0
        z0 = np.linalg.solve(L, c)
        res = lsq_linear(L.T, z0, bounds=(-np.inf, np.inf), max_iter=4000)
        w156 = res.x
    elif method == "lasso":
        Gn = G / W; cn = c / W
        d = Gn.diagonal().copy(); d[d < 1e-12] = 1e-12
        w156 = np.zeros(N_BASIC); Gw = np.zeros(N_BASIC)
        for _ in range(400):
            delta = 0.0
            for j in range(N_BASIC):
                rho = cn[j] - Gw[j] + d[j] * w156[j]
                nw = np.sign(rho) * max(abs(rho) - lam, 0.0) / d[j]
                if nw != w156[j]:
                    Gw += Gn[:, j] * (nw - w156[j]); delta = max(delta, abs(nw - w156[j])); w156[j] = nw
            if delta < 1e-10:
                break
    else:
        raise ValueError(method)
    w = np.zeros(N_FEAT); w[:N_BASIC] = w156      # pad to 372, zeros on the non-basic block
    return w, mg.ybar


def main():
    (SCRATCH / "fits").mkdir(parents=True, exist_ok=True)
    results = []
    for space in ("raw", "shaped"):
        for mixname, mix in MIXES.items():
            mg = LP.MixGram(space, mix)
            for method, lam in [("ridge", 1e-3), ("bvls", 0.0), ("lasso", 2e-3)]:
                w, bias = solve_basic156(mg, method, lam)
                key = f"ADD156_{mixname}_{space}_{method}"
                np.savez(SCRATCH / "fits" / f"{key}.npz",
                         w=w, bias=bias, mu=mg.mu, sd=mg.sd, space=space)
                out = OUT / f"{key}.bin"
                info = LP.bake_candidate(key, tau=0.0, out_path=out)  # zerobias+f16+spline+bake
                n_basic_active = int((np.abs(w[:N_BASIC]) > 0).sum())
                info["n_basic_active"] = n_basic_active
                results.append((key, info))
                print(f"  {key}: n_active={info['n_active']} (basic {n_basic_active}/156) "
                      f"size={info['size']} knots={info['knots']} sha={info['sha256']}")
    print("\nBaked", len(results), "additive-basic-156 candidates → eval with bake_verdict next.")


if __name__ == "__main__":
    main()
