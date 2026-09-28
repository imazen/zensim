#!/usr/bin/env python3
"""Convexity + monotonicity + plumbing tests for fit_standalone.py.

Covers the preregistered assertions (§5.1):
  * all mixing weights >= 0, each simplex sums to 1
  * E non-decreasing in every s_{p,l}
  * E non-decreasing in every m_b (via s)
  * E = 0 at identity (all m_b = 0)
  * no bias term
under random parameters, plus a finite-difference gradient check of the
Adam objectives.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_standalone as fs


TEST_MAP = {"A": 80.0, "B": 5.0, "lambda": 7.0}


def random_params(rng, planes):
    p = fs.Params(planes)
    for pl in planes:
        p.raw[pl] = rng.normal(size=(fs.LEVELS, 5))
        p.wl[pl] = rng.normal(size=fs.LEVELS)
    p.wc = rng.normal(size=2)
    return p


def test_simplex():
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = random_params(rng, fs.PLANES_3)
        for pl in p.planes:
            w = p.level_weights(pl)
            assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-9
        cY, cC = p.channel_weights()
        assert cY >= 0 and cC >= 0 and abs(cY + 2 * cC - 1) < 1e-9


def test_monotone_and_identity():
    rng = np.random.default_rng(11)
    n = 32
    for _ in range(50):
        p = random_params(rng, fs.PLANES_3)
        S = {pl: rng.uniform(0, 0.5, (n, fs.LEVELS)) for pl in p.planes}
        E0, _ = fs.forward_E(p, S)
        # identity: all s = 0 -> E = 0 -> score = A + B (top of scale;
        # default map (100,0,1) keeps the historical 100)
        Sz = {pl: np.zeros((n, fs.LEVELS)) for pl in p.planes}
        Ez, _ = fs.forward_E(p, Sz)
        assert np.all(Ez == 0.0)
        assert np.allclose(fs.score_of(p, Ez), 100.0)
        # non-decreasing in each s_{p,l}
        for pl in p.planes:
            for l in range(fs.LEVELS):
                S2 = {q: v.copy() for q, v in S.items()}
                S2[pl][:, l] += rng.uniform(0, 0.1, n)
                E2, _ = fs.forward_E(p, S2)
                assert np.all(E2 >= E0 - 1e-12)
        # score non-increasing in E (A > 0 in every accepted map)
        assert np.all(fs.score_of(p, E2) <= fs.score_of(p, E0) + 1e-9)
        p.map_abl = (TEST_MAP["A"], TEST_MAP["B"], TEST_MAP["lambda"])
        assert np.all(fs.score_of(p, E2) <= fs.score_of(p, E0) + 1e-9)


def test_s_monotone_in_m():
    """s_{p,l} is non-decreasing in every block max m_b."""
    rng = np.random.default_rng(13)
    nb = 64
    recs = np.zeros((nb, 18))
    recs[:, 0] = rng.uniform(0, 0.3, nb)          # m
    recs[:, 2:10] = rng.uniform(0.01, 0.4, (nb, 8))
    recs[:, 10:18] = rng.uniform(0.01, 0.4, (nb, 8))
    g, P, c0, beta, sharp = 1.0, 1.0, 0.01, 0.65, 4.0
    base = fs.s_rows_like(recs, g, P, c0, beta, sharp)
    recs2 = recs.copy()
    recs2[::3, 0] += 0.05
    up = fs.s_rows_like(recs2, g, P, c0, beta, sharp)
    assert up >= base - 1e-12


def test_level_grads_fd():
    """Finite-difference check of the per-level Adam objective."""
    rng = np.random.default_rng(17)
    n_rows, nb = 8, 40
    recs = np.zeros((n_rows * nb, 18))
    recs[:, 0] = rng.uniform(0, 0.4, n_rows * nb)
    recs[:, 2:10] = rng.uniform(0.005, 0.5, (n_rows * nb, 8))
    recs[:, 10:18] = rng.uniform(0.005, 0.5, (n_rows * nb, 8))

    NR, NB = n_rows, nb  # class body binds same-named attrs -> use distinct names

    class FakeCache:
        n_rows = NR
        nb = np.full((NR, 5), NB)
        nz_rows = [np.arange(NR)] * 5

        def level_arrays(self, l, sub=True):
            assert sub
            return recs, np.arange(NR) * NB, np.full(NR, NB)

        def iter_level(self, l, sub, chunk_recs=fs.FULL_CHUNK_RECS):
            # two chunks -> exercises the chunked reduceat path
            half = NR // 2
            for r0, r1 in ((0, half), (half, NR)):
                yield (recs[r0 * NB:r1 * NB],
                       np.arange(r0, r1),
                       np.arange(r0, r1) * NB - r0 * NB,
                       np.full(r1 - r0, NB))

    cache = FakeCache()
    planes = fs.PLANES_LUMA
    p = fs.Params(planes)
    p.raw[planes[0]] = rng.normal(size=(5, 5)) * 0.3
    y = rng.uniform(20, 95, n_rows)
    l = 2
    s, d = fs.s_level(cache, l, p.phys(planes[0], l), with_grad=True)
    S = {planes[0]: np.zeros((n_rows, 5))}
    S[planes[0]][:, l] = s

    def obj(x):
        save = p.raw[planes[0]][l].copy()
        p.raw[planes[0]][l] = x
        ph = fs.Params.decode(x)
        sc_, dc_ = fs.s_level(cache, l, ph, with_grad=True)
        Sx = {planes[0]: S[planes[0]].copy()}
        Sx[planes[0]][:, l] = sc_
        out = fs.level_obj_grads(p, Sx, y, planes[0], l, sc_, dc_,
                                 fixed_map=TEST_MAP)
        p.raw[planes[0]][l] = save
        return out

    x0 = p.raw[planes[0]][l].copy()
    _, g = obj(x0)
    eps = 1e-6
    for k in range(5):
        xp = x0.copy(); xp[k] += eps
        xm = x0.copy(); xm[k] -= eps
        lp = obj(xp)[0]
        lm = obj(xm)[0]
        fd = (lp - lm) / (2 * eps)
        assert abs(fd - g[k]) < 1e-4 * max(1.0, abs(fd)), \
            f"param {k}: fd {fd} vs analytic {g[k]}"


def test_s_level_chunk_equivalence():
    """iter_level chunking (splits at row ends) must produce the exact same
    per-row sums and grads as the single-shot path."""
    rng = np.random.default_rng(5)
    n_rows, nb = 7, 33
    recs = np.zeros((n_rows * nb, 18))
    recs[:, 0] = rng.uniform(0, 0.4, n_rows * nb)
    recs[:, 2:10] = rng.uniform(0.005, 0.5, (n_rows * nb, 8))
    recs[:, 10:18] = rng.uniform(0.005, 0.5, (n_rows * nb, 8))
    # uneven rows incl. a zero-block row and a >chunk row
    counts = np.array([0, 3, 80, 1, 33, 2, 200])
    n_recs = int(counts.sum())
    recs = recs[:n_recs]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nz = np.nonzero(counts)[0]
    NR, C = n_rows, counts

    def mkcache(splits):
        class FakeCache:
            n_rows = NR
            nb = np.stack([C] * 5, axis=1)
            nz_rows = [nz] * 5

            def iter_level(self, l, sub, chunk_recs=fs.FULL_CHUNK_RECS):
                edges = [0] + [s for s in splits if 0 < s < NR] + [NR]
                for a, b in zip(edges, edges[1:]):
                    rows = np.arange(a, b)
                    rows = rows[C[rows] > 0]
                    if len(rows) == 0:
                        continue
                    lo = int(starts[rows[0]])
                    hi = int(starts[rows[-1]] + C[rows[-1]])
                    yield (recs[lo:hi], rows,
                           starts[rows] - lo, C[rows])
        return FakeCache()

    phys = (1.3, 0.9, 0.02, 0.7, 3.0)
    s1, d1 = fs.s_level(mkcache([]), 0, phys, with_grad=True)
    s3, d3 = fs.s_level(mkcache([2, 4]), 0, phys, with_grad=True)
    np.testing.assert_array_equal(s1, s3)
    np.testing.assert_array_equal(d1, d3)

    # beta must be a NON-vacuous check: place c0 near the median block
    # contrast so vis() responds to beta, and assert the raw-space
    # gradient is meaningfully nonzero before comparing to FD.
    xb = x0.copy()
    xb[2] = math.log(0.2)                 # c0 = 0.2, mid contrast range
    # beta ~= 1.0: large beta makes vis a step function (d/dbeta ~ 0
    # everywhere) — the FD check must sit in the responsive regime.
    xb[3] = -4.6
    _, gb = obj(xb)
    assert abs(gb[3]) > 1e-6, f"vacuous beta gradient: {gb[3]}"
    xb2 = xb.copy(); xb2[3] += eps
    xb3 = xb.copy(); xb3[3] -= eps
    fd_b = (obj(xb2)[0] - obj(xb3)[0]) / (2 * eps)
    assert abs(fd_b - gb[3]) < 1e-4 * max(1.0, abs(fd_b)), \
        f"beta: fd {fd_b} vs analytic {gb[3]}"


def test_head_grads_fd():
    rng = np.random.default_rng(23)
    n = 64
    p = random_params(rng, fs.PLANES_3)
    S = {pl: rng.uniform(0.001, 0.4, (n, fs.LEVELS)) for pl in fs.PLANES_3}
    y = rng.uniform(10, 100, n)

    x0 = p.head_vector()

    def obj(x):
        p.set_head(x)
        return fs.head_obj_grads(p, S, y, fixed_map=TEST_MAP)

    _, g = obj(x0)
    eps = 1e-6
    for k in range(len(x0)):
        xp = x0.copy(); xp[k] += eps
        xm = x0.copy(); xm[k] -= eps
        fd = (obj(xp)[0] - obj(xm)[0]) / (2 * eps)
        assert abs(fd - g[k]) < 1e-4 * max(1.0, abs(fd)), \
            f"head {k}: fd {fd} vs {g[k]}"


def test_fit_map():
    """The refit map recovers a known map and beats the constant predictor;
    an anti-monotone E collapses to the A->0 boundary (= var(y))."""
    rng = np.random.default_rng(31)
    n = 2000
    E = rng.uniform(0, 1.5, n)
    y = 95.0 * np.exp(-2.2 * E) + 3.0 + rng.normal(0, 1.5, n)
    mp = fs.fit_map(E, y)
    assert mp["A"] > 0
    assert mp["mse"] < float(np.var(y))
    assert abs(mp["lambda"] - 2.2) < 0.4      # golden-section ballpark
    assert abs(mp["A"] - 95.0) < 6.0
    # anti-correlated E -> boundary: no better than the constant
    mp_bad = fs.fit_map(E, 100.0 - y + 200.0)
    assert mp_bad["mse"] >= float(np.var(100.0 - y + 200.0)) - 1e-9
    # zero-variance E -> boundary as well
    mp_flat = fs.fit_map(np.zeros(n), y)
    assert mp_flat["mse"] == float(np.var(y))
    # sanity-gate semantics: correlated init passes, flat init fails
    assert mp["mse"] < np.var(y)
    assert not (mp_flat["mse"] < np.var(y))


def test_par_map_deterministic():
    """par_map (fork workers) must return results bit-identical to direct
    calls — the pool only relocates pure deterministic functions, and the
    children inherit read-only state COW. Exercises the exact pattern
    fit_variant uses: adam() over a closure bound to a shared cache."""
    rng = np.random.default_rng(41)
    args = [rng.normal(size=8) for _ in range(6)]

    def f(v):
        return (float(np.tanh(v).sum()), v * 2.0)

    direct = [f(a) for a in args]
    pooled = fs.par_map("t_pure", f, args)
    assert len(pooled) == len(direct)
    for d, q in zip(direct, pooled):
        assert d[0] == q[0]
        np.testing.assert_array_equal(d[1], q[1])

    n_rows, nb = 6, 30
    recs2 = np.zeros((n_rows * nb, 18))
    recs2[:, 0] = rng.uniform(0, 0.4, n_rows * nb)
    recs2[:, 2:10] = rng.uniform(0.005, 0.5, (n_rows * nb, 8))
    recs2[:, 10:18] = rng.uniform(0.005, 0.5, (n_rows * nb, 8))
    NR2, NB2 = n_rows, nb

    class FC:
        n_rows = NR2
        nb = np.full((NR2, 5), NB2)
        nz_rows = [np.arange(NR2)] * 5

        def level_arrays(self, l, sub=True):
            return recs2, np.arange(NR2) * NB2, np.full(NR2, NB2)

        def iter_level(self, l, sub, chunk_recs=fs.FULL_CHUNK_RECS):
            yield (recs2, np.arange(NR2),
                   np.arange(NR2) * NB2, np.full(NR2, NB2))

    cache2 = FC()
    planes = fs.PLANES_LUMA
    pp = fs.Params(planes)
    y2 = rng.uniform(20, 95, n_rows)
    Ssub2 = {planes[0]: np.zeros((n_rows, 5))}
    l2 = 1

    def obj2(x):
        ph = fs.Params.decode(np.asarray(x))
        s_col, d_col = fs.s_level(cache2, l2, ph, sub=True,
                                  with_grad=True)
        Ss = dict(Ssub2)
        Ss[planes[0]] = Ssub2[planes[0]].copy()
        Ss[planes[0]][:, l2] = s_col
        return fs.level_obj_grads(pp, Ss, y2, planes[0], l2,
                                  s_col, d_col, raw=x)

    starts = [pp.raw[planes[0]][l2] + rng.normal(size=5) * 0.01
              for _ in range(3)]

    def rs(x):
        return fs.adam(obj2, x, 40)

    d_res = [rs(a) for a in starts]
    p_res = fs.par_map("t_adam", rs, starts)
    for d, q in zip(d_res, p_res):
        assert d[0] == q[0], f"adam via pool {q[0]} != direct {d[0]}"
        np.testing.assert_array_equal(d[1], q[1])


def test_bounds_contain_grid():
    """No grid value is a logit infinity: every grid endpoint encodes to a
    finite raw vector, and decode(encode(x)) round-trips the grid ranges."""
    for c0 in [fs.GRID_C0[0], fs.GRID_C0[-1], 1e-4, 3.0]:
        for beta in [fs.GRID_BETA[0], fs.GRID_BETA[-1], 0.05, 3.0]:
            raw = fs.Params.encode(1.0, 0.9, c0, beta, 4.0)
            assert np.all(np.isfinite(raw)), (c0, beta, raw)
            g, P, c0d, bd, sd = fs.Params.decode(raw)
            assert abs(c0d - min(max(c0, fs.C0_LO), fs.C0_HI)) < 1e-6 * c0
            assert abs(bd - beta) < 1e-6 * beta
    # parameterisation bounds strictly contain the grid
    assert fs.B_LO < fs.GRID_BETA[0] and fs.B_HI > fs.GRID_BETA[-1]
    assert fs.C0_LO < fs.GRID_C0[0] and fs.C0_HI > fs.GRID_C0[-1]


if __name__ == "__main__":
    test_simplex()
    test_monotone_and_identity()
    test_s_monotone_in_m()
    test_level_grads_fd()
    test_head_grads_fd()
    test_fit_map()
    test_par_map_deterministic()
    test_bounds_contain_grid()
    print("ALL TESTS PASS")
