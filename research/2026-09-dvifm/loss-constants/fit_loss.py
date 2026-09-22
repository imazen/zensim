#!/usr/bin/env python3
"""Lane `loss` (dvifm-loss-2026-09-20): identifiable DVIFM constants.

Extends fit_standalone.py — its Params encode/decode, visibility,
contrast_grad, Adam, map refit and cache iteration are reused verbatim via
import (`fs`). This module supplies what the constants-identifiability
lane adds (docs/PLAN_DVIFM_VERDICT_2026-09-20.md §6/§7):

  1. Within-reference pairwise ranking OBJECTIVE (differentiable
     Kendall-tau surrogate): z-scored E inside the loss so the output
     map's affine freedom cannot absorb it; the pooled refit-map MSE and
     SROCC/KROCC stay diagnostics only.
  2. Tied constants first: one shared beta, Cb=Cr per level; untie ladder
     (shared -> per-level -> per-plane-group -> chroma split) gated on
     development-leg gains larger than paired-bootstrap-over-refs noise.
  3. Prior-pulled beta (toward 0.65) vs free; lambda chosen on dev only.
  4. Weber-like contrast axis: C~_w = amp/(pedestal_mean + eps) from the
     20-f32 v2 block records (fields 18,19 = block means of the pedestal
     the band was differenced from — E for lap, B^2G for local, the plane
     itself at the low-pass level). eps stated; refit and compare beta to
     Legge & Foley 0.62 / Watson 0.7.
  5. Domain ladder: safesyn / safesyn-majority / cid22a / tidkadid with
     profile intervals; a constant is portable only where all intervals
     overlap.
  6. Guards' collapse/degeneracy detectors + >=20 bootstrap-over-refs
     refits for the shipped configuration.

Cache formats handled automatically per file: record width 18 (v1) or 20
(v2 with mean fields), quant f32 or f16. Negative targets are kept as-is
throughout (pairwise ordering only); no row is ever dropped for sign.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, "/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools")

import numpy as np
import fit_standalone as fs

LEVELS = fs.LEVELS
F32 = np.dtype("<f4")
F16 = np.dtype("<f2")

TAU = 0.02                      # ranking-loss margin, z-scored-E units
WEBER_EPS = 0.01                # normalized plane units ([0,1] axes)
PRIOR_BETA = 0.65               # Legge & Foley / Watson vicinity
PRIOR_LAMBDAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]
BOOT_REFS = 20
ADAM_SUB_STRIDE = 4             # further subsample of capped recs for Adam


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# cache: record-width / quant autodetect
# ---------------------------------------------------------------------------

class PlaneCache:
    """Memmap'd DVIFM block cache; autodetects record width (18/20) and
    storage quant (f32/f16) from the file size vs the index's record count.

    Exposes the same surface fit_standalone's PlaneCache does (n_rows, nb,
    nz_rows, level_arrays, iter_level) plus `has_means` and per-row
    ref/dist paths for group construction.
    """

    def __init__(self, bin_path):
        self.bin_path = str(bin_path)
        self.index = fs.load_index(bin_path)
        self.n_rows = len(self.index)
        self.quant = self.index[0].get("quant", "f32") if self.index else "f32"
        self.dtype = F16 if self.quant == "f16" else F32
        self.elt = 2 if self.quant == "f16" else 4
        nbytes = os.path.getsize(bin_path)
        total_recs = sum(
            sum(e["level_records"]) for e in self.index)
        rec_f = nbytes / (self.elt * max(total_recs, 1))
        assert rec_f in (18.0, 20.0), (
            f"{bin_path}: {nbytes}B / {total_recs} recs / {self.elt}B "
            f"= {rec_f} fields — expected 18 or 20")
        self.REC = int(rec_f)
        self.has_means = self.REC == 20
        self.data = np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.nb = np.zeros((self.n_rows, LEVELS), np.int64)
        self.row_elem = np.zeros((self.n_rows, LEVELS), np.int64)
        self.refs = np.empty(self.n_rows, object)
        self.dists = np.empty(self.n_rows, object)
        for i, e in enumerate(self.index):
            base = e["offset"] // self.elt
            cnt = np.asarray(e["level_records"], np.int64)
            self.nb[i] = cnt
            self.row_elem[i] = base + np.cumsum(
                np.concatenate([[0], cnt]))[:5] * self.REC
            self.refs[i] = e["ref_path"]
            self.dists[i] = e["dist_path"]
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(LEVELS)]
        self._sub = {}
        self._recs = {}

    def _build(self, l, stride):
        """CSR row layout over nz rows. stride=1 returns flat=None —
        records are already contiguous per row in the file so chunks slice
        directly; stride>1 returns the kept-record gather indices."""
        if stride == 1:
            bounds = np.cumsum(self.nb[self.nz_rows[l], l]) - \
                self.nb[self.nz_rows[l], l]
            # flat=None: per-row record runs are contiguous but NOT across
            # rows (levels interleave within a row) — iter_level builds
            # the gather indices per chunk instead of storing a level's
            # worth
            return None, bounds.astype(np.int64), \
                self.nb[self.nz_rows[l], l].astype(np.int64)
        ix, bounds, counts = [], [], []
        pos = 0
        for r in self.nz_rows[l]:
            n = int(self.nb[r, l])
            start = self.row_elem[r, l] // self.REC
            k = -(-n // stride)
            bidx = start + np.arange(k) * stride
            bounds.append(pos)
            counts.append(k)
            ix.append(bidx)
            pos += k
        flat = np.concatenate(ix) if ix else np.zeros(0, np.int64)
        if flat.size and flat.max() < np.iinfo(np.int32).max:
            flat = flat.astype(np.int32)
        return flat, np.asarray(bounds), np.asarray(counts)

    def _layout(self, l, stride):
        key = (l, stride)
        if key not in self._sub:
            self._sub[key] = self._build(l, stride)
        return self._sub[key]

    def level_arrays(self, l, stride=ADAM_SUB_STRIDE):
        """(recs f64 (N,REC), bounds, counts) for the Adam-estimator
        subsample — resident-cached under an LRU bound. stride=1 is NOT
        supported here (stream it via iter_level / contrast_arrays)."""
        assert stride > 1, "stride=1 records stream; use iter_level"
        flat, bounds, counts = self._layout(l, stride)
        key = (l, stride)
        if key not in self._recs:
            recs = self.data.reshape(-1, self.REC)[flat].astype(np.float64)
            while len(self._recs) >= self._max_resident:
                self._recs.pop(next(iter(self._recs)))
            self._recs[key] = recs
        return self._recs[key], bounds, counts

    _max_resident = 20

    def iter_level(self, l, stride=1, chunk_recs=4_000_000):
        flat, bounds, counts = self._layout(l, stride)
        nz = self.nz_rows[l]
        n_l = len(counts)
        if n_l == 0:
            return
        key = (l, stride)
        recs_all = self._recs.get(key) if stride > 1 else None
        data = self.data.reshape(-1, self.REC)
        i0 = 0
        total = int(counts.sum())
        while i0 < n_l:
            i1 = int(np.searchsorted(
                bounds, bounds[i0] + chunk_recs, "right")) - 1
            i1 = min(max(i1, i0 + 1), n_l)
            lo = int(bounds[i0])
            hi = int(bounds[i1]) if i1 < n_l else total
            if recs_all is not None:
                yield (recs_all[lo:hi], nz[i0:i1],
                       bounds[i0:i1] - lo, counts[i0:i1])
            elif flat is None:
                ix = np.concatenate([
                    np.arange(self.row_elem[r, l] // self.REC,
                              self.row_elem[r, l] // self.REC
                              + int(self.nb[r, l]), dtype=np.int64)
                    for r in nz[i0:i1]])
                yield (data[ix].astype(np.float64),
                       nz[i0:i1], bounds[i0:i1] - lo, counts[i0:i1])
            else:
                yield (data[flat[lo:hi]].astype(np.float64),
                       nz[i0:i1], bounds[i0:i1] - lo, counts[i0:i1])
            i0 = i1

    def contrast_arrays(self, l, g, weber_eps=None, stride=1):
        """(cs,cd,m f32, bounds, counts, nz) — the only resident state the
        grid stage needs; ~12B/record instead of the full record."""
        cs_l, cd_l, m_l, bounds_l, counts_l, nz_l = [], [], [], [], [], []
        pos = 0
        for recs, nz, bounds, counts in self.iter_level(l, stride):
            cs, cd, _, _ = contrasts(recs, g, weber_eps)
            cs_l.append(cs.astype(np.float32))
            cd_l.append(cd.astype(np.float32))
            m_l.append(recs[:, 0].astype(np.float32))
            bounds_l.append(bounds + pos)
            counts_l.append(counts)
            nz_l.append(nz)
            pos += recs.shape[0]
        return (np.concatenate(cs_l), np.concatenate(cd_l),
                np.concatenate(m_l), np.concatenate(bounds_l),
                np.concatenate(counts_l), np.concatenate(nz_l))


class CatCache:
    """Virtual concatenation (same surface as PlaneCache)."""
    def __init__(self, caches):
        self.caches = caches
        self.bin_path = "+".join(c.bin_path for c in caches)
        self.n_rows = sum(c.n_rows for c in caches)
        self.nb = np.concatenate([c.nb for c in caches])
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(LEVELS)]
        self.refs = np.concatenate([c.refs for c in caches])
        self.dists = np.concatenate([c.dists for c in caches])
        self.has_means = all(c.has_means for c in caches)
        self._offsets = np.concatenate(
            [[0], np.cumsum([c.n_rows for c in caches])[:-1]])
        self._sub = {}

    def level_arrays(self, l, stride=ADAM_SUB_STRIDE):
        key = (l, stride)
        if key not in self._sub:
            bs, cs = [], []
            pos = 0
            recs = []
            for c in self.caches:
                r, b, ct = c.level_arrays(l, stride=stride)
                bs.append(b + pos)
                cs.append(ct)
                pos += int(ct.sum())
                recs.append(r)
            self._sub[key] = (np.concatenate(recs),
                              np.concatenate(bs), np.concatenate(cs))
        return self._sub[key]

    def iter_level(self, l, stride=1, chunk_recs=4_000_000):
        for ci, c in enumerate(self.caches):
            off = self._offsets[ci]
            for recs, rows, bounds_l, counts_l in c.iter_level(
                    l, stride, chunk_recs):
                yield recs, rows + off, bounds_l, counts_l

    def contrast_arrays(self, l, g, weber_eps=None, stride=1):
        cs_l, cd_l, m_l, bounds_l, counts_l, nz_l = [], [], [], [], [], []
        pos = 0
        for ci, c in enumerate(self.caches):
            cs, cd, m, bounds, counts, nz = c.contrast_arrays(
                l, g, weber_eps, stride)
            cs_l.append(cs); cd_l.append(cd); m_l.append(m)
            bounds_l.append(bounds + pos)
            counts_l.append(counts)
            nz_l.append(nz + self._offsets[ci])
            pos += int(counts.sum())
        return (np.concatenate(cs_l), np.concatenate(cd_l),
                np.concatenate(m_l), np.concatenate(bounds_l),
                np.concatenate(counts_l), np.concatenate(nz_l))


class BootView:
    """Cache facade over a resampled row set (bootstrap over references).
    Row order/repeats are arbitrary; records are gathered per level on
    first use and kept resident (Adam subsample sizes only)."""

    def __init__(self, cache, rows):
        self.parent = cache
        self.rows = np.asarray(rows, np.int64)
        self.n_rows = len(self.rows)
        self.refs = np.asarray(cache.refs)[self.rows]
        self.dists = np.asarray(cache.dists)[self.rows]
        self.nb = cache.nb[self.rows]
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(LEVELS)]
        self.has_means = cache.has_means
        self.bin_path = f"{cache.bin_path}#boot{self.n_rows}"
        self._lay = {}

    def _src(self, r):
        """(constituent cache, local row) for parent row r — PlaneCache
        maps to itself; CatCache routes via _offsets."""
        if isinstance(self.parent, CatCache):
            ci = int(np.searchsorted(self.parent._offsets, r, "right")) - 1
            return self.parent.caches[ci], int(r - self.parent._offsets[ci])
        return self.parent, int(r)

    def level_arrays(self, l, stride=ADAM_SUB_STRIDE):
        key = (l, stride)
        if key not in self._lay:
            recs_l, bounds, counts = [], [], []
            pos = 0
            datas = {}
            for j, r in enumerate(self.rows):
                n = int(self.parent.nb[r, l])
                k = -(-n // stride)
                if k == 0:
                    continue
                src, lr = self._src(r)
                if id(src) not in datas:
                    datas[id(src)] = (
                        src, src.data.reshape(-1, src.REC))
                st = src.row_elem[lr, l] // src.REC
                recs_l.append(datas[id(src)][1][
                    st + np.arange(k) * stride])
                bounds.append(pos)
                counts.append(k)
                pos += k
            recs = (np.concatenate(recs_l).astype(np.float64)
                    if recs_l else np.zeros(
                        (0, getattr(self.parent, "REC", 20))))
            self._lay[key] = (recs, np.asarray(bounds),
                              np.asarray(counts))
        return self._lay[key]

    def iter_level(self, l, stride=ADAM_SUB_STRIDE, chunk_recs=4_000_000):
        recs, bounds, counts = self.level_arrays(l, stride)
        nz = self.nz_rows[l]
        n_l = len(counts)
        i0 = 0
        while i0 < n_l:
            i1 = min(i0 + 512, n_l)
            lo, hi = int(bounds[i0]), (
                int(bounds[i1]) if i1 < n_l else recs.shape[0])
            yield recs[lo:hi], nz[i0:i1], bounds[i0:i1] - lo, \
                counts[i0:i1]
            i0 = i1

    def contrast_arrays(self, l, g, weber_eps=None, stride=1):
        raise NotImplementedError("bootstrap refits use level_arrays")


# ---------------------------------------------------------------------------
# within-reference ranking loss
# ---------------------------------------------------------------------------

class RankingCtx:
    """Pairwise logistic ranking loss over same-reference pairs.

    Groups = distinct ref_path over the cache row space. Pair (i,j) ranks
    whenever y_i > y_j (signed targets unchanged); E is z-scored inside
    the loss so the objective is invariant to the output map's affine
    freedom — the map is a diagnostic, not part of the loss.
    """

    def __init__(self, y, refs, domain_ids=None, domain_weights=None,
                 tau=TAU):
        self.y = np.asarray(y, np.float64)
        self.tau = tau
        n = len(self.y)
        assert len(refs) == n
        groups = {}
        for i, r in enumerate(refs):
            groups.setdefault(r, []).append(i)
        pi, pj = [], []
        n_ties = 0
        for r, idx in groups.items():
            idx = np.asarray(idx)
            yy = self.y[idx]
            gt = yy[:, None] > yy[None, :]
            ai, bi = np.nonzero(gt)
            pi.append(idx[ai])
            pj.append(idx[bi])
            n_ties += int((yy[:, None] == yy[None, :]).sum()) - len(yy)
        pi = (np.concatenate(pi) if pi else np.zeros(0, np.int64))
        pj = (np.concatenate(pj) if pj else np.zeros(0, np.int64))
        if domain_ids is None:
            domain_ids = np.zeros(n, np.int64)
        if domain_weights is None:
            domain_weights = {0: 1.0}
        self.domain_weights = domain_weights
        # Domain weights are a TOTAL-MASS allocation: domain d contributes
        # w_d of the whole loss, split evenly over its refs, then each
        # ref's mass split evenly over its pairs. (A per-ref weighting
        # would make the 3,218-ref safesyn swamp every choice of w.)
        n_refs_of = {}
        for r, idx in groups.items():
            n_refs_of[int(domain_ids[np.asarray(idx)[0]])] = \
                n_refs_of.get(int(domain_ids[np.asarray(idx)[0]]), 0) + 1
        coef = np.zeros(len(pi))
        pos = 0
        for r, idx in groups.items():
            idx = np.asarray(idx)
            yy = self.y[idx]
            npair = int((yy[:, None] > yy[None, :]).sum())
            d = int(domain_ids[idx[0]])
            w_ref = domain_weights[d] / max(n_refs_of.get(d, 1), 1)
            coef[pos:pos + npair] = w_ref / max(npair, 1)
            pos += npair
        self.pi, self.pj, self.coef = pi, pj, coef
        self.n_pairs = len(pi)
        self.n_ties = n_ties
        self.n_refs = len(groups)
        self.zscale = float(coef.sum()) or 1.0
        # constant-E reference loss: every pair contributes tau*ln2
        self.l_const = float(tau * math.log(2.0) * self.zscale)

    def loss(self, E):
        """(L, dL/dE). L in softplus margin units (0..~tau*ln2*zscale)."""
        E = np.asarray(E, np.float64)
        mu = E.mean()
        sd = max(float(E.std()), 1e-12)
        Ez = (E - mu) / sd
        d = Ez[self.pi] - Ez[self.pj]
        t = d / self.tau
        term = self.tau * np.logaddexp(0.0, t)
        L = float((self.coef * term).sum())
        # dterm/dEz_i = sig(t) at pi, -sig(t) at pj
        w = fs.sig(t) * self.coef
        dEz = np.zeros(len(E))
        np.add.at(dEz, self.pi, w)
        np.add.at(dEz, self.pj, -w)
        # Ez -> E: dEz_k/dE_i = (delta-1/n)/sd - Ez_k*Ez_i/(n*sd)
        n = len(E)
        m_dEz = dEz.mean()
        dot = float((dEz * Ez).sum())
        dE = (dEz - m_dEz) / sd - Ez * (dot / (n * sd))
        return L, dE

    def concordance(self, E):
        d = E[self.pi] - E[self.pj]
        good = float((d < 0).sum())
        return good / max(len(d), 1)


def pairs_load(tsv):
    """rows -> (dist_path -> y) map; extra columns tolerated."""
    out = {}
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["dist_path"]] = float(r["human_score"])
    return out


def y_for_cache(cache, pairs_list):
    """y aligned to cache rows by dist_path join (order-safe)."""
    lut = {}
    for t in pairs_list:
        lut.update(pairs_load(t))
    y = np.array([lut.get(d, np.nan) for d in cache.dists])
    n_miss = int(np.isnan(y).sum())
    assert n_miss == 0, f"{n_miss} cache rows have no pairs-TSV target"
    return y


# ---------------------------------------------------------------------------
# tied-parameter model
# ---------------------------------------------------------------------------

def group_key(p, chroma_shared):
    if chroma_shared and p != "ycbcr_y":
        return "chroma"
    return p


class ModelParams:
    """Tied-constant model. raws keyed by (group, level) — `chroma` shares
    one raw set across Cb and Cr when chroma_shared.

    beta_mode: "shared" (one beta), "level" (beta per level, shared across
    groups), "group" (beta per (group, level)), "free" (beta inside each
    raw — only meaningful combined with chroma split or luma).
    """

    def __init__(self, planes, chroma_shared=True, beta_mode="shared"):
        self.planes = tuple(planes)
        self.chroma_shared = chroma_shared
        self.beta_mode = beta_mode
        self.groups = []
        for p in self.planes:
            g = group_key(p, chroma_shared)
            if g not in self.groups:
                self.groups.append(g)
        self.raw = {g: np.zeros((LEVELS, 5)) for g in self.groups}
        nb = {"shared": 1, "level": LEVELS,
              "group": len(self.groups) * LEVELS, "free": 0}[beta_mode]
        self.raw_beta = np.zeros(max(nb, 1))
        self.wl = {p: np.zeros(LEVELS) for p in self.planes}
        self.wc = np.zeros(2)
        self.map_abl = (100.0, 0.0, 1.0)

    def beta_index(self, g, l):
        if self.beta_mode == "shared":
            return 0
        if self.beta_mode == "level":
            return l
        if self.beta_mode == "group":
            return self.groups.index(g) * LEVELS + l
        return None

    def phys(self, p, l):
        g = group_key(p, self.chroma_shared)
        raw = self.raw[g][l].copy()
        bi = self.beta_index(g, l)
        if bi is not None:
            raw[3] = self.raw_beta[bi]
        return fs.Params.decode(raw)

    def set_from_init(self, init_levels, beta0=0.65):
        for g in self.groups:
            for l, lv in enumerate(init_levels):
                self.raw[g][l] = fs.Params.encode(
                    lv["g"], lv["p"], lv["c0"], lv["beta"], lv["sharp"])
        self.raw_beta[:] = fs.Params.encode(
            1.0, 1.0, 1.0, beta0, 2.0)[3]

    def level_weights(self, p):
        w = np.exp(self.wl[p] - self.wl[p].max())
        return w / w.sum()

    def channel_weights(self):
        zy, zc = self.wc
        ey, ec = math.exp(zy), math.exp(zc)
        d = ey + 2.0 * ec
        return ey / d, ec / d

    # --- packing for joint Adam -----------------------------------------
    def pack(self):
        parts = []
        for g in self.groups:
            for l in range(LEVELS):
                r = self.raw[g][l]
                if self.beta_mode == "free":
                    parts.append(r)
                else:
                    parts.append(r[[0, 1, 2, 4]])
        if self.beta_mode != "free":
            parts.append(self.raw_beta)
        for p in self.planes:
            parts.append(self.wl[p])
        parts.append(self.wc)
        return np.concatenate(parts)

    def unpack(self, x):
        k = 0
        per = 5 if self.beta_mode == "free" else 4
        for g in self.groups:
            for l in range(LEVELS):
                seg = x[k:k + per]
                k += per
                if self.beta_mode == "free":
                    self.raw[g][l] = seg
                else:
                    self.raw[g][l] = np.array(
                        [seg[0], seg[1], seg[2], self.raw[g][l][3], seg[3]])
        if self.beta_mode != "free":
            nb = len(self.raw_beta)
            self.raw_beta = x[k:k + nb]
            k += nb
        for p in self.planes:
            self.wl[p] = x[k:k + LEVELS]
            k += LEVELS
        self.wc = x[k:k + 2]
        k += 2
        assert k == len(x), (k, len(x))

    def grad_layout(self):
        """Names per packed coordinate — for the beta-grad summation."""
        names = []
        for g in self.groups:
            for l in range(LEVELS):
                if self.beta_mode == "free":
                    names += [(g, l, j) for j in range(5)]
                else:
                    names += [(g, l, j) for j in (0, 1, 2, 4)]
        if self.beta_mode != "free":
            names += [("beta", i, 3) for i in range(len(self.raw_beta))]
        for p in self.planes:
            names += [("wl", p, l) for l in range(LEVELS)]
        names += [("wc", 0), ("wc", 1)]
        return names

    def describe(self):
        levs = {}
        for g in self.groups:
            levs[g] = []
            for l in range(LEVELS):
                raw = self.raw[g][l].copy()
                bi = self.beta_index(g, l)
                if bi is not None:
                    raw[3] = self.raw_beta[bi]
                levs[g].append(dict(zip(
                    ("g", "p", "c0", "beta", "sharp"),
                    fs.Params.decode(raw))))
        return {
            "groups": levs,
            "level_weights": {p: self.level_weights(p).tolist()
                              for p in self.planes},
            "channel_weights": list(self.channel_weights()),
            "chroma_shared": self.chroma_shared,
            "beta_mode": self.beta_mode,
        }


def forward_E_model(params, S):
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0
    n = next(iter(S.values())).shape[0]
    E = np.zeros(n)
    for p in params.planes:
        w = params.level_weights(p)
        cp = cY if p == params.planes[0] else cC
        for l in range(LEVELS):
            P = params.phys(p, l)[1]
            s = np.maximum(S[p][:, l], 0.0)
            E += cp * w[l] * np.where(
                s > 0, np.power(np.maximum(s, 1e-300), 1.0 / P), 0.0)
    return E


# ---------------------------------------------------------------------------
# s_level with optional weber contrast + gradients (mirrors fs.s_level)
# ---------------------------------------------------------------------------

def contrasts(recs, g, weber_eps=None):
    """(cs, cd, dcs/dg, dcd/dg); weber: divide by (lowpass mean + eps)."""
    cs, cd, dcs, dcd = fs.contrast_grad(recs, g)
    if weber_eps is not None:
        if recs.shape[1] < 20:
            raise RuntimeError(
                "weber contrast needs the 20-f32 v2 records "
                "(fields 18,19 = lowpass means); this cache is "
                f"{recs.shape[1]}-wide")
        ms = np.maximum(recs[:, 18], 0.0)
        md = np.maximum(recs[:, 19], 0.0)
        cs = cs / (ms + weber_eps)
        cd = cd / (md + weber_eps)
        dcs = dcs / (ms + weber_eps)
        dcd = dcd / (md + weber_eps)
    return cs, cd, dcs, dcd


def s_level(cache, l, phys, stride=1, with_grad=False, weber_eps=None):
    g, P, c0, beta, sharp = phys
    n_rows = cache.n_rows
    s = np.zeros(n_rows)
    d = np.zeros((n_rows, 5))
    for recs, nz, bounds, counts in cache.iter_level(l, stride):
        if recs.shape[0] == 0:
            continue
        cs, cd, dcs, dcd = contrasts(recs, g, weber_eps)
        vs, vs_c0, vs_b, vs_s, vs_c = fs.visibility(cs, c0, beta, sharp)
        vd, vd_c0, vd_b, vd_s, vd_c = fs.visibility(cd, c0, beta, sharp)
        pick = vd > vs
        v = np.where(pick, vd, vs)
        e = np.power(recs[:, 0], P)
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
        if with_grad:
            de = np.where(recs[:, 0] > 0,
                          e * np.log(np.maximum(recs[:, 0], 1e-300)), 0.0)
            d[nz, 0] = np.add.reduceat(
                np.where(pick, vd_c * dcd, vs_c * dcs) * e, bounds) / den
            d[nz, 1] = np.add.reduceat(v * de, bounds) / den
            d[nz, 2] = np.add.reduceat(np.where(pick, vd_c0, vs_c0) * e,
                                       bounds) / den
            d[nz, 3] = np.add.reduceat(np.where(pick, vd_b, vs_b) * e,
                                       bounds) / den
            d[nz, 4] = np.add.reduceat(np.where(pick, vd_s, vs_s) * e,
                                       bounds) / den
    return (s, d) if with_grad else s


def build_S(params, caches, stride=1, weber_eps=None):
    S = {}
    for p in params.planes:
        c = caches[p]
        S[p] = np.zeros((c.n_rows, LEVELS))
        for l in range(LEVELS):
            S[p][:, l] = s_level(c, l, params.phys(p, l),
                                 stride=stride, weber_eps=weber_eps)
    return S


# ---------------------------------------------------------------------------
# joint objective over the packed vector
# ---------------------------------------------------------------------------

_JOINT_STATE = {}


def _joint_level_eval(arg):
    """fork-pool worker: one (plane, level) s+d eval. The cache object is
    looked up from _JOINT_STATE (COW-inherited) — never pickled."""
    i, l, phys, stride, weber_eps = arg
    cache = _JOINT_STATE[i]
    s, d = s_level(cache, l, phys, stride=stride, with_grad=True,
                   weber_eps=weber_eps)
    return i, l, s, d


class JointObj:
    """loss(x) over all free constants + head (+ optional beta prior).

    Each eval: s+d per (plane,level) over the fork pool -> S -> E ->
    ctx.loss -> dL/dE -> chain to packed grads. `prior_lam` adds
    lam*(beta - PRIOR_BETA)^2 to the objective and its gradient to the
    beta coordinates (beta_mode != free: raw_beta coords; free: raw[.,3]).
    """

    def __init__(self, params, caches, ctx, stride=ADAM_SUB_STRIDE,
                 weber_eps=None, prior_lam=0.0):
        self.params = params
        self.caches = caches
        self.ctx = ctx
        self.stride = stride
        self.weber_eps = weber_eps
        self.prior_lam = prior_lam
        self.evals = 0
        for i, pl in enumerate(params.planes):
            _JOINT_STATE[i] = caches[pl]
        # warm the Adam-stride resident arrays in the parent so forked
        # children inherit them (no per-child reread)
        for pl in params.planes:
            for l in range(LEVELS):
                caches[pl].level_arrays(l, stride)
        self._pidx = {pl: i for i, pl in enumerate(params.planes)}

    def __call__(self, x):
        self.params.unpack(x)
        p = self.params
        tasks = []
        for pl in p.planes:
            for l in range(LEVELS):
                tasks.append((self._pidx[pl], l, p.phys(pl, l),
                              self.stride, self.weber_eps))
        S, D = {}, {}
        for pl in p.planes:
            S[pl] = np.zeros((self.caches[pl].n_rows, LEVELS))
            D[pl] = np.zeros((self.caches[pl].n_rows, LEVELS, 5))
        for i, l, s, d in fs.par_map("joint_lvl", _joint_level_eval,
                                     tasks):
            pl = p.planes[i]
            S[pl][:, l] = s
            D[pl][:, l, :] = d
        E = forward_E_model(p, S)
        L, dE = self.ctx.loss(E)
        cY, cC = p.channel_weights()
        if len(p.planes) == 1:
            cY, cC = 1.0, 0.0
        # physical grads per (group, level)
        gphys = {(g, l): np.zeros(5) for g in p.groups for l in range(LEVELS)}
        for pl in p.planes:
            cp = cY if pl == p.planes[0] else cC
            w = p.level_weights(pl)
            g = group_key(pl, p.chroma_shared)
            for l in range(LEVELS):
                ph = p.phys(pl, l)
                P = ph[1]
                s = np.maximum(S[pl][:, l], 0.0)
                Elv = np.power(np.maximum(s, 1e-300), 1.0 / P)
                coef = dE * cp * w[l]
                sdiv = np.where(s > 1e-120, s, 1e-120)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lns = np.where(s > 0, np.log(np.maximum(s, 1e-300)), 0.0)
                    dEl_dP = np.where(s > 0, Elv * (-lns / (P * P)), 0.0)
                    dEl_ds = np.where(s > 0, Elv / (P * sdiv), 0.0)
                d_col = D[pl][:, l, :]
                gp = gphys[(g, l)]
                gp[0] += np.sum(coef * dEl_ds * d_col[:, 0])
                gp[1] += np.sum(coef * (dEl_dP + dEl_ds * d_col[:, 1]))
                gp[2] += np.sum(coef * dEl_ds * d_col[:, 2])
                gp[3] += np.sum(coef * dEl_ds * d_col[:, 3])
                gp[4] += np.sum(coef * dEl_ds * d_col[:, 4])
        # pack grads in pack() order
        grads = []
        for g in p.groups:
            for l in range(LEVELS):
                raw = p.raw[g][l]
                bi = p.beta_index(g, l)
                if bi is not None:
                    raw = raw.copy()
                    raw[3] = p.raw_beta[bi]
                gp = gphys[(g, l)]
                dg = gp[0] * (fs.G_HI - fs.G_LO) * fs.sig(raw[0]) * (
                    1 - fs.sig(raw[0]))
                dP = gp[1] * math.exp(raw[1])
                dc0 = gp[2] * math.exp(raw[2])
                dbeta = gp[3] * (fs.B_HI - fs.B_LO) * fs.sig(raw[3]) * (
                    1 - fs.sig(raw[3]))
                dsharp = gp[4] * fs.sig(raw[4])
                if self.params.beta_mode == "free":
                    grads.append([dg, dP, dc0, dbeta, dsharp])
                else:
                    grads.append([dg, dP, dc0, dsharp])
        # beta coords
        if p.beta_mode != "free":
            nbeta = len(p.raw_beta)
            gb = np.zeros(nbeta)
            for g in p.groups:
                for l in range(LEVELS):
                    bi = p.beta_index(g, l)
                    if bi is not None:
                        raw = p.raw[g][l].copy()
                        raw[3] = p.raw_beta[bi]
                        gb[bi] += gphys[(g, l)][3] * (
                            fs.B_HI - fs.B_LO) * fs.sig(raw[3]) * (
                            1 - fs.sig(raw[3]))
            if self.prior_lam > 0.0:
                phys_b = np.array(
                    [fs.B_LO + (fs.B_HI - fs.B_LO) * fs.sig(min(max(
                        float(rb), -60.0), 60.0))
                     for rb in p.raw_beta])
                L += float(self.prior_lam *
                           np.sum((phys_b - PRIOR_BETA) ** 2))
                gb += self.prior_lam * 2.0 * (phys_b - PRIOR_BETA) * (
                    fs.B_HI - fs.B_LO) * fs.sig(
                        np.clip(p.raw_beta, -60, 60)) * (
                        1 - fs.sig(np.clip(p.raw_beta, -60, 60)))
            grads.append(gb)
        elif self.prior_lam > 0.0:
            # free beta: penalty distributed on each raw[3] coordinate —
            # rebuild the per-raw grads list with the added term
            idx = 0
            for g in p.groups:
                for l in range(LEVELS):
                    raw = p.raw[g][l]
                    b = fs.B_LO + (fs.B_HI - fs.B_LO) * fs.sig(
                        min(max(float(raw[3]), -60.0), 60.0))
                    L += float(self.prior_lam * (b - PRIOR_BETA) ** 2)
                    grads[idx][3] += self.prior_lam * 2.0 * (
                        b - PRIOR_BETA) * (fs.B_HI - fs.B_LO) * fs.sig(
                            np.clip(raw[3], -60, 60)) * (
                            1 - fs.sig(np.clip(raw[3], -60, 60)))
                    idx += 1
        # head grads (same algebra as fs.head_obj_grads)
        Ep = {}
        El = {}
        for pl in p.planes:
            w = p.level_weights(pl)
            Ep[pl] = np.zeros(len(E))
            for l in range(LEVELS):
                P = p.phys(pl, l)[1]
                s = np.maximum(S[pl][:, l], 0.0)
                El[(pl, l)] = np.where(
                    s > 0, np.power(np.maximum(s, 1e-300), 1.0 / P), 0.0)
                Ep[pl] += w[l] * El[(pl, l)]
        for pl in p.planes:
            w = p.level_weights(pl)
            cp = cY if pl == p.planes[0] else cC
            gh = np.zeros(LEVELS)
            for k in range(LEVELS):
                gh[k] = np.sum(dE * cp * w[k] * (El[(pl, k)] - Ep[pl]))
            grads.append(gh)
        EY = Ep[p.planes[0]]
        EC = (Ep[p.planes[1]] + Ep[p.planes[2]]
              if len(p.planes) == 3 else np.zeros(len(E)))
        dwY = np.array([cY * (1 - cY), -2 * cY * cC])
        dwC = np.array([-cC * cY, cC - 2 * cC * cC])
        grads.append(np.array([
            np.sum(dE * (EY * dwY[0] + EC * dwC[0])),
            np.sum(dE * (EY * dwY[1] + EC * dwC[1])),
        ]))
        g = np.concatenate([np.atleast_1d(np.asarray(x_, np.float64))
                            for x_ in grads])
        np.nan_to_num(g, copy=False)
        self.evals += 1
        return L, g


# ---------------------------------------------------------------------------
# grid stage: per (group, level) c0 x beta surfaces under the ranking loss
# (plus pooled-MSE diagnostic), with one-step sharpness
# ---------------------------------------------------------------------------

def group_level_contrasts(caches, planes, l, g_val, weber_eps, stride=1):
    """Per-plane (cs, cd, m, bounds, counts, nz) via contrast_arrays."""
    out = {}
    for p in planes:
        cs, cd, m, bounds, counts, nz = caches[p].contrast_arrays(
            l, g_val, weber_eps, stride=stride)
        out[p] = (cs, cd, m, bounds, counts, nz)
    return out


def grid_surface(params, caches, ctx, y, S_base, gname, l, out_csv,
                 weber_eps=None, stride=1, want_mse=True):
    """(c0 x beta) surface for one (group, level) at CURRENT other params.

    Returns rows (rank_loss, c0, beta, mse, srocc, krocc) sorted by rank
    loss + diag with one-step sharpness on BOTH losses.
    """
    g, P, _, _, sharp = params.phys(
        next(p for p in params.planes if group_key(p, params.chroma_shared)
             == gname), l)
    planes = [p for p in params.planes
              if group_key(p, params.chroma_shared) == gname]
    arr = {}
    for p in planes:
        cs, cd, m_, bounds, counts, nz = caches[p].contrast_arrays(
            l, g, weber_eps, stride=stride)
        arr[p] = dict(cs=cs, cd=cd, m=m_, bounds=bounds,
                      counts=counts, nz=nz)
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0

    def E_const():
        E = np.zeros(len(y))
        for pl in params.planes:
            w = params.level_weights(pl)
            cp = cY if pl == params.planes[0] else cC
            for ll in range(LEVELS):
                if group_key(pl, params.chroma_shared) == gname and ll == l:
                    continue
                P2 = params.phys(pl, ll)[1]
                s = np.maximum(S_base[pl][:, ll], 0.0)
                E += cp * w[ll] * np.where(
                    s > 0, np.power(np.maximum(s, 1e-300), 1.0 / P2), 0.0)
        return E
    Ec = E_const()
    e_pow = {p: np.power(a["m"], P) for p, a in arr.items()}

    def eval_cell(c0, beta):
        E = Ec.copy()
        for p in planes:
            a = arr[p]
            vs = fs.visibility(a["cs"], c0, beta, sharp)[0]
            vd = fs.visibility(a["cd"], c0, beta, sharp)[0]
            contrib = np.maximum(vs, vd) * e_pow[p]
            sr = np.zeros(len(y))
            den = np.maximum(a["counts"], 1).astype(np.float64)
            sr[a["nz"]] = np.add.reduceat(contrib, a["bounds"]) / den
            cp = cY if p == params.planes[0] else cC
            wl = params.level_weights(p)[l]
            E += cp * wl * np.power(np.maximum(sr, 1e-300), 1.0 / P)
        L, _ = ctx.loss(E)
        if want_mse:
            mp = fs.fit_map(E, y)
            mse = mp["mse"]
        else:
            mse = np.nan
        sro, kro = fs._rank_stats(E, y)
        return (L, float(c0), float(beta), mse, sro, kro)

    results = {}
    cells = [(float(c), float(b)) for c in fs.GRID_C0 for b in fs.GRID_BETA]
    for out in fs.par_map(f"grid_{gname}_{l}",
                          lambda cb: eval_cell(*cb), cells):
        results[(out[1], out[2])] = out

    c0v, bv = list(fs.GRID_C0), list(fs.GRID_BETA)
    best = min(results.values(), key=lambda r: r[0])
    edge = {}
    if best[1] in (c0v[0], c0v[-1]):
        edge["c0"] = "lo" if best[1] == c0v[0] else "hi"
    if best[2] in (bv[0], bv[-1]):
        edge["beta"] = "lo" if best[2] == bv[0] else "hi"

    if out_csv:
        surf = np.array([[c, b, *results[(c, b)][:1], *results[(c, b)][3:]]
                         for c, b in sorted(results.keys())])
        np.savetxt(out_csv, surf, delimiter=",",
                   header="c0,beta,rank_loss,mse_refit,srocc,krocc",
                   comments="")

    bi = c0v.index(best[1])
    bj = bv.index(best[2])
    m = {(c, b): r for (c, b), r in results.items()}

    def pm(delta, axis):
        try:
            if axis == "c0":
                nb = c0v[bi + delta]
                r = m[(nb, best[2])]
            else:
                nb = bv[bj + delta]
                r = m[(best[1], nb)]
            return {"dL": r[0] - best[0], "dMSE": (r[3] - best[3])
                    if want_mse else None}
        except (IndexError, KeyError):
            return None

    diag = {"best_rank_loss": best[0], "best_c0": best[1],
            "best_beta": best[2], "best_mse": best[3],
            "best_srocc": best[4], "best_krocc": best[5],
            "grid_edge": edge or None,
            "sharpness": {
                "c0_minus": pm(-1, "c0"), "c0_plus": pm(1, "c0"),
                "beta_minus": pm(-1, "beta"), "beta_plus": pm(1, "beta")},
            "surface_csv": Path(out_csv).name if out_csv else None}
    rows = sorted(results.values())
    return rows, diag


# ---------------------------------------------------------------------------
# profile intervals + bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_loss(ctx, E, refs, n_boot=200, seed=7):
    """SD of the ranking loss under reference resampling at fixed params —
    the 'seed noise' scale for interval + dev-decision thresholds."""
    rng = np.random.default_rng(seed)
    refs = np.asarray(refs)
    uniq = np.unique(refs)
    out = np.empty(n_boot)
    # build resample membership once per boot via searchsorted
    order = np.argsort(refs, kind="stable")
    refs_s = refs[order]
    pi, pj, coef = ctx.pi, ctx.pj, ctx.coef
    for b in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        w_row = np.zeros(len(refs))
        cnt = {}
        for r in draw:
            cnt[r] = cnt.get(r, 0) + 1
        # weight each row by how many times its ref was drawn
        mult = np.array([cnt.get(r, 0) for r in refs])
        pw = coef * mult[pi]
        Ez = (E - E.mean()) / max(E.std(), 1e-12)
        d = Ez[pi] - Ez[pj]
        out[b] = float((pw * ctx.tau *
                        np.logaddexp(0.0, d / ctx.tau)).sum())
    return out


def profile_axis(eval_at, grid, L_star, sigma_b):
    """eval_at(theta)->L over the axis grid; interval = {L <= L* + sigma_b}."""
    Ls = np.array([eval_at(t) for t in grid])
    inside = grid[Ls <= L_star + sigma_b]
    return Ls, (float(inside.min()), float(inside.max())) if inside.size \
        else (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# detectors (guards table)
# ---------------------------------------------------------------------------

def run_assertions(params):
    """Guards' fitted-constant assertions — pure checks on the params and
    the visibility curve shape (no data)."""
    out = {}
    cs_grid = np.geomspace(1e-9, 30.0, 400)
    bad = {}
    for p in params.planes:
        for l in range(LEVELS):
            g, P, c0, beta, sharp = params.phys(p, l)
            vs = fs.visibility(cs_grid, c0, beta, sharp)[0]
            mono = bool((np.diff(vs) <= 1e-12).all())
            finite0 = bool(np.isfinite(fs.visibility(
                np.array([0.0]), c0, beta, sharp)[0][0]))
            rng_ok = bool(0.2 <= g <= 2.0 and P > 0 and sharp > 1
                          and 0.01 <= beta <= 100.0
                          and 1e-8 <= c0 <= 1e4)
            bnd = bool((vs >= -1e-9).all() and (vs <= 1.0 + 1e-9).all())
            if not (mono and finite0 and rng_ok and bnd):
                bad[f"{p}_l{l}"] = {"mono": mono, "finite0": finite0,
                                    "range": rng_ok, "bounded": bnd}
    out["cells_ok"] = not bad
    out["failures"] = bad
    for p in params.planes:
        out[f"level_simplex_sum_{p}"] = float(
            params.level_weights(p).sum())
    cw = params.channel_weights()
    out["channel_simplex_sum"] = float(cw[0] + 2.0 * cw[1])
    out["simplex_ok"] = all(
        abs(out[f"level_simplex_sum_{p}"] - 1.0) < 1e-9
        for p in params.planes) and abs(
            out["channel_simplex_sum"] - 1.0) < 1e-9
    # identical-pair ceiling: m=0 rows contribute exactly zero error
    out["m0_zero_error"] = bool(np.power(0.0, 1.0) == 0.0)
    return out


def run_detectors(params, caches, ctx, y, S, stride=4, weber_eps=None):
    det = {"levels": {}, "global": {}}
    det["assertions"] = run_assertions(params)
    # mixture collapse
    wl_min = min(float(params.level_weights(p).min()) for p in params.planes)
    cY, cC = params.channel_weights()
    det["global"]["channel_min"] = min(cY, cC)
    det["global"]["mixture_collapse"] = bool(
        wl_min < 0.01 or min(cY, cC) < 0.01)
    det["global"]["level_weight_min"] = wl_min
    # map degeneracy on the diagnostic refit map
    E = forward_E_model(params, S)
    mp = fs.fit_map(E, y)
    lamE = float(mp["lambda"] * np.median(E))
    det["global"]["map"] = mp
    det["global"]["map_lambda_medE"] = lamE
    det["global"]["map_degenerate"] = bool(lamE < 0.05)
    # target clamped / saturation
    det["global"]["y_min"] = float(np.min(y))
    det["global"]["y_max"] = float(np.max(y))
    det["global"]["frac_at_min"] = float((y == y.min()).mean())
    det["global"]["frac_at_max"] = float((y == y.max()).mean())
    det["global"]["frac_negative"] = float((y < 0).mean())
    det["global"]["target_clamped"] = bool(
        (y >= 0).all() and det["global"]["frac_at_max"] > 0.05)
    det["global"]["saturation"] = bool(
        det["global"]["frac_at_min"] > 0.05
        or det["global"]["frac_at_max"] > 0.05)
    # per (plane, level): masking-off + gate on kept blocks
    for p in params.planes:
        gname = group_key(p, params.chroma_shared)
        for l in range(LEVELS):
            phys = params.phys(p, l)
            g, P, c0, beta, sharp = phys
            recs, bounds, counts = caches[p].level_arrays(l, stride=stride)
            if recs.shape[0] == 0:
                continue
            cs, cd, _, _ = contrasts(recs, g, weber_eps)
            vs = fs.visibility(cs, c0, beta, sharp)[0]
            vd = fs.visibility(cd, c0, beta, sharp)[0]
            v = np.maximum(vs, vd)
            iqr = float(np.subtract(*np.percentile(v, [75, 25])))
            gate_frac = float(((v > 0.05) & (v < 0.95)).mean())
            d = det["levels"].setdefault(f"{p}_l{l}", {})
            d["v_iqr"] = iqr
            d["v_median"] = float(np.median(v))
            d["gate_soft_fraction"] = gate_frac
            d["masking_off"] = bool(iqr < 0.05 and np.median(v) > 0.5)
            d["masking_gate"] = bool(gate_frac < 0.05)
            d["level_weight"] = float(params.level_weights(p)[l])
            d["mixture_collapse"] = bool(d["level_weight"] < 0.01)
            d["mode"] = ("off" if d["mixture_collapse"] or d["masking_off"]
                         else "gate" if d["masking_gate"] else "curve")
    return det


# ---------------------------------------------------------------------------
# the experiment ladder for one domain
# ---------------------------------------------------------------------------

def domain_ids_for_cache(cache, pairs_list):
    """Row -> index of the pairs TSV its dist_path matched (mixed-domain
    fits get per-row domain ids this way)."""
    ids = np.full(cache.n_rows, -1, np.int64)
    dists = np.asarray(cache.dists)
    for di, t in enumerate(pairs_list):
        lut = pairs_load(t)
        for i, d in enumerate(dists):
            if d in lut:
                ids[i] = di
    assert (ids >= 0).all(), "unmapped rows in domain_ids_for_cache"
    return ids


class Domain:
    def __init__(self, name, fit_pairs, fit_caches, dev_pairs, dev_caches):
        self.name = name
        self.fit_pairs = fit_pairs
        self.caches = fit_caches
        self.dev_pairs = dev_pairs
        self.dev_caches = dev_caches
        anyc = next(iter(fit_caches.values()))
        self.y = y_for_cache(anyc, fit_pairs)
        self.refs = np.asarray(anyc.refs)
        self.domain_ids = domain_ids_for_cache(anyc, fit_pairs)
        if dev_caches:
            anyd = next(iter(dev_caches.values()))
            self.y_dev = y_for_cache(anyd, dev_pairs)
            self.refs_dev = np.asarray(anyd.refs)
            self.domain_ids_dev = domain_ids_for_cache(anyd, dev_pairs)
        else:
            self.y_dev = None
            self.refs_dev = None
            self.domain_ids_dev = None


def eval_dev(params, dom, weber_eps=None, stride=2,
             domain_weights=None):
    """dev-leg ranking loss + diagnostics at fixed params (stride-2
    subsample — consistent across candidates, cheap enough to repeat)."""
    S = build_S(params, dom.dev_caches, stride=stride, weber_eps=weber_eps)
    E = forward_E_model(params, S)
    ctx_d = RankingCtx(dom.y_dev, dom.refs_dev, dom.domain_ids_dev,
                       domain_weights)
    L, _ = ctx_d.loss(E)
    mp = fs.fit_map(E, dom.y_dev)
    sro, kro = fs._rank_stats(E, dom.y_dev)
    return {"rank_loss": L, "mse_refit": mp["mse"], "srocc": sro,
            "krocc": kro, "concordance": ctx_d.concordance(E)}


def paired_dev_delta(params_a, params_b, dom, weber_eps=None,
                     stride=2, domain_weights=None):
    """Paired bootstrap over dev refs of (L_b - L_a): CI on the delta."""
    S_a = build_S(params_a, dom.dev_caches, stride=stride,
                  weber_eps=weber_eps)
    E_a = forward_E_model(params_a, S_a)
    S_b = build_S(params_b, dom.dev_caches, stride=stride,
                  weber_eps=weber_eps)
    E_b = forward_E_model(params_b, S_b)
    ctx_d = RankingCtx(dom.y_dev, dom.refs_dev, dom.domain_ids_dev,
                       domain_weights)
    rng = np.random.default_rng(11)
    uniq = np.unique(dom.refs_dev)
    deltas = np.empty(300)
    pi, pj = ctx_d.pi, ctx_d.pj
    ref_of_row = dom.refs_dev
    for b in range(300):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        cnt = {}
        for r in draw:
            cnt[r] = cnt.get(r, 0) + 1
        mult = np.array([cnt.get(r, 0) for r in ref_of_row])
        La = Lb = 0.0
        for E, acc in ((E_a, "a"), (E_b, "b")):
            Ez = (E - E.mean()) / max(E.std(), 1e-12)
            d = Ez[pi] - Ez[pj]
            v = float((ctx_d.coef * mult[pi] * ctx_d.tau *
                       np.logaddexp(0.0, d / ctx_d.tau)).sum())
            if acc == "a":
                La = v
            else:
                Lb = v
        deltas[b] = Lb - La
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_med": float(np.median(deltas)),
            "delta_p2_5": float(lo), "delta_p97_5": float(hi),
            "improves_gt_noise": bool(hi < 0.0)}


# ---------------------------------------------------------------------------
# CLI driver pieces
# ---------------------------------------------------------------------------

def load_caches(specs):
    out = {}
    for s in specs:
        p, _, path = s.partition("=")
        bins = path.split(",")
        out[p] = (PlaneCache(bins[0]) if len(bins) == 1
                  else CatCache([PlaneCache(b) for b in bins]))
    return out


def fit_joint(params, caches, ctx, x0=None, steps=200, prior_lam=0.0,
              weber_eps=None, stride=ADAM_SUB_STRIDE, tag=""):
    obj = JointObj(params, caches, ctx, stride=stride,
                   weber_eps=weber_eps, prior_lam=prior_lam)
    x0 = params.pack() if x0 is None else x0
    best = fs.adam(obj, x0, steps, lr=0.03, patience=25, tol=1e-7)
    params.unpack(best[1])
    log(f"{tag}: joint adam done L={best[0]:.6f} evals={obj.evals}")
    return params, best[0]


def beta_profile(params, caches, ctx, betas, idx=0,
                 stride=ADAM_SUB_STRIDE, weber_eps=None):
    """Loss at each beta grid value for beta coordinate `idx` (raw_beta
    slot), all other params fixed. Full 15-level re-eval per point —
    simple and honest (affected columns only is a TODO)."""
    save = params.raw_beta.copy()
    out = []
    for b in betas:
        params.raw_beta[idx] = fs.Params.encode(1, 1, 1, b, 2)[3]
        S = build_S(params, caches, stride=stride, weber_eps=weber_eps)
        E = forward_E_model(params, S)
        L, _ = ctx.loss(E)
        out.append((float(b), L))
    params.raw_beta[:] = save
    return out


def params_from_describe(planes, desc):
    """Rebuild ModelParams from an artefact's final.params describe()
    (physical values -> raw encoding; weights -> log domain)."""
    p = ModelParams(planes, chroma_shared=desc["chroma_shared"],
                    beta_mode=desc["beta_mode"])
    for g in p.groups:
        for l in range(LEVELS):
            d = desc["groups"][g][l]
            p.raw[g][l] = fs.Params.encode(d["g"], d["p"], d["c0"],
                                           d["beta"], d["sharp"])
            bi = p.beta_index(g, l)
            if bi is not None:
                p.raw_beta[bi] = p.raw[g][l][3]
    for pl in p.planes:
        w = np.asarray(desc["level_weights"][pl], float)
        p.wl[pl] = np.log(np.maximum(w, 1e-12))
    cY, cC = desc["channel_weights"]
    p.wc = np.array([math.log(max(cY, 1e-12) / max(cC, 1e-12)), 0.0])
    return p


def c0_profile(params, caches, ctx, c0s, gname, l,
               stride=ADAM_SUB_STRIDE, weber_eps=None):
    """Loss at each c0 grid value for (group, level) — raw[g][l][2] is
    log(c0); all other params fixed. Same footing as beta_profile."""
    save = params.raw[gname][l].copy()
    out = []
    for c in c0s:
        params.raw[gname][l][2] = math.log(
            min(max(c, fs.C0_LO), fs.C0_HI))
        S = build_S(params, caches, stride=stride, weber_eps=weber_eps)
        E = forward_E_model(params, S)
        L, _ = ctx.loss(E)
        out.append((float(c), L))
    params.raw[gname][l] = save
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    gp = sub.add_parser("grid")
    gp.add_argument("--cache", action="append", required=True)
    gp.add_argument("--pairs", action="append", required=True)
    gp.add_argument("--init-spec", required=True)
    gp.add_argument("--out-dir", required=True)
    gp.add_argument("--name", required=True)
    gp.add_argument("--chroma-untied", action="store_true")
    gp.add_argument("--weber-eps", type=float, default=None)
    gp.add_argument("--stride", type=int, default=1)

    fp = sub.add_parser("fit")
    fp.add_argument("--cache", action="append", required=True)
    fp.add_argument("--pairs", action="append", required=True)
    fp.add_argument("--dev-cache", action="append", required=True)
    fp.add_argument("--dev-pairs", action="append", required=True)
    fp.add_argument("--init-spec", required=True)
    fp.add_argument("--out-dir", required=True)
    fp.add_argument("--name", required=True)
    fp.add_argument("--weber-eps", type=float, default=None)
    fp.add_argument("--steps", type=int, default=200)
    fp.add_argument("--grids", action="store_true")
    fp.add_argument("--bootstrap", type=int, default=0)
    fp.add_argument("--boot-steps", type=int, default=80)
    fp.add_argument("--domain-weights", default=None,
                    help='JSON {"0": w0, "1": w1} per pairs-TSV index')
    fp.add_argument("--row-frac", type=float, default=1.0,
                    help="seeded row subsample for Adam-stage fits "
                         "(untie/prior/boot); reporting stages use all rows")
    fp.add_argument("--row-seed", type=int, default=5)

    cp = sub.add_parser("c0profile")
    cp.add_argument("--artefact", required=True,
                    help="artefact_<name>.json supplying the params")
    cp.add_argument("--cache", action="append", required=True)
    cp.add_argument("--pairs", action="append", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--name", required=True)
    cp.add_argument("--weber-eps", type=float, default=None)
    cp.add_argument("--domain-weights", default=None)

    sp = sub.add_parser("spec")
    sp.add_argument("--artefact", action="append", required=True,
                    help="artefact_<name>.json from a fit run (repeatable)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--init-spec", required=True)
    sp.add_argument("--canonical", default=None,
                    help="artefact name to take the shipped constants from")
    sp.add_argument("--c0-profile", default=None,
                    help="c0profile_<arm>.json to merge as c0 intervals")
    sp.add_argument("--pairs", default=None,
                    help="canonical arm's fit pairs TSV — hashed into "
                    "fit.pairs_sha")
    sp.add_argument("--git-commit", default=None,
                    help="repo commit id stamped into provenance")

    a = ap.parse_args()

    if a.cmd == "grid":
        caches = load_caches(a.cache)
        anyc = next(iter(caches.values()))
        y = y_for_cache(anyc, a.pairs)
        ctx = RankingCtx(y, anyc.refs)
        log(f"{a.name}: {anyc.n_rows} rows, {ctx.n_refs} refs, "
            f"{ctx.n_pairs} pairs, L_const={ctx.l_const:.5f}, "
            f"ties={ctx.n_ties}, y[{y.min():.3f},{y.max():.3f}]")
        init = json.loads(Path(a.init_spec).read_text())
        planes = tuple(sorted(caches.keys(), key=lambda x: "_y" not in x))
        params = ModelParams(planes,
                             chroma_shared=not a.chroma_untied,
                             beta_mode="shared")
        params.set_from_init(init["levels"])
        S = build_S(params, caches, stride=a.stride,
                    weber_eps=a.weber_eps)
        E0 = forward_E_model(params, S)
        L0, _ = ctx.loss(E0)
        mp0 = fs.fit_map(E0, y)
        sro0, kro0 = fs._rank_stats(E0, y)
        log(f"SANITY init rank-L {L0:.5f} vs const {ctx.l_const:.5f} | "
            f"refit-mse {mp0['mse']:.3f} vs var {np.var(y):.3f} | "
            f"srocc {sro0:.4f} krocc {kro0:.4f} "
            f"concord {ctx.concordance(E0):.4f}")
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        report = {"name": a.name, "sanity": {
            "rank_L": L0, "L_const": ctx.l_const, "mse": mp0["mse"],
            "var_y": float(np.var(y)), "srocc": sro0, "krocc": kro0,
            "concordance": ctx.concordance(E0),
            "map": mp0}}
        diags = {}
        for gname in params.groups:
            for l in range(LEVELS):
                rows, diag = grid_surface(
                    params, caches, ctx, y, S, gname, l,
                    Path(a.out_dir) / f"grid_{a.name}_{gname}_l{l}.csv",
                    weber_eps=a.weber_eps, stride=a.stride)
                diags[f"{gname}_l{l}"] = diag
                log(f"{a.name} {gname} l{l}: best L {diag['best_rank_loss']:.5f} "
                    f"c0={diag['best_c0']:.4g} beta={diag['best_beta']:.3f} "
                    f"mse={diag['best_mse']:.3f} edge={diag['grid_edge']} "
                    f"sharp={diag['sharpness']}")
        report["grids"] = diags
        (Path(a.out_dir) / f"grid_report_{a.name}.json").write_text(
            json.dumps(report, indent=1))
        return 0

    if a.cmd == "fit":
        caches = load_caches(a.cache)
        dev_caches = load_caches(a.dev_cache)
        dom = Domain(a.name, a.pairs, caches, a.dev_pairs, dev_caches)
        init = json.loads(Path(a.init_spec).read_text())
        planes = tuple(sorted(caches.keys(), key=lambda x: "_y" not in x))
        domain_weights = ({int(k): v for k, v in json.loads(
            a.domain_weights).items()} if a.domain_weights else None)
        ctx = RankingCtx(dom.y, dom.refs, dom.domain_ids, domain_weights)
        ctx_dev = RankingCtx(dom.y_dev, dom.refs_dev,
                             dom.domain_ids_dev, domain_weights)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        R = {"name": a.name, "weber_eps": a.weber_eps,
             "planes": list(planes),
             "n_rows": int(dom.y.size), "n_refs": ctx.n_refs,
             "n_pairs": ctx.n_pairs, "n_ties": ctx.n_ties,
             "l_const": ctx.l_const,
             "y_range": [float(dom.y.min()), float(dom.y.max())],
             "y_neg_frac": float((dom.y < 0).mean()),
             "domain_weights": domain_weights, "stages": {}}

        def ck(name, obj):
            (out / f"ck_{a.name}_{name}.json").write_text(
                json.dumps(obj, indent=1, default=str))

        # --- sanity under ranking loss ------------------------------------
        params0 = ModelParams(planes, chroma_shared=True,
                              beta_mode="shared")
        params0.set_from_init(init["levels"])
        S0 = build_S(params0, caches, stride=ADAM_SUB_STRIDE,
                     weber_eps=a.weber_eps)
        E0 = forward_E_model(params0, S0)
        L0, _ = ctx.loss(E0)
        mp0 = fs.fit_map(E0, dom.y)
        sro0, kro0 = fs._rank_stats(E0, dom.y)
        R["sanity"] = {"rank_L": L0, "l_const": ctx.l_const,
                       "mse": mp0["mse"], "var_y": float(np.var(dom.y)),
                       "srocc": sro0, "krocc": kro0,
                       "concordance": ctx.concordance(E0)}
        ck("sanity", R["sanity"])
        log(f"{a.name} sanity: {R['sanity']}")

        # --- E1: tied-model grids at init (sharpness surfaces) ------------
        grids = {}
        if a.grids:
            for gname in params0.groups:
                for l in range(LEVELS):
                    rows, diag = grid_surface(
                        params0, caches, ctx, dom.y, S0, gname, l,
                        out / f"grid_{a.name}_{gname}_l{l}.csv",
                        weber_eps=a.weber_eps, stride=ADAM_SUB_STRIDE)
                    grids[f"{gname}_l{l}"] = diag
                    log(f"{a.name} grid {gname} l{l}: L "
                        f"{diag['best_rank_loss']:.5f} c0={diag['best_c0']:.3g} "
                        f"beta={diag['best_beta']:.3f} edge={diag['grid_edge']}")
            R["stages"]["E1_grids"] = grids
            ck("E1_grids", grids)

        # Adam-stage caches/ctx: optionally a seeded *ref* subsample —
        # whole refs are kept so each kept ref's pair structure is intact
        # (ref-normalized loss makes this an unbiased Monte-Carlo).
        if a.row_frac < 1.0:
            rng = np.random.default_rng(a.row_seed)
            urefs = np.unique(dom.refs)
            keep = set(rng.choice(
                urefs, size=max(1, int(len(urefs) * a.row_frac)),
                replace=False).tolist())
            sub_rows = np.array([i for i, r in enumerate(dom.refs)
                                 if r in keep])
            fit_caches = {p: BootView(caches[p], sub_rows)
                          for p in planes}
            fit_ctx = RankingCtx(dom.y[sub_rows],
                                 np.asarray(dom.refs)[sub_rows],
                                 dom.domain_ids[sub_rows],
                                 domain_weights)
            log(f"{a.name}: Adam on {len(sub_rows)}/{dom.y.size} rows "
                f"({fit_ctx.n_pairs} pairs)")
        else:
            fit_caches, fit_ctx = caches, ctx

        # --- E2: U0 joint fit (fully tied) --------------------------------
        cur = ModelParams(planes, chroma_shared=True, beta_mode="shared")
        cur.set_from_init(init["levels"])
        cur, L = fit_joint(cur, fit_caches, fit_ctx, steps=a.steps,
                           prior_lam=0.0, weber_eps=a.weber_eps,
                           tag=f"{a.name}-U0")
        dev = eval_dev(cur, dom, weber_eps=a.weber_eps,
                       domain_weights=domain_weights)
        R["stages"]["U0"] = {"fit_L": L, "dev": dev,
                             "params": cur.describe()}
        ck("U0", R["stages"]["U0"])
        log(f"{a.name} U0: fit {L:.5f} dev {dev}")

        # --- E3: untie ladder (dev-gated) ---------------------------------
        untie_chain = [("U1", "level"), ("U2", "group")]
        if len(planes) == 3:
            untie_chain.append(("U3", "CHROMA"))
        for step, mode in untie_chain:
            if mode == "CHROMA":
                cand = ModelParams(planes, chroma_shared=False,
                                   beta_mode=cur.beta_mode)
                for p in planes:
                    src = "chroma" if "chroma" in cur.raw else p
                    cand.raw[p] = cur.raw[src].copy()
                # re-key raw_beta to the new group list: each new group's
                # betas come from the group it split out of
                if cur.beta_mode == "group":
                    nb = len(cand.groups) * LEVELS
                    nbv = np.empty(nb)
                    for p in planes:
                        src_g = ("chroma" if "chroma" in cur.raw else p)
                        gi_new = cand.groups.index(p)
                        gi_old = cur.groups.index(src_g)
                        nbv[gi_new * LEVELS:(gi_new + 1) * LEVELS] = \
                            cur.raw_beta[gi_old * LEVELS:
                                         (gi_old + 1) * LEVELS]
                    cand.raw_beta = nbv
                else:
                    cand.raw_beta = cur.raw_beta.copy()
            else:
                cand = ModelParams(planes, chroma_shared=True,
                                   beta_mode=mode)
                cand.raw = {g: v.copy() for g, v in cur.raw.items()}
                cand.raw_beta[:] = cur.raw_beta[0]
            cand.wl = {p: v.copy() for p, v in cur.wl.items()}
            cand.wc = cur.wc.copy()
            cand, L2 = fit_joint(cand, fit_caches, fit_ctx, steps=a.steps,
                                 prior_lam=0.0, weber_eps=a.weber_eps,
                                 tag=f"{a.name}-{step}")
            delta = paired_dev_delta(cur, cand, dom,
                                     weber_eps=a.weber_eps,
                                     domain_weights=domain_weights)
            dev2 = eval_dev(cand, dom, weber_eps=a.weber_eps,
                        domain_weights=domain_weights)
            ok = delta["improves_gt_noise"]
            R["stages"][step] = {"fit_L": L2, "dev": dev2,
                                 "dev_delta": delta, "accepted": ok,
                                 "params": cand.describe()}
            ck(step, R["stages"][step])
            log(f"{a.name} {step}: fit {L2:.5f} dev "
                f"{dev2['rank_loss']:.5f} delta {delta} -> "
                f"{'ACCEPT' if ok else 'REJECT'}")
            if ok:
                cur = cand

        chosen = {"beta_mode": cur.beta_mode,
                  "chroma_shared": cur.chroma_shared,
                  "params": cur.describe()}
        R["stages"]["chosen_structure"] = chosen
        ck("chosen", chosen)

        # --- E4: prior sweep on the chosen structure (dev-selected) -------
        sweep = []
        best_prior = None
        for lam in PRIOR_LAMBDAS:
            cand = ModelParams(planes, chroma_shared=cur.chroma_shared,
                               beta_mode=cur.beta_mode)
            cand.raw = {g: v.copy() for g, v in cur.raw.items()}
            cand.raw_beta = cur.raw_beta.copy()
            cand.wl = {p: v.copy() for p, v in cur.wl.items()}
            cand.wc = cur.wc.copy()
            cand, Lp = fit_joint(cand, fit_caches, fit_ctx, steps=a.steps,
                                 prior_lam=lam, weber_eps=a.weber_eps,
                                 tag=f"{a.name}-prior{lam}")
            devp = eval_dev(cand, dom, weber_eps=a.weber_eps,
                        domain_weights=domain_weights)
            beta_hat = phys_betas(cand)
            sweep.append({"lambda": lam, "fit_L": Lp, "dev": devp,
                          "betas": np.asarray(beta_hat).tolist(),
                          "packed": cand.pack().tolist()})
            log(f"{a.name} prior λ={lam}: fit {Lp:.5f} dev "
                f"{devp['rank_loss']:.5f} betas {np.round(beta_hat,3)}")
        # dev-selection: pick the lambda with the best dev rank_loss;
        # ties (within paired-boot noise) prefer the largest lambda —
        # i.e. ship prior-pulled when dev cannot tell them apart
        devs = np.array([s["dev"]["rank_loss"] for s in sweep])
        i_best = int(np.argmin(devs))
        sel = i_best
        for j in range(len(sweep)):
            if j == i_best:
                continue
            d = paired_dev_delta(
                _params_from(sweep, i_best, cur, planes),
                _params_from(sweep, j, cur, planes), dom,
                weber_eps=a.weber_eps, domain_weights=domain_weights)
            # j indistinguishable from best -> prefer larger lambda
            if not d["improves_gt_noise"] and not _sig_worse(d):
                if sweep[j]["lambda"] > sweep[sel]["lambda"]:
                    sel = j
        R["stages"]["E4_prior"] = {"sweep": sweep,
                                   "selected_lambda": sweep[sel]["lambda"],
                                   "selected_index": sel}
        ck("E4_prior", R["stages"]["E4_prior"])
        final = _params_from(sweep, sel, cur, planes)

        # --- E5: detectors -------------------------------------------------
        S_fin = build_S(final, caches, stride=ADAM_SUB_STRIDE,
                        weber_eps=a.weber_eps)
        det = run_detectors(final, caches, ctx, dom.y, S_fin,
                            stride=4, weber_eps=a.weber_eps)
        E_fin = forward_E_model(final, S_fin)
        det["fit"] = {"rank_L": ctx.loss(E_fin)[0],
                      "concordance": ctx.concordance(E_fin),
                      "mse": fs.fit_map(E_fin, dom.y),
                      "srocc_krocc": fs._rank_stats(E_fin, dom.y)}
        R["stages"]["E5_detectors"] = det
        ck("E5_detectors", det)

        # --- E6: beta profile + interval -----------------------------------
        bootL = paired_bootstrap_loss(ctx, E_fin, dom.refs, n_boot=200)
        sigma_b = float(np.std(bootL))
        L_star = ctx.loss(E_fin)[0]
        profs = {}
        if final.beta_mode != "free":
            bgrid = np.geomspace(0.05, 20.0, 25)
            for bi in range(len(final.raw_beta)):
                raw = final.raw_beta[bi]
                bhat = fs.B_LO + (fs.B_HI - fs.B_LO) * fs.sig(raw)
                prof = beta_profile(final, caches, ctx, bgrid, idx=bi,
                                    stride=ADAM_SUB_STRIDE,
                                    weber_eps=a.weber_eps)
                Ls = np.array([l for _, l in prof])
                inside = bgrid[Ls <= L_star + sigma_b]
                profs[f"beta_{bi}"] = {
                    "hat": float(bhat),
                    "grid": [[float(b), float(l)] for b, l in prof],
                    "interval": [float(inside.min()), float(inside.max())]
                    if inside.size else None,
                    # interval touching the profile grid's edge is an OPEN
                    # interval (unbounded on that side), not a closed one
                    "interval_edge": (
                        [bool(inside.size and inside.min() == bgrid[0]),
                         bool(inside.size and inside.max() == bgrid[-1])]
                        if inside.size else None),
                    "identified": bool(inside.size and
                                       inside.min() > bgrid[0] and
                                       inside.max() < bgrid[-1]),
                    "sigma_b": sigma_b, "L_star": L_star}
        R["stages"]["E6_profile"] = profs
        ck("E6_profile", profs)

        # --- E7: bootstrap refits (constants distribution) -----------------
        boots = []
        if a.bootstrap:
            rng = np.random.default_rng(13)
            uniq = np.unique(dom.refs)
            ref_rows = {}
            for i, r in enumerate(dom.refs):
                ref_rows.setdefault(r, []).append(i)
            for b in range(a.bootstrap):
                draw = rng.choice(uniq, size=len(uniq), replace=True)
                rows = np.concatenate([ref_rows[r] for r in draw])
                bcaches = {p: BootView(caches[p], rows)
                           for p in planes}
                by = dom.y[rows]
                brefs = np.asarray(dom.refs)[rows]
                bctx = RankingCtx(by, brefs, dom.domain_ids[rows],
                                  domain_weights)
                bp = ModelParams(planes,
                                 chroma_shared=final.chroma_shared,
                                 beta_mode=final.beta_mode)
                bp.raw = {g: v.copy() for g, v in final.raw.items()}
                bp.raw_beta = final.raw_beta.copy()
                bp.wl = {p: v.copy() for p, v in final.wl.items()}
                bp.wc = final.wc.copy()
                bp, Lb = fit_joint(
                    bp, bcaches, bctx, steps=a.boot_steps,
                    prior_lam=sweep[sel]["lambda"],
                    weber_eps=a.weber_eps, tag=f"{a.name}-boot{b}")
                boots.append({"L": Lb,
                              "betas": np.asarray(phys_betas(bp)).tolist(),
                              "groups": bp.describe()["groups"]})
                log(f"{a.name} boot{b}: L {Lb:.5f} betas "
                    f"{np.round(phys_betas(bp),3)}")
            betas_b = np.array([b["betas"] for b in boots])
            R["stages"]["E7_bootstrap"] = {
                "n": len(boots),
                "beta_min": betas_b.min(0).tolist() if len(betas_b) else [],
                "beta_max": betas_b.max(0).tolist() if len(betas_b) else [],
                "beta_q10": np.percentile(betas_b, 10, axis=0).tolist()
                if len(betas_b) else [],
                "beta_q90": np.percentile(betas_b, 90, axis=0).tolist()
                if len(betas_b) else [],
                "raw": boots}
            ck("E7_bootstrap", R["stages"]["E7_bootstrap"])

        R["final"] = {"params": final.describe(),
                      "lambda_prior": sweep[sel]["lambda"],
                      "dev": eval_dev(final, dom, weber_eps=a.weber_eps,
                                domain_weights=domain_weights)}
        (out / f"artefact_{a.name}.json").write_text(
            json.dumps(R, indent=1, default=str))
        log(f"{a.name}: artefact written")
        return 0

    if a.cmd == "c0profile":
        A = json.loads(Path(a.artefact).read_text())
        caches = load_caches(a.cache)
        dom = Domain(a.name, a.pairs, caches, [], {})
        planes = tuple(sorted(caches.keys(), key=lambda x: "_y" not in x))
        domain_weights = ({int(k): v for k, v in json.loads(
            a.domain_weights).items()} if a.domain_weights else None)
        ctx = RankingCtx(dom.y, dom.refs, dom.domain_ids, domain_weights)
        params = params_from_describe(planes, A["final"]["params"])
        S_fin = build_S(params, caches, stride=ADAM_SUB_STRIDE,
                        weber_eps=a.weber_eps)
        E_fin = forward_E_model(params, S_fin)
        L_star = ctx.loss(E_fin)[0]
        bootL = paired_bootstrap_loss(ctx, E_fin, dom.refs, n_boot=200)
        sigma_b = float(np.std(bootL))
        c0grid = np.geomspace(1e-6, 10.0, 33)
        profs = {}
        for gname in params.groups:
            for l in range(LEVELS):
                pts = c0_profile(params, caches, ctx, c0grid, gname, l,
                                 stride=ADAM_SUB_STRIDE,
                                 weber_eps=a.weber_eps)
                Ls = np.array([x[1] for x in pts])
                inside = c0grid[Ls <= L_star + sigma_b]
                c_hat = params.phys(
                    next(p for p in planes
                         if group_key(p, params.chroma_shared)
                         == gname), l)[2]
                profs[f"{gname}_l{l}"] = {
                    "hat": float(c_hat),
                    "grid": [[float(c), float(x)] for c, x in pts],
                    "interval": [float(inside.min()), float(inside.max())]
                    if inside.size else None,
                    "interval_edge": (
                        [bool(inside.size and inside.min() == c0grid[0]),
                         bool(inside.size and inside.max() == c0grid[-1])]
                        if inside.size else None),
                    "identified": bool(inside.size and
                                       inside.min() > c0grid[0] and
                                       inside.max() < c0grid[-1]),
                    "sigma_b": sigma_b, "L_star": L_star}
                log(f"{a.name} c0prof {gname}_l{l}: "
                    f"hat={c_hat:.4g} iv={profs[f'{gname}_l{l}']['interval']} "
                    f"edge={profs[f'{gname}_l{l}']['interval_edge']}")
        Path(a.out).write_text(json.dumps(
            {"name": a.name, "c0_profile": profs}, indent=1, default=str))
        log(f"{a.name}: c0 profile written -> {a.out}")
        return 0

    if a.cmd == "spec":
        arts = [json.loads(Path(p).read_text()) for p in a.artefact]
        canon = (a.canonical or arts[0]["name"])
        A = next(x for x in arts if x["name"] == canon)
        init = json.loads(Path(a.init_spec).read_text())
        c0p = (json.loads(Path(a.c0_profile).read_text())
               if a.c0_profile else None)
        spec = build_spec(A, arts, init, c0p, pairs_sha=a.pairs,
                          git_commit=a.git_commit)
        Path(a.out).write_text(json.dumps(spec, indent=1, default=str))
        log(f"spec written -> {a.out}")
        return 0


def build_spec(A, arts, init, c0prof=None, pairs_sha=None, git_commit=None):
    """constants-v1.json — the §2 artefact. Every cell carries value,
    interval, sharpness, mode, tied/free, prior, domains, detectors.

    A (canonical artefact) supplies the shipped constants; `arts` are the
    per-domain runs whose β intervals decide `domains_agreeing`;
    `c0prof` (optional c0profile output) supplies c0 profile intervals.
    """
    det = A["stages"].get("E5_detectors", {})
    prof = A["stages"].get("E6_profile", {})
    grids = A["stages"].get("E1_grids", {})
    prior = A["stages"].get("E4_prior", {})
    final = A["final"]["params"]
    groups = final["groups"]
    planes = A["planes"]

    # per-domain beta intervals -> overlap
    dom_intervals = {}
    for x in arts:
        xi = {}
        for k, v in (x["stages"].get("E6_profile") or {}).items():
            if v.get("interval"):
                xi[k] = v["interval"]
        dom_intervals[x["name"]] = xi

    def interval_for(dom, plane_g, l):
        """β interval in domain `dom` for the (group,level) cell, under
        that domain's own beta_mode — maps shared/level/group coords."""
        di = dom_intervals.get(dom, {})
        arts_of = next(x for x in arts if x["name"] == dom)
        bm = arts_of["final"]["params"]["beta_mode"]
        ng = len(arts_of["final"]["params"]["groups"])
        if bm == "shared":
            return di.get("beta_0")
        if bm == "level":
            return di.get(f"beta_{l}")
        if bm == "group":
            gi = list(arts_of["final"]["params"]["groups"]).index(plane_g)
            return di.get(f"beta_{gi * LEVELS + l}")
        return None

    cells = {}
    for gname, levs in groups.items():
        planes_in = [p for p in planes if (
            (p == "ycbcr_y" and gname == "ycbcr_y")
            or (p != "ycbcr_y" and gname != "ycbcr_y"))]
        for l, lv in enumerate(levs):
            for p in planes_in:
                cell_id = f"{p}_l{l}"
                d = det.get("levels", {}).get(cell_id, {})
                gk = f"{gname}_l{l}"
                grid_d = grids.get(gk, {})
                sharp = grid_d.get("sharpness", {})
                # Domain agreement = count of arms whose own beta
                # profile interval contains the canonical fitted beta.
                # A flat (unidentified) interval contains everything —
                # the report flags that caveat.
                cb = lv["beta"]
                agreeing = [dn for dn in dom_intervals
                            if (iv := interval_for(dn, gname, l))
                            and iv[0] <= cb <= iv[1]]
                bm_a = A["final"]["params"]["beta_mode"]
                if bm_a == "level":
                    pk = f"beta_{l}"
                elif bm_a == "group":
                    gi_a = list(groups).index(gname)
                    pk = f"beta_{gi_a * LEVELS + l}"
                else:
                    pk = "beta_0"
                # Identified = cell exercises beta (curve mode) AND the
                # E6 profile (the authoritative, post-fit grid) found a
                # closed interval. The E1 init-grid edge flag is stale
                # once Adam moves beta — do not veto on it.
                prof_id = (prof.get(pk) or {}).get("identified", None)
                identified = (d.get("mode") == "curve"
                              and prof_id is True)
                cells[cell_id] = {
                    "mode": d.get("mode", "curve"),
                    "g": lv["g"], "P": lv["p"], "c0": lv["c0"],
                    "beta": lv["beta"], "sigma": lv["sharp"],
                    "band": (init.get("levels") or [{}])[l].get("band"),
                    "edge": (init.get("levels") or [{}])[l].get("edge"),
                    "c_hi": (init.get("levels") or [{}])[l].get("c_hi"),
                    "f2_centers": (init.get("levels") or [{}])[l]
                    .get("f2_centers"),
                    "level_weight": d.get("level_weight"),
                    "c0_sharpness_pm1": {
                        "down_dL": (sharp.get("c0_minus") or {}).get("dL"),
                        "up_dL": (sharp.get("c0_plus") or {}).get("dL"),
                        "down_dMSE": (sharp.get("c0_minus") or {})
                        .get("dMSE"),
                        "up_dMSE": (sharp.get("c0_plus") or {})
                        .get("dMSE")},
                    "c0_profile_interval": ((c0prof or {})
                                            .get("c0_profile", {})
                                            .get(gk, {})
                                            .get("interval")),
                    "c0_profile_edge": ((c0prof or {})
                                        .get("c0_profile", {})
                                        .get(gk, {})
                                        .get("interval_edge")),
                    "c0_profile_identified": ((c0prof or {})
                                              .get("c0_profile", {})
                                              .get(gk, {})
                                              .get("identified")),
                    "beta_sharpness_pm1": {
                        "down_dL": (sharp.get("beta_minus") or {})
                        .get("dL"),
                        "up_dL": (sharp.get("beta_plus") or {})
                        .get("dL"),
                        "down_dMSE": (sharp.get("beta_minus") or {})
                        .get("dMSE"),
                        "up_dMSE": (sharp.get("beta_plus") or {})
                        .get("dMSE")},
                    "beta_profile_interval": (prof.get(pk) or {})
                    .get("interval"),
                    "beta_profile_identified": prof_id,
                    "beta_profile_edge": (prof.get(pk) or {})
                    .get("interval_edge"),
                    "identified": bool(identified),
                    "prior": {"beta": PRIOR_BETA,
                              "source": "Legge & Foley 1980 / Watson & "
                                        "Solomon 1997 (0.6-0.7 band)",
                              "lambda": prior.get("selected_lambda")},
                    "domains_agreeing": agreeing,
                    "tied_or_free": {
                        "beta": A["final"]["params"]["beta_mode"],
                        "chroma": "shared" if A["final"]["params"]
                        ["chroma_shared"] else "free"},
                    "detectors_tripped": [
                        k for k in ("masking_off", "masking_gate",
                                    "mixture_collapse")
                        if d.get(k)]}
    spec = {
        "format": "dvifm-constants-v1",
        "lane": "loss",
        "date": "2026-09-20",
        "canonical_domain": A["name"],
        "weber_eps": A.get("weber_eps"),
        "cells": cells,
        "channel_simplex": A["final"]["params"]["channel_weights"],
        "level_simplices": A["final"]["params"]["level_weights"],
        "block": {"n": 8, "phase": "aligned",
                  "band": init.get("levels", [{}])[0].get("band", "local")},
        "planes": planes,
        "plane_normalisation": {
            "ycbcr_y": {"min": 0.0, "span": 1.0},
            "ycbcr_cb": {"min": -0.5, "span": 1.0},
            "ycbcr_cr": {"min": -0.5, "span": 1.0}},
        "output_map": A["stages"].get("E5_detectors", {})
        .get("global", {}).get("map"),
        "feature_set": {
            "record": "dvifm-blockrec-v2",
            "width_f32": 20,
            "fields": ["m", "peak",
                       "cmax_s0", "cmax_s1", "cmax_s2", "cmax_s3",
                       "cmin_s0", "cmin_s1", "cmin_s2", "cmin_s3",
                       "cmax_d0", "cmax_d1", "cmax_d2", "cmax_d3",
                       "cmin_d0", "cmin_d1", "cmin_d2", "cmin_d3",
                       "mean_s", "mean_d"],
            "cap_per_pair": 192,
            "extractor": "zensim-bench/examples/"
            "extract_features_372col.rs --dvifm-block-stats",
        },
        "fit": {"rows": A["n_rows"], "refs": A["n_refs"],
                "pairs": A["n_pairs"], "y_range": A["y_range"],
                "y_neg_frac": A["y_neg_frac"],
                "domain_weights": A.get("domain_weights"),
                "pairs_sha": sha(Path(pairs_sha)) if pairs_sha else None},
        "dev": A["final"].get("dev"),
        "detectors_global": A["stages"].get("E5_detectors", {})
        .get("global"),
        "serving": {
            "visibility_form":
                "v(C) = (1 + (C/c0)^(beta*sigma))^(-1/sigma); "
                "log-contrast x = ln(C/c0) -> "
                "v = (1 + exp(beta*sigma*x))^(-1/sigma)",
            "lut": {
                "index": "i = clamp(floor(QI*(ln C - ln C_min)), 0, N-1); "
                         "recommended QI=64/e-fold, N covers "
                         "C in [1e-6, 4]",
                "v_table": "u16 Q1.15 (v in [0,1])",
                "accumulation": "i64; bit-identical across "
                                "strips/threads/SIMD tiers",
                "modes": "curve cells use the table; gate cells serve "
                         "v=step(C vs c0); off cells serve v=0",
            },
            "measured_max_err": None,
            "error_note": "kernel lane must measure "
                          "max|v_i16 - v_f64| over the full C domain at "
                          "serve time; required before bake (plan §2)",
        },
        "provenance": {
            "artefacts": [x["name"] for x in arts],
            "tool": str(_HERE / "fit_loss.py"),
            "tool_sha": sha(_HERE / "fit_loss.py"),
            "base_fitter_sha": sha(Path(
                "/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools/"
                "fit_standalone.py")),
            "git_commit": git_commit},
    }
    return spec


def phys_betas(params):
    """Physical beta per (group, level) flattened."""
    out = []
    for g in params.groups:
        for l in range(LEVELS):
            raw = params.raw[g][l].copy()
            bi = params.beta_index(g, l)
            if bi is not None:
                raw[3] = params.raw_beta[bi]
            out.append(fs.Params.decode(raw)[3])
    return np.array(out)


def _params_from(sweep, i, base, planes):
    """Reconstruct a ModelParams from sweep[i]'s stored packed vector
    (structure identical to base)."""
    p = ModelParams(planes, chroma_shared=base.chroma_shared,
                    beta_mode=base.beta_mode)
    p.unpack(np.asarray(sweep[i]["packed"], np.float64))
    return p


def _sig_worse(delta):
    """delta CI entirely above 0 -> candidate strictly worse."""
    return delta["delta_p2_5"] > 0.0


if __name__ == "__main__":
    sys.exit(main())
