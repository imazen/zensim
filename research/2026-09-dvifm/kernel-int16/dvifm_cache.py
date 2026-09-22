#!/usr/bin/env python3
"""DVIFM block-record cache reader + pooling replay.

Reads the extractor's `--dvifm-block-stats` output: a flat f32 bin of
18-float records (level-major, block-row-major over the full-block grid)
plus its `<bin>.index.jsonl` (row_index, offset, grid, level counts,
input sha256s). Replays `pool_block`/`level_out` from zensim/src/dvifm.rs
in numpy so constants can be refit without re-extracting pixels.

Record order: [m, peak, cmax_s[4], cmin_s[4], cmax_d[4], cmin_d[4]].
"""
import json
import numpy as np
from pathlib import Path

F32 = np.dtype("<f4")
REC = 18
LEVELS = 5
BINS = 5
PER_LEVEL = 1 + BINS


def load_index(bin_path):
    idx_path = str(bin_path) + ".index.jsonl"
    rows = [json.loads(x) for x in Path(idx_path).read_text().splitlines() if x.strip()]
    rows.sort(key=lambda e: e["row_index"])
    return rows


def read_records(bin_path, entry):
    """Return [level] -> (nblocks, 18) f32 array for one index entry."""
    n = sum(entry["level_records"])
    with open(bin_path, "rb") as f:
        f.seek(entry["offset"])
        buf = np.frombuffer(f.read(n * REC * 4), dtype=F32).astype(np.float64)
    out, off = [], 0
    for cnt in entry["level_records"]:
        out.append(buf[off:off + cnt * REC].reshape(cnt, REC))
        off += cnt * REC
    return out


def phi_g(x, g):
    return np.sign(x) * np.abs(x) ** g


def log1pexp(x):
    return np.logaddexp(0.0, x)


def visibility(c, lp):
    c = np.asarray(c, dtype=np.float64)
    out = np.ones_like(c)
    pos = c > 0.0
    if not np.any(pos):
        return out
    k = lp["beta"] * lp["sharp"]
    ell = np.log(c[pos])
    lower = log1pexp(k * (ell - np.log(lp["c0"])))
    upper = log1pexp(k * (ell - np.log(lp["c_hi"]))) if np.isfinite(lp["c_hi"]) else 0.0
    out[pos] = np.exp(-(lower - upper) / lp["sharp"])
    return out


def contrast_g(recs, side, g, edge):
    """recs: (n,18). side 0 -> cmax_s/cmin_s (cols 2:6 / 6:10); side 1 -> 10:14 / 14:18."""
    cmax = recs[:, 2 + side * 8:6 + side * 8]
    cmin = recs[:, 6 + side * 8:10 + side * 8]
    if edge:
        cq = phi_g(cmax, g) - phi_g(cmin, g)
        return cq.min(axis=1)
    return phi_g(cmax.max(axis=1), g) - phi_g(cmin.min(axis=1), g)


def hat_memberships(ell, centers):
    """Triangular memberships, ends clamped (searchsorted side='left')."""
    centers = np.asarray(centers, dtype=np.float64)
    ell = np.clip(ell, centers[0], centers[-1])
    i = np.searchsorted(centers, ell, side="left")
    j = np.clip(i - 1, 0, len(centers) - 2)
    t = (ell - centers[j]) / (centers[j + 1] - centers[j])
    h = np.zeros((ell.shape[0], len(centers)))
    n = np.arange(ell.shape[0])
    h[n, j] = 1.0 - t
    h[n, j + 1] = t
    return h


def pool_level(recs, lp):
    """(n,18) -> [F1, F2_0..F2_4] means over blocks; zeros when empty."""
    if recs.shape[0] == 0:
        return np.zeros(PER_LEVEL)
    cs = contrast_g(recs, 0, lp["g"], lp["edge"])
    cd = contrast_g(recs, 1, lp["g"], lp["edge"])
    vb = np.maximum(visibility(cs, lp), visibility(cd, lp))
    e = recs[:, 0] ** lp["p"]
    f1 = (vb * e).mean()
    ell = np.log(np.minimum(cs, cd) + 1e-6)
    h = hat_memberships(ell, lp["f2_centers"])
    f2 = (h * e[:, None]).mean(axis=0)
    return np.concatenate([[f1], f2])


def pool_pair(recs_by_level, spec):
    """ -> 30 features (5 levels x [F1,F2x5]), matching f956..f985 order."""
    return np.concatenate([pool_level(recs_by_level[l], spec["levels"][l]) for l in range(LEVELS)])


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for lv in spec["levels"]:
        if lv.get("c_hi") is None:
            lv["c_hi"] = np.inf
        lv.setdefault("edge", True)
    return spec
