#!/usr/bin/env python3
"""Build verified aggregate JSON (<30 KB) for the public zensim evaluations page.

Reads completed frozen-A score caches and verifies every scalar against the
committed RESULTS.json records before emitting aggregate 8x8 density grids,
tie-preserving quantile medians/IQRs, CLIC margin bins, and KADID breakdowns.
Explanatory copy and authored report tables live in evaluations.html / evaluations.js.
Never emits row-level targets, stimulus IDs, local paths, or private corpus metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

A_SHA256 = "d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a"
ZENSIM_DOC_REV = "f1cb811a50dfe969bbf548725a1f8f5d661bd4e3"
ZENSIMX_RECORD_REV = "f3bc01a9ac8fd664c4798250aab628c6e050243c"
VIZ_SOURCE_REV = "5d2c1a50cd2af9ea0111dae4e6169caf0e0c7b37"
MAX_JSON_BYTES = 30_000

MODEL_SHORT = {
    "A": "A",
    "SSIMULACRA2": "S2",
    "butteraugli3_quality": "BA",
    "B": "B",
    "by_v2fy": "V2",
}

FORBIDDEN = re.compile(
    r"/home/|/mnt/|/var/|/srv/|/tmp/|source_paths|zenpapers|imazen-26|"
    r"\b(?:tower|r7900x|r5900xt|i265|i270|i134|r3500|r3800x|lilith-wsl)\b|"
    r"\.lan\b|aic[-_ ]?2026|public_test|normalized_pred",
    re.I,
)

MCL_CLEAN40_EXCLUDED = {1, 3, 6, 7, 9, 17, 28, 31, 45, 50}


def r6(v: float | None) -> float | None:
    return None if v is None else round(float(v), 6)


def r3(v: float | None) -> float | None:
    return None if v is None else round(float(v), 3)


def load_continuous_set(root: Path, set_name: str):
    keys = json.loads((root / "inputs" / set_name / "KEYS.json").read_text())["rows"]
    with (root / "scores" / set_name / "pred_A.tsv").open() as f:
        preds = list(csv.DictReader(f, delimiter="\t"))
    assert len(keys) == len(preds), (set_name, len(keys), len(preds))
    scores = {
        "A": {k["pair_key"]: float(p["pred"]) for k, p in zip(keys, preds, strict=True)}
    }
    with (root / "scores" / set_name / "peers.tsv").open() as f:
        peers = list(csv.DictReader(f, delimiter="\t"))
    scores["SSIMULACRA2"] = {p["id"]: float(p["ssim2"]) for p in peers}
    scores["butteraugli3_quality"] = {p["id"]: -float(p["ba_p3"]) for p in peers}
    if set_name == "kadid_terminal":
        for comp in ("B", "by_v2fy"):
            with (root / "scores" / set_name / f"pred_{comp}.tsv").open() as f:
                cp = list(csv.DictReader(f, delimiter="\t"))
            assert len(cp) == len(keys)
            scores[comp] = {
                k["pair_key"]: float(list(row.values())[-1])
                for k, row in zip(keys, cp, strict=True)
            }
    eval_rows = json.loads((root / "scores" / set_name / "eval_rows.json").read_text())
    return eval_rows, scores


def tie_preserving_quantile_bins(x: np.ndarray, max_bins: int) -> list[np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(xs)
    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        groups.append((i, j))
        i = j
    if len(groups) <= max_bins:
        return [order[a:b] for a, b in groups]
    bins: list[np.ndarray] = []
    cur_start = 0
    for b_idx in range(1, max_bins):
        target_end = round(b_idx * n / max_bins)
        best = None
        for _, g_end in groups:
            if g_end <= cur_start:
                continue
            if best is None or abs(g_end - target_end) < abs(best - target_end):
                best = g_end
        if best is not None and best < n:
            bins.append(order[cur_start:best])
            cur_start = best
    if cur_start < n:
        bins.append(order[cur_start:n])
    assert sum(len(b) for b in bins) == n
    return bins


def density_grid_8x8(
    x: np.ndarray, y: np.ndarray, y_range: tuple[float, float], max_qbins: int
) -> dict:
    nb = 8
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = y_range
    xedges = np.linspace(xmin, xmax, nb + 1)
    yedges = np.linspace(ymin, ymax, nb + 1)
    xi = np.searchsorted(xedges[1:-1], x, side="right")
    yi = np.searchsorted(yedges[1:-1], y, side="right")
    counts = np.zeros((nb, nb), dtype=int)
    for ix, iy in zip(xi, yi):
        counts[int(ix), int(iy)] += 1
    assert int(counts.sum()) == len(x)
    cells: list[int] = []
    for ix in range(nb):
        for iy in range(nb):
            c = int(counts[ix, iy])
            if c > 0:
                cells.extend((ix * nb + iy, c))
    qbins = tie_preserving_quantile_bins(x, max_qbins)
    quantiles = []
    for idx in qbins:
        xb = x[idx]
        yb = y[idx]
        quantiles.append(
            [
                r3(float(np.median(xb))),
                r3(float(np.median(yb))),
                r3(float(np.quantile(yb, 0.25, method="linear"))),
                r3(float(np.quantile(yb, 0.75, method="linear"))),
                int(len(idx)),
            ]
        )
    assert sum(q[4] for q in quantiles) == len(x)
    return {"xr": [r3(xmin), r3(xmax)], "c": cells, "q": quantiles}


def extract_metric_panel(mdata: dict) -> list:
    # Compact 5-tuple: [srocc, krocc, plcc, rmse, within_or_null], each [val, lo, hi]
    out = []
    for k in ("srocc_signed", "krocc", "plcc", "rmse"):
        d = mdata[k]
        out.append([r6(d["value"]), r6(d["lo"]), r6(d["hi"])])
    w = mdata["within_reference_srocc"]
    out.append(None if w["value"] is None else [r6(w["value"]), r6(w["lo"]), r6(w["hi"])])
    return out


def build_clic_margins(root: Path) -> dict:
    keys = json.loads((root / "inputs/clic22/KEYS.json").read_text())["rows"]
    with (root / "scores/clic22/pred_A.tsv").open() as f:
        preds = list(csv.DictReader(f, delimiter="\t"))
    scores = {
        "A": {k["pair_key"]: float(p["pred"]) for k, p in zip(keys, preds, strict=True)}
    }
    with (root / "scores/clic22/peers.tsv").open() as f:
        peers = list(csv.DictReader(f, delimiter="\t"))
    scores["SSIMULACRA2"] = {p["id"]: float(p["ssim2"]) for p in peers}
    scores["butteraugli3_quality"] = {p["id"]: -float(p["ba_p3"]) for p in peers}
    er = json.loads((root / "scores/clic22/eval_rows.json").read_text())
    res = json.loads((root / "scores/clic22/RESULTS.json").read_text())

    models_out = {}
    for m in ("A", "SSIMULACRA2", "butteraugli3_quality"):
        mp = scores[m]
        deltas = np.empty(len(er), dtype=np.float64)
        agree = np.empty(len(er), dtype=np.float64)
        for i, r in enumerate(er):
            d = mp[r["a"]] - mp[r["b"]]
            t = float(r["target"])
            deltas[i] = abs(d)
            agree[i] = 0.5 if d == 0.0 or t == 0.5 else (t if d > 0.0 else 1.0 - t)
        full_mean = float(agree.mean())
        rec = float(res["models"][m]["agreement"])
        assert abs(full_mean - rec) < 1e-9, (m, full_mean, rec)

        zero_mask = deltas == 0.0
        bins_list = []
        if int(zero_mask.sum()) > 0:
            bins_list.append([0.0, 0.0, int(zero_mask.sum()), r6(float(agree[zero_mask].mean()))])
        pos_idx = np.flatnonzero(~zero_mask)
        qbins = tie_preserving_quantile_bins(deltas[pos_idx], 10)
        for sub in qbins:
            idx = pos_idx[sub]
            db = deltas[idx]
            ab = agree[idx]
            bins_list.append([r3(float(db.min())), r3(float(db.max())), int(len(idx)), r6(float(ab.mean()))])
        tot_n = sum(b[2] for b in bins_list)
        assert tot_n == len(er)
        weighted = sum(b[2] * b[3] for b in bins_list) / tot_n
        assert abs(weighted - rec) < 5e-6
        models_out[MODEL_SHORT[m]] = {
            "maj": [r6(res["models"][m]["agreement"]), r6(res["models"][m]["lo"]), r6(res["models"][m]["hi"])],
            "soft": [
                r6(res["supplementary_soft_vote"]["models"][m]["agreement"]),
                r6(res["supplementary_soft_vote"]["models"][m]["lo"]),
                r6(res["supplementary_soft_vote"]["models"][m]["hi"]),
            ],
            "bins": bins_list,
        }

    return {"rows": int(res["rows"]), "refs": int(res["references"]), "models": models_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-read-root", type=Path, required=True)
    ap.add_argument("--live-overlap-py", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.live_overlap_py.parent))
    import external_sets  # type: ignore

    live_overlap: set[str] = set(external_sets.CONTENT_OVERLAP["live"])

    views_spec = [
        ("cid22_b", "all", "cid22_b", 10, lambda r: True),
        ("csiq", "all", "csiq", 10, lambda r: True),
        ("live", "all", "live_all", 10, lambda r: True),
        ("live", "clean10", "live_clean10", 8, lambda r: (r["group"] + ".bmp") not in live_overlap),
        ("kadid_terminal", "all", "kadid_terminal", 10, lambda r: True),
        ("aic4", "all", "aic4", 8, lambda r: True),
        ("aic4", "SDR25", "sdr25", 5, lambda r: r["codec"] == "JPEG-AI"),
        ("mcljci", "all", "mcljci_all", 10, lambda r: True),
        ("mcljci", "R7a_clean40", "mcljci_clean40", 10, lambda r: r["source"] not in MCL_CLEAN40_EXCLUDED),
        ("konjnd_terminal", "all", "konjnd_terminal", 5, lambda r: True),
    ]

    cached_sets: dict[str, tuple] = {}
    views_out = []
    for set_dir, sub_dir, vid, max_qbins, pred_fn in views_spec:
        if set_dir not in cached_sets:
            cached_sets[set_dir] = load_continuous_set(args.final_read_root, set_dir)
        er_all, score_maps = cached_sets[set_dir]
        rows = [r for r in er_all if pred_fn(r)]
        res = json.loads((args.final_read_root / "scores" / set_dir / sub_dir / "RESULTS.json").read_text())
        assert len(rows) == res["rows"], (vid, len(rows), res["rows"])
        y = np.array([float(r["target"]) for r in rows], dtype=np.float64)
        y_range = (float(y.min()), float(y.max()))

        model_order = (
            ["A", "B", "by_v2fy", "SSIMULACRA2", "butteraugli3_quality"]
            if set_dir == "kadid_terminal"
            else ["A", "SSIMULACRA2", "butteraugli3_quality"]
        )
        m_out = {}
        sc_out = {}
        for m in model_order:
            x = np.array([score_maps[m][r["a"]] for r in rows], dtype=np.float64)
            sr = float(spearmanr(x, y).statistic)
            rec = float(res["models"][m]["srocc_signed"]["value"])
            assert abs(sr - rec) < 1e-12, (vid, m, sr, rec)
            mk = MODEL_SHORT[m]
            m_out[mk] = extract_metric_panel(res["models"][m])
            sc_out[mk] = density_grid_8x8(x, y, y_range, max_qbins)

        views_out.append(
            {
                "id": vid,
                "rows": int(res["rows"]),
                "refs": int(res["references"]),
                "yr": [r3(y_range[0]), r3(y_range[1])],
                "m": m_out,
                "sc": sc_out,
            }
        )

    kadid_res = json.loads((args.final_read_root / "scores/kadid_terminal/all/RESULTS.json").read_text())
    conf = kadid_res["confirmation"]
    kadid_conf = {
        "B": [
            r6(kadid_res["models"]["B"]["srocc_signed"]["value"]),
            r6(conf["delta_A_minus_B"]),
            r6(conf["lower_limit"]),
            r6(conf["lo"]),
            r6(conf["hi"]),
            bool(conf["pass_B"]),
        ],
        "V2": [
            r6(kadid_res["models"]["by_v2fy"]["srocc_signed"]["value"]),
            r6(conf["delta_A_minus_by_v2fy"]),
            r6(conf["by_v2fy_limit"]),
            r6(conf["by_v2fy_delta_interval"]["lo"]),
            r6(conf["by_v2fy_delta_interval"]["hi"]),
            bool(conf["pass_by_v2fy"]),
        ],
    }

    kadid_types = {}
    for code in sorted(kadid_res["per_distortion_srocc"].keys()):
        entry = kadid_res["per_distortion_srocc"][code]
        kadid_types[code] = {
            MODEL_SHORT[m]: [
                r6(entry[m]["signed"]["estimate"]),
                r6(entry[m]["signed"]["lo"]),
                r6(entry[m]["signed"]["hi"]),
            ]
            for m in ("A", "B", "by_v2fy", "SSIMULACRA2", "butteraugli3_quality")
        }

    kadid_diag = {}
    for m in ("A", "B", "by_v2fy", "SSIMULACRA2", "butteraugli3_quality"):
        sc = json.loads((args.final_read_root / f"scores/kadid_terminal/all/scatter-{m}.json").read_text())
        kadid_diag[MODEL_SHORT[m]] = {
            "geo": {k: r6(v) for k, v in sc["geo"].items()},
            "raw": {k: r6(v) for k, v in sc["raw"].items()},
        }

    payload = {
        "schema": "zensim-evals-v1",
        "commits": {
            "bake_sha256": A_SHA256,
            "zensim": ZENSIM_DOC_REV,
            "zensimx": ZENSIMX_RECORD_REV,
            "viz": VIZ_SOURCE_REV,
        },
        "views": views_out,
        "clic22": build_clic_margins(args.final_read_root),
        "kadid_conf": kadid_conf,
        "kadid_types": kadid_types,
        "kadid_diag": kadid_diag,
    }

    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    m = FORBIDDEN.search(encoded.decode("utf-8"))
    if m:
        raise ValueError(f"Forbidden token in evaluations JSON: {m.group(0)!r}")
    if len(encoded) >= MAX_JSON_BYTES:
        raise ValueError(f"evaluations JSON {len(encoded)} bytes >= {MAX_JSON_BYTES} limit")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(encoded)
    print(json.dumps({"bytes": len(encoded), "limit": MAX_JSON_BYTES, "views": len(views_out)}))


if __name__ == "__main__":
    main()
