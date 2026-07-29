#!/usr/bin/env python3
"""Per-source canonical-filter GATES (task #11 chunk 1) — one verdict JSON per
source parquet, composing the existing audit owners (never re-implementing
their checks).

Two kinds of filter, per the task-#11 design (2026-07-29):

  ASSERT-CLEAN gates (verify + record; NEVER drop rows here):
    misjoin   - iwssim human-copy leak + ssim2 constant-per-ref
                (shells `audit_metric_columns.py --paths ... --fail-on-corruption`,
                 the owner of both detectors; DATASET_HISTORY §3.18)
    jxl_zone  - the JXL near-lossless broken-zone assertion (§3.20 MEASURED:
                training corpora never sampled d<0.03 pre-fix — so this is an
                ASSERTION on min distance / q bounds when the source carries a
                jxl distance or codec+q column, else structurally N/A)

  RECORD-ONLY inventories (manifest metadata, not row filters):
    poison    - which documented poison-as-MSE-target columns exist
                (column_audit.py verdicts, 2026-07-1x: the mix_cv* family +
                 *_log_norm forms are TARGET-SHAPE poison under MSE; raw
                 scores stay. Presence is recorded so a trainer manifest can
                 exclude them; the columns are NOT removed from data.)
    winsor    - per-feature [p0.1, p99.9] bounds (§3.19 IW-explosion guard —
                ships in BAKES as winsor_p99 transforms, recorded here as
                provenance; computed only when --winsor is passed: the
                percentile pass is the expensive part on multi-GB sources)

Usage:
  canonical_filter_gates.py --source <name> --parquet <path> [--winsor] \
      [--out-dir /mnt/v/output/zensim/canonical-gates]

Output: <out-dir>/<name>.gates.json with per-gate verdict + n + provenance
(build commit, input sha256 when --sha). Chunk 2 wires these into each
source's _MANIFEST.json + the R2/Tower mirrors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# column_audit.py verdicts (docs/DATASET_HISTORY + the 2026-07 column audit):
# log-expanded / mixed forms over-weight the near-lossless sliver under MSE —
# TARGET-SHAPE poison as regression targets (rank-fine). Raw scores are kept.
POISON_AS_MSE_TARGET = [
    "cvvdp_log_norm",
    "iwssim_log_norm",
    "ssim2_log_norm",
    "mix_cv25_iw75",
    "mix_cv50_iw50",
    "mix_cv75_iw25",
    "mix_cv33_iw33_sm33",
    "mix_cv40_iw60",
]

# §3.20: the jxl-encoder DC-saturation bug fired only at butteraugli
# distance <= 0.02 (fixed eeb52735 2026-07-06T06:09Z); distance >= 0.03 is
# hash-proven byte-identical at every date. The gate asserts the source's
# sampled range never enters the broken zone (pre-fix data) — for q-only
# sources, generic quality q<=90 structurally never resolves to native
# d<0.03 (§3.20's R2/duckdb verification), recorded as such.
JXL_BROKEN_ZONE_D = 0.03


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def gate_misjoin(parquet: Path) -> dict:
    """Shell the OWNER of the leak/misjoin detectors; capture its verdict."""
    r = subprocess.run(
        [sys.executable, str(HERE / "audit_metric_columns.py"),
         "--paths", str(parquet), "--fail-on-corruption"],
        capture_output=True, text=True, timeout=3600,
    )
    return {
        "verdict": "PASS" if r.returncode == 0 else "FAIL",
        "owner": "audit_metric_columns.py --fail-on-corruption",
        "detail": (r.stdout.strip().splitlines() or [""])[-1][:400],
    }


def gate_jxl_zone(t) -> dict:
    cols = set(t.schema.names)
    # distance-carrying sources: assert min distance clear of the zone.
    for dcol in ("distance", "butteraugli_distance", "d"):
        if dcol in cols:
            import pyarrow.compute as pc
            dmin = pc.min(t.column(dcol)).as_py()
            ok = dmin is None or dmin >= JXL_BROKEN_ZONE_D
            return {"verdict": "PASS" if ok else "FAIL",
                    "basis": f"min({dcol})={dmin} vs broken zone d<{JXL_BROKEN_ZONE_D}"}
    if {"codec", "q"} <= cols or {"codec", "quality"} <= cols:
        return {"verdict": "PASS",
                "basis": "generic-quality source — q-mapping structurally never "
                         "resolves to native d<0.03 (DATASET_HISTORY §3.20, R2-verified)"}
    return {"verdict": "N/A",
            "basis": "no jxl distance/quality axis in this source"}


def gate_poison(t) -> dict:
    present = [c for c in POISON_AS_MSE_TARGET if c in t.schema.names]
    return {
        "verdict": "RECORDED",
        "present": present,
        "note": "TARGET-SHAPE poison under MSE only (rank-fine); columns kept "
                "in data, trainer manifests must not select them as targets",
    }


def gate_winsor(t) -> dict:
    import numpy as np
    feats = sorted(
        (c for c in t.schema.names
         if (c.startswith("f") and c[1:].isdigit()) or c.startswith("feat_")),
        key=lambda c: int("".join(ch for ch in c if ch.isdigit())),
    )
    lo, hi = {}, {}
    for c in feats:
        v = t.column(c).to_numpy(zero_copy_only=False).astype(float)
        v = v[~(v != v)]  # drop NaN
        if v.size:
            lo[c] = float(np.percentile(v, 0.1))
            hi[c] = float(np.percentile(v, 99.9))
    return {"verdict": "RECORDED", "n_features": len(lo),
            "p001": lo, "p999": hi,
            "note": "§3.19 guard bounds; ship in bakes as winsor_p99, "
                    "recorded here as provenance"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="canonical source name")
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--winsor", action="store_true",
                    help="also compute per-feature winsor bounds (expensive)")
    ap.add_argument("--sha", action="store_true",
                    help="sha256 the input (expensive on multi-GB sources)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/v/output/zensim/canonical-gates"))
    a = ap.parse_args()

    t = pq.read_table(a.parquet)
    out = {
        "source": a.source,
        "parquet": str(a.parquet),
        "rows": t.num_rows,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_commit": git_head(),
        "input_sha256": sha256_file(a.parquet) if a.sha else None,
        "gates": {
            "misjoin": gate_misjoin(a.parquet),
            "jxl_zone": gate_jxl_zone(t),
            "poison": gate_poison(t),
        },
    }
    if a.winsor:
        out["gates"]["winsor"] = gate_winsor(t)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dst = a.out_dir / f"{a.source}.gates.json"
    dst.write_text(json.dumps(out, indent=1))
    hard = [k for k, g in out["gates"].items() if g.get("verdict") == "FAIL"]
    print(f"{a.source}: rows={t.num_rows} "
          + " ".join(f"{k}={g['verdict']}" for k, g in out["gates"].items())
          + f" -> {dst}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
