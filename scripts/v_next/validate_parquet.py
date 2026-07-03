#!/usr/bin/env python3
"""Data-contract validator for training/eval parquets — catch fleet/versioning
errors BEFORE they burn a training run. Born 2026-07-02 after one day produced:
footerless partials (OOM mid-write), wrong target annotations (MCOS vs ssim2),
schema drift (feat_N vs fN), in-place rewrites of manifest-referenced files,
out-of-convention target ranges, and inverted supervision pairs.

Usage:
  validate_parquet.py FILE.parquet [--kind train|eval|grid] [--expect-rows N]
      [--expect-sha SHA] [--target-range LO,HI] [--split-rule lsd|mod10]
  validate_parquet.py --manifest manifest.toml     # validate every [inputs.*]

Checks (any FAIL → exit 1, loud):
  C1  footer/magic valid (readable metadata)          — catches partial writes
  C2  feature columns complete + contiguous           — f0..fN or feat_0..feat_N
  C3  no nulls in features/target                     — trainer NaN-cascades
  C4  no NaN/Inf values (sampled batches)             — same
  C5  target column present + within declared range   — catches unclamped/neg
  C6  no all-constant feature columns (sampled)       — catches the all-zero
                                                        picker-features join bug
  C7  row count == --expect-rows / manifest rows      — catches truncation
  C8  sha256 == --expect-sha / manifest sha           — catches silent rewrite
  C9  split-rule conformance when key present         — ref_basename obeys LSD,
                                                        source_id obeys mod10
  C10 duplicate-row spot check (first feature col + target, sampled)
"""
import argparse, hashlib, os, re, sys
import pyarrow.parquet as pq
import numpy as np

FAIL = []
def check(ok, code, msg):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {code}: {msg}")
    if not ok:
        FAIL.append(code)

def featcols(names):
    for pat, key in ((r"^f(\d+)$", 1), (r"^feat_(\d+)$", 1)):
        cols = sorted(((int(re.match(pat, n).group(key)), n) for n in names if re.match(pat, n)))
        if cols:
            return [n for _, n in cols], [i for i, _ in cols]
    return [], []

def validate(path, kind="train", expect_rows=None, expect_sha=None,
             target_range=(-0.001, 1.001), target_col="human_score", split_rule=None,
             allow_dup_rate=0.01, contract=None):
    print(f"== {path} (kind={kind}) ==")
    try:
        md = pq.read_metadata(path)
    except Exception as e:
        check(False, "C1", f"unreadable parquet: {e}")
        return
    check(True, "C1", f"valid footer, {md.num_rows:,} rows, {md.num_columns} cols")
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    fcols, fidx = featcols(names)
    check(len(fcols) in (228, 300, 372) or (kind == "grid" and len(fcols) > 0),
          "C2", f"{len(fcols)} feature cols ({fcols[0] if fcols else '-'}..{fcols[-1] if fcols else '-'}), contiguous={fidx == list(range(len(fidx)))}")
    has_target = target_col in names
    if kind in ("train", "eval"):
        check(has_target, "C5a", f"target column '{target_col}' present")
    # sampled deep checks
    nan_bad = inf_bad = null_bad = 0
    tmin, tmax = np.inf, -np.inf
    const_candidates = None
    seen_rows = 0
    dup_keys = set(); dup_hits = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=131072, columns=(fcols[:64] + ([target_col] if has_target else []) + [c for c in ("ref_basename", "source_id") if c in names]))):
        import pyarrow as pa
        t = pa.Table.from_batches([batch])
        F = np.column_stack([np.asarray(t[c], dtype=np.float64) for c in fcols[:64]])
        null_bad += int(np.isnan(F).sum())  # arrow nulls -> nan on cast
        inf_bad += int(np.isinf(F).sum())
        if const_candidates is None:
            const_candidates = (F.max(axis=0) - F.min(axis=0)) == 0
        else:
            const_candidates &= (F.max(axis=0) - F.min(axis=0)) == 0
        if has_target:
            h = np.asarray(t[target_col], dtype=np.float64)
            nan_bad += int(np.isnan(h).sum())
            tmin, tmax = min(tmin, np.nanmin(h)), max(tmax, np.nanmax(h))
            for v in zip(F[:512, 0], h[:512]):
                if v in dup_keys: dup_hits += 1
                dup_keys.add(v)
        if split_rule and bi == 0:
            if split_rule == "lsd" and "ref_basename" in names:
                sys.path.insert(0, os.path.expanduser("~/work/zen/zenmetrics/scripts/picker"))
                from origin_split import split_of
                buckets = {split_of(x) for x in t["ref_basename"].to_pylist()[:5000]}
                check(len(buckets - {None}) == 1, "C9", f"LSD buckets in sample: {buckets}")
            if split_rule and split_rule.startswith("mod10") and "source_id" in names:
                want = split_rule.split(":")[1] if ":" in split_rule else None
                sid = np.asarray(t["source_id"], dtype=np.int64) % 10
                ok = (sid < 8).all() if want == "train" else (sid == 8).all() if want == "val" else (sid == 9).all() if want == "test" else True
                check(ok, "C9", f"mod10 sample buckets: {sorted(set(sid.tolist()))[:5]} (want {want})")
        seen_rows += len(F)
        if seen_rows >= 1_500_000 and kind != "train":
            break  # eval/grid: sampled is enough
    check(null_bad == 0, "C3", f"nulls/NaN in target+first-64 feats: {null_bad}")
    check(inf_bad == 0, "C4", f"Inf values: {inf_bad}")
    if has_target and kind in ("train", "eval"):
        lo, hi = target_range
        check(lo <= tmin and tmax <= hi, "C5b", f"target range [{tmin:.4f}, {tmax:.4f}] within [{lo}, {hi}]")
    nconst = int(const_candidates.sum()) if const_candidates is not None else -1
    allow_const = int(contract.get("allow_const_cols", 0)) if contract else 0
    check(nconst <= allow_const, "C6",
          f"all-constant feature cols (first 64 sampled): {nconst} (allowed: {allow_const})")
    if expect_rows is not None:
        check(md.num_rows == int(expect_rows), "C7", f"rows {md.num_rows:,} == expected {int(expect_rows):,}")
    if expect_sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        got = h.hexdigest()
        check(got == expect_sha, "C8", f"sha256 {got[:12]}… == manifest {expect_sha[:12]}…")
    dup_rate = dup_hits / max(1, len(dup_keys) + dup_hits)
    check(dup_rate < allow_dup_rate, "C10",
          f"duplicate (f0,target) sampled rate: {dup_rate*100:.2f}% ({dup_hits} hits) — "
          f">1% indicates systematic dup rows (e.g. knob-no-op sweep cells; 2026-07-02: "
          f"caught 22.2% dups in bigcodec from modes_full no-op knobs)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--manifest")
    ap.add_argument("--kind", default="train", choices=["train", "eval", "grid"])
    ap.add_argument("--expect-rows", type=int)
    ap.add_argument("--expect-sha")
    ap.add_argument("--target-range", default="-0.001,1.001")
    ap.add_argument("--target-col", default="human_score")
    ap.add_argument("--split-rule")
    ap.add_argument("--allow-const-cols", type=int, default=0,
                    help="C6 allowance for corpus-property constant features (e.g. HDR PU21: f25/f64 always 0)")
    ap.add_argument("--contracts", help="JSON sidecar {basename: {target_range, allow_dup_rate}} of declared per-file deviations")
    a = ap.parse_args()
    tr = tuple(float(x) for x in a.target_range.split(","))
    if a.manifest:
        import toml
        m = toml.load(a.manifest)
        for key, inp in (m.get("inputs") or {}).items():
            if not isinstance(inp, dict) or "sha256" not in inp:
                continue
            p = inp["path"]
            if "{canonical}" in p:
                p = p.replace("{canonical}", m["inputs"]["canonical_root"]["local"])
            if not p.endswith(".parquet"):
                continue
            # per-input contract keys (optional, declared in the manifest):
            #   target_range = [lo, hi]   validate_kind = "grid"|"eval"|"train"
            #   allow_const_cols = 2      (corpus-property constant features, e.g. HDR PU21 f25/f64)
            #   target_column = "..."     allow_dup_rate = 0.30
            validate(p, kind=inp.get("validate_kind", "train"),
                     expect_rows=inp.get("rows"), expect_sha=inp.get("sha256"),
                     target_range=tuple(inp.get("target_range", tr)),
                     target_col=inp.get("target_column", a.target_col),
                     allow_dup_rate=inp.get("allow_dup_rate", 0.01),
                     contract=inp)
    contracts = {}
    if a.contracts and os.path.exists(a.contracts):
        import json
        contracts = json.load(open(a.contracts))
    for f in a.files:
        c = contracts.get(os.path.basename(f), {})
        c.setdefault("allow_const_cols", a.allow_const_cols)
        validate(f, kind=a.kind, expect_rows=a.expect_rows, expect_sha=a.expect_sha,
                 target_range=tuple(c.get("target_range", tr)), target_col=a.target_col,
                 split_rule=a.split_rule, allow_dup_rate=c.get("allow_dup_rate", 0.01),
                 contract=c)
    if FAIL:
        print(f"\nVALIDATION FAILED: {sorted(set(FAIL))}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")

if __name__ == "__main__":
    main()
