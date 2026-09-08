#!/usr/bin/env python3
"""Corruption-corpus gate: for every structural-corruption entry, check
score(ref, corruption) < score(ref, q20-honest-lq) through a zensim bake.

The gate is the property zensim's negative tail must satisfy for the
regression-test use case (codec-corpus#7): a structurally-broken decode must
rank BELOW an honestly-lossy encode, so a test catches the bug instead of
passing it.

Usage:
  python3 scripts/v_next/corruption_gate_eval.py <bake.bin> <corruption_out_dir> <ref.png> [label]
"""
import sys, os, glob, subprocess, re
from concurrent.futures import ThreadPoolExecutor

SCORE = "./target/release/score_pair_with_bake"
TILE = "./target/release/score_tiles_with_bake"
TILE_MIN = os.environ.get("TILE_MIN") == "1"  # task #33: tile-min pooling


def audit_report(argv):
    """Analyze complete Rust-surface audit records, without another scorer."""
    import argparse, hashlib, json, math
    from pathlib import Path
    from collections import defaultdict
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-jsonl", required=True)
    ap.add_argument("--inputs-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model-context", choices=["historical", "canonical-fit"], default="historical")
    a = ap.parse_args(argv)
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    def sha(path):
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    require(not Path(a.out_json).exists(), "report output must be fresh")
    inputs = json.loads(Path(a.inputs_json).read_text())
    require(inputs["schema"] == "canonical-corruption-serving-inputs-v1", "input schema")
    for path, expected in inputs["files_sha256"].items():
        require(sha(path) == expected, f"changed input: {path}")
    expected = {r["index"]: r for r in inputs["records"]}
    require(len(expected) == len(inputs["records"]) and bool(expected), "duplicate/empty input keys")
    seen, rows, model_inputs, precision_mode = set(), [], None, None
    with open(a.audit_jsonl) as f:
        for line in f:
            record = json.loads(line)
            require(record["schema"] == "canonical-feature-audit-v1", "audit schema")
            key = record["human_score"]
            require(isinstance(key, (int, float)) and math.isfinite(key) and int(key) == key, "invalid audit key")
            key = int(key)
            require(key in expected and key not in seen, "duplicate/unknown audit key")
            seen.add(key); meta = expected[key]
            require(record["reference"] == meta["reference"] and record["distorted"] == meta["distorted"], "audit path mismatch")
            require(record["distorted_file_sha256"] == meta["expected_distorted_file_sha256"], "distorted SHA mismatch")
            if "expected_distorted_pixels_sha256" in meta:
                require(record["distorted_pixels_sha256"] == meta["expected_distorted_pixels_sha256"], "distorted pixel mismatch")
            require(record["pixels_identical"] == (record["reference_pixels_sha256"] == record["distorted_pixels_sha256"]), "identity evidence mismatch")
            numbers = [record[n] for n in ("base_score", "pixel_composed_score", "cached_composed_score",
                       "literal_feature_composed_score", "head_probability", "head_threshold", "max_consumed_feature_abs_delta")]
            require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers), "nonfinite audit result")
            require(0 <= record["head_probability"] <= 1 and 0 <= record["head_threshold"] <= 1, "probability range")
            require(abs(record["pixel_composed_score"] - record["cached_composed_score"]) <= 1e-4, "surface score mismatch")
            has_precision = "stored_f32_composed_score" in record
            if precision_mode is None:
                precision_mode = has_precision
            require(has_precision == precision_mode, "mixed precision-check coverage")
            if has_precision:
                ps, pp = record["stored_f32_composed_score"], record["stored_f32_head_probability"]
                require(math.isfinite(ps) and math.isfinite(pp) and 0 <= pp <= 1, "nonfinite stored-f32 result")
                require(abs(ps-record["pixel_composed_score"]) <= 1e-4, "stored-f32 score mismatch")
                require((pp > record["head_threshold"]) == (record["head_probability"] > record["head_threshold"]), "stored-f32 fire mismatch")
            if record["pixels_identical"]:
                require(record["base_score"] == record["pixel_composed_score"] == 100, "identity score mismatch")
            require(not meta["label"] or not record["pixels_identical"], "positive identity label")
            if model_inputs is None:
                model_inputs = record["model_inputs"]
                require(len(model_inputs) == 2, "expected complete base and companion")
                for path, digest in model_inputs:
                    require(sha(path) == digest, "model SHA mismatch")
            require(record["model_inputs"] == model_inputs, "mixed model composition")
            rows.append(dict(meta=meta, audit=record))
    require(seen == set(expected), "missing audit rows")
    anchors = {}
    for r in rows:
        m = r["meta"]
        if m["kind"] == "honest_anchor":
            quality = Path(m["distorted"]).stem
            require(quality in ("anchor-q10", "anchor-q20"), "unknown anchor")
            key = (m["role"], m["origin"], quality)
            require(key not in anchors, "duplicate anchor")
            anchors[key] = r["audit"]
    for role, origin in {(r["meta"]["role"], r["meta"]["origin"]) for r in rows}:
        require(all((role, origin, f"anchor-q{q}") in anchors for q in (10, 20)), "missing anchor")
    def rate(values):
        return dict(n=len(values), count=sum(values), rate=sum(values)/len(values) if values else None)
    def summarize(group):
        unique = {}
        for r in group:
            m, v = r["meta"], r["audit"]
            key = (m["origin"], v["reference_pixels_sha256"], v["distorted_pixels_sha256"])
            if key in unique:
                old = unique[key]
                require(old["meta"]["label"] == m["label"], "duplicate label conflict")
                require(all(old["audit"][f] == v[f] for f in ("head_probability", "base_score", "pixel_composed_score")), "duplicate score conflict")
            unique.setdefault(key, r)
        rs = list(unique.values()); pos = [r for r in rs if r["meta"]["label"]]; neg = [r for r in rs if not r["meta"]["label"]]
        out = dict(raw_rows=len(group), unique_pixel_pairs=len(rs), removed_duplicates=len(group)-len(rs), positives=len(pos), negatives=len(neg),
                   detection=rate([r["audit"]["head_probability"] > r["audit"]["head_threshold"] for r in pos]),
                   head_fp=rate([r["audit"]["head_probability"] > r["audit"]["head_threshold"] for r in neg]),
                   honest_score_lowered=rate([r["audit"]["pixel_composed_score"] < r["audit"]["base_score"] for r in neg]),
                   identities=sum(r["audit"]["pixels_identical"] for r in rs))
        for q in (10, 20):
            for score, name in [("base_score", "base"), ("pixel_composed_score", "composed")]:
                out[f"{name}_below_q{q}"] = rate([r["audit"][score] < anchors[(r["meta"]["role"], r["meta"]["origin"], f"anchor-q{q}")][score] for r in pos])
        return out
    result = dict(schema="canonical-corruption-serving-report-v1", model_qualified=False,
                  claim=("development screen of historical frozen models; no new fit or holdout qualification"
                         if a.model_context == "historical" else
                         "development screen of exact canonical head fit composed with frozen base; no product qualification"),
                  audit_sha256=sha(a.audit_jsonl), inputs_sha256=sha(a.inputs_json), model_inputs=model_inputs,
                  complete_rows=len(rows), max_feature_abs_delta=max(r["audit"]["max_consumed_feature_abs_delta"] for r in rows),
                  max_score_abs_delta=max(abs(r["audit"]["pixel_composed_score"]-r["audit"]["cached_composed_score"]) for r in rows),
                  identity_rows=sum(r["audit"]["pixels_identical"] for r in rows), splits={})
    result["stored_f32_check_rows"] = len(rows) if precision_mode else 0
    if precision_mode:
        result["max_stored_f32_score_delta"] = max(abs(r["audit"]["stored_f32_composed_score"]-r["audit"]["pixel_composed_score"]) for r in rows)
        result["max_stored_f32_probability_delta"] = max(abs(r["audit"]["stored_f32_head_probability"]-r["audit"]["head_probability"]) for r in rows)
    for role in sorted({r["meta"]["role"] for r in rows}):
        group = [r for r in rows if r["meta"]["role"] == role]
        summary = summarize(group)
        for field in ("origin", "content_class", "family", "kind", "codec"):
            keys = sorted({r["meta"].get(field) for r in group if r["meta"].get(field) is not None})
            summary[f"by_{field}"] = {key: summarize([r for r in group if r["meta"].get(field) == key]) for key in keys}
        result["splits"][role] = summary
    Path(a.out_json).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"complete: {len(rows)} rows; identity={result['identity_rows']}; model remains unqualified")


def score(bake, ref, dist):
    if TILE_MIN:
        # tile-min: localized-defect signal. Output cols: global min p2 p5 median n.
        r = subprocess.run([TILE, "--bake", bake, "--bake-post", "clamp", "--ref", ref,
                            "--dist", dist, "--tile", os.environ.get("TILE_SIZE", "64"),
                            "--overlap", "0.5"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return None
        try:
            return float(r.stdout.strip().split()[1])  # min tile
        except (ValueError, IndexError):
            return None
    r = subprocess.run([SCORE, "--bake", bake, "--bake-post", "raw", "--ref", ref, "--dist", dist],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--audit-jsonl":
        audit_report(sys.argv[1:])
        return
    bake, out_dir, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(bake)
    corruptions = sorted(glob.glob(os.path.join(out_dir, "*__corruption.png")))
    print(f"[{label}] {len(corruptions)} corruption entries vs q20/q10 anchors", file=sys.stderr)

    def work(cpath):
        key = cpath[:-len("__corruption.png")]
        q20p, q10p = key + "__q20.png", key + "__q10.png"
        name = os.path.basename(key)
        m = re.match(r".*?__([a-z_0-9]+?)__(whole|frac2|frac4|sq64|sq16|sq8)__(op20|op50|op100)$", name)
        fam, region, sev = (m.group(1), m.group(2), m.group(3)) if m else (name, "?", "?")
        sc = score(bake, ref, cpath)
        sq20 = score(bake, ref, q20p) if os.path.exists(q20p) else None
        sq10 = score(bake, ref, q10p) if os.path.exists(q10p) else None
        return dict(name=name, fam=fam, region=region, sev=sev, sc=sc, sq20=sq20, sq10=sq10)

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(work, corruptions))

    rows = [r for r in rows if r["sc"] is not None and r["sq20"] is not None]
    n = len(rows)
    passed = [r for r in rows if r["sc"] < r["sq20"]]
    print(f"\n[{label}] GATE score(corruption) < score(q20): {len(passed)}/{n} = {len(passed)/n*100:.1f}% PASS")
    # also vs q10 (more aggressive anchor)
    rows10 = [r for r in rows if r["sq10"] is not None]
    p10 = [r for r in rows10 if r["sc"] < r["sq10"]]
    print(f"[{label}] GATE score(corruption) < score(q10): {len(p10)}/{len(rows10)} = {len(p10)/max(len(rows10),1)*100:.1f}% PASS")

    # per-family pass rate
    fams = {}
    for r in rows:
        fams.setdefault(r["fam"], []).append(r["sc"] < r["sq20"])
    print("\nper-family gate(vs q20) pass rate:")
    for fam in sorted(fams):
        v = fams[fam]
        print(f"  {fam:34s} {sum(v):3d}/{len(v):3d}  {sum(v)/len(v)*100:5.1f}%")

    # per-region (subtlety axis: smaller region = harder)
    regs = {}
    for r in rows:
        regs.setdefault(r["region"], []).append(r["sc"] < r["sq20"])
    print("\nper-region gate(vs q20) pass rate (smaller = harder/subtler):")
    for reg in ["whole", "frac2", "frac4", "sq64", "sq16", "sq8"]:
        if reg in regs:
            v = regs[reg]
            print(f"  {reg:8s} {sum(v):3d}/{len(v):3d}  {sum(v)/len(v)*100:5.1f}%")

    # worst FAILURES (corruption scored ABOVE q20 = metric let a bug pass)
    fails = sorted([r for r in rows if r["sc"] >= r["sq20"]], key=lambda r: r["sc"] - r["sq20"], reverse=True)
    print(f"\n{len(fails)} FAILURES (corruption >= q20). worst 20:")
    print(f"  {'name':52s} {'corr':>8s} {'q20':>8s}")
    for r in fails[:20]:
        print(f"  {r['name']:52s} {r['sc']:8.1f} {r['sq20']:8.1f}")


if __name__ == '__main__':
    main()
