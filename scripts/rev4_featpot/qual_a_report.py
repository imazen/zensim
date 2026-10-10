"""QUAL-A report: E33 candidate A on every runnable release gate, with C and production seed 0 report-only.

Reads only the QUAL-A outputs (gate runner, Rust-surface audit, runtime grid, test logs) and the gate map's bars;
recomputes nothing a gate owner computes except order statistics (p95) of the retained paired timing rounds.
Writes QUAL_A.json and QUAL_A.md in the evidence root.
"""

import argparse
import json
import math
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "demos")]
from e33_gates import identity_rows  # noqa: E402

MODELS = ("a", "c", "seed0")
NAMES = {"a": "A", "c": "C", "seed0": "seed 0"}
HUNDRED = struct.unpack("<q", struct.pack("<d", 100.0))[0]


def p95(values):
    v = sorted(values)
    return v[min(len(v) - 1, math.ceil(0.95 * len(v)) - 1)]


def verdict(root, m, grid):
    g = json.loads((root / f"verdict-s0-{m}" / grid / "gaddr.json").read_text())
    checks = {c["id"]: (c["measured"], c["state"]) for c in g["checks"] if c["id"].startswith("C")}
    d = g["measured"]["grid"]
    return dict(checks=checks, p5=d["p5"], p95=d["p95"], mono=d["mono"], negtail=g["measured"]["negtail"])


def runtime(root):
    import speedq_run as owner
    out = {}
    tdir = root / "runtime" / "timing"
    if not tdir.is_dir():
        return None
    for cell in sorted(p for p in tdir.iterdir() if p.is_dir() and "." not in p.name):
        done = json.loads((cell / "COMPLETE.json").read_text())
        if done.get("status") != "PASS":
            return None
        inner = json.loads((cell / "zenbench.inner.json").read_text())
        values, _ = owner.select_clean_rounds(inner)
        out[cell.name] = {arm: dict(n=len(v), p95_ms=p95(v) / 1e6, median_ms=sorted(v)[len(v) // 2] / 1e6)
                          for arm, v in values.items()}
    rss = {}
    for f in sorted((root / "runtime" / "rss").glob("*.json")):
        rss[f.stem] = json.loads(f.read_text())["max_rss_kib"]
    return dict(cells=out, rss_kib=rss)


def build(root):
    st = {m: verdict(root, m, "standard") for m in MODELS}
    la = {m: verdict(root, m, "ladder") for m in MODELS}
    rows, cands = identity_rows(root / "identity-s0.jsonl")
    shas = json.loads((root / "MODELS.json").read_text())
    # Raw identity proof (feature path, every dispatch tier) ran on A and C; seed 0's raw identity is C5's band check.
    order = [c["sha256"] for c in cands]
    raw = {m: (len(rows) == 620 and all(r["candidate_score_bits"][order.index(shas[m]["sha256"])] == HUNDRED for r in rows))
           if shas[m]["sha256"] in order else None for m in MODELS}
    served_lines = [line.split("\t") for line in (root / "served-identity.tsv").read_text().splitlines()]
    header, body = served_lines[0], served_lines[1:]
    col = {m: header.index(shas[m]["path"]) for m in MODELS}
    pixel = {m: len(body) == 62 and all(float(r[col[m]]) == 100.0 for r in body) for m in MODELS}
    nearid = {m: json.loads((root / "nearid-s0" / f"candidate-{m}.GATES.json").read_text()) for m in MODELS}
    steer = {m: json.loads((root / f"steer-s0-{m}.json").read_text())["rows"] for m in MODELS}
    inspect = json.loads((root / "inspect.json").read_text())
    probe = {line.split("\t")[0]: line.split("\t")[2] for line in (root / "head-probe.tsv").read_text().splitlines()[1:]}
    native = json.loads((root / "native-wasm-summary.json").read_text())
    rt = runtime(root)
    tests = {}
    for name in ("rev5-parity", "workspace-tests"):
        rc = root / f"{name}.rc"
        log = root / f"{name}.log"
        if rc.exists():
            text = log.read_text(errors="replace")
            failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED", text, re.M)))
            tests[name] = dict(rc=int(rc.read_text().strip()), failed=failed,
                               results=re.findall(r"test result: \w+\. \d+ passed; \d+ failed", text)[-3:])
    res = {"models": {m: shas[m]["sha256"] for m in MODELS}, "gates": {}}
    G = res["gates"]

    def row(name, bar, per, extra=None):
        G[name] = dict(bar=bar, **{m: per[m] for m in MODELS}, **(extra or {}))

    def ok(b):
        return "pass" if b else "fail"
    row("Table provenance (canonical inspector)", "qualified Rev5 provenance, 7 admitted tables, epoch 119",
        {m: dict(state=ok(inspect[m]["rc"] == 0 and inspect[m]["qualified_provenance"] and inspect[m]["admitted_tables"] == 7
                          and inspect[m]["checkpoint_epoch"] == "119" and inspect[m]["formula_revision"] == 5),
                 detail=f"{inspect[m]['feature_set_id']}, epoch {inspect[m]['checkpoint_epoch']}, "
                        f"{inspect[m]['admitted_tables']} tables") for m in MODELS})
    if native:
        seeds = {m: native["seeds"][i] for i, m in enumerate(MODELS)}
        row("Rust surface: pixel/cache/feature parity, 28 synthetic pairs × 10 native dispatch permutations + WASM128",
            "0 mismatches, finite, pixel identity exactly 100, no distortion above identity",
            {m: dict(state=ok(s["finite"] and s["pixel_cache_mismatches"] == 0 and s["consumed_feature_mismatches"] == 0
                              and s["cached_feature_mismatches"] == 0 and s["native_tier_row_mismatches"] == 0
                              and s["wasm_row_mismatches"] == 0 and s["wasm_finite"]
                              and s["pixel_identity_exact_100"] and s["no_distortion_above_identity"]),
                     detail=f"declared reads {s['declared_reads']}, caller width {s['caller_width']}, "
                            f"synthetic feature identity in [97.5,100]: {s['feature_identity_band_97_5_to_100']}")
             for m, s in seeds.items()})
    row("ZCTH v4 companion compatibility (attachment probe)", "head 568380f1 attaches",
        {m: dict(state="pass" if probe.get(shas[m]["path"]) == "ACCEPTED" else "fail", detail=probe.get(shas[m]["path"]))
         for m in MODELS})
    row("G-DIAL (standard 4,424)", "p5 ≤ 25, p95 ≥ 85, monotonicity ≥ .93",
        {m: dict(state=ok(st[m]["p5"] <= 25 and st[m]["p95"] >= 85 and st[m]["mono"] >= 0.93),
                 detail=f"p5 {st[m]['p5']:.2f}, p95 {st[m]['p95']:.2f}, mono {st[m]['mono']:.4f}") for m in MODELS})
    for cid, what in (("C1", "monotonicity ≥ .93"), ("C2", "ties ≤ .05"), ("C3", "an all-negative-truth probe row < 0"),
                      ("C4", "deepest probe < 0"), ("C6", "no grid cell above identity")):
        row(f"{cid} (standard / ladder)", what,
            {m: dict(state=ok(st[m]["checks"][cid][1] == "pass" and la[m]["checks"][cid][1] == "pass"),
                     detail=f"{st[m]['checks'][cid][0]:.4g} / {la[m]['checks'][cid][0]:.4g}") for m in MODELS})
    row("Negative tails (2,000-row probe)", "C3 and C4 pass; no dial clamp",
        {m: dict(state=ok(st[m]["checks"]["C3"][1] == "pass" and st[m]["checks"]["C4"][1] == "pass"),
                 detail=f"{st[m]['negtail']['frac_below_zero'] * 2000:.0f}/2000 below 0; min {st[m]['negtail']['min']:.3f}, "
                        f"p5 {st[m]['negtail']['p5']:.3f}") for m in MODELS})
    row("Identity, raw: feature inference on the 38 identity probes (C5)", "all 38 in [97.5, 100]",
        {m: dict(state=ok(st[m]["checks"]["C5"][1] == "pass"),
                 detail=f"{st[m]['checks']['C5'][0]:.0f} of 38 outside the band") for m in MODELS})
    row("Identity, raw: exact 100.0 through the feature path on every dispatch tier (E33 proof)",
        "38 probes + 24 NEARID sources × 10 tiers = 620 rows, exact bits",
        {m: dict(state="pass" if raw[m] else "fail" if raw[m] is False else "fail",
                 detail="620/620 exact" if raw[m] else "not run: seed 0 fails C5 (raw identity 92.2-97.5)" if raw[m] is None
                 else "not exact") for m in MODELS})
    row("Identity, served: BakeScorer::compute on identical pixels", "62 sources (38 probes + 24 NEARID) serve 100",
        {m: dict(state=ok(pixel[m]), detail="62/62 at 100.000000000 (printed to 9 decimals)" if pixel[m] else "not 100")
         for m in MODELS})
    row("Near-identity N1/N2/N3 (E33 registration; report)", "one-pixel ≥ 99, highest ≥ 99, ≥ 122/144 ladders",
        {m: dict(state=ok(nearid[m]["N1"]["pass"] and nearid[m]["N2"]["pass"] and nearid[m]["N3"]["pass"]),
                 detail=f"one-pixel min {nearid[m]['N1']['one_pixel_score_range'][0]:.3f}, ladders "
                        f"{nearid[m]['N3']['monotone_ladders']}/144") for m in MODELS})
    row("G-STEER (135 cases)", "release qualification: all 135 pass (M2 ≥ .99, M3f ≥ .70)",
        {m: dict(state=ok(sum(r["pass"] for r in steer[m]) == 135),
                 detail=f"{sum(r['pass'] for r in steer[m])}/135; failing "
                        + ", ".join(f"{r['key']} (b{r['blocks'][0]['bounds'][2]})" for r in steer[m] if not r["pass"]))
         for m in MODELS})
    if rt:
        arm = {"a": "e33_a", "c": "e33_c", "seed0": "by_v2fy_r5"}
        mp = {"a": "e33_a_map", "c": "e33_c_map", "seed0": "e33_seed0_map"}
        for cell, bar_ms, px in (("v4x-t1-1024x1024", 50, 1024 * 1024), ("v4x-t1-2048x2048", 200, 2048 * 2048)):
            c = rt["cells"][cell]
            d, fs, fm = c["zensim_D"]["p95_ms"], c["fast_ssim2"]["p95_ms"], c["fast_ssim2_main"]["p95_ms"]
            row(f"Scalar performance {cell.split('-')[-1]} (uncached complete, one worker, p95)",
                f"≤ {bar_ms} ms, ≤ 1.25× frozen D, ≤ fast-ssim2",
                {m: dict(state=ok(c[arm[m]]["p95_ms"] <= bar_ms and c[arm[m]]["p95_ms"] <= 1.25 * d
                                  and c[arm[m]]["p95_ms"] <= fs),
                         detail=f"{c[arm[m]]['p95_ms']:.2f} ms (D {d:.2f}, fast-ssim2 0.8.2 {fs:.2f}, main {fm:.2f})")
                 for m in MODELS})
            row(f"Spatial cost {cell.split('-')[-1]} (cached-reference score+map, p95)", "≤ 3× the uncached p95",
                {m: dict(state=ok(c[mp[m]]["p95_ms"] <= 3 * c[arm[m]]["p95_ms"]),
                         detail=f"{c[mp[m]]['p95_ms']:.2f} ms = {c[mp[m]]['p95_ms'] / c[arm[m]]['p95_ms']:.2f}× uncached")
                 for m in MODELS})
            bar_kib = (128 * px + 64 * 2**20) / 1024
            key = cell
            base = {m: rt["rss_kib"][f"{key}-e33_baseline_{m}"] for m in MODELS}
            inc = {m: (rt["rss_kib"][f"{key}-{arm[m]}"] - base[m], rt["rss_kib"][f"{key}-{mp[m]}"] - base[m]) for m in MODELS}
            row(f"Memory {cell.split('-')[-1]} (peak incremental RSS = fresh-process peak − baseline with inputs, model "
                "and scorer loaded)", f"≤ 128 B/pixel + 64 MiB = {bar_kib:,.0f} KiB",
                {m: dict(state=ok(max(inc[m]) <= bar_kib),
                         detail=f"score {inc[m][0]:,} KiB, score+map {inc[m][1]:,} KiB "
                                f"({inc[m][1] * 1024 / px:.0f} B/pixel incl. the 64 MiB allowance)")
                 for m in MODELS})
    res["tests"] = tests
    res["runtime"] = rt
    (root / "QUAL_A.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["| Gate | Bar | A | C | Seed 0 |", "|---|---|---|---|---|"]
    for name, g in G.items():
        lines.append(f"| {name} | {g['bar']} | " + " | ".join(f"**{g[m]['state'].upper()}** {g[m]['detail']}" for m in MODELS)
                     + " |")
    (root / "QUAL_A.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    build(p.parse_args().root)
