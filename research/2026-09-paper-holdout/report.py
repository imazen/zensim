#!/usr/bin/env python3
"""paper-holdout lane: build the committed record from stats.json (canonical
panel via zenstats) and the bootstrap owner's text outputs. Parses; computes
nothing statistical.

Writes benchmarks/paper_holdout_2026-09-23.{json,md} in the repo (small).
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
A = os.environ.get("OUT", "/var/tmp/paper-holdout/a")
CORPORA = ["cid22a", "csiq", "aic3", "aic4", "aic4full", "sdr25", "konjnd"]
ARMS = ["ssim2", "B", "GMSD", "DVtalk", "DVours", "DVgate"]
LABEL = {"ssim2": "fast-ssim2 (SSIMULACRA2)", "B": "zensim B (frozen)",
         "GMSD": "GMSD", "DVtalk": "DVIFM-ish talk-faithful-luma",
         "DVours": "DVIFM-ish ours-full-luma", "DVgate": "DVIFM-ish serving-gate-ycbcr3"}
CNAME = {"cid22a": "CID22-A(25) human (MCOS)", "csiq": "CSIQ",
         "aic3": "AIC-3 CTC (EPFL, full resolution as distributed)",
         "aic4": "AIC-4 sample, 620x800 PTC crops", "aic4full": "AIC-4 sample, full resolution",
         "sdr25": "JPEG-AI SDR25", "konjnd": "KonJND-1k JPEG-504"}


def parse_boot(path):
    out = {"pooled": {}, "pooled_ci": {}, "pooled_delta": {}, "within": {},
           "within_ci": {}, "within_delta": {}, "header": None, "skipped_within": False}
    if not os.path.exists(path):
        return None
    mode = None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("# corpus="):
            out["header"] = line
        if "within-image axis SKIPPED" in line:
            out["skipped_within"] = True
        m = re.match(r"^(\S+)\tper_ref_mean=([-\d.]+)\tmedian=([-\d.]+)\tn=(\d+)$", line)
        if m:
            out["within"][m.group(1)] = dict(mean=float(m.group(2)), median=float(m.group(3)),
                                            refs=int(m.group(4)))
            continue
        if line == "arm\tpooled_srocc":
            mode = "pooled"; continue
        if line == "arm\tpooled\tCI95_lo\tCI95_hi":
            mode = "pooled_ci"; continue
        if line == "arm\tmean\tCI95_lo\tCI95_hi":
            mode = "within_ci"; continue
        if line.startswith("candidate\tpooled\t"):
            mode = "pooled_delta"; continue
        if line.startswith("candidate\tmean\t"):
            mode = "within_delta"; continue
        if not line or line.startswith("#"):
            if line.startswith("#"):
                mode = None
            continue
        p = line.split("\t")
        if mode == "pooled" and len(p) == 2:
            out["pooled"][p[0]] = float(p[1])
        elif mode in ("pooled_ci", "within_ci") and len(p) == 4:
            out[mode][p[0]] = dict(point=float(p[1]), lo=float(p[2]), hi=float(p[3]))
        elif mode in ("pooled_delta", "within_delta") and len(p) == 7:
            out[mode][p[0]] = dict(cand=float(p[1]), ref=float(p[2]), delta=float(p[3]),
                                   lo=float(p[4]), hi=float(p[5]), p_win=float(p[6]))
    return out


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def ci(d):
    return "—" if not d else f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def dl(d):
    if not d:
        return "—"
    tag = " ✓" if d["lo"] > 0 else (" ✗" if d["hi"] < 0 else "")
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{tag}"


def main():
    st = json.load(open(f"{A}/stats.json"))
    boots = {c: {ref: parse_boot(f"{A}/boot/{c}_vs_{ref}.txt") for ref in ("ssim2", "B")}
             for c in CORPORA}
    ro, re_ = f"{A}/boot/regression_csiq_orig.txt", f"{A}/boot/regression_csiq_edited.txt"
    reg = (open(ro).read() == open(re_).read()) if os.path.exists(ro) and os.path.exists(re_) else None
    record = dict(
        schema="paper-holdout-v1", date="2026-09-23",
        purpose="zensim-2026 paper peer scoring: GMSD and DVIFM-ish on the held-out human table",
        exposure=dict(path=f"{A}/EXPOSURE.json",
                      ledger="docs/DATASET_HISTORY.md + docs/DATA_SPLITS.md, 2026-09-23"),
        binaries=open(f"{A}/BINARIES.sha256").read().splitlines(),
        bootstrap=dict(owner="benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py",
                       resamples=10000, seed=20260901, unit="reference (cluster)",
                       regression_owner_edit_csiq_boot2000_byte_identical=reg),
        checks=st["checks"], corpora={})
    for c in CORPORA:
        s = st["stats"][c]
        b = boots[c]["ssim2"] or {}
        bb = boots[c]["B"] or {}
        arms = {}
        for a, v in s["arms"].items():
            e = dict(v)
            if a in ARMS:
                e["pooled_ci"] = (b.get("pooled_ci") or {}).get(a)
                e["within_boot"] = (b.get("within_ci") or {}).get(a)
                e["delta_vs_ssim2"] = dict(pooled=(b.get("pooled_delta") or {}).get(a),
                                           within=(b.get("within_delta") or {}).get(a))
                e["delta_vs_B"] = dict(pooled=(bb.get("pooled_delta") or {}).get(a),
                                       within=(bb.get("within_delta") or {}).get(a))
            arms[a] = e
        record["corpora"][c] = dict(name=CNAME[c], n=s["n"], refs=s["refs"],
                                    target_orientation=s["target_orientation"],
                                    expected_sign=s["expected_sign"], arms=arms,
                                    boot_header=b.get("header"))
    jp = f"{REPO}/benchmarks/paper_holdout_2026-09-23.json"
    # Committed JSON must stay under 30 KB: the full record (with per-arm
    # signed/PLCC/KROCC detail) goes to block storage; the committed copy keeps
    # headline pooled/within stats and paired deltas only.
    json.dump(record, open(f"{A}/paper_holdout_2026-09-23.full.json", "w"), indent=1)
    slim = dict(record)
    slim["full_record"] = f"{A}/paper_holdout_2026-09-23.full.json"
    slim["corpora"] = {}
    for c, r in record["corpora"].items():
        arms = {}
        for a, e in r["arms"].items():
            arms[a] = {k: v for k, v in e.items()
                       if k in ("srocc", "pooled_ci", "within_boot",
                                "delta_vs_ssim2", "delta_vs_B")}
        slim["corpora"][c] = dict(r, arms=arms)
    json.dump(slim, open(jp, "w"), separators=(",", ":"), allow_nan=False)
    # ---------------------------------------------------------------- md
    L = []
    L.append("## Table H1 — pooled |SROCC| with reference-clustered 95% CI (10,000 draws, seed 20260901)\n")
    L.append("| corpus (n / refs) | " + " | ".join(LABEL[a] for a in ARMS) + " |")
    L.append("|---" * (len(ARMS) + 1) + "|")
    for c in CORPORA:
        r = record["corpora"][c]
        cells = []
        for a in ARMS:
            e = r["arms"].get(a)
            cells.append("not scored" if e is None else ci(e.get("pooled_ci")) if e.get("pooled_ci")
                         else f4(e["srocc"]))
        L.append(f"| {r['name']} ({r['n']} / {r['refs']}) | " + " | ".join(cells) + " |")
    L.append("\n## Table H2 — within-image (mean per-reference |SROCC|), reference-clustered 95% CI\n")
    L.append("| corpus | " + " | ".join(LABEL[a] for a in ARMS) + " |")
    L.append("|---" * (len(ARMS) + 1) + "|")
    for c in CORPORA:
        r = record["corpora"][c]
        if c == "konjnd":
            L.append(f"| {r['name']} | " + " | ".join(["one row per reference"] * len(ARMS)) + " |")
            continue
        cells = []
        for a in ARMS:
            e = r["arms"].get(a)
            cells.append("not scored" if e is None else ci(e.get("within_boot")))
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    for ref, key in (("fast-ssim2", "delta_vs_ssim2"), ("B", "delta_vs_B")):
        L.append(f"\n## Table H3{'a' if ref == 'fast-ssim2' else 'b'} — paired Δ vs {ref} "
                 "(candidate − reference; ✓ CI above 0, ✗ CI below 0)\n")
        cand = [a for a in ARMS if a != ("ssim2" if ref == "fast-ssim2" else "B")]
        L.append("| corpus | axis | " + " | ".join(LABEL[a] for a in cand) + " |")
        L.append("|---" * (len(cand) + 2) + "|")
        for c in CORPORA:
            r = record["corpora"][c]
            for ax in ("pooled", "within"):
                if c == "konjnd" and ax == "within":
                    continue
                cells = []
                for a in cand:
                    e = r["arms"].get(a)
                    cells.append("not scored" if e is None else dl((e.get(key) or {}).get(ax)))
                L.append(f"| {c} | {ax} | " + " | ".join(cells) + " |")
    L.append("\n## Table H4 — full panel (canonical zenstats panel): |SROCC| / signed SROCC / PLCC / KROCC\n")
    L.append("| corpus | arm | |SROCC| | signed | PLCC | KROCC | source |")
    L.append("|---|---|---|---|---|---|---|")
    for c in CORPORA:
        r = record["corpora"][c]
        for a, e in r["arms"].items():
            L.append(f"| {c} | {LABEL.get(a, a)} | {e['srocc']:.4f} | {e['srocc_signed']:+.4f} | "
                     f"{e['plcc']:.4f} | {e['krocc']:.4f} | `{os.path.basename(str(e['source']))}` |")
    mp = f"{A}/tables_md.txt"
    open(mp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {jp} and {mp}")


if __name__ == "__main__":
    main()
