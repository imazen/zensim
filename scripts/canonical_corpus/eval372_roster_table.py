#!/usr/bin/env python3
"""Stored-vs-current era table for the 372-class roster.

READS the numbers `bake_verdict` produced (`rank.<corpus>.srocc_signed`,
`z_rmse`, the `dial` block, `composite`). Computes NO statistic — per the
one-owner rule, every IQA number here came out of `zensim_validate::panel`.

Emits: a per-(bake, corpus) TSV, a markdown table, and the ORDERING-FLIP list —
pairs of bakes whose relative order on a corpus reverses between the two eras,
which is the decision-relevant output.

Usage: eval372_roster_table.py [ROSTER_DIR] [OUT_STEM]
"""
import json
import os
import sys
from itertools import combinations

DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/eval372-roster-2026-08-30"
STEM = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DIR, "roster_era_table")

# report order: the era-affected corpora first, then the measured-identical ones
CORPORA = ["cid22", "konjnd", "kon504", "kadid", "tid", "aic3",
           "csiq", "live", "pipal", "aic4", "nonphoto", "imazen26",
           "hf_nearlossless", "hfnlproxy", "sdr25"]
# corpora whose table is BYTE-IDENTICAL across the two roots (copied or measured
# bit-identical) -> any delta there would be a bug in this harness, not an era shift
IDENTICAL = {"csiq", "live", "pipal", "aic4", "nonphoto", "imazen26",
             "hf_nearlossless", "hfnlproxy", "sdr25"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    labels = sorted({os.path.basename(p).rsplit("_", 1)[0]
                     for p in os.listdir(os.path.join(DIR, "json")) if p.endswith(".json")})
    data = {}
    for lab in labels:
        old = load(os.path.join(DIR, "json", f"{lab}_old.json"))
        new = load(os.path.join(DIR, "json", f"{lab}_new.json"))
        k_old = load(os.path.join(DIR, "kon504", f"{lab}_old.json"))
        k_new = load(os.path.join(DIR, "kon504", f"{lab}_new.json"))
        if not (old and new):
            print(f"SKIP {lab}: missing verdict", file=sys.stderr)
            continue
        rec = {"n_inputs": new.get("n_inputs"), "composite": (old.get("composite"), new.get("composite")),
               "dial": (old.get("dial"), new.get("dial")), "corpora": {}}
        for c in CORPORA:
            if c == "kon504":
                a = (k_old or {}).get("rank", {}).get("konjnd")
                b = (k_new or {}).get("rank", {}).get("konjnd")
            else:
                a = old.get("rank", {}).get(c)
                b = new.get("rank", {}).get(c)
            if not a or not b:
                continue
            rec["corpora"][c] = {
                "n": (a.get("n"), b.get("n")),
                "srocc": (a.get("srocc_signed"), b.get("srocc_signed")),
                "z_rmse": (a.get("z_rmse"), b.get("z_rmse")),
                "per_ref": (a.get("per_ref_mean"), b.get("per_ref_mean")),
                "train_eq_val": b.get("train_eq_val"),
            }
        data[lab] = rec

    with open(STEM + ".tsv", "w") as f:
        f.write("bake\tn_inputs\tcorpus\tn_old\tn_new\tsrocc_stored\tsrocc_current\td_srocc\t"
                "zrmse_stored\tzrmse_current\td_zrmse\tperref_stored\tperref_current\ttrain_eq_val\n")
        for lab, rec in data.items():
            for c, v in rec["corpora"].items():
                so, sn = v["srocc"]
                zo, zn = v["z_rmse"]
                po, pn = v["per_ref"]
                f.write(f"{lab}\t{rec['n_inputs']}\t{c}\t{v['n'][0]}\t{v['n'][1]}\t"
                        f"{so:.5f}\t{sn:.5f}\t{sn - so:+.5f}\t{zo:.5f}\t{zn:.5f}\t{zn - zo:+.5f}\t"
                        f"{'' if po is None else f'{po:.5f}'}\t{'' if pn is None else f'{pn:.5f}'}\t"
                        f"{v['train_eq_val']}\n")

    # ---- ordering flips: relative order of two bakes reverses between eras ----
    flips = []
    for c in CORPORA:
        for a, b in combinations([l for l in data if c in data[l]["corpora"]], 2):
            sa = data[a]["corpora"][c]["srocc"]
            sb = data[b]["corpora"][c]["srocc"]
            # compare on the reported (signed) value; JND corpora are negative by
            # construction, so compare |.| there to keep "better" = larger.
            fa = (abs(sa[0]), abs(sa[1]))
            fb = (abs(sb[0]), abs(sb[1]))
            old_cmp = fa[0] - fb[0]
            new_cmp = fa[1] - fb[1]
            if old_cmp * new_cmp < 0:
                flips.append((c, a, b, fa[0], fb[0], fa[1], fb[1]))

    lines = []
    lines.append("| bake | n_in | " + " | ".join(CORPORA) + " |")
    lines.append("|---|---:|" + "---:|" * len(CORPORA))
    for lab, rec in data.items():
        cells = []
        for c in CORPORA:
            v = rec["corpora"].get(c)
            if not v:
                cells.append("—")
                continue
            so, sn = v["srocc"]
            cells.append(f"{so:+.4f} → **{sn:+.4f}** ({sn - so:+.4f})" if abs(sn - so) > 5e-5
                         else f"{sn:+.4f} (=)")
        lines.append(f"| {lab} | {rec['n_inputs']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"## ordering flips ({len(flips)})")
    if flips:
        lines.append("| corpus | A | B | \\|A\\| stored | \\|B\\| stored | \\|A\\| current | \\|B\\| current |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for c, a, b, ao, bo, an, bn in flips:
            lines.append(f"| {c} | {a} | {b} | {ao:.4f} | {bo:.4f} | {an:.4f} | {bn:.4f} |")
    else:
        lines.append("none")
    with open(STEM + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # harness self-check: an identical-bytes corpus must not move
    bad = [(lab, c) for lab, rec in data.items() for c, v in rec["corpora"].items()
           if c in IDENTICAL and abs(v["srocc"][1] - v["srocc"][0]) > 1e-12]
    print(f"wrote {STEM}.tsv / .md — {len(data)} bakes, {len(flips)} ordering flips")
    if bad:
        print(f"WARNING: {len(bad)} identical-bytes corpora moved (harness bug?): {bad[:5]}")
    else:
        print("self-check OK: every byte-identical corpus reports an exactly zero delta")


if __name__ == "__main__":
    main()
