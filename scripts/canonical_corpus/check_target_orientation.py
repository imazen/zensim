#!/usr/bin/env python3
"""Corpus TARGET-ORIENTATION gate — assert a feature table's `human_score` points the
same way as the corpus's ground-truth human labels, BEFORE it is used to train or eval.

**Why this exists.** On 2026-08-04 the ext-lineage KADID tables
(`ext720`/`ext924`/`ext944` `ext_kadid.parquet`) were found to store
`human_score = (5 − dmos)/4` — the exact inverse of the canonical `(dmos − 1)/4`.
`scripts/canonical_corpus/build_fr_corpus_pairs.py:build_kadid()` had applied the
standard invert-a-DMOS reflex (correct for CSIQ's `1 − DMOS` and LIVE's
`1 − dmos_new/100`) to a column that is a **MOS in disguise**: KADID's `dmos` FALLS with
severity (raw crowdsourced DCR 4.0789 → 2.0072 across levels 1–5, 349,800 ratings), so
it was already quality-oriented. Nothing caught it for six weeks. The cost: every KADID
number in the SOTA-944 campaign was published as an unsigned magnitude of a sign-flipped
quantity, 110 of 188 board bakes trained/scored anti-correlated with KADID's real human
MOS, and a registered gate (`KADID ≥ 0.70`) was passed by the three most-inverted arms
and failed by the only correctly-oriented one. Full determination:
`benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F.

**The gate.** For each corpus with a recoverable ground truth, assert
`sign(SROCC(table.human_score, ground_truth_quality)) > 0`. This is deliberately a
SIGN test, not an equality test: it does not care which normalization a builder chose,
only that the table is not backwards. Run it at build time and record the verdict in the
dir `_MANIFEST.json` (`target_orientation`), so "is this table oriented correctly?" is a
grep, not a forensic audit.

Ground truths are the RAW human labels wherever they exist, never a derived column:
  kadid  — mean DCR per distorted image from `raw_crowdsource_data.csv` (349,800 ratings)
  tid    — published MOS from `mos_with_names.txt` (quality-oriented)
  csiq   — DMOS from the corpus xlsx (distortion-oriented; expected transform 1 − DMOS)
  live   — realigned `dmos_new` (distortion-oriented; expected 1 − dmos_new/100)
Corpora with no recoverable raw ground truth (safesyn, bigcodec, cid22_train, kadis …)
are reported SKIPPED — a skip is "not checked", never "passed".

Usage:
    check_target_orientation.py <parquet> [--corpus kadid] [--json]
    check_target_orientation.py --all-roots          # sweep every known eval root
Exit 0 = every checked table correctly oriented; exit 1 = at least one INVERTED;
exit 2 = usage/IO error.  Statistics come from `zenstats` via `scripts/lib/zen_stats`
(no stat math is implemented here, per the no-duplication rule).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.zen_stats import panel  # noqa: E402

KADID_RAW = "/mnt/v/dataset/kadid10k/raw_crowdsource_data.csv"
KADID_DMOS = "/mnt/v/dataset/kadid10k/dmos.csv"
TID_MOS = "/mnt/v/dataset/tid2013/mos_with_names.txt"

# Known eval roots for --all-roots. (root, {corpus: filename})
KNOWN_ROOTS = [
    ("/mnt/v/zen/zensim-training/2026-05-15-full-features",
     {"kadid": "kadid_features_372col_2026-05-15.parquet",
      "tid": "tid_features_372col_2026-05-15.parquet"}),
    ("/mnt/v/zen/zensim-training/canonical-2026-05-21/train",
     {"kadid": "kadid.parquet", "tid": "tid.parquet"}),
    ("/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22",
     {"kadid": "ext_kadid.parquet", "tid": "ext_tid.parquet"}),
    ("/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27",
     {"kadid": "ext_kadid.parquet", "tid": "ext_tid.parquet"}),
    ("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01",
     {"kadid": "ext_kadid.parquet", "tid": "ext_tid.parquet"}),
]


def _signed_srocc(x, y) -> float:
    """|SROCC| from zenstats, signed by the rank-covariance direction.

    `zen_stats.panel` returns |SROCC| (the project convention). The SIGN is recovered
    from the covariance of the midranks — a direction, not a second statistic."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = panel(list(x), list(y))["srocc"]

    def midrank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        # average ties
        s = np.sort(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[np.isin(v, s[i])] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    cov = np.cov(midrank(x), midrank(y))[0, 1]
    return float(mag) * (1.0 if cov >= 0 else -1.0)


def kadid_ground_truth():
    """Mean raw DCR per distorted image, in `dmos.csv` row order. Quality-oriented."""
    acc = collections.defaultdict(list)
    src = KADID_RAW
    if os.path.exists(src):
        with open(src, newline="") as f:
            for r in csv.DictReader(f):
                u = r.get("dist_url") or ""
                if not u.startswith("kon10k_png/"):
                    continue
                try:
                    acc[os.path.basename(u)].append(float(r["dcr"]))
                except (TypeError, ValueError):
                    continue
    rows = list(csv.DictReader(open(KADID_DMOS)))
    if acc:
        gt = np.array([float(np.mean(acc[r["dist_img"]])) for r in rows])
        note = f"raw crowdsourced DCR, {sum(len(v) for v in acc.values())} ratings"
    else:  # raw file absent — fall back to published DMOS (itself quality-oriented)
        gt = np.array([float(r["dmos"]) for r in rows])
        note = "published dmos.csv (raw ratings file absent)"
    return gt, note, len(rows)


def tid_ground_truth():
    mos, n = [], 0
    for line in open(TID_MOS):
        p = line.split()
        if len(p) == 2:
            mos.append(float(p[0]))
            n += 1
    return np.array(mos), "published TID MOS (mos_with_names.txt)", n


GROUND_TRUTH = {"kadid": kadid_ground_truth, "tid": tid_ground_truth}

# ---------------------------------------------------------------------------
# TARGET PROVENANCE of every leg in the SOTA-944 training mix, and therefore
# WHICH legs an external orientation check can reach at all.
#
# This map exists because the 2026-08-04 data-integrity audit (campaign Appendix
# G) established a structural fact that had never been written down: of the 11
# groups in the incumbent mix, only TWO carry human labels. The other nine carry
# a METRIC-derived target (ssim2, or a teacher model's prediction), so there is
# no human ground truth to point them at, and `sign(SROCC(target, humans)) > 0`
# is not a question that can be asked of them.
#
# The consequence is worth stating plainly: KADID's six-week-old inversion was
# found because KADID is one of the only two legs where an external check was
# ever possible. The other nine are not known-good; they are UNCHECKED, and no
# amount of running this gate will change that. Internal consistency (a target
# falling monotonically along a known quality ladder) is the only handle those
# legs have — see `check_table_integrity.py` and Appendix G check A4.
# ---------------------------------------------------------------------------
MIX_TARGET_PROVENANCE = {
    "kadid":          ("human", "mean DCR over 349,800 raw crowdsourced ratings"),
    "tid":            ("human", "published TID2013 MOS"),
    "safesyn":        ("metric", "ssim2-anchored synthetic sweep"),
    "cid22_train":    ("metric", "ssim2_gpu; CID22 human MOS is VALIDATION-ONLY and "
                                 "is never a training target (zensim/CLAUDE.md)"),
    "bigcodec":       ("metric", "ssim2-anchored multi-codec sweep"),
    "kadis":          ("metric", "score_ssim2_gpu over the KADIS-700k distortion grid"),
    "konjnd_bpg":     ("metric", "gpu_ssimulacra2/100 over the KonJND BPG ladder; the "
                                 "corpus's human PJND is NOT carried into the table"),
    "konjnd_bpg_val": ("metric", "same as konjnd_bpg (validation split)"),
    "tsafesyn":       ("teacher", "teacher model forward over the safesyn rows"),
    "ttbig":          ("teacher", "teacher model forward over the bigcodec rows"),
    "tkadis":         ("teacher", "teacher model forward over the kadis rows"),
}


def provenance_report() -> list[dict]:
    """Which mix legs an external orientation check can reach, and which it cannot."""
    out = []
    for g, (kind, note) in sorted(MIX_TARGET_PROVENANCE.items()):
        out.append({
            "group": g, "target_kind": kind, "note": note,
            "externally_checkable": kind == "human",
            "verdict": "CHECKABLE" if kind == "human" else "NOT-CHECKABLE",
        })
    return out


def guess_corpus(path: str) -> str | None:
    b = os.path.basename(path).lower()
    for c in GROUND_TRUTH:
        if re.search(rf"(^|[_/]){c}([_.]|$)", b):
            return c
    return None


def check(path: str, corpus: str | None = None) -> dict:
    corpus = corpus or guess_corpus(path)
    out = {"path": path, "corpus": corpus}
    if corpus not in GROUND_TRUTH:
        out.update(verdict="SKIPPED", reason="no recoverable ground truth for this corpus")
        return out
    gt, note, n_gt = GROUND_TRUTH[corpus]()
    hs = np.asarray(pq.read_table(path, columns=["human_score"])["human_score"].to_pylist(), float)
    if len(hs) != n_gt:
        out.update(verdict="SKIPPED",
                   reason=f"row count {len(hs)} != ground truth {n_gt}; positional join unsafe")
        return out
    s = _signed_srocc(hs, gt)
    out.update(verdict="OK" if s > 0 else "INVERTED", signed_srocc=round(s, 6),
               n=len(hs), ground_truth=note)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", nargs="?")
    ap.add_argument("--corpus")
    ap.add_argument("--all-roots", action="store_true")
    ap.add_argument("--provenance", action="store_true",
                    help="report which SOTA-944 mix legs an external orientation check "
                         "can reach at all (2 of 11) and which carry metric/teacher "
                         "targets with no human ground truth to check against")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.provenance:
        rep = provenance_report()
        if a.json:
            print(json.dumps(rep, indent=2))
        else:
            n_ok = sum(r["externally_checkable"] for r in rep)
            print(f"SOTA-944 mix target provenance — {n_ok} of {len(rep)} legs carry "
                  f"human labels and can be orientation-checked externally:\n")
            for r in rep:
                mark = "CHECKABLE   " if r["externally_checkable"] else "NOT-CHECKABLE"
                print(f"  {mark}  {r['group']:16s} [{r['target_kind']:7s}] {r['note']}")
            print("\nA NOT-CHECKABLE leg is UNCHECKED, not known-good. Internal "
                  "consistency (check_table_integrity.py, Appendix G A4) is the only\n"
                  "handle those legs have.")
        return 0
    results = []
    if a.all_roots:
        for root, files in KNOWN_ROOTS:
            for corpus, fn in files.items():
                p = os.path.join(root, fn)
                if os.path.exists(p):
                    results.append(check(p, corpus))
    elif a.parquet:
        results.append(check(a.parquet, a.corpus))
    else:
        ap.error("give a parquet path or --all-roots")
        return 2
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            mark = {"OK": "OK      ", "INVERTED": "INVERTED", "SKIPPED": "SKIPPED "}[r["verdict"]]
            extra = (f"signed SROCC {r['signed_srocc']:+.6f} vs {r['ground_truth']} (n={r['n']})"
                     if r["verdict"] != "SKIPPED" else r["reason"])
            print(f"{mark}  {r.get('corpus') or '?':6s}  {r['path']}\n          {extra}")
    bad = [r for r in results if r["verdict"] == "INVERTED"]
    if bad:
        print(f"\nFAIL — {len(bad)} table(s) store human_score BACKWARDS vs the corpus's "
              f"human labels. Fix the builder's transform; do NOT flip it at read time.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
