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
SDR25_RESP = "/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data"
SDR25_RECON = ("/mnt/v/output/zensim-multicodec-probe/"
               "sdr25_jnd_reconstructed_2026-07-02.parquet")
KONFIG_RAW = ("/mnt/v/dataset/konfig-iqa/KonFiG-IQA/DATA/EXP_III/data3.csv")
KONFIG_PAIRS = os.environ.get(
    "KONFIG_PAIRS", "/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv")

# ---------------------------------------------------------------------------
# DECLARED TARGET ORIENTATION per eval corpus.
#
# Added 2026-08-04 after the JPEG-AI-SDR25 determination (campaign Appendix I).
# The KADID bug taught "a table can be backwards"; SDR25 taught the harder
# lesson: **"backwards" is only meaningful relative to a DECLARED convention.**
#
# `ext_sdr25.parquet` stores `human_score = q_jnd`, a JND *distance* from the
# pristine original (original pinned at 0, rising with distortion). That is not
# a defect — it is the honest native unit of a triplet-comparison JND study, and
# the seed-selection oracle consumes |SROCC|, so it works correctly today. It is
# a LANDMINE only for a future training leg, which would fit the model to
# anti-correlate with quality — precisely the KADID failure.
#
# So this gate does NOT ask "is the sign positive?". It asks "does the table
# match the orientation it declares?" A `distortion` corpus reading
# distortion-oriented is OK. A corpus reading the OPPOSITE of its declaration is
# INVERTED. That makes the convention a machine-checkable fact rather than tribal
# knowledge, and it forces anyone adding a training leg to confront the sign.
#
# MEASURED across the 188 board fullevals (`rank.<corpus>.srocc_signed`), which
# is what makes these declarations empirical rather than assumed:
#   distortion-oriented : aic4 188/188 neg, sdr25 171/171 neg, konjnd 187/188 neg
#   quality-oriented    : cid22/aic3/imazen26/nonphoto 0 neg, csiq/live/tid 1-2 neg
#   kadid               : 78 neg / 110 pos — the Appendix F inversion, now fixed
# The three distortion-oriented corpora are exactly the JND/threshold-scaled
# ones. Orientation tracks the LABEL FAMILY (JND distance vs MOS), not the corpus.
# ---------------------------------------------------------------------------
QUALITY, DISTORTION = "quality", "distortion"
EXPECTED_ORIENTATION = {
    "kadid": QUALITY,      # (dmos-1)/4 over a MOS-in-disguise; Appendix F
    "tid": QUALITY,        # published MOS
    "csiq": QUALITY,       # 1 - DMOS
    "live": QUALITY,       # 1 - dmos_new/100
    "cid22": QUALITY,      # MCOS/100
    "aic3": QUALITY,       # stored quality-oriented (188/188 positive)
    "imazen26": QUALITY,   # score_ssim2
    "nonphoto": QUALITY,   # score_ssim2
    "hfnlproxy": QUALITY,  # score_ssim2 band
    "sdr25": DISTORTION,   # q_jnd — JND distance from the original
    "aic4": DISTORTION,    # q_jnd, same reconstruction family as sdr25
    "konjnd": DISTORTION,  # PJND threshold; freeze_check already takes |SROCC|
    # konfig (added 2026-08-05, campaign Appendix L §L.5): the LABEL FAMILY is a
    # JND design grid (distortion family), but the STORED target is the already-
    # converted `human_score = 1 - q_jnd/3.2` — quality-oriented BY TRANSFORM.
    # The aic4/sdr25 lesson applied at build time: the declaration names what the
    # table STORES, and the keyed checker verifies it against the 75,519 raw
    # EXP_III DCR votes (degradation-oriented, rises with level in 70/70 ladders).
    "konfig": QUALITY,
}

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
     {"kadid": "ext_kadid.parquet", "tid": "ext_tid.parquet",
      "sdr25": "ext_sdr25.parquet"}),
    ("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01",
     {"kadid": "ext_kadid.parquet", "tid": "ext_tid.parquet",
      "sdr25": "ext_sdr25.parquet", "konfig": "konfig_944.parquet"}),
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


def sdr25_ground_truth_keyed(hs: np.ndarray):
    """RAW-VOTE ground truth for JPEG-AI-SDR25, keyed on `human_score` (= q_jnd).

    Positional joins are unsafe here (the table carries only `ref_basename` +
    `human_score`, 10 rows per ref), so we key on the q_jnd value, which is
    unique per stimulus and matches the reconstruction parquet exactly — VERIFIED
    50/50 with zero misses, codec id 6 = JPEG-AI, dlevels 1..10 x 5 refs.

    The ground truth itself is the RAW crowd votes, never the reconstructed
    scale: for each stimulus, the fraction of head-to-head appearances in which
    workers named it the MORE DISTORTED side (the response reading the
    reconstruction script verified against the traps — under a 'closer' reading
    383 of 386 workers fail them). Trap and bias triplets are excluded: traps
    pair the pristine original against a heavy distortion and bias triplets show
    the same stimulus twice, so neither carries stimulus-ordering signal.

    Returned as a QUALITY-oriented quantity (`-distortion_rate`) so the caller's
    sign convention is uniform across corpora. This is a crude scale — each
    stimulus meets different opponents, so the rate is not an interval measure —
    but a SIGN test needs only the ordering direction, and it is the rawest
    human signal the corpus can produce.
    """
    if not os.path.isdir(SDR25_RESP) or not os.path.exists(SDR25_RECON):
        raise FileNotFoundError("SDR25 raw responses or reconstruction absent")
    rec = pq.read_table(SDR25_RECON).to_pydict()
    by_q = {round(q, 9): (rec["img_num"][i], str(rec["codec"][i]), rec["dlevel"][i])
            for i, q in enumerate(rec["q_jnd"])}

    seen = collections.defaultdict(int)   # stimulus -> appearances
    named = collections.defaultdict(int)  # stimulus -> times named more-distorted
    n_rows = 0
    for method in ("BTC", "PTC"):
        f = f"{SDR25_RESP}/JPEG_AIC_SDR_{method}_JPEG_AI_responses_2025.02.28_v1.csv"
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if r["response"] not in ("left", "right"):
                continue
            if r.get("is_trap") == "1" or r.get("is_bias") == "1":
                continue
            n_rows += 1
            img = int(r["img_num"])
            for side in ("left", "right"):
                s = (img, r[f"codec_{side}"], int(r[f"dlevel_{side}"]))
                if s[1] == "0" and s[2] == 0:
                    continue  # the pristine original is not a stimulus
                seen[s] += 1
                if r["response"] == side:
                    named[s] += 1

    gt, missing = [], 0
    for h in hs:
        s = by_q.get(round(float(h), 9))
        if s is None or seen.get(s, 0) == 0:
            gt.append(np.nan)
            missing += 1
        else:
            gt.append(-(named[s] / seen[s]))  # quality-oriented
    if missing:
        raise ValueError(f"{missing} of {len(hs)} rows had no raw-vote ground truth")
    note = (f"raw more-distorted vote rate over {n_rows} BTC+PTC triplet responses "
            f"(traps/bias excluded), negated to quality orientation")
    return np.asarray(gt, float), note


def konfig_ground_truth_keyed(hs: np.ndarray):
    """RAW-VOTE ground truth for KonFiG-IQA (campaign Appendix L §L.5).

    Ground truth = per-stimulus mean of the 75,519 raw EXP_III DCR votes
    (`data3.csv`, keyed Source x Distortion x Level; Answer in [0,4], a
    degradation scale — verified rising with level in 70/70 ladders, and the
    distribution's own `scores.csv` aggregation re-derives from it 910/910 with
    zero mismatches). Negated to quality orientation (sdr25 convention).

    Join: `human_score` is NOT unique per stimulus (37 distinct values over
    1,090 rows), so per-row stimulus identity comes from the build's pairs TSV
    (KONFIG_PAIRS, env-overridable) — valid because `v2_ab_extract` preserves
    input row order and the canonical-corpus promote preserves CSV order. The
    join is VALIDATED per row: the TSV's human_score must equal the table's
    exactly (all KonFiG targets are exact binary fractions: 1 - 5k/64 for
    PartA, 1 - k/32 for PartB, so byte-level round-trips are lossless).

    EXP_III covers PartA only (910 stimuli); PartB rows get NaN and are dropped
    by the caller with the count reported — the registered "850 of 1,090
    externally checked" scope (Appendix L §L.5, limitation L.11.5).
    """
    if not os.path.exists(KONFIG_RAW):
        raise FileNotFoundError(f"KonFiG raw votes absent: {KONFIG_RAW}")
    if not os.path.exists(KONFIG_PAIRS):
        raise FileNotFoundError(
            f"KonFiG pairs TSV absent: {KONFIG_PAIRS} (set KONFIG_PAIRS)")
    acc = collections.defaultdict(list)
    with open(KONFIG_RAW, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["Source"], r["Distortion Type"], int(r["Distortion Level"]))
            acc[key].append(float(r["Answer"]))
    n_votes = sum(len(v) for v in acc.values())
    mean_dcr = {k: sum(v) / len(v) for k, v in acc.items()}
    rows = list(csv.DictReader(open(KONFIG_PAIRS), delimiter="\t"))
    if len(rows) != len(hs):
        raise ValueError(
            f"pairs TSV rows {len(rows)} != table rows {len(hs)}; "
            f"positional identity broken")
    gt = []
    for i, r in enumerate(rows):
        if abs(float(r["human_score"]) - float(hs[i])) > 0.0:
            raise ValueError(
                f"row {i}: pairs human_score {r['human_score']} != table "
                f"{hs[i]}; positional identity broken")
        # EXP_III keys are PartA stimuli. A PartB row at level<=12 would FALSELY
        # match PartA's motionblur key (same level index, DIFFERENT stimulus:
        # q = level*0.1 vs level*0.25) — so PartB is excluded by PART, not by
        # key-miss. Caught on the first gate run: n came back 950, not 850.
        if r["part"] != "PartA":
            gt.append(np.nan)
            continue
        key = (r["source"], r["distortion"], int(r["level"]))
        m = mean_dcr.get(key)
        gt.append(np.nan if m is None else -m)  # quality-oriented
    note = (f"raw EXP_III DCR vote mean over {n_votes} ratings (PartA stimuli), "
            f"negated to quality orientation; PartB rows have no DCR and are "
            f"excluded")
    return np.asarray(gt, float), note


GROUND_TRUTH = {"kadid": kadid_ground_truth, "tid": tid_ground_truth}
# Corpora whose ground truth must be joined on a key rather than row position.
KEYED_GROUND_TRUTH = {"sdr25": sdr25_ground_truth_keyed,
                      "konfig": konfig_ground_truth_keyed}

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
    "konfig":         ("human", "JND design grid calibrated by boosted triplet "
                                "comparisons (Men 2021, 1.05M responses); "
                                "orientation cross-checked vs 75,519 raw EXP_III "
                                "DCR votes (Appendix L)"),
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
    for c in list(GROUND_TRUTH) + list(KEYED_GROUND_TRUTH):
        if re.search(rf"(^|[_/]){c}([_.]|$)", b):
            return c
    return None


def check(path: str, corpus: str | None = None) -> dict:
    corpus = corpus or guess_corpus(path)
    expect = EXPECTED_ORIENTATION.get(corpus, QUALITY)
    out = {"path": path, "corpus": corpus, "expected_orientation": expect}
    if corpus not in GROUND_TRUTH and corpus not in KEYED_GROUND_TRUTH:
        out.update(verdict="SKIPPED", reason="no recoverable ground truth for this corpus")
        return out
    hs = np.asarray(pq.read_table(path, columns=["human_score"])["human_score"].to_pylist(), float)
    if corpus in KEYED_GROUND_TRUTH:
        try:
            gt, note = KEYED_GROUND_TRUTH[corpus](hs)
        except (FileNotFoundError, ValueError, KeyError) as e:
            out.update(verdict="SKIPPED", reason=f"keyed ground truth unavailable: {e}")
            return out
    else:
        gt, note, n_gt = GROUND_TRUTH[corpus]()
        if len(hs) != n_gt:
            out.update(verdict="SKIPPED",
                       reason=f"row count {len(hs)} != ground truth {n_gt}; positional join unsafe")
            return out
    # Keyed ground truths may mark rows with no external signal as NaN (e.g.
    # konfig's PartB rows, which EXP_III does not cover). Drop them from the
    # sign test and report the checked subset size — a NaN row is out-of-scope,
    # never silently counted.
    gt = np.asarray(gt, float)
    keep = ~np.isnan(gt)
    n_dropped = int((~keep).sum())
    s = _signed_srocc(hs[keep], gt[keep])
    # `gt` is always quality-oriented, so s > 0 means the TABLE is quality-oriented.
    measured = QUALITY if s > 0 else DISTORTION
    out.update(verdict="OK" if measured == expect else "INVERTED",
               signed_srocc=round(s, 6), measured_orientation=measured,
               n=int(keep.sum()), ground_truth=note)
    if n_dropped:
        out["n_no_ground_truth"] = n_dropped
    if measured == DISTORTION:
        out["training_warning"] = (
            "target is DISTORTION-oriented: negate before any training use, or the "
            "model learns to anti-correlate with quality (campaign Appendix F/I)")
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
            if r["verdict"] == "SKIPPED":
                extra = r["reason"]
            else:
                extra = (f"signed SROCC {r['signed_srocc']:+.6f} vs {r['ground_truth']} "
                         f"(n={r['n']}); measured {r['measured_orientation']}, "
                         f"declared {r['expected_orientation']}")
            print(f"{mark}  {r.get('corpus') or '?':6s}  {r['path']}\n          {extra}")
            if r.get("training_warning"):
                print(f"          NOTE: {r['training_warning']}")
    bad = [r for r in results if r["verdict"] == "INVERTED"]
    if bad:
        print(f"\nFAIL — {len(bad)} table(s) store human_score in the OPPOSITE orientation "
              f"to the one declared in EXPECTED_ORIENTATION. Either the builder's transform "
              f"is backwards (fix the builder; do NOT flip it at read time) or the "
              f"declaration is wrong (fix the declaration, with evidence).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
