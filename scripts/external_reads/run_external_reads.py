#!/usr/bin/env python3
"""Canonical seven-domain external-read runner (decision-surface audit gap 3).

The 2026-07-28/29 external-validation wave (UPIQ-HDR cross-route, SI-HDR,
HDR-VDC, AVT-VQDB-UHD-1-HDR, CHUG, Rousselot HDdtb/4Kdtb, plus the
BANDVIS/CSFW LOO instruments) left each study's analysis runner only in its
`/mnt/v/output/zensim/<study>/` artifact dir. This is the COMMITTED, named
runner the freeze plan's Phase 4 requires ("re-run the external read set at
the final bakes"): it rescores the STORED feature tables + labels — minutes,
no video decode — under a chosen scorer, and checks the recorded numbers.

Modes / scorers
---------------
  --from-stored            (default and only compute mode) rescore stored
                           tables. Scorer via --scorer:
    probe944  (default)    the registered UPIQ-SDR 944 ridge probe — the
                           instrument every 2026-07-29 study used. Re-fit
                           deterministically from the stored UPIQ tables and
                           GATED against the recorded hdr-dmean head
                           (q3_heads["944"]) to 5e-7 BEFORE any study look,
                           exactly as the registered protocols did.
    s228                   the stored `score228` column (the streamed
                           production score at extraction build) — zero-fit.
    bake:<path.bin>        a ZNPR bake forwarded over the stored 944-feature
                           tables via `predict_features_with_bake` (the
                           feature-cache fast path). THIS is the Phase-4
                           final-bake mode: point it at the final V1 bakes.
                           The bake must accept the 944 feature contract —
                           the stored tables are 944-regime v2 extractions;
                           there is deliberately no first-N column slicing
                           (a 372-input bake expects v1 features, which
                           these are not).
  --reads a,b,...          subset of: upiq sihdr hdrvdc avt chug rousselot
                           loo944 loo956 (default: all). loo944/loo956 are
                           VERIFY reads: they recompute the LOO delta tables
                           from the stored per-drop bake_verdict JSONs and
                           check them against the stored deltas files +
                           report the registered headline Σ (no scorer).
  --check-recorded         (default on) compare against the recorded values
                           embedded below (provenance: each study's
                           results.json); exit 1 on any mismatch > --tol.
                           Only reads with a recorded value are checked —
                           bake:<path> reads have none (they are the new
                           measurement).
  --tol 5e-7               the studies' own registered gate tolerance.
  --json OUT.json          machine-readable dump of every read.
  --list-extract           print the full re-extraction pointers per study
                           (COMMANDS.md + committed extractor examples) and
                           exit. Re-extraction is hours of video decode; it
                           is DOCUMENTED, not automated — the stored tables
                           above are the canonical inputs (honesty over
                           automation theater).

Stats provenance: every correlation is the canonical Rust `panel` binary via
`scripts/lib/zen_stats.panel_batch` (tie-correct midrank Spearman; one batch
process per group of reads). SROCC values are reported SIGNED (the studies
recorded signed scipy values; all recorded reads are positive). The ridge
probe is the studies' registered instrument machinery (sklearn Ridge +
GroupKFold), reproduced verbatim — it is an instrument, not a shipped model.

Data dependencies (stored tables; see README.md for sha256s + Tower mirrors):
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/       UPIQ SDR/HDR 944|956 CSVs
  /mnt/v/output/zensim/sihdr-transfer-2026-07-29/  sihdr_feats_944.csv
  /mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/hdrvdc_pooled_944.csv
  /mnt/v/output/zensim/avthdr-validation-2026-07-29/{avthdr_pooled_944.csv,
      chug_feats_frame_level/, chug_sample.tsv}
  /mnt/v/output/zensim/rousselot-chroma-2026-07-29/{*_feats_944_k179.csv,
      pairs_manifest.json}
  /mnt/v/output/zensim/bandvis-loo-2026-07-28/{verdicts944/,loo944_deltas.json}
  /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/{verdicts956/,loo956_deltas.json}
  /mnt/v/datasets/{upiq,si-hdr,hdr-vdc,avt-vqdb-uhd-1-hdr,chug}/  labels

Registered-run reproduction (2026-07-31, --scorer probe944 + --scorer s228):
every recorded value below reproduced to <= the 5e-7 gate from stored tables,
including Korshunov 0.9346 / Narwaria 0.7688 (hdr-dmean q3_heads.944) and the
AVT pooled probe 0.7742.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from scripts.lib import zen_stats  # noqa: E402

B = "/mnt/v/output/zensim"
DMEAN = f"{B}/hdr-dmean-2026-07-29"
SIHDR = f"{B}/sihdr-transfer-2026-07-29"
HDRVDC = f"{B}/hdrvdc-conditions-2026-07-29"
AVTHDR = f"{B}/avthdr-validation-2026-07-29"
ROUSSELOT = f"{B}/rousselot-chroma-2026-07-29"
LOO944 = f"{B}/bandvis-loo-2026-07-28"
LOO956 = f"{B}/csfw-g6-loo-2026-07-29"

# ----------------------------------------------------------------------
# Recorded values (provenance: each study's results.json in its artifact
# dir; extraction build commits in the asrun/*/PROVENANCE.txt files).
# These are the numbers the freeze plan cites; --check-recorded gates
# reproduction to --tol (5e-7, the studies' own registered bound).
# ----------------------------------------------------------------------
RECORDED = {
    # hdr-dmean results.json q3_heads["944"] — the registered UPIQ-SDR 944
    # ridge probe head (this IS the probe gate every study reproduced).
    "probe_gate": {
        "n_cols_kept": 689,
        "lambda": 100.0,
        "sdr_cv_srocc": 0.9362861433052736,
        "hdr_pooled": 0.7596713929714486,
        "hdr_narwaria": 0.7688482648531633,
        "hdr_korshunov": 0.9346082397263838,
    },
    # hdr-dmean results.json q1_readout — score228 zero-fit readout.
    ("upiq", "s228"): {
        "pooled": 0.7145258696572109,
        "narwaria": 0.7144868907305767,
        "korshunov": 0.9456101668431741,
    },
    # sihdr results.json l1 — zero-shot on the 324 labeled pairs.
    ("sihdr", "probe944"): {"pooled": 0.4208035318668324},
    ("sihdr", "s228"): {"pooled": 0.34395207726257887},
    # hdrvdc results.json q2.pooled — condition-matched legs i/ii/iii.
    ("hdrvdc", "probe944"): {
        "leg_i": 0.6694842305956337,
        "leg_ii": 0.6516551623391339,
        "leg_iii": 0.7461711717122695,
    },
    ("hdrvdc", "s228"): {
        "leg_i": 0.7140507785627813,
        "leg_ii": 0.7161823118462308,
        "leg_iii": 0.8114144105401488,
    },
    # avthdr results.json q1 — pooled + per-codec.
    ("avt", "probe944"): {
        "pooled": 0.7741882112891073,
        "av1": 0.7553020244940146,
        "hevc": 0.8409858776149138,
        "vvc": 0.7071091723202965,
    },
    ("avt", "s228"): {
        "pooled": 0.7245392929042379,
        "av1": 0.6873497668628157,
        "hevc": 0.7994972701314031,
        "vvc": 0.6482817815234772,
    },
    # avthdr chug_results.json — the ONE registered CHUG look.
    ("chug", "probe944"): {"pooled": 0.7244987166524072},
    ("chug", "s228"): {"pooled": 0.7524714719052432},
    # rousselot results.json comparators — score228 zero-fit only (the
    # study's L1/L2 numbers are LOSO nested-CV instrument outputs; those
    # live in the as-run copy, not in this rescorer).
    ("rousselot", "s228"): {
        "hddtb_all": 0.8840898687154805,
        "k4dtb_all": 0.8282047550707016,
        "hddtb_chromapure": 0.8353504117820824,
    },
    # LOO instruments carry no embedded constants: the verify reads
    # recompute the delta tables from the stored verdict JSONs and take
    # the registered headline Σ from the stored deltas file itself
    # (append2 Σ -0.0687 PASS <= 0; csfw Σ +0.0608 G6 FAIL by design —
    # summary.txt in each artifact dir).
}

LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]


# ----------------------------------------------------------------------
# Stat plumbing — ALL correlations go through one panel --batch process
# per call site. Values are SIGNED midrank Spearman (srocc_signed).
# ----------------------------------------------------------------------
def sroccs_signed(jobs) -> list[float]:
    """jobs: [(label, x, y)] -> signed SROCC per job, one batch process."""
    rows = zen_stats.panel_batch(jobs, stats="srocc")
    return [r["srocc_signed"] for r in rows]


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_feat_csv(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        return header, list(rdr)


def load_upiq():
    """The shared UPIQ side: SDR train table + HDR eval table."""
    hsdr, rsdr = load_feat_csv(f"{DMEAN}/upiq_sdr_956.csv")
    h944, r944 = load_feat_csv(f"{DMEAN}/upiq_hdr_944.csv")
    assert len(rsdr) == 3779 and len(r944) == 380, (len(rsdr), len(r944))
    off = 5  # condition_id,dataset,is_hdr,jod,score228
    d = {
        "cid_s": [r[0] for r in rsdr],
        "jod_s": np.array([float(r[3]) for r in rsdr]),
        "s228_s": np.array([float(r[4]) for r in rsdr]),
        "X_s": np.array([[float(v) for v in r[off:]] for r in rsdr])[:, :944],
        "ds_h": np.array([r[1] for r in r944]),
        "jod_h": np.array([float(r[3]) for r in r944]),
        "s228_h": np.array([float(r[4]) for r in r944]),
        "X_h": np.array([[float(v) for v in r[off:]] for r in r944]),
    }
    assert d["X_h"].shape[1] == 944
    d["nar"] = d["ds_h"] == "narwaria"
    d["kor"] = d["ds_h"] == "korshunov"
    assert d["nar"].sum() == 140 and d["kor"].sum() == 240
    return d


def upiq_groups(cid_s):
    """Merged content groups on the UPIQ SDR side (verbatim protocol
    machinery from the studies — tid<->live repeated-content union)."""
    cond_meta = {}
    with open("/mnt/v/datasets/upiq/upiq_subjective_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            cond_meta[row["condition_id"]] = row
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for c in cid_s:
        m = cond_meta[c]
        key = f"{m['dataset']}:{m['content_id']}"
        find(key)
        rep = m["repeating_content_id"].strip()
        if rep and rep != "-":
            other_ds = "live" if rep.startswith("l-") else "tid2013"
            other_ct = ("l-i" + rep[2:]) if rep.startswith("l-") else ("t-i" + rep[2:])
            union(key, f"{other_ds}:{other_ct}")
    return np.array([find(f"{cond_meta[c]['dataset']}:{cond_meta[c]['content_id']}")
                     for c in cid_s])


# ----------------------------------------------------------------------
# The registered probe (studies' instrument machinery, reproduced
# verbatim; gated vs the recorded head before any study look).
# ----------------------------------------------------------------------
def fit_probe944(u, tol):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    groups = upiq_groups(u["cid_s"])
    keep = u["X_s"].std(axis=0) > 1e-12
    Xtr = u["X_s"][:, keep]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Ztr = (Xtr - mu) / sd
    y = u["jod_s"]
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Ztr, y, groups))
    # All 25 (lambda, fold) held-out sroccs in ONE batch process.
    jobs = []
    for lam in LAMBDAS:
        for fi, (tr, te) in enumerate(folds):
            m = Ridge(alpha=lam)
            m.fit(Ztr[tr], y[tr])
            jobs.append((f"cv_{lam}_{fi}", m.predict(Ztr[te]), y[te]))
    vals = sroccs_signed(jobs)
    cv = {}
    for li, lam in enumerate(LAMBDAS):
        cv[lam] = float(np.mean(vals[li * 5:(li + 1) * 5]))
    lam = max(cv, key=cv.get)
    probe = Ridge(alpha=lam)
    probe.fit(Ztr, y)

    pred_h = probe.predict((u["X_h"][:, keep] - mu) / sd)
    ph, pn, pk = sroccs_signed([
        ("hdr_pooled", pred_h, u["jod_h"]),
        ("hdr_narwaria", pred_h[u["nar"]], u["jod_h"][u["nar"]]),
        ("hdr_korshunov", pred_h[u["kor"]], u["jod_h"][u["kor"]]),
    ])
    gate = {
        "n_cols_kept": int(keep.sum()), "lambda": lam,
        "sdr_cv_srocc": cv[lam],
        "hdr_pooled": ph, "hdr_narwaria": pn, "hdr_korshunov": pk,
    }
    rec = RECORDED["probe_gate"]
    for k, want in rec.items():
        got = gate[k]
        ok = (got == want) if isinstance(want, int) else abs(got - want) < tol
        assert ok, f"PROBE GATE FAIL {k}: got {got!r} want {want!r}"
    print("probe gate PASS (recorded hdr-dmean q3_heads.944 reproduced "
          f"to <{tol:g}): kor={pk:.4f} nar={pn:.4f} pooled={ph:.4f} "
          f"cv={cv[lam]:.4f} lam={lam:g} kept={int(keep.sum())}")

    def predict(X944: np.ndarray) -> np.ndarray:
        return probe.predict((X944[:, keep] - mu) / sd)

    return predict, gate


# ----------------------------------------------------------------------
# Bake scorer — predict_features_with_bake (the feature-cache fast path).
# ----------------------------------------------------------------------
def make_bake_scorer(bake_path: str):
    bin_path = os.path.join(_REPO, "target/release/predict_features_with_bake")
    if not os.path.exists(bin_path):
        sys.exit("build first: cargo build --release -p zensim-validate "
                 "--bin predict_features_with_bake")

    def predict(X944: np.ndarray) -> np.ndarray:
        n_rows, n_feat = X944.shape
        with tempfile.NamedTemporaryFile(suffix=".blob", delete=False) as f:
            tmp = f.name
            f.write(struct.pack("<II", n_feat, n_rows))
            f.write(X944.astype("<f4").tobytes())
        try:
            out = subprocess.run(
                [bin_path, "--bake", bake_path, "--features-file", tmp],
                capture_output=True, text=True, check=True, timeout=600)
        except subprocess.CalledProcessError as e:
            sys.exit(f"predict_features_with_bake failed (does the bake "
                     f"accept {n_feat} features?): {e.stderr.strip()}")
        finally:
            os.unlink(tmp)
        preds = np.array([float(v) for v in out.stdout.split()])
        assert len(preds) == n_rows, (len(preds), n_rows)
        return preds

    return predict


# ----------------------------------------------------------------------
# Per-study stored-table evaluators. Each returns [(read, value, n)].
# ----------------------------------------------------------------------
def eval_upiq(scorer_name, score_fn, u):
    if scorer_name == "probe944":
        # The probe gate already IS the UPIQ read (fit on SDR, one
        # registered HDR look) — reported there; nothing to re-run.
        return []
    if scorer_name == "s228":
        preds = u["s228_h"]
    else:
        preds = score_fn(u["X_h"])
    v = sroccs_signed([
        ("pooled", preds, u["jod_h"]),
        ("narwaria", preds[u["nar"]], u["jod_h"][u["nar"]]),
        ("korshunov", preds[u["kor"]], u["jod_h"][u["kor"]]),
    ])
    return [("pooled", v[0], 380), ("narwaria", v[1], 140), ("korshunov", v[2], 240)]


def eval_sihdr(scorer_name, score_fn):
    h, rows = load_feat_csv(f"{SIHDR}/sihdr_feats_944.csv")
    off = h.index("f0")
    assert h[off:] == [f"f{k}" for k in range(944)]
    cid = [r[0] for r in rows]
    s228 = np.array([float(r[h.index("score228")]) for r in rows])
    X = np.array([[float(v) for v in r[off:]] for r in rows])
    lab = {}
    with open("/mnt/v/datasets/si-hdr/experiment_results/experiment_results.csv",
              newline="") as f:
        for row in csv.DictReader(f):
            if row["scene"] == "all" or row["method"] in ("input", "original"):
                continue
            lab[(row["image"], row["clip_level"], row["method"])] = float(row["jod"])
    assert len(lab) == 324, len(lab)
    idx, jod = [], []
    for i, c in enumerate(cid):
        key = tuple(c.split("-"))
        if key in lab:
            idx.append(i)
            jod.append(lab[key])
    idx, jod = np.array(idx), np.array(jod)
    assert len(idx) == 324
    preds = s228[idx] if scorer_name == "s228" else score_fn(X[idx])
    (v,) = sroccs_signed([("pooled", preds, jod)])
    return [("pooled", v, len(idx))]


HDRVDC_LEGS = {
    "i": {c: "A" for c in (("bright", "near"), ("bright", "far"),
                           ("dim", "near"), ("dim", "far"))},
    "ii": {("bright", "near"): "B", ("bright", "far"): "B",
           ("dim", "near"): "C", ("dim", "far"): "C"},
    "iii": {("bright", "near"): "B", ("bright", "far"): "D",
            ("dim", "near"): "C", ("dim", "far"): "E"},
}


def eval_hdrvdc(scorer_name, score_fn):
    h, rows = load_feat_csv(f"{HDRVDC}/hdrvdc_pooled_944.csv")
    assert h[:4] == ["cid", "vid", "config", "score228_mean"]
    keys = [(r[0], r[1], r[2]) for r in rows]
    X = np.array([[float(v) for v in r[4:]] for r in rows])
    s228 = np.array([float(r[3]) for r in rows])
    score_by_key = dict(zip(keys, s228 if scorer_name == "s228" else score_fn(X)))
    lab = list(csv.DictReader(open("/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv")))
    lrows = []
    for r in lab:
        if r["is_reference"] == "True":
            continue
        lrows.append({"cid": r["content_id"],
                      "vid": f"{r['crf']}_{r['resolution']}",
                      "cond": (r["luminance_level"], r["viewing_distance"]),
                      "jod": float(r["jod"])})
    assert len(lrows) == 464, len(lrows)
    jod = np.array([r["jod"] for r in lrows])
    jobs = []
    for leg, cfg_of in HDRVDC_LEGS.items():
        preds = np.array([score_by_key[(r["cid"], r["vid"], cfg_of[r["cond"]])]
                          for r in lrows])
        jobs.append((f"leg_{leg}", preds, jod))
    vals = sroccs_signed(jobs)
    return [(jobs[i][0], vals[i], 464) for i in range(len(jobs))]


AVT_PAT = re.compile(r"^(\d+)_(\d+)_(\w+)_(av1|hevc|vvc)_(.+)\.mkv$")


def eval_avt(scorer_name, score_fn):
    h, rows = load_feat_csv(f"{AVTHDR}/avthdr_pooled_944.csv")
    assert h[:3] == ["content", "vid", "score228_mean"]
    keys = [(r[0], r[1]) for r in rows]
    X = np.array([[float(v) for v in r[3:]] for r in rows])
    s228 = np.array([float(r[2]) for r in rows])
    score_by_key = dict(zip(keys, s228 if scorer_name == "s228" else score_fn(X)))
    lrows = []
    labels = "/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv"
    for r in csv.DictReader(open(labels)):
        m = AVT_PAT.match(r["stimuli_file"])
        if not m:
            continue
        w, hh, br, codec, content = m.groups()
        lrows.append({"content": content, "vid": f"{codec}_{w}_{hh}_{br}",
                      "codec": codec, "mos": float(r["mos"])})
    assert len(lrows) == 195, len(lrows)
    mos = np.array([r["mos"] for r in lrows])
    preds = np.array([score_by_key[(r["content"], r["vid"])] for r in lrows])
    codecs = np.array([r["codec"] for r in lrows])
    jobs = [("pooled", preds, mos)]
    for c in ("av1", "hevc", "vvc"):
        m = codecs == c
        jobs.append((c, preds[m], mos[m]))
    vals = sroccs_signed(jobs)
    return [(jobs[i][0], vals[i], len(jobs[i][1])) for i in range(len(jobs))]


def eval_chug(scorer_name, score_fn):
    frame_feats = defaultdict(list)
    frame_s228 = defaultdict(list)
    for path in sorted(glob.glob(f"{AVTHDR}/chug_feats_frame_level/chug_batch_*.csv")):
        with open(path, newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr)
            assert header[:4] == ["key", "dim", "peak_nits", "score228"]
            for rec in rdr:
                content, rung, fj, _cfg = rec[0].split("|")
                frame_feats[(content, rung)].append(
                    (int(fj[1:]), np.array([float(v) for v in rec[4:]])))
                frame_s228[(content, rung)].append((int(fj[1:]), float(rec[3])))
    pooled_X, pooled_s228 = {}, {}
    for k, lst in frame_feats.items():
        lst.sort()
        assert len(lst) == 8, (k, len(lst))
        pooled_X[k] = np.mean([v for _, v in lst], axis=0)
        pooled_s228[k] = float(np.mean([s for _, s in sorted(frame_s228[k])]))
    mosj = {}
    for r in csv.DictReader(open("/mnt/v/datasets/chug/chug.csv")):
        if r["ref"] == "0":
            mosj[(r["content_name"], r["bitladder"])] = float(r["mos_j"])
    sample = list(csv.DictReader(open(f"{AVTHDR}/chug_sample.tsv"), delimiter="\t"))
    keys = [(s["content"], s["rung"]) for s in sample if (s["content"], s["rung"]) in pooled_X]
    assert len(keys) == 300, len(keys)
    mos = np.array([mosj[k] for k in keys])
    if scorer_name == "s228":
        preds = np.array([pooled_s228[k] for k in keys])
    else:
        preds = score_fn(np.stack([pooled_X[k] for k in keys]))
    (v,) = sroccs_signed([("pooled", preds, mos)])
    return [("pooled", v, len(keys))]


def eval_rousselot(scorer_name, score_fn):
    manifest = json.load(open(f"{ROUSSELOT}/pairs_manifest.json"))
    meta = {}
    for row in manifest:
        stem = os.path.splitext(os.path.basename(row["dist"]))[0]
        meta[(row["dataset"], stem)] = row
    out = []
    jobs = []
    metas = []
    for ds, fname in (("hddtb", "hddtb_feats_944_k179.csv"),
                      ("4kdtb", "k4dtb_feats_944_k179.csv")):
        h, rows = load_feat_csv(f"{ROUSSELOT}/{fname}")
        off = h.index("f0")
        ids = [r[0] for r in rows]
        X = np.array([[float(v) for v in r[off:]] for r in rows])
        s228 = np.array([float(r[h.index("score228")]) for r in rows])
        mos = np.array([meta[(ds, i)]["mos"] for i in ids])
        fam = np.array([meta[(ds, i)]["family"] for i in ids])
        assert len(rows) == 96
        preds = s228 if scorer_name == "s228" else score_fn(X)
        tag = "hddtb" if ds == "hddtb" else "k4dtb"
        jobs.append((f"{tag}_all", preds, mos))
        metas.append((f"{tag}_all", len(rows)))
        if ds == "hddtb":
            cp = np.isin(fam, ["cnoise", "gamut"])
            assert cp.sum() == 40
            jobs.append(("hddtb_chromapure", preds[cp], mos[cp]))
            metas.append(("hddtb_chromapure", int(cp.sum())))
    vals = sroccs_signed(jobs)
    for (label, n), v in zip(metas, vals):
        out.append((label, v, n))
    return out


# ----------------------------------------------------------------------
# LOO verify reads — recompute the delta tables from the STORED per-drop
# bake_verdict JSONs and check them against the stored deltas files.
# (Re-running the instruments end-to-end = extraction + twin fit + 12+
# verdicts, hours; see --list-extract and the asrun/ harness copies.)
# ----------------------------------------------------------------------
def _verdict_sroccs(path):
    d = json.load(open(path))
    return {c["display"]: float(c["srocc"]) for c in d["corpora"]}


def verify_loo(base, full_name, deltas_file, headline_fam, headline_key):
    stored = json.load(open(f"{base}/{deltas_file}"))
    vdir = f"{base}/verdicts{'944' if '944' in deltas_file else '956'}"
    full = _verdict_sroccs(f"{vdir}/{full_name}.json")
    max_diff = 0.0
    n_checked = 0
    for fam, dmap in stored["deltas"].items():
        drop = _verdict_sroccs(f"{vdir}/drop_{fam}.json")
        for corpus, want in dmap.items():
            got = drop[corpus] - full[corpus]
            max_diff = max(max_diff, abs(got - want))
            n_checked += 1
    sigma_subset = [c for c in stored["sigma_subset"] if c in full]
    sig_max = 0.0
    for fam, want in stored["sigma"].items():
        drop = _verdict_sroccs(f"{vdir}/drop_{fam}.json")
        got = sum(drop[c] - full[c] for c in sigma_subset)
        sig_max = max(sig_max, abs(got - want))
    headline = stored["sigma"][headline_fam]
    return {
        "n_delta_cells_checked": n_checked,
        "max_delta_recompute_diff": max_diff,
        "max_sigma_recompute_diff": sig_max,
        "headline": {headline_key: headline},
        "consistent": max_diff < 1e-12 and sig_max < 1e-12,
    }


# ----------------------------------------------------------------------
# Full re-extraction pointers (documented, not automated).
# ----------------------------------------------------------------------
EXTRACT_DOCS = """\
Full re-extraction (hours of decode; per-study exact commands + shas live in
each artifact dir's COMMANDS.md; extractor examples are committed):

  upiq / hdr-dmean : {dmean}/COMMANDS.md
      zensim/examples/upiq_features_extract.rs (+ upiq_hdr924_score,
      hdr_sdr_consistency); build c4632d62.
  sihdr            : {sihdr}/COMMANDS.md
      zensim/examples/sihdr_features_extract.rs; build 34cbd9cf;
      asrun/sihdr/run_extraction.sh is the batch driver.
  hdrvdc           : {hdrvdc}/COMMANDS.md
      zensim/examples/hdrvdc_features_extract.rs; build 6b3505a5;
      asrun/hdrvdc/driver.py is the decode driver.
  avt + chug       : {avthdr}/COMMANDS.md
      same hdrvdc_features_extract example (generic); build 1f0f92d5;
      asrun/avthdr/{{driver,chug_driver,chug_scope}}.py.
  rousselot        : {rousselot}/COMMANDS.md
      zensim/examples/rousselot_features_extract.rs; build 73734d88.
  loo944 (BANDVIS) : {loo944}/COMMANDS.md
      v2_ab_extract (ZENSIM_AB_MODE=foldapp2) -> twin fit -> 13 verdicts;
      build b1d4bc25; harness = asrun/bandvis_loo_944/. Instrument parquet:
      /mnt/v/zen/zensim-training/ext944-instrument-2026-07-28/.
  loo956 (CSFW G6) : {loo956}/COMMANDS.md
      v2_ab_extract (ZENSIM_AB_MODE=foldcsfw) -> twin fit -> verdicts;
      build 7bfd511d; harness = asrun/csfw_g6_loo_956/. Instrument parquet:
      /mnt/v/zen/zensim-training/ext956-instrument-2026-07-29/.
""".format(dmean=DMEAN, sihdr=SIHDR, hdrvdc=HDRVDC, avthdr=AVTHDR,
           rousselot=ROUSSELOT, loo944=LOO944, loo956=LOO956)

ALL_READS = ["upiq", "sihdr", "hdrvdc", "avt", "chug", "rousselot",
             "loo944", "loo956"]


def main():
    ap = argparse.ArgumentParser(
        description="canonical seven-domain external-read runner (stored tables)")
    ap.add_argument("--from-stored", action="store_true", default=True,
                    help="rescore stored feature tables (the default and only "
                         "compute mode; full re-extraction is documented via "
                         "--list-extract)")
    ap.add_argument("--scorer", default="probe944",
                    help="probe944 | s228 | bake:<path.bin>")
    ap.add_argument("--reads", default=",".join(ALL_READS))
    ap.add_argument("--check-recorded", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--tol", type=float, default=5e-7,
                    help="recorded-value gate (the studies' registered bound)")
    ap.add_argument("--json", default=None, help="write machine-readable results")
    ap.add_argument("--list-extract", action="store_true")
    a = ap.parse_args()

    if a.list_extract:
        print(EXTRACT_DOCS)
        return 0

    reads = [r.strip() for r in a.reads.split(",") if r.strip()]
    bad = [r for r in reads if r not in ALL_READS]
    if bad:
        sys.exit(f"unknown reads {bad}; choose from {ALL_READS}")

    scorer = a.scorer
    out = {"scorer": scorer, "tol": a.tol, "reads": {}}
    failures = []

    needs_scorer = [r for r in reads if not r.startswith("loo")]
    score_fn = None
    if needs_scorer:
        if scorer == "probe944":
            u = load_upiq()
            score_fn, gate = fit_probe944(u, a.tol)
            out["probe_gate"] = gate
        elif scorer == "s228":
            u = load_upiq() if "upiq" in reads else None
        elif scorer.startswith("bake:"):
            u = load_upiq() if "upiq" in reads else None
            score_fn = make_bake_scorer(scorer.split(":", 1)[1])
        else:
            sys.exit(f"unknown scorer {scorer!r}")

    scorer_key = "probe944" if scorer == "probe944" else (
        "s228" if scorer == "s228" else "bake")

    for read in reads:
        if read == "loo944":
            res = verify_loo(LOO944, "lin944", "loo944_deltas.json",
                             "append2", "sigma_append2")
            out["reads"][read] = res
            ok = res["consistent"] and res["headline"]["sigma_append2"] <= 0
            print(f"loo944  verify: {res['n_delta_cells_checked']} delta cells "
                  f"recomputed from stored verdicts, max diff "
                  f"{res['max_delta_recompute_diff']:.2e}; "
                  f"append2 Σ = {res['headline']['sigma_append2']:+.4f} "
                  f"(registered PASS bar <= 0) -> {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append("loo944 verify")
            continue
        if read == "loo956":
            res = verify_loo(LOO956, "lin956", "loo956_deltas.json",
                             "csfw", "sigma_csfw")
            out["reads"][read] = res
            ok = res["consistent"]
            print(f"loo956  verify: {res['n_delta_cells_checked']} delta cells "
                  f"recomputed from stored verdicts, max diff "
                  f"{res['max_delta_recompute_diff']:.2e}; "
                  f"csfw Σ = {res['headline']['sigma_csfw']:+.4f} "
                  f"(registered G6 FAIL -> lanes default-OFF, as designed) "
                  f"-> {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append("loo956 verify")
            continue

        ev = {"upiq": lambda: eval_upiq(scorer_key, score_fn, u),
              "sihdr": lambda: eval_sihdr(scorer_key, score_fn),
              "hdrvdc": lambda: eval_hdrvdc(scorer_key, score_fn),
              "avt": lambda: eval_avt(scorer_key, score_fn),
              "chug": lambda: eval_chug(scorer_key, score_fn),
              "rousselot": lambda: eval_rousselot(scorer_key, score_fn)}[read]
        rows = ev()
        rec = RECORDED.get((read, scorer_key), {})
        out["reads"][read] = {}
        for label, val, n in rows:
            want = rec.get(label)
            entry = {"srocc_signed": val, "srocc_abs": abs(val), "n": n}
            status = ""
            if want is not None and a.check_recorded:
                diff = abs(val - want)
                entry["recorded"] = want
                entry["recorded_diff"] = diff
                if diff < a.tol:
                    status = f"  == recorded {want:.4f} (diff {diff:.1e}) OK"
                else:
                    status = f"  != recorded {want:.4f} (diff {diff:.3e}) FAIL"
                    failures.append(f"{read}/{label}")
            out["reads"][read][label] = entry
            print(f"{read:9s} {label:18s} SROCC {val:+.4f} (n={n}){status}")

    if scorer_key == "probe944" and "upiq" in reads and a.check_recorded:
        # The probe gate already asserted the UPIQ read (incl. Korshunov +
        # Narwaria); surface it in the read table for the record.
        g = out["probe_gate"]
        print(f"upiq      korshunov (gate)   SROCC {g['hdr_korshunov']:+.4f} "
              f"(n=240)  == recorded 0.9346 OK")
        print(f"upiq      narwaria  (gate)   SROCC {g['hdr_narwaria']:+.4f} "
              f"(n=140)  == recorded 0.7688 OK")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")

    if failures:
        print(f"\nFAIL: {len(failures)} read(s) diverged: {failures}")
        return 1
    print("\nall requested reads OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
