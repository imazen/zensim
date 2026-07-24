#!/usr/bin/env python3
"""Train + rigorously evaluate the structural-corruption DETECTOR (2nd head).

The head is 372-FEATURE (v1 f0..f371 — a subset of the 720 already extracted at
deployment, so zero extra cost) so it can compose directly with the breakthrough
**negrich** severe-honest hard negatives (266k rows, 372-feat, provenance-gapped →
cannot be re-extracted at 720). Classes:
  - POSITIVES: structural corruptions from build_corruption_corpus.py (many
    imazen-26 sources × the codec-corpus catalog) — the MISSING multi-source
    positives (negrich's companion was single-source gb82).
  - HARD negatives: negrich = severe-but-HONEST KADIS degradations (heavy blur/
    noise/motion/color) that look corruption-like in feature space. THE boundary.
  - EASY negatives: matched honest q10/q20 anchors (same sources) + a broad sample
    of the existing honest 720 corpora (span q0..100, diverse content).

SOURCE-HELD-OUT split on the corruption positives + broad honest (no source image
in two folds → detection generalization to UNSEEN images, the gb82_dog gap).
negrich has no source_id (provenance gap) → row-split; it is negatives, so what
matters is the FP measured on its held-out rows.

Reports: held-out-source detection (overall + per family/content/severity), FP by
negative subclass — the KEY new one being FP on held-out SEVERE-HONEST (does the
head correctly NOT fire on heavy honest degradation?) — deadband threshold curve,
and the perceptual-miss value-add.
"""
import argparse, json, os, subprocess, struct
import numpy as np, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

NFEAT = 372
CANON = "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22"
FOLD = "/mnt/v/zen/zensim-training/ext720-foldable-2026-07-24"
NEGRICH = ("/mnt/v/zen/zensim-training/kadis-negrich-regen-2026-07-24/"
           "kadis_negrich_srcid.parquet")  # regenerated WITH source_id → leak-free split
DST = "/mnt/v/output/zensim/signedpow-clean-2026-07-24"
PERC = f"{DST}/ideal_p0p2_L0p003_F0p005_bd.bin"


def load_X(path, nfeat, extra=()):
    cols = [f"f{i}" for i in range(nfeat)]
    t = pq.read_table(path, columns=cols + list(extra))
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in cols])
    ex = {e: np.asarray(t.column(e).to_pylist()) for e in extra}
    return X, ex


def split_ids(ids, rng, fracs=(0.6, 0.2, 0.2)):
    uniq = sorted(set(ids)); rng.shuffle(uniq)
    n = len(uniq); a = int(fracs[0]*n); b = int((fracs[0]+fracs[1])*n)
    s = {**{i: "train" for i in uniq[:a]}, **{i: "val" for i in uniq[a:b]},
         **{i: "test" for i in uniq[b:]}}
    return np.array([s[i] for i in ids])


def perc_score(X720, tag):
    tmp = os.path.expanduser(f"~/tmp/corrbuild/{tag}.bin")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as g:
        g.write(struct.pack("<II", 720, len(X720)))
        g.write(X720.astype(np.float32).tobytes(order="C"))
    r = subprocess.run(["./target/release/predict_features_with_bake", "--bake", PERC,
                        "--bake-post", "raw", "--features-file", tmp], capture_output=True, text=True)
    return np.array([float(x) for x in r.stdout.split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--negrich", default=NEGRICH)
    ap.add_argument("--out", default=f"{DST}/corruption_head_372.json")
    ap.add_argument("--honest-per-corpus", type=int, default=12000)
    ap.add_argument("--negrich-n", type=int, default=120000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    # positives + matched anchors (load 720 so we can score the perceptual model;
    # the detector uses only the first 372)
    Xc720, ex = load_X(a.corpus, 720, extra=("is_corruption", "family", "content_class",
                                             "severity", "ref_id"))
    isc = ex["is_corruption"].astype(int)
    print(f"corpus: {len(Xc720)} rows ({int(isc.sum())}c/{int((isc==0).sum())} matched-honest), "
          f"{len(set(ex['ref_id']))} sources, {len(set(ex['family']))} families")

    parts = []  # (X372, y, source, family, content, severity, subclass, X720_or_None)
    parts.append((Xc720[:, :NFEAT], isc, ex["ref_id"], ex["family"], ex["content_class"],
                  ex["severity"], np.where(isc == 1, "corruption", "matched_anchor"), Xc720))

    # negrich severe-honest hard negatives (372) — regenerated WITH source_id, so
    # the split is LEAK-FREE (one KADIS reference → 5 severity rows must not straddle
    # folds). This is the fix the provenance-gapped original could not support.
    if os.path.exists(a.negrich):
        Xn, exn = load_X(a.negrich, NFEAT, extra=("source_id",))
        if len(Xn) > a.negrich_n:
            idx = rng.choice(len(Xn), a.negrich_n, replace=False)
            Xn, sid = Xn[idx], exn["source_id"][idx]
        else:
            sid = exn["source_id"]
        s = np.array([f"negrich/{v}" for v in sid])  # real source_id → leak-free
        lab = lambda v: np.array([v] * len(Xn))
        parts.append((Xn, np.zeros(len(Xn), dtype=int), s, lab("severe_honest"),
                      lab("severe_honest"), lab("severe_honest"), lab("severe_honest"), None))
        print(f"negrich: {len(Xn)} severe-honest hard negatives (372-feat, "
              f"{len(set(sid))} unique KADIS source_ids → leak-free split)")
    else:
        print("WARN: negrich missing — no severe-honest hard negatives!")

    # broad honest easy negatives (span q, diverse content)
    for name, p in [("safesyn", f"{FOLD}/ext_safesyn_full.parquet"),
                    ("cid22val", f"{FOLD}/ext_cid22val.parquet"),
                    ("nonphoto", f"{CANON}/ext_nonphoto_720_nn_full.parquet"),
                    ("csiq", f"{FOLD}/ext_csiq.parquet"), ("live", f"{FOLD}/ext_live.parquet")]:
        if not os.path.exists(p):
            continue
        Xb, exb = load_X(p, NFEAT, extra=("ref_basename",))
        idx = rng.choice(len(Xb), min(a.honest_per_corpus, len(Xb)), replace=False)
        s = np.array([f"{name}/{r}" for r in exb["ref_basename"][idx]])
        lab = lambda v: np.array([v]*len(idx))
        parts.append((Xb[idx], np.zeros(len(idx), dtype=int), s, lab("broad_honest"),
                      lab("broad_honest"), lab("broad_honest"), lab("broad_honest"), None))

    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    src = np.concatenate([p[2] for p in parts])
    fam = np.concatenate([p[3] for p in parts])
    cc = np.concatenate([p[4] for p in parts])
    sub = np.concatenate([p[6] for p in parts])
    which = split_ids(src, rng)
    tr, te = which == "train", which == "test"
    print(f"total {len(X)} rows | split train {tr.sum()} / test {te.sum()} | "
          f"subclasses {dict(zip(*np.unique(sub, return_counts=True)))}")

    sc = StandardScaler().fit(X[tr]); Z = lambda M: np.clip(sc.transform(M), -8, 8)
    clf = CalibratedClassifierCV(LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000),
                                 method="isotonic", cv=3).fit(Z(X[tr]), y[tr])
    P = lambda M: clf.predict_proba(Z(M))[:, 1]

    corr_te = te & (y == 1)
    if corr_te.sum() == 0:
        print("WARN: no held-out-source corruptions in test fold (too few corruption "
              "sources — expected only in a tiny pilot). Skipping detection report.")
        return
    print("\n=== HELD-OUT-SOURCE detection + false-positive by subclass ===")
    for T in (0.5, 0.9, 0.99):
        det = float((P(X[corr_te]) > T).mean())
        line = f"  T={T}: detection={det*100:5.1f}%"
        for sc_name in ("severe_honest", "broad_honest", "matched_anchor"):
            m = te & (y == 0) & (sub == sc_name)
            if m.sum():
                line += f"  FP[{sc_name}]={float((P(X[m])>T).mean())*100:.2f}%"
        print(line)

    print("\n=== per-corruption-family detection (held-out sources, T=0.9) ===")
    for f in sorted(set(fam[corr_te])):
        m = corr_te & (fam == f)
        if m.sum() >= 5:
            print(f"  {f:26s}: {float((P(X[m])>0.9).mean())*100:5.1f}%  (n={int(m.sum())})")

    print("\n=== per-content-class corruption detection (held-out sources, T=0.9) ===")
    for c in sorted(set(cc[corr_te])):
        m = corr_te & (cc == c)
        if m.sum():
            print(f"  {c:10s}: {float((P(X[m])>0.9).mean())*100:5.1f}% (n={int(m.sum())})")

    # perceptual-miss value-add: score the perceptual model (full 720) on the
    # corpus corruption rows in the test fold. The corpus is the FIRST block of the
    # concatenated arrays, so which[:len(Xc720)] are its split labels.
    try:
        te_corpus = which[:len(Xc720)]
        mask = (te_corpus == "test") & (isc == 1)
        if mask.sum():
            ps = perc_score(Xc720[mask], "perc_corr_te")
            det = P(Xc720[mask][:, :NFEAT])
            miss = ps > 40
            if miss.sum():
                print(f"\n=== VALUE-ADD: perceptual scores {float(miss.mean())*100:.1f}% of held-out "
                      f"corruptions >40 ('looks OK'); head catches {float((det[miss]>0.9).mean())*100:.1f}% of THOSE ===")
    except Exception as e:
        print(f"(value-add skipped: {e})")

    lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Z(X[tr]), y[tr])
    json.dump({"nfeat": NFEAT, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
               "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "clip": 8.0},
              open(a.out, "w"))
    print(f"\nsaved 372-feat head → {a.out}")


if __name__ == "__main__":
    main()
