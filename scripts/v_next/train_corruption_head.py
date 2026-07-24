#!/usr/bin/env python3
"""Train + rigorously evaluate the structural-corruption DETECTOR (2nd head).

Positives: structural corruptions from build_corruption_corpus.py (many imazen-26
sources × the codec-corpus catalog). Negatives: the matched honest q10/q20 anchors
(same sources) + a broad sample of the existing honest 720 corpora (span q0..100,
diverse content). SOURCE-HELD-OUT split (no source image in two folds) so detection
generalization to UNSEEN images is what's measured — the gap the single-image
(gb82_dog) prototype could not test. Calibrated probabilities for the deadband min().

Reports: detection recall on held-out-source corruptions (overall + per family /
content / severity), false-positive rate on held-out honest, the deadband
threshold curve, and the perceptual-miss value-add (corruptions the perceptual
model scores 'looks OK' that the head still catches).
"""
import argparse, glob, json, os, subprocess, struct
import numpy as np, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

FCOL = [f"f{i}" for i in range(720)]
CANON = "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22"
FOLD = "/mnt/v/zen/zensim-training/ext720-foldable-2026-07-24"
DST = "/mnt/v/output/zensim/signedpow-clean-2026-07-24"
PERC = f"{DST}/ideal_p0p2_L0p003_F0p005_bd.bin"  # perceptual model (dialed)


def load_feats(path, cols=FCOL, extra=()):
    t = pq.read_table(path, columns=list(cols) + list(extra))
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in cols])
    ex = {e: np.asarray(t.column(e).to_pylist()) for e in extra}
    return X, ex


def split_by_source(ids, rng, fracs=(0.6, 0.2, 0.2)):
    uniq = sorted(set(ids))
    rng.shuffle(uniq)
    n = len(uniq); a = int(fracs[0] * n); b = int((fracs[0] + fracs[1]) * n)
    tr, va, te = set(uniq[:a]), set(uniq[a:b]), set(uniq[b:])
    which = np.array(["train" if i in tr else "val" if i in va else "test" for i in ids])
    return which


def perc_score(X, tag):
    tmp = os.path.expanduser(f"~/tmp/corrbuild/{tag}.bin")
    with open(tmp, "wb") as g:
        g.write(struct.pack("<II", 720, len(X)))
        g.write(X.astype(np.float32).tobytes(order="C"))
    r = subprocess.run(["./target/release/predict_features_with_bake", "--bake", PERC,
                        "--bake-post", "raw", "--features-file", tmp], capture_output=True, text=True)
    return np.array([float(x) for x in r.stdout.split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="im26_corruption_720.parquet")
    ap.add_argument("--out", default=f"{DST}/corruption_head.json")
    ap.add_argument("--honest-per-corpus", type=int, default=12000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    # --- positives + matched honest anchors ---
    Xc, ex = load_feats(a.corpus, extra=("is_corruption", "family", "content_class",
                                          "severity", "ref_id"))
    isc = ex["is_corruption"].astype(int)
    src = ex["ref_id"]
    print(f"corpus: {len(Xc)} rows ({int(isc.sum())} corruption / {int((isc==0).sum())} matched-honest), "
          f"{len(set(src))} sources, {len(set(ex['family']))} families")

    # --- broad honest negatives from existing 720 corpora (span q, diverse content) ---
    Xh_list, hsrc_list = [], []
    for name, p in [("safesyn", f"{FOLD}/ext_safesyn_full.parquet"),
                    ("cid22val", f"{FOLD}/ext_cid22val.parquet"),
                    ("nonphoto", f"{CANON}/ext_nonphoto_720_nn_full.parquet"),
                    ("csiq", f"{FOLD}/ext_csiq.parquet"),
                    ("live", f"{FOLD}/ext_live.parquet")]:
        if not os.path.exists(p):
            continue
        Xb, exb = load_feats(p, extra=("ref_basename",))
        idx = rng.choice(len(Xb), min(a.honest_per_corpus, len(Xb)), replace=False)
        Xh_list.append(Xb[idx])
        hsrc_list.append(np.array([f"{name}/{r}" for r in exb["ref_basename"][idx]]))
    Xh = np.vstack(Xh_list); hsrc = np.concatenate(hsrc_list)
    print(f"broad honest: {len(Xh)} rows from {len(Xh_list)} corpora")

    # --- assemble + SOURCE-held-out split ---
    X = np.vstack([Xc, Xh])
    y = np.r_[isc, np.zeros(len(Xh), dtype=int)]
    allsrc = np.concatenate([src, hsrc])
    fam = np.concatenate([ex["family"], np.array(["honest"] * len(Xh))])
    cc = np.concatenate([ex["content_class"], np.array(["honest"] * len(Xh))])
    sev = np.concatenate([ex["severity"], np.array(["honest"] * len(Xh))])
    which = split_by_source(allsrc, rng)
    tr, va, te = which == "train", which == "val", which == "test"
    print(f"split (source-held-out): train {tr.sum()} / val {va.sum()} / test {te.sum()}")

    # --- train calibrated classifier on standardized+clipped features ---
    sc = StandardScaler().fit(X[tr])
    Z = lambda M: np.clip(sc.transform(M), -8, 8)
    base = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Z(X[tr]), y[tr])
    P = lambda M: clf.predict_proba(Z(M))[:, 1]
    pte = P(X[te])

    # --- rigorous held-out-source evaluation ---
    corr_te = te & (y == 1)
    hon_te = te & (y == 0)
    print("\n=== HELD-OUT-SOURCE detection + false-positive ===")
    for T in (0.5, 0.9, 0.99):
        det = float((P(X[corr_te]) > T).mean())
        fp = float((P(X[hon_te]) > T).mean())
        print(f"  T={T}: detection={det*100:.1f}%  honest-FP={fp*100:.2f}%")
    pc = P(X[corr_te]); ph = P(X[hon_te])
    print(f"  corruption P: P10={np.percentile(pc,10):.3f} P50={np.median(pc):.3f} | "
          f"honest P: P90={np.percentile(ph,90):.3f} P99={np.percentile(ph,99):.3f} MAX={ph.max():.3f}")

    print("\n=== per-corruption-family detection (held-out sources, T=0.9) ===")
    for f in sorted(set(fam[corr_te])):
        m = corr_te & (fam == f)
        if m.sum() >= 5:
            print(f"  {f:26s}: {float((P(X[m])>0.9).mean())*100:5.1f}%  (n={int(m.sum())})")

    print("\n=== per-content-class (held-out sources, T=0.9) ===")
    for c in sorted(set(cc[te])):
        mc = te & (cc == c) & (y == 1); mh = te & (cc == c) & (y == 0)
        d = float((P(X[mc])>0.9).mean())*100 if mc.sum() else float('nan')
        fpr = float((P(X[mh])>0.9).mean())*100 if mh.sum() else float('nan')
        print(f"  {c:10s}: detection={d:5.1f}% (n={int(mc.sum())})  FP={fpr:5.2f}% (n={int(mh.sum())})")

    # --- perceptual-miss value-add: corruptions the perceptual model scores 'OK' ---
    try:
        ps = perc_score(X[corr_te], "perc_corr_te")
        miss = ps > 40
        if miss.sum():
            catch = float((P(X[corr_te][miss]) > 0.9).mean())
            print(f"\n=== VALUE-ADD: perceptual model scores {float(miss.mean())*100:.1f}% of held-out "
                  f"corruptions >40 ('looks OK'); the head catches {catch*100:.1f}% of THOSE ===")
    except Exception as e:
        print(f"(perceptual value-add skipped: {e})")

    # --- persist head (weights via a plain logistic refit for a portable JSON) ---
    lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Z(X[tr]), y[tr])
    json.dump({"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
               "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
               "clip": 8.0, "note": "P=sigmoid(clip((x-mean)/scale)·coef+intercept); calibrate separately"},
              open(a.out, "w"))
    print(f"\nsaved head → {a.out}")


if __name__ == "__main__":
    main()
