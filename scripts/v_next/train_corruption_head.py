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

## The D-companion mode (2026-09-05)

Shipped Profile D (`ADD156`) reads 28 of 156 BASIC lines and zero pool lines, and
its walk runs `V1PoolsMode::Peaks` — which `fold_engine.rs` documents costs the
SAME as `Off` ("the peak accumulators are the fused V-blur kernel's unconditional
L8/max tier"), while masked/IW (`f228..371`) would force `Full`. So a head for D
is free on `f0..155` AND on `f0..227`, and NOT free past that. The 2026-07-24
head read all 372 including masked/IW, so it is not a D companion at any price.

Select the slice with `--feat-range 0:156` / `--feat-range 0:228`.

`--broad-honest LABEL:PATH[:IDCOL]` (repeatable) replaces the hard-coded ext720
list, so the broad negatives can be an era-matched instrument instead of whatever
720 tables happened to exist. `--corpus-nfeat` is auto-detected; the perceptual
value-add needs a 720 corpus and is skipped (loudly) otherwise.
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


def load_X(path, idx, extra=()):
    cols = [f"f{i}" for i in idx]
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


def emit_znpr(out_path, bake_bin, caller_width, feat_idx, mean, scale, coef,
              intercept, clip, iso, u_lo, u_hi, provenance):
    """Emit the logistic+isotonic head as a ZNPR v3 bake.

    Every piece maps onto the existing wire format — no new sections, and the
    JSON pipeline mandate is honoured (emit BakeRequestJson, shell to
    `zenpredict-bake`; never hand-roll ZNPR bytes):

      * `clip((x-mean)/scale, +-clip)`  ->  a raw-space `winsor_p99` transform at
        `mean +- clip*scale` followed by the scaler. Transforms run BEFORE the
        scaler and the scaler stats were fit on UNCLIPPED data, so the raw bound
        reproduces the training-time clip exactly.
      * lines outside `feat_idx`        ->  `drop` (arity 0). The bake therefore
        still ACCEPTS `caller_width` features while forwarding only len(feat_idx)
        — which is what `bake_verdict`'s `head.caller_input_width() ==
        grid.n_features` check requires.
      * logistic weights                ->  one identity 1-output layer, NEGATED.
      * isotonic calibration            ->  `zentrain.output_calibration_spline`.

    The negation is load-bearing. `corruption_gate` is `score(corruption) <
    score(q20)`, i.e. HIGHER = better quality, while the logistic emits P(corrupt)
    which is higher = worse. Negating the layer makes `u = -z` increase with
    quality, so the spline `u -> 100*(1-P(-u))` is monotone INCREASING and the
    format's strictly-increasing-x contract holds with no sign tricks downstream.

    Because that map is monotone, the gate's pass rate depends only on the linear
    ordering — the calibration cannot change it. The spline is still baked so the
    OUTPUT is interpretable: emitted score = 100*(1-P), so a deadband threshold
    `P > T` reads as `score < 100*(1-T)`.
    """
    import struct, tempfile
    n_used = len(feat_idx)
    used = set(int(i) for i in feat_idx)
    tf, tp = [], []
    pos = {int(f): k for k, f in enumerate(feat_idx)}
    for i in range(caller_width):
        if i in used:
            k = pos[i]
            tf.append("winsor_p99")
            tp.append(f"{mean[k] - clip * scale[k]:.9g},{mean[k] + clip * scale[k]:.9g}")
        else:
            tf.append("drop")
            tp.append("")
    # Spline knots. A linspace alone is NOT enough: isotonic regression is a STEP
    # function, and PCHIP through evenly-spaced samples rounds its plateau edges
    # (measured: 97.002 -> 97.122 on a re-bake of the 2026-07-24 head). So take
    # the union of a dense linspace and the isotonic's OWN breakpoints, mapped
    # back through u = -logit(p), bracketed +-eps so each step is represented by
    # a point on each side. Ordering is monotone either way — this is for the
    # OUTPUT value (and hence the deadband), not for the gate.
    span = max(u_hi - u_lo, 1e-6)
    lo, hi = u_lo - 0.05 * span, u_hi + 0.05 * span
    xs = list(np.linspace(lo, hi, 512))
    pt = np.clip(np.asarray(iso.X_thresholds_, dtype=np.float64), 1e-12, 1 - 1e-12)
    for u in -np.log(pt / (1.0 - pt)):
        if lo < u < hi:
            xs += [u - 1e-4, u + 1e-4]
    xs = np.unique(np.asarray(xs, dtype=np.float64))
    P = iso.predict(1.0 / (1.0 + np.exp(np.clip(xs, -60, 60))))   # sigmoid(-u) = sigmoid(z)
    ys = 100.0 * (1.0 - P)
    keep = [0]
    for i in range(1, len(xs)):                      # strictly increasing x in f32
        if np.float32(xs[i]) > np.float32(xs[keep[-1]]):
            keep.append(i)
    xs, ys = xs[keep], ys[keep]
    payload = struct.pack("<I", len(xs)) + b"".join(
        struct.pack("<ff", float(x), float(y)) for x, y in zip(xs, ys))
    req = {
        "schema_hash": 0,
        "flags": 0,
        "scaler_mean": [float(v) for v in mean],
        "scaler_scale": [float(v) for v in scale],
        "layers": [{"in_dim": n_used, "out_dim": 1, "activation": "identity",
                    "dtype": "f32",
                    "weights": [float(-c) for c in coef],
                    "biases": [float(-intercept)]}],
        "metadata": [
            {"key": "zentrain.feature_transforms", "type": "utf8",
             "text": "\n".join(tf)},
            {"key": "zentrain.feature_transform_params", "type": "utf8",
             "text": "\n".join(tp)},
            {"key": "zentrain.output_calibration_spline", "type": "bytes",
             "hex": payload.hex()},
            {"key": "zentrain.repro", "type": "utf8",
             "text": json.dumps(provenance, sort_keys=True)},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     dir=os.path.expanduser("~/tmp")) as f:
        json.dump(req, f)
        reqp = f.name
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    r = subprocess.run([bake_bin, reqp, out_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"zenpredict-bake failed rc={r.returncode}: "
                         f"{r.stderr.strip()[:400]}")
    print(f"BAKED -> {out_path} ({os.path.getsize(out_path)} B; caller width "
          f"{caller_width}, forwards {n_used}; score = 100*(1-P), so a deadband "
          f"P>T reads as score < {100:.0f}*(1-T))")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--negrich", default=NEGRICH)
    ap.add_argument("--out", default="/mnt/v/output/zensim/corruption-head-2026-07-24/"
                    "corruption_head_372.json")
    ap.add_argument("--honest-per-corpus", type=int, default=12000)
    ap.add_argument("--negrich-n", type=int, default=120000)
    ap.add_argument("--ablate", action="store_true",
                    help="sweep #features (top-K by |coef|) → the minimal detector for perf")
    ap.add_argument("--nfeat", type=int, default=372,
                    help="372 (v1, uses the native-372 negrich) or 720 (v1++v2; the "
                         "corpus is 720 so this tests whether v2 helps — negrich is "
                         "skipped at 720 until it is regenerated at 720 via kadis-distort)")
    ap.add_argument("--feat-range", default=None,
                    help="A:B — use features fA..f(B-1). The D-companion slices are "
                         "0:156 (basic) and 0:228 (basic+peaks); both are free at D's "
                         "runtime because its walk already runs V1PoolsMode::Peaks.")
    ap.add_argument("--broad-honest", action="append", default=None,
                    help="LABEL:PATH[:IDCOL] broad-honest negatives (repeatable). "
                         "IDCOL defaults to ref_basename. Overrides the legacy list.")
    ap.add_argument("--thresholds", default="0.5,0.9,0.95,0.99",
                    help="comma-separated detection thresholds to report")
    ap.add_argument("--bake-out", default=None,
                    help="also emit the head as a ZNPR v3 bake at this path, so "
                         "`bake_verdict --corruption-head` can score it (that flag "
                         "takes a BAKE, never the JSON — which is why no 372 head "
                         "has ever been through the gate). Needs zenpredict-bake.")
    ap.add_argument("--bake-bin", default=os.path.expanduser(
                        "~/work/zen/zenanalyze/target/release/zenpredict-bake"))
    ap.add_argument("--bake-extra-width", type=int, action="append", default=None,
                    help="ALSO emit the head at this caller width (repeatable). The "
                         "372 v1 layout is what the corruption GRID and the 372 eval "
                         "root speak; the runtime fold emits the 944 layout, whose "
                         "f0..155 basic and f156..227 peaks sit at the SAME indices "
                         "(zensim-bench's `peaks156_no_raw` arm is a 156+peaks head "
                         "in the 944 layout). Same coefficients, wider drop list, so "
                         "one fit serves both without a second training run. Written "
                         "next to --bake-out as `<stem>_w<width>.bin`.")
    ap.add_argument("--feat-subset", default=None,
                    help="npy of feature indices to restrict to (e.g. the dial+diffmap "
                         "foldable subset). Default: f0..f{nfeat-1}.")
    ap.add_argument("--severe-720", default=None,
                    help="a 720-feat severe-honest parquet (matched kadis-distort) to use "
                         "as the severe_honest negatives INSTEAD of native-372 negrich — "
                         "required for a v2/foldable subset. Leak-free split on its ref_id.")
    a = ap.parse_args()
    global NFEAT
    NFEAT = a.nfeat
    rng = np.random.default_rng(0)
    if a.feat_subset:
        FEAT_IDX = np.load(a.feat_subset).astype(int).tolist()
        fdesc = "subset " + os.path.basename(a.feat_subset)
    elif a.feat_range:
        lo, hi = (int(v) for v in a.feat_range.split(":"))
        FEAT_IDX = list(range(lo, hi))
        fdesc = f"f{lo}..f{hi-1}"
    else:
        FEAT_IDX = list(range(NFEAT))
        fdesc = f"f0..f{NFEAT-1}"
    print(f"feature set: {len(FEAT_IDX)} features ({fdesc})")
    THRESHOLDS = tuple(float(x) for x in a.thresholds.split(","))

    # positives + matched anchors. The corpus width is READ, not assumed — a 372
    # corpus (the current-extractor D-companion build) has no f372.. at all, and
    # asking for range(720) there is an unhelpful pyarrow KeyError.
    import pyarrow.parquet as _pq
    _names = _pq.ParquetFile(a.corpus).schema_arrow.names
    CORPUS_NFEAT = 1 + max(int(c[1:]) for c in _names
                           if c.startswith("f") and c[1:].isdigit())
    print(f"corpus width: {CORPUS_NFEAT} features ({os.path.basename(a.corpus)})")
    if max(FEAT_IDX) >= CORPUS_NFEAT:
        raise SystemExit(f"--feat-range/--feat-subset asks for f{max(FEAT_IDX)} but "
                         f"the corpus only has f0..f{CORPUS_NFEAT-1}")
    Xc720, ex = load_X(a.corpus, range(CORPUS_NFEAT), extra=("is_corruption", "family",
                       "content_class", "severity", "ref_id"))
    isc = ex["is_corruption"].astype(int)
    print(f"corpus: {len(Xc720)} rows ({int(isc.sum())}c/{int((isc==0).sum())} matched-honest), "
          f"{len(set(ex['ref_id']))} sources, {len(set(ex['family']))} families")

    parts = []  # (X_subset, y, source, family, content, severity, subclass, X720_or_None)
    parts.append((Xc720[:, FEAT_IDX], isc, ex["ref_id"], ex["family"], ex["content_class"],
                  ex["severity"], np.where(isc == 1, "corruption", "matched_anchor"), Xc720))

    # severe-honest hard negatives — leak-free split on source. Prefer the matched
    # 720 kadis-distort set (works with ANY feature subset incl. v2/foldable);
    # else the native-372 negrich (regenerated WITH source_id).
    def add_severe(Xn, srcids, n_uniq, tag):
        s = np.array([f"severe/{v}" for v in srcids])
        lab = lambda v: np.array([v] * len(Xn))
        parts.append((Xn, np.zeros(len(Xn), dtype=int), s, lab("severe_honest"),
                      lab("severe_honest"), lab("severe_honest"), lab("severe_honest"), None))
        print(f"severe_honest: {len(Xn)} hard negatives ({tag}, {n_uniq} unique "
              f"sources → leak-free split)")
    if a.severe_720 and os.path.exists(a.severe_720):
        Xn, exn = load_X(a.severe_720, FEAT_IDX, extra=("ref_id",))
        if len(Xn) > a.negrich_n:
            k = rng.choice(len(Xn), a.negrich_n, replace=False); Xn, sid = Xn[k], exn["ref_id"][k]
        else:
            sid = exn["ref_id"]
        add_severe(Xn, sid, len(set(sid)), f"matched-720 {os.path.basename(a.severe_720)}")
    elif max(FEAT_IDX) < 372 and os.path.exists(a.negrich):
        Xn, exn = load_X(a.negrich, FEAT_IDX, extra=("source_id",))
        if len(Xn) > a.negrich_n:
            k = rng.choice(len(Xn), a.negrich_n, replace=False); Xn, sid = Xn[k], exn["source_id"][k]
        else:
            sid = exn["source_id"]
        add_severe(Xn, sid, len(set(sid)), "native-372 negrich")
    else:
        print("WARN: no severe-honest source for this feature subset "
              "(need --severe-720 for a v2/foldable subset)!")

    # broad honest easy negatives (span q, diverse content)
    if a.broad_honest:
        broad = []
        for spec in a.broad_honest:
            bits = spec.split(":")
            broad.append((bits[0], bits[1], bits[2] if len(bits) > 2 else "ref_basename"))
    else:
        broad = [(n, p, "ref_basename") for n, p in
                 [("safesyn", f"{FOLD}/ext_safesyn_full.parquet"),
                  ("cid22val", f"{FOLD}/ext_cid22val.parquet"),
                  ("nonphoto", f"{CANON}/ext_nonphoto_720_nn_full.parquet"),
                  ("csiq", f"{FOLD}/ext_csiq.parquet"), ("live", f"{FOLD}/ext_live.parquet")]]
    for name, p, idcol in broad:
        if not os.path.exists(p):
            print(f"broad_honest {name}: MISSING {p} — skipped")
            continue
        Xb, exb = load_X(p, FEAT_IDX, extra=(idcol,))
        exb["ref_basename"] = exb[idcol]
        print(f"broad_honest {name}: {len(Xb)} rows, id column '{idcol}'")
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
    tr, va, te = which == "train", which == "val", which == "test"
    print(f"total {len(X)} rows | split train {tr.sum()} / val {va.sum()} / test {te.sum()} | "
          f"subclasses {dict(zip(*np.unique(sub, return_counts=True)))}")

    sc = StandardScaler().fit(X[tr]); Z = lambda M: np.clip(sc.transform(M), -8, 8)

    # --- feature ablation: how few features does the detector actually need? ---
    if getattr(a, "ablate", False):
        def feat_desc(i):
            # v1-372 layout: basic-156 = 13/ch/scale × 3ch × 4scale (f0..155);
            # masked/iw/peak = f156..371 (18/ch/scale × 3ch × 4scale).
            if i < 156:
                blk, loc = "basic", i
                scale = loc // 39; ch = (loc % 39) // 13
            else:
                blk, loc = "mask/iw/peak", i - 156
                scale = loc // 54; ch = (loc % 54) // 18
            return f"{blk} s{scale} c{ch}"
        base = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Z(X[tr]), y[tr])
        order = np.argsort(-np.abs(base.coef_[0]))  # importance on standardized feats
        corr_te = te & (y == 1)
        print("\n=== ABLATION: detection + FP vs #features (top-K by |coef|, T=0.9) ===")
        print(f"  {'K':>4} {'detection':>10} {'severe_FP':>10} {'broad_FP':>9}")
        for K in [5, 8, 12, 16, 24, 32, 48, 64, 96, 156, 372]:
            cols = order[:K]
            Zk = Z(X)[:, cols]
            ck = CalibratedClassifierCV(LogisticRegression(C=0.05, class_weight="balanced",
                 max_iter=3000), method="isotonic", cv=3).fit(Zk[tr], y[tr])
            pk = ck.predict_proba(Zk)[:, 1]
            det = float((pk[corr_te] > 0.9).mean())
            sfp = te & (y == 0) & (sub == "severe_honest")
            bfp = te & (y == 0) & (sub == "broad_honest")
            sv = float((pk[sfp] > 0.9).mean()) if sfp.sum() else float("nan")
            bv = float((pk[bfp] > 0.9).mean()) if bfp.sum() else float("nan")
            print(f"  {K:>4} {det*100:>9.1f}% {sv*100:>9.2f}% {bv*100:>8.2f}%")
        print("\n  top-16 features (index : scale/channel/family):")
        for i in order[:16]:
            print(f"    f{int(i):<3} coef={base.coef_[0][i]:+.3f}  {feat_desc(int(i))}")
        return

    clf = CalibratedClassifierCV(LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000),
                                 method="isotonic", cv=3).fit(Z(X[tr]), y[tr])
    P = lambda M: clf.predict_proba(Z(M))[:, 1]

    corr_te = te & (y == 1)
    if corr_te.sum() == 0:
        print("WARN: no held-out-source corruptions in test fold (tiny pilot). Skipping.")
        return

    metrics = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
               "subclass_counts": {k: int(v) for k, v in zip(*np.unique(sub, return_counts=True))},
               "threshold_curve": [], "per_family_detection_T0.9": {},
               "per_content_T0.9": {}}
    print("\n=== HELD-OUT-SOURCE detection + false-positive by subclass ===")
    pcorr = P(X[corr_te])
    for T in THRESHOLDS:
        row = {"T": T, "detection": float((pcorr > T).mean())}
        line = f"  T={T}: detection={row['detection']*100:5.1f}%"
        for scn in ("severe_honest", "broad_honest", "matched_anchor"):
            m = te & (y == 0) & (sub == scn)
            if m.sum():
                row[f"fp_{scn}"] = float((P(X[m]) > T).mean())
                line += f"  FP[{scn}]={row[f'fp_{scn}']*100:.2f}%"
        metrics["threshold_curve"].append(row); print(line)

    print("\n=== per-corruption-family detection (held-out sources, T=0.9) ===")
    for f in sorted(set(fam[corr_te])):
        m = corr_te & (fam == f)
        if m.sum() >= 5:
            d = float((P(X[m]) > 0.9).mean()); metrics["per_family_detection_T0.9"][f] = d
            print(f"  {f:26s}: {d*100:5.1f}%  (n={int(m.sum())})")

    print("\n=== per-content-class corruption detection (held-out sources, T=0.9) ===")
    for c in sorted(set(cc[corr_te])):
        m = corr_te & (cc == c)
        if m.sum():
            d = float((P(X[m]) > 0.9).mean()); metrics["per_content_T0.9"][c] = d
            print(f"  {c:10s}: {d*100:5.1f}% (n={int(m.sum())})")

    if CORPUS_NFEAT != 720 or not os.path.exists(PERC):
        print(f"\n(value-add SKIPPED: needs a 720 corpus + {PERC}; "
              f"corpus is {CORPUS_NFEAT}-wide)")
    try:
        if CORPUS_NFEAT != 720 or not os.path.exists(PERC):
            raise RuntimeError("no 720 perceptual reference for this corpus")
        te_corpus = which[:len(Xc720)]
        mask = (te_corpus == "test") & (isc == 1)
        if mask.sum():
            ps = perc_score(Xc720[mask], "perc_corr_te"); det = P(Xc720[mask][:, FEAT_IDX])
            miss = ps > 40
            if miss.sum():
                metrics["value_add"] = {"perceptual_miss_frac": float(miss.mean()),
                                        "head_catches_of_misses": float((det[miss] > 0.9).mean())}
                print(f"\n=== VALUE-ADD: perceptual scores {float(miss.mean())*100:.1f}% of held-out "
                      f"corruptions >40; head catches {float((det[miss]>0.9).mean())*100:.1f}% of THOSE ===")
    except Exception as e:
        print(f"(value-add skipped: {e})")

    # recommend the deadband threshold: lowest T with severe-honest FP < 1%
    rec = 0.99
    for row in metrics["threshold_curve"]:
        if row.get("fp_severe_honest", 1.0) < 0.01:
            rec = row["T"]; break
    metrics["recommended_deadband_T"] = rec

    # --- PERSIST durably to block storage: portable head + calibration + metrics + manifest ---
    from sklearn.isotonic import IsotonicRegression
    lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Z(X[tr]), y[tr])
    raw_va = lr.decision_function(Z(X[va])) if va.sum() else lr.decision_function(Z(X[tr]))
    y_va = y[va] if va.sum() else y[tr]
    iso = IsotonicRegression(out_of_bounds="clip").fit(1 / (1 + np.exp(-raw_va)), y_va)
    outdir = os.path.dirname(a.out); os.makedirs(outdir, exist_ok=True)
    head = {"nfeat": len(FEAT_IDX), "feat_idx": [int(i) for i in FEAT_IDX],
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "clip": 8.0,
            "calibration": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
            "recommended_deadband_T": rec,
            "deploy": "x=feat[feat_idx]; P=isotonic(sigmoid(clip((x-mean)/scale,±8)·coef+intercept)); "
                      "gate=100 unless P>recommended_deadband_T; final=min(perceptual,gate)"}
    json.dump(head, open(a.out, "w"))
    if a.bake_out:
        u_tr = -lr.decision_function(Z(X[tr]))
        emit_znpr(a.bake_out, a.bake_bin, CORPUS_NFEAT, FEAT_IDX,
                  sc.mean_[:], sc.scale_[:], lr.coef_[0], float(lr.intercept_[0]),
                  8.0, iso, float(u_tr.min()), float(u_tr.max()),
                  {"corpus": a.corpus, "corpus_nfeat": CORPUS_NFEAT,
                   "negrich": a.negrich, "feat_range": a.feat_range,
                   "n_features": len(FEAT_IDX), "seed": 0,
                   "recommended_deadband_T": rec,
                   "broad_honest": [list(b) for b in broad]})
        for w in (a.bake_extra_width or []):
            if w < (max(FEAT_IDX) + 1):
                raise SystemExit(f"--bake-extra-width {w} is narrower than the "
                                 f"head's highest read line f{max(FEAT_IDX)}")
            stem, ext = os.path.splitext(a.bake_out)
            emit_znpr(f"{stem}_w{w}{ext}", a.bake_bin, w, FEAT_IDX,
                      sc.mean_[:], sc.scale_[:], lr.coef_[0], float(lr.intercept_[0]),
                      8.0, iso, float(u_tr.min()), float(u_tr.max()),
                      {"corpus": a.corpus, "corpus_nfeat": CORPUS_NFEAT,
                       "negrich": a.negrich, "feat_range": a.feat_range,
                       "n_features": len(FEAT_IDX), "seed": 0, "caller_width": w,
                       "recommended_deadband_T": rec,
                       "broad_honest": [list(b) for b in broad]})
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=1)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    json.dump({"artifact": os.path.basename(a.out), "build_commit": commit,
               "corpus": a.corpus, "negrich": a.negrich, "perceptual_model": PERC,
               "nfeat": len(FEAT_IDX), "feat_range": a.feat_range,
               "corpus_nfeat": CORPUS_NFEAT,
               "broad_honest": [list(b) for b in broad],
               "thresholds": list(THRESHOLDS),
               "split": "source-held-out (ref_id / KADIS source_id / broad id), seed 0",
               "n_sources_corruption": len(set(ex["ref_id"])),
               "recommended_deadband_T": rec,
               "key_result": {k: metrics.get(k) for k in ("value_add",)},
               "deadband_row": next((r for r in metrics["threshold_curve"]
                                     if r["T"] == rec), None)},
              open(os.path.join(outdir, "_MANIFEST.json"), "w"), indent=1)
    print(f"\nPERSISTED → {outdir}/")
    print(f"  corruption_head_372.json (weights+calibration), metrics.json, _MANIFEST.json")
    print(f"  recommended deadband T={rec}")


if __name__ == "__main__":
    main()
