#!/usr/bin/env python3
"""Reconstruct per-stimulus JND-scale quality from the JPEG-AI-SDR25 triplet
responses (BTC + PTC), following the AIC-3 / QoMEX'25 methodology (Jenadeleh,
Sneyers, Jia, Mohammadi, Ascenso, Saupe — arXiv:2504.06301): every triplet
shows the pristine original as pivot and asks which side is closer to it, so
with latent distortion magnitudes q >= 0 (original pinned at 0) the response
names the side judged MORE DISTORTED (verified: under a 'closer' reading 383
of 386 workers fail the traps; under 'more distorted' they pass), an
ordered-probit judgment on delta = q_left - q_right:

    P(left)    = Phi((delta - tau) / sigma_m)      # left more distorted
    P(right)   = Phi((-delta - tau) / sigma_m)
    P(notsure) = 1 - P(left) - P(right)

sigma_BTC := 1 defines the scale unit (boosted-sigma units; the boosted
display is the sensitivity reference), sigma_PTC is fitted per image, tau >= 0
is a shared indecision threshold. Per-image joint MLE (all codecs together —
the cross-codec triplets are what tie codec ladders onto one scale) via
L-BFGS. Workers are dropped when their trap accuracy < 0.8 (paper-style
cleaning); 'skip' responses are dropped.

Outputs:
  sdr25_jnd_reconstructed_<date>.parquet  (img_num, codec, dlevel, q_jnd,
                                            n_resp, method_mix)
  stdout sanity: per-ladder monotonicity, trap-filter counts, scale ranges.

This is the T0 eval anchor build (docs/PLAN_BEAT_A.md instrument prereq (a));
DATA_SPLITS.md registers SDR25 as eval-only.
"""
import csv, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

D = "/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim-multicodec-probe/sdr25_jnd_reconstructed_2026-07-02.parquet"

def load(method):
    f = f"{D}/JPEG_AIC_SDR_{method}_JPEG_AI_responses_2025.02.28_v1.csv"
    return list(csv.DictReader(open(f)))

def trap_filter(rows):
    """Drop workers whose trap accuracy < 0.8. A trap pairs the original
    against a heavily distorted image — the distorted side is (codec!=0 or
    dlevel>0); correct = answering that the original side is closer."""
    acc = {}
    for r in rows:
        if r["is_trap"] != "1" or r["response"] not in ("left", "right"):
            continue
        ldist = (r["codec_left"], int(r["dlevel_left"])) != ("0", 0)
        rdist = (r["codec_right"], int(r["dlevel_right"])) != ("0", 0)
        if ldist == rdist:
            continue  # malformed trap
        correct = (r["response"] == "left") == ldist  # pick the DISTORTED side
        a = acc.setdefault(r["worker"], [0, 0])
        a[0] += int(correct); a[1] += 1
    bad = {w for w, (c, n) in acc.items() if n >= 3 and c / n < 0.8}
    kept = [r for r in rows if r["worker"] not in bad]
    return kept, len(bad), len(acc)

def fit_image(img, responses):
    """Joint MLE over this image's stimuli. responses: list of
    (li, ri, resp_code, method_is_ptc) with stimulus indices into stims."""
    stims = sorted({(r["codec_left"], int(r["dlevel_left"])) for r in responses}
                   | {(r["codec_right"], int(r["dlevel_right"])) for r in responses})
    stims = [s for s in stims if s != ("0", 0)]
    idx = {s: i for i, s in enumerate(stims)}
    obs = []
    for r in responses:
        if r["response"] not in ("left", "right", "notsure"):
            continue
        sl, sr = (r["codec_left"], int(r["dlevel_left"])), (r["codec_right"], int(r["dlevel_right"]))
        li = -1 if sl == ("0", 0) else idx[sl]
        ri = -1 if sr == ("0", 0) else idx[sr]
        code = {"left": 0, "right": 1, "notsure": 2}[r["response"]]
        obs.append((li, ri, code, 1.0 if r["method"] == "PTC" else 0.0))
    li = np.array([o[0] for o in obs]); ri = np.array([o[1] for o in obs])
    code = np.array([o[2] for o in obs]); isptc = np.array([o[3] for o in obs])
    n = len(stims)

    def nll(theta):
        q = np.concatenate([[0.0], np.abs(theta[:n])])       # index 0 = original
        tau = np.abs(theta[n]); lsp = theta[n + 1]
        sig = np.where(isptc > 0.5, np.exp(lsp), 1.0)
        delta = q[li + 1] - q[ri + 1]
        pl = norm.cdf((delta - tau) / sig)
        pr = norm.cdf((-delta - tau) / sig)
        pn = np.clip(1.0 - pl - pr, 1e-9, 1.0)
        p = np.where(code == 0, np.clip(pl, 1e-9, 1), np.where(code == 1, np.clip(pr, 1e-9, 1), pn))
        return -np.log(p).sum()

    # init: mean dlevel scaled to ~3 JND across the ladder
    x0 = np.concatenate([np.array([0.3 * dl for (_, dl) in stims]), [0.3, 0.0]])
    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 2000})
    q = np.abs(res.x[:n]); tau = abs(res.x[n]); sig_ptc = float(np.exp(res.x[n + 1]))
    counts = np.bincount(np.concatenate([li[li >= 0], ri[ri >= 0]]), minlength=n)
    return stims, q, tau, sig_ptc, counts, res.fun / max(1, len(obs))

rows = load("BTC") + load("PTC")
rows, nbad, nworkers = trap_filter(rows)
rows = [r for r in rows if r["is_trap"] == "0" and r["response"] in ("left", "right", "notsure")]
print(f"workers: {nworkers} total, {nbad} dropped by trap filter; usable responses: {len(rows):,}")

import pyarrow as pa, pyarrow.parquet as pq
# stimulus -> filename map (observed from response rows; sides carry filenames)
fname = {}
for r in rows:
    fname[(r["img_num"], r["codec_left"], int(r["dlevel_left"]))] = r["img_left"]
    fname[(r["img_num"], r["codec_right"], int(r["dlevel_right"]))] = r["img_right"]
    fname[(r["img_num"], r["codec_pivot"], int(r["dlevel_pivot"]))] = r["img_pivot"]
recs = []
for img in sorted({r["img_num"] for r in rows}, key=int):
    rimg = [r for r in rows if r["img_num"] == img]
    stims, q, tau, sig_ptc, counts, nllps = fit_image(img, rimg)
    # sanity: within-ladder monotonicity (q should rise with dlevel per codec)
    inv = tot = 0
    for c in sorted({c for c, _ in stims}):
        lad = sorted([(dl, q[i]) for (cc, dl), i in ((s, stims.index(s)) for s in stims) if cc == c])
        for a, b in zip(lad, lad[1:]):
            tot += 1; inv += int(b[1] < a[1] - 1e-9)
    print(f"img {img}: {len(stims)} stimuli, q range 0..{q.max():.2f}, tau={tau:.3f}, "
          f"sigma_PTC={sig_ptc:.2f}, ladder inversions {inv}/{tot}, nll/resp={nllps:.3f}")
    for (c, dl), qq, cnt in zip(stims, q, counts):
        recs.append({"img_num": int(img), "codec": int(c), "dlevel": int(dl),
                     "q_jnd": float(qq), "n_resp": int(cnt),
                     "filename": fname.get((img, str(c), int(dl)), "")})
    recs.append({"img_num": int(img), "codec": 0, "dlevel": 0, "q_jnd": 0.0, "n_resp": 0,
                 "filename": fname.get((img, "0", 0), "")})

t = pa.Table.from_pylist(recs)
pq.write_table(t, OUT, compression="zstd")
print(f"wrote {OUT}: {t.num_rows} stimuli")
