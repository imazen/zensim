#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/pin_sigma_720.py
# sha256(source): cfc32706329827617ec773d9912b0c7c535c49259f96457ec9462f32786bf6d4
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Pin the e2 LOO Σ definition empirically: recompute per-family per-corpus
LOO deltas (drop − lin720, in |SROCC|) from today's bake_verdict re-runs of
the ARCHIVED lintwin bins, then find which corpus subset's Σ reproduces the
published Σ values (e2_optimal_model_720_vs_372_2026-07-23.md table:
banding +0.401, peak +0.227, transducer_bank +0.072, edge_width +0.038,
pjnd_core +0.027, blockiness −0.172, ringing −0.608, gms −1.024, iw −1.100,
masked −1.643, basic −2.103). Sign convention: positive = family HURT
(dropping it improved the held-out SROCC)."""
import itertools
import json
from pathlib import Path

P = Path("/mnt/v/output/zensim/bandvis-loo-2026-07-28/pin720")
FAMS = ["banding", "basic", "blockiness", "edge_width", "gms", "iw", "masked",
        "peak", "pjnd_core", "ringing", "transducer_bank"]
PUB = {"banding": 0.401, "peak": 0.227, "transducer_bank": 0.072,
       "edge_width": 0.038, "pjnd_core": 0.027, "blockiness": -0.172,
       "ringing": -0.608, "gms": -1.024, "iw": -1.100, "masked": -1.643,
       "basic": -2.103}


def sroccs(path: Path) -> dict[str, float]:
    d = json.load(open(path))
    out = {}
    for c in d["corpora"]:
        out[c["display"]] = float(c["srocc"])
    return out


full = sroccs(P / "lin720.json")
corpora = list(full)
deltas = {}
for f in FAMS:
    drop = sroccs(P / f"drop_{f}.json")
    deltas[f] = {c: drop[c] - full[c] for c in corpora}

print("per-family per-corpus deltas (drop − lin720, |SROCC|):")
hdr = "family".ljust(16) + "".join(c.rjust(10) for c in corpora)
print(hdr)
for f in FAMS:
    print(f.ljust(16) + "".join(f"{deltas[f][c]:+.4f}".rjust(10) for c in corpora))

best = None
for r in range(6, len(corpora) + 1):
    for sub in itertools.combinations(corpora, r):
        err = 0.0
        for f in FAMS:
            s = sum(deltas[f][c] for c in sub)
            err = max(err, abs(s - PUB[f]))
        if best is None or err < best[0]:
            best = (err, sub)
print(f"\nbest subset (max |Σ − published| = {best[0]:.4f}, size {len(best[1])}):")
print(" ", ", ".join(best[1]))
for f in FAMS:
    s = sum(deltas[f][c] for c in best[1])
    print(f"  {f}: Σ={s:+.4f} (published {PUB[f]:+.3f})")
json.dump({"subset": best[1], "max_err": best[0],
           "deltas": deltas, "full": full},
          open(P / "sigma_pin.json", "w"), indent=1)
print(f"wrote {P/'sigma_pin.json'}")
