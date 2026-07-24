#!/usr/bin/env python3
"""Build a soft-sign 720-feature screen from an existing smooth screen.

Every compressive transform (signed_cbrt/signed_log1p/log1p/yeo_johnson)
becomes `soft_sign` with a per-feature scale = p<PCTL>(|x|) on the honest
reference corpus (safesyn). Identity stays identity. The scale sets where
the smooth saturation knee sits: values below it stay near-linear (rank
preserved), values above it (heavy honest tail + corruption extremes)
saturate smoothly in (-1,1) — bounded derivative everywhere, so the
central-difference diffmap sensitivities the fold consumes stay clean
(unlike winsor/quantile step transforms).

Usage:
  build_softsign_screen.py <in_smooth_screen.tsv> <out_softsign_screen.tsv> \
      [--ref <safesyn.parquet>] [--pctl 95] [--floor 1e-6]
"""
import argparse
import csv
import sys
import numpy as np
import pyarrow.parquet as pq

COMPRESSIVE = {"signed_cbrt", "signed_log1p", "log1p", "yeo_johnson",
               "signed_sqrt", "log"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_screen")
    ap.add_argument("out_screen")
    ap.add_argument("--ref", default="/mnt/v/zen/zensim-training/"
                    "ext720-foldable-2026-07-24/ext_safesyn_full.parquet")
    ap.add_argument("--pctl", type=float, default=95.0)
    ap.add_argument("--floor", type=float, default=1e-6)
    ap.add_argument("--nfeat", type=int, default=720)
    ap.add_argument("--transform",
                    choices=["soft_sign", "soft_clip", "signed_pow"],
                    default="soft_sign",
                    help="soft_sign: params=[scale=p<pctl>]. "
                         "soft_clip: params=[knee=p<pctl>, soft]. "
                         "signed_pow: params=[pow] (fixed, no percentile pass).")
    ap.add_argument("--soft", type=float, default=1.0,
                    help="soft_clip tail softness (2nd param)")
    ap.add_argument("--pow", type=float, default=0.25,
                    help="signed_pow exponent p (fixed for all features)")
    ap.add_argument("--only-cbrt", action="store_true",
                    help="signed_pow: convert ONLY signed_cbrt entries (keep "
                         "yeo_johnson/log1p/etc). This is the winning HYBRID — "
                         "per-feature mixed smooth screen + aggressive power core.")
    args = ap.parse_args()

    # signed_pow uses a fixed exponent — no per-feature percentile scale needed.
    if args.transform == "signed_pow":
        # --only-cbrt keeps the mixed screen's non-cbrt transforms (the win);
        # otherwise convert the whole COMPRESSIVE family to a uniform power.
        convert = {"signed_cbrt"} if args.only_cbrt else COMPRESSIVE
        with open(args.in_screen) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        fields = list(rows[0].keys()) if rows else []
        n_pow = n_id = n_kept = 0
        with open(args.out_screen, "w", newline="") as g:
            w = csv.DictWriter(g, fieldnames=fields, delimiter="\t")
            w.writeheader()
            for row in rows:
                tok = (row["best_transform"] or "identity").strip()
                if tok in convert:
                    row["best_transform"] = "signed_pow"
                    row["params_csv"] = f"{args.pow:g}"
                    n_pow += 1
                elif tok == "identity":
                    n_id += 1
                else:
                    n_kept += 1
                w.writerow(row)
        print(f"[signed_pow] p={args.pow} only_cbrt={args.only_cbrt} wrote "
              f"{args.out_screen}: {n_pow} signed_pow, {n_id} identity, "
              f"{n_kept} kept-as-is", flush=True)
        return 0

    # Per-feature p<pctl>(|x|) on the honest reference. Feature columns are
    # named f0..f<N-1> in the canonical corpora.
    have = set(pq.read_schema(args.ref).names)
    cols = [f"f{i}" for i in range(args.nfeat)]
    missing = [c for c in cols if c not in have]
    if missing:
        print(f"[softsign] ERROR: {len(missing)} feature cols missing "
              f"(e.g. {missing[:3]}) in {args.ref}", file=sys.stderr)
        return 2
    tbl = pq.read_table(args.ref, columns=cols)
    scales = np.empty(args.nfeat, dtype=np.float64)
    for i, c in enumerate(cols):
        x = np.abs(np.asarray(tbl.column(c).to_numpy(zero_copy_only=False),
                              dtype=np.float64))
        x = x[np.isfinite(x)]
        s = np.percentile(x, args.pctl) if x.size else 0.0
        scales[i] = max(s, args.floor)
    print(f"[softsign] ref={args.ref} pctl={args.pctl} "
          f"scale range [{scales.min():.4g}, {scales.max():.4g}] "
          f"median {np.median(scales):.4g}", flush=True)

    with open(args.in_screen) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fields = list(rows[0].keys()) if rows else []
    n_soft = n_id = n_kept = 0
    with open(args.out_screen, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            i = int(row["feat_idx"])
            tok = (row["best_transform"] or "identity").strip()
            if tok in COMPRESSIVE and i < args.nfeat:
                row["best_transform"] = args.transform
                if args.transform == "soft_clip":
                    row["params_csv"] = f"{scales[i]:.6g},{args.soft:g}"
                else:
                    row["params_csv"] = f"{scales[i]:.6g}"
                n_soft += 1
            elif tok == "identity":
                n_id += 1
            else:
                n_kept += 1
            w.writerow(row)
    print(f"[softsign] wrote {args.out_screen}: "
          f"{n_soft} soft_sign, {n_id} identity, {n_kept} other", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
