#!/usr/bin/env python3
"""build_tbig_200k.py — the committed rebuild of the E-M campaign's bigcodec
training slice (`tbig_924_200k`) at any regime width.

The 924-era builder was uncommitted scratch (`~/tmp`-class, coh924 session); this
is its durable owner (SOTA-944 pre-reg §5/§8, benchmarks/sota944_campaign_2026-08-03.md).
Mechanism (recovered + verified against the surviving
/mnt/v/zen/zensim-training/tbig_924_200k.parquet, 208,169 rows; the view ORDER
and key form were pinned empirically from that file's segment boundaries):

  for each of the 4 lossy TRAIN views in the order
  (zenjpeg_lossy, zenwebp_lossy, zenjxl_lossy, zenavif_lossy):
      step = n_rows // 50000
      keep rows 0, step, 2*step, ...          (deterministic global stride)
      human_score = clip(score_ssim2 / 100, 0, 1)
      ref_basename = ref_filename             (already carries ".png")
  concat -> one parquet: ref_basename, human_score, encoded_filename, f0..f<N-1>

`encoded_filename` is NEW vs the 924 slice (the free join-key fix the 924 build
forgot); feature columns come from the view verbatim.

Gates (all hard, run in-process):
  G-T1: (ref_basename, human_score) sequence EXACTLY equals the reference
        slice's (row-for-row) when --verify-against is given.
  G-T2: f0..f923 bitwise-identical to the reference slice on every row
        (columnar equality; f64 bit pattern via cast to int64 view).

KEY EMISSION (--emit-keys / --keys-only, added 2026-08-30 for W-LIN round 7):
the same defect `build_eval_slices_944.py` had — the leg keeps `encoded_filename`
but drops `codec` / `q` / `origin_id` / `score_ssim2`, so its rows cannot be
resolved to BYTES and therefore cannot be re-extracted at another regime.
`--emit-keys` writes a sidecar `keys_<name>.parquet` holding exactly the rows the
leg holds, in the SAME order, with the identity columns carried through;
`--keys-only` skips the feature columns entirely (the stride arithmetic is
identical, so the key rows are the leg's rows by construction — a cheap read that
touches no feature column). `--keys-only` requires no feature columns to exist in
the view at all, which is what makes it usable against a regime whose views have
not been built yet.

ASSEMBLY (--from-features, added 2026-08-30): rebuild the leg at a NEW regime from
(a) the key sidecar and (b) a features table produced by an extraction of those
keys. The join is `join_safety.safe_key_join_arrow` on `encoded_filename` (a
per-pair distortion identifier, never a ref-only key), the key table's row ORDER is
preserved, and 100 % resolution is required — a partial join aborts. `human_score`
comes from the KEY table (`clip(score_ssim2/100,0,1)`, the same expression the
feature path uses), never from the features table, so the target is identical
across regimes by construction.

Usage:
  python3 scripts/canonical_corpus/build_tbig_200k.py \
      --views-root /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec \
      --suffix _944 --n-feat 944 \
      --verify-against /mnt/v/zen/zensim-training/tbig_924_200k.parquet \
      --out /mnt/v/zen/zensim-training/tbig_944_200k.parquet
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from join_safety import safe_key_join_arrow  # noqa: E402

VIEWS = ["zenjpeg_lossy", "zenwebp_lossy", "zenjxl_lossy", "zenavif_lossy"]
TARGET_PER_VIEW = 50000
KEY_COLS = ["origin_id", "ref_filename", "encoded_filename", "codec", "q",
            "knob_tuple_json", "score_ssim2", "score_zensim"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit() -> str:
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def assemble_from_features(a) -> int:
    """Rebuild the leg at a NEW regime from a key sidecar + a features table.

    The key table owns row ORDER and the TARGET; the features table owns only the
    feature columns. The join is `safe_key_join_arrow` on `encoded_filename`, which
    refuses a ref-only key, a missing key and a non-unique right side — so a partial
    or broadcasting join aborts instead of silently producing wrong rows.
    """
    if not a.keys_in:
        print("FATAL: --from-features requires --keys-in", flush=True)
        return 2
    keys = pq.read_table(a.keys_in)
    feats = pq.read_table(a.from_features)
    if a.features_key_col != "encoded_filename":
        feats = feats.rename_columns(
            ["encoded_filename" if c == a.features_key_col else c for c in feats.column_names])
    feat_cols = [f"f{i}" for i in range(a.n_feat)]
    missing = [c for c in feat_cols if c not in feats.column_names]
    if missing:
        print(f"FATAL: features table lacks {len(missing)} of {a.n_feat} columns "
              f"(first: {missing[:4]})", flush=True)
        return 2
    if a.features_key_col == "__rowid__":
        # ROW-ALIGNED mode. `v2_ab_extract` preserves its pairs-TSV row order (R1b
        # §8.5a — it is the only v1/944 extractor that does), and the pairs TSV is
        # emitted from the key table's own row order, so position IS the key here
        # and is STRONGER than a name join: it identifies the exact (ref, dist)
        # pair, not a filename that could repeat. The gate is correspondingly
        # strict — BOTH the reference-basename sequence AND the target must match
        # row for row, or the assembly aborts.
        if feats.num_rows != keys.num_rows:
            print(f"G-J FAIL: features {feats.num_rows} rows, keys {keys.num_rows}",
                  flush=True)
            return 2
        # v2_ab_extract writes the reference STEM (`I01.png` -> `I01`), while the
        # key table carries the full rendition filename, whose "extension" may be a
        # scale token (`o_9158.png.scale85x128.png`). Strip only a real image
        # extension, on both sides, so the comparison is well defined.
        img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif", ".jxl"}

        def _norm(x: str) -> str:
            q = Path(str(x)).name
            return Path(q).stem if Path(q).suffix.lower() in img_ext else q

        kb = [_norm(x) for x in keys["ref_basename"].to_pylist()]
        fb = [_norm(x) for x in feats["ref_basename"].to_pylist()]
        if kb != fb:
            bad = next(i for i, (x, y) in enumerate(zip(kb, fb)) if x != y)
            print(f"G-J FAIL: ref_basename differs at row {bad}: {kb[bad]!r} vs {fb[bad]!r}",
                  flush=True)
            return 2
        yk = np.asarray(keys["human_score"].combine_chunks(), dtype=np.float64)
        yf = np.asarray(feats["human_score"].combine_chunks(), dtype=np.float64)
        if not np.allclose(yk, yf, rtol=0, atol=1e-9):
            print(f"G-J FAIL: human_score differs, max|d| {np.abs(yk - yf).max():.3e}",
                  flush=True)
            return 2
        print(f"G-J PASS (row-aligned): {keys.num_rows} rows, ref_basename sequence "
              f"equal, target max|d| {np.abs(yk - yf).max():.3e}", flush=True)
        joined = feats.select(feat_cols)
    else:
        feats = feats.select(["encoded_filename"] + feat_cols)
        joined = safe_key_join_arrow(keys.select(["encoded_filename"]), feats,
                                     join_keys=["encoded_filename"], bring=feat_cols)
    if joined.num_rows != keys.num_rows:
        print(f"FATAL: join produced {joined.num_rows} rows for {keys.num_rows} keys", flush=True)
        return 2
    probe = np.asarray(joined[feat_cols[0]].combine_chunks().to_numpy(zero_copy_only=False),
                       dtype=np.float64)
    n_null = int(np.isnan(probe).sum()) + int(joined[feat_cols[0]].null_count)
    if n_null:
        print(f"FATAL: {n_null} key rows have no features row (100% resolution required)",
              flush=True)
        return 2
    out_cols = {"ref_basename": keys["ref_basename"],
                "human_score": keys["human_score"],
                "encoded_filename": keys["encoded_filename"]}
    if a.regime:
        out_cols["regime"] = pa.array([a.regime] * keys.num_rows)
    for c in feat_cols:
        out_cols[c] = joined[c]
    full = pa.table(out_cols)
    # G-R1: the pool block must be live for a *pools regime and zero for a folded one.
    pool = np.stack([np.asarray(full[f"f{i}"].combine_chunks(), dtype=np.float64)
                     for i in range(156, min(372, a.n_feat))], axis=1)
    n_live = int((np.abs(pool).max(axis=0) > 0).sum())
    if a.regime and a.regime.endswith("pools") and n_live != pool.shape[1]:
        print(f"G-R1 FAIL: regime {a.regime} but only {n_live}/{pool.shape[1]} pool slots live",
              flush=True)
        return 3
    if a.regime and not a.regime.endswith("pools") and n_live != 0:
        print(f"G-R1 FAIL: regime {a.regime} but {n_live} pool slots are non-zero", flush=True)
        return 3
    print(f"G-R1 OK: {n_live}/{pool.shape[1]} f156-371 slots live (regime {a.regime})")
    out = Path(a.out)
    md = {b"zensim_regime": a.regime.encode()} if a.regime else None
    if md:
        full = full.replace_schema_metadata({**(full.schema.metadata or {}), **md})
    pq.write_table(full, out, compression="zstd", compression_level=7)
    manifest = {"file": str(out), "sha256": sha256_file(out), "rows": full.num_rows,
                "n_feat": a.n_feat, "regime": a.regime, "build_commit": _commit(),
                "mechanism": "assembled from key sidecar + features table via "
                             "join_safety.safe_key_join_arrow on encoded_filename; row order "
                             "and target from the KEY table",
                "keys_in": {"path": a.keys_in, "sha256": sha256_file(Path(a.keys_in)),
                            "rows": keys.num_rows},
                "features_in": {"path": a.from_features,
                                "sha256": sha256_file(Path(a.from_features)),
                                "rows": feats.num_rows}}
    mp = out.with_suffix(".parquet._MANIFEST.json")
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out} ({out.stat().st_size} B) + {mp}")
    return 0


BAND_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]


def stratified_slice(a) -> int:
    """The W-LIN round-7b FULL-RANGE hf cut (`benchmarks/wlin_round7_rawframe_2026-08-30.md`
    §13.1) — a RE-CUT of an existing leg, never a re-extraction.

    Round 7's `tbig_hf` kept only `human_score >= 0.90`, so the head it fed had
    never seen a low-quality encode and its raw output saturated (measured: raw
    span 0.366 on the dial grid against 0.75-0.84 for every full-range head).
    This cut keeps the hf band whole AND gives every lower band real mass:

        keep ALL rows with human_score >= --strat-keep-above   (may be unset)
        plus up to --strat-per-band rows from each band of BAND_EDGES below it

    Selection inside a band is a deterministic global stride (`idx[::step]`), the
    same mechanism the leg itself is built with, so the cut carries NO RNG and is
    reproducible from the flags alone.
    """
    t = pq.read_table(a.band_from)
    y = np.asarray(t["human_score"].combine_chunks(), dtype=np.float64)
    keep: list[int] = []
    rows_desc = []
    hi_cut = a.strat_keep_above
    for i in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        idx = np.flatnonzero((y >= lo) & (y < hi))
        if hi_cut is not None and lo >= hi_cut:
            sel = idx                      # the hf band is kept whole
        elif len(idx) > a.strat_per_band > 0:
            step = max(1, len(idx) // a.strat_per_band)
            sel = idx[::step][: a.strat_per_band]
        else:
            sel = idx
        keep.extend(sel.tolist())
        rows_desc.append({"band": [lo, hi], "available": int(len(idx)), "kept": int(len(sel))})
        print(f"  band [{lo:.2f},{hi:.2f}) available {len(idx):7d} kept {len(sel):7d}", flush=True)
    keep_arr = np.array(sorted(keep), dtype=np.int64)
    out_t = t.take(pa.array(keep_arr))
    yk = np.asarray(out_t["human_score"].combine_chunks(), dtype=np.float64)
    print(f"stratified cut: {out_t.num_rows} of {t.num_rows} rows; target mean {yk.mean():.4f} "
          f"[{yk.min():.4f}, {yk.max():.4f}]", flush=True)
    outp = Path(a.out)
    pq.write_table(out_t, outp, compression="zstd", compression_level=7)
    man = {"file": str(outp), "sha256": sha256_file(outp), "rows": out_t.num_rows,
           "mode": "stratified-band-slice", "band_edges": BAND_EDGES,
           "strat_per_band": a.strat_per_band, "strat_keep_above": a.strat_keep_above,
           "mechanism": "deterministic global stride within each band; the band at/above "
                        "--strat-keep-above is kept whole. No RNG.",
           "bands": rows_desc, "target_mean": float(yk.mean()),
           "source": {"path": str(a.band_from), "sha256": sha256_file(Path(a.band_from)),
                      "rows": t.num_rows}}
    (outp.parent / (outp.name + "._MANIFEST.json")).write_text(json.dumps(man, indent=1))
    print(f"wrote {outp} + manifest")
    return 0


def band_slice(a) -> int:
    """The `tbig_hf` leg = the TOP TARGET BAND of the tbig leg, nothing else.

    VERIFIED, not assumed (2026-08-30): the set of `encoded_filename` in the
    registered `tbig_hf_954` leg is EXACTLY `{rows of tbig_954 with human_score
    >= 0.90}` — 11,941 of 192,714, set-equal, and not equal at 0.895. So the hf
    near-lossless leg is a band rule over the same corpus, not a separate one,
    and it is reproduced here by that rule rather than by re-deriving a cut.
    """
    t = pq.read_table(a.band_from)
    y = np.asarray(t["human_score"].combine_chunks(), dtype=np.float64)
    keep = np.flatnonzero(y >= a.band_min)
    out_t = t.take(pa.array(keep))
    print(f"band >= {a.band_min}: {len(keep)} of {t.num_rows} rows", flush=True)
    outp = Path(a.out)
    pq.write_table(out_t, outp, compression="zstd", compression_level=7)
    man = {"file": str(outp), "sha256": sha256_file(outp), "rows": out_t.num_rows,
           "mode": "band-slice", "band_min": a.band_min,
           "mechanism": "rows of --band-from with human_score >= --band-min; "
                        "verified set-equal to the registered tbig_hf_954 leg at 0.90",
           "source": {"path": str(a.band_from), "sha256": sha256_file(Path(a.band_from)),
                      "rows": t.num_rows}}
    (outp.parent / (outp.name + "._MANIFEST.json")).write_text(json.dumps(man, indent=1))
    print(f"wrote {outp} + manifest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root")
    ap.add_argument("--suffix", default="_944", help="view filename suffix (train<suffix>.parquet)")
    ap.add_argument("--n-feat", type=int, default=944)
    ap.add_argument("--verify-against", help="reference slice for G-T1/G-T2 (the 924 file)")
    ap.add_argument("--emit-keys", action="store_true",
                    help="also write keys_<out-stem>.parquet: the SAME rows in the SAME order "
                         "with the identity columns (origin_id/ref_filename/encoded_filename/"
                         "codec/q/knob_tuple_json/score_ssim2/score_zensim) carried through")
    ap.add_argument("--keys-only", action="store_true",
                    help="write ONLY the key sidecar (no feature columns read at all) — usable "
                         "against a views root whose regime has no feature table yet")
    ap.add_argument("--from-features",
                    help="assemble the leg at a NEW regime: join this features parquet onto "
                         "--keys-in by encoded_filename (safe_key_join_arrow, 100%% required), "
                         "preserving the key table's row order")
    ap.add_argument("--keys-in", help="key sidecar for --from-features")
    ap.add_argument("--features-key-col", default="encoded_filename",
                    help="column in --from-features carrying the encoded_filename value; "
                         "'__rowid__' selects ROW-ALIGNED assembly (position is the key, "
                         "gated on the ref_basename sequence AND the target)")
    ap.add_argument("--regime", help="regime tag written as a column + parquet metadata")
    ap.add_argument("--band-from", help="build the tbig_hf leg: the top target band of this leg")
    ap.add_argument("--strat-per-band", type=int, default=0,
                    help="round-7b full-range cut: max rows kept per band below "
                         "--strat-keep-above (deterministic stride; 0 = plain --band-min cut)")
    ap.add_argument("--strat-keep-above", type=float, default=None,
                    help="keep EVERY row at or above this target (the hf band, kept whole)")
    ap.add_argument("--band-min", type=float, default=0.90,
                    help="band floor on human_score (0.90 reproduces the registered "
                         "tbig_hf leg set-exactly)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.keys_only:
        a.emit_keys = True
    if a.from_features:
        return assemble_from_features(a)
    if a.band_from and (a.strat_per_band or a.strat_keep_above is not None):
        return stratified_slice(a)
    if a.band_from:
        return band_slice(a)
    if not a.views_root:
        print("FATAL: --views-root required (or use --from-features / --band-from)", flush=True)
        return 2

    feat_cols = [f"f{i}" for i in range(a.n_feat)]
    parts = []
    key_parts = []
    prov = []
    for ds in VIEWS:
        p = Path(a.views_root) / ds / f"train{a.suffix}.parquet"
        n = pq.read_metadata(p).num_rows
        step = n // TARGET_PER_VIEW
        idx = np.arange(0, n, step, dtype=np.int64)
        base = ["ref_filename", "encoded_filename", "score_ssim2"]
        cols = base if a.keys_only else base + feat_cols
        if a.emit_keys:
            have = set(pq.read_schema(p).names)
            cols = cols + [c for c in KEY_COLS if c in have and c not in cols]
        t = pq.read_table(p, columns=cols).take(pa.array(idx))
        hs = pc.min_element_wise(
            pc.max_element_wise(pc.divide(t["score_ssim2"], 100.0), 0.0), 1.0
        )
        out_cols = {
            "ref_basename": t["ref_filename"],
            "human_score": pc.cast(hs, pa.float64()),
            "encoded_filename": t["encoded_filename"],
        }
        if a.emit_keys:
            kc = {"ref_basename": t["ref_filename"],
                  "human_score": pc.cast(hs, pa.float64()),
                  "view": pa.array([ds] * len(idx)),
                  "view_row": pa.array(idx)}
            for c in KEY_COLS:
                if c in t.column_names:
                    kc[c] = t[c]
            key_parts.append(pa.table(kc))
        if not a.keys_only:
            for c in feat_cols:
                out_cols[c] = t[c]
            parts.append(pa.table(out_cols))
        prov.append({"view": str(p), "sha256": sha256_file(p), "n_rows": n, "step": step,
                     "picked": len(idx)})
        print(f"  {ds}: n={n} step={step} picked={len(idx)}", flush=True)

    if a.emit_keys:
        keys = pa.concat_tables(key_parts)
        kp = Path(a.out).parent / ("keys_" + Path(a.out).stem + ".parquet")
        pq.write_table(keys, kp, compression="zstd", compression_level=7)
        print(f"wrote key sidecar {kp} ({keys.num_rows} rows)", flush=True)
        if a.keys_only:
            km = {"file": str(kp), "sha256": sha256_file(kp), "rows": keys.num_rows,
                  "kind": "key-sidecar", "build_commit": _commit(),
                  "mechanism": "same global stride as the feature leg; identity columns carried",
                  "views": prov}
            kmp = kp.with_suffix(".parquet._MANIFEST.json")
            kmp.write_text(json.dumps(km, indent=1))
            print(f"wrote {kmp}")
            return 0

    full = pa.concat_tables(parts)
    print(f"total rows: {full.num_rows}")

    if a.verify_against:
        ref = pq.read_table(a.verify_against)
        assert full.num_rows == ref.num_rows, (
            f"G-T1 FAIL: rows {full.num_rows} != ref {ref.num_rows}")
        rb_new = full["ref_basename"].combine_chunks()
        rb_old = ref["ref_basename"].combine_chunks()
        assert rb_new.equals(rb_old), "G-T1 FAIL: ref_basename sequence differs"
        hs_new = np.asarray(full["human_score"].combine_chunks(), dtype=np.float64)
        hs_old = np.asarray(ref["human_score"].combine_chunks(), dtype=np.float64)
        assert (hs_new.view(np.int64) == hs_old.view(np.int64)).all(), (
            "G-T1 FAIL: human_score bit patterns differ")
        n_ref_feat = sum(1 for c in ref.schema.names if c.startswith("f") and c[1:].isdigit())
        mism = []
        for i in range(n_ref_feat):
            c = f"f{i}"
            x = np.asarray(full[c].combine_chunks(), dtype=np.float64).view(np.int64)
            y = np.asarray(ref[c].combine_chunks(), dtype=np.float64).view(np.int64)
            if not (x == y).all():
                mism.append(c)
        assert not mism, f"G-T2 FAIL: {len(mism)} feature cols differ vs ref: {mism[:8]}"
        print(f"G-T1 + G-T2 PASS: keys + f0..f{n_ref_feat - 1} bitwise-identical "
              f"to {a.verify_against}")

    out = Path(a.out)
    pq.write_table(full, out, compression="zstd", compression_level=7)
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    manifest = {
        "file": str(out), "sha256": sha256_file(out), "rows": full.num_rows,
        "n_feat": a.n_feat, "build_commit": commit,
        "mechanism": "global-row-stride step=n//50000 per lossy TRAIN view; "
                     "human_score=clip(score_ssim2/100,0,1); row-identical to "
                     "tbig_924_200k (G-T1/G-T2 gated when --verify-against)",
        "views": prov,
        "verified_against": a.verify_against,
    }
    mp = out.with_suffix(".parquet._MANIFEST.json")
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out} ({out.stat().st_size} B) + {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
