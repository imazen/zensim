#!/usr/bin/env python3
"""Stage the 196,086-row "big safesyn leg" for wave-r4 (era-2 x radius-4).

WHY THIS EXISTS
---------------
The wave root's `ext_safesyn_full.parquet` is 111,068 rows = exactly the
JPEG-family subset of safesyn (mozjpeg-rs-420-e4 + zenjpeg-420-e2 +
zenjpeg-420-xyb-e2). The hybrid lane measured that the 156-student class is
LEG-LIMITED, not method-limited: 111k rows -> CID22 0.8139, 196k rows ->
0.8643, same recipe / slice / lambda family (hybrid_candidate_2026-09-01.md
Sec 12.1, Sec 15 item 1). The missing 85,018 rows are the avif/jxl/webp
families.

They are missing for a MECHANICAL reason, not a data reason: every safesyn
`dist_path` points at a `q<N>.png` in the ~403 GiB decode cache that was
deleted 2026-06-22 (zensim/CLAUDE.md). The BITSTREAMS survive beside them.
`v2_ab_extract` decodes through `zen_io` = zenpng/zenjpeg/zenbitmaps only, so
it reads `.jpg` directly but cannot read `.avif`/`.jxl`/`.webp`. Those 85,018
rows therefore need a decode-to-PNG step first.

WHAT THIS SCRIPT IS (and is NOT)
--------------------------------
It is a THIN STAGER. It writes no pixels and implements no decode. The decode
owner is `zensim-bench/examples/verify_bitstream_decode --decode-list`
(added 2026-08-30 for the R1b keyed rebuild, which reuses that file's per-codec
zencodec decode functions verbatim -- "no second decode path").

That owner writes `<out_dir>/<input file stem>.png`, keyed on the input file
STEM ONLY. R1b's inputs were content-hash-named so their stems were globally
unique. Ours are `q5`, `q10`, ... -- 85,018 inputs would collapse onto ~30
output files. So this script builds a SYMLINK FARM with globally-unique names
(`<ref_stem>__<codec_dir>__q<N>.<ext>`) pointing at the real bitstreams, and
feeds those symlink paths to the owner. `file_stem()` is taken from the path
as given (not the resolved target) and the decoder is chosen by EXTENSION, so
the owner runs verbatim, unmodified, and emits the flat globally-unique
`bytes/decoded/` layout that r1b-pools944 and wlin7-pools944 already use.

THE JOIN (stated, gated, never best-effort)
-------------------------------------------
`canonical-2026-05-21/train/safesyn.parquet` carries `ref_basename` +
`human_score` but NO distorted-side identifier, so it cannot be key-joined to a
pairs list on its own. The attachment is POSITIONAL against
`2026-05-18-extfeat/safesyn_pairs.tsv`, which is the pairs file that parquet
was extracted from, gated row-wise on `ref_basename`:

    MEASURED: 196086 / 196086 rows match splitext(basename(ref_path))[0]
    MEASURED: dist_path is unique across all 196086 rows (a real key)
    MEASURED: (ref_basename, human_score) is ALSO unique here (0 duplicates)

This is the same positional-attachment-under-a-row-wise-gate pattern the wave
registered in Sec 2.1.1. The gate is re-asserted on the full table every run and
ABORTS rather than padding or partially covering. NOTE the ordering trap: the
other 196,086-row pairs file (`zensim-eval/safesyn_cvvdp_pairs_2026-05-17.tsv`)
holds the SAME pair set in a DIFFERENT order (it starts at `_512sq`, the
parquet and this file start at `_1022x818`), so it is 0/196086 positionally and
must not be substituted.

Outputs (all under the wave root's `bytes/` + `pairs/`):
  bytes/links/<uniq>.<ext>     symlinks -> real bitstreams (non-JPEG only)
  pairs/decode_list_safesyn_big.tsv   input list for the decode owner
  pairs/pairs_safesyn_big.tsv  ref_path<TAB>dist_path<TAB>human_score, 196086
  pairs/keymap_safesyn_big.tsv provenance per row (idx, orig png, final dist)
  pairs/_MANIFEST_stage.json   counts + gate results + sha256s
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

# ---------------------------------------------------------------- constants
PARQUET = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet")
PAIRS = Path("/mnt/v/zen/zensim-training/2026-05-18-extfeat/safesyn_pairs.tsv")
ROOT = Path(os.environ.get("BIGLEG_ROOT", "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01"))
# The parent lane's 111k JPEG pairs file, used only as a CROSS-CHECK of the
# human_score column (it is NOT this script's source of truth).
PARENT_111K = Path("/mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv")

EXPECT_ROWS = 196086

# codec directory -> bitstream extension. Verified by directory listing:
# the three JPEG families read directly through zen_io; the other three do not.
CODEC_EXT = {
    "mozjpeg-rs-420-e4": "jpg",
    "zenjpeg-420-e2": "jpg",
    "zenjpeg-420-xyb-e2": "jpg",
    "zenavif-s5-e6": "avif",
    "zenjxl-e7": "jxl",
    "zenwebp-default-m4": "webp",
}
JPEG_EXTS = {"jpg"}


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    bytes_dir = ROOT / "bytes"
    links_dir = bytes_dir / "links"
    decoded_dir = bytes_dir / "decoded"
    pairs_dir = ROOT / "pairs"
    for d in (links_dir, decoded_dir, pairs_dir):
        d.mkdir(parents=True, exist_ok=True)

    gates: dict[str, object] = {}

    # ---- load the label side -------------------------------------------
    tbl = pq.read_table(PARQUET, columns=["ref_basename", "human_score"])
    ref_basename = tbl.column("ref_basename").to_pylist()
    human_score = tbl.column("human_score").to_pylist()
    if len(ref_basename) != EXPECT_ROWS:
        die(f"{PARQUET} has {len(ref_basename)} rows, expected {EXPECT_ROWS}")

    # ---- load the pairs side -------------------------------------------
    refs: list[str] = []
    dists: list[str] = []
    with open(PAIRS, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            refs.append(row["ref_path"])
            dists.append(row["dist_path"])
    if len(refs) != EXPECT_ROWS:
        die(f"{PAIRS} has {len(refs)} rows, expected {EXPECT_ROWS}")

    # ---- GATE 1: positional ref_basename agreement (hard) ---------------
    ok = sum(
        1
        for i in range(EXPECT_ROWS)
        if os.path.splitext(os.path.basename(refs[i]))[0] == ref_basename[i]
    )
    gates["gate1_positional_ref_basename"] = f"{ok}/{EXPECT_ROWS}"
    if ok != EXPECT_ROWS:
        die(f"positional ref_basename gate {ok}/{EXPECT_ROWS} -- refusing to attach labels")

    # ---- GATE 2: dist_path is a real key --------------------------------
    n_uniq = len(set(dists))
    gates["gate2_dist_path_unique"] = f"{n_uniq}/{EXPECT_ROWS}"
    if n_uniq != EXPECT_ROWS:
        die(f"dist_path not unique ({n_uniq}/{EXPECT_ROWS})")

    # ---- resolve every distorted side to a real bitstream ---------------
    missing: list[str] = []
    per_codec: dict[str, int] = {}
    rows: list[tuple[str, str, float, str, str]] = []  # ref, final_dist, hs, codec, orig
    link_specs: list[tuple[Path, Path]] = []  # (link, target)

    for i in range(EXPECT_ROWS):
        dp = dists[i]
        parts = dp.split("/")
        codec = parts[-2]
        qstem = os.path.splitext(parts[-1])[0]  # "q5"
        ref_stem = parts[-3]
        ext = CODEC_EXT.get(codec)
        if ext is None:
            die(f"row {i}: unknown codec dir {codec!r}")
        bitstream = Path(dp).with_suffix("." + ext)
        if not bitstream.is_file():
            missing.append(str(bitstream))
            if len(missing) > 40:
                break
            continue
        per_codec[codec] = per_codec.get(codec, 0) + 1
        if ext in JPEG_EXTS:
            final = bitstream  # zen_io reads JPEG natively
        else:
            uniq = f"{ref_stem}__{codec}__{qstem}"
            link = links_dir / f"{uniq}.{ext}"
            link_specs.append((link, bitstream))
            final = decoded_dir / f"{uniq}.png"
        rows.append((refs[i], str(final), human_score[i], codec, dp))

    gates["gate3_bitstreams_present"] = f"{len(rows)}/{EXPECT_ROWS}"
    if missing:
        for m in missing[:10]:
            print(f"  MISSING {m}", file=sys.stderr)
        die(f"{len(missing)}+ bitstreams missing -- refusing partial coverage")
    if len(rows) != EXPECT_ROWS:
        die(f"resolved {len(rows)} of {EXPECT_ROWS}")

    # ---- GATE 4: unique output names (the whole point of the farm) ------
    finals = [r[1] for r in rows]
    if len(set(finals)) != EXPECT_ROWS:
        die(f"final dist paths not unique ({len(set(finals))}/{EXPECT_ROWS}) -- stem collision")
    gates["gate4_final_dist_unique"] = f"{len(set(finals))}/{EXPECT_ROWS}"

    n_jpeg = sum(1 for r in rows if CODEC_EXT[r[3]] in JPEG_EXTS)
    n_decode = EXPECT_ROWS - n_jpeg
    gates["n_jpeg_direct"] = n_jpeg
    gates["n_needing_decode"] = n_decode
    gates["per_codec"] = per_codec

    # ---- write the symlink farm ----------------------------------------
    made = 0
    for link, target in link_specs:
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == str(target):
                continue
            link.unlink()
        link.symlink_to(target)
        made += 1
    gates["symlinks_made"] = made
    gates["symlinks_total"] = len(link_specs)

    # ---- write the decode list -----------------------------------------
    decode_list = pairs_dir / "decode_list_safesyn_big.tsv"
    with open(decode_list, "w") as f:
        for link, _ in link_specs:
            f.write(f"{link}\n")

    # ---- the XYB-corrected variant --------------------------------------
    # MEASURED this session against the 16 surviving q<N>.png cache files that
    # escaped the 2026-06-22 deletion (the only ground truth left for safesyn):
    # reading a `zenjpeg-420-xyb-e2` bitstream DIRECTLY through zen_io does not
    # apply the XYB->sRGB inverse, so the colorspace error (~0.112 in f0) buries
    # the codec-quality signal (~0.0003 at q20/q50, i.e. ~370x smaller). The
    # q-ladder collapses: f0 spans only 0.1123..0.1145 across q5..q100 and is
    # not monotone (q100 = 0.1142 scores WORSE than q10 = 0.1124). Routing the
    # same bitstream through the zencodec decode owner instead lands within
    # max|d| 0.076 of the original cache versus 0.966 for the direct read --
    # 12.7x closer -- and restores the ladder.
    #
    # So this emits a SECOND pairs TSV, identical to the primary except the
    # 28,182 XYB rows point at decoded PNGs. The primary stays exactly as
    # briefed so the JPEG-subset agreement gate against the 111k leg stays
    # meaningful; this variant is the one to train on.
    xyb_codec = "zenjpeg-420-xyb-e2"
    xyb_links: list[tuple[Path, Path]] = []
    fixed_rows: list[tuple[str, str, float]] = []
    for ref, final, hs, codec, orig in rows:
        if codec == xyb_codec:
            parts = orig.split("/")
            uniq = f"{parts[-3]}__{codec}__{os.path.splitext(parts[-1])[0]}"
            link = links_dir / f"{uniq}.jpg"
            xyb_links.append((link, Path(final)))
            final = str(decoded_dir / f"{uniq}.png")
        fixed_rows.append((ref, final, hs))
    made_x = 0
    for link, target in xyb_links:
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == str(target):
                continue
            link.unlink()
        link.symlink_to(target)
        made_x += 1
    gates["xybfix_rows"] = len(xyb_links)
    gates["xybfix_symlinks_made"] = made_x
    if len({r[1] for r in fixed_rows}) != EXPECT_ROWS:
        die("xybfix final dist paths not unique")

    xyb_list = pairs_dir / "decode_list_safesyn_xybfix.tsv"
    with open(xyb_list, "w") as f:
        for link, _ in xyb_links:
            f.write(f"{link}\n")
    pairs_fixed = pairs_dir / "pairs_safesyn_bigfix.tsv"
    with open(pairs_fixed, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for ref, final, hs in fixed_rows:
            f.write(f"{ref}\t{final}\t{hs!r}\n")

    # ---- write the pairs TSV (input order preserved) --------------------
    pairs_out = pairs_dir / "pairs_safesyn_big.tsv"
    with open(pairs_out, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for ref, final, hs, _codec, _orig in rows:
            f.write(f"{ref}\t{final}\t{hs!r}\n")

    keymap = pairs_dir / "keymap_safesyn_big.tsv"
    with open(keymap, "w") as f:
        f.write("idx\tref_basename\thuman_score\tcodec\torig_dist_png\tfinal_dist\n")
        for i, (ref, final, hs, codec, orig) in enumerate(rows):
            f.write(
                f"{i}\t{os.path.splitext(os.path.basename(ref))[0]}\t{hs!r}\t{codec}\t{orig}\t{final}\n"
            )

    # ---- CROSS-CHECK: our human_score vs the parent lane's 111k TSV -----
    # Reported, never enforced: the parent's leg is a different label vintage.
    xcheck: dict[str, object] = {"status": "not-run"}
    if PARENT_111K.is_file():
        parent: dict[str, float] = {}
        with open(PARENT_111K, newline="") as f:
            rd = csv.DictReader(f, delimiter="\t")
            for row in rd:
                parent[row["dist_path"]] = float(row["human_score"])
        deltas = []
        for ref, final, hs, codec, _orig in rows:
            if CODEC_EXT[codec] in JPEG_EXTS and final in parent:
                deltas.append(abs(parent[final] - hs))
        if deltas:
            deltas.sort()
            xcheck = {
                "status": "measured",
                "matched_rows": len(deltas),
                "parent_rows": len(parent),
                "max_abs_delta": deltas[-1],
                "median_abs_delta": deltas[len(deltas) // 2],
                "n_exact": sum(1 for d in deltas if d == 0.0),
                "note": (
                    "our human_score comes from canonical-2026-05-21/train/safesyn.parquet "
                    "(the task's named source); the parent lane's 111k TSV is a different "
                    "label vintage. Reported, not enforced."
                ),
            }
    gates["human_score_xcheck_vs_parent_111k"] = xcheck

    manifest = {
        "description": "wave-r4 A6 big safesyn leg staging (196,086 rows, era-2 x radius-4)",
        "built_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "label_source": str(PARQUET),
        "pairs_source": str(PAIRS),
        "join": "POSITIONAL, gated row-wise on ref_basename (see module docstring)",
        "decode_owner": (
            "zensim-bench/examples/verify_bitstream_decode --decode-list "
            "(features verify-all); this script writes no pixels"
        ),
        "outputs": {
            "pairs_tsv": str(pairs_out),
            "decode_list": str(decode_list),
            "keymap": str(keymap),
            "links_dir": str(links_dir),
            "decoded_dir": str(decoded_dir),
            "pairs_tsv_xybfix": str(pairs_fixed),
            "decode_list_xybfix": str(xyb_list),
        },
        "gates": gates,
    }
    manifest["outputs_sha256"] = {
        "pairs_tsv": sha256_file(pairs_out),
        "decode_list": sha256_file(decode_list),
        "pairs_tsv_xybfix": sha256_file(pairs_fixed),
        "decode_list_xybfix": sha256_file(xyb_list),
    }
    with open(pairs_dir / "_MANIFEST_stage.json", "w") as f:
        json.dump(manifest, f, indent=1)

    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
