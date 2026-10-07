"""Label adapter for the confirmatory read: ORIGINAL dataset manifests -> Rev4 `pair_key`, never by bank `row_id`.

The bank has no label file for a sealed set. The originals are TSV/CSV files with no `pair_key`; a row maps to a Rev4 key by
(ref_path, dist_path) against `<bank>/<set>/keys.parquet`, falling back to decoded-pixel hashes through an explicitly pinned
`path -> pixel sha256` table (built with the pinned decoder; Rev4 keys carry ref/dist pixel hashes). Accounting is strict:
  * every non-identical Rev4 key receives exactly `n_stimuli` label rows (collapsed stimuli need the pixel table),
  * label rows that map to no key must equal the identical stimuli not matched by path (0 where nothing is identical),
  * rows outside the pinned `select` rule are counted as unselected, never silently used; keys outside it are outside the
    read (the rule defines the population on both sides; their predictions are never paired with a label).
Validated ONLY on open sets (`python3 v2c_labels.py validate`): it must reproduce AIC-3's and KADID-SELECT's admitted
`labels__human.parquet` exactly. This module never opens a sealed label file by itself; the read passes pinned, sha-checked specs.

spec: {"path", "sha256", "format": "tsv"|"csv"|"json" (+ "rows_key"), "ref_col", "dist_col", "label_col",
       "select": {"ref_path_in": [...]} | {"ref_stem_in": [...]} | {"ref_number": {"regex", "modulus", "in"|"not_in": [...]}} | null,
       "usecols": [...] | null, "via_pairs": {"path", "sha256", "on": [[label_col, pairs_col], ...]} | null}
       # via_pairs: the label file carries no paths (MCL-JCI-style); it is keyed to a pinned pairs file by explicit columns
"""

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v2_common import sha


def _read(data: bytes, fmt: str, usecols: list[str] | None = None, rows_key: str | None = None) -> pd.DataFrame:
    """Parse only the retained bytes whose digest the caller checked."""
    if fmt == "json":  # {"<rows_key>": [ {col: value, ...}, ... ]}
        return pd.DataFrame(json.loads(data)[rows_key]).astype(str)
    return pd.read_csv(io.BytesIO(data), sep="\t" if fmt == "tsv" else ",", dtype=str, keep_default_na=False, usecols=usecols)


def _stem(path: str) -> str:
    return Path(path).stem


def select_mask(frame: pd.DataFrame, rule: dict | None) -> np.ndarray:
    if not rule:
        return np.ones(len(frame), dtype=bool)
    if "ref_path_in" in rule:
        return frame["ref_path"].isin(set(rule["ref_path_in"])).to_numpy()
    stems = frame["ref_path"].map(_stem)
    if "ref_stem_in" in rule:
        return stems.isin(set(rule["ref_stem_in"])).to_numpy()
    if "ref_number" in rule:
        r = rule["ref_number"]
        num = stems.str.extract(r["regex"])[0].astype(float)
        mod = num % r["modulus"]
        ok = mod.isin(r["in"]) if "in" in r else ~mod.isin(r["not_in"])
        return (ok & num.notna()).to_numpy()
    raise ValueError(f"unknown select rule {sorted(rule)}")


def load_label_rows(spec: dict, root_check: bool = True) -> pd.DataFrame:
    """Columns ref_path, dist_path, label (float), file_row; after the sha256 check."""
    path = Path(spec["path"])
    data = path.read_bytes()
    if root_check and hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"{path}: sha256 differs from the pin")
    frame = _read(data, spec["format"], spec.get("usecols"), spec.get("rows_key"))
    if spec.get("via_pairs"):
        vp = spec["via_pairs"]
        pp = Path(vp["path"])
        pair_data = pp.read_bytes()
        if hashlib.sha256(pair_data).hexdigest() != vp["sha256"]:
            raise ValueError(f"{pp}: sha256 differs from the pin")
        pairs = _read(pair_data, "tsv")
        left, right = [a for a, _ in vp["on"]], [b for _, b in vp["on"]]
        n = len(frame)
        pairs = pairs[[*dict.fromkeys([*right, "ref_path", "dist_path"])]]  # only the key columns: never a label column of the pairs file
        frame = frame.merge(pairs, left_on=left, right_on=right, how="left", validate="many_to_one", indicator=True)  # joinsafety-ok: label file -> pairs file by explicit pinned columns, coverage asserted
        if len(frame) != n or not (frame["_merge"] == "both").all():
            raise ValueError("label rows not covered by the pairs file")
        ref_col, dist_col = "ref_path", "dist_path"
    else:
        ref_col, dist_col = spec["ref_col"], spec["dist_col"]
    out = pd.DataFrame({"ref_path": frame[ref_col], "dist_path": frame[dist_col],
                        "label": frame[spec["label_col"]].astype(float), "file_row": np.arange(len(frame))})
    if not np.isfinite(out.label).all():
        raise ValueError("non-finite label value")
    return out


def adapt(rows: pd.DataFrame, keys: pd.DataFrame, rule: dict | None = None, pixel_sha: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Map label rows to Rev4 keys. Returns (frame[pair_key, label, file_row] for NON-identical keys, accounting)."""
    sel = select_mask(rows, rule)
    unselected = int((~sel).sum())
    rows = rows.loc[sel].reset_index(drop=True)
    by_path = {(r, d): i for i, (r, d) in enumerate(zip(keys.ref_path, keys.dist_path))}
    idx = np.array([by_path.get((r, d), -1) for r, d in zip(rows.ref_path, rows.dist_path)])
    via_pixels = 0
    if pixel_sha:
        by_pix = {(a, b): i for i, (a, b) in enumerate(zip(keys.ref_pixels_sha256, keys.dist_pixels_sha256))}
        for j in np.flatnonzero(idx < 0):
            a, b = pixel_sha.get(rows.ref_path[j]), pixel_sha.get(rows.dist_path[j])
            if a is not None and b is not None and (a, b) in by_pix:
                idx[j] = by_pix[(a, b)]
                via_pixels += 1
    matched = idx >= 0
    per_key = np.bincount(idx[matched], minlength=len(keys))
    ident = keys.pixels_identical.to_numpy().astype(bool)
    need = keys.n_stimuli.to_numpy()
    # The select rule defines the population on BOTH sides: a key whose reference the rule excludes is outside the read
    # (e.g. CID22-B(23): the bank holds 24 references, one of them a duplicate of a CID22-A picture; DATA_SPLITS 2026-09-22).
    in_pop = select_mask(keys, rule)
    if per_key[~in_pop].any():
        raise ValueError("a selected label row maps to a key outside the select rule")
    bad = np.flatnonzero(in_pop & ~ident & (per_key != need))
    if len(bad):
        raise ValueError(f"{len(bad)} non-identical keys do not receive n_stimuli label rows (first: {keys.pair_key[bad[0]]}, "
                         f"{per_key[bad[0]]} vs {need[bad[0]]}); collapsed stimuli need a pinned pixel-hash table")
    unmatched = int((~matched).sum())
    ident_unmatched = int(need[ident & in_pop].sum() - per_key[ident & in_pop].sum())
    if unmatched != ident_unmatched:
        raise ValueError(f"{unmatched} label rows match no Rev4 key but only {ident_unmatched} identical stimuli are unaccounted for")
    keep = matched & ~ident[np.where(matched, idx, 0)]
    out = pd.DataFrame({"pair_key": keys.pair_key.to_numpy()[idx[keep]], "label": rows.label.to_numpy()[keep],
                        "file_row": rows.file_row.to_numpy()[keep]})
    acct = {"label_rows": int(len(rows)), "unselected_rows": unselected, "matched_by_pixel_hash": via_pixels,
            "rows_on_identical_keys_or_unmatched_identical": int(len(rows) - keep.sum()), "keys": int(len(keys)),
            "identical_keys": int(ident.sum()), "rows_used": int(keep.sum()), "keys_outside_select": int((~in_pop).sum())}
    return out, acct


# ------------------------------------------------------------------ validation on OPEN sets only
OPEN = {
    "aic3": {"spec": {"path": "/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv", "format": "tsv", "ref_col": "ref_path",
                      "dist_col": "dist_path", "label_col": "human_score", "select": None},
             "members": ["aic3"]},
    # KADID-SELECT's admitted labels come from the ceiling INPUTS table (reference, distorted, target per row)
    "kadid_select": {"spec": {"path": str(Path.home() / "work/zensim-validation-2026-09-13/ceiling/final/INPUTS.json"), "format": "json",
                              "rows_key": "rows", "ref_col": "reference", "dist_col": "distorted", "label_col": "target",
                              "select": "from-keys"},
                     "members": ["kadid_select"]},
}
BANK = Path("/var/tmp/rev4-featbank-r4")


def validate(name: str) -> dict:
    """Reproduce an OPEN set's admitted labels from its original manifest. Refuses any other set."""
    if name not in OPEN:
        raise SystemExit(f"{name}: validation runs on open sets only ({sorted(OPEN)})")
    from admit_bank import BANK as OLD_BANK
    spec = dict(OPEN[name]["spec"])
    spec["sha256"] = sha(Path(spec["path"]))
    if spec.get("via_pairs"):
        spec["via_pairs"] = {**spec["via_pairs"], "sha256": sha(Path(spec["via_pairs"]["path"]))}
    keys = pq.read_table(BANK / name / "keys.parquet").to_pandas()
    if spec["select"] == "from-keys":
        spec["select"] = {"ref_path_in": sorted(set(keys.ref_path))}
    rows = load_label_rows(spec)
    got, acct = adapt(rows, keys, spec["select"])
    adm = pq.read_table(OLD_BANK / name / "labels__human.parquet").to_pandas()
    ident_keys = set(keys.pair_key[keys.pixels_identical])
    adm = adm.loc[~adm.pair_key.isin(ident_keys)]
    a = {k: sorted(g.human_score.tolist()) for k, g in adm.groupby("pair_key")}
    b = {k: sorted(g.label.tolist()) for k, g in got.groupby("pair_key")}
    same_sets = a == b
    # sequence check: source order of the admitted rows equals the file order of the adapter's rows
    seq = adm.sort_values("source_row_id").human_score.tolist() == got.sort_values("file_row").label.tolist()
    result = {"set": name, "spec_path": spec["path"], "spec_sha256": spec["sha256"], "accounting": acct,
              "keys_with_labels": len(b), "admitted_keys": len(a), "per_key_label_multisets_equal": same_sets,
              "file_order_sequence_equal": bool(seq), "exact": bool(same_sets and seq)}
    return result


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "validate":
        print(__doc__)
        return 2
    results = [validate(n) for n in sys.argv[2:]]
    print(json.dumps(results, indent=1))
    return 0 if all(r["exact"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
