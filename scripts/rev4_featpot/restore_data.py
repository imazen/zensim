"""Pinned sidecar join for the Rev4 POTENTIAL candidate and restore arms.

POTENTIAL - ceiling, not a model score. Arms add columns to the Rev3 944 bank; the
added columns are packed after f943 as f944+ (`candidate_data` delegates every arm
name below to this module). Two pinned sources, each hash-checked at every load:

* restore-cuts sidecars (`rev4_featpot_restore_arms_2026-09-25.json`): mapdev, z1max,
  gmsnative, dvifmgate (f1502-f1824);
* Part B sidecars in the promoted bank (`rev4_featpot_partb_arms_2026-09-25.json`):
  csfw_dvifm (f944-f985), c1c4 (f986-f1321), gmsbank (f1322-f1501), plus the
  reviewed P2 peer columns (gmsd, gmsm) for arm p3, negative canonical ids -1, -2.

`side_join` reads no label; `load` adds role-allowed labels through the baseline
loader after the pins verify. Identical-pair convention and permuted controls:
`apply_identical_convention` (amendment rev4_featpot_identical_pair_amendment).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from admit_bank import OUT as ADMITTED
from data import FEATURES as BANK_FEATURES, load as load_bank


REPO = Path(__file__).resolve().parents[2]
RESTORE_JSON = REPO / "benchmarks/rev4_featpot_restore_arms_2026-09-25.json"
RESTORE_SHA256 = "6b920f2a619a105acea5915f563b57a968a1f635ac6598a60d7ccea41529251c"
PARTB_JSON = REPO / "benchmarks/rev4_featpot_partb_arms_2026-09-25.json"
PARTB_SHA256 = "9cfaecd402b628113e4f18115957e9d21b81b1f66ede7d19dd45ee3be1b302a9"
# family -> (source, first id, end id)
FAMILIES = {"mapdev": ("restore", 1502, 1562), "z1max": ("restore", 1562, 1790),
            "gmsnative": ("restore", 1790, 1820), "dvifmgate": ("restore", 1820, 1825),
            "csfw_dvifm": ("partb", 944, 986), "c1c4": ("partb", 986, 1322),
            "gmsbank": ("partb", 1322, 1502)}
# arm name -> (json source, key in that JSON's "arms")
ARM_KEYS = {"a1": ("restore", "A1"), "a1m": ("restore", "A1m"), "a1w": ("restore", "A1w"),
            "b2": ("restore", "B2"), "b2m": ("restore", "B2m"),
            **{name: ("partb", name) for name in
               ("c1", "c2", "c3", "c4", "all", "csfw", "c7", "p1", "p3", "b1", "b1s", "c8n", "rall")}}
PEER_IDS = (-1, -2)  # canonical ids for the peer columns gmsd, gmsm


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(source: str) -> dict:
    path, expected = (RESTORE_JSON, RESTORE_SHA256) if source == "restore" else (PARTB_JSON, PARTB_SHA256)
    if sha(path) != expected:
        raise ValueError(f"{path.name} differs from its committed pin")
    return json.loads(path.read_text())


def arm_ids(arm: str) -> list[int]:
    family = arm.removesuffix("_perm")
    if family not in ARM_KEYS:
        raise ValueError(f"{arm}: not a runnable candidate/restore arm")
    source, key = ARM_KEYS[family]
    spec = _json(source)["arms"][key]
    return list(spec["add"]) + (list(PEER_IDS) if spec.get("peer") else [])


def columns(arm: str) -> tuple[list[str], list[int]]:
    ids = arm_ids(arm)
    return BANK_FEATURES + [f"f{i}" for i in range(944, 944 + len(ids))], ids


def _family_of(feature_id: int) -> str:
    for family, (_, lo, hi) in FAMILIES.items():
        if lo <= feature_id < hi:
            return family
    raise ValueError(f"f{feature_id}: no pinned sidecar family")


def _pin(name: str, family: str) -> tuple[dict, dict]:
    source = FAMILIES[family][0]
    pins = _json(source)["sidecar_pins"].get(name)
    if pins is None:
        raise ValueError(f"{name}: absent from the {source} pin")
    return pins, pins["families"][family]


def preflight(name: str, ids: list[int]) -> dict:
    """Verify every pinned file the arm reads; returns {family: (path, expected_rows)}."""
    needed = sorted({_family_of(i) for i in ids if i >= 0})
    out = {}
    for family in needed:
        pins, fam = _pin(name, family)
        source, lo, hi = FAMILIES[family]
        path = Path(fam["file"])
        if sha(path) != fam["sha256"]:
            raise ValueError(f"{name}: {family} sidecar changed after pin")
        if source == "restore":
            manifest = path.parent / "_MANIFEST_restore.json"
            if sha(manifest) != pins["manifest_sha256"]:
                raise ValueError(f"{name}: restore manifest changed after pin")
            rows = pins["row_count"]
        else:
            manifest = path.parent / "_MANIFEST.json"
            if sha(manifest) != pins["manifest_sha256"] or sha(path.parent / "keys.parquet") != pins["keys_sha256"]:
                raise ValueError(f"{name}: promoted bank manifest or keys changed after pin")
            rows = pins["unique_pair_keys"]
        schema = pq.ParquetFile(path).schema_arrow
        if schema.names != ["pair_key"] + [f"f{i}" for i in range(lo, hi)]:
            raise ValueError(f"{name}: {family} column order/schema mismatch")
        for field in schema:
            if field.name != "pair_key" and str(field.type) != "float":
                raise ValueError(f"{name}: {field.name} is not f32")
        out[family] = (path, rows)
    return out


def side_join(name: str, ids: list[int]):
    """Pinned sidecar columns for the non-peer `ids`, one row per unique pair_key. No label read."""
    files = preflight(name, ids)
    frames = []
    expected = None
    for family, (path, rows) in files.items():
        take = [i for i in ids if i >= 0 and _family_of(i) == family]
        frame = pq.read_table(path, columns=["pair_key"] + [f"f{i}" for i in take]).to_pandas()
        if frame.pair_key.duplicated().any() or len(frame) != rows:
            raise ValueError(f"{name}: {family} sidecar key cardinality differs from pin")
        expected = rows
        frames.append(frame.set_index("pair_key"))
    side = frames[0].join(frames[1:], how="inner") if len(frames) > 1 else frames[0]
    if len(side) != expected:
        raise ValueError(f"{name}: sidecar families disagree on keys")
    side = side[[f"f{i}" for i in ids if i >= 0]]
    if not np.isfinite(side.to_numpy()).all():
        raise ValueError(f"{name}: nonfinite added feature")
    return side.reset_index()


def peer_side(name: str):
    """Reviewed P2 peer columns (gmsd, gmsm), one row per unique pair_key, verified as p2_data does."""
    import p2_data as p2
    from admit_bank import BANK, load_pinned_set
    manifest = json.loads((p2.PEER / "_MANIFEST.json").read_text())
    verification = json.loads((p2.PEER / "verification.json").read_text())
    if (sha(p2.PEER / "_MANIFEST.json") != p2.PEER_MANIFEST_SHA or
            sha(p2.PEER / "verification.json") != p2.PEER_VERIFY_SHA or
            manifest["binary_sha256"] != p2.PEER_BINARY_SHA or
            verification["parquet_key_score_mismatches"] != 0):
        raise ValueError("peer provenance changed")
    info = manifest["sets"][name]
    path = p2.PEER / f"{name}.parquet"
    bank = load_pinned_set(name)
    if (sha(path) != info["parquet_sha256"] or sha(BANK / name / "keys.parquet") != info["keys_sha256"] or
            not info["key_check"] or info["unique_pair_keys"] != bank["unique_pair_keys"]):
        raise ValueError(f"{name}: peer/bank source mismatch")
    peer = pq.read_table(path, columns=["pair_key", "gmsd", "gmsm"]).to_pandas()
    unique = peer.drop_duplicates("pair_key", keep="first")
    check = peer.merge(unique, on="pair_key", suffixes=("", "_first"), validate="many_to_one")  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    if (len(unique) != info["unique_pair_keys"] or not (check.gmsd == check.gmsd_first).all()
            or not (check.gmsm == check.gmsm_first).all()
            or not np.isfinite(unique[["gmsd", "gmsm"]].to_numpy()).all()):
        raise ValueError(f"{name}: peer scores inconsistent")
    return unique.reset_index(drop=True), {"peer_manifest_sha256": p2.PEER_MANIFEST_SHA,
                                            "peer_parquet_sha256": info["parquet_sha256"]}


def identical_flags(name: str):
    """pair_key -> pixels_identical from the admitted table (feature-side metadata, no label)."""
    table = pq.read_table(ADMITTED / f"POT_{name}_rev3_944.parquet",
                          columns=["pair_key", "pixels_identical"]).to_pandas()
    if table.groupby("pair_key").pixels_identical.nunique().max() > 1:
        raise ValueError(f"{name}: pixels_identical differs within a pair_key")
    return table.drop_duplicates("pair_key").set_index("pair_key").pixels_identical.astype(bool)


def _permute_within_reference(out, extra: list[str], selector, rng) -> None:
    for ref in sorted(out.ref_basename.unique()):
        sel = (out.ref_basename == ref).to_numpy() & selector
        subset = out.loc[sel, ["pair_key"] + extra].drop_duplicates("pair_key")
        keys = subset.pair_key.to_numpy()
        mapped = dict(zip(keys, subset[extra].to_numpy()[rng.permutation(len(keys))]))
        positions = out.index[sel]
        out.loc[positions, extra] = np.stack([mapped[k] for k in out.loc[positions, "pair_key"]])


def apply_identical_convention(out, extra: list[str], name: str, permute: bool,
                               peer_cols: list[str] | None = None):
    """Amendment identical-pair convention: fabricated zeros on pixels_identical keys.

    Zero-masks every added column in `extra` on identical rows, then (control arms)
    permutes them jointly within each reference over the NON-identical pair_keys only,
    seed 20260923, so the fabricated zeros never move. `peer_cols` (the reviewed P2
    peer pair) are not masked; their control is the registered P2 one: a joint key-level
    permutation over ALL rows within each reference, drawn after the added-column one.
    Returns (frame, receipt).
    """
    peer_cols = peer_cols or []
    flags = identical_flags(name)
    mask = out.pair_key.map(flags)
    if mask.isna().any():
        raise ValueError(f"{name}: pair_key missing from the pixels_identical table")
    mask = mask.to_numpy(dtype=bool)
    nonzero_before = int((out[extra].to_numpy()[mask] != 0).sum()) if extra else 0
    if extra:
        out.loc[mask, extra] = 0.0
    if permute:
        rng = np.random.default_rng(20260923)
        if extra:
            _permute_within_reference(out, extra, ~mask, rng)
        if peer_cols:
            _permute_within_reference(out, peer_cols, np.ones(len(out), dtype=bool), rng)
    return out, {"identical_rows": int(mask.sum()), "identical_keys": int(flags.sum()),
                 "added_cells_nonzero_before_mask": nonzero_before,
                 "peer_columns_unmasked": peer_cols,
                 "amendment": "rev4_featpot_identical_pair_amendment_2026-09-25"}


def load(name: str, arm: str, features: bool = True):
    cols, ids = columns(arm)
    sidecar_ids = [i for i in ids if i >= 0]
    files = preflight(name, ids)
    base, meta = load_bank(name, features=features)
    meta = dict(meta)
    meta.update({"candidate_arm": arm, "candidate_ids": ids,
                 "restore_arms_sha256": RESTORE_SHA256, "partb_arms_sha256": PARTB_SHA256,
                 "sidecar_sha256": {f: sha(p) for f, (p, _) in files.items()}})
    if not features:
        return base, meta
    out = base
    if sidecar_ids:
        side = side_join(name, ids)
        out = out.merge(side, on="pair_key", how="left", sort=False,  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
                        validate="many_to_one", indicator=True)
        if len(out) != len(base) or not (out._merge == "both").all():
            raise ValueError(f"{name}: sidecar join coverage mismatch")
        out = out.drop(columns="_merge")
    packed = {f"f{c}": f"f{944 + j}" for j, c in enumerate(ids) if c >= 0}
    peer_cols = []
    if any(i < 0 for i in ids):
        peer, peer_meta = peer_side(name)
        meta.update(peer_meta)
        names = [f"f{944 + j}" for j, c in enumerate(ids) if c < 0]
        peer = peer.rename(columns=dict(zip(["gmsd", "gmsm"], names)))
        out = out.merge(peer, on="pair_key", how="left", sort=False,  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
                        validate="many_to_one", indicator=True)
        if len(out) != len(base) or not (out._merge == "both").all():
            raise ValueError(f"{name}: peer join coverage mismatch")
        out = out.drop(columns="_merge")
        peer_cols = names
    out.index = np.arange(len(out))
    out = out.rename(columns=packed)
    added = [f"f{944 + j}" for j, c in enumerate(ids) if c >= 0]
    out, meta["identical_convention"] = apply_identical_convention(
        out, added, name, permute=arm.endswith("_perm"), peer_cols=peer_cols)
    meta["candidate_packed_columns"] = cols[944:]
    return out, meta


def bounds_tsv(arm: str) -> Path:
    """Sign-mask TSV for BVLS. Every added column is free (partb-arms amendment 2026-09-25):
    the run's mask is the pinned legacy f0-f371 file only, so this returns it unchanged."""
    columns(arm)  # validates the arm name
    return REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv"


def is_a1w(arm: str) -> bool:
    return arm.removesuffix("_perm") == "a1w"


def a1w_dropped(key: str) -> list[int]:
    """Label-free A1w drop list (60 R0 ids) for a fit population key, e.g. 'D1/kadid_train/o0',
    'D1/kadid_train/full' or 'D2/without_aic3'. Pinned through the restore arms JSON."""
    info = _json("restore")["drop_lists_file"]
    path = Path(info["path"])
    if not path.is_absolute():
        path = REPO / path
    if sha(path) != info["sha256"]:
        raise ValueError("A1w drop-list file differs from its pin")
    lists = json.loads(path.read_text())["drop_lists"]
    if key not in lists or len(lists[key]) != 60:
        raise ValueError(f"A1w: no 60-id drop list for {key}")
    return lists[key]


def a1w_slice(key: str, n_columns: int) -> Path:
    """Coordinate-slice file for bake_dial_refit --slice-file: every column except the drop list."""
    dropped = set(a1w_dropped(key))
    out = Path("/var/tmp/rev4-featpot/restore/slices") / (key.replace("/", "__") + f".w{n_columns}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(i) for i in range(n_columns) if i not in dropped) + "\n")
    return out
