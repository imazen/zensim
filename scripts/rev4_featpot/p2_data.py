"""Reviewed, label-free peer GMSD join for the registered P2 diagnostic arm."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from admit_bank import BANK, load_pinned_set
from data import FEATURES as BANK_FEATURES, load as load_bank


PEER = Path("/var/tmp/gmsbank/peer_gmsd")
ROOT = Path("/var/tmp/rev4-featpot")
PEER_MANIFEST_SHA = "8d2799c8a9ebaa3f25fa7518b3d7591ee7742b696d7f5592b9ace9974d0d27fc"
PEER_VERIFY_SHA = "5429a455120bd41a0c729514e61670c124b85be576dc2fc49a8a9cf5bf695f5f"
PEER_BINARY_SHA = "2ed8f6765e2f9a7bca5a4ea4a3b8b7b266fb10c57b5b0c9f5563f978c2108ea6"
FEATURES = BANK_FEATURES + ["gmsd", "gmsm"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str, arm: str = "p2", features: bool = True):
    if arm not in ("p2", "p2_perm"):
        raise ValueError(arm)
    if sha(PEER / "_MANIFEST.json") != PEER_MANIFEST_SHA or sha(PEER / "verification.json") != PEER_VERIFY_SHA:
        raise ValueError("peer manifest or verification receipt changed")
    manifest = json.loads((PEER / "_MANIFEST.json").read_text())
    verification = json.loads((PEER / "verification.json").read_text())
    if (manifest["schema"] != "gmsbank-peer-gmsd-v1" or
            manifest["binary_sha256"] != PEER_BINARY_SHA or
            verification["binary_sha256"] != PEER_BINARY_SHA or
            verification["parquet_key_score_mismatches"] != 0):
        raise ValueError("peer provenance or key verification mismatch")
    bank_manifest = load_pinned_set(name)
    peer_info = manifest["sets"][name]
    peer_path = PEER / f"{name}.parquet"
    if (sha(peer_path) != peer_info["parquet_sha256"] or
            sha(BANK / name / "keys.parquet") != peer_info["keys_sha256"] or
            not peer_info["key_check"] or peer_info["rows"] != bank_manifest["row_count"] or
            peer_info["unique_pair_keys"] != bank_manifest["unique_pair_keys"]):
        raise ValueError(f"{name}: peer/bank source mismatch")
    base, label_meta = load_bank(name, features=features)
    if not features:
        return base, label_meta
    peer = pq.read_table(peer_path, columns=["pair_key", "gmsd", "gmsm"]).to_pandas()
    if len(peer) != len(base):
        raise ValueError(f"{name}: peer row count mismatch")
    if peer[["gmsd", "gmsm"]].isna().any().any() or not np.isfinite(peer[["gmsd", "gmsm"]].to_numpy()).all():
        raise ValueError(f"{name}: nonfinite peer scores")
    unique = peer.drop_duplicates("pair_key", keep="first")
    if len(unique) != peer_info["unique_pair_keys"]:
        raise ValueError(f"{name}: peer unique-key count mismatch")
    joined_check = peer.merge(unique, on="pair_key", suffixes=("", "_first"), validate="many_to_one")  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    if not (joined_check.gmsd == joined_check.gmsd_first).all() or not (joined_check.gmsm == joined_check.gmsm_first).all():
        raise ValueError(f"{name}: duplicate key has nonidentical peer score")
    out = base.merge(unique, on="pair_key", how="left", sort=False, validate="many_to_one", indicator=True)  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    if len(out) != len(base) or not (out._merge == "both").all():
        raise ValueError(f"{name}: peer join coverage mismatch")
    out = out.drop(columns="_merge")
    out.index = np.arange(len(out))
    if arm == "p2_perm":
        # One common permutation for the two columns, at key level within each
        # reference. Collapsed stimulus rows retain identical peer values.
        rng = np.random.default_rng(20260923)
        for ref in sorted(out.ref_basename.unique()):
            subset = out.loc[out.ref_basename == ref, ["pair_key", "gmsd", "gmsm"]].drop_duplicates("pair_key")
            keys = subset.pair_key.to_numpy()
            values = subset[["gmsd", "gmsm"]].to_numpy()
            perm = rng.permutation(len(keys))
            mapped = dict(zip(keys, values[perm]))
            positions = out.index[out.ref_basename == ref]
            assigned = np.stack([mapped[k] for k in out.loc[positions, "pair_key"]])
            out.loc[positions, ["gmsd", "gmsm"]] = assigned
    label_meta = dict(label_meta)
    label_meta["peer_manifest_sha256"] = PEER_MANIFEST_SHA
    label_meta["peer_parquet_sha256"] = peer_info["parquet_sha256"]
    label_meta["peer_arm"] = arm
    return out, label_meta
