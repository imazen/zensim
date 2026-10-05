"""Design log E13: curated SafeSyn teacher legs for v2 cells (recipe token `ts<name>`, rules in v2_common.TEACHER_SUBSETS).

The SafeSyn fit table carries reference and target but not codec or quality, so a per-row strata file travels with the fit
program (`data/e13/safesyn_fit_strata.npz`, built by e13_teacher.py from the bank keys). It records the sha256 of the
fit table's row-identity sidecar, which the wide receipt also pins, so a strata file can only be applied to the rows it
was built from. A curated leg is a row-filtered (and for the floor rules, target-clipped) copy of the fit table, written to
scratch and deleted after training.
"""

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2_common import COVERAGE_FAMILIES, REPO, TEACHER_CODECS, TEACHER_SUBSETS, V2

STRATA_NAME = "data/e13/safesyn_fit_strata.npz"
# Design log E14: the KADIS ordinal ladder table (e14_kadis_ordinal.py table), pinned by sha.
ORDINAL_NAME = "data/e14/kadis_ordinal.parquet"
ORDINAL_WIDTH = 1825  # extracted columns f0..f1824; the rest of the table's width is NaN
ORDINAL_SHA = "ffc245a0e39bd85d7527a08fd96bdd93b9fb266ad0334d4401eca06c95812012"
STRATA_SCHEMA = "rev4-featpot-e13-strata-v1"
MONO_DROP = 5.0


def strata_path() -> Path:
    """The program's packed strata file, else the coordinator's copy under the instrument root."""
    packed = REPO / STRATA_NAME
    return packed if packed.is_file() else V2 / "e13" / Path(STRATA_NAME).name


def load_strata(keys_sha256: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(codec name per row, quality per row, sha256 of the strata file) for the fit table whose keys sidecar hashes to
    `keys_sha256`; refuses a strata file built for other rows."""
    path = strata_path()
    with np.load(path, allow_pickle=False) as z:
        if str(z["schema"]) != STRATA_SCHEMA or str(z["keys_sha256"]) != keys_sha256:
            raise ValueError(f"{path}: strata do not belong to the SafeSyn fit rows ({keys_sha256[:12]})")
        codecs = z["codec_names"].astype(str)
        codec = codecs[z["codec_idx"]]
        quality = z["quality"].astype(np.int64)
    return codec, quality, hashlib.sha256(path.read_bytes()).hexdigest()


def curate(rule: str, ref: np.ndarray, target: np.ndarray, codec: np.ndarray, quality: np.ndarray,
           floor_lo: float) -> tuple[np.ndarray, np.ndarray]:
    """(keep mask, new targets) of one rule over the fit rows; targets change only under the floor rules."""
    if rule not in TEACHER_SUBSETS or rule == "none":
        raise ValueError(f"curate: no row rule for {rule!r}")
    keep = np.ones(len(target), bool)
    new = target.copy()
    if rule == "win":
        new = np.maximum(target, floor_lo)
    elif rule == "floor0":
        new = np.maximum(target, 0.0)
    elif rule == "neg":
        keep = target >= 0
    elif rule == "q20":
        keep = quality > 15
    elif rule == "mono5":
        frame = pd.DataFrame({"ref": ref, "codec": codec, "q": quality, "t": target, "row": np.arange(len(target))})
        frame = frame.sort_values(["ref", "codec", "q"], kind="stable")
        drop = frame.groupby(["ref", "codec"], sort=False).t.diff() < -MONO_DROP
        bad = set(map(tuple, frame.loc[drop, ["ref", "codec"]].to_numpy()))
        keep = np.array([(r, c) not in bad for r, c in zip(ref, codec)])
    else:  # x<codec>
        keep = codec != TEACHER_CODECS[rule[1:]]
    return keep, new


def write_curated(src: Path, dest: Path, keep: np.ndarray, new_target: np.ndarray) -> dict:
    """Row-filtered copy of a wide table with `human_score` replaced, streamed batch by batch; returns its record."""
    from v2c_wide import safe_path, sha
    src, dest = safe_path(src), safe_path(dest)
    if dest.exists() or Path(f"{dest}.manifest.json").exists() or key_path(dest).exists():
        raise ValueError(f"{dest}: curated output must be fresh")
    source_manifest = Path(f"{src}.manifest.json")
    declaration = json.loads(source_manifest.read_text()) if source_manifest.is_file() else {}
    if declaration:
        from v2_common import refuse_immutable_output, table_input_roots
        roots = table_input_roots(src)
        for path in (dest, Path(f"{dest}.manifest.json"), key_path(dest)):
            refuse_immutable_output(path, roots)
    keep = np.asarray(keep, dtype=bool)
    pf = pq.ParquetFile(src)
    if pf.metadata.num_rows != len(keep) or len(new_target) != len(keep):
        raise ValueError(f"{src}: {pf.metadata.num_rows} rows, mask has {len(keep)}")
    source_keys = key_path(src)
    keys = None
    if declaration.get("feature_set_id"):
        if (sha(src) != declaration.get("table_sha256") or not source_keys.is_file()
                or sha(source_keys) != declaration.get("keys_sha256")):
            raise ValueError(f"{src}: admitted source table/keys changed")
        keys = pq.read_table(source_keys)
        if keys.num_rows != len(keep) or row_keys_sha(keys) != declaration.get("row_keys_sha256"):
            raise ValueError(f"{src}: admitted source row keys changed")
    schema = pf.schema_arrow
    col = schema.get_field_index("human_score")
    writer = pq.ParquetWriter(dest, schema, compression="zstd", compression_level=1)
    start = 0
    targets_changed = 0
    try:
        for batch in pf.iter_batches(batch_size=16384):
            n = batch.num_rows
            m = keep[start:start + n]
            if m.any():
                cols = batch.columns
                original = cols[col].to_numpy(zero_copy_only=False)
                replacement = np.asarray(new_target[start:start + n], dtype=original.dtype)
                targets_changed += int(np.count_nonzero(original[m] != replacement[m]))
                cols[col] = pa.array(new_target[start:start + n], type=schema.field(col).type)
                writer.write_batch(pa.RecordBatch.from_arrays(cols, schema=schema).filter(pa.array(m)))
            start += n
    finally:
        writer.close()
    if start != len(keep):
        raise ValueError(f"{src}: streamed {start} rows, expected {len(keep)}")
    record = {"rows_in": int(len(keep)), "rows_kept": int(keep.sum()),
              "row_selection_sha256": selection_sha(np.flatnonzero(keep)), "targets_changed": targets_changed}
    if declaration:
        inherited = {**declaration, "source_table_sha256": sha(src), "source_manifest_sha256": sha(source_manifest),
                     "source_row_selection_sha256": declaration.get("row_selection_sha256"),
                     "source_row_keys_sha256": declaration.get("row_keys_sha256"),
                     "table_sha256": sha(dest), **record}
        if keys is not None:
            selected = keys.filter(pa.array(keep))
            pq.write_table(selected, key_path(dest), compression="zstd")
            inherited.update({"keys_sha256": sha(key_path(dest)), "row_keys_sha256": row_keys_sha(selected)})
            record.update({"keys_sha256": inherited["keys_sha256"], "row_keys_sha256": inherited["row_keys_sha256"]})
        Path(f"{dest}.manifest.json").write_text(json.dumps(inherited, indent=1) + "\n")
    return record


def curated_leg(rule: str, fit_path: Path, keys_sha256: str, floor_lo: float, scratch: Path) -> tuple[Path, dict]:
    """Write the curated SafeSyn fit table for `rule` under `scratch`; returns (path, record for result.json)."""
    codec, quality, strata_sha = load_strata(keys_sha256)
    meta = pq.read_table(fit_path, columns=["ref_basename", "human_score"])
    ref = meta["ref_basename"].to_numpy(zero_copy_only=False).astype(str)
    target = meta["human_score"].to_numpy().astype(np.float64)
    if len(ref) != len(codec):
        raise ValueError(f"{fit_path}: {len(ref)} rows, strata have {len(codec)}")
    keep, new = curate(rule, ref, target, codec, quality, floor_lo)
    # Containers on one host share scratch and reuse small pids, so the name must be unique across processes.
    dest = scratch / f"safesyn_fit_ts{rule}_{uuid.uuid4().hex}.parquet"
    record = write_curated(fit_path, dest, keep, new)
    record.update({"rule": rule, "strata_sha256": strata_sha, "targets_changed": int((new != target)[keep].sum()),
                   "floor_lo": floor_lo if rule == "win" else None})
    return dest, record


def ordinal_leg() -> tuple[Path, dict]:
    """(path, record) of the pinned KADIS ordinal ladder table: the program's packed copy, else the coordinator's."""
    packed = REPO / ORDINAL_NAME
    path = packed if packed.is_file() else V2 / "e14" / Path(ORDINAL_NAME).name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ORDINAL_SHA:
        raise ValueError(f"{path}: not the registered E14 ordinal table ({digest[:12]})")
    return path, {"table_sha256": digest, "rows": pq.ParquetFile(path).metadata.num_rows}


# Design log E15: the coverage pool (e15_coverage.py table) and its row keys, pinned by sha; filled when the pool is built.
POOL_NAME = "data/e15/coverage_pool.parquet"
POOL_KEYS_NAME = "data/e15/coverage_pool.keys.parquet"
POOL_SHA = "6b00349c8aca6613aeb1591f8411e738e3c70798274c7df9017dfbc8844848b3"
POOL_KEYS_SHA = "bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf"
# Rev5 (spec rev5_spec_2026-10-04.md §6): the same 42,021 rungs and keys extracted at Rev5 (basic+peaks+v2, other slots NaN).
# The pool's own sidecar manifest declares its formula revision, which selects the pin; keys are revision-independent.
POOL_SHA_REV5 = "6bf584ac70579bdf9a0242b7ccfd688ff0182cb5c75e35085ba71967207f8f5e"


def selection_sha(indices) -> str:
    """SHA-256 of ordered zero-based source row indices, little-endian u64."""
    return hashlib.sha256(np.asarray(indices, dtype="<u8").tobytes()).hexdigest()


def row_keys_sha(keys: pa.Table) -> str:
    """Ordered label-free key identity: compact UTF-8 JSON, values as strings."""
    rows = list(zip(*(keys[c].to_pylist() for c in keys.column_names)))
    payload = {"columns": keys.column_names, "rows": [[str(v) for v in row] for row in rows]}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def key_path(table: Path) -> Path:
    return table.with_suffix(".keys.parquet")


def _packed_or_root(name: str) -> Path:
    packed = REPO / name
    return packed if packed.is_file() else V2 / "e15" / Path(name).name


def coverage_leg(mask: int, scratch: Path, *, admitted_root: Path | None = None) -> tuple[Path, dict]:
    """Rows of the pinned coverage pool whose family bit is set in `mask`, copied to scratch; returns (path, record)."""
    from v2c_wide import safe_path
    if admitted_root is not None:
        from v2_common import admission_input_roots, refuse_immutable_output
        refuse_immutable_output(scratch, admission_input_roots(admitted_root))
    pool, keys = ((safe_path(admitted_root) / "e15" / Path(POOL_NAME).name,
                   safe_path(admitted_root) / "e15" / Path(POOL_KEYS_NAME).name) if admitted_root is not None else
                  (_packed_or_root(POOL_NAME), _packed_or_root(POOL_KEYS_NAME)))
    pool, keys, scratch = safe_path(pool), safe_path(keys), safe_path(scratch)
    man = Path(f"{pool}.manifest.json")
    revision = int(json.loads(man.read_text()).get("formula_revision", 4)) if man.is_file() else 4
    pool_sha = {4: POOL_SHA, 5: POOL_SHA_REV5}.get(revision)
    if pool_sha is None:
        raise ValueError(f"{pool}: no registered E15 coverage pool for formula revision {revision}")
    for path, want in ((pool, pool_sha), (keys, POOL_KEYS_SHA)):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise ValueError(f"{path}: not the registered E15 coverage pool ({got[:12]})")
    fams = list(COVERAGE_FAMILIES)
    chosen = [f for i, f in enumerate(fams) if mask >> i & 1]
    family = pq.read_table(keys, columns=["family"])["family"].to_numpy(zero_copy_only=False).astype(str)
    keep = np.isin(family, chosen)
    if not keep.any():
        raise ValueError(f"coverage mask {mask:#x} selects no rows")
    target = pq.read_table(pool, columns=["human_score"])["human_score"].to_numpy().astype(np.float64)
    dest = scratch / f"coverage_cf{mask:x}_{uuid.uuid4().hex}.parquet"
    record = write_curated(pool, dest, keep, target)
    record.update({"mask": mask, "families": chosen, "pool_sha256": pool_sha, "pool_formula_revision": revision})
    sidecar = Path(f"{dest}.manifest.json")
    if sidecar.is_file():
        declaration = json.loads(sidecar.read_text())
        declaration.update({"coverage_mask": mask, "coverage_families": chosen})
        sidecar.write_text(json.dumps(declaration, indent=1) + "\n")
    return dest, record
