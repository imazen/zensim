"""Design log E13: curated SafeSyn teacher legs for v2 cells (recipe token `ts<name>`, rules in v2_common.TEACHER_SUBSETS).

The SafeSyn fit table carries reference and target but not codec or quality, so a per-row strata file travels with the fit
program (`data/e13/safesyn_fit_strata.npz`, built by e13_teacher.py from the bank keys). It records the sha256 of the
fit table's row-identity sidecar, which the wide receipt also pins, so a strata file can only be applied to the rows it
was built from. A curated leg is a row-filtered (and for the floor rules, target-clipped) copy of the fit table, written to
scratch and deleted after training.
"""

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2_common import REPO, TEACHER_CODECS, TEACHER_SUBSETS, V2

STRATA_NAME = "data/e13/safesyn_fit_strata.npz"
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
    pf = pq.ParquetFile(src)
    if pf.metadata.num_rows != len(keep):
        raise ValueError(f"{src}: {pf.metadata.num_rows} rows, mask has {len(keep)}")
    schema = pf.schema_arrow
    col = schema.get_field_index("human_score")
    writer = pq.ParquetWriter(dest, schema, compression="zstd", compression_level=1)
    start = 0
    try:
        for batch in pf.iter_batches(batch_size=16384):
            n = batch.num_rows
            m = keep[start:start + n]
            if m.any():
                cols = batch.columns
                cols[col] = pa.array(new_target[start:start + n], type=schema.field(col).type)
                writer.write_batch(pa.RecordBatch.from_arrays(cols, schema=schema).filter(pa.array(m)))
            start += n
    finally:
        writer.close()
    if start != len(keep):
        raise ValueError(f"{src}: streamed {start} rows, expected {len(keep)}")
    return {"rows_in": int(len(keep)), "rows_kept": int(keep.sum())}


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
