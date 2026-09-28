"""Read POTENTIAL features and role-allowed targets from separate owners.

Feature/metadata rows come from the admitted diagnostic table. Every target
value is read directly from the promoted bank's allowed labels__*.parquet,
whose manifest and file SHA-256 are pinned and checked at each read. The join
uses both pair_key and source_row_id so collapsed stimuli remain distinct.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from admit_bank import BANK, EXPECTED_TARGET, OUT, load_pinned_set


FEATURES = [f"f{i}" for i in range(944)]


def load(name: str, features: bool = True, with_codec: bool = False):
    if name not in EXPECTED_TARGET:
        raise ValueError(f"{name}: not ruling-admitted")
    manifest = load_pinned_set(name)
    target = EXPECTED_TARGET[name]
    admitted = OUT / f"POT_{name}_rev3_944.parquet"
    schema = pq.ParquetFile(admitted).schema_arrow
    if target in schema.names:
        raise ValueError(f"{name}: admitted table must not copy target values")
    columns = ["pair_key", "source_row_id", "ref_basename"]
    if with_codec:
        columns.append("codec")
    if features:
        columns += FEATURES
    x = pq.read_table(admitted, columns=columns).to_pandas()
    label_file = next(filename for filename in manifest["files"] if filename.startswith("labels__"))
    y = pq.read_table(BANK / name / label_file,
                      columns=["pair_key", "source_row_id", target]).to_pandas()
    if x.duplicated(["pair_key", "source_row_id"]).any():
        raise ValueError(f"{name}: duplicate admitted stimulus key")
    if y.duplicated(["pair_key", "source_row_id"]).any():
        raise ValueError(f"{name}: duplicate bank label stimulus key")
    joined = x.merge(y, on=["pair_key", "source_row_id"], how="left", sort=False,  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
                     validate="one_to_one", indicator=True)
    if len(joined) != manifest["row_count"] or not (joined["_merge"] == "both").all():
        raise ValueError(f"{name}: role-allowed bank label join coverage mismatch")
    joined = joined.drop(columns="_merge")
    if joined[target].isna().any() or not np.isfinite(joined[target]).all():
        raise ValueError(f"{name}: missing or nonfinite target")
    joined = joined.rename(columns={target: "target"})
    joined.index = np.arange(len(joined))
    return joined, {"target": target, "label_scale": manifest["label_scale"],
                    "label_file": str(Path(BANK / name / label_file))}
