"""POTENTIAL — ceiling, not a model score. Feature-only cache compatibility probe."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/home/lilith/work/zensim-validation-2026-09-14/rev3-public-human-eval/features-rev3")
KON_CSV = Path("/mnt/v/output/zensim/konfig944/build/konfig_944.csv")
KON_PARQUET = ROOT / "ext_konfig.parquet"
COLS = ["ref_basename", "f0"]


def source_id(names):
    return names.str.extract(r"(SRC\d+)", expand=False)


def main():
    csv = pd.read_csv(KON_CSV, usecols=COLS)
    rev3 = pq.read_table(KON_PARQUET, columns=COLS).to_pandas()
    csv["source"] = source_id(csv.ref_basename)
    rev3["source"] = source_id(rev3.ref_basename)
    by_source = []
    for source in sorted(rev3.source.unique()):
        old = csv.loc[csv.source == source, "f0"]
        new = rev3.loc[rev3.source == source, "f0"]
        by_source.append(
            {
                "source": source,
                "csv_rows": len(old),
                "rev3_rows": len(new),
                "f0_exact_intersection": len(set(old) & set(new)),
                "csv_f0_min": float(old.min()),
                "rev3_f0_min": float(new.min()),
            }
        )
    print(
        json.dumps(
            {
                "status": "POTENTIAL — ceiling, not a model score",
                "labels_read": False,
                "konfig_csv_rows": len(csv),
                "konfig_rev3_val_rows": len(rev3),
                "konfig_rev3_val_references": rev3.ref_basename.nunique(),
                "per_source": by_source,
                "exact_f0_matches_total": sum(x["f0_exact_intersection"] for x in by_source),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
