"""Registered gates for the v2-canon tables (`v2c_wide.py verify`). Reads no human label of a sealed set.

Gates (each prints one JSON line; the exit code is 1 if any fails):
  cells        every feature cell of every table equals the Rev4 bank's float64 value cast to float32 (all rows of the
               five human sources and of the confirmatory tables; a seeded sample of the teacher legs), compared
               through an independent read path (whole-column pyarrow -> astype), bit for bit
  rows         per human source: rows == sum of n_stimuli over the non-identical Rev4 keys, distinct pair_key order ==
               the Rev4 key order, label coverage 100% (finite target for every row)
  rev3keys     (when the Rev3 v2 root exists) the human keys files equal the Rev3 v2 keys exactly: same stimuli, same
               order, same labels -- the label join is the same as v2's
  permutations p1..p3 tables keep every non-added column, the within-reference multiset of distinct keys' added-column vectors, and
               key-level joint permutation (rows sharing a pair_key share their vector)
  teacher      the bank's SSIMULACRA2 targets per reference equal the pinned R915 tables' (sub-multiset; the remainder
               is exactly the dropped identical keys)
  confirm      confirmatory tables: features equal the bank cast, human_score all 0, keys carry no target column
  receipts     every receipt file hash matches
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v2_common import (AUX_GMSBANK, AUX_ORACLE, AUX_PEERS, FAMILIES, SOURCE_ORDER, SOURCES, TEACHERS, VARIANTS, WIDTH, sha,
                       table_path)

REV3_V2 = Path("/var/tmp/rev4-featpot/v2")


def read_wide(path: Path, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    names = [f"f{i}" for i in range(width)]
    if table.column_names != ["ref_basename", "human_score", *names]:
        raise ValueError(f"{path}: unexpected schema")
    x = np.empty((table.num_rows, width), dtype=np.float32)
    for j, name in enumerate(names):
        x[:, j] = table.column(name).to_numpy()
    return (np.asarray(table.column("ref_basename").to_pylist()), table.column("human_score").to_numpy(), x)


def bank_f64_rows(bank: Path, name: str, keys: np.ndarray) -> np.ndarray:
    """float64 -> float32 cells of the bank's f0..f1824 for the given pair_keys, by whole-column pyarrow reads."""
    from v2c_wide import bank_file
    table = pq.read_table(bank_file(bank, name, "features.parquet"), filters=[("pair_key", "in", list(set(keys.tolist())))])
    got = np.asarray(table.column("pair_key").to_pylist())
    pos = pd.Index(got).get_indexer(keys)
    if (pos < 0).any():
        raise ValueError(f"{name}: {int((pos < 0).sum())} keys not found in the bank")
    out = np.empty((len(keys), WIDTH), dtype=np.float32)
    for j in range(WIDTH):
        out[:, j] = table.column(f"f{j}").to_numpy()[pos].astype(np.float32)
    return out


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def check_cells(bank: Path, keys: pd.DataFrame, x: np.ndarray, rows: np.ndarray | None = None,
                columns: list[int] | None = None) -> tuple[int, int]:
    """(rows checked, differing cells) over `columns` (default f0..f1824) of table x vs the bank, per member set."""
    columns = list(range(WIDTH)) if columns is None else columns
    rows = np.arange(len(keys)) if rows is None else rows
    checked = diff = 0
    for member, idx in pd.Series(rows).groupby(keys.member_set.to_numpy()[rows]):
        sel = idx.to_numpy()
        want = bank_f64_rows(bank, str(member), keys.pair_key.to_numpy()[sel])
        diff += int((np.ascontiguousarray(x[sel][:, columns]).view(np.uint32)
                     != np.ascontiguousarray(want[:, columns]).view(np.uint32)).sum())
        checked += len(sel)
    return checked, diff


def verify(bank: Path, out: Path, extras, sample: int) -> int:
    from v2c_wide import CONFIRM_SETS, bank_file, total_width
    width = total_width(extras)
    results, failed = [], False

    def gate(name: str, ok: bool, **detail) -> None:
        nonlocal failed
        failed |= not ok
        results.append({"gate": name, "ok": bool(ok), **detail})
        print(json.dumps(results[-1]), flush=True)

    receipt_path = out / "wide" / "main" / "real" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    legs = receipt["legs"]
    # ---- rows / cells / rev3keys on the five human sources (main, real)
    for source in SOURCE_ORDER:
        dest = out / "wide" / "main" / "real"
        keys = pq.read_table(dest / f"{source}.keys.parquet").to_pandas()
        ref, score, x = read_wide(dest / f"{source}.parquet", width)
        members = SOURCES[source]
        expected, order = 0, []
        for member in members:
            rk = pq.read_table(bank_file(bank, member, "keys.parquet"), columns=["pair_key", "n_stimuli", "pixels_identical"]).to_pandas()
            rk = rk.loc[~rk.pixels_identical]
            expected += int(rk.n_stimuli.sum())
            order += rk.pair_key.tolist()
        distinct = keys.pair_key.drop_duplicates().tolist()
        gate("rows", len(keys) == expected and distinct == order and np.isfinite(keys.target).all()
             and not keys.duplicated(["pair_key", "source_row_id"]).any() and len(x) == len(keys),
             leg=source, rows=len(keys), expected_rows=expected, key_order_equal=distinct == order,
             label_coverage=float(np.isfinite(keys.target).mean()))
        n, diff = check_cells(bank, keys, x)
        gate("cells", diff == 0, leg=source, rows=n, differing_cells=diff, columns=WIDTH)
        if width > WIDTH:
            for e in extras:
                side = np.vstack([pq.read_table(e.path(m), columns=["pair_key", *[f"f{i}" for i in e.ids]]).to_pandas()
                                  .set_index("pair_key").reindex(keys.pair_key[keys.member_set == m]).to_numpy(np.float64)
                                  .astype(np.float32) for m in members])
                gate("cells", same_bits(x[:, e.first:e.first + e.width], side), leg=source, extra=e.name)
        old = REV3_V2 / "wide" / "main" / "real" / f"{source}.keys.parquet"
        if old.is_file() and not (out.resolve() == REV3_V2.resolve()):
            ok = pq.read_table(old).to_pandas()
            gate("rev3keys", ok.equals(keys), leg=source, rows=len(keys))
    # ---- aux family (needs the peer output): bank columns, gmsbank and the peer pair equal their sources
    aux_dir = out / "wide" / "aux" / "real"
    if (aux_dir / "receipt.json").is_file():
        for source in SOURCE_ORDER:
            keys = pq.read_table(aux_dir / f"{source}.keys.parquet").to_pandas()
            ref, score, x = read_wide(aux_dir / f"{source}.parquet", width)
            cols = [*range(944), *AUX_GMSBANK]
            n, diff = check_cells(bank, keys, x, None, cols)
            peers_ok = True
            for member in SOURCES[source]:
                rec_peers = json.loads((out / "wide" / "main" / "real" / "receipt.json").read_text())["bank"][member].get("peers")
                peer = pq.read_table(rec_peers["path"], columns=["pair_key", "gmsd", "gmsm"]).to_pandas().drop_duplicates("pair_key")
                sel = (keys.member_set == member).to_numpy()
                got = peer.set_index("pair_key").reindex(keys.pair_key[sel])[["gmsd", "gmsm"]].to_numpy(np.float64).astype(np.float32)
                peers_ok &= same_bits(x[sel][:, [AUX_PEERS["gmsd"], AUX_PEERS["gmsm"]]], np.ascontiguousarray(got))
            zero = [c for c in range(width) if c not in {*range(944), *AUX_GMSBANK, *AUX_PEERS.values(), *AUX_ORACLE.values()}]
            gate("aux", diff == 0 and peers_ok and not x[:, zero].any(), leg=source, differing_cells=diff, peers_equal=peers_ok,
                 other_columns_zero=bool(not x[:, zero].any()))
    # ---- permutations
    for family in FAMILIES:
        real_dir = out / "wide" / family / "real"
        if not (real_dir / "receipt.json").is_file():
            continue
        added = list(range(944, width)) if family == "main" else sorted({*AUX_PEERS.values(), *AUX_ORACLE.values(), *AUX_GMSBANK})
        fixed = [c for c in range(width) if c not in set(added)]
        for variant in VARIANTS[1:]:
            vdir = out / "wide" / family / variant
            if not (vdir / "receipt.json").is_file():
                continue
            for source in SOURCE_ORDER:
                ref0, _, x0 = read_wide(real_dir / f"{source}.parquet", width)
                ref1, _, x1 = read_wide(vdir / f"{source}.parquet", width)
                keys = pq.read_table(real_dir / f"{source}.keys.parquet").to_pandas()
                ok = bool((ref0 == ref1).all() and same_bits(x0[:, fixed], x1[:, fixed]))
                keyed = True
                for r in sorted(set(ref0.tolist())):
                    rows = np.flatnonzero(ref0 == r)
                    # the permutation is over pair_keys (stimuli that share a key share their features), so the
                    # multiset is over distinct keys' vectors, one per key
                    first = rows[np.unique(keys.pair_key.to_numpy()[rows], return_index=True)[1]]
                    a = Counter(map(bytes, np.ascontiguousarray(x0[first][:, added]).view(np.uint8).reshape(len(first), -1)))
                    b = Counter(map(bytes, np.ascontiguousarray(x1[first][:, added]).view(np.uint8).reshape(len(first), -1)))
                    ok &= a == b
                    for _, grp in pd.Series(rows).groupby(keys.pair_key.to_numpy()[rows]):
                        g = grp.to_numpy()
                        keyed &= bool((x1[g][:, added] == x1[g[0], added]).all())
                gate("permutations", ok and keyed, family=family, variant=variant, leg=source,
                     within_reference_multiset_equal=ok, same_key_same_vector=keyed)
    # ---- teacher targets vs the pinned R915 tables, and a SafeSyn/CID22 cell sample
    import v2_wide
    rng = np.random.default_rng(20261001)
    for leg, (bank_name, r915, _) in TEACHERS.items():
        if leg not in legs:
            gate("teacher", False, leg=leg, reason="leg not built")
            continue
        files = v2_wide.pinned(leg)
        dest = out / "wide" / "main" / "real"
        for split in ("fit", "dev"):
            keys = pq.read_table(dest / f"{leg}_{split}.keys.parquet").to_pandas()
            r915_tab = pq.read_table(files[f"r915_{'fit' if split == 'fit' else 'dev'}"], columns=["ref_basename", "human_score"]).to_pandas()
            stripped = keys.ref_basename.str.split(":", n=1).str[1]
            ours = pd.DataFrame({"r": stripped, "y": keys.target.to_numpy(np.float32)})
            theirs = pd.DataFrame({"r": r915_tab.ref_basename, "y": r915_tab.human_score.to_numpy(np.float32)})
            extra_in_theirs = 0
            ok = set(ours.r) <= set(theirs.r)
            tg = {r: Counter(g.y.tolist()) for r, g in theirs.groupby("r")}
            for r, g in ours.groupby("r"):
                diff = tg[r] - Counter(g.y.tolist())
                ok &= not (Counter(g.y.tolist()) - tg[r])
                extra_in_theirs += sum(diff.values())
            ident = int(len(theirs) - len(ours))
            gate("teacher", bool(ok) and extra_in_theirs == ident, leg=leg, split=split, rows=len(ours),
                 r915_rows=len(theirs), rows_only_in_r915=extra_in_theirs, identical_dropped=ident)
            ref, score, x = read_wide(dest / f"{leg}_{split}.parquet", width)
            pick = np.sort(rng.choice(len(keys), size=min(sample // 2, len(keys)), replace=False))
            n, diff = check_cells(bank, keys, x, pick)
            gate("cells", diff == 0, leg=leg, split=split, sampled_rows=n, differing_cells=diff, columns=WIDTH,
                 targets_equal_table=bool(np.array_equal(score, keys.target.to_numpy())))
    # ---- confirmatory tables
    confirm = out / "wide" / "confirm" / "receipt.json"
    if confirm.is_file():
        record = json.loads(confirm.read_text())
        for name in CONFIRM_SETS:
            rec = record["sets"][name]
            for family, by_variant in rec["tables"].items():
                for variant, table in by_variant.items():
                    if variant == "skipped":
                        continue
                    dest = table_path(table).parent
                    keys = pq.read_table(dest / f"{name}.keys.parquet").to_pandas()
                    ref, score, x = read_wide(dest / f"{name}.parquet", width)
                    bank_keys = pq.read_table(bank_file(bank, name, "keys.parquet"),
                                              columns=["pair_key", "pixels_identical"]).to_pandas()
                    want = bank_keys.pair_key[~bank_keys.pixels_identical].tolist()
                    n, diff = check_cells(bank, keys, x) if (family == "main" and variant == "real") else (len(keys), 0)
                    gate("confirm", diff == 0 and keys.pair_key.tolist() == want and not score.any()
                         and not {"target", "human_score", "label"} & set(keys.columns)
                         and sha(table_path(table)) == table["sha256"], set=name, family=family, variant=variant,
                         rows=n, differing_cells=diff, key_order_equal=keys.pair_key.tolist() == want,
                         human_score_all_zero=not score.any())
    # ---- receipts: file hashes
    bad = []
    for fam in FAMILIES:
        for variant in VARIANTS:
            rp = out / "wide" / fam / variant / "receipt.json"
            if not rp.is_file():
                continue
            for lname, rec in json.loads(rp.read_text())["legs"].items():
                for part in ([rec["full"]] if "full" in rec else []) + [rec[s] for s in ("fit", "dev") if s in rec]:
                    tp = table_path(part)
                    if sha(tp) != part["sha256"] or sha(Path(f"{tp}.manifest.json")) != part["manifest_sha256"]:
                        bad.append(str(tp))
    gate("receipts", not bad, changed=bad)
    (out / "wide" / "verify.json").write_text(json.dumps({"gates": results, "all_ok": not failed}, indent=1) + "\n")
    print(json.dumps({"verify": str(out / "wide" / "verify.json"), "all_ok": not failed}))
    return int(failed)


if __name__ == "__main__":
    sys.exit("run via: v2c_wide.py verify")
