"""Readers for named zensim feature sets.

  benchmarks/feature_sets_registry.json        sets, compute tokens, eras, aliases, roots
  benchmarks/costset2_2026-10-03.candidate_ids.json   by_v2fy (420 IDs)
  benchmarks/e33_registration_2026-10-09.json   E33 arms and the fx1 d × fragility pairing

`slots_hash8` mirrors `zensim::feature_set_id::slots_hash8` (FNV-1a/64 over the
sorted, de-duplicated decimal IDs joined by ',', folded high ^ low). It is used
only to check the registry's recorded hashes; the Rust function stays the owner.
"""
from __future__ import annotations

import re

from .mdparse import SourceShapeError

REGISTRY = "benchmarks/feature_sets_registry.json"
BY_V2FY = "benchmarks/costset2_2026-10-03.candidate_ids.json"
E33 = "benchmarks/e33_registration_2026-10-09.json"


def parse_slots(s: str | None) -> list[int]:
    if s is None:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise SourceShapeError(f"bad slot range {part!r}")
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        out.extend(range(lo, hi + 1))
    return out


def compress(ids) -> str:
    ids = sorted(set(ids))
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def slots_hash8(ids) -> str:
    h = 0xCBF29CE484222325
    for b in ",".join(str(i) for i in sorted(set(ids))).encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{(h >> 32) ^ (h & 0xFFFFFFFF):08x}"


def registry(ctx) -> dict:
    d = ctx.json(REGISTRY, "feature_sets")
    for k in ("_schema", "compute_tokens", "eras", "sets", "roots", "aliases"):
        if k not in d:
            raise SourceShapeError(f"{REGISTRY}: missing top-level key {k}")
    need = {"compute", "layout", "era", "slots", "slots_hash8", "n_slots", "role", "kind", "evidence"}
    sets = []
    for sid, e in d["sets"].items():
        miss = need - set(e)
        if miss:
            raise SourceShapeError(f"{REGISTRY}: set {sid} lacks {sorted(miss)}")
        ids = parse_slots(e["slots"])
        hash_ok = None
        if e["slots"] is not None:
            hash_ok = slots_hash8(ids) == e["slots_hash8"]
            if len(ids) != e["n_slots"]:
                raise SourceShapeError(f"{REGISTRY}: set {sid} n_slots {e['n_slots']} != {len(ids)} parsed")
        sets.append({"id": sid, **{k: e.get(k) for k in ("compute", "layout", "era", "slots", "slots_hash8", "n_slots",
                                                         "role", "kind", "evidence", "legacy_name", "slot_selection",
                                                         "note", "producer_id", "superseded_by")},
                     "ids": ids, "hash_ok": hash_ok})
    tokens = {t: {"slots": v.get("slots"), "ids": parse_slots(v.get("slots")), "owner": v.get("owner"), "note": v.get("note")}
              for t, v in d["compute_tokens"].items()}
    ctx.count(REGISTRY, len(sets))
    return {"path": REGISTRY, "description": d["_schema"].get("description"), "sets": sets, "tokens": tokens,
            "eras": d["eras"], "roots": d["roots"], "aliases": d["aliases"], "regimes": d.get("regime_strings", {})}


def named_sets(ctx) -> dict:
    b = ctx.json(BY_V2FY, "named_sets")
    if b.get("schema") != "costset-candidates-v1" or "by_v2fy" not in b.get("candidates", {}):
        raise SourceShapeError(f"{BY_V2FY}: expected schema costset-candidates-v1 with candidates.by_v2fy")
    by = sorted(b["candidates"]["by_v2fy"])
    e = ctx.json(E33, "named_sets")
    if not str(e.get("schema", "")).startswith("e33-registration"):
        raise SourceShapeError(f"{E33}: unexpected schema {e.get('schema')!r}")
    fx1 = e.get("fx1") or {}
    cols = fx1.get("table_columns")
    if cols != ["id", "signal", "scale", "channel", "fragility_id"]:
        raise SourceShapeError(f"{E33}: fx1.table_columns changed: {cols}")
    table = [dict(zip(cols, r)) for r in fx1["table"]]
    frag = list(fx1["fragility_ids"])
    arms = e.get("arms") or {}
    for k in ("control", "A", "C"):
        if k not in arms:
            raise SourceShapeError(f"{E33}: arms.{k} missing")
    direct = sorted(r["id"] for r in table)
    ctx.count(BY_V2FY, len(by))
    ctx.count(E33, len(table))
    return {"by_v2fy": by, "by_v2fy_path": BY_V2FY, "e33_path": E33, "fx1_table": table, "fragility_ids": frag,
            "e33_arms": arms, "e33_direct": direct,
            "sets": [
                {"name": "by_v2fy (420)", "ids": by, "source": BY_V2FY,
                 "note": "The 420 IDs the by_v2fy recipe reads; production D1 fits use it."},
                {"name": "E33 control (420)", "ids": sorted(direct + frag), "source": E33,
                 "note": "fx1 direct inputs plus the ten pjnd_fragility inputs; equals by_v2fy."},
                {"name": "E33 arm A (410)", "ids": direct, "source": E33,
                 "note": "by_v2fy without the ten pjnd_fragility inputs."},
                {"name": "E33 arm C (410 + 410 products)", "ids": direct, "source": E33,
                 "note": "Arm A plus one derived d × fragility(cell(d)) product per direct input (positions 410+i)."},
            ]}
