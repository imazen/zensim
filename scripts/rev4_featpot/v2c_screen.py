"""Screen statistics for amendment R2.3: per-family importance from the screen models' bakes (no new fits).

The screen cells are the two all-columns specs `screen_main@h<w>` and `screen_aux@h<w>` (v2c grid (a)): per head, held-out source and
seed one bake trained on the other four sources. For a candidate family F (an arm's added columns) and a held-out source s:
  * permute F's columns within reference in the held-out table (3 draws, joint key-level permutation, the v2 control permutation),
  * predict with the cell's bake (`bake_dial_refit predict`, the production predictor), and take the drop in SIGNED SROCC against the
    unpermuted prediction of the same bake.
Importance(head, F) = mean drop over seeds, draws and the five sources. TARGETED importance = the same drop on the rows of R0's ten
worst distortion types only (KADID 20, 08, 07, 03, 21; TID2013 17, 18, 14, 12, 23; KonFiG highsharpen, multinoise, colordiffusion;
design log E1), mean over the sources that have those types. Selection for full arms (R2.3 item 4) is computed mechanically.

  python3 v2c_screen.py run --root ROOT --human-weight 32 [--sources ...] [--heads N F] [--seeds 0..9] [--jobs 4]
  python3 v2c_screen.py select --screen ROOT/compare/screen.json     # recompute the selection from a stored screen.json

Exploratory (R2.3 item 6): families not selected are reported as "not tested in full", never as null results.
"""

import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import v2_common
from v2_common import CANDIDATES, HEADS, SOURCE_ORDER, SOURCES, V2, arm_columns, extra_arms, sha, table_path

SCREEN_SEED = 20261001
DRAWS = 3
CHUNK_BLOCKS = 12
# R0's ten worst distortion types by (source, bank-keys `codec` value) — design log E1 (r0_errors_by_type). The codec labels of the
# bank keys are `kadid_NN`, `tid_NN` (NN = the dataset's distortion-type number) and the KonFiG type names.
WORST_TYPES = {"kadid": {"kadid_20", "kadid_08", "kadid_07", "kadid_03", "kadid_21"},
               "tid2013": {"tid_17", "tid_18", "tid_14", "tid_12", "tid_23"},
               "konfig": {"highsharpen", "multinoise", "colordiffusion"}}
MIN_TARGETED_ROWS = 20
# Design log E4: slots sign-consistent on all five sources, per family (csfw 7/12, mapdev 7/36, z1max 12/228). Keyed by the
# family's arm name; an arm without an E4 entry is not eligible for the sign-consistency pick.
E4_SIGN_CONSISTENT = {"csfw": (7, 12), "a1": (7, 36), "b2": (12, 228)}
N_TOP, N_EXTRA_MAX, N_TOP_TARGETED = 6, 2, 3
# Amendment R2.3 clarification: union arms are not families. `all` (the C1-C4 union) and `rall` (every research column)
# would win the ranking by construction; they are reported as an upper bound and never take a selection slot.
UNIONS = ("all", "rall")


# ------------------------------------------------------------------ families and types
def screen_families(spec: str) -> list[tuple[str, list[int]]]:
    """(arm, added column indices in the spec's table) for every candidate family evaluated on the spec's table."""
    fam = arm_columns(spec)[0]
    out = []
    for arm in (*CANDIDATES, *extra_arms()["arms"]):
        f, _, cols = arm_columns(arm)
        if f == fam:
            out.append((arm, cols[944:]))
    return out


def worst_mask(source: str, codecs: np.ndarray) -> np.ndarray:
    return np.isin(codecs, sorted(WORST_TYPES.get(source, ())))


def codecs_for(source: str, pair_keys: np.ndarray, bank: Path) -> np.ndarray:
    """Distortion-type label per held-out row, by pair_key from the Rev4 bank keys of the source's member sets."""
    maps = []
    for member in SOURCES[source]:
        k = pq.read_table(bank / member / "keys.parquet", columns=["pair_key", "codec"]).to_pandas()
        maps.append(k.set_index("pair_key").codec)
    lookup = pd.concat(maps)
    got = lookup.reindex(pair_keys)
    if got.isna().any():
        raise ValueError(f"{source}: {int(got.isna().sum())} rows without a bank codec")
    return got.to_numpy().astype(str)


# ------------------------------------------------------------------ core
def default_predict(bake: Path, table: Path, out: Path) -> np.ndarray:
    import v2_lodo_mlp
    return v2_lodo_mlp.predict(bake, table, out)


def default_srocc(jobs: list[tuple]) -> list[float]:
    from lib.zen_stats import panel_batch
    return [r["srocc_signed"] for r in panel_batch(jobs, stats="srocc")]


def permuted_block(x: np.ndarray, cols: list[int], refs: np.ndarray, keys: np.ndarray, source_i: int, fam_i: int, draw: int) -> np.ndarray:
    from v2c_wide import permute_within_reference
    rng = np.random.default_rng([SCREEN_SEED, source_i, fam_i, draw])
    out = x.copy()
    out[:, cols] = permute_within_reference(refs, keys, x[:, cols], np.ones(len(x), dtype=bool), rng)
    return out


def screen_source(table_x: np.ndarray, refs: np.ndarray, pair_keys: np.ndarray, y: np.ndarray, targeted: np.ndarray | None,
                  families: list[tuple[str, list[int]]], bakes: dict, source_i: int, work: Path, write_table, predict_fn=default_predict,
                  srocc_fn=default_srocc, draws: int = DRAWS, chunk_blocks: int = CHUNK_BLOCKS, jobs: int = 1) -> dict:
    """Drops for one held-out source. `bakes` = {(head, seed): bake path}. Returns {"base": {(head,seed): s}, "base_t": ..., "drop":
    {(head, fam, seed, draw): d}, "drop_t": ...}. Row blocks (unpermuted + each family x draw) are packed into temporary tables of
    at most `chunk_blocks` blocks, each predicted by every bake and removed afterwards."""
    blocks = [("base", None, -1)] + [(arm, cols, d) for arm, cols in families for d in range(draws)]
    fam_index = {arm: i for i, (arm, _) in enumerate(families)}
    n = len(table_x)
    res = {"base": {}, "base_t": {}, "drop": {}, "drop_t": {}}
    for start in range(0, len(blocks), chunk_blocks):
        chunk = blocks[start:start + chunk_blocks]
        xs = [table_x if cols is None else permuted_block(table_x, cols, refs, pair_keys, source_i, fam_index[arm], d)
              for arm, cols, d in chunk]
        big = np.concatenate(xs)
        del xs
        # the packed table is a new table; its refs/keys only matter to the writer
        path = work / f"chunk_{start:04d}.parquet"
        write_table(path, np.tile(refs, len(chunk)), np.zeros(len(big)), big)
        del big

        def one(item):
            (head, seed), bake = item
            pred = predict_fn(bake, path, work / f"pred_{head}_{seed}_{start}.tsv")
            return (head, seed), pred

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            preds = list(pool.map(one, sorted(bakes.items())))
        sr_jobs, meta = [], []
        for (head, seed), pred in preds:
            for b, (arm, cols, d) in enumerate(chunk):
                p = pred[b * n:(b + 1) * n]
                sr_jobs.append((f"{head}_{seed}_{arm}_{d}", p, y))
                meta.append((head, seed, arm, d, False))
                if targeted is not None:
                    sr_jobs.append((f"{head}_{seed}_{arm}_{d}_t", p[targeted], y[targeted]))
                    meta.append((head, seed, arm, d, True))
        vals = srocc_fn(sr_jobs)
        for (head, seed, arm, d, t), v in zip(meta, vals):
            key = (head, seed)
            if arm == "base":
                res["base_t" if t else "base"][key] = v
            else:
                res["drop_t" if t else "drop"][(head, arm, seed, d)] = v  # holds the permuted SROCC; the drop is formed below
        path.unlink(missing_ok=True)
        Path(f"{path}.manifest.json").unlink(missing_ok=True)
        for p in work.glob(f"pred_*_{start}.tsv"):
            p.unlink()
    for name, base in (("drop", "base"), ("drop_t", "base_t")):
        res[name] = {k: res[base][(k[0], k[2])] - v for k, v in res[name].items()}
    return res


def summarise(per_source: dict[str, dict], heads=HEADS) -> dict:
    """importance[head][family] = mean over sources of (mean over seeds and draws of the drop); targeted over sources that have rows."""
    out = {"importance": {}, "targeted_importance": {}, "per_source": {}}
    for head in heads:
        fams = sorted({k[1] for r in per_source.values() for k in r["drop"] if k[0] == head})
        out["importance"][head], out["targeted_importance"][head] = {}, {}
        for fam in fams:
            by_src, by_src_t = {}, {}
            for src, r in per_source.items():
                v = [x for k, x in r["drop"].items() if k[0] == head and k[1] == fam]
                vt = [x for k, x in r["drop_t"].items() if k[0] == head and k[1] == fam]
                if v:
                    by_src[src] = float(np.mean(v))
                if vt:
                    by_src_t[src] = float(np.mean(vt))
            out["importance"][head][fam] = float(np.mean(list(by_src.values()))) if by_src else None
            out["targeted_importance"][head][fam] = float(np.mean(list(by_src_t.values()))) if by_src_t else None
            out["per_source"][f"{head}/{fam}"] = {"drop": by_src, "drop_targeted": by_src_t}
    return out


def select(importance: dict, targeted: dict, e4: dict = E4_SIGN_CONSISTENT) -> dict:
    """R2.3 item 4: the 6 families with the largest head-N importance (ties: head-F importance), plus up to 2 more: first any family
    in the top 3 by head-N targeted importance not already chosen (in that order), then any family with >= 50% sign-consistent slots
    in design log E4 not already chosen; cap 8."""
    n_imp, f_imp = importance["N"], importance.get("F", {})
    fams = [f for f, v in n_imp.items() if v is not None and f not in UNIONS]
    rank = sorted(fams, key=lambda f: (-n_imp[f], -(f_imp.get(f) or 0.0), f))
    chosen = rank[:N_TOP]
    reasons = {f: "top-6 head-N importance" for f in chosen}
    extra = []
    t_rank = sorted([f for f, v in targeted["N"].items() if v is not None and f not in UNIONS], key=lambda f: (-targeted["N"][f], -(f_imp.get(f) or 0.0), f))
    for f in t_rank[:N_TOP_TARGETED]:
        if f not in chosen and f not in extra:
            extra.append(f)
            reasons[f] = "top-3 head-N targeted importance"
    for f in sorted(e4, key=lambda g: -(e4[g][0] / e4[g][1])):
        if e4[f][0] / e4[f][1] >= 0.5 and f in fams and f not in chosen and f not in extra:
            extra.append(f)
            reasons[f] = f"design log E4 sign-consistent slots {e4[f][0]}/{e4[f][1]}"
    extra = extra[:N_EXTRA_MAX]
    selected = chosen + extra
    return {"selected": selected, "reasons": {f: reasons[f] for f in selected},
            "not_tested_in_full": [f for f in rank if f not in selected], "rank_by_head_N_importance": rank,
            "unions_upper_bound": {u: {"N": n_imp.get(u), "F": f_imp.get(u)} for u in UNIONS if u in n_imp},
            "note": "exploratory screen; unselected families are 'not tested in full', never null results (R2.3 item 6)"}


# ------------------------------------------------------------------ run
def cell_bakes(root: Path, spec_full: str, source: str, heads, seeds) -> dict:
    out = {}
    for head in heads:
        for seed in seeds:
            d = root / "cells" / f"{spec_full}__{head}" / f"without_{source}_s{seed}"
            res = json.loads((d / "result.json").read_text())
            bake = d / "refit" / Path(res["selected_bake"]).name
            if sha(bake) != res["selected_bake_sha256"]:
                raise ValueError(f"{bake}: differs from its cell receipt")
            out[(head, seed)] = bake
    return out


def run(args) -> dict:
    from v2c_wide import write_table
    root = Path(args.root)
    w = args.human_weight
    sources = args.sources or list(SOURCE_ORDER)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    result = {"schema": "rev4-featpot-v2c-screen-v1", "label": "POTENTIAL — exploratory screen (R2.3), not a verdict", "human_weight": w,
              "draws": DRAWS, "screen_seed": SCREEN_SEED, "specs": {}}
    for spec in (args.specs or v2_common.SCREEN):
        spec_full = f"{spec}@h{w:g}"
        family, variant, _ = arm_columns(spec)
        receipt = json.loads((root / "wide" / family / variant / "receipt.json").read_text())
        fams = screen_families(spec)
        per_source, targeted_rows = {}, {}
        for si, source in enumerate(sources):
            leg = receipt["legs"][source]
            table = table_path(leg["full"])
            keys = pq.read_table(table_path(leg["full"]).with_name(f"{source}.keys.parquet")).to_pandas()
            x = pq.read_table(table).select([f"f{i}" for i in range(receipt["width"])]).to_pandas().to_numpy(np.float32)
            codecs = codecs_for(source, keys.pair_key.to_numpy(), Path(args.bank))
            tm = worst_mask(source, codecs)
            targeted = tm if tm.sum() >= MIN_TARGETED_ROWS else None
            targeted_rows[source] = int(tm.sum())
            seeds = args.seeds
            bakes = cell_bakes(root, spec_full, source, args.heads, seeds)
            sub = work / f"{spec}_{source}"
            sub.mkdir(exist_ok=True)

            def writer(path, ref, score, big, fam=family, wd=receipt["width"]):
                write_table(sub, path, ref, score, big, fam, "screen scratch", wd)

            res = screen_source(x, keys.ref_basename.to_numpy(), keys.pair_key.to_numpy(), keys.target.to_numpy(np.float64), targeted,
                                fams, bakes, si, sub, writer, jobs=args.jobs)
            # the unpermuted prediction must reproduce each cell's own held-out SROCC
            ok = {}
            for (head, seed), bake in bakes.items():
                cell = json.loads((bake.parent.parent / "result.json").read_text())
                ok[f"{head}_{seed}"] = abs(cell["score"]["srocc_signed"] - res["base"][(head, seed)]) < 1e-9
            res["base_matches_cell"] = ok
            per_source[source] = res
            shutil.rmtree(sub, ignore_errors=True)
        summ = summarise(per_source, args.heads)
        summ["base_matches_cells"] = all(all(r["base_matches_cell"].values()) for r in per_source.values())
        summ["targeted_rows"] = targeted_rows
        result["specs"][spec] = summ
    merged = {h: {} for h in args.heads}
    merged_t = {h: {} for h in args.heads}
    for spec_summ in result["specs"].values():
        for h in args.heads:
            merged[h].update(spec_summ["importance"][h])
            merged_t[h].update(spec_summ["targeted_importance"][h])
    result["selection"] = select(merged, merged_t)
    out = root / "compare" / ("screen.json" if not (args.specs or args.sources or args.seeds != list(range(10))) else "screen_partial.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"screen": str(out), "selected": result["selection"]["selected"]}))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--human-weight", type=float, default=32.0)
    r.add_argument("--sources", nargs="*", choices=SOURCE_ORDER)
    r.add_argument("--heads", nargs="*", default=list(HEADS), choices=HEADS)
    r.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    r.add_argument("--specs", nargs="*", choices=v2_common.SCREEN, help="default: both screen specs")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--bank", default="/var/tmp/rev4-featbank-r4")
    r.add_argument("--work", default="/var/tmp/canontab/screen_work")
    s = sub.add_parser("select")
    s.add_argument("--screen", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        rec = json.loads(Path(args.screen).read_text())
        merged = {h: {} for h in HEADS}
        merged_t = {h: {} for h in HEADS}
        for spec in rec["specs"].values():
            for h in HEADS:
                merged[h].update(spec["importance"].get(h, {}))
                merged_t[h].update(spec["targeted_importance"].get(h, {}))
        print(json.dumps(select(merged, merged_t), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
