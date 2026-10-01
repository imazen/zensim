"""The single confirmatory read (amendment R2): sealed labels x frozen predictions -> per-arm verdicts.

THE COORDINATOR RUNS THIS, ONCE, AFTER DESIGN FREEZES. The CANONTAB lane wrote it and tested it only on synthetic labels;
it was never run on a sealed label. It refuses to run unless it is given, explicitly:

  --confirmatory-read        the acknowledgement that this exposes the sealed labels (one read, frozen candidates)
  --pin PIN.json             {"schema": "rev4-featpot-v2c-confirm-pin-v1", "reference": "r0", "candidates": [...],
                              "shortlist": [{"arm": ..., "head": "N"|"F"}, ... <= 6],
                              "heads": ["N","F"], "wide_receipts": {"main/real": sha, ...}, "confirm_receipt_sha256",
                              "keep_lists_sha256",
                              "binaries": {name: sha}, "program_sha"?, "data_sha"?, "local"?: bool}
                             The candidate list must equal the arms requested, and every cell must carry the pinned
                             receipt/binary hashes (and, for fleet cells, fleet_receipt.json program/data shas).
                             The pin also carries "labels": {set: {"path": ..., "column": ..., "join": "pair_key"|"row_id"}}
                             for every confirmatory set: the ORIGINAL dataset manifests (never the `_sealed/` echoes),
                             named explicitly; nothing is discovered.
  --root ROOT                the canon instrument root (cells under ROOT/confirm/cells)

PRIMARY TEST (amendment R2.1, "short list + Holm"): the pin carries a frozen `shortlist` of <= 6 {arm, head} entries. Per
entry: the mean permutation excess E over the five sealed sets (equal weights; TERMINAL excluded), one-sided bootstrap p =
share of the hierarchical-bootstrap draws of that mean that are <= 0, Holm step-down at 0.05 over the list. CONFIRMED =
Holm-significant and no set with E's upper bound < -0.005. Head-F-only confirmations read "helps a free head". The
per-set V1/V2 below are SECONDARY output.

Statistics, per confirmatory set s and head h (as v2, via v2_compare.contrast/family): seed-paired SROCC difference
vs R0, permutation excess vs `<arm>~p1..p3`, hierarchical bootstrap over seeds x references (B = 2000), each set's
declared target orientation (below), V1 on >= 2 of the 5 sets with no regression (E upper bound < -0.005), V2 seed
consistency >= 7/10 on the V1-passing sets. KonJND-JPEG TERMINAL is reported as a sanity guard only and never counted.

DECLARED TARGET ORIENTATIONS: taken from documents, never from label values (sources in the table below). A label is
turned into quality orientation (higher = better, like the model output) before correlation: DISTORTION labels are
negated. The coordinator confirms this table in CANONTAB_decisions.md before the read.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2_compare  # noqa: E402
from lib.zen_stats import panel_batch_indexed  # noqa: E402
from v2_common import BOOT_B, HEADS, N_PERMS, REPO, V2, sha, table_path  # noqa: E402

QUALITY, DISTORTION = "quality", "distortion"
# set -> (orientation, document that declares it). Documents only; no label value was read to fill this in.
ORIENTATION = {
    "cid22_b": (QUALITY, "bank manifest cid22_a25 label_scale 'CID22 human MCOS/100' (same label family); "
                         "check_target_orientation.py EXPECTED_ORIENTATION['cid22']"),
    "aic4": (DISTORTION, "docs/DATA_SPLITS.md AIC-4 row: 'TARGET IS DISTORTION-ORIENTED (q_jnd)'; "
                         "check_target_orientation.py EXPECTED_ORIENTATION['aic4']"),
    "konjnd_jpeg_select": (DISTORTION, "check_target_orientation.py EXPECTED_ORIENTATION['konjnd'] (PJND threshold)"),
    "konjnd_jpeg_terminal": (DISTORTION, "same as konjnd_jpeg_select"),
    "csiq": (QUALITY, "check_target_orientation.py EXPECTED_ORIENTATION['csiq'] ('1 - DMOS'; declares what a built "
                      "table stores, so a raw-DMOS label file would be distortion-oriented: coordinator to confirm)"),
    "mcljci": (DISTORTION, "benchmarks/mcl-jci.pointer.md + DATASETS_DONE.md: jnd_dist, higher = more visible distortion"),
}
CONFIRMATORY = ("cid22_b", "aic4", "konjnd_jpeg_select", "csiq", "mcljci")  # R2: five sets count for V1
GUARD = ("konjnd_jpeg_terminal",)  # touch-once sanity guard, never a ranking surface
SEEDS = 10
PIN_SCHEMA = "rev4-featpot-v2c-confirm-pin-v1"
MAX_SHORTLIST = 6
ALPHA = 0.05


class Refusal(SystemExit):
    pass


def refuse(message: str) -> None:
    raise Refusal(f"v2_confirm_read refuses: {message}")


# ------------------------------------------------------------------ pin and cells
def cell_path(root: Path, spec: str, head: str, seed: int) -> Path:
    return root / "confirm" / "cells" / f"{spec}__{head}" / f"full_s{seed}"


def load_pin(path: Path, arms: list[str], root: Path) -> dict:
    pin = json.loads(path.read_text())
    if pin.get("schema") != PIN_SCHEMA:
        refuse(f"{path}: not a {PIN_SCHEMA} file")
    if sorted(pin["candidates"]) != sorted(arms):
        refuse(f"requested arms {sorted(arms)} differ from the pinned candidate list {sorted(pin['candidates'])}")
    short = pin.get("shortlist")
    if (not short or len(short) > MAX_SHORTLIST or len({(e["arm"], e["head"]) for e in short}) != len(short)
            or any(e["head"] not in HEADS for e in short) or sorted({e["arm"] for e in short}) != sorted(arms)):
        refuse(f"pin shortlist must be 1..{MAX_SHORTLIST} unique (arm, head) entries over exactly the requested arms")
    if pin.get("reference") != "r0" or sorted(pin["heads"]) != sorted(HEADS):
        refuse("pin must name reference r0 and heads N and F")
    for key, path_ in (("confirm_receipt_sha256", root / "wide" / "confirm" / "receipt.json"),
                       ("keep_lists_sha256", root / "wide" / "keep_lists.json")):
        if sha(path_) != pin[key]:
            refuse(f"{path_.name} differs from the pin ({key})")
    return pin


def load_cell(root: Path, pin: dict, spec: str, head: str, seed: int) -> dict:
    d = cell_path(root, spec, head, seed)
    result = d / "result.json"
    if not result.is_file():
        return {}
    cell = json.loads(result.read_text())
    mismatch = [k for k in ("confirm_receipt_sha256", "keep_lists_sha256") if cell.get(k) != pin[k]]
    if cell.get("wide_receipt_sha256") != pin["wide_receipts"].get(f"{cell.get('family')}/{cell.get('variant')}"):
        mismatch.append("wide_receipt_sha256")
    if cell.get("binaries") != pin["binaries"]:
        mismatch.append("binaries")
    fleet = d / "fleet_receipt.json"
    if fleet.is_file():
        fr = json.loads(fleet.read_text())
        for key in ("program_sha", "data_sha"):
            if key in pin and fr.get(key) != pin[key]:
                mismatch.append(f"fleet {key}")
    elif not pin.get("local") and ("program_sha" in pin or "data_sha" in pin):
        mismatch.append("no fleet_receipt.json (pin names program/data hashes; set pin.local for local cells)")
    if cell.get("schema") != "rev4-featpot-v2c-confirm-cell-v1" or cell["spec"] != spec or cell["head"] != head \
            or cell["seed_index"] != seed or mismatch:
        refuse(f"{result}: cell does not match the pin ({', '.join(mismatch) or 'identity'})")
    return cell


# ------------------------------------------------------------------ labels
def load_labels(spec: dict, name: str, keys: pd.DataFrame) -> pd.DataFrame:
    """Label rows aligned to prediction rows: columns [pred_row, ref_basename, y_quality]. Labels of identical-pair rows
    (dropped from the prediction tables) are ignored; every other label row must find its prediction row."""
    join = spec.get("join", "pair_key")
    if join not in ("pair_key", "row_id"):
        refuse(f"{name}: join must be pair_key or row_id")
    table = pq.read_table(Path(spec["path"]), columns=[join, spec["column"]]).to_pandas()
    index = pd.Index(keys[join]).get_indexer(table[join])
    kept = index >= 0
    if kept.sum() == 0:
        refuse(f"{name}: no label row joins the prediction keys")
    dropped = int((~kept).sum())
    out = pd.DataFrame({"pred_row": index[kept], "label": table[spec["column"]].to_numpy(np.float64)[kept]})
    if not np.isfinite(out.label).all():
        refuse(f"{name}: non-finite label")
    covered = out.pred_row.nunique()
    if covered != len(keys):
        refuse(f"{name}: labels cover {covered} of {len(keys)} predicted keys (dropped label rows: {dropped})")
    sign = 1.0 if ORIENTATION[name][0] == QUALITY else -1.0
    out["y_quality"] = sign * out.label
    out["ref_basename"] = keys.ref_basename.to_numpy()[out.pred_row.to_numpy()]
    out.attrs["label_rows_not_predicted"] = dropped
    return out


# ------------------------------------------------------------------ statistics
def make_model_fn(root: Path, pin: dict, labels: dict[str, pd.DataFrame]):
    cache: dict = {}

    def cell_boot(spec: str, head: str, source: str, seed: int):
        key = (spec, head, source, seed)
        if key not in cache:
            cell = load_cell(root, pin, spec, head, seed)
            if not cell:
                cache[key] = None
            else:
                lab = labels[source]
                pred = np.asarray(cell["predictions"][source]["pred"], dtype=np.float64)[lab.pred_row.to_numpy()]
                y = lab.y_quality.to_numpy()
                rows = panel_batch_indexed({"p": pred, "y": y}, None, stats="srocc", timeout=7200,
                                           rendered_jobs=v2_compare.rendered_jobs(source, lab))
                # SIGNED SROCC of the model against the quality-oriented label: the panel's `srocc` is a magnitude
                # (v2_compare reads it), which would score a wrongly oriented model as well as a right one. The
                # confirmatory read must not.
                by = {r["label"]: r["srocc_signed"] for r in rows}
                cache[key] = (float(by["POINT"]), np.asarray([by[f"B{b}"] for b in range(v2_compare.BOOT_B)]))
        return cache[key]

    def model(spec: str, head: str, source: str):
        got = [cell_boot(spec, head, source, i) for i in range(SEEDS)]
        missing = [i for i, g in enumerate(got) if g is None]
        if missing:
            return None, missing
        return (np.array([g[0] for g in got]), np.stack([g[1] for g in got])), []

    return model


def holm(pvalues: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm step-down: sort ascending, reject p_(k) while p_(k) <= alpha / (m - k + 1) (k from 1); stop at the first miss."""
    m = len(pvalues)
    reject = [False] * m
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        if pvalues[i] > alpha / (m - rank):
            break
        reject[i] = True
    return reject


def primary_entry(arm: str, head: str, model_fn) -> dict:
    """Mean permutation excess over the five sealed sets and its one-sided bootstrap p (R2.1)."""
    perms = [f"{arm}~p{k}" for k in range(1, N_PERMS + 1)]
    per = {s: v2_compare.contrast(arm, perms, head, s, model_fn, keep_boot=True) for s in CONFIRMATORY}
    if any(v["status"] != "OK" for v in per.values()):
        return {"arm": arm, "head": head, "status": "INCOMPLETE", "sources": {s: v for s, v in per.items()
                                                                              if v["status"] != "OK"}}
    boot = np.mean([per[s]["excess_boot"] for s in CONFIRMATORY], axis=0)  # equal weights, per-draw
    mean_e = float(np.mean([per[s]["excess"] for s in CONFIRMATORY]))
    regress = [s for s in CONFIRMATORY if per[s]["regression"]]
    return {"arm": arm, "head": head, "status": "OK", "mean_excess": mean_e,
            "mean_excess_ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "p_one_sided": float(np.mean(boot <= 0.0)),
            "boot_draws": int(len(boot)), "regressions": regress,
            "per_set_excess": {s: per[s]["excess"] for s in CONFIRMATORY}}


def confirm_read(root: Path, pin: dict, arms: list[str], labels: dict[str, pd.DataFrame]) -> dict:
    model_fn = make_model_fn(root, pin, labels)
    out = {"schema": "rev4-featpot-v2c-confirm-read-v2", "label": "POTENTIAL — ceiling, not a model score",
           "orientation": {s: ORIENTATION[s] for s in (*CONFIRMATORY, *GUARD)}, "alpha": ALPHA,
           "primary": [], "secondary_v1_v2": {}, "sanity_guard": {}}
    entries = [primary_entry(e["arm"], e["head"], model_fn) for e in pin["shortlist"]]
    ok = [e for e in entries if e["status"] == "OK"]
    # an INCOMPLETE entry is never a pass; it still counts toward m (the list is frozen)
    pvals = [e["p_one_sided"] if e["status"] == "OK" else 1.0 for e in entries]
    reject = holm(pvals)
    for e, rej in zip(entries, reject):
        e["holm_significant"] = bool(rej and e["status"] == "OK")
        e["confirmed"] = bool(e["holm_significant"] and not e.get("regressions"))
        e["verdict"] = ("INCOMPLETE" if e["status"] != "OK" else
                        "confirmed" if e["confirmed"] and e["head"] == "N" else
                        "confirmed under the free head only: helps a free head" if e["confirmed"] else
                        "Holm-significant but a set regresses" if e["holm_significant"] else "not confirmed")
    out["primary"] = entries
    for arm in arms:
        for head in HEADS:
            rec = v2_compare.family(arm, head, CONFIRMATORY, model_fn)
            out["secondary_v1_v2"][f"{arm}_{head}"] = rec
            guard = {s: v2_compare.contrast(arm, [f"{arm}~p{k}" for k in range(1, N_PERMS + 1)], head, s, model_fn)
                     for s in GUARD}
            out["sanity_guard"][f"{arm}_{head}"] = {
                s: {k: g.get(k) for k in ("status", "r0_mean", "arm_mean", "delta", "delta_ci95", "excess", "excess_ci95")}
                for s, g in guard.items()}
    return out


# ------------------------------------------------------------------ exposure ledger
def ledger_stub(ledger: Path, pin_path: Path, pin: dict, labels_json: dict, arms: list[str]) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime())
    lines = [f"\n## Exposure ledger — {stamp}: confirmatory read (amendment R2)\n",
             "Status: **pending, update after read.** `v2_confirm_read.py` is about to read the sealed labels of: "
             + ", ".join(f"{s} (`{labels_json[s]['path']}`)" for s in (*CONFIRMATORY, *GUARD)) + ".",
             f"Frozen candidates: {', '.join(arms)} (+ r0, permuted controls); pin `{pin_path}` sha256 `{sha(pin_path)}`.",
             "KonJND-JPEG TERMINAL is read as a touch-once sanity guard only. No design change may follow from this read; "
             "a later change needs a new holdout.\n"]
    with Path(ledger).open("a") as stream:
        stream.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmatory-read", action="store_true", help="acknowledge: this exposes the sealed labels")
    ap.add_argument("--pin", type=Path)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--root", type=Path, help="canon instrument root (also read by v2_common from argv)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ledger", type=Path, default=REPO / "docs" / "DATA_SPLITS.md")
    args = ap.parse_args(argv)
    if not args.confirmatory_read:
        refuse("--confirmatory-read not given")
    if not (args.pin and args.root and args.out and args.arm):
        refuse("--pin, --root, --out and at least one --arm are all required")
    if Path(V2).resolve() != args.root.resolve():
        refuse("--root must be on argv so v2_common resolves the same root")
    pin = load_pin(args.pin, args.arm, args.root)
    labels_json = pin.get("labels", {})
    if sorted(labels_json) != sorted((*CONFIRMATORY, *GUARD)):
        refuse(f"the pin must name label paths for exactly {sorted((*CONFIRMATORY, *GUARD))}")
    receipt = json.loads((args.root / "wide" / "confirm" / "receipt.json").read_text())
    ledger_stub(args.ledger, args.pin, pin, labels_json, args.arm)  # recorded before any label byte is read
    labels = {}
    for name in (*CONFIRMATORY, *GUARD):
        table = receipt["sets"][name]["tables"]["main"]["real"]
        keys_path = table_path(table).parent / f"{name}.keys.parquet"
        if sha(keys_path) != table["keys_sha256"]:
            refuse(f"{name}: keys file changed after the confirm receipt")
        keys = pq.read_table(keys_path).to_pandas()
        labels[name] = load_labels(labels_json[name], name, keys)
    result = confirm_read(args.root, pin, args.arm, labels)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"confirm_read": str(args.out), "arms": {f"{e['arm']}_{e['head']}": e["verdict"] for e in result["primary"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
