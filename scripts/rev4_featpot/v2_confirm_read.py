"""The single confirmatory read (amendments R2, R2.1, R2.2): sealed labels x frozen predictions -> per-entry verdicts.

THE COORDINATOR RUNS THIS, ONCE, AFTER DESIGN FREEZES. The CANONTAB lane wrote it and tested it only on synthetic labels and
(through the label adapter) on the open sets AIC-3 and KADID-SELECT; it was never run on a sealed label. Nothing is exposed
until every check that needs no label has passed: `preflight` validates the pin, the frozen receipts, the code and panel
hashes, every label file's sha256, and every required cell BEFORE the exposure-ledger stub is written.

  python3 v2_confirm_read.py --confirmatory-read --pin PIN.json --root ROOT --out OUT.json --arm A [--arm B ...] [--bank BANK]
  (ZEN_PANEL_BIN must be set to the pinned panel binary)

PIN (`rev4-featpot-v2c-confirm-pin-v2`), every field required unless marked optional:
  reference "r0"; candidates [arms]; heads ["N","F"]; shortlist [{arm, head}, ... <= 6]; human_weight (null or the R3 weight, applied
  as `@h<w>` to EVERY spec incl. r0 and the permuted controls); frozen_sha256 (hash of ROOT/wide/frozen.json, written by
  `v2c_wide.py freeze`); wide_receipts {"main/real": sha, ...}; confirm_receipt_sha256; keep_lists_sha256; binaries
  {zensim_mlp_train, bake_dial_refit, panel: sha}; program_sha; data_sha (fleet cells only: each cell dir holds fleet_receipt.json);
  code {"rev4_featpot/v2_confirm_read.py": sha, v2_compare.py, v2_common.py, v2c_labels.py, v2c_wide.py, "lib/zen_stats.py"};
  shortlist_provenance {calibration: {path, sha256}, compare: {"<arm>_<head>": {path, sha256}}}; pixel_hashes (optional)
  {path, sha256} (TSV path<TAB>pixel_sha256, built with the pinned decoder, for collapsed stimuli);
  labels {set: adapter spec of v2c_labels (path, sha256, format, columns, select, via_pairs) for the five sealed sets and the TERMINAL
  guard}: the ORIGINAL dataset manifests (never the `_sealed/` echoes), named explicitly, nothing discovered.

PRIMARY TEST (R2.1 as amended by R2.2): per shortlisted (arm, head) entry, the mean permutation excess E over the four sealed sets
with multi-pair references (CID22-B, AIC-4, CSIQ, MCL-JCI), equal weights; one-sided bootstrap p = share of the hierarchical-bootstrap
draws of that mean <= 0; Holm step-down at 0.05 over the frozen list. KonJND-JPEG SELECT (one pair per reference: its within-reference
permutation is the identity) contributes its seed-paired delta vs R0 with a bootstrap CI and carries the regression veto on delta
(upper bound < -0.005). TERMINAL is a sanity guard only. CONFIRMED = Holm-significant, no primary set with E upper bound < -0.005, and
no KonJND regression. Reference resamples are INDEPENDENT per set (seed [BOOT_SEED, set index]); seed resamples are shared (one fit
predicts every set). R2's per-set V1/V2 are SECONDARY output, for the shortlisted entries only.

ORIENTATION is keyed by (set, label file name, label column) and taken from documents (the table below); an unlisted combination
refuses. A label is negated to quality orientation (higher = better, like the model output) when DISTORTION, and the SIGNED SROCC is used.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import v2_compare  # noqa: E402
import v2c_labels  # noqa: E402
from lib.zen_stats import panel_batch_indexed  # noqa: E402
from v2_common import (BOOT_SEED, FAMILIES, HEADS, N_PERMS, NOMINAL_WEIGHT, REPO, V2, load_frozen, sha,  # noqa: E402
                       table_path)
from v2c_wide import CANON_BANK  # noqa: E402
from v2c_wide import safe_path as _safe_path  # noqa: E402


def safe_path(path):
    """v2c_wide.safe_path (refuses any `_sealed` component) as a Refusal."""
    try:
        return _safe_path(path)
    except PermissionError as e:
        raise Refusal(f"v2_confirm_read refuses: {e}") from None


class Refusal(SystemExit):
    pass


QUALITY, DISTORTION = "quality", "distortion"
# (set, label file name, label column) -> orientation. Documents only; no label value was read to fill this in.
ORIENTATION = {
    ("cid22_b", "cid22val_pairs_ab.tsv", "human_score"): (QUALITY, "CID22 MCOS/100 (bank manifest cid22_a25 label_scale; check_target_orientation `cid22`)"),
    ("aic4", "aic4_pairs.tsv", "human_score"): (DISTORTION, "docs/DATA_SPLITS.md AIC-4 row: distortion-oriented q_jnd"),
    ("konjnd_jpeg_select", "konjnd_jpeg_val_pairs.tsv", "human_score"): (DISTORTION, "check_target_orientation `konjnd` (PJND)"),
    ("konjnd_jpeg_terminal", "konjnd_jpeg_val_pairs.tsv", "human_score"): (DISTORTION, "same file as SELECT"),
    ("csiq", "csiq_pairs.tsv", "human_score"): (QUALITY, "build_fr_corpus_pairs.py:98 writes 1.0 - DMOS (coordinator 2026-10-01)"),
    ("mcljci", "mcljci_labels.csv", "jnd_dist"): (DISTORTION, "benchmarks/mcl-jci.pointer.md: higher jnd_dist = more visible distortion"),
}
PRIMARY_SETS = ("cid22_b", "aic4", "csiq", "mcljci")   # R2.2: multi-pair references
KONJND = "konjnd_jpeg_select"                          # R2.2: delta vs R0 + regression veto
GUARD = ("konjnd_jpeg_terminal",)                      # touch-once sanity guard, never counted
SEALED = (*PRIMARY_SETS, KONJND)
ALL_SETS = (*SEALED, *GUARD)
SEEDS = 10
PIN_SCHEMA = "rev4-featpot-v2c-confirm-pin-v2"
MAX_SHORTLIST = 6
ALPHA = 0.05
CODE_FILES = {"rev4_featpot/v2_confirm_read.py": HERE / "v2_confirm_read.py", "rev4_featpot/v2_compare.py": HERE / "v2_compare.py",
              "rev4_featpot/v2_common.py": HERE / "v2_common.py", "rev4_featpot/v2c_labels.py": HERE / "v2c_labels.py",
              "rev4_featpot/v2c_wide.py": HERE / "v2c_wide.py", "lib/zen_stats.py": HERE.parent / "lib" / "zen_stats.py"}
REQUIRED_PIN = ("reference", "candidates", "heads", "shortlist", "human_weight", "frozen_sha256", "wide_receipts",
                "confirm_receipt_sha256", "keep_lists_sha256", "binaries", "program_sha", "data_sha", "code",
                "shortlist_provenance", "labels")


def refuse(message: str) -> None:
    raise Refusal(f"v2_confirm_read refuses: {message}")


def full_spec(spec: str, weight) -> str:
    """`c1~p1` -> `c1~p1@h2` under a human weight (grid builder naming); bare when the weight is null."""
    return spec if weight is None else f"{spec}@h{weight:g}"


def cell_path(root: Path, spec: str, head: str, seed: int) -> Path:
    return root / "confirm" / "cells" / f"{spec}__{head}" / f"full_s{seed}"


# ------------------------------------------------------------------ pin
def load_pin(path: Path, arms: list[str]) -> dict:
    pin = json.loads(path.read_text())
    if pin.get("schema") != PIN_SCHEMA:
        refuse(f"{path}: not a {PIN_SCHEMA} file")
    missing = [k for k in REQUIRED_PIN if k not in pin]
    if missing:
        refuse(f"pin lacks required fields {missing}")
    if sorted(pin["candidates"]) != sorted(arms):
        refuse(f"requested arms {sorted(arms)} differ from the pinned candidate list {sorted(pin['candidates'])}")
    short = pin["shortlist"]
    if (not short or len(short) > MAX_SHORTLIST or len({(e["arm"], e["head"]) for e in short}) != len(short)
            or any(e["head"] not in HEADS for e in short) or sorted({e["arm"] for e in short}) != sorted(arms)):
        refuse(f"pin shortlist must be 1..{MAX_SHORTLIST} unique (arm, head) entries over exactly the requested arms")
    if pin["reference"] != "r0" or sorted(pin["heads"]) != sorted(HEADS):
        refuse("pin must name reference r0 and heads N and F")
    if not (isinstance(pin["program_sha"], str) and len(pin["program_sha"]) == 64 and isinstance(pin["data_sha"], str)
            and len(pin["data_sha"]) == 64):
        refuse("program_sha and data_sha are required (64 hex)")
    if sorted(pin["labels"]) != sorted(ALL_SETS):
        refuse(f"the pin must name label sources for exactly {sorted(ALL_SETS)}")
    return pin


def spec_arms_needed(pin: dict) -> set[tuple[str, str]]:
    """(bare spec, head) pairs the read needs: the reference and each shortlisted arm with its three controls."""
    need = set()
    for e in pin["shortlist"]:
        for spec in (pin["reference"], e["arm"], *[f"{e['arm']}~p{k}" for k in range(1, N_PERMS + 1)]):
            need.add((spec, e["head"]))
    return need


# ------------------------------------------------------------------ preflight: everything that needs no label
def check_cell(cell: dict, d: Path, spec: str, head: str, seed: int, pin: dict, receipt: dict, frozen_sha: str) -> list[str]:
    w = pin["human_weight"]
    full = full_spec(spec, w)
    bad = []
    expect_variant = f"p{spec.partition('~p')[2]}" if "~p" in spec else "real"
    if cell.get("schema") != "rev4-featpot-v2c-confirm-cell-v1":
        bad.append("schema")
    if cell.get("spec") != full or cell.get("head") != head or cell.get("seed_index") != seed:
        bad.append("identity")
    if cell.get("variant") != expect_variant or cell.get("eval_variant") != expect_variant:
        bad.append(f"variant (spec {spec} needs {expect_variant}, matched null)")
    family = cell.get("family")
    if family not in FAMILIES:
        bad.append("family")
    if cell.get("wide_receipt_sha256") != pin["wide_receipts"].get(f"{family}/{expect_variant}"):
        bad.append("wide_receipt_sha256")
    for key, want in (("confirm_receipt_sha256", pin["confirm_receipt_sha256"]), ("keep_lists_sha256", pin["keep_lists_sha256"]),
                      ("frozen_sha256", frozen_sha)):
        if cell.get(key) != want:
            bad.append(key)
    if cell.get("binaries") != pin["binaries"]:
        bad.append("binaries")
    if cell.get("human_nominal_weight") != (NOMINAL_WEIGHT["human"] if w is None else w):
        bad.append("human_nominal_weight")
    fleet = d / "fleet_receipt.json"
    if not fleet.is_file():
        bad.append("fleet_receipt.json missing (program_sha/data_sha are mandatory)")
    else:
        fr = json.loads(fleet.read_text())
        if fr.get("program_sha") != pin["program_sha"] or fr.get("data_sha") != pin["data_sha"]:
            bad.append("fleet program_sha/data_sha")
    preds = cell.get("predictions", {})
    if sorted(preds) != sorted(ALL_SETS):
        bad.append("prediction sets")
    for name in ALL_SETS:
        rec = receipt["sets"][name]
        table = rec["tables"].get(family or "", {}).get(expect_variant)
        real = rec["tables"].get(family or "", {}).get("real")
        p = preds.get(name)
        if not table or not p:
            bad.append(f"{name}: table/prediction missing")
            continue
        pred = p.get("pred", [])
        if (p.get("table_sha256") != table["sha256"] or p.get("keys_sha256") != table["keys_sha256"]
                or table["keys_sha256"] != real["keys_sha256"] or len(pred) != rec["rows"] or p.get("rows") != rec["rows"]
                or not np.isfinite(np.asarray(pred, dtype=np.float64)).all()):
            bad.append(f"{name}: prediction/table/keys identity or length")
    return bad


def perm_identity_share(receipt: dict, families: set[str]) -> dict:
    """Per set (main family): share of added-column cells unchanged between the real and each permuted confirmatory table.
    Label-free. KonJND shows ~1.0 (one pair per reference: the within-reference permutation is the identity), amendment R2.2."""
    out = {}
    for name in ALL_SETS:
        if "main" not in families:
            continue
        tables = receipt["sets"][name]["tables"]["main"]
        cols = [f"f{i}" for i in range(944, receipt["width"])]
        real = pq.read_table(table_path(tables["real"]), columns=cols).to_pandas().to_numpy()
        out[name] = [float((real == pq.read_table(table_path(tables[f"p{k}"]), columns=cols).to_pandas().to_numpy()).mean())
                     for k in range(1, N_PERMS + 1)]
    return out


def preflight(root: Path, pin: dict) -> dict:
    """Everything that needs no label. Raises Refusal on the first problem; returns the validated cells and the report."""
    for rel, path in CODE_FILES.items():
        if pin["code"].get(rel) != sha(path):
            refuse(f"code hash differs from the pin: {rel}")
    panel_env = os.environ.get("ZEN_PANEL_BIN")
    if not panel_env:
        refuse("ZEN_PANEL_BIN must be set to the pinned panel binary")
    if sha(Path(panel_env)) != pin["binaries"].get("panel"):
        refuse("ZEN_PANEL_BIN differs from pin.binaries.panel")
    try:
        frozen, frozen_sha = load_frozen(root)
    except ValueError as e:
        refuse(str(e))
    if frozen_sha != pin["frozen_sha256"]:
        refuse("frozen.json differs from the pin")
    if (frozen["wide_receipts"] != pin["wide_receipts"] or frozen["confirm_receipt_sha256"] != pin["confirm_receipt_sha256"]
            or frozen["keep_lists_sha256"] != pin["keep_lists_sha256"]):
        refuse("pinned receipts differ from the frozen ones")
    prov = pin["shortlist_provenance"]
    for kind, rec in [("calibration", prov["calibration"]), *[(f"compare {k}", v) for k, v in prov["compare"].items()]]:
        if sha(Path(rec["path"])) != rec["sha256"]:
            refuse(f"shortlist provenance file changed or differs from the pin: {kind}")
    # label sources: pinned, sha-checked (bytes hashed, nothing parsed), never under _sealed, orientation declared
    label_report = {}
    for name in ALL_SETS:
        spec = pin["labels"][name]
        p = safe_path(spec["path"])
        key = (name, p.name, spec.get("label_col"))
        if key not in ORIENTATION:
            refuse(f"no declared orientation for {key}")
        if not p.is_file() or sha(p) != spec["sha256"]:
            refuse(f"{name}: label file missing or sha256 differs from the pin")
        if spec.get("via_pairs") and sha(safe_path(spec["via_pairs"]["path"])) != spec["via_pairs"]["sha256"]:
            refuse(f"{name}: pairs file sha256 differs from the pin")
        label_report[name] = {"file": str(p), "sha256": spec["sha256"], "orientation": ORIENTATION[key][0]}
    if pin.get("pixel_hashes") and sha(safe_path(pin["pixel_hashes"]["path"])) != pin["pixel_hashes"]["sha256"]:
        refuse("pixel_hashes file differs from the pin")
    receipt = json.loads((root / "wide" / "confirm" / "receipt.json").read_text())
    cells, problems = {}, []
    for spec, head in sorted(spec_arms_needed(pin)):
        for seed in range(SEEDS):
            d = cell_path(root, full_spec(spec, pin["human_weight"]), head, seed)
            if not (d / "result.json").is_file():
                problems.append(f"missing {d}")
                continue
            cell = json.loads((d / "result.json").read_text())
            bad = check_cell(cell, d, spec, head, seed, pin, receipt, frozen_sha)
            if bad:
                problems.append(f"{d}: {', '.join(bad)}")
            cells[(spec, head, seed)] = cell
    if problems:
        refuse(f"{len(problems)} cell problems before exposure, first: {problems[0]}")
    families = {c["family"] for c in cells.values()}
    keys_of = {}
    for name in ALL_SETS:
        table = receipt["sets"][name]["tables"][sorted(families)[0]]["real"]
        keys_path = table_path(table).with_name(f"{name}.keys.parquet")
        if sha(keys_path) != table["keys_sha256"]:
            refuse(f"{name}: keys file changed after the confirm receipt")
        keys_of[name] = keys_path
    return {"cells": cells, "receipt": receipt, "keys": keys_of, "frozen_sha256": frozen_sha, "labels": label_report,
            "perm_identity_share": perm_identity_share(receipt, families), "panel_sha256": sha(Path(panel_env))}


# ------------------------------------------------------------------ labels (after exposure)
def load_labels(pin: dict, name: str, keys_path: Path, bank: Path, pixel_sha: dict | None) -> pd.DataFrame:
    """Label rows aligned to the prediction rows of set `name`: [pred_row, label, y_quality, ref_basename]."""
    spec = pin["labels"][name]
    pred_keys = pq.read_table(keys_path).to_pandas()
    bank_keys = pq.read_table(safe_path(bank / name / "keys.parquet")).to_pandas()
    nonid = bank_keys.loc[~bank_keys.pixels_identical].reset_index(drop=True)
    if not np.array_equal(nonid.pair_key.to_numpy(), pred_keys.pair_key.to_numpy()):
        refuse(f"{name}: prediction keys are not the bank's non-identical keys in bank order")
    try:
        rows = v2c_labels.load_label_rows(spec)
        got, acct = v2c_labels.adapt(rows, bank_keys, spec.get("select"), pixel_sha)
    except ValueError as e:
        refuse(f"{name}: label adapter: {e}")
    pred_row = pd.Index(pred_keys.pair_key).get_indexer(got.pair_key)
    if (pred_row < 0).any():
        refuse(f"{name}: a label maps to a key without a prediction row")
    sign = 1.0 if ORIENTATION[(name, Path(spec["path"]).name, spec["label_col"])][0] == QUALITY else -1.0
    out = pd.DataFrame({"pred_row": pred_row, "label": got.label.to_numpy(), "y_quality": sign * got.label.to_numpy(),
                        "ref_basename": pred_keys.ref_basename.to_numpy()[pred_row]})
    out.attrs["accounting"] = acct
    return out


# ------------------------------------------------------------------ statistics
def make_model_fn(pin: dict, cells: dict, labels: dict[str, pd.DataFrame]):
    cache: dict = {}

    def cell_boot(spec: str, head: str, source: str, seed: int):
        key = (spec, head, source, seed)
        if key not in cache:
            cell = cells.get((spec, head, seed))
            if cell is None:
                cache[key] = None
            else:
                lab = labels[source]
                pred = np.asarray(cell["predictions"][source]["pred"], dtype=np.float64)[lab.pred_row.to_numpy()]
                rows = panel_batch_indexed({"p": pred, "y": lab.y_quality.to_numpy()}, None, stats="srocc", timeout=7200,
                                           rendered_jobs=v2_compare.rendered_jobs(source, lab))
                # SIGNED SROCC against the quality-oriented label: the panel's `srocc` is a magnitude and would score a wrongly
                # oriented model as well as a right one.
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


def primary_entry(arm: str, head: str, model_fn, reference: str) -> dict:
    """R2.2: mean permutation excess over the four multi-pair sets + its one-sided bootstrap p; KonJND SELECT delta + veto."""
    perms = [f"{arm}~p{k}" for k in range(1, N_PERMS + 1)]
    per = {s: v2_compare.contrast(arm, perms, head, s, model_fn, keep_boot=True, reference=reference) for s in PRIMARY_SETS}
    kon = v2_compare.contrast(arm, [], head, KONJND, model_fn, reference=reference)
    bad = {s: v for s, v in per.items() if v["status"] != "OK"}
    if kon["status"] != "OK":
        bad[KONJND] = kon
    if bad:
        return {"arm": arm, "head": head, "status": "INCOMPLETE", "sources": bad}
    boot = np.mean([per[s]["excess_boot"] for s in PRIMARY_SETS], axis=0)  # equal weights, per draw
    regress = [s for s in PRIMARY_SETS if per[s]["regression"]]
    kon_regress = bool(kon["delta_ci95"][1] < v2_compare.REGRESSION)
    return {"arm": arm, "head": head, "status": "OK", "mean_excess": float(np.mean([per[s]["excess"] for s in PRIMARY_SETS])),
            "mean_excess_ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "p_one_sided": float(np.mean(boot <= 0.0)),
            "boot_draws": int(len(boot)), "regressions": regress, "per_set_excess": {s: per[s]["excess"] for s in PRIMARY_SETS},
            "konjnd_select": {"delta": kon["delta"], "delta_ci95": kon["delta_ci95"], "r0_mean": kon["r0_mean"],
                              "arm_mean": kon["arm_mean"], "regression": kon_regress}}


def verdicts(entries: list[dict]) -> None:
    """Holm over the frozen list, then per-entry wording (free-head label derived per family)."""
    pvals = [e["p_one_sided"] if e["status"] == "OK" else 1.0 for e in entries]  # an INCOMPLETE entry counts in m, never passes
    for e, rej in zip(entries, holm(pvals)):
        e["holm_significant"] = bool(rej and e["status"] == "OK")
        e["confirmed"] = bool(e["holm_significant"] and not e.get("regressions") and not e.get("konjnd_select", {}).get("regression"))
    for e in entries:
        n_ok = any(o["arm"] == e["arm"] and o["head"] == "N" and o["confirmed"] for o in entries)
        if e["status"] != "OK":
            e["verdict"] = "INCOMPLETE"
        elif e["confirmed"] and e["head"] == "N":
            e["verdict"] = "confirmed; V3 dial gates next"
        elif e["confirmed"]:
            e["verdict"] = ("confirmed under the free head (the N entry is also confirmed)" if n_ok else
                            "confirmed under the free head only: helps a free head")
        elif e["holm_significant"]:
            e["verdict"] = "Holm-significant but a set regresses"
        else:
            e["verdict"] = "not confirmed"


def confirm_read(pin: dict, cells: dict, labels: dict[str, pd.DataFrame]) -> dict:
    ref = pin["reference"]
    model_fn = make_model_fn(pin, cells, labels)
    v2_compare.REF_SEEDS.clear()
    v2_compare.REF_SEEDS.update({s: [BOOT_SEED, i] for i, s in enumerate(ALL_SETS)})  # independent per-set reference streams
    v2_compare._ref_draws.clear()
    v2_compare._rendered.clear()
    entries = [primary_entry(e["arm"], e["head"], model_fn, ref) for e in pin["shortlist"]]
    verdicts(entries)
    out = {"schema": "rev4-featpot-v2c-confirm-read-v3", "label": "POTENTIAL — ceiling, not a model score", "alpha": ALPHA,
           "primary_sets": PRIMARY_SETS, "konjnd_set": KONJND, "guard_sets": GUARD,
           "orientation": {s: ORIENTATION[(s, Path(pin["labels"][s]["path"]).name, pin["labels"][s]["label_col"])]
                           for s in ALL_SETS}, "primary": entries, "secondary_v1_v2": {}, "sanity_guard": {}}
    for e in pin["shortlist"]:  # secondary and guard: shortlisted entries only (nothing off the frozen list is read)
        arm, head = e["arm"], e["head"]
        out["secondary_v1_v2"][f"{arm}_{head}"] = v2_compare.family(arm, head, SEALED, model_fn, reference=ref)
        guard = {s: v2_compare.contrast(arm, [f"{arm}~p{k}" for k in range(1, N_PERMS + 1)], head, s, model_fn, reference=ref)
                 for s in GUARD}
        out["sanity_guard"][f"{arm}_{head}"] = {s: {k: g.get(k) for k in ("status", "r0_mean", "arm_mean", "delta", "delta_ci95")}
                                                for s, g in guard.items()}
    return out


# ------------------------------------------------------------------ exposure ledger
def ledger_stub(ledger: Path, pin_path: Path, pin: dict) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime())
    shortlist = ", ".join(f"{e['arm']}/{e['head']}" for e in pin["shortlist"])
    lines = [f"\n## Exposure ledger — {stamp}: confirmatory read (amendments R2, R2.1, R2.2)\n",
             "Status: **pending, update after read.** `v2_confirm_read.py` is about to read the sealed labels of: "
             + ", ".join(f"{s} (`{pin['labels'][s]['path']}`, sha256 `{pin['labels'][s]['sha256'][:12]}…`)" for s in ALL_SETS) + ".",
             f"Frozen short list ({len(pin['shortlist'])}): {shortlist} (+ r0 and the permuted controls); human weight "
             f"{pin['human_weight']}; pin `{pin_path}` sha256 `{sha(pin_path)}`; frozen root `{pin['frozen_sha256'][:12]}…`.",
             "Primary: mean permutation excess over CID22-B, AIC-4, CSIQ, MCL-JCI with Holm at 0.05 (R2.1/R2.2); KonJND-JPEG SELECT reports "
             "delta vs R0 with a regression veto; TERMINAL is a touch-once sanity guard only. No design change may follow from this read; "
             "a later change needs a new holdout.\n"]
    with Path(ledger).open("a") as stream:
        stream.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmatory-read", action="store_true", help="acknowledge: this exposes the sealed labels")
    ap.add_argument("--pin", type=Path)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--root", type=Path, help="canon instrument root (also read by v2_common from argv)")
    ap.add_argument("--bank", type=Path, default=CANON_BANK)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ledger", type=Path, default=REPO / "docs" / "DATA_SPLITS.md")
    args = ap.parse_args(argv)
    if not args.confirmatory_read:
        refuse("--confirmatory-read not given")
    if not (args.pin and args.root and args.out and args.arm):
        refuse("--pin, --root, --out and at least one --arm are all required")
    if Path(V2).resolve() != args.root.resolve():
        refuse("--root must be on argv so v2_common resolves the same root")
    pin = load_pin(args.pin, args.arm)
    pre = preflight(args.root, pin)                               # nothing exposed yet; any mismatch refuses here
    ledger_stub(args.ledger, args.pin, pin)                       # recorded before any label byte is parsed
    pixel_sha = None
    if pin.get("pixel_hashes"):
        pixel_sha = dict(line.split("\t", 1) for line in Path(pin["pixel_hashes"]["path"]).read_text().splitlines() if line)
    labels = {name: load_labels(pin, name, pre["keys"][name], args.bank, pixel_sha) for name in ALL_SETS}
    result = confirm_read(pin, pre["cells"], labels)
    result["provenance"] = {
        "pin_sha256": sha(args.pin), "frozen_sha256": pre["frozen_sha256"], "panel_sha256": pre["panel_sha256"],
        "code_sha256": pin["code"], "labels": pre["labels"], "perm_identity_share": pre["perm_identity_share"],
        "per_set": {s: {"rows": int(pre["receipt"]["sets"][s]["rows"]), "references": int(labels[s].ref_basename.nunique()),
                        "label_rows_used": int(len(labels[s])), "accounting": labels[s].attrs["accounting"]} for s in ALL_SETS},
        "ref_seeds": {s: [BOOT_SEED, i] for i, s in enumerate(ALL_SETS)}, "boot_b": v2_compare.BOOT_B}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"confirm_read": str(args.out), "entries": {f"{e['arm']}_{e['head']}": e["verdict"] for e in result["primary"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
