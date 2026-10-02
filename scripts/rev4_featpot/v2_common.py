"""Instrument v2 shared definitions.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md (with its revision R1). POTENTIAL —
ceiling, not a model score. Every value here is fixed by that amendment; change the amendment first.

Wide-table layout (revision R1.1): two table families, both WIDTH = 1825 columns, so a kept column's
first-layer initial weights are identical in every arm of either family (--keep-features keeps full width).
  main: f0..f943 bank Rev3 944 surface (39 structural zeros filled with 0, as admit_bank does);
        f944..f1824 the Rev4 research vector at its canonical IDs.
  aux:  f0..f943 bank; f944 gmsd, f945 gmsm (the reviewed peer pair, packed at registered IDs exactly as the
        registered P2 arm did); f946 oracle_lo, f947 oracle_hi; f1322..f1501 gmsbank (arm p3); all other
        columns 0. Every column a bake reads is a registered feature ID, which the product predictor requires.
Variants: "real" and "p1".."p3" (the family's added columns permuted jointly within reference over pair keys).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/var/tmp/rev4-featpot")


def _root_override(argv: list[str]) -> str | None:
    """`--root DIR` / `--root=DIR` anywhere in argv, else $REV4_V2_ROOT, else None (CANONTAB: the Rev3 v2 instrument and
    the v2-canon instrument share these scripts and coexist under different roots; argv carries the root, so the
    declared cells' argv hashes, hence their job ids, differ)."""
    for i, arg in enumerate(argv):
        if arg == "--root" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--root="):
            return arg.split("=", 1)[1]
    return os.environ.get("REV4_V2_ROOT")


V2 = Path(_root_override(sys.argv) or ROOT / "v2")
# Erratum R1.1 binary set locally (v8 trainer and panel, predictor from main 86fc02bb, admitted by
# benchmarks/rev4_featpot_v2_predictor_parity_2026-10-01.json); in a fit-cell container the executor links the same
# three binaries under target/debug.
_LOCAL = Path("/var/tmp/fitv2/bin-v2")
BIN_DIR = Path(os.environ.get("REV4_V2_BIN_DIR", str(_LOCAL if _LOCAL.is_dir() else ROOT / "target/debug")))
TRAINER = BIN_DIR / "zensim_mlp_train"
FITBIN = BIN_DIR / "bake_dial_refit"
PANEL = BIN_DIR / "panel"

# Evaluated human sources, in fold order. Member sets are bank names admitted by admit_bank.
SOURCES = {
    "kadid": ("kadid_train", "kadid_select"),
    "tid2013": ("tid2013",),
    "konfig": ("konfig_train", "konfig_val"),
    "cid22_a25": ("cid22_a25",),
    "aic3": ("aic3",),
}
SOURCE_ORDER = tuple(SOURCES)
# R915 teacher legs (never evaluated): bank set, R915 dedup-table leg, validation weight.
TEACHERS = {"safesyn": ("safesyn", "safesyn", 0.5), "cid22": ("cid22_train", "cid22", 2.0)}
R915_TABLES = Path("/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables")
TEACHER_PIN = REPO / "benchmarks/rev4_featpot_v2_teacher_pin_2026-09-30.json"
NOMINAL_WEIGHT = {"safesyn": 1.0, "cid22": 1.0, "human": 0.5}
HUMAN_VAL_WEIGHT = 1.0
HUMAN_DEV_MODULUS = 5  # label-free: a human reference is dev when sha256(ref) mod 5 == 0

INIT_SEEDS = (1101, 1103, 1107, 1109, 1117, 1123, 1129, 1151, 1153, 1163)
SAMPLE_SEEDS = tuple(101 + k * 100_000_000 for k in range(10))
EPOCHS = 120
# Which epoch's weights a cell keeps (design log E5/E5b, amendment R4): "best_dev" = the trainer's best dev-aggregate
# epoch (R1-R3); "last" = the final-epoch checkpoint (dumped by the trainer, identical to its state after training).
EPOCH_RULE = "last"  # amendment R4 (design log E5/E5b)
PAIRS_PER_EPOCH = 50_000
HIDDEN = 32
HEADS = ("N", "F")

CANDIDATES = ("c1", "c2", "c3", "c4", "all", "csfw", "c7", "p1", "p3", "b1", "b1s",
              "c8n", "rall", "a1", "a1m", "b2", "b2m")
CALIBRATION = ("oracle_lo", "oracle_hi", "minus_basic")
# Amendment R2.3: one all-columns model per family table (no permuted controls): screen_main = bank + every main-family candidate
# column + every appended (extra) column; screen_aux = bank + gmsd + gmsm + gmsbank.
SCREEN = ("screen_main", "screen_aux")
ORACLE_SIGMA = {"oracle_lo": 1.5, "oracle_hi": 0.5}
FAMILIES = ("main", "aux")
WIDTH = 1825
# Design log E9: the seven legacy blocks of the Rev3 944 bank (FEATURE_SET_IDS.md) and the lean core (basic + peaks,
# R915's 228-column regime). Block specs: `r0-<block>` (R0 without the block), `core`, `core+<block or arm>`.
LEGACY_BLOCKS = {"basic": range(0, 156), "peaks": range(156, 228), "masked": range(228, 300), "iw": range(300, 372),
                 "v2": range(372, 720), "append": range(720, 924), "append2": range(924, 944)}
CORE = range(0, 228)
AUX_PEERS = {"gmsd": 944, "gmsm": 945}
AUX_ORACLE = {"oracle_lo": 946, "oracle_hi": 947}
AUX_GMSBANK = tuple(range(1322, 1502))
AUX_ADDED = tuple(sorted({*AUX_PEERS.values(), *AUX_ORACLE.values(), *AUX_GMSBANK}))
N_PERMS = 3
VARIANTS = ("real", *(f"p{k}" for k in range(1, N_PERMS + 1)))
PERM_SEED_BASE = 20260930
ORACLE_SEED_BASE = 20260930
BOOT_B = 2000
BOOT_SEED = 20260930
REPLAY = "Rev4 POTENTIAL Instrument v2 diagnostic: pinned Rev3 944 plus pinned sidecars; never ship"


def table_path(record: dict) -> Path:
    """A receipt record's table file: the root-relative `rel` of a v2-canon receipt (so the tree can be promoted or packed
    anywhere), else the absolute `path` of a Rev3 v2 receipt."""
    return V2 / record["rel"] if "rel" in record else Path(record["path"])


FROZEN_SCHEMA = "rev4-featpot-v2c-frozen-v1"


def load_frozen(root: Path | None = None) -> tuple[dict, str]:
    """(record, sha256) of `<root>/wide/frozen.json` after re-hashing every file it pins (confirm receipt, wide receipts,
    keep lists, extra arms). Raises if the canon tables changed after the freeze or no freeze exists."""
    root = V2 if root is None else root
    path = Path(root) / "wide" / "frozen.json"
    if not path.is_file():
        raise ValueError(f"{path}: the canon root is not frozen (run `v2c_wide.py freeze`)")
    record = json.loads(path.read_text())
    if record.get("schema") != FROZEN_SCHEMA:
        raise ValueError(f"{path}: unexpected schema")
    wide = Path(root) / "wide"
    pins = {**{f"wide/{k}/receipt.json": v for k, v in record["wide_receipts"].items()},
            "wide/confirm/receipt.json": record["confirm_receipt_sha256"], "wide/keep_lists.json": record["keep_lists_sha256"]}
    if record.get("extra_arms_sha256"):
        pins["wide/extra_arms.json"] = record["extra_arms_sha256"]
    for rel, want in pins.items():
        if sha(Path(root) / rel) != want:
            raise ValueError(f"{rel}: changed after the freeze")
    return record, sha(path)


def split_weight(spec: str) -> tuple[str, float | None]:
    """'oracle_hi~p1@h2' -> ('oracle_hi~p1', 2.0): the instrument-retune sweep's human-weight override (design log E2).
    No suffix -> (spec, None), i.e. NOMINAL_WEIGHT['human']."""
    core, _, w = spec.partition("@h")
    if not w:
        return core, None
    value = float(w.split(":")[0])  # ":H<n>:gl<x>" recipe tokens follow the weight (recipe_of, design log E8)
    if not 0 < value <= 64:
        raise ValueError(f"bad human weight in {spec!r}")
    return core, value


def recipe_of(spec: str) -> dict:
    """Training-recipe tokens after the human weight (design log E8): 'r0@h32:H128:gl0.0001' -> {'hidden': 128,
    'group_l1': 0.0001}. No tokens -> {} (the registered recipe: H = HIDDEN, no group lasso), so every existing spec,
    cell name and trainer argv is unchanged."""
    _, _, w = spec.partition("@h")
    out: dict = {}
    for tok in w.split(":")[1:]:
        if tok.startswith("H") and tok[1:].isdigit() and 8 <= int(tok[1:]) <= 512 and "hidden" not in out:
            out["hidden"] = int(tok[1:])
        elif tok.startswith("gl") and "group_l1" not in out and 0 < float(tok[2:]) <= 100:
            out["group_l1"] = float(tok[2:])
        else:
            raise ValueError(f"bad recipe token {tok!r} in {spec!r}")
    return out


def extra_arms() -> dict:
    """Arms over columns appended after f1824 (v2-canon only): `<root>/wide/extra_arms.json`
    {"schema": "rev4-featpot-v2c-extra-arms-v1", "width": W, "arms": {name: [column ids, each 1825 <= id < W]}},
    written by `v2c_wide.py keeplists --extra-arm`. Absent file -> no extra arms (the Rev3 v2 root has none)."""
    path = V2 / "wide" / "extra_arms.json"
    if not path.is_file():
        return {"width": WIDTH, "arms": {}}
    record = json.loads(path.read_text())
    if record.get("schema") != "rev4-featpot-v2c-extra-arms-v1":
        raise ValueError(f"{path}: unexpected schema")
    return record


def parse_spec(spec: str) -> tuple[str, int]:
    """'c1' -> ('c1', 0); 'c1~p2' -> ('c1', 2); an '@h<w>' suffix is accepted and ignored here (split_weight).
    Permutations exist for candidates (and extra arms), oracle_lo and oracle_hi (the sweep's null)."""
    core, _ = split_weight(spec)
    if block_spec(core):  # design log E9: no permuted controls (a direct Δ against r0 or core)
        return core, 0
    base, _, perm = core.partition("~p")
    k = int(perm) if perm else 0
    extras = extra_arms()["arms"]
    if base not in ("r0", *CANDIDATES, *CALIBRATION, *SCREEN, *extras) or not 0 <= k <= N_PERMS:
        raise ValueError(f"bad v2 arm spec {spec!r}")
    if k and base not in (*CANDIDATES, *extras, "oracle_lo", "oracle_hi"):
        raise ValueError(f"{spec!r}: permuted controls exist only for candidates and the oracles")
    return base, k


def all_specs() -> list[str]:
    specs = ["r0", *CALIBRATION, *SCREEN, *(f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1)),
             *(f"oracle_hi~p{k}" for k in range(1, N_PERMS + 1))]
    for arm in (*CANDIDATES, *extra_arms()["arms"]):
        specs += [arm, *(f"{arm}~p{k}" for k in range(1, N_PERMS + 1))]
    return specs


def selection_id(cols) -> str:
    """Name of a column subset in a `sel:<id>` spec (design log E9′ method 2 refits): the first 12 hex digits of the
    SHA-256 of the sorted column ids joined by commas. The ids themselves travel in the cell argv (`--columns`)."""
    return hashlib.sha256(",".join(map(str, sorted(cols))).encode()).hexdigest()[:12]


def block_spec(core: str) -> bool:
    """True for the E9 block specs: 'core', 'r0-<legacy block>', 'core+<legacy block | candidate arm | extra arm>',
    'set:<groups>', and 'sel:<12 hex>' (a column subset whose ids are passed with --columns)."""
    if core == "core":
        return True
    if core.startswith("sel:"):
        return len(core) == 16 and all(c in "0123456789abcdef" for c in core[4:])
    if core.startswith("set:"):  # E9″: any groups, basic and peaks included, nothing exempt
        parts = core[4:].split("+")
        ok = (*LEGACY_BLOCKS, *[a for a in (*CANDIDATES, *extra_arms()["arms"]) if a not in ("all", "rall")])
        return bool(core[4:]) and len(parts) == len(set(parts)) and all(x in ok for x in parts)
    if core.startswith("r0-"):
        return core[3:] in LEGACY_BLOCKS
    if core.startswith("core+"):
        parts = core[5:].split("+")
        ok = (*LEGACY_BLOCKS, *[a for a in (*CANDIDATES, *extra_arms()["arms"]) if a not in ("all", "rall")])
        return len(parts) == len(set(parts)) and all(x in ok for x in parts)
    return False


def pinned_arm(arm: str) -> tuple[str, str, list[int]]:
    """(table family, variant, kept columns) of one registered arm, read from the root's pinned keep lists
    (`wide/keep_lists.json`: sha-pinned by frozen.json and shipped in the fit-data archive). The fit program does not
    pack restore_data, so E9 `set:`/`core+` specs resolve their candidate arms here (2026-10-02: E9″ round 1 failed
    every cell with `No module named 'restore_data'`). A root without keep lists, or an arm they lack, falls back to
    arm_columns, which needs restore_data and fails loudly where it is absent."""
    path = V2 / "wide" / "keep_lists.json"
    if path.is_file():
        lists = json.loads(path.read_text())
        if lists.get("schema") != "rev4-featpot-v2-keeplists-v2":
            raise ValueError(f"{path}: keep-list schema mismatch")
        entry = lists["specs"].get(arm)
        if entry is not None:
            return entry["family"], entry["variant"], list(entry["keep"])
    return arm_columns(arm)


def arm_columns(spec: str) -> tuple[str, str, list[int]]:
    """(table family, variant, kept wide-column indices) for a spec (any '@h' suffix ignored)."""
    base, k = parse_spec(spec)
    if base.startswith("sel:"):
        raise ValueError(f"{spec}: a sel: spec carries its columns in the cell argv (--columns), not in a registry")
    variant = f"p{k}" if k else "real"
    bank = list(range(944))
    if base == "r0":
        return "main", variant, bank
    if base == "core":
        return "main", variant, list(CORE)
    if base.startswith("r0-"):
        drop = set(LEGACY_BLOCKS[base[3:]])
        return "main", variant, [c for c in bank if c not in drop]
    if base.startswith("core+") or base.startswith("set:"):  # E9′ core + S + X; E9″ any set of groups, no exempt core
        cols, fams = (set(CORE), set()) if base.startswith("core+") else (set(), set())
        for x in base.partition("+")[2].split("+") if base.startswith("core+") else base[4:].split("+"):
            if x in ("all", "rall"):
                raise ValueError(f"{spec}: unions are not lean-base arms")
            if x in LEGACY_BLOCKS:
                cols |= set(LEGACY_BLOCKS[x])
                continue
            fam, _, ids = pinned_arm(x)
            fams.add(fam)
            cols |= {c for c in ids if c >= 944}
        if len(fams) > 1:  # the peer pair exists only in the aux table, the main research columns only in main
            raise ValueError(f"{spec}: mixes aux-table and main-table families")
        return (fams.pop() if fams else "main"), variant, sorted(cols)
    if base == "minus_basic":
        return "main", variant, list(range(228, 944))
    if base in AUX_ORACLE:
        return "aux", variant, bank + [AUX_ORACLE[base]]
    extras = extra_arms()
    if base == "screen_aux":
        return "aux", variant, bank + [AUX_PEERS["gmsd"], AUX_PEERS["gmsm"], *AUX_GMSBANK]
    if base == "screen_main":
        cols = set()
        for arm in CANDIDATES:
            fam, _, ids_ = arm_columns(arm)
            if fam == "main":
                cols |= set(ids_[944:])
        for ids_ in extras["arms"].values():
            cols |= set(ids_)
        return "main", variant, bank + sorted(cols)
    if base in extras["arms"]:  # columns appended after f1824 (v2-canon)
        ids = extras["arms"][base]
        if any(not WIDTH <= c < extras["width"] for c in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{spec}: extra-arm columns outside {WIDTH}..{extras['width']}")
        return "main", variant, bank + list(ids)
    import restore_data  # registered arm definitions (pinned JSONs); not in the fit program, so only reached here

    ids = restore_data.arm_ids(base)
    if any(c < 0 for c in ids):  # the peer pair: arm p3 lives in the aux family
        added = [AUX_PEERS["gmsd"] if c == -1 else AUX_PEERS["gmsm"] if c == -2 else c for c in ids]
        if not set(added) <= set(AUX_ADDED):
            raise ValueError(f"{spec}: peer arm columns outside the aux family")
        return "aux", variant, bank + added
    if any(not 944 <= c < WIDTH for c in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{spec}: added columns outside the main layout")
    return "main", variant, bank + list(ids)


def seeds(heldout: str, seed_index: int) -> tuple[int, int]:
    fold = SOURCE_ORDER.index(heldout)
    return INIT_SEEDS[seed_index], SAMPLE_SEEDS[(seed_index + fold) % len(SAMPLE_SEEDS)]


def confirm_seeds(seed_index: int) -> tuple[int, int]:
    """Full-data (confirmatory) fits hold out no source, so there is no fold offset: init[i], sample[i]."""
    return INIT_SEEDS[seed_index], SAMPLE_SEEDS[seed_index]


def human_dev(ref: str) -> bool:
    return int(hashlib.sha256(ref.encode()).hexdigest(), 16) % HUMAN_DEV_MODULUS == 0


def acceptance_weight(nominal: float, refs) -> float:
    """R915's within-ref correction: w / mean over refs with n >= 2 of (1 - 1/n)."""
    import collections
    counts = collections.Counter(refs)
    eligible = [n for n in counts.values() if n >= 2]
    return nominal / (sum(1 - 1 / n for n in eligible) / len(eligible))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
