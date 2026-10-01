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
PAIRS_PER_EPOCH = 50_000
HIDDEN = 32
HEADS = ("N", "F")

CANDIDATES = ("c1", "c2", "c3", "c4", "all", "csfw", "c7", "p1", "p3", "b1", "b1s",
              "c8n", "rall", "a1", "a1m", "b2", "b2m")
CALIBRATION = ("oracle_lo", "oracle_hi", "minus_basic")
ORACLE_SIGMA = {"oracle_lo": 1.5, "oracle_hi": 0.5}
FAMILIES = ("main", "aux")
WIDTH = 1825
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


def split_weight(spec: str) -> tuple[str, float | None]:
    """'oracle_hi~p1@h2' -> ('oracle_hi~p1', 2.0): the instrument-retune sweep's human-weight override (design log E2).
    No suffix -> (spec, None), i.e. NOMINAL_WEIGHT['human']."""
    core, _, w = spec.partition("@h")
    if not w:
        return core, None
    value = float(w)
    if not 0 < value <= 64:
        raise ValueError(f"bad human weight in {spec!r}")
    return core, value


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
    base, _, perm = core.partition("~p")
    k = int(perm) if perm else 0
    extras = extra_arms()["arms"]
    if base not in ("r0", *CANDIDATES, *CALIBRATION, *extras) or not 0 <= k <= N_PERMS:
        raise ValueError(f"bad v2 arm spec {spec!r}")
    if k and base not in (*CANDIDATES, *extras, "oracle_lo", "oracle_hi"):
        raise ValueError(f"{spec!r}: permuted controls exist only for candidates and the oracles")
    return base, k


def all_specs() -> list[str]:
    specs = ["r0", *CALIBRATION, *(f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1)),
             *(f"oracle_hi~p{k}" for k in range(1, N_PERMS + 1))]
    for arm in (*CANDIDATES, *extra_arms()["arms"]):
        specs += [arm, *(f"{arm}~p{k}" for k in range(1, N_PERMS + 1))]
    return specs


def arm_columns(spec: str) -> tuple[str, str, list[int]]:
    """(table family, variant, kept wide-column indices) for a spec (any '@h' suffix ignored)."""
    import restore_data  # registered arm definitions (pinned JSONs)
    base, k = parse_spec(spec)
    variant = f"p{k}" if k else "real"
    bank = list(range(944))
    if base == "r0":
        return "main", variant, bank
    if base == "minus_basic":
        return "main", variant, list(range(228, 944))
    if base in AUX_ORACLE:
        return "aux", variant, bank + [AUX_ORACLE[base]]
    extras = extra_arms()
    if base in extras["arms"]:  # columns appended after f1824 (v2-canon)
        ids = extras["arms"][base]
        if any(not WIDTH <= c < extras["width"] for c in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{spec}: extra-arm columns outside {WIDTH}..{extras['width']}")
        return "main", variant, bank + list(ids)
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
