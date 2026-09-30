"""Instrument v2 shared definitions.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md (with its revision R1). POTENTIAL —
ceiling, not a model score. Every value here is fixed by that amendment; change the amendment first.

Wide-table layout (one table per leg per variant; a cell selects its arm with --keep-features):
  f0..f943     bank Rev3 944 surface (39 structural zeros filled with 0, as admit_bank does)
  f944..f1824  Rev4 research vector at its canonical IDs (csfw/DVIFM, C1-C4, gmsbank, restore families)
  f1825, f1826 reviewed peer columns gmsd, gmsm (arm p3)
  f1827, f1828 calibration columns oracle_lo, oracle_hi
Variants: "real" and "p1".."p3" (columns f944..f1828 permuted jointly within reference over pair keys).
"""

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/var/tmp/rev4-featpot")
V2 = ROOT / "v2"
BIN_DIR = Path(os.environ.get("REV4_V2_BIN_DIR", "/var/tmp/fleet-fits/bin-v8"))
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
COL_GMSD, COL_GMSM, COL_ORACLE = 1825, 1826, {"oracle_lo": 1827, "oracle_hi": 1828}
WIDTH = 1829
N_PERMS = 3
VARIANTS = ("real", *(f"p{k}" for k in range(1, N_PERMS + 1)))
PERM_SEED_BASE = 20260930
ORACLE_SEED_BASE = 20260930
BOOT_B = 2000
BOOT_SEED = 20260930
REPLAY = "Rev4 POTENTIAL Instrument v2 diagnostic: pinned Rev3 944 plus pinned sidecars; never ship"


def parse_spec(spec: str) -> tuple[str, int]:
    """'c1' -> ('c1', 0); 'c1~p2' -> ('c1', 2). Permutations only for candidates and oracle_lo."""
    base, _, perm = spec.partition("~p")
    k = int(perm) if perm else 0
    if base not in ("r0", *CANDIDATES, *CALIBRATION) or not 0 <= k <= N_PERMS:
        raise ValueError(f"bad v2 arm spec {spec!r}")
    if k and base not in (*CANDIDATES, "oracle_lo"):
        raise ValueError(f"{spec!r}: permuted controls exist only for candidates and oracle_lo")
    return base, k


def all_specs() -> list[str]:
    specs = ["r0", *CALIBRATION, *(f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1))]
    for arm in CANDIDATES:
        specs += [arm, *(f"{arm}~p{k}" for k in range(1, N_PERMS + 1))]
    return specs


def arm_columns(spec: str) -> tuple[str, list[int]]:
    """(table variant, kept wide-column indices) for a spec."""
    import restore_data  # registered arm definitions (pinned JSONs)
    base, k = parse_spec(spec)
    variant = f"p{k}" if k else "real"
    bank = list(range(944))
    if base == "r0":
        return variant, bank
    if base == "minus_basic":
        return variant, list(range(228, 944))
    if base in COL_ORACLE:
        return variant, bank + [COL_ORACLE[base]]
    added = []
    for cid in restore_data.arm_ids(base):
        added.append({-1: COL_GMSD, -2: COL_GMSM}.get(cid, cid))
    if any(not 944 <= c < WIDTH for c in added) or len(set(added)) != len(added):
        raise ValueError(f"{spec}: added columns outside the wide layout")
    return variant, bank + added


def seeds(heldout: str, seed_index: int) -> tuple[int, int]:
    fold = SOURCE_ORDER.index(heldout)
    return INIT_SEEDS[seed_index], SAMPLE_SEEDS[(seed_index + fold) % len(SAMPLE_SEEDS)]


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
