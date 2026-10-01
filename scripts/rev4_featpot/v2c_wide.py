"""v2-canon wide tables from the re-extracted Rev4 bank (CANONTAB lane, 2026-10-01).

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md (R1, R1.1, R2, R3). POTENTIAL — ceiling, not a
model score. The Rev3 builder `v2_wide.py` is untouched, so the Rev3 tables stay reproducible; this one reads
`/var/tmp/rev4-featbank-r4/<set>/{features.parquet,keys.parquet,_MANIFEST.json}` (pair_key, row_id, f0..f1824, float64,
formula Rev4, era tiercanon_c3negfold) and writes the same table layout under `--out`.

  python v2c_wide.py build    [--legs human|teachers|all] [--family F] [--variant V] [--extra FAM=FIRST:WIDTH:TMPL]
  python v2c_wide.py confirm  [--family F]      # features-only tables for the sealed confirmatory sets
  python v2c_wide.py keeplists [--extra-arm NAME=ID,ID,...]
  python v2c_wide.py verify                      # the registered gates (cast equality, row/key order, permutations)
  python v2c_wide.py freeze                      # after a clean verify of the FINAL build: pins every receipt in wide/frozen.json

Layout (same two 1,825+ wide families as R1.1, so kept columns keep identical first-layer initial weights):
  main: f0..f1824 straight from the bank (float64 cast to float32) + appended sidecar families (f1825.. width W)
  aux : bank f0..f943, gmsd f944, gmsm f945, oracle_lo f946, oracle_hi f947 (R3 distance orientation),
        gmsbank f1322..f1501, zeros elsewhere (width W). Needs the REEXTRACT peer output; without it only main builds.
Legs (as v2): five human sources (full table, keys, human_without_<held> fit/dev) + `human_all` fit/dev (union of the
five, same dev rule) + the two R915 teacher legs. pixels_identical keys are dropped everywhere. Human labels come from
the OLD bank's labels__*.parquet through `data.load`, exactly as v2 does; teacher SSIMULACRA2 targets come from the old
bank's labels__ssim2_oracle by pair_key under the committed teacher pin (the R915 dedup tables carry no pair_key; they
define the fit/dev reference split, and `verify` checks their targets against the bank's per reference).

SEALED LABELS: nothing here opens anything under `_sealed`. The confirmatory tables read only features.parquet,
keys.parquet and _MANIFEST.json of the Rev4 bank, through `bank_file`, which refuses any other path.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2_common import (AUX_GMSBANK, AUX_ORACLE, AUX_PEERS, FAMILIES, N_PERMS, ORACLE_SEED_BASE, ORACLE_SIGMA,
                       PERM_SEED_BASE, REPO, SOURCE_ORDER, SOURCES, TEACHERS, VARIANTS, WIDTH, human_dev, sha)

CANON_BANK = Path("/var/tmp/rev4-featbank-r4")
CANON_PEERS = CANON_BANK / "peer_gmsd"
OUT_DEFAULT = Path("/var/tmp/canontab/v2c")
SCHEMA = "rev4-featpot-v2c-wide-v1"
CONFIRM_SCHEMA = "rev4-featpot-v2c-confirm-v1"
# What the re-extraction (REEXTRACT lane) pinned; every manifest is refused if it differs.
CANON_FEATURE_SET_ID = ("basic+peaks+masked+iw+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+"
                        "mapdev+z1max+gmsnative+dvifmgate@w1825/tiercanon_c3negfold#d57e9571")
CANON_ERA = "tiercanon_c3negfold"
CANON_BINARY = "8c6f4c03695660fbb4fdce46e05859b40d9080497622cd726a7db75fd8323aa8"
CANON_BUILD = "259045b0acc7045cb3d4804217b81a9bfa1cbbad"
CANON_ERAS_TAIL = "c3negfold"
FORMULA_REVISION = 4
HUMAN_SETS = tuple(n for names in SOURCES.values() for n in names)
CONFIRM_SETS = ("cid22_b", "aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "csiq", "mcljci")
BANK_FILES = ("features.parquet", "keys.parquet", "_MANIFEST.json")
LEG_ORDER = (*SOURCE_ORDER, *TEACHERS)  # leg index drives the permutation and oracle seeds, as v2_wide
META = ["pair_key", "source_row_id", "ref_basename", "member_set", "target"]
KEY_COLUMNS = ["pair_key", "row_id", "ref_group", "pixels_identical"]


# ------------------------------------------------------------------ path guard (sealed labels)
def safe_path(path) -> Path:
    """Refuse any path with a `_sealed` component: the sealed label directory is never opened, listed or copied."""
    p = Path(path)
    if any(part == "_sealed" or part.startswith("_sealed") for part in p.parts):
        raise PermissionError(f"refusing a path under _sealed: {p}")
    return p


def bank_file(bank: Path, name: str, filename: str) -> Path:
    """The only way this module names a bank file: <bank>/<set>/<features|keys|manifest> of a known set."""
    if filename not in BANK_FILES:
        raise ValueError(f"{filename}: not a bank feature/key/manifest file")
    if name not in (*HUMAN_SETS, *CONFIRM_SETS, "cid22_train", "safesyn", "konjnd_bpg_train", "konjnd_bpg_val",
                    "kadid_terminal"):
        raise ValueError(f"{name}: not a Rev4 bank set")
    return safe_path(Path(bank) / name / filename)


# ------------------------------------------------------------------ extras (appended families)
@dataclass(frozen=True)
class Extra:
    name: str
    first: int
    width: int
    template: str  # path template with {set}

    def path(self, set_name: str) -> Path:
        return safe_path(self.template.format(set=set_name))

    @property
    def ids(self) -> list[int]:
        return list(range(self.first, self.first + self.width))


def parse_extras(specs: list[str]) -> list[Extra]:
    """`texgain=1825:12:/dir/{set}/features__signed_texgain.parquet`; contiguous from f1825 in the order given."""
    out, nxt = [], WIDTH
    for spec in specs or []:
        name, _, rest = spec.partition("=")
        first, width, template = rest.split(":", 2)
        extra = Extra(name, int(first), int(width), template)
        if extra.first != nxt or extra.width <= 0 or "{set}" not in template:
            raise ValueError(f"{spec}: extra families must be contiguous from f{nxt} and name {{set}}")
        nxt += extra.width
        out.append(extra)
    return out


def total_width(extras: list[Extra]) -> int:
    return WIDTH + sum(e.width for e in extras)


# ------------------------------------------------------------------ bank reading
@dataclass
class BankSet:
    name: str
    keys: pd.DataFrame      # KEY_COLUMNS in bank row order
    X: np.ndarray           # (rows, W) float32: f0..f1824 then appended families
    peers: np.ndarray | None  # (rows, 2) float32 gmsd, gmsm, or None
    receipt: dict


def check_manifest(manifest: dict, name: str) -> None:
    """Refuse a sidecar whose era/feature-set identity differs from the pinned canon (REEXTRACT consumer rule)."""
    want = {"set": name, "schema": "rev4-featbank-r4-v1", "feature_width": WIDTH, "formula_revision": "Rev4",
            "era_label": CANON_ERA, "feature_set_id": CANON_FEATURE_SET_ID, "binary_sha256": CANON_BINARY,
            "build_commit": CANON_BUILD, "dtype": "float64"}
    for key, value in want.items():
        if manifest.get(key) != value:
            raise ValueError(f"{name}: manifest {key}={manifest.get(key)!r} differs from the pinned canon {value!r}")
    eras = manifest.get("formula_revision_eras") or []
    if not eras or eras[-1] != CANON_ERAS_TAIL:
        raise ValueError(f"{name}: formula_revision_eras does not end in {CANON_ERAS_TAIL}")


def read_matrix(path: Path, ids: list[int], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """features.parquet -> (float32 matrix of columns f<ids>, pair_key array) in batches; float64 -> float32 once."""
    parquet = pq.ParquetFile(safe_path(path))
    names = parquet.schema_arrow.names
    cols = [f"f{i}" for i in ids]
    if names[:2] != ["pair_key", "row_id"] or any(c not in names for c in cols):
        raise ValueError(f"{path}: unexpected schema")
    out = np.empty((n_rows, len(cols)), dtype=np.float32)
    keys = []
    at = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=["pair_key", *cols]):
        n = batch.num_rows
        keys.append(batch.column(0).to_numpy(zero_copy_only=False))
        for j in range(len(cols)):
            out[at:at + n, j] = batch.column(j + 1).to_numpy(zero_copy_only=False)
        at += n
    if at != n_rows:
        raise ValueError(f"{path}: {at} rows, keys say {n_rows}")
    return out, np.concatenate(keys)


def gather_by_key(path: Path, columns: list[str], keys: np.ndarray, what: str) -> np.ndarray:
    """Columns of a per-set sidecar parquet (pair_key + columns), rows ordered by `keys`, float32; refuses missing,
    duplicate or non-finite rows."""
    table = pq.read_table(safe_path(path), columns=["pair_key", *columns]).to_pandas()
    if table.pair_key.duplicated().any():
        # a peer scorer emits one row per stimulus; collapsed stimuli repeat a key with equal values
        first = table.drop_duplicates("pair_key", keep="first")
        check = table.merge(first, on="pair_key", suffixes=("", "_first"), validate="many_to_one")  # joinsafety-ok: pair_key-keyed duplicate-consistency check, not a metric join
        for c in columns:
            if not (check[c] == check[f"{c}_first"]).all():
                raise ValueError(f"{what}: duplicate keys with different values in {c}")
        table = first
    index = pd.Index(table.pair_key).get_indexer(keys)
    if (index < 0).any():
        raise ValueError(f"{what}: {(index < 0).sum()} bank keys missing from {path}")
    values = table[columns].to_numpy()[index].astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{what}: non-finite values")
    return values


def load_bank_set(bank: Path, name: str, extras: list[Extra], peer_dir: Path | None = None) -> BankSet:
    manifest_path = bank_file(bank, name, "_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    check_manifest(manifest, name)
    keys_path, feats_path = bank_file(bank, name, "keys.parquet"), bank_file(bank, name, "features.parquet")
    keys_sha, feats_sha = sha(keys_path), sha(feats_path)
    if keys_sha != manifest["keys_sha256"] or feats_sha != manifest["features_parquet_sha256"]:
        raise ValueError(f"{name}: keys or features changed after the manifest")
    keys = pq.read_table(keys_path, columns=KEY_COLUMNS).to_pandas()
    if len(keys) != manifest["rows"] or keys.pair_key.duplicated().any() or not keys.row_id.is_monotonic_increasing:
        raise ValueError(f"{name}: keys disagree with the manifest")
    X, feat_keys = read_matrix(feats_path, list(range(WIDTH)), len(keys))
    if not np.array_equal(feat_keys, keys.pair_key.to_numpy()):
        raise ValueError(f"{name}: features.parquet and keys.parquet rows are not the same keys in the same order")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite feature cell")
    receipt = {"rows": len(keys), "manifest_sha256": sha(manifest_path), "keys_sha256": keys_sha,
               "features_sha256": feats_sha, "feature_set_id": manifest["feature_set_id"],
               "extractor_build_commit": manifest["build_commit"], "extractor_binary_sha256": manifest["binary_sha256"],
               "identical_rows": int(keys.pixels_identical.sum())}
    if extras:
        X = np.concatenate([X, np.empty((len(keys), sum(e.width for e in extras)), np.float32)], axis=1)
        for e in extras:
            path = e.path(name)
            cols = [f"f{i}" for i in e.ids]
            X[:, e.first:e.first + e.width] = gather_by_key(path, cols, keys.pair_key.to_numpy(), f"{name}/{e.name}")
            receipt.setdefault("extras", {})[e.name] = {"path": str(path), "sha256": sha(path),
                                                         "first": e.first, "width": e.width}
    peers = None
    if peer_dir is not None:
        pp = safe_path(Path(peer_dir) / f"{name}.parquet")
        if pp.is_file():
            peers = gather_by_key(pp, ["gmsd", "gmsm"], keys.pair_key.to_numpy(), f"{name}/peers")
            receipt["peers"] = {"path": str(pp), "sha256": sha(pp)}
            manifest = safe_path(Path(peer_dir) / "_MANIFEST.json")
            if manifest.is_file():
                receipt["peers"]["manifest_sha256"] = sha(manifest)
    return BankSet(name, keys, X, peers, receipt)


# ------------------------------------------------------------------ legs
@dataclass
class Leg:
    name: str
    meta: pd.DataFrame            # META (+ split for teachers)
    X: np.ndarray                 # (n, W) float32
    peers: np.ndarray | None
    bounds: list[float]
    y01: np.ndarray
    oracle: dict                  # oracle_lo / oracle_hi float32 columns


def oracle_columns(target: np.ndarray, leg_index: int) -> tuple[list[float], np.ndarray, dict]:
    """Per-leg q0.001/q0.999 bounds, y01, and the distance-oriented oracle columns of amendment R3 (as v2_wide)."""
    target = np.asarray(target, dtype=np.float64)
    lo, hi = (float(v) for v in np.quantile(target, [0.001, 0.999]))
    y01 = np.clip((target - lo) / (hi - lo), 0.0, 1.0)
    rng = np.random.default_rng(ORACLE_SEED_BASE + leg_index)
    sd = float(np.std(y01))
    cols = {}
    for name in ("oracle_lo", "oracle_hi"):
        cols[name] = ((1.0 - y01) + rng.normal(0.0, ORACLE_SIGMA[name] * sd, len(target))).astype(np.float32)
    return [lo, hi], y01, cols


def rows_from_bank(stim: pd.DataFrame, bank: BankSet) -> tuple[np.ndarray, np.ndarray | None]:
    """Feature (and peer) rows for stimulus rows keyed by pair_key; refuses any key missing from the bank."""
    index = pd.Index(bank.keys.pair_key).get_indexer(stim.pair_key)
    if (index < 0).any():
        raise ValueError(f"{bank.name}: {(index < 0).sum()} stimulus keys missing from the Rev4 bank")
    return bank.X[index], (None if bank.peers is None else bank.peers[index])


def human_member(name: str, bank: BankSet) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """One member set: stimulus rows (old bank labels by key, as data.load), identical keys dropped, Rev4 features."""
    import restore_data
    from data import load as load_old
    frame, meta = load_old(name, features=False)  # pair_key, source_row_id, ref_basename, target; one row per stimulus
    bank.receipt["labels"] = {"file": meta["label_file"], "sha256": sha(Path(meta["label_file"])),
                              "target": meta["target"], "label_scale": meta["label_scale"]}
    flags = restore_data.identical_flags(name)
    old = frame.pair_key.map(flags).astype(bool).to_numpy()
    new = bank.keys.set_index("pair_key").pixels_identical.reindex(frame.pair_key).to_numpy()
    if np.isnan(new.astype(float)).any() or not np.array_equal(old, new.astype(bool)):
        raise ValueError(f"{name}: pixels_identical differs between the admitted table and the Rev4 keys")
    stim = frame.loc[~old].reset_index(drop=True)
    ref_group = bank.keys.set_index("pair_key").ref_group.reindex(stim.pair_key).to_numpy()
    if not np.array_equal(ref_group.astype(str), stim.ref_basename.astype(str).to_numpy()):
        raise ValueError(f"{name}: ref_basename differs from the Rev4 ref_group")
    X, peers = rows_from_bank(stim, bank)
    stim["member_set"] = name
    return stim[META[:2] + ["ref_basename", "member_set", "target"]], X, peers


def make_leg(name: str, meta: pd.DataFrame, X: np.ndarray, peers) -> Leg:
    bounds, y01, oracle = oracle_columns(meta.target.to_numpy(), LEG_ORDER.index(name))
    return Leg(name, meta.reset_index(drop=True), X, peers, bounds, y01, oracle)


def human_leg(source: str, bank: Path, extras: list[Extra], peer_dir: Path | None, receipts: dict) -> Leg:
    metas, xs, ps = [], [], []
    for member in SOURCES[source]:
        b = load_bank_set(bank, member, extras, peer_dir)
        receipts[member] = b.receipt
        m, x, p = human_member(member, b)
        metas.append(m), xs.append(x), ps.append(p)
        del b
    refsets = [set(m.ref_basename.astype(str)) for m in metas]
    if len(metas) > 1 and refsets[0] & refsets[1]:
        raise ValueError(f"{source}: member sets share references")
    peers = None if any(p is None for p in ps) else np.concatenate(ps)
    return make_leg(source, pd.concat(metas, ignore_index=True), np.concatenate(xs), peers)


def teacher_leg(leg: str, bank: Path, extras: list[Extra], peer_dir: Path | None, receipts: dict) -> Leg:
    """R915 teacher leg: Rev4 features, SSIMULACRA2 targets from the pinned old-bank label file by pair_key, split by the
    pinned R915 reference partition (fit / dev / excluded), identical keys dropped."""
    import v2_wide
    bank_name, _, _ = TEACHERS[leg]
    files = v2_wide.pinned(leg)
    b = load_bank_set(bank, bank_name, extras, peer_dir)
    receipts[bank_name] = b.receipt
    b.receipt["labels"] = {"teacher_pin": str(v2_wide.TEACHER_PIN), "teacher_pin_sha256": sha(v2_wide.TEACHER_PIN),
                           **{k: {"file": str(files[k]), "sha256": sha(files[k])} for k in ("labels", "r915_fit", "r915_dev")}}
    old_keys = pq.read_table(files["keys"], columns=["pair_key"]).to_pandas().pair_key.to_numpy()
    if not np.array_equal(old_keys, b.keys.pair_key.to_numpy()):
        raise ValueError(f"{leg}: Rev4 key order differs from the old bank's")
    labels = pq.read_table(files["labels"], columns=["pair_key", "source_row_id", "ssim2_oracle"]).to_pandas()
    index = pd.Index(b.keys.pair_key).get_indexer(labels.pair_key)
    if (index < 0).any() or labels.pair_key.duplicated().any():
        raise ValueError(f"{leg}: label keys do not map one-to-one into the Rev4 bank")
    ident = b.keys.pixels_identical.to_numpy()[index].astype(bool)
    fit = set(pq.read_table(files["r915_fit"], columns=["ref_basename"])["ref_basename"].to_pylist())
    dev = set(pq.read_table(files["r915_dev"], columns=["ref_basename"])["ref_basename"].to_pylist())
    ref_group = b.keys.ref_group.to_numpy()[index].astype(str)
    stripped = pd.Series(ref_group).str.split(":", n=1).str[1]
    split = np.where(stripped.isin(fit), "fit", np.where(stripped.isin(dev), "dev", "excluded"))
    keep = ~ident & (split != "excluded")
    sel = index[keep]
    meta = pd.DataFrame({"pair_key": labels.pair_key.to_numpy()[keep], "source_row_id": labels.source_row_id.to_numpy()[keep],
                         "ref_basename": ref_group[keep], "member_set": bank_name,
                         "target": labels.ssim2_oracle.to_numpy(dtype=np.float64)[keep], "split": split[keep]})
    peers = None if b.peers is None else b.peers[sel]
    X = b.X[sel]
    del b
    return make_leg(leg, meta, X, peers)


# ------------------------------------------------------------------ views, permutations, writing
def permute_within_reference(refs: np.ndarray, keys: np.ndarray, vals: np.ndarray, selector: np.ndarray, rng) -> np.ndarray:
    """Joint key-level permutation of the columns of `vals` within each reference (numpy form of
    restore_data._permute_within_reference, EFFAUDIT D7: same draws, same result; checked against it in the tests)."""
    new = vals.copy()
    for ref in sorted(set(refs.tolist())):
        rows = np.flatnonzero((refs == ref) & selector)
        _, first, inverse = np.unique(keys[rows], return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        source = rows[first[order]]
        new[rows] = vals[source[rng.permutation(len(order))]][rank[inverse.reshape(-1)]]
    return new


def aux_columns(width: int) -> list[int]:
    return sorted({*AUX_PEERS.values(), *AUX_ORACLE.values(), *AUX_GMSBANK})


def family_matrix(leg: Leg, family: str, width: int) -> tuple[np.ndarray, list[int]]:
    """(n, width) float32 matrix for a family and the permutable (added) column indices of that family."""
    if family == "main":
        return leg.X, list(range(944, width))
    x = np.zeros((len(leg.X), width), dtype=np.float32)
    x[:, :944] = leg.X[:, :944]
    x[:, AUX_PEERS["gmsd"]] = leg.peers[:, 0]
    x[:, AUX_PEERS["gmsm"]] = leg.peers[:, 1]
    for name, col in AUX_ORACLE.items():
        x[:, col] = leg.oracle[name]
    x[:, AUX_GMSBANK[0]:AUX_GMSBANK[-1] + 1] = leg.X[:, AUX_GMSBANK[0]:AUX_GMSBANK[-1] + 1]
    return x, aux_columns(width)


def variant_matrix(leg: Leg, family: str, variant: str, width: int) -> np.ndarray:
    x, added = family_matrix(leg, family, width)
    k = 0 if variant == "real" else int(variant[1:])
    if not k:
        return x
    salt = 0 if family == "main" else 500
    rng = np.random.default_rng(PERM_SEED_BASE + salt + k + 1000 * LEG_ORDER.index(leg.name))
    out = x.copy()
    out[:, added] = permute_within_reference(leg.meta.ref_basename.to_numpy(), leg.meta.pair_key.to_numpy(),
                                             x[:, added], np.ones(len(x), dtype=bool), rng)
    return out


def write_table(root: Path, path: Path, ref: np.ndarray, score: np.ndarray, X: np.ndarray, family: str, note: str,
                width: int) -> dict:
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"f{i}" for i in range(width)]
    arrays = [pa.array(ref.astype(str)), pa.array(np.asarray(score, dtype=np.float64))]
    arrays += [pa.array(np.ascontiguousarray(X[:, j])) for j in range(width)]
    table = pa.Table.from_arrays(arrays, names=["ref_basename", "human_score", *names])
    # byte-stream-split floats, as v2_wide.write since d029bebc (decoded values identical; EFFAUDIT D6)
    pq.write_table(table, path, compression="zstd", use_dictionary=["ref_basename"],
                   use_byte_stream_split=["human_score", *names])
    layout = (f"Rev4 bank f0-f1824 (formula Rev4, era {CANON_ERA})" + (f" + appended families to f{width - 1}" if width > WIDTH else "")
              if family == "main" else
              "bank f0-f943, gmsd f944, gmsm f945, oracle_lo f946, oracle_hi f947, gmsbank f1322-f1501, zeros elsewhere")
    Path(f"{path}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": CANON_FEATURE_SET_ID,
        "composite": f"Rev4 POTENTIAL Instrument v2-canon {family} wide table ({layout}); diagnostic only; " + note,
        "formula_revision": FORMULA_REVISION}) + "\n")
    return {"rel": str(path.relative_to(root)), "sha256": sha(path), "manifest_sha256": sha(Path(f"{path}.manifest.json")),
            "rows": len(ref), "references": int(len(set(ref.astype(str).tolist())))}


def write_keys(path: Path, frame: pd.DataFrame, columns: list[str]) -> str:
    pq.write_table(pa.Table.from_pandas(frame[columns].reset_index(drop=True), preserve_index=False), safe_path(path),
                   compression="zstd")
    return sha(path)


def code_identity() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    here = Path(__file__).resolve().parent
    return {"build_commit": run(["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"]),
            "working_copy_clean": run(["jj", "diff", "--stat", "scripts/rev4_featpot"]) == "",
            "script_sha256": {n: sha(here / n) for n in ("v2c_wide.py", "v2_common.py")}}


def build(bank: Path, out: Path, legs: str, families: list[str], variants: list[str], extras: list[Extra],
          peer_dir: Path | None) -> None:
    refuse_if_frozen(out)
    width = total_width(extras)
    names = set(legs.split(","))
    if "all" in names:
        names |= {"human", *TEACHERS}
    if "teachers" in names:
        names |= set(TEACHERS)
    if not names <= {"human", "all", "teachers", *TEACHERS}:
        raise ValueError(f"unknown --legs {legs!r}")
    wanted_human = "human" in names
    identity = code_identity()
    all_receipts: dict[str, dict] = {}
    human_parts: dict[tuple, dict] = {}
    recs: dict[tuple, dict] = {}
    todo = (list(SOURCE_ORDER) if wanted_human else []) + [t for t in TEACHERS if t in names]
    skipped_aux = None
    for leg_name in todo:
        bank_receipts: dict = {}
        leg = (human_leg if leg_name in SOURCES else teacher_leg)(leg_name, bank, extras, peer_dir, bank_receipts)
        all_receipts.update(bank_receipts)
        for family in families:
            if family == "aux" and leg.peers is None:
                skipped_aux = "peer columns (gmsd/gmsm) missing from the REEXTRACT peer output; aux not built"
                continue
            for variant in variants:
                x = variant_matrix(leg, family, variant, width)
                dest = out / "wide" / family / variant
                rec = {"bounds": leg.bounds}
                ref = leg.meta.ref_basename.to_numpy()
                if leg_name in SOURCES:
                    score = 100.0 * leg.y01
                    rec["full"] = write_table(out, dest / f"{leg_name}.parquet", ref, score, x, family,
                                              f"human source {leg_name} (evaluation)", width)
                    rec["keys_sha256"] = write_keys(dest / f"{leg_name}.keys.parquet", leg.meta, META)
                    dev = np.array([human_dev(r) for r in ref.astype(str)])
                    human_parts.setdefault((family, variant), {})[leg_name] = (leg.meta[META], x, score, dev)
                else:
                    score = leg.meta.target.to_numpy(dtype=np.float64)  # raw signed SSIMULACRA2, --target-scale 1
                    for split in ("fit", "dev"):
                        m = (leg.meta.split == split).to_numpy()
                        rec[split] = write_table(out, dest / f"{leg_name}_{split}.parquet", ref[m], score[m], x[m], family,
                                                 f"teacher {leg_name} {split}", width)
                        # row-identity sidecar (pair_key per table row) so verify can check every cell against the bank
                        rec[split]["keys_sha256"] = write_keys(dest / f"{leg_name}_{split}.keys.parquet", leg.meta[m], META)
                recs.setdefault((family, variant), {})[leg_name] = rec
                del x
        print(json.dumps({"leg": leg_name, "rows": len(leg.meta)}), flush=True)
        del leg
    for (family, variant), parts in human_parts.items():
        dest = out / "wide" / family / variant
        if set(parts) != set(SOURCE_ORDER):
            continue
        groups = {f"human_without_{held}": [s for s in SOURCE_ORDER if s != held] for held in SOURCE_ORDER}
        groups["human_all"] = list(SOURCE_ORDER)
        for gname, members in groups.items():
            rec = {}
            for split, want_dev in (("fit", False), ("dev", True)):
                chunks = [(parts[t][0][parts[t][3] == want_dev], parts[t][1][parts[t][3] == want_dev],
                           parts[t][2][parts[t][3] == want_dev]) for t in members]
                ref = np.concatenate([c[0].ref_basename.to_numpy() for c in chunks])
                rec[split] = write_table(out, dest / f"{gname}_{split}.parquet", ref, np.concatenate([c[2] for c in chunks]),
                                         np.concatenate([c[1] for c in chunks]), family,
                                         f"human training leg {gname} ({'+'.join(members)}), {split}", width)
            recs.setdefault((family, variant), {})[gname] = rec
    for (family, variant), legs_rec in recs.items():
        path = out / "wide" / family / variant / "receipt.json"
        old = json.loads(path.read_text()) if path.is_file() else {"legs": {}, "bank": {}}
        if old.get("schema") not in (None, SCHEMA) or old.get("width", width) != width:
            raise ValueError(f"{path}: existing receipt has another schema or width; build into a clean --out")
        old["legs"].update(legs_rec)
        old["bank"].update(all_receipts)
        old.update({"schema": SCHEMA, "label": "POTENTIAL — ceiling, not a model score", "family": family,
                    "variant": variant, "width": width, "feature_set_id": CANON_FEATURE_SET_ID, "era": CANON_ERA,
                    "formula_revision": FORMULA_REVISION, "table_code": identity,
                    "extras": [{"name": e.name, "first": e.first, "width": e.width, "template": e.template} for e in extras],
                    "required_legs": [*SOURCE_ORDER, *(f"human_without_{h}" for h in SOURCE_ORDER), "human_all", *TEACHERS]})
        old["complete"] = all(leg in old["legs"] for leg in old["required_legs"])
        path.write_text(json.dumps(old, indent=1) + "\n")
        print(json.dumps({"receipt": str(path), "sha256": sha(path), "complete": old["complete"]}), flush=True)
    if skipped_aux:
        print(json.dumps({"aux_skipped": skipped_aux}), flush=True)


# ------------------------------------------------------------------ confirmatory tables (features only)
def build_confirm(bank: Path, out: Path, families: list[str], extras: list[Extra], peer_dir: Path | None,
                  variants: list[str] | None = None, sets: tuple[str, ...] = CONFIRM_SETS) -> dict:
    """Features-only tables for the sealed sets: `human_score` constant 0, keys without any target. Reads only
    features.parquet + keys.parquet + _MANIFEST.json of the Rev4 bank (bank_file refuses everything else). All variants by
    default: permuted ones are label-free (a within-reference key permutation needs only keys) and are the matched null.
    Layout: confirm/<family>/<set>.parquet (real), confirm/<family>/<variant>/<set>.parquet (permuted)."""
    refuse_if_frozen(out)
    width = total_width(extras)
    variants = variants or list(VARIANTS)  # matched null: every variant has its own label-free confirmatory table
    path = out / "wide" / "confirm" / "receipt.json"
    record = json.loads(path.read_text()) if path.is_file() else {
        "schema": CONFIRM_SCHEMA, "label": "features only; no label read or written", "width": width,
        "feature_set_id": CANON_FEATURE_SET_ID, "sets": {}}
    if record["schema"] != CONFIRM_SCHEMA or record["width"] != width:
        raise ValueError(f"{path}: existing confirm receipt has another schema or width; build into a clean --out")
    record["table_code"] = code_identity()
    for name in sets:
        b = load_bank_set(bank, name, extras, peer_dir)
        keep = ~b.keys.pixels_identical.to_numpy().astype(bool)
        keys = b.keys.loc[keep].reset_index(drop=True)
        meta = pd.DataFrame({"pair_key": keys.pair_key, "row_id": keys.row_id, "ref_basename": keys.ref_group,
                             "member_set": name})
        rec = record["sets"].setdefault(name, {"tables": {}})
        rec.update({"bank": b.receipt, "rows": int(keep.sum()), "identical_rows_dropped": int((~keep).sum()),
                    "dropped_pair_keys_sha256": hashlib.sha256("\n".join(b.keys.pair_key[~keep]).encode()).hexdigest()})
        leg = Leg(name, meta, b.X[keep], None if b.peers is None else b.peers[keep], [0.0, 1.0], np.zeros(keep.sum()),
                  {n: np.zeros(keep.sum(), np.float32) for n in AUX_ORACLE})
        for family in families:
            if family == "aux" and b.peers is None:
                rec["tables"].setdefault(family, {})["skipped"] = "peer columns missing from the REEXTRACT peer output"
                continue
            x, added = family_matrix(leg, family, width)  # aux oracle columns stay zero: an oracle needs a label
            for variant in variants:
                xv = x
                if variant != "real":
                    salt = 0 if family == "main" else 500
                    seed = PERM_SEED_BASE + salt + int(variant[1:]) + 1000 * (len(LEG_ORDER) + CONFIRM_SETS.index(name))
                    xv = x.copy()
                    xv[:, added] = permute_within_reference(meta.ref_basename.to_numpy(), meta.pair_key.to_numpy(),
                                                            x[:, added], np.ones(len(x), dtype=bool),
                                                            np.random.default_rng(seed))
                dest = out / "wide" / "confirm" / family / ("" if variant == "real" else variant)
                table = write_table(out, dest / f"{name}.parquet", meta.ref_basename.to_numpy(), np.zeros(len(meta)), xv,
                                    family, f"confirmatory set {name} ({variant}, features only)", width)
                table["keys_sha256"] = write_keys(dest / f"{name}.keys.parquet", meta,
                                                  ["pair_key", "row_id", "ref_basename", "member_set"])
                rec["tables"].setdefault(family, {}).pop("skipped", None)
                rec["tables"][family][variant] = table
        print(json.dumps({"confirm": name, "rows": rec["rows"], "variants": variants}), flush=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"receipt": str(path), "sha256": sha(path)}), flush=True)
    return record


# ------------------------------------------------------------------ keep lists
def refuse_if_frozen(out: Path) -> None:
    if (Path(out) / "wide" / "frozen.json").is_file():
        raise ValueError(f"{out}: frozen (wide/frozen.json); a changed table needs a new root or an explicit unfreeze by the coordinator")


def freeze(out: Path) -> str:
    """Freeze the canon root: record the hash of every receipt the cells and the read must see, after a clean `verify`."""
    import v2_common
    wide = Path(out) / "wide"
    refuse_if_frozen(out)
    verify = json.loads((wide / "verify.json").read_text())
    if not verify.get("all_ok"):
        raise ValueError("verify.json is not all_ok; fix and re-verify before the freeze")
    wide_receipts, widths, ids = {}, set(), set()
    for family in FAMILIES:
        for variant in VARIANTS:
            rp = wide / family / variant / "receipt.json"
            if not rp.is_file():
                raise ValueError(f"{family}/{variant}: receipt missing (build every family and variant before the freeze)")
            rec = json.loads(rp.read_text())
            if not rec.get("complete"):
                raise ValueError(f"{family}/{variant}: receipt incomplete")
            wide_receipts[f"{family}/{variant}"] = sha(rp)
            widths.add(rec["width"]), ids.add(rec["feature_set_id"])
    confirm = json.loads((wide / "confirm" / "receipt.json").read_text())
    for name in CONFIRM_SETS:
        tables = confirm["sets"][name]["tables"]
        for family in FAMILIES:
            for variant in VARIANTS:
                if variant not in tables.get(family, {}):
                    raise ValueError(f"confirm {name}/{family}/{variant}: table missing")
    if len(widths) != 1 or len(ids) != 1 or confirm["width"] not in widths:
        raise ValueError("widths or feature_set_ids disagree across receipts")
    extra = wide / "extra_arms.json"
    record = {"schema": v2_common.FROZEN_SCHEMA, "width": widths.pop(), "feature_set_id": ids.pop(), "wide_receipts": wide_receipts,
              "confirm_receipt_sha256": sha(wide / "confirm" / "receipt.json"), "keep_lists_sha256": sha(wide / "keep_lists.json"),
              "extra_arms_sha256": sha(extra) if extra.is_file() else None, "verify_sha256": sha(wide / "verify.json"),
              "frozen_at": __import__("time").strftime("%Y-%m-%d %H:%M %Z"), "table_code": code_identity()}
    path = wide / "frozen.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"frozen": str(path), "sha256": sha(path), "width": record["width"]}))
    return sha(path)


def write_keeplists(out: Path, extra_arms: dict[str, list[int]], width: int) -> None:
    import v2_common
    refuse_if_frozen(out)
    if extra_arms:
        path = out / "wide" / "extra_arms.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "rev4-featpot-v2c-extra-arms-v1", "width": width, "arms": extra_arms}) + "\n")
    if Path(v2_common.V2).resolve() != Path(out).resolve():
        raise ValueError("keeplists: pass --root <out> so spec parsing reads this root's extra arms")
    lists = {spec: dict(zip(("family", "variant", "keep"), v2_common.arm_columns(spec))) for spec in v2_common.all_specs()}
    path = out / "wide" / "keep_lists.json"
    path.write_text(json.dumps({"schema": "rev4-featpot-v2-keeplists-v2", "specs": lists}) + "\n")
    print(json.dumps({"keep_lists": str(path), "sha256": sha(path), "specs": len(lists)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "confirm", "keeplists", "verify", "freeze"])
    ap.add_argument("--bank", type=Path, default=CANON_BANK)
    ap.add_argument("--out", "--root", dest="out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--legs", default="all", help="comma list of: human, safesyn, cid22, teachers, all")
    ap.add_argument("--family", choices=FAMILIES, action="append")
    ap.add_argument("--variant", choices=VARIANTS, action="append")
    ap.add_argument("--extra", action="append", default=[], help="FAM=FIRST:WIDTH:TEMPLATE (TEMPLATE has {set})")
    ap.add_argument("--extra-arm", action="append", default=[], help="NAME=ID,ID,... (keeplists)")
    ap.add_argument("--peer-dir", type=Path, default=CANON_PEERS)
    ap.add_argument("--sample", type=int, default=10_000, help="verify: SafeSyn sample rows")
    args = ap.parse_args()
    import v2_common
    v2_common.V2 = args.out  # receipts hold root-relative table paths (table_path); the root is --out
    extras = parse_extras(args.extra)
    fams = args.family or list(FAMILIES)
    if args.action == "build":
        build(args.bank, args.out, args.legs, fams, args.variant or list(VARIANTS), extras, args.peer_dir)
    elif args.action == "confirm":
        build_confirm(args.bank, args.out, fams, extras, args.peer_dir, args.variant)
    elif args.action == "keeplists":
        arms = {}
        for spec in args.extra_arm:
            name, _, ids = spec.partition("=")
            arms[name] = [int(i) for i in ids.split(",")]
        write_keeplists(args.out, arms, total_width(extras))
    elif args.action == "freeze":
        freeze(args.out)
    else:
        from v2c_verify import verify
        sys.exit(verify(args.bank, args.out, extras, args.sample))


if __name__ == "__main__":
    main()
