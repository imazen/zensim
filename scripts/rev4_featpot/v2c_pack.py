"""Pack the v2-canon wide tables into one content-addressed fit-data archive (zenfleet-fit-data-v1).

Same archive format as zenmetrics `scripts/jobsys/pack_fit_data_v2.py` (which is bound to the Rev3 v2 root and its
receipt schema); members are `rev4-featpot/<name>/wide/...`, so a container that binds /var/tmp/rev4-featpot to the extracted
archive sees the canon root at /var/tmp/rev4-featpot/<name> (default `v2c`), the path every cell receives as `--root`.
Receipts hold root-relative table paths, so the tree can be built anywhere (default /var/tmp/canontab/v2c) and packed here.

  python v2c_pack.py --root /var/tmp/canontab/v2c --kind lodo    --out ~/tmp/canon-lodo.tar.gz
  python v2c_pack.py --root /var/tmp/canontab/v2c --kind confirm --out ~/tmp/canon-confirm.tar.gz

kinds: lodo = every leg of every selected variant dir (v2_lodo_mlp cells); confirm = only what v2_confirm_fit reads (the
two teacher legs, human_all fit/dev, receipts, the confirmatory feature tables and keys, keep lists); all = both.
Every table is hash-checked against its receipt before it is copied. The caller logs the transport in the exposure ledger.
"""

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

from v2_common import FAMILIES, VARIANTS, sha

CONFIRM_LEGS = ("safesyn", "cid22", "human_all")


def add_file(tar: tarfile.TarFile, path: Path, name: str) -> None:
    member = tar.gettarinfo(str(path), arcname=name)
    member.mtime = 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    with path.open("rb") as stream:
        tar.addfile(member, stream)


def keys_of(path: Path) -> Path:
    return path.with_name(path.name.replace(".parquet", ".keys.parquet"))


def receipt_parts(root: Path, rec: dict) -> list[Path]:
    """Every file one leg's receipt record names (tables, their manifests, row-identity keys), hash-checked."""
    out = []
    for part in ("full", "fit", "dev"):
        if part not in rec:
            continue
        path = root / rec[part]["rel"]
        if sha(path) != rec[part]["sha256"] or sha(Path(f"{path}.manifest.json")) != rec[part]["manifest_sha256"]:
            raise ValueError(f"{path}: table changed after its receipt")
        out += [path, Path(f"{path}.manifest.json")]
        expected = rec["keys_sha256"] if part == "full" and "keys_sha256" in rec else rec[part].get("keys_sha256")
        if expected is not None:
            if sha(keys_of(path)) != expected:
                raise ValueError(f"{keys_of(path)}: keys changed after the receipt")
            out.append(keys_of(path))
    return out


def members_for(root: Path, kind: str, selected: list[tuple[str, str]]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    files["wide/keep_lists.json"] = root / "wide" / "keep_lists.json"
    # v2_confirm_fit (and anything else calling v2_common.load_frozen) refuses a root without its freeze record, so every
    # archive carries it; load_frozen re-hashes the receipts / keep lists / extra arms it pins, all of which are packed too.
    if (root / "wide" / "frozen.json").is_file():
        files["wide/frozen.json"] = root / "wide" / "frozen.json"
        frozen = json.loads(files["wide/frozen.json"].read_text())
        if frozen.get("schema") == "rev5-recipe-admission-freeze-v1":
            from v2_common import load_frozen
            load_frozen(root, training_only=True)
            for rel in frozen["auxiliary_files"]:
                files[rel] = root / rel
    elif kind in ("confirm", "all"):
        raise ValueError(f"{root}: not frozen; the confirmatory fits refuse an unfrozen root (run `v2c_wide.py freeze`)")
    if (root / "wide" / "extra_arms.json").is_file():
        files["wide/extra_arms.json"] = root / "wide" / "extra_arms.json"
    for family, variant in selected:
        vdir = root / "wide" / family / variant
        receipt = json.loads((vdir / "receipt.json").read_text())
        if receipt.get("schema") != "rev4-featpot-v2c-wide-v1" or receipt["family"] != family or receipt["variant"] != variant:
            raise ValueError(f"{family}/{variant}: wide receipt identity mismatch")
        if not receipt.get("complete"):
            raise ValueError(f"{family}/{variant}: receipt incomplete (a leg is missing)")
        if "hdr" in receipt["legs"]:
            if kind != "lodo":
                raise ValueError("E26 HDR leg is registered for LODO only; VAL/confirmation packing refused")
            import v2_teacher
            import e21_cheap_recipe as e21
            v2_teacher.hdr_leg(receipt["legs"]["hdr"], e21.columns("by_v2fy"))
        files[f"wide/{family}/{variant}/receipt.json"] = vdir / "receipt.json"
        for leg, rec in receipt["legs"].items():
            if kind == "confirm" and leg not in CONFIRM_LEGS:
                continue
            for path in receipt_parts(root, rec):
                files[str(path.relative_to(root))] = path
    if kind in ("confirm", "all"):
        confirm = json.loads((root / "wide" / "confirm" / "receipt.json").read_text())
        files["wide/confirm/receipt.json"] = root / "wide" / "confirm" / "receipt.json"
        for name, rec in confirm["sets"].items():
            for family, by_variant in rec["tables"].items():
                for variant, table in by_variant.items():
                    if variant == "skipped" or (family, variant) not in {(f, v) for f, v in selected} | {(f, "real") for f in FAMILIES}:
                        continue
                    path = root / table["rel"]
                    if sha(path) != table["sha256"] or sha(Path(f"{path}.manifest.json")) != table["manifest_sha256"]:
                        raise ValueError(f"{path}: confirmatory table changed after its receipt")
                    keys = path.with_name(f"{name}.keys.parquet")
                    if sha(keys) != table["keys_sha256"]:
                        raise ValueError(f"{keys}: confirmatory keys changed after the receipt")
                    for p in (path, Path(f"{path}.manifest.json"), keys):
                        files[str(p.relative_to(root))] = p
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--name", default="v2c", help="root directory name inside the archive")
    ap.add_argument("--kind", choices=["lodo", "confirm", "all"], required=True)
    ap.add_argument("--select", action="append", default=None, metavar="FAMILY/VARIANT")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    every = [(f, v) for f in FAMILIES for v in VARIANTS]
    chosen = every if args.select is None else [tuple(s.split("/", 1)) for s in args.select]
    if any(c not in every for c in chosen) or len(set(chosen)) != len(chosen):
        raise SystemExit(f"--select: unknown or repeated variant directories {chosen}")
    receipt_path = args.root / "wide/main/real/receipt.json"
    if receipt_path.is_file() and "hdr_consensus" in json.loads(receipt_path.read_text()).get("legs", {}):
        from e29_consensus import preflight
        from v2_human_role import PRODUCTION_SOURCES
        for fold in PRODUCTION_SOURCES:
            for arm in ("hb4", "hc4"):
                preflight(args.root, fold, arm)
    selected = [c for c in every if c in chosen and (args.root / "wide" / c[0] / c[1] / "receipt.json").is_file()]
    if not selected:
        raise SystemExit("no selected variant receipts exist")
    revisions = {int(json.loads((args.root / "wide" / f / v / "receipt.json").read_text()).get("formula_revision", 4))
                 for f, v in selected}
    if len(revisions) != 1:
        raise SystemExit(f"mixed formula revisions in selected receipts: {revisions}")
    revision = revisions.pop()
    members = {f"rev4-featpot/{args.name}/{k}": v for k, v in members_for(args.root, args.kind, selected).items()}
    inventory = {"build_commit": json.loads((args.root / "wide/frozen.json").read_text()).get("build_commit"), "schema": "zenfleet-fit-data-v1", "label": "POTENTIAL — ceiling, not a model score",
                 "program": f"Rev{revision} potential Instrument v2-canon ({args.kind})", "variant_dirs": [f"{f}/{v}" for f, v in selected],
                 "files": {name: sha(path) for name, path in sorted(members.items())}}
    inv = json.dumps(inventory, sort_keys=True, indent=2).encode() + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            info = tarfile.TarInfo("input_inventory.json")
            info.size = len(inv)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(inv))
            for name, path in sorted(members.items()):
                add_file(tar, path, name)
    print(json.dumps({"sha256": sha(args.out), "bytes": args.out.stat().st_size, "members": len(members), "out": str(args.out)}))


if __name__ == "__main__":
    sys.exit(main())
