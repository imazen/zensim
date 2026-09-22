#!/usr/bin/env python3
"""Phase-2c DVIFM screen — codec-panel segment/admission builder.

Reproduces the September-13 scale study's codec-proxy panel selection
VERBATIM (scales944 executed driver, frozen 19,958-row admission), then
re-roles it under split-policy train-eval-only-v1:

  train  <- the frozen admission's 18 fit-role source families  (373 rows)
  eval   <- the frozen admission's 6 dev-role source families   (121 rows)
  DROP   <- the 6 test-role families (origins 1634/1220/8134/7050/7004/7058)
            are never opened, extracted or retagged (Sept-13 ruling).

Rows are the registered anchor panel: JPEG/WebP/AVIF-SVT/JXL knob sweeps over
the imazen-26 k-means references, five approximately equally spaced distinct
knob settings per source x codec (numeric knob order, both endpoints) plus one
identity row per source. Targets are score_ssim2 — signed, unclamped
SSIMULACRA2 full-reference PROXY labels, not human judgments.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/v/output/zensim/dvifm-screen2c-2026-09-19")
SEG = ROOT / "segments"
ANCHOR = Path("/mnt/v/output/zensim/ladder-2026-09-05/anchor")
LADDER_MANIFEST = Path("/mnt/v/output/zensim/ladder-2026-09-05/_MANIFEST.json")
ANCHOR_MANIFEST = ANCHOR / "out/_MANIFEST_anchor.json"
FAMILY_MAP = Path.home() / "work/zensim-validation-2026-09-08/canonical-corruption/split_map_family.tsv"
FAMILY_MAP_SHA = "9d07a0f63ef5fa167c5333535010f44b4ab9a087f04e560521b6d1aa1961820c"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ordered(values):
    return sorted(set(values), key=lambda x: hashlib.sha256(str(x).encode()).hexdigest())


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def main():
    SEG.mkdir(parents=True, exist_ok=True)
    assert sha(FAMILY_MAP) == FAMILY_MAP_SHA
    families = {r["id"]: r for r in read_tsv(FAMILY_MAP)}
    pinned = {"family_map": FAMILY_MAP_SHA, "ladder_manifest": sha(LADDER_MANIFEST),
              "anchor_manifest": sha(ANCHOR_MANIFEST), "pairs": {}, "labels": {}}

    # --- verbatim Sept-13 panel selection ---
    codec_rows = []
    for codec in ("jpeg", "webp", "avif_svt", "jxl"):
        pairs_file, labels_file = (ANCHOR / "grid" / sub / (codec + ".tsv")
                                   for sub in ("pairs", "tsv"))
        pinned["pairs"][codec] = sha(pairs_file)
        pinned["labels"][codec] = sha(labels_file)
        key = lambda r: (r["image_path"], r["codec"], r["q"], r["knob_tuple_json"])
        labels = {key(r): r for r in read_tsv(labels_file)}
        grouped = collections.defaultdict(dict)
        for r in read_tsv(pairs_file):
            r = dict(r, target=float(labels[key(r)]["score_ssim2"]))
            grouped[r["image_path"]][(r["q"], r["knob_tuple_json"])] = r
        assert len(grouped) == (30 if codec == "jxl" else 32)
        for ref, values in sorted(grouped.items()):
            origin = Path(ref).stem.split(".")[0]
            if families[origin]["split"] != "train":
                # Later family admission supersedes the older even-digit anchor.
                continue
            grid = sorted(values.values(), key=lambda r: (float(r["q"]),
                tuple(sorted(json.loads(r["knob_tuple_json"]).items()))))
            for i in sorted({round(j * (len(grid) - 1) / 4) for j in range(5)}):
                codec_rows.append((codec, grid[i]))
    origins = ordered(Path(r["ref_path"]).stem.split(".")[0] for _, r in codec_rows)
    assert len(origins) == 30
    source_families = {x: families[x]["family"] or "origin:" + x for x in origins}
    family_order = ordered(source_families.values())
    assert len(family_order) == 29
    # The Sept-13 frozen admission's family-level inner split, reproduced.
    family_roles = {x: "fit" if i < 18 else "dev" if i < 24 else "test"
                    for i, x in enumerate(family_order)}
    roles = {x: family_roles[source_families[x]] for x in origins}
    for origin, role in sorted(roles.items()):
        assert families[origin]["split"] == "train", origin  # all train-side content
    dropped = sorted(f for f, r in family_roles.items() if r == "test")
    dropped_origins = sorted(o for o in origins if roles[o] == "test")
    assert len(dropped) == 5 and len(dropped_origins) == 6

    legs = {"train": [], "eval": []}
    identities = {}
    for codec, r in codec_rows:
        origin = Path(r["ref_path"]).stem.split(".")[0]
        old = roles[origin]
        if old == "test":
            continue
        role = "train" if old == "fit" else "eval"
        legs[role].append(dict(task="codec", corpus="imazen_anchor", origin=origin,
                               source_family=source_families[origin],
                               family=codec, target=r["target"],
                               reference=r["ref_path"], distorted=r["dist_path"],
                               knob=r["knob_tuple_json"]))
        identities.setdefault(origin, (role, r["ref_path"]))
    for origin, (role, ref) in sorted(identities.items()):
        legs[role].append(dict(task="codec", corpus="imazen_anchor", origin=origin,
                               source_family=source_families[origin],
                               family="identity", target=100.0,
                               reference=ref, distorted=ref, knob="{}"))
    assert len(legs["train"]) == 373 and len(legs["eval"]) == 121

    rule = ("imazen_anchor codec-proxy panel: ladder-2026-09-05 anchor grid "
            "(5 knob settings + identity per source x codec, score_ssim2 proxy "
            "labels). All 30 origins are TRAIN-split under the Sept-8 canonical "
            "family map; the inner fit/dev partition is the Sept-13 frozen "
            "admission's family-level assignment reproduced verbatim; the six "
            "families it labelled 'test' are excluded entirely.")
    for role, rows in legs.items():
        fams = ordered(r["source_family"] for r in rows)
        sources = [dict(corpus="imazen_anchor", origin=o,
                        source_family=source_families[o], split=role)
                   for o in ordered({r["origin"] for r in rows})]
        authority = dict(
            path=str(FAMILY_MAP), sha256=FAMILY_MAP_SHA,
            ladder_manifest=str(LADDER_MANIFEST), ladder_manifest_sha256=sha(LADDER_MANIFEST),
            anchor_manifest=str(ANCHOR_MANIFEST), anchor_manifest_sha256=sha(ANCHOR_MANIFEST),
            rule=rule, dropped_families=dropped)
        admission = SEG / f"codec-{role}-admission.json"
        segment = SEG / f"codec-{role}-segment.json"
        save(admission, dict(schema="zensim-source-admission-v1",
                             authority={**authority, "dropped_origins": dropped_origins}, sources=sources))
        save(segment, dict(schema="zensim-feature-segment-v1", role=role, rows=rows))
        print(role, len(rows), len(sources), "families", len(fams),
              sha(segment), sha(admission))
    save(SEG / "codec_panel_pinned.json", pinned)


if __name__ == "__main__":
    main()
