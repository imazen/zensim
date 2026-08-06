#!/usr/bin/env python3
"""check_table_era.py — C3 (opus-review, campaign appendix W): given a table
sha256 (full or prefix), answer in one call whether it is a CURRENT canonical
table, a SUPERSEDED pre-correction artifact, or unknown.

The era-scoped-correction hazard this serves: the ext_kadid rebuild
(appendix H part 1) corrected the target IN PLACE and preserved the inverted
original; a bake's embedded `zentrain.repro` cites a sha, and whether that sha
is pre- or post-correction decides whether re-running the argv reproduces the
bake (it does NOT if the sha is the preserved inverted one — substitute the
`_INVERTED_` file). The manifests already record all of this
(`target_orientation` blocks with `corrected_sha256` /
`preserved_inverted_sha256`); this tool is the missing one-call reader.

    scripts/check_table_era.py 4dde6be2            # prefix ok
    scripts/check_table_era.py --json 286f1b23...

Exit codes: 0 CURRENT · 3 SUPERSEDED (pre-correction — repro hazard applies)
· 4 UNKNOWN (not in any scanned manifest).
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOTS = [
    "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01",
    "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27",
    "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-26",
    "/mnt/v/zen/zensim-training/canonical-2026-05-21",
    "/mnt/v/zen/zensim-training/canonical-2026-07-15",
]


def scan_manifest(path: Path, sha: str, hits: list) -> None:
    try:
        j = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    for e in j.get("entries", []) if isinstance(j, dict) else []:
        corpus = e.get("corpus", "?")
        cur = e.get("sha256", "")
        if cur.startswith(sha):
            hits.append({"era": "CURRENT", "corpus": corpus, "manifest": str(path),
                         "sha256": cur, "parquet": e.get("parquet")})
        corr = e.get("target_orientation") or {}
        inv = corr.get("preserved_inverted_sha256", "")
        if inv and inv.startswith(sha):
            hits.append({
                "era": "SUPERSEDED", "corpus": corpus, "manifest": str(path),
                "sha256": inv,
                "corrected_sha256": corr.get("corrected_sha256"),
                "corrected_utc": corr.get("corrected_utc"),
                "transform_applied": corr.get("transform_applied"),
                "preserved_file": corr.get("preserved_inverted_file"),
                "repro_hazard": corr.get("repro_hazard"),
            })
        # future correction-block shapes: any *_sha256 key inside a dict-valued
        # entry field is scanned, so a new correction class is found without
        # editing this tool.
        for k, v in e.items():
            if isinstance(v, dict) and k != "target_orientation":
                for kk, vv in v.items():
                    if kk.endswith("_sha256") and isinstance(vv, str) and vv.startswith(sha):
                        era = "SUPERSEDED" if "preserved" in kk or "old" in kk else "RELATED"
                        hits.append({"era": era, "corpus": corpus, "manifest": str(path),
                                     "sha256": vv, "via": f"{k}.{kk}"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sha", help="table sha256, full or prefix (>= 8 hex chars)")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS,
                    help="canonical roots whose _MANIFEST.json files are scanned")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sha = a.sha.lower().removeprefix("sha256:")
    if len(sha) < 8 or any(c not in "0123456789abcdef" for c in sha):
        sys.exit("check_table_era: sha must be >= 8 hex chars")

    hits: list = []
    for root in a.roots:
        r = Path(root)
        if not r.is_dir():
            continue
        for mf in sorted(r.glob("**/_MANIFEST.json")):
            scan_manifest(mf, sha, hits)

    if a.json:
        print(json.dumps({"query": sha, "hits": hits}, indent=1))
    else:
        if not hits:
            print(f"UNKNOWN {sha} — not in any scanned manifest "
                  f"({len(a.roots)} roots); pass --roots to widen")
        for h in hits:
            print(f"{h['era']}  {h['corpus']}  sha={h['sha256'][:16]}…  ({h['manifest']})")
            if h["era"] == "SUPERSEDED":
                print(f"  corrected {h.get('corrected_utc')} → {str(h.get('corrected_sha256'))[:16]}…"
                      f"  transform: {h.get('transform_applied')}")
                if h.get("repro_hazard"):
                    print(f"  REPRO HAZARD: {h['repro_hazard']}")
    if not hits:
        return 4
    if any(h["era"] == "SUPERSEDED" for h in hits):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
