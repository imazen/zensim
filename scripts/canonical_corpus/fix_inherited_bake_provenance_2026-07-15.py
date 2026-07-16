#!/usr/bin/env python3
"""One-shot repair: strip/recompute the outcome fields that 123 manifests
INHERITED from `v47_strict_qat.toml` by fork-and-edit.

# What was wrong

Measured 2026-07-15 across `zensim/weights/manifests/*.toml` (n=142):

  * **128 manifests recorded the same `[bake].sha256`** — `d0ef7a30…`, the
    sha of the *shipped Profile A bake* — and **the same `[eval]` block**
    (`cid22_srocc = 0.8657`, Profile A's number). Exactly ONE of them
    (`v47_strict_qat.toml`) actually describes that bake.
  * 142 manifests carry only **57 distinct `[training]` recipes** (seed
    excluded) and **7 distinct `[eval]` blocks**.

Every new experiment was created by copying `v47_strict_qat.toml`, editing
`seed` / `groups` / hyperparams, and leaving the outcome fields untouched.
A manifest's entire job is to identify a bake; 123 identified someone
else's.

# Why nothing caught it

`RawBake` parsed only `file`. `sha256` and `file_bytes` were dropped by
serde — inert prose in a TOML costume. `verify_inputs()` checked every
*input* hash; no code read the *output* hash. Fixed the same day:
`ManifestConfig::{bake_sha256, bake_file_bytes}` + `verify_bake()` +
`zensim-validate/tests/shipped_bake_provenance.rs`.

# What this script does

Only to manifests whose `[bake].sha256` == v47's AND whose `[bake].file`
is NOT the v47 bake (i.e. provably inherited):

  * `[bake].sha256` / `[bake].file_bytes` — **recomputed** from the bake at
    `[bake].file` when it is on disk (that is the honest value, and it makes
    `--manifest` self-checking); **removed** when it is not (claiming a hash
    for a bake you do not have is how this started).
  * `[eval]` — **removed**. It is a byte-copy of v47's numbers, so no unique
    data is lost, and it cannot be recomputed without re-running
    `bake_verdict` on 123 probe bakes.

`[inputs.*].sha256` is NEVER touched: those are correct and load-bearing
(`verify_inputs` depends on them). The script asserts the input-hash count
is unchanged per file.

Idempotent: a second run is a no-op (nothing still claims v47's sha).
"""

import hashlib
import pathlib
import re
import sys

V47_SHA = "d0ef7a3054d1ed9e70086d306cda69b71fc95072c6ef3351f362f27da096d4fc"
MANIFESTS = pathlib.Path(__file__).resolve().parents[2] / "zensim/weights/manifests"

NOTE = (
    "# [bake].sha256 / file_bytes were INHERITED from v47_strict_qat.toml by\n"
    "# fork-and-edit and described the shipped Profile A bake, not this one\n"
    "# (128 of 142 manifests carried the same wrong hash). Repaired 2026-07-15\n"
    "# by scripts/canonical_corpus/fix_inherited_bake_provenance_2026-07-15.py.\n"
    "# The inherited [eval] block was removed for the same reason: it was a\n"
    "# copy of v47's numbers (cid22_srocc = 0.8657), not this recipe's.\n"
    "# Enforced now by zensim-validate/tests/shipped_bake_provenance.rs.\n"
)


def bake_section_bounds(lines):
    """Return (start, end) line indices of the [bake] table body."""
    start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "[bake]":
            start = i + 1
            continue
        if start is not None and s.startswith("[") and not s.startswith("[["):
            return start, i
    return (start, len(lines)) if start is not None else (None, None)


def strip_table(lines, name):
    """Drop a whole top-level [name] table. Returns (lines, dropped?)."""
    out, dropping, dropped = [], False, False
    for ln in lines:
        s = ln.strip()
        if s == f"[{name}]":
            dropping, dropped = True, True
            continue
        if dropping and s.startswith("[") and not s.startswith("[["):
            dropping = False
        if not dropping:
            out.append(ln)
    return out, dropped


def main():
    apply = "--apply" in sys.argv
    changed = recomputed = removed = 0
    for f in sorted(MANIFESTS.glob("*.toml")):
        text = f.read_text()
        m = re.search(r'^sha256\s*=\s*"([^"]+)"', text, re.M)
        if not m or m.group(1).lower() != V47_SHA:
            continue
        fm = re.search(r'^file\s*=\s*"([^"]+)"', text, re.M)
        if fm and "v47_strict_qat_native" in fm.group(1):
            continue  # the one manifest that legitimately describes v47

        # A manifest may record v47's sha *correctly*: `v47_mainfix_repro`
        # is a REPRODUCTION of v47 whose bake hashes to d0ef7a30… exactly —
        # which is evidence that v47 reproduces byte-identically, not a
        # copy-paste. Repair only what is actually WRONG: if the recorded
        # sha already matches the bake on disk, leave the file alone. (This
        # also makes the script idempotent; the first draft re-flagged
        # v47_mainfix_repro on every run and, worse, stripped its
        # legitimately-v47-matching [eval].)
        if fm:
            _raw = fm.group(1)
            _p = (
                pathlib.Path(_raw)
                if _raw.startswith("/")
                else (MANIFESTS / _raw).resolve()
            )
            if _p.exists() and hashlib.sha256(_p.read_bytes()).hexdigest() == V47_SHA:
                continue

        lines = text.splitlines(keepends=True)
        b0, b1 = bake_section_bounds(lines)
        if b0 is None:
            print(f"  SKIP {f.name}: no [bake] table")
            continue

        inputs_before = len(re.findall(r'^sha256\s*=\s*"', text, re.M))

        # Resolve the bake this manifest actually produces.
        raw = fm.group(1) if fm else None
        bake = None
        if raw:
            p = pathlib.Path(raw) if raw.startswith("/") else (MANIFESTS / raw).resolve()
            bake = p if p.exists() else None

        new_bake, real_sha, real_len = [], None, None
        if bake:
            data = bake.read_bytes()
            real_sha, real_len = hashlib.sha256(data).hexdigest(), len(data)

        for ln in lines[b0:b1]:
            key = ln.split("=", 1)[0].strip() if "=" in ln else ""
            if key == "sha256":
                if real_sha:
                    new_bake.append(re.sub(r'"[^"]+"', f'"{real_sha}"', ln, count=1))
                continue
            if key == "file_bytes":
                if real_len is not None:
                    new_bake.append(re.sub(r"=\s*\d+", f"= {real_len}", ln, count=1))
                continue
            new_bake.append(ln)

        out = lines[:b0] + new_bake + lines[b1:]
        out, had_eval = strip_table(out, "eval")
        # Note goes above [bake] so it is the first thing read.
        bake_hdr = next(i for i, l in enumerate(out) if l.strip() == "[bake]")
        out = out[:bake_hdr] + [NOTE] + out[bake_hdr:]
        new_text = "".join(out)

        inputs_after = len(re.findall(r'^sha256\s*=\s*"', new_text, re.M))
        expect = inputs_before - 1 + (1 if real_sha else 0)
        assert inputs_after == expect, (
            f"{f.name}: sha256 line count {inputs_before}->{inputs_after}, expected {expect}. "
            "REFUSING to write — [inputs.*].sha256 must never be touched."
        )

        changed += 1
        recomputed += 1 if real_sha else 0
        removed += 1 if not real_sha else 0
        if apply:
            f.write_text(new_text)

    verb = "repaired" if apply else "WOULD repair (dry run; pass --apply)"
    print(
        f"{verb} {changed} manifests that inherited v47's outcome fields:\n"
        f"  {recomputed} had their bake on disk -> sha256/file_bytes RECOMPUTED\n"
        f"  {removed} had no bake on disk       -> sha256/file_bytes REMOVED\n"
        f"  [eval] (a copy of v47's numbers) removed from all {changed}"
    )


if __name__ == "__main__":
    main()
