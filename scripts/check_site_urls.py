#!/usr/bin/env python3
"""Audit URL references in the zensim site for 404s.

Probes both kinds of URL the site emits:

1. **Local relative paths** — paths like `data/parquet/cid22.parquet`,
   `data/bakes/V0_18_seed42.json`, `weights/v0_16.bin`. These resolve
   against `site/` on disk; the check is "file exists."

2. **Remote R2 / CDN URLs** — `https://zentrain-r2.imazen.org/...`
   and `https://cdn.jsdelivr.net/...`. The check is an HTTP HEAD
   returning a 2xx.

URLs are discovered three ways:
- Static strings in `site/**/*.js` and `site/**/*.html` (anything that
  looks like a relative URL or HTTPS URL ending in `.json`, `.bin`,
  `.parquet`, or `.html`).
- Templated paths the JS expands at runtime — currently
  `weights/${bakeId}.bin`, `data/step5_bands/${label}.json`,
  `data/scatter/${label}.json`. The script enumerates the label set
  by reading the corresponding hard-coded JS arrays (matching
  `const labels = [...]` / `const scatterLabels = [...]`).
- The `data/index.json` manifest's `bakes[].json` field, which lists
  bake-summary JSONs under `data/bakes/`.

Exit code 0 on all-pass, 1 on any 404 / file-not-found.

Usage:
    python3 scripts/check_site_urls.py [--no-remote]

Optional flag:
- `--no-remote` skips HTTPS HEAD probes (use in offline / CI dry-runs).
- `--strict` treats any remote 404 as a failure (default: warn-only).

Output: a small report with a pass/fail per URL.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1] / "site"

# Regex for relative URL strings inside JS/HTML.
REL_PATTERNS = [
    re.compile(r'"((?:data|weights)/[A-Za-z0-9_./-]+\.(?:json|bin|parquet|html))"'),
    re.compile(r"`((?:data|weights)/[A-Za-z0-9_./-]+\.(?:json|bin|parquet|html))`"),
]

# Regex for remote https URLs in JS/HTML.
REMOTE_PATTERNS = [
    re.compile(r'"(https://[^"]+\.(?:json|bin|parquet))"'),
    re.compile(r"`(https://[^`]+\.(?:json|bin|parquet))`"),
]

# Templated paths the JS expands at runtime. Each entry is
# (file, regex-on-arrays-of-labels, path-template-with-${label}).
TEMPLATE_LABELS = [
    # app.js step-5 + scatter loops use the same labels list
    {
        "file": "site/js/app.js",
        "label_regex": re.compile(
            r"const\s+(?:labels|scatterLabels)\s*=\s*\[([^\]]+)\];"
        ),
        "templates": [
            "data/step5_bands/{label}.json",
            "data/scatter/{label}.json",
        ],
    },
    # compare-worker.js bakeUrl uses weights/<id>.bin; the dropdown
    # IDs come from compare.js's `score_zensim_v0_<n>(_jsmlp)?` entries.
    # We pull those from the dropdown labels and strip the prefix.
    {
        "file": "site/js/compare.js",
        "label_regex": re.compile(
            r'id:\s*"score_zensim_v0_(\d+|\d+_[a-z_0-9]+)"'
        ),
        "templates": [
            # Only attempted on dropdown change → match bake file
            # convention. We DO NOT probe weights/v0_<id>.bin
            # unconditionally because many dropdown rows route to a
            # parquet column instead of a JS-MLP bake; instead, only
            # probe a hard-coded list of known JS-MLP labels:
            #   v0_4_jsmlp, v0_16_jsmlp, v0_20, v0_22
            # plus the new V0_18 SHIP if a bake gets uploaded.
        ],
    },
]

# Known JS-MLP weights expected under site/weights/.
KNOWN_JSMLP_BAKES = ["v0_4", "v0_16", "v0_20", "v0_22"]


def collect_local_urls():
    """Walk site/ for static .js/.html files and pull relative URL refs."""
    found = set()
    for f in SITE_ROOT.rglob("*.js"):
        text = f.read_text()
        for pat in REL_PATTERNS:
            for m in pat.findall(text):
                found.add(m)
    for f in SITE_ROOT.rglob("*.html"):
        text = f.read_text()
        for pat in REL_PATTERNS:
            for m in pat.findall(text):
                found.add(m)
    return found


def collect_remote_urls():
    """Walk site/ for static .js/.html and pull remote URL refs.

    Also expand the JS `${R2_BASE}/path/...` templated URLs by reading
    the R2_BASE constant out of compare.js + compare-worker.js."""
    found = set()
    for f in SITE_ROOT.rglob("*.js"):
        text = f.read_text()
        for pat in REMOTE_PATTERNS:
            for m in pat.findall(text):
                found.add(m)
    for f in SITE_ROOT.rglob("*.html"):
        text = f.read_text()
        for pat in REMOTE_PATTERNS:
            for m in pat.findall(text):
                found.add(m)
    # Expand R2_BASE-prefixed template URLs in compare.js +
    # compare-worker.js. The JS literals look like
    # `${R2_BASE}/parquets/codec-sweeps/unified_v13_zenjpeg.parquet`.
    r2_base = None
    cjs = SITE_ROOT / "js/compare.js"
    if cjs.exists():
        m = re.search(r'const\s+R2_BASE\s*=\s*"([^"]+)"', cjs.read_text())
        if m:
            r2_base = m.group(1).rstrip("/")
    if r2_base:
        tmpl_pat = re.compile(
            r"`\$\{R2_BASE\}(/[A-Za-z0-9_./-]+\.(?:json|bin|parquet))`"
        )
        for f in SITE_ROOT.rglob("*.js"):
            text = f.read_text()
            for m in tmpl_pat.findall(text):
                found.add(r2_base + m)
    return found


def collect_templated_urls():
    """Expand the known runtime-templated paths into concrete URLs."""
    urls = set()
    # app.js step-5 + scatter
    app_js = (SITE_ROOT / "js/app.js").read_text()
    m = re.search(r"const\s+labels\s*=\s*\[([^\]]+)\]", app_js)
    if m:
        # Strip quotes + commas to get individual labels
        labels = re.findall(r"'([^']+)'", m.group(1))
        for lab in labels:
            urls.add(f"data/step5_bands/{lab}.json")
            urls.add(f"data/scatter/{lab}.json")
    # compare-worker.js JS-MLP bakes
    for bake in KNOWN_JSMLP_BAKES:
        urls.add(f"weights/{bake}.bin")
    return urls


def collect_bake_manifest_urls():
    """Read data/index.json and collect each bake's json file path."""
    urls = set()
    idx = SITE_ROOT / "data/index.json"
    if not idx.exists():
        return urls
    data = json.loads(idx.read_text())
    for b in data.get("bakes", []):
        if "json" in b:
            urls.add(f"data/{b['json']}")
    return urls


def check_local(url):
    """url is relative to site/. Return True if file exists."""
    full = SITE_ROOT / url
    return full.is_file()


def check_remote(url, timeout=5):
    """HTTP HEAD against url. Return (status_code, error_str|None)."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-remote", action="store_true",
                    help="Skip HTTPS HEAD probes (CI offline mode).")
    ap.add_argument("--strict", action="store_true",
                    help="Treat remote 404s as test failures (default: warn).")
    args = ap.parse_args()

    local = set()
    local |= collect_local_urls()
    local |= collect_templated_urls()
    local |= collect_bake_manifest_urls()
    remote = collect_remote_urls()

    print(f"Auditing {len(local)} local URLs + {len(remote)} remote URLs")
    print()

    failed_local = []
    for url in sorted(local):
        ok = check_local(url)
        mark = "OK " if ok else "404"
        print(f"  {mark}  local   {url}")
        if not ok:
            failed_local.append(url)

    failed_remote = []
    if not args.no_remote:
        print()
        for url in sorted(remote):
            status, err = check_remote(url)
            if err is not None:
                mark = f"ERR  {err}"
            elif status is None:
                mark = "???"
            elif 200 <= status < 400:
                mark = f"{status}"
            else:
                mark = f"{status}"
            print(f"  {mark}  remote  {url}")
            if status is None or status >= 400:
                failed_remote.append((url, status, err))

    print()
    print(f"local: {len(local) - len(failed_local)}/{len(local)} pass; "
          f"{len(failed_local)} fail")
    if not args.no_remote:
        print(f"remote: {len(remote) - len(failed_remote)}/{len(remote)} pass; "
              f"{len(failed_remote)} fail")

    if failed_local:
        print()
        print("LOCAL FAILURES:")
        for url in failed_local:
            print(f"  - {url}")

    if failed_remote and (args.strict or not args.no_remote):
        print()
        print("REMOTE FAILURES:")
        for url, status, err in failed_remote:
            print(f"  - {url}: status={status} err={err}")

    # Exit code: always fail on local 404. Remote fails only fail when
    # --strict (remote URLs are external and may transiently 404).
    fail_count = len(failed_local) + (len(failed_remote) if args.strict else 0)
    sys.exit(1 if fail_count else 0)


if __name__ == "__main__":
    main()
