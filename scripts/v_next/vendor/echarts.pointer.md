# Vendored Apache ECharts — pointer (bytes live in block storage, NOT in git)

The gauntlet dashboard (`scripts/v_next/gauntlet.py`) inlines Apache ECharts into the
emitted offline HTML at build time (the page's CSP allows no external requests). The
minified bundle is >30 KB so it is **never committed** — it lives in block storage and
this pointer records exactly which bytes the build must use. `build_html` verifies the
file's sha256 against this pointer and fails loudly on mismatch or absence.

- version: 5.6.0
- file: /mnt/v/zen/vendor/echarts/echarts-5.6.0.min.js
- sha256: bf4a223524e40b77c304bec67e1222cf551f14880cf42c69dc046558e11c07b1
- bytes: 1034102
- upstream: https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js
- license: Apache-2.0 (header retained in the bundle)
- fetched: 2026-08-04

Override the file location with env `ZEN_ECHARTS_JS` (sha256 is still checked).

To (re)fetch the exact bytes:

```sh
mkdir -p /mnt/v/zen/vendor/echarts
curl -sL -o /mnt/v/zen/vendor/echarts/echarts-5.6.0.min.js \
  https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js
sha256sum /mnt/v/zen/vendor/echarts/echarts-5.6.0.min.js
# must print bf4a223524e40b77c304bec67e1222cf551f14880cf42c69dc046558e11c07b1
```

Verified at vendor time: the bundle contains no `</script>` substring (safe to inline
in a script tag) and passes `node --check`. To bump the version: download the new
release the same way, update every field above IN THIS FILE (version, file, sha256,
bytes, upstream, fetched), and re-run the gauntlet gates — the build refuses a
file/pointer mismatch, so a half-done bump fails closed.
