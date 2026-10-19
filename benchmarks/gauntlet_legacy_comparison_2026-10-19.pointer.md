# Matched gauntlet B/D comparison artifacts

Host checked: dev. Local final HTML/JSON:
`/mnt/v/output/zensim/model-comparison-2026-10-19/`.
NAS mirror: `/mnt/tower/output/zensim/model-comparison-2026-10-19/`.

The root `index.html` is the concise comparison; `gauntlet.html` is the
shared interactive owner with all measured corpus columns.
`COMPARISON.json` retains full statistics; `fulleval/` contains the six
unchanged original rows and two new frozen B/D rows. `legacy/PLAN.json`
freezes populations/tools/models; `legacy/ROWS.json` binds exact ordered
pixels; `legacy/PARITY.json` records native parity. Fresh Rev1 features
and prediction/audit inputs remain separate from every Rev5 bank.

`_MANIFEST.json` and `ARCHIVE_FILES.json` own source/tool/file hashes and
the complete artifact inventory. `LAST_MIRROR_VERIFIED.json` records
full SHA-256 mirror verification. All final reports and JSON stay local
for the gallery. No R2 copy is claimed.

Reproduction: use the existing native extractor on `legacy/pairs.tsv`
with the copied B bake, revision1, default372 feature contract and native
audit; `just gauntlet-profile-audit` records that command. Then run
`just gauntlet-legacy-assess <original-gauntlet-root> <comparison-root>`
on a fresh prepared output. This requires the explicitly pinned PLAN,
ROWS and D audit pair inputs; it does not discover label corpora. Existing
outputs are preserved and refused as fresh-stage outputs.

Original study:
`/mnt/v/output/zensim/gauntlet-2026-10-18/`.
Fixed repair study:
`/mnt/v/output/zensim/color-noise-repair-2026-10-18/`.
Those completed artifact trees are unchanged.
