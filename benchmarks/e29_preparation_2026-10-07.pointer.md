# E29 local preparation pointer — 2026-10-07

Status: prepared for review with an unintended legacy HDR VAL read disclosed.
No full scientific E29 fit, fleet enrollment, source/image push or product
qualification. Coordinator registration must cover the four-source amendment,
exposure addendum and one matched-control choice before fitting.

Local artifacts: `/mnt/v/output/zensim/e29-2026-10-07/`.
Tower mirror: `/mnt/tower/output/zensim/e29-2026-10-07/`.
Prepared scorer root: `v2e29/` in the local artifact directory.

Data SHA256: `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`.
Program SHA256: `44ca24737293b404e29e150ebce78632cc21d5de7b6416c3058fb53c2100ef58`.
Image: `ghcr.io/imazen/zenfleet-worker:fit-e29-consensus-v41-w925f9783329f`,
ID `sha256:3b85136b13c37d30ae74d30a767bed49dca1e4d13643b26ad0ce4ae59335cb15`.

`PINNED_ARTIFACTS.json` (SHA256 `6cadfefaf8cdd20b34369371c95582c2fb9eae6d6b3b3ad4fc6662d8c3ea4148`) binds all binaries,
producer commits, 80-arm/40-proposed-control manifests, actual executor
smokes, admissions, resource records and the exposure receipt. Data build
`9d563474`, trainer `a4d1c3bd`, guarded program `f634504e`, profile `f15cb920`.
All 40 completed E30 nA3 cells are pinned in `E30_COMPLETE_PINS.json`; exact
E29 program reuse parity is not asserted. The fresh control remains a proposal.

`MIRROR_RECEIPT.json` contains every mirrored file hash and three randomly
selected file checks. Superseded v40 program/image/smokes are retained.
`UNINTENDED_EXPOSURE.json` records the import-test legacy 22,860-row/952-column
HDR VAL read. No student prediction or fit used those rows. The guarded import
and absent-control tripwires now pass, but the read cannot be undone.
Full worklog and role/exposure ledger: [E29_WORKLOG.md](E29_WORKLOG.md) and
[DATA_SPLITS.md](../docs/DATA_SPLITS.md). Large artifacts remain outside git.

Coordinator update: the solo archives/image listed above are historical
preparation evidence. E29/E31/E32 will use one combined v40 package and
one shared fresh matched control. The prepared data remains available
locally; no new solo package/image is produced. Full baseline parity evidence
is `control-parity-final/PARITY.json` in the same artifact root.
`CLEANUP_RECEIPT.json` binds the owned-stage/cache removal after tower audit;
tower `ARCHIVE_MIRROR_RECEIPT.json` preserves the original 11,020-file proof.
The final local subset and current code evidence have a separate
`FINAL_LOCAL_MIRROR_RECEIPT.json`; they are not a claim that all original
extraction stages still exist locally.
