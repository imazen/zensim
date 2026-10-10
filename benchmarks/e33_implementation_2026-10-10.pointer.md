# E33 implementation evidence pointer (2026-10-10)

- Local root: `/mnt/v/output/zensim/e33-impl-2026-10-09/`.
- Tower mirror: `/mnt/tower/output/zensim-e33-impl-2026-10-09/`. Every file except the smokes'
  `container-scratch-*` data extractions and the data archive hardlink is mirrored (2,048 files, sizes
  verified). Three random files were SHA-256-checked (`random.Random(20261010)`).
- Data archive: identical to the V40 control's `e29-fit-data.tar.gz` (`9c3eff1b…`), held in the V40 bundle and
  its mirror.
- Key files:

| File | SHA-256 |
|---|---|
| `packet/PACKET.json` (freeze inventory) | `fa0acbdffe23e4d9a1d1748a8227a320fba0f03a0bae31dd22ce3c7251885b18` |
| `packet/program.tar.gz` | `94b7c4c0bd38ba3e77b6ba7ced30c8182242669bb7f16056de141aa4b3bf0eda` |
| `packet/e33-fit-contract.json` | `0679e019e1471567b874892ad09e60a560b55fb146b8a22ec384785835b10a6f` |
| `E3_SOURCES.json` | `c15211539fc36f783ff351c23073df78d3cc172a53221def27834a7585891d88` |
| `scripts/rev4_featpot/e33_fx1_declaration.json` (repo) | `a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648` |

- Image (local, not pushed): `ghcr.io/imazen/zenfleet-worker:fit-e33-94b7c4c0bd38-wc581bdb55f88`,
  id `sha256:f9f4930c1d83b0c91086855726352bca020c785b386de9bcede3230a73d60354`.
- Re-resolve: `test -d /mnt/tower/output/zensim-e33-impl-2026-10-09/packet && sha256sum <file>` on both roots;
  `docker image inspect -f '{{.Id}}' <image>`.
