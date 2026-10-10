# E33 implementation evidence pointer (2026-10-10)

- Local root: `/mnt/v/output/zensim/e33-impl-2026-10-09/`.
- Tower mirror: `/mnt/tower/output/zensim-e33-impl-2026-10-09/`. Every file except the smokes'
  `container-scratch-*` data extractions and the data archive hardlink is mirrored (2,509 files after the
  review-fix re-mirror, sizes verified for every file). Three random files (`random.Random(20261010)`) plus
  `PACKET.json`, `program.tar.gz` and `PREDICTOR_PARITY.json` were SHA-256-checked. Two dense bakes left in the
  tower's `packet/` by earlier mirrors were renamed `*.stale-attempt{2,3}.bak`; identical copies are in
  `packet-attempt2/` and `packet-attempt3/`.
- Data archive: identical to the V40 control's `e29-fit-data.tar.gz` (`9c3eff1b…`), held in the V40 bundle and
  its mirror.
- Key files:

| File | SHA-256 |
|---|---|
| `packet/PACKET.json` (freeze inventory) | `3c96351e51ce6c8e0d214d53b84fcfba83d2f7286adcdd8e975d88340d249693` |
| `packet/program.tar.gz` | `ff419879cb2e67cfea14a0b95cf3027463dd087c639fac22a8da526bbf4929c6` |
| `packet/e33-fit-contract.json` | `3bb4705b8a07307b3dec3956f43d99ae649ea333b6c95817ac389167713ec4d7` |
| `predictor-parity-final2/PREDICTOR_PARITY.json` (review item 1, program binary) | `c12917218eea0eb15b0f6b5746d7ec8c6a7233f433f947055bfbab0e9fc30452` |
| `parity-kadid-final2/PARITY.json` (V40 one-cell parity, program trainer) | `c1a5289f8e7417dba51b15272c951cfd45072c28a74fb63b07295d67721291c1` |
| `E3_SOURCES.json` | `c15211539fc36f783ff351c23073df78d3cc172a53221def27834a7585891d88` |
| `scripts/rev4_featpot/e33_fx1_declaration.json` (repo) | `a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648` |

- Image (local, not pushed): `ghcr.io/imazen/zenfleet-worker:fit-e33-ff419879cb2e-wc581bdb55f88`,
  id `sha256:0b8a442cdcf9bb4341565821982d9e87728a9294cf90f3c1027511a24364ad91`.
- Re-resolve: `test -d /mnt/tower/output/zensim-e33-impl-2026-10-09/packet && sha256sum <file>` on both roots;
  `docker image inspect -f '{{.Id}}' <image>`.
