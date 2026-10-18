# R8 severe-quantization evidence — 2026-10-18

Host checked: dev. Local `/mnt/v/zensim/r8-severe-quant-2026-10-18/run-01/`; full verified mirror `/mnt/tower/output/zensim/r8-severe-quant-2026-10-18/run-01/`. Every source→mirror→source SHA256 matched: 1483 files / 90835969 bytes. Three random JPEG payloads also verified. `LAST_MIRROR_VERIFIED.json` records selections and verifier identity; receipts exclude themselves from the inventory.

Both candidates pass corrected R8:48/48 (100%), strict base score below same-source q20; detector absent. All16 q20 anchor replays pass1e-4. Six levels frozen before validation encoding/new scoring; same48 encodes scored by both models. [All96 comparisons](r8_severe_quant_2026-10-18.tsv); [result append](r8_r10_2026-10-18.md). No labels or untouched confirmation opened. Scope: eight exposed battery validation families at native geometries, not broad product qualification.

Native source `99ab8b9264c3`, report adapter `d46a81a65f7f`; `BUILD_PINS.json` and `SOURCE_REVISIONS.json` bind source hashes. Binary SHA256 `857ec4b69ebca07f160366a9209dc1dd031d0603b56ec8cd7ad872bd3ad110e1`. Runtime base `95a6c7f4`; zenjpeg `4196fa93d422bc367d1e3ddadbaddfe672b9cdfc`, recompress-expert enabled, portable release. C model `89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`; W-C `f73059ff8f67ff177265d35dfd5ff5db61d8fc825a6b519cf6925dc8fc1d5e2a`.

Command: `just r8-severe-phase PHASE INPUTS AUDITS PLAN REGISTRATION BINARY /usr/bin/djpeg OUTPUT`; phases prepare,encode,score,report in order. Expanded argv and env in retained logs/command receipts. Serial shared lock plus `run-heavy --mem 8G --jobs 1`, disk TMPDIR. Final fixture/Clippy/build PASS (rc0,67s,peak-RSS1.14GiB,min-available29314MiB,peak-load10.89). Encode rc0,0s,peak-RSS0.03GiB,min-available37883MiB,peak-load2.78; score rc0,1s,peak-RSS0.03GiB,min-available38054MiB,peak-load2.26; archive rc0,13s,peak-RSS0.03GiB,min-available37788MiB,peak-load2.77. Resource receipts only, no speed claim. Minimum logged disk43,915,415,552bytes during encode/score; floor40GiB.

Exact component DQTs: DC-only(DC16/64/256,AC32767), coarse-AC(DC16,AC256/512/1024), YCbCr444, AQ/deringing/trellis off, neutral zero bias. Every DC-only AC coefficient zero. All48 outputs independently decode with native RGB identity.44 coefficient-preserving serializations restore requested literal DQTs only at zero coefficients; coefficients and both decoders' initial/final RGB remain exact. Original literal-DQT expectation retained. Five coarse rows become DC-only and remain included:5315/ac1024,6823/ac512,6823/ac1024,8113/ac1024,9113/ac1024.

`INPUTS.json` binds source/anchor hashes, saved anchor scores, original serving/audit/model identities, levels and99% rule. Source/anchor PNG copies verified. `encodes/`: initial/final JPEGs, DQT/coefficient census, RGB/PNG and oracle decodes. `scores/`:112 scalar/native-map pairs and features. `artifacts/`: CAS bitstreams/RGB/maps. Binary, source, models, initial failed fixture, all attempt logs, resource receipts, `_MANIFEST.json` and `COMPLETE.json` preserved. Initial instrument/admission failures caused no score-dependent selection.

File SHA256 identities:

- `REGISTRATION.md`: `f1b6d36572ea269da52fd4cf9cc432feb10e4288c86c85e30d47bcb819ad4b61`.

- `INPUTS.json`: `f620c40117efc0496212124dae2e19bbff5361bfe4aeaa30dbdb58748ba42e04`.

- `SUMMARY.json`: `a29ac5034b5db705b36151e0955af4d7392dcf914cb02e74dff373487c48f429`.

- `ROWS.tsv`: `64a4977a92e59b96abace78b823e62d53ccd9c9485b2ae0658ee09a596c06b88`.

- `ARCHIVE_FILES.json`: `b70e45b3ba4720fa9993459ecf3ab88ef15b972f69aa5b74a6a7489e6e073a62`.
