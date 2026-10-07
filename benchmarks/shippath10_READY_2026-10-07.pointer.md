# SHIPPATH10 — D1 / E30 / production preparation, 2026-10-07

Prepared only. The 40 E30 cells and three production fits have not run at the
registered budget. No job was queued, no image/data was published, no protected
confirmation/T0 label or AIC-family table payload was read. E28 runs first.
Full-model composition, assessment and final qualification remain outstanding.

D1 is the owner decision recorded at ledger commit
`1d3bf35a78a149f0925029d71014c6df8535f550`. The four production sources are
KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and CID22-A. Historical five-source
routes retain their defaults. Strict wrappers check the decision, original
receipt/freeze binding, human population and label-free row identities before
creating a fit directory. Added AIC-family sources fail closed.

The fresh admitted view is `/mnt/v/output/zensim/shippath10-2026-10-07/v2d1`; the portable root is
`/var/tmp/rev4-featpot/v2d1`. Ten approved source/teacher/human-union tables are
byte-identical copies. The full union uses the existing human_without_aic3
fit/dev rows. Eight LODO tables are ordered subsets of those rows; every feature
and target bit, including NaN payloads, is preserved. The unchanged 42,021-row
Rev5 coverage pool travels with its admitted metadata. The archive contains 19
table payloads, 63 pinned files, and no AIC table or confirmation payload. Original
source receipt/freeze metadata remain as bindings; their hashes are unchanged.

E30 declares nA3 at `sel:59f0bbc2f290@h32:H128:cv16:cf98`, head N, 420 IDs,
four folds and seeds 0–9. Forty existing E24 control cells are separately pinned;
no control refit or AIC fold is declared. `e30_four_source.py` adapts the existing
E21 statistics and external-set owners for explicit later assessment. The cost
report has no adoption decision. No assessment subcommand ran during preparation.

Production declares full-data four-source fits, seeds 0–2, 120 epochs,
50,000 pairs/epoch, fixed final epoch 119. `v2_confirm_fit --pack-production`
uses the canonical densifier and `bake_dial_refit pack`: f16 packing precedes the
final TRAIN calibration, with the CID22 oracle fit table explicitly supplied.
The selected epoch dump and packed model preserve qualified Rev5 admission,
feature-set, producer/decoder and uniform-sampling metadata. No checkpoint or
development metric selects a production epoch.

All five prepared routes pass seven-table admission. Local and installed-image
short smokes use two epochs / 128 pairs with full input tables, explicitly outside
the registered fleet budget. The real executor verifies data, uses output paths
outside the admitted root, and emits hash-bound output receipts. The canonical
harvester verifies both training-only blobs and refuses incomplete admission or
wrong epochs. Canonical packed inference is finite and ordered on 3,785 CID22
development rows; local/image selected weights and packed predictions match.
The Rust model loader verifies decoded compressed metadata on both epoch dumps
and the packed model. This proves transport/admission, not scientific adoption.

The local worker image pins one Rayon thread and the v3 tier. Its six-GiB cap is
inherited from the existing E28 envelope. The final short container measured
1647517696 bytes in memory.peak, with zero max/oom/oom_kill
events. Full 120-epoch resource use is not measured. The final Docker smoke's
run-heavy line is recorded in `pinned-env-smoke.log`; its 0.03-GiB peak-RSS is the
Docker client, while the container counter measures the fit.

Artifacts and launch authorization templates live at `/mnt/v/output/zensim/shippath10-2026-10-07`. Source pins and
receipts are in `ARTIFACT_PINS.json`; the program is the zenmetrics `v2d1` profile
at `9a55587e1c506b0a344fea6075465066beb187d8`. It depends on the preserved local
E28 profile history. A separate concurrent worker/justfile change appeared after
that commit and is excluded from this lane's source pins.

- Program SHA-256: `6bc158276777b4bd9ebd6eb2df9abecc113713121360820cf0258913ed6a7fa0`
- Data SHA-256: `8f2dae5e024670c02bbb81e5ed18254a5132f4423fc672f6aeb1059560bba084`
- E30 manifest SHA-256: `ffaef44e1410623aa8b2af35976374b3b453f2e3b0bf5877cde2634c4314aac5`
- Production manifest SHA-256: `328d7495a3bddda1a707a7811f5d344e9d3a12c1a840e2fc9d0b5202331363c6`
- Image: `ghcr.io/imazen/zenfleet-worker:fit-d1-e30-v38-w8dc42d4e`
- Local image ID: `sha256:2efae6bf28c7f8bc7b241b6664a45e83fa1261652ca9d58b1b682cfea67d2a49`
- Decision SHA-256: `1baaa0ae980757d69e351240cb9c0c4d3c5369bde2eb52addafcd49845dbe94a`
- New view frozen SHA-256: `4e1c0d997acea3a985f21c858ced0231d832a6ee27f34568dbc98ac450653999`

Both private local jobset_caps entries exist. Their hashes are recorded without
copying the host map into this document. Both launchers refuse before any fleet
command unless the owner supplies the matching LAUNCH_AUTHORIZATION file,
confirms source landing/pin publication/review and E28 completion. No such
file was created. Actual refusal tests preserve the fleet queue byte for byte.
Launchers append behind existing work and use the existing queue/fillers.

Validation: 23 SHIPPATH Python regressions, 43 existing fit-tool tests, two new
strict-harvest regressions, scoped formatting/clippy, script lint, full bundle
hash audit, all-fold admission, actual executor/harvest/model-loader checks and
3,785-row packed inference. Commands are in the justfile; full logs and evidence
are under `/mnt/v/output/zensim/shippath10-2026-10-07` and `/home/lilith/tmp/shippath10`.
