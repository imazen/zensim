# SHIPPATH11 — review fixes and v39 preparation, 2026-10-07

Prepared locally; nothing pushed, published or enqueued. The registered 40 E30
cells and three production fits remain unrun. E28 stays first; full-model
assessment and qualification remain outstanding.

Population admission now has a metadata/key phase before any payload hash or
decode. Strict output guards read only freeze metadata and label-free keys.
The CLI checks the approved D1 fold/decision, human declarations, key membership
and bank-member allowlist; the lower table owner checks every group's population
before any group's payload access. Preparation checks all selected populations
before copying or hashing payloads. Open tripwires exercise forbidden AIC fold,
key and bank-member cases with zero payload opens and no destination creation.
Historical population and training defaults are unchanged.

The program carries the committed qualified-fit contract, bound to the same
unchanged data archive SHA. Harvest requires that exact program archive and the
canonical inspector matching its binary pin. Expected budget, epoch, receipt,
freeze, decision, consumed IDs, seeds and seven table admissions come from the
trusted contract. Selected and packed checkpoint metadata must agree. Executor
lexical paths match only the declared data-hash extraction. Registered jobs must
follow this contract even if a result alters its training-only flag.

Local smokes use explicit `--local-smoke-budget 2:128`, distinct argv/job
identities and separate cell names. They can be verified with
`--allow-local-smoke` but cannot install through either CLI or lower owner.
Actual blobs reject short results under full jobs, epoch-119 claims with epoch-1
checkpoints, zeroed admission and altered result mode, after recomputing receipt
hashes. Both routes pass positive verification through the canonical loader.

Zensim was rebased onto `main@origin` `079c793845368d82c77a82891f5f991572f44169`.
Both justfile recipe sets, including the later E28 executor recipe, and both
Known Bugs records are preserved. The three original zenmetrics commits were
rebased onto `master@origin` `f26c61cb11e12e48cd9026ebae21c89a2afa4d72`; the E28 v36 profile
and reviewed worker changes are preserved. Source pins match the built program
after the final rebase. The temporary zenmetrics workspace is removed at finish,
with local commits retained by `quarantine/codex/shippath11`.

The inspected E28 driver `/var/tmp/fitv2/harvest_driver_v2.py` delegates to its
existing runtime verifier without registered budget/decoded checkpoint binding;
its installed-copy shortcut compares identity fields only. Neither running file
was modified or executed. That gap requires separate coordinator sign-off.

Artifacts: `/mnt/v/output/zensim/shippath11-2026-10-07`. The archive still contains 19 table payloads and 63 data files;
the refreshed program has 27 pinned files. Full manifests remain 120 epochs,
50,000 pairs/epoch and final epoch 119. Data and original roots are unchanged;
no AIC-family, protected confirmation or T0 payload was read.

- Program SHA-256: `6d491eb165a00b70297de0864ec2e749818ca81491207a7da3fa6e36c2028cd0`
- Data SHA-256: `8f2dae5e024670c02bbb81e5ed18254a5132f4423fc672f6aeb1059560bba084`
- E30 manifest SHA-256: `acf29dcb9c061870ee23c892e2143e56a91e8d5730f0fa0599cf5a824a1a0282`
- Production manifest SHA-256: `236eb2feb7861c05e5bc437b1063674266dca13f7c46a865c5b5e5c0e6025e84`
- Image: `ghcr.io/imazen/zenfleet-worker:fit-d1-e30-v39-w925f9783329f`
- Image ID: `sha256:f9a7ff6436fd9e9d6d8fbb1f108b615753f65dc2ff1974dd2e22f956383b5e5c`
- Worker build ID: `925f9783329f`
- Worker SHA-256: `925f9783329f45db2963dc1c5ef16c4d8a0f62c072618fb4e1b929ae37b80439`

The worker was rebuilt in the CI-pinned 23-sibling snapshot with Cargo.lock
unchanged, including the reviewed claim fix; installed worker bytes match.
Its 11 ownership regressions pass. The image's actual `/usr/local/bin/fit-cell-exec`
entry ran both short fits on fresh scratch with real extraction, links, argv and
receipts. No subprocess interception or budget monkeypatch was used. Each
selected epoch and packed model retains seven qualified Rev5 admissions.
Canonical densify/f16 packing precedes final TRAIN calibration. Selected weights
and all 3,785 finite ordered packed predictions match SHIPPATH10's short baseline.

The final cold short container measured 1898991616 bytes
in memory.peak, with zero max/oom/oom_kill events. Full 120-epoch resource use is
not measured. Its run-heavy line and all other logs are preserved in the bundle;
Docker-client peak-RSS is separate from the container memory counter.

Validation: 27 SHIPPATH tests; 43 existing fit-tool tests plus 10 strict-harvest
regressions; scoped Rust formatting/clippy; script lint; full bundle audit;
actual executor, loader, harvest negatives and unchanged numerical parity.
Launch authorization templates bind the new image, manifests, evidence and
worker identities. No authorization file was created; both launchers refuse
before any fleet command and preserve the queue byte for byte.
