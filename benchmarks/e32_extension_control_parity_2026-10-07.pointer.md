# E32 trainer extension: full control parity

Source base: `13ece3619b8f74656aef2066211e74bd69cdefa1` (main at lane start).
Tested implementation tip: `09ddbda7` with ancestors `8a2f5473`, `2b60b69c`,
`6f576cac`, `6c57d6de`, `0dc7a166`. All commits are local; nothing was pushed.
Transport contract: [E32_PALETTE_TRAINING_TRANSPORT.md](../docs/E32_PALETTE_TRAINING_TRANSPORT.md).

## Full-budget measured result

**PASS** for E30 nA3 `without_kadid_s0`, final epoch 119, 120 epochs × 50,000
pairs, initialization seed 1101 and sampling seed 101, one Rayon thread, v3.
The final extended trainer's complete model bytes match E30 after the
canonical strip owner removes only `zentrain.repro`. Reproduction records
also match after normalization of timestamps and run/build locations.
Input SHA values, weights, seeds, budgets, sampling and admission facts were
retained. Raw bakes differ because their reproduction metadata differs.

Final evidence root:
`/mnt/v/output/zensim/e32-extension-2026-10-07/kadid-s0-final/`.

| Artifact | SHA-256 |
| --- | --- |
| `PARITY.json` | `8634214bf65c7c57362ca01f94094551485188d1003a5ed3e2750dfae9a7e446` |
| `cell/result.json` | `8aec6777b1b254c8554f889b0c6e658bd7e17b9a83fc6bf1f3fccba00a42096e` |
| Both models without reproduction metadata | `4fc21dc98d83cdcfba25f38623921e90595e976c75ad2fa8a4f8b2eb7f1a95f2` |
| Both normalized reproduction JSON files | `222d90abda6708c582f4ba62c247d7581e4a99003386da3c76a5c0c0fd6164c7` |
| Extended trainer binary | `43747a5e9d6b8464a232aec91b374923c70d3c0be6648583b7fe4e6f31602810` |
| Frozen E30 control inventory | `304f67a5d0b7bc78778b9deec731c213825b6a3811671c4fc3bdbf699978a07f` |

The trainer binary is retained at
`/home/lilith/tmp/e32-target/release/zensim_mlp_train`.
Compiler: `rustc 1.99.0 (b940084d7 2026-09-28)`, release thin LTO,
no native CPU build flag. Prepared control inputs:
`/mnt/v/output/zensim/shippath11-2026-10-07/v2d1`.
Original control:
`/var/tmp/rev4-featpot/e30-results/cells/sel:59f0bbc2f290@h32:H128:cv16:cf98__N/without_kadid_s0`.

Command, from the E32 workspace (destination must be fresh):

```sh
just e32-control-parity \
  "$HOME/tmp/e32-ext/run-scratch" "$HOME/tmp/e32-target/release" \
  /mnt/v/output/zensim/shippath11-2026-10-07/v2d1 \
  '/var/tmp/rev4-featpot/e30-results/cells/sel:59f0bbc2f290@h32:H128:cv16:cf98__N/without_kadid_s0' \
  /mnt/v/output/zensim/e32-2026-10-07/CONTROL_PINS.json \
  /mnt/v/output/zensim/e32-extension-2026-10-07/kadid-s0-final
```

The recipe runs through `run-heavy --mem 16G --jobs 1`, sets v3 and one
Rayon thread, and calls the existing strict fit owner. Full logs:
`cell/train.log`, `cell.log` under the evidence root, plus
`/home/lilith/tmp/e32-ext/full-control-final.log`.
Measured resource line:
`run-heavy: done rc=0 288s | peak-RSS 0.95GiB | min-avail 43359MiB | peak-load 17.10`.

An earlier pre-null-fix replay is retained at sibling `kadid-s0/` and also
passed (283 s, peak RSS 0.85 GiB). The final result above binds the rebuilt
null-refusing trainer; no earlier result substitutes for that verification.

## Scope

The extension admits named palette inputs, preserves old auxiliaries,
checks projected roles/populations/pins, refuses nonfinite and null primary
values, and adds no serving support. Synthetic gates include an actual
462-input trainer invocation, all loader shapes, late forbidden populations,
wrong fit/development roles and zero label-bearing payload opens on refusal.
The bank and instrument metadata SHA values and semantic instrument contract
were verified against the registered pins without reopening feature/label
payloads.

No real palette arm was fit; no real joined arm root, archive, image,
zenmetrics profile, grid or fleet package was built. Original inheritance
and observation joins still need proof when the combined v40 data package is
assembled. No KADID TERMINAL, AIC-family, CID22-B, HDR or external label
payload was opened by this work. The coordinator's fresh shared 40-cell v40
control remains mandatory; this regression result does not authorize E30 reuse.
