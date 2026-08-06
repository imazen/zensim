# Extractor slot forensics — the 39 never-populated slots of the 944 vector (2026-08-05)

Follow-up to the data-integrity audit's **F-4** finding
(`benchmarks/data_integrity_audit_2026-08-04.md`, appendix G of
`benchmarks/sota944_campaign_2026-08-03.md`): 39 of 944 feature slots are
constant-zero in **all 11** ext944 training tables, outside the `f156..f371`
structural-zero block. F-4 classified them wholesale as "class (ii) — a
property of the extractor itself". This pass reads the extractor source slot
by slot and answers *which kind* of extractor property each one is, per the
registered taxonomy:

- **(a) condition-never-fires on the training data** — the code path exists
  and works; the corpora never exercise it (proof = a synthetic input that
  fires it);
- **(b) wired to zero by design** — deprecate-by-absence / documented skip;
  no code path exists that could populate it in any SDR extraction;
- **(c) BUG** — a computed value discarded or mis-indexed.

**Headline: 31 slots are class (b), 8 slots are class (a), 0 slots are
class (c).** No fixes were needed; no extraction output changed (no REGIME
event). One test was strengthened to make the class-(a) proof cover all 8
slots (see §4).

Source of truth read for every row: `zensim/src/feature_v2.rs` at
`7577dfa6` (layouts: append `f720 + scale*51 + ch*17 + local` with
`idx_append` locals; append2 `f924 + scale*5 + local` with `idx_append2`
locals; channels 0=X, 1=Y, 2=B).

## 1. Per-slot table

| feat_idx | scale | ch | local | class | mechanism |
|---|---|---|---|---|---|
| 720 | 0 | X | `XMASK_TRANSDUCER` | (b) | Y-only by design (see §2.1) |
| 721 | 0 | X | `LUM_TRANSDUCER` | (b) | Y-only by design (§2.1) |
| 754–770 | 0 | B | all 17 locals | (b) | `APPEND_SKIP_B_SCALE0` whole-cell skip (§2.2) |
| 771 | 1 | X | `XMASK_TRANSDUCER` | (b) | §2.1 |
| 772 | 1 | X | `LUM_TRANSDUCER` | (b) | §2.1 |
| 805 | 1 | B | `XMASK_TRANSDUCER` | (b) | §2.1 |
| 806 | 1 | B | `LUM_TRANSDUCER` | (b) | §2.1 |
| 822 | 2 | X | `XMASK_TRANSDUCER` | (b) | §2.1 |
| 823 | 2 | X | `LUM_TRANSDUCER` | (b) | §2.1 |
| 856 | 2 | B | `XMASK_TRANSDUCER` | (b) | §2.1 |
| 857 | 2 | B | `LUM_TRANSDUCER` | (b) | §2.1 |
| 873 | 3 | X | `XMASK_TRANSDUCER` | (b) | §2.1 |
| 874 | 3 | X | `LUM_TRANSDUCER` | (b) | §2.1 |
| 907 | 3 | B | `XMASK_TRANSDUCER` | (b) | §2.1 |
| 908 | 3 | B | `LUM_TRANSDUCER` | (b) | §2.1 |
| 927 | 0 | Y | `HL_BIN1` | (a) | HDR-route-gated (§3) |
| 928 | 0 | Y | `HL_BIN2` | (a) | §3 |
| 932 | 1 | Y | `HL_BIN1` | (a) | §3 |
| 933 | 1 | Y | `HL_BIN2` | (a) | §3 |
| 937 | 2 | Y | `HL_BIN1` | (a) | §3 |
| 938 | 2 | Y | `HL_BIN2` | (a) | §3 |
| 942 | 3 | Y | `HL_BIN1` | (a) | §3 |
| 943 | 3 | Y | `HL_BIN2` | (a) | §3 |

Counts: **(b) = 31** (8 X-channel transducer slots + 6 B-channel transducer
slots at scales 1–3 + the 17-slot B@scale0 cell), **(a) = 8** (HL bins ×
4 scales), **(c) = 0**.

## 2. The 31 class-(b) slots — wired to zero, verified in the kernel, not just the docstrings

### 2.1 `XMASK_TRANSDUCER` + `LUM_TRANSDUCER` on X and B (14 slots)

Both transducers are **Y-channel-only by documented design**
(`idx_append` docs, `feature_v2.rs:248-264`): the cross-channel masking
direction is chroma→luma per the ColorVideoVDP-trained matrix (luma does
NOT mask chroma), and the 2026-07-19 luma-gate ablation measured chroma
transducers as a broad CID22 cost
(`benchmarks/v2_trainability_ab_2026-07-19.md:209-216`). Verified wiring,
both ends:

- **Kernel side**: `append_block_kernel_generic<_, const CROSS, _>`
  (`feature_v2.rs:3236-3253`) computes the two transducer chains inside
  `if CROSS { … }`; the X/B dispatch instantiates `CROSS=false`
  (`append_block_kernel_entry_nocross`), so the chain is **const-compiled
  out** — nothing is computed and discarded.
- **Finalize side**: `finish_append` (`feature_v2.rs:3742-3751`) writes
  literal `0.0` to both slots when `cross == false` (`cross` is `ch == 1`
  at the call site, line 6246).

Not a bug by construction: no accumulator for these slots ever exists on
X/B, so there is no value to mis-route. Pinned by the existing test
`append_bounds_and_chroma_xmask_zero` (asserts both slots exactly 0.0 on
ch 0 and 2 at every scale, AND the Y slot fires as a positive control —
re-run green this pass).

### 2.2 The B@scale0 whole-cell skip (17 slots, f754–770)

`APPEND_SKIP_B_SCALE0 = true` (`feature_v2.rs:237-244`): the append kernel
is **never dispatched** for (ch=B, scale=0) — `append_cell_active()`
(line 5088) gates both the kernel and `finish_append` (line 6237), so the
17 output slots keep their zero initialization. Documented psychophysical
grounds in the constant's doc: yellow-violet foveal resolution ~53 ppd vs
94 achromatic (Ashraf/Chapiro/Mantiuk 2025); butteraugli carries no B
channel in its two highest-frequency bands; scale 0 is ~75% of pyramid
pixels, so the skip buys 25% of the append block's pixel cost for signal
the eye cannot resolve. The fused-entry retention zero-fills the cell's
`bs2` plane explicitly for determinism (line 5185-5189). Pinned by the
same test (all 17 asserted 0.0).

## 3. The 8 class-(a) slots — HDR-route-gated HL bins

`HL_BIN1`/`HL_BIN2` (`idx_append2` locals 3/4, all 4 scales) weigh error
above PU-luminance anchors `HL1_Y_ANCHOR = 1.01` (gray at 100 cd/m² ≈ SDR
white) and `HL2_Y_ANCHOR = 1.649` (gray at 1000 cd/m²).

**The condition that never fires on the training data:** the front-end
route. `foldapp_streaming_walk` sets `hl_bins: false` for `FrontEnd::Sdr`
and `true` for `FrontEnd::Hdr(_)` (`feature_v2.rs:5954-5965`); the kernel's
`if HL { … }` block (lines 3264-3275) is const-compiled out on SDR, and the
`WeightedSum::finish()` of the never-accumulated sums emits 0. **Every one
of the 11 ext944 training tables was extracted through the SDR route**
(the 924/944 set is SDR-by-design — CLAUDE.md "coming later: additional HDR
features"), so the bins are structurally zero there. On the cbrt/SDR Y
scale the anchors are additionally unreachable (Y ≲ 1.0 < 1.01), so even a
hypothetical SDR-route HL=true build would emit 0 on SDR content — the
gate and the content bound agree.

**Proof the code path works (the class-(a) requirement):** the existing
synthetic-HDR test `append2_hdr_hl_bins_and_pu_bandvis` (a 50→3000-nit
log ramp with error injected only above 300 nits, through
`compute_folded720_append2_features_hdr`) fires both bins. **Extended this
pass** to assert per-scale: all 8 slots (f927/932/937/942 + f928/933/938/943)
now individually asserted > 1e-3 on the HDR route — measured scale-0 values
HL1 0.1118, HL2 0.1519 — and the same test pins them to exactly 0.0 on
≤80-nit content. The SDR-route zeros are separately pinned by
`append2_layout_identity_and_first924_bit_stable`.

## 4. Class (c): none — and what was checked to say so

For every one of the 39 slots the check was: (1) does any kernel
accumulate a value destined for this slot in the configuration that
produced the tables? (no — the chains are const-generic-compiled out or
the cell never dispatches); (2) does finalize write anything other than a
literal/structural zero? (no — explicit `0.0`, or never called, or
`finish()` of an empty `WeightedSum`); (3) could a live value land in the
wrong slot? (no — the existing bounds+zero tests pin the exact indices
while positive controls fire on Y, so a mis-index would trip them).
No compute-and-discard was found anywhere: the zeros cost nothing at
runtime beyond the slot bytes themselves.

**No REGIME event**: nothing in this pass changes extraction output for
any input on any route. The only code change is additive test assertions.

## 5. Reconciliation with the audit + doc updates

- F-4's "all 39 are class (ii), a property of the extractor" stands, but
  the audit's registered taxonomy could not distinguish permanent zeros
  from route-gated ones. The refined split: **31 permanent (deprecate-by-
  absence), 8 route-gated (populate on the declared-HDR route)**. The item
  brief's assumption that "HDR-gated HL bins" were excluded from the 39 was
  wrong — the 8 HL slots ARE among the audit's 39 (the audit scanned SDR
  tables, where they are indistinguishable from permanent zeros).
- Consequence for dead-column pruning (campaign appendix E.9 hazard): all
  39 are correct prune candidates **for SDR-trained models**; the 8 HL
  slots must NOT be pruned from any future HDR-route (Q-appendix) model's
  input contract.
- Consequence for the ext944 manifest's PROVISIONAL marking of f924–943:
  8 of the 20 append2 slots being zero in every SDR table is expected
  behavior, not a data defect.

## Verification commands

```sh
cargo test -p zensim --features feature-regime-v2,custom-profiles --lib -- \
  append2_hdr_hl_bins_and_pu_bandvis append_bounds_and_chroma_xmask_zero \
  append2_layout_identity_and_first924_bit_stable
```

All green at this commit (run 2026-08-05, this workspace).
