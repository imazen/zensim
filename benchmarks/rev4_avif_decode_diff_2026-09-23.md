# Rev4 AVIF decode path difference — 2026-09-23

## Finding

The September 14 SafeSyn extraction and September 23 CVVDP smoke used **the same locked zenavif 0.2.0 version and dependency revisions**, but different decode surfaces. The former used `zenavif::AvifDecoderConfig` through `zencodec`; the latter used `ManagedAvifDecoder::decode_full` directly. The wrapper attaches the signaled CICP transfer to the RGB16 `PixelDescriptor` before `zenpixels_convert::RowConverter` reduces it to RGB8. The direct path leaves transfer `Unknown`. This changes sparse RGB8 rounding by at most one code value and reproduces the exact SSIMULACRA2 disagreement. The single-thread setting is not the cause.

No zenavif source file was modified. The proposed `/var/tmp/avif-decode-diff/zenmetrics-avif-cicp-test.patch` (SHA256 `5f22a9011878b5ae2974897ce3c12217baa1bf1e5528849979c1024f35b8e2eb`) was apply-checked but must not be applied: its expected RGB8 hash pins the inexact tagged-route output. A corrected zenmetrics test must assert (a) the descriptor carries CICP transfer 13 before reduction and (b) RGB8 equals exact rounding of the native 10-bit samples, rather than a captured hash.

## Inputs and identity

The preregistration is `benchmarks/avif_decode_diff_prereg_2026-09-23.md`, commit `e6f01b29f164dd5098acaca16d4c0ae8af9d5056`. The fixed selection is the 11 smoke AVIF files plus 20 other SafeSyn AVIF files from distinct references, selected by SHA256(path) rank. All 31 file hashes matched `SAFESYN_ADMISSION.json`; they span 21 references and 20,116,577 pixels. `selected.tsv` SHA256 is `cbb2081f7c5e2f94d46bc6adc3577d34da9a0d231d5be7d02740699da5c4c435`. All are TRAIN-role; no human labels or held-out content were read.

The September 14 producer was `/home/lilith/work/zensim-validation-2026-09-14/native-integrity-admission/extract-native-admission` (SHA256 `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`). Its `zensim-bench/Cargo.lock` at build time (SHA256 `d6234c1fd46442563e33d95b2079e09d3e4289f08df3b7e91e4ed5ccdc7980b1`, September 14 timestamp) resolves **local path** zenavif 0.2.0, rav1d-safe `e73811f5d4dad81b75195ca18554fd8a5df19515`, zenavif-parse 0.7.0 from that zenavif workspace. The zenavif checkout's HEAD at and after that build was `a7c56be9e77dc708f379eeaa288ab607f6d3f6e1` by repository history (September 8 commit; no later commits). Its uncommitted September 14 state was not attested. Re-decoding all 31 through that source's `zencodec` path reproduced the audit's stored RGB8 SHA256 and SSIMULACRA2 score exactly, which supports the source binding for these inputs.

The September 23 zenmetrics lock also resolves local zenavif 0.2.0, the same rav1d-safe revision and zenavif-parse 0.7.0. The 0.1.7 tag is **not** the September 14 producer. It was therefore not treated as an investigation arm.

## Stage isolation and correctness

For each of 31 AVIFs, four calls used the same zenavif build: `AvifDecoderConfig` and direct `ManagedAvifDecoder`, each at automatic and one decoder thread. Packed native RGB16 bytes are identical in **31/31** across all four calls. The two thread settings give identical RGB8 in **31/31** within each path. The `zencodec` path's RGB8 hash matches the September 14 audit in **31/31**; direct managed RGB8 differs in **31/31**. Thus the first differing stage is **CICP transfer attachment to the pixel descriptor before RGB16→RGB8 reduction**. An independent AV1 Y/U/V dump was not needed to distinguish these two caller paths, but was not performed.

All selected bitstreams signal CICP primaries `1`, transfer `13`, matrix `6`, full range, 10-bit, 4:4:4; none has alpha or ICC. No chroma upsampling is required for 4:4:4. Identical native RGB16 bytes rule out AV1 reconstruction, film grain, YUV→RGB matrix, and range as the *difference between these callers*. The result does not establish that the underlying AV1 decoder is normatively correct against an independent implementation.

The [AV1 specification §6.4.2](https://aomediacodec.github.io/av1-spec/av1-spec.pdf) assigns semantic meaning to `transfer_characteristics`, matrix and range; [ITU-T H.273](https://www.itu.int/rec/T-REC-H.273-202407-I) identifies transfer code 13 as the IEC 61966-2-1 sRGB transfer in this context. The [AVIF specification](https://aomediacodec.github.io/av1-avif/v1.2.0.html) uses the AV1 and CICP color semantics for image items. The file signals sRGB, so passing a buffer marked `Unknown` into a conversion to `RGB8_SRGB` loses known source semantics. That is a **zenmetrics caller metadata defect**. The native RGB16 is 10-bit bit-replicated (`v16 = v10<<6 | v10>>4`, all 31 cells). The direct, untagged route produces exact round-to-nearest `round(v10·255/1023)` on 60,349,731 of 60,349,731 channel values. The sRGB-tagged RGB16→RGB8 `RowConverter` route rounds 96,160 values (0.16%) down by one code at near-half levels, with none above exact. This is a precision deviation in the tagged conversion (`zenpixels-convert`; cause not localised). The September 14 stored SafeSyn AVIF features and labels use that slightly inexact route.

The direct route dates to at least `e9e2ef71`; `e7fe45ab` introduced a direct `zenavif::decode` call, but its transfer-tag behavior was not verified. `e9e2ef71` used `ManagedAvifDecoder::decode_full` for the HDR tripwire and noted that the returned buffer is tagged `Unknown`, but still converted it directly. No commit in `zenavif v0.1.7..HEAD` explains this comparison; a purported zenavif version regression is unsupported by the pinned producer identity and paired hashes.

## Measured impact, fixed 31-pair sample only

The managed path changes 92,464 of 20,116,577 pixels (**0.459641%**). Changed channel samples: R 37,530, G 22,399, B 36,231. Max absolute change is `[1,1,1]`. Only 1,185 changed pixels lie within the outer two rows/columns; the effect is not confined to borders. Since the files are 4:4:4 without alpha, it is neither a chroma-upsampling nor alpha-edge effect. Per-file bounding boxes and hashes are in the JSON companion.

On the same 31 pairs, managed minus September 14 path gives SSIMULACRA2 mean `+0.002284454` (range `−0.065460266` to `+0.085065325`), and zensim Profile B mean `−0.000684103` (range `−0.058188434` to `+0.052514367`). These are descriptive paired differences on the selected sample, not population estimates. The old fast-ssim2 0.8.2 scorer reproduces **31/31** September 14 audit values and **11/11** smoke values; fast-ssim2 0.9.0 is a separate scorer and was not used for these deltas. The reference PNG RGB8 hashes match the audit **31/31**.

The SafeSyn packet contains **34,001** AVIF bitstreams. Its September 14 features and teacher scores are bound to the `zencodec` RGB8 path; fresh zenmetrics direct-path scores cannot be joined as if they used those pixels. AVIF-specific training and instrument populations also need decode-era stamps when rescored: `avif944` has 459,780 TRAIN and 104,520 `eval8` rows, and `avif-autotune` has 79,368 cells (`docs/DATA_SPLITS.md`); their score shift was **not measured here**. The board's AVIF ladder and AVIF-containing rows may read stored sidecars or decoded PNGs rather than these bitstreams, so this sample does not establish a board-value change. No board row was regraded.

## Recompute and artifacts

From `/var/tmp/avif-decode-diff/`, these one-line commands regenerate results from the fixed AVIF/PNG inputs and the September 14 audit:

```sh
python3 select.py
python3 analyze.py
python3 native.py
python3 scores.py
python3 summarize.py
git -C /home/lilith/work/zen/zenmetrics apply --check /var/tmp/avif-decode-diff/zenmetrics-avif-cicp-test.patch
```

Actual `summarize.py` output in the final replay:

```text
sample rows=31 references=21 smoke_rows=11
native_equal=31/31 baseline_rgb_hash_match=31/31 managed_changed=31/31
changed_pixels=92464/20116577 (0.459641%) max_abs_rgb=[1, 1, 1]
changed_channels_rgb=[37530, 22399, 36231] edge_2px=1185
ssim2_delta mean=+0.002284454 min=-0.065460266 max=+0.085065325
zensim_b_delta mean=-0.000684103 min=-0.058188434 max=+0.052514367
signalling=info transfer=13 primaries=ColorPrimaries(1) matrix=MatrixCoefficients(6) depth=10 alpha=false icc=false range=Full chroma=Cs444
safesyn_avif_files=34001
```

Large raw RGB/native outputs and per-cell detail remain under `/var/tmp/avif-decode-diff/`; SHA256: `pixels.json` `8fc21c9f4efe2032495783ce69920f16b5c5733c0cca0f3fc63f931c34d25aab`, `native.json` `46cfa76a9ac84c236decfbfbb755fcffdb7894aed98e4f03005b99a8e10cfc14`, `scores.json` `329f17444e1573caeb77fdfb44871ce6fb22f1e3ae020b244eec4c3dc781a90e`. The compiled probe is `cf4d86d507ae9d096122b5cdfa11cc9331b3e151bcd66941057b69a4c6761f1a`; its Rust source and compile command are beside it. The compact, committed per-row JSON is `rev4_avif_decode_diff_2026-09-23.json` (SHA256 `00a8122fb3f5564a88e560ee97c537794de05395d200fe4fcb3f732412b99851`).

## Limits

- No independent Y/U/V comparison or `zenav1-aom` cross-check was run. The first-stage finding rests on native RGB16 identity across the two caller paths, which is sufficient for this mismatch.
- The zenmetrics test patch was apply-checked, but the expected hash pins the inexact route and the patch must not be applied. The zenmetrics test suite was not run; no sibling source was edited.
- Process footprint: this measurement lane worked in the primary zensim checkout; the prereg was committed there, and the record, JSON, pointer and worklog were then committed and published before review. This correction was made after Opus review in a separate landing workspace.
- No population score shift, board regrade, or AVIF train/eval ladder re-extraction was attempted.
