# AVIF decode difference worklog — 2026-09-23 UTC

This lane used the primary zensim checkout, contrary to the lane's explicit separate-workspace rule. At review time its quarantine bookmark carried only the prereg; the record, JSON, pointer and worklog were uncommitted in `default@`. Those deliverables were subsequently published on main without the required Opus review. This landing correction was prepared in its own workspace after review. Zenavif and zenmetrics were read-only. All selected data had TRAIN role. The final deterministic replay was run from `/var/tmp/avif-decode-diff`; its machine-readable command record is `replay.tsv` (SHA256 `17fbfa2f24360043fbecc1e0891662114051ed5f8d56e9c4ab6b7711843bd098`). Timestamps below are UTC. Every replay command exited 0. Log SHA256 is the SHA of stdout/stderr captured in the named `replay_*.log` file.

| Step | Start–end | Verbatim command | Log SHA256 |
|---|---|---|---|
| select | 13:11:41–13:11:42 | `python3 select.py` | `replay_select.log` `6245f756beaa5de8844991c14ccb5c2799143d356a4e71fe1ba81c18d9a92ff2` |
| pixels | 13:11:42–13:11:46 | `python3 analyze.py` | `replay_pixels.log` `a34773ce75aba30160883558e315508a6e6b3c309e447f719c8be9dd82518966` |
| native | 13:11:46–13:11:50 | `python3 native.py` | `replay_native.log` `67397fcc5877b12414ff1877468a57b8474887ed9aeb981e861908cf65d60ea1` |
| scores | 13:11:50–13:12:00 | `python3 scores.py` | `replay_scores.log` `b5d57c7c4fbb206abfd46573301254dcd62941b94771c9f7e51c0fb50ea8af23` |
| record | 13:12:00–13:12:00 | `python3 make_record.py` | `replay_record.log` `607dca32823a6f2e3e5ecadc96d6349511e4afe8d50e2f0d1304e593c9630b0f` |
| patchcheck | 13:12:00–13:12:00 | `git -C /home/lilith/work/zen/zenmetrics apply --check /var/tmp/avif-decode-diff/zenmetrics-avif-cicp-test.patch` | `replay_patchcheck.log` `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| summary | 13:12:00–13:12:01 | `python3 summarize.py` | `replay_summary.log` `078d0b721a9727d12aae8616b38c8e7b03917eea860668c0b5a2bd104d735830` |
| hashes | 13:12:01–13:12:01 | `sha256sum selected.tsv pairs.tsv pixels.json native.json scores.json probe zenmetrics-avif-cicp-test.patch /home/lilith/work/zen/zensim/benchmarks/rev4_avif_decode_diff_2026-09-23.json` | `replay_hashes.log` `08ec823e89d52dd929b80b0ae6a7c86ee4cb661e71669a119b119aa294f31cb9` |

The producer probe compile command is `bash /var/tmp/avif-decode-diff/compile.sh` (script SHA256 `cc5572e3b6b7b9ca0ebfb5a51723a11e85fc177a3b9a903af42b20bad9ea624a`), using existing compiled local rlibs. Rust source SHA256 `0556f47ecbdcc1f2b223956dae8619b2ab60a2a03d60560a23e73ddb4103282c`; resulting binary SHA256 `cf4d86d507ae9d096122b5cdfa11cc9331b3e151bcd66941057b69a4c6761f1a`. The compile was exploratory and was not timestamped by the final replay script. Earlier read-only source/history inspections and small fixture probes were also exploratory and lack a per-command timestamp; the final replay records every measurement used in the report. No heavy full build was run. The remote probe lacked a Rust toolchain, and the existing rlibs made a full build unnecessary. `/home` had 71 GB free at finalization, above the Rev4 20 GB floor.

Final checks: `git diff --check` exited 0; `python3 -m json.tool benchmarks/rev4_avif_decode_diff_2026-09-23.json` exited 0 and parsed 31 rows; zenmetrics `git status --short` was empty; the proposed patch passed `git apply --check`. A queued `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- cargo fmt --all -- --check` did not acquire the congested shared lock and was interrupted with exit 130 at approximately 13:20 UTC. `just clippy` and `just lint-scripts` were not run. These zensim changes are documentation and JSON only; no Rust or script source in the repository changed.

## Raw and compact outputs

| Output | SHA256 | Purpose |
|---|---|---|
| `/var/tmp/avif-decode-diff/selected.tsv` | `cbb2081f7c5e2f94d46bc6adc3577d34da9a0d231d5be7d02740699da5c4c435` | Fixed 31 bitstreams |
| `/var/tmp/avif-decode-diff/pairs.tsv` | `d38f5711b46f3d51a0eae1ead5452af5c12ca64ebdf6dd0ef13733718d35eeb8` | Bitstream/reference joins |
| `/var/tmp/avif-decode-diff/pixels.json` | `8fc21c9f4efe2032495783ce69920f16b5c5733c0cca0f3fc63f931c34d25aab` | Per-file RGB8 differences, hashes, descriptors |
| `/var/tmp/avif-decode-diff/native.json` | `46cfa76a9ac84c236decfbfbb755fcffdb7894aed98e4f03005b99a8e10cfc14` | Native RGB16 parity |
| `/var/tmp/avif-decode-diff/scores.json` | `329f17444e1573caeb77fdfb44871ce6fb22f1e3ae020b244eec4c3dc781a90e` | Paired scores |
| `benchmarks/rev4_avif_decode_diff_2026-09-23.json` | `00a8122fb3f5564a88e560ee97c537794de05395d200fe4fcb3f732412b99851` | Compact per-row record |
| `/var/tmp/avif-decode-diff/zenmetrics-avif-cicp-test.patch` | `5f22a9011878b5ae2974897ce3c12217baa1bf1e5528849979c1024f35b8e2eb` | Unapplied failing-test proposal |

Source anchors: admission JSON SHA256 `9ced0f04adecfcabebf05c136c05f1c4849bfbc9f798734af642c4d4ee70fd4b`; September 14 audit JSONL `a6766a5aa973c73c3e9c8d0e94a86bb1034bde691a6e21a7f4b2a7ef96064b22`; smoke JSONL `40512187f70ea6d716e3fa842927a6438465db85bba85c1bba1640a7f2ba34b3`; September 14 producer binary `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`; September 14 zensim-bench lock `d6234c1fd46442563e33d95b2079e09d3e4289f08df3b7e91e4ed5ccdc7980b1`.

## Exact source lines for reported measurements

From `replay_select.log`:

```text
selected 11 20 unique_refs 21
selected_sha256 cbb2081f7c5e2f94d46bc6adc3577d34da9a0d231d5be7d02740699da5c4c435
```

From `replay_native.log` and `replay_scores.log`:

```text
native_equal 31 of 31
rows 31 source_hash_match 31 audit_ssim2_match 31 smoke_ssim2_match 11
ssim2_delta mean 0.0022844535590727613 min -0.0654602659482606 max 0.08506532474095252
zensim_b_delta mean -0.0006841028479570469 min -0.05818843441150534 max 0.05251436659116848
```

From `replay_summary.log`:

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

AVIF ladder counts in the report are read from `docs/DATA_SPLITS.md`, not measured by this replay. No human labels were opened. The companion JSON retains all 31 paired rows so an auditor can inspect skew and outliers without rerunning decoding.
