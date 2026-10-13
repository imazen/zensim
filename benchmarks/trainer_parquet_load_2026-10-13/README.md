# Trainer Parquet load cost, 2026-10-13

The current zensim trainer has two relevant Parquet paths: an ordinary flat
F64 feature matrix, and a compact F32 projection when a supported plain
`--keep-features` run selects eligible columns. This study times those **existing
loaders** on the complete 699,999-row, 944-feature KADIS Float32 Parquet file.
It does not run a training epoch.

The input is 2,523,688,313 bytes (11 row groups and 15 non-feature columns).
Both paths also load `score_iwssim_gpu` at scale 1.0. The compact run selects
feature IDs 440–503, 64 of 944. Three fresh processes per path alternate order;
the file and release binary hashes match before and after all six. The loader's
own output and each `/usr/bin/time -v` record are retained.

| Loader | Values materialized | Load seconds, 3 trials | Median | Worst | Max process RSS |
| --- | ---: | --- | ---: | ---: | ---: |
| Flat F64, all 944 features | 660,799,056 | 7.538, 7.698, 7.633 | 7.633 s | 7.698 s | 5,554,792 KiB |
| Compact F32, selected 64 features | 44,799,936 | 0.483, 0.473, 0.470 | 0.473 s | 0.483 s | 223,640 KiB |

The workloads differ in both projected columns and output type. Their time
ratio is **not** a per-column decoder speedup. The compact path materializes
179,199,744 feature bytes; the flat path materializes 5,286,392,448. The
source file was SHA-256 checked before and after, which warms the host cache;
cold-device time was not established. The trainer also hashes each source for
reproduction identity after loading, which this timer excludes. No fraction of
end-to-end training time follows from these results because no epoch or full
trainer startup was timed.

The benchmark source landed as `d882d7c2`; a result-parser-only correction is
`9beaa7af`. The release binary SHA-256 is
`beb9a06c96eeccf268c84b225a688871fc43db4c3f2a6fce87b4ef407f349600`.
The first attempt stopped at its JSON parser after one successful flat load;
it is retained. The corrected six-run job passed `rc=0` under
`run-heavy --mem 16G --jobs 8` in 37 seconds, with whole-job peak RSS 5.30 GiB,
minimum available 49,067 MiB and peak load 1.89. The per-process RSS above
comes from `/usr/bin/time -v`, not the wrapper. Raw outputs, commands, failed
attempt and seal are in the [evidence pointer](data.pointer.md).
