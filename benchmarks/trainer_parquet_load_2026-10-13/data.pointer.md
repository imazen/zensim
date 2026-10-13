# Trainer loader evidence

The immutable dated evidence is on the `r5900xt` Tower NFS mount:

`/mnt/v/output/zensim/trainer_parquet_load_2026-10-13/`

`run2/` contains six fresh-process stdout, parsed JSON and `/usr/bin/time -v`
files, plus before/after hashes. `run1/` and `run1-wrapper.log` preserve the
first parser failure after its successful flat load. Build/check logs and the
corrected wrapper log are also retained. The 32-file `SHA256SUMS` manifest has
SHA-256 `19afc6c3d64a8152fff86fe3779e4e99953624e499a211364f08cc591f22ad6e`.

The Parquet input is
`/mnt/v/output/zensim/kadis-944-2026-08-01/kadis700k_944.parquet`,
SHA-256 `6ab12d50c41f94bef871d4b63257e05bbe481c95b7f1896d496ce5e817daf9a9`.
The measured release binary SHA-256 is
`beb9a06c96eeccf268c84b225a688871fc43db4c3f2a6fce87b4ef407f349600`.
`run2/source.txt` names benchmark source commit `9beaa7af`; the benchmark
Rust file is byte-identical to its initial `d882d7c2` commit. The runner
requires this exact input size, row/column counts and both stored-value counts.

To verify the archive and repeat with a **new** output directory:

```sh
cd /mnt/v/output/zensim/trainer_parquet_load_2026-10-13
sha256sum -c SHA256SUMS
cd /home/lilith/work/zen/zensim
just trainer-parquet-load-build
~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  just trainer-parquet-load-study /mnt/v/output/zensim/trainer_parquet_load_2026-10-13/new-run
```

The source and output target are host-specific. The results measure loader
startup and materialization only, not end-to-end training or cold storage.
