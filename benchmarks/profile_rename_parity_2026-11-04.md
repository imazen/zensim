# Public profile rename parity — 2026-11-04

Host CPU: AMD Ryzen 9 9950X3D, x86-64 Linux. Source before: `473a4cfb`.
Source after: rename `38b16050` plus default-feature build repair `c8dd10a7`.
The helper below was temporarily appended to `zensim/tests/profiles.rs` in each
revision in the same jj workspace; the research crate was not used. The model
bakes, scoring parameters and historical GOLDEN values were unchanged.

```rust
#[test]
fn rename_baseline_dump() {
    let _token = archmage::testing::lock_token_testing();
    for &(name, id, _) in GOLDEN {
        println!("RENAME_ORACLE\t{name}\t{id}\t{:016x}",
            score(by_name(name), &pair(id)).to_bits());
    }
}
```

Command in both source snapshots (resource runner capped at 16 GiB, 28 jobs;
`TMPDIR` on disk, no target-cpu override):

```sh
cargo test -p zensim --features classification --test profiles rename_baseline_dump -- --exact --nocapture
```

[All 70 rows](profile_rename_parity_2026-11-04.tsv) match exactly after mapping
`F` to public `A` and development `A` to `LegacyA`; the other names are unchanged.
The pre-existing historical-golden test allows revision-1 rounding differences.
Two LegacyA and thirteen PreviewV0_2 values differ from the historical reference
bits on this host in both source snapshots; their existing tolerance passes.
For example, legacy `16x16-photo-noise` is `40500aefe3698de7` in both snapshots,
while the historical golden is `40500aefe491ef04`. No golden was re-pinned.

Resource runner results:

```text
before: run-heavy: done rc=0 13s | peak-RSS 0.61GiB | min-avail 37675MiB | peak-load 3.02
after:  run-heavy: done rc=0 14s | peak-RSS 0.62GiB | min-avail 37635MiB | peak-load 1.52
```

These times describe the correctness-check commands, including compilation;
they are not performance comparisons.
