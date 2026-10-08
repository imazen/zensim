# E31 control/admission evidence

Local root: `/mnt/v/output/zensim/e31-preparation-2026-10-07/`.
Mirror: `/mnt/tower/output/zensim-e31-preparation-2026-10-07/`.
R2: none uploaded. These artifacts record a blocked gate, not a prepared fit.
Source implementation: `6b525cf2086f1790c9a0125a34a9d74c484f9623`.

| Relative file | Bytes | SHA-256 |
|---|---:|---|
| E30_CONTROL_PINS.json | 3878745 | `d12450b7ec926f41b5b59ce3b3796557ba79a5d9316b8cc06b90b85eca996f78` |
| pinned-admission/PINNED_TRAINER_ADMISSION.json | 3569 | `4e2d3f9dc23cd39957395122c91038d42c3d91ffbcbffae15fda7b53a3c774d2` |
| TOWER_CONTROL_CHECK.json | 451 | `770a0b0b40a6ca8bb8ce8625974b2417bac29f751e69556fb0e3a90a4ea5a0b9` |

The first binds all 40 E30 nA3 final-119 results, selected bakes, ordered
features, embedded training receipts, job argv and program/binary inventories.
The second records the unchanged trainer's UPIQ null-ID refusal and synthetic
mixed-decoder refusal, both exit 2 with no dataset payload opens or fit output.
`pinned-admission/*.strace` and stderr/stdout files preserve the syscall evidence.
The final file compares 320 installed files with Tower's existing control index
and records three independently hashed Tower files. After copying this evidence
root, three deterministic random local/mirror file hashes agreed.
