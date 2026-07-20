# Tower basement server vs local box — v2 720-feature extraction throughput (2026-07-20)

Directional box-capacity benchmark (n=40 real aic3 pairs, ~1-5 MP, 3 runs best-of;
NOT a source-informing sweep — a relative CPU comparison). The real 720-feature
extractor (`v2_ab_extract`, append-only v1-372 ++ v2-348) run NATIVELY on both,
runtime CPU dispatch picking each box's best SIMD tier.

## Boxes

| | Local | Tower (basement) |
|---|---|---|
| CPU | AMD Ryzen 9 7950X (Zen4, 16C/32T) | AMD Threadripper 2950X (Zen1+, 16C/32T) |
| SIMD tier used | **AVX512** (magetypes `v4x`) | **AVX2** (`v4`) — no AVX512 |
| Env | WSL2, 28 threads visible, RAM 59 GiB | Unraid OS 7.3.1 bare metal, 32T, 62 GiB |
| Reach | — | `root@192.168.50.170` (passwordless key), glibc 2.43 |
| Load during run | 2.4–4.9 (shared w/ agents) | 1.5–1.7 |

## Results (best of 3, 40 pairs)

| metric | Local | Tower | tower/local |
|---|--:|--:|--:|
| **per-core** (RAYON_NUM_THREADS=1) | 7.51 s → 188 ms/pair | 11.81 s → 295 ms/pair | **0.64×** (1.57× slower/core) |
| **aggregate** (all threads) | 1.37 s → 34 ms/pair → **29.3 pairs/s** | 1.57 s → 39 ms/pair → **25.4 pairs/s** | **0.87×** |

Per-core gap (0.64×) = Zen4 IPC + AVX512 over Zen1+AVX2 — smaller than feared
(the v2 kernel's AVX2 path is competitive). Aggregate gap nearly closes (0.87×):
the tower's 32 real threads vs the local's 28 WSL threads under ~2.4 load.

## Cross-arch parity (the load-bearing correctness check)

AVX512 (local) vs AVX2 (tower) features on the same 40 pairs × 720:
**max abs 1.0e-8, max rel 9.1e-6, 0/28800 exceed the ≤5e-4 v2 numeric policy.**
⟹ tower-computed features are **mergeable** with local + fleet features. A mixed
tower/Hetzner/local training corpus is byte-safe within tolerance.

## Verdict: the tower is a viable FREE, always-on CPU extraction worker

At ~25 pairs/s it does the 2.32M bigcodec corpus solo in ~25 h, or offloads the
paid Hetzner fleet for $0. ~87% of the (loaded) local box's real throughput —
effectively a second comparable machine.

### To use it as a fleet worker — ONE blocker
Its docker image is **STALE**: `ghcr.io/imazen/zenfleet-worker-exec` created
2026-06-24 (digest `25733c31…`) = pre-V2Ab ⟹ would silently emit **372**, not 720.
`docker pull ghcr.io/imazen/zenfleet-worker:exec` (the fresh V2Ab image,
`9d2898e2…`) before enrolling it. (Native `v2_ab_extract` — used for THIS bench —
sidesteps docker entirely and is already staged at
`/mnt/user/coefficient/towerbench/` on the tower.)

## Caveats
- n=40, one fixed set, 3 runs — directional box-ratio, not a size/quality sweep.
- Local was under agent/fleet load (2.4–4.9); vs an idle local box the tower's
  ratio would drop below 0.87 — but local is realistically always shared, so
  0.87× of "available local" is the honest operational number.
- Tower aggregate had run-variance (1.57–1.76 s); best-case reported.

Bench fixtures kept: `/mnt/v/output/zensim/towerbench/` (local) +
`/mnt/user/coefficient/towerbench/` (tower) for re-runs.
