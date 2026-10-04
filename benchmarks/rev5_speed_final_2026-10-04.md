# Rev5 speed measurements — 2026-10-04

**UNQUALIFIED:** strict zenbench reports resource interference. These measurements
are provisional and do not certify the quiet-box or kernel stop gate. Raw
interleaved rounds, execution orders and gate flags are retained in
`/var/tmp/rev5/owner-speed-matrix/`.

Inputs are the benchmark’s deterministic square RGB8 pairs (64², 256²,
1024², 2048²), not a codec/corpus throughput result. Both bakes use the
same network weights across revision stamps; these are timing arms, not
retrained Rev5 quality models. Scores call `BakeScorer::compute`; maps use
`prepare_steering` and reuse the worker. Setup plus three warmup calls are
untimed. Every reported time includes the parent/worker pipe round-trip.

Thirty rounds per group; one call per arm per round, randomized interleaving.
Single-thread CPU 16, eight-thread CPUs 16–23; `RAYON_NUM_THREADS=1|8`.
Native v4x uses the available AVX-512 tier; v3 is forced with
`ZENSIM_MAX_TIER=v3`. `uptime` is recorded at the start of every run.
The final executable is `/var/tmp/rev5/xp_owner_final`. Reproduction drivers:
`/var/tmp/rev5/owner-speed-matrix.py` and the `ZEN_XP_EXTERNAL_MODELS` path in
`zensim/benches/extract_paths_bench.rs`. All heavy commands run through
`scripts/run-heavy --mem 16G --jobs 8 --`, with private target/scratch.

## Mean milliseconds

| tier | threads | path | bake | revision | 64² | 256² | 1 MP | 4 MP |
|---|---:|---|---|---:|---:|---:|---:|---:|
| v4x | 1 | score | by_v2fy | 3 | 0.3573 | 2.0026 | 27.7583 | 112.5339 |
| v4x | 1 | score | by_v2fy | 4 | 0.3854 | 2.2557 | 34.2143 | 136.7219 |
| v4x | 1 | score | by_v2fy | 5 | 0.4295 | 2.0716 | 23.1004 | 92.9226 |
| v4x | 1 | score | v2basic | 3 | 0.4330 | 2.8483 | 49.2399 | 205.8385 |
| v4x | 1 | score | v2basic | 4 | 0.4738 | 3.1967 | 63.1449 | 278.5411 |
| v4x | 1 | score | v2basic | 5 | 0.5739 | 3.4571 | 42.0103 | 166.1356 |
| v4x | 1 | map | by_v2fy | 3 | 3.5319 | 6.6179 | 61.8644 | 236.9782 |
| v4x | 1 | map | by_v2fy | 4 | 3.5667 | 6.8418 | 67.6762 | 276.1305 |
| v4x | 1 | map | by_v2fy | 5 | 3.5445 | 5.8912 | 40.7353 | 153.3019 |
| v4x | 1 | map | v2basic | 3 | 4.9138 | 9.9805 | 110.9438 | 432.1534 |
| v4x | 1 | map | v2basic | 4 | 4.9244 | 10.3972 | 124.5434 | 502.8684 |
| v4x | 1 | map | v2basic | 5 | 4.9110 | 8.9905 | 73.4300 | 274.5864 |
| v4x | 8 | score | by_v2fy | 3 | 0.4264 | 1.3422 | 15.7522 | 60.0070 |
| v4x | 8 | score | by_v2fy | 4 | 0.4187 | 1.4496 | 17.7496 | 67.8176 |
| v4x | 8 | score | by_v2fy | 5 | 0.4224 | 1.4204 | 15.6754 | 63.3732 |
| v4x | 8 | score | v2basic | 3 | 0.4296 | 1.4048 | 16.9568 | 68.1191 |
| v4x | 8 | score | v2basic | 4 | 0.4347 | 1.5620 | 19.7870 | 77.7972 |
| v4x | 8 | score | v2basic | 5 | 0.4047 | 1.4144 | 14.8419 | 60.4530 |
| v4x | 8 | map | by_v2fy | 3 | 1.3430 | 3.2680 | 30.8050 | 113.0833 |
| v4x | 8 | map | by_v2fy | 4 | 1.3457 | 3.3699 | 32.8866 | 127.0827 |
| v4x | 8 | map | by_v2fy | 5 | 1.2490 | 2.8367 | 26.3264 | 108.8965 |
| v4x | 8 | map | v2basic | 3 | 1.6893 | 4.2910 | 44.9353 | 180.3202 |
| v4x | 8 | map | v2basic | 4 | 1.7855 | 4.3695 | 47.3555 | 188.1428 |
| v4x | 8 | map | v2basic | 5 | 1.5698 | 3.3094 | 32.4513 | 134.6006 |
| v3 | 1 | score | by_v2fy | 3 | 0.3554 | 2.1450 | 47.1235 | 172.7227 |
| v3 | 1 | score | by_v2fy | 4 | 0.4055 | 2.5712 | 40.0593 | 172.4446 |
| v3 | 1 | score | by_v2fy | 5 | 0.4543 | 2.5483 | 32.1459 | 126.5169 |
| v3 | 1 | score | v2basic | 3 | 0.4341 | 3.0014 | 99.7610 | 277.2477 |
| v3 | 1 | score | v2basic | 4 | 0.5698 | 3.8621 | 77.8548 | 316.3368 |
| v3 | 1 | score | v2basic | 5 | 0.6207 | 4.3784 | 60.3811 | 237.7848 |
| v3 | 1 | map | by_v2fy | 3 | 3.5389 | 6.7071 | 93.3898 | 304.0159 |
| v3 | 1 | map | by_v2fy | 4 | 3.5805 | 7.1864 | 75.1679 | 306.1307 |
| v3 | 1 | map | by_v2fy | 5 | 3.5513 | 6.3711 | 48.7670 | 188.6103 |
| v3 | 1 | map | v2basic | 3 | 4.7879 | 10.0787 | 197.4628 | 507.3180 |
| v3 | 1 | map | v2basic | 4 | 4.8627 | 11.2240 | 142.4859 | 563.7447 |
| v3 | 1 | map | v2basic | 5 | 4.8755 | 9.8231 | 88.4035 | 347.7136 |
| v3 | 8 | score | by_v2fy | 3 | 0.4202 | 1.3872 | 20.9300 | 74.1110 |
| v3 | 8 | score | by_v2fy | 4 | 0.4428 | 1.7803 | 21.8130 | 84.4038 |
| v3 | 8 | score | by_v2fy | 5 | 0.4203 | 1.6265 | 19.5173 | 78.1548 |
| v3 | 8 | score | v2basic | 3 | 0.4343 | 1.4754 | 25.5120 | 82.1229 |
| v3 | 8 | score | v2basic | 4 | 0.4373 | 1.7555 | 23.7480 | 93.5847 |
| v3 | 8 | score | v2basic | 5 | 0.4411 | 1.5931 | 18.4100 | 73.9649 |
| v3 | 8 | map | by_v2fy | 3 | 1.2612 | 3.2934 | 36.7919 | 140.6050 |
| v3 | 8 | map | by_v2fy | 4 | 1.2764 | 3.5218 | 35.3972 | 149.5017 |
| v3 | 8 | map | by_v2fy | 5 | 1.1468 | 3.0255 | 29.7106 | 123.8737 |
| v3 | 8 | map | v2basic | 3 | 1.6274 | 5.3339 | 57.6255 | 203.3637 |
| v3 | 8 | map | v2basic | 4 | 1.7105 | 4.5938 | 50.7798 | 214.3518 |
| v3 | 8 | map | v2basic | 5 | 1.6665 | 3.4988 | 36.2070 | 152.9296 |

## Fixed + per-pixel fit

OLS over the four sizes: `time_ns = fixed_ns + ns_per_pixel × pixels`.
The fixed term includes IPC. Negative fitted intercepts are left visible;
they are regression coefficients, not physical negative setup costs.
Residuals quantify the limits of the linear approximation.

| tier | threads | path | bake | revision | fixed ms | ns/pixel | R² | max residual ms |
|---|---:|---|---|---:|---:|---:|---:|---:|
| v4x | 1 | score | by_v2fy | 3 | 0.0897 | 26.7845 | 0.99997 | 0.4170 |
| v4x | 1 | score | by_v2fy | 4 | 0.1576 | 32.5546 | 1.00000 | 0.0944 |
| v4x | 1 | score | by_v2fy | 5 | 0.3433 | 22.0519 | 0.99996 | 0.3660 |
| v4x | 1 | score | v2basic | 3 | -0.6777 | 49.1426 | 0.99987 | 1.6121 |
| v4x | 1 | score | v2basic | 4 | -2.2040 | 66.6676 | 0.99944 | 4.5572 |
| v4x | 1 | score | v2basic | 5 | 0.6387 | 39.4581 | 0.99999 | 0.2324 |
| v4x | 1 | map | by_v2fy | 3 | 3.2109 | 55.7455 | 1.00000 | 0.2464 |
| v4x | 1 | map | by_v2fy | 4 | 1.9359 | 65.2181 | 0.99980 | 2.6457 |
| v4x | 1 | map | by_v2fy | 5 | 3.4184 | 35.7268 | 1.00000 | 0.1454 |
| v4x | 1 | map | v2basic | 3 | 3.8926 | 102.1026 | 0.99999 | 0.6035 |
| v4x | 1 | map | v2basic | 4 | 2.4481 | 119.1416 | 0.99992 | 2.8337 |
| v4x | 1 | map | v2basic | 5 | 5.0626 | 64.3137 | 0.99998 | 0.9296 |
| v4x | 8 | score | by_v2fy | 3 | 0.5185 | 14.2030 | 0.99993 | 0.3407 |
| v4x | 8 | score | by_v2fy | 4 | 0.5192 | 16.0675 | 0.99993 | 0.3824 |
| v4x | 8 | score | by_v2fy | 5 | 0.2668 | 15.0257 | 0.99994 | 0.3470 |
| v4x | 8 | score | v2basic | 3 | 0.2592 | 16.1644 | 0.99997 | 0.2520 |
| v4x | 8 | score | v2basic | 4 | 0.3756 | 18.4618 | 1.00000 | 0.0528 |
| v4x | 8 | score | v2basic | 5 | 0.2473 | 14.3294 | 0.99989 | 0.4308 |
| v4x | 8 | map | by_v2fy | 3 | 1.8002 | 26.5973 | 0.99979 | 1.1155 |
| v4x | 8 | map | by_v2fy | 4 | 1.3500 | 29.9830 | 1.00000 | 0.1271 |
| v4x | 8 | map | by_v2fy | 5 | 0.6525 | 25.7314 | 0.99970 | 1.3075 |
| v4x | 8 | map | v2basic | 3 | 1.1468 | 42.6632 | 0.99994 | 0.9471 |
| v4x | 8 | map | v2basic | 4 | 1.2960 | 44.5118 | 0.99998 | 0.6145 |
| v4x | 8 | map | v2basic | 5 | 0.7143 | 31.8256 | 0.99969 | 1.6345 |
| v3 | 1 | score | by_v2fy | 3 | 0.9681 | 41.1244 | 0.99936 | 3.0332 |
| v3 | 1 | score | by_v2fy | 4 | -0.8138 | 41.1737 | 0.99964 | 2.3007 |
| v3 | 1 | score | by_v2fy | 5 | 0.5035 | 30.0520 | 0.99999 | 0.1723 |
| v3 | 1 | score | v2basic | 3 | 8.1247 | 65.4955 | 0.98597 | 22.9594 |
| v3 | 1 | score | v2basic | 4 | -0.6678 | 75.5376 | 0.99998 | 0.9282 |
| v3 | 1 | score | v2basic | 5 | 0.6806 | 56.5538 | 0.99999 | 0.3995 |
| v3 | 1 | map | by_v2fy | 3 | 6.9724 | 71.4845 | 0.99703 | 11.4605 |
| v3 | 1 | map | by_v2fy | 4 | 1.8783 | 72.3861 | 0.99984 | 2.6128 |
| v3 | 1 | map | by_v2fy | 5 | 3.1548 | 44.1750 | 0.99997 | 0.7087 |
| v3 | 1 | map | v2basic | 3 | 22.4808 | 118.5361 | 0.97929 | 50.6879 |
| v3 | 1 | map | v2basic | 4 | 3.1182 | 133.6175 | 0.99999 | 1.1972 |
| v3 | 1 | map | v2basic | 5 | 3.9650 | 81.8738 | 0.99997 | 1.4124 |
| v3 | 8 | score | by_v2fy | 3 | 0.9070 | 17.5473 | 0.99901 | 1.6233 |
| v3 | 8 | score | by_v2fy | 4 | 0.5271 | 20.0153 | 0.99997 | 0.2983 |
| v3 | 8 | score | by_v2fy | 5 | 0.2916 | 18.5510 | 0.99998 | 0.2265 |
| v3 | 8 | score | v2basic | 3 | 1.6219 | 19.3989 | 0.99614 | 3.5488 |
| v3 | 8 | score | v2basic | 4 | 0.3535 | 22.2327 | 1.00000 | 0.0818 |
| v3 | 8 | score | v2basic | 5 | 0.2974 | 17.5472 | 0.99997 | 0.2870 |
| v3 | 8 | map | by_v2fy | 3 | 1.3453 | 33.2367 | 0.99996 | 0.5954 |
| v3 | 8 | map | by_v2fy | 4 | 0.3630 | 35.4343 | 0.99959 | 2.1213 |
| v3 | 8 | map | by_v2fy | 5 | 0.4864 | 29.3291 | 0.99969 | 1.5296 |
| v3 | 8 | map | v2basic | 3 | 3.3353 | 47.9263 | 0.99917 | 4.0358 |
| v3 | 8 | map | v2basic | 4 | 0.3009 | 50.8671 | 0.99963 | 2.8592 |
| v3 | 8 | map | v2basic | 5 | 0.4747 | 36.2170 | 0.99955 | 2.2439 |

## Evidence

| run | completed rounds per size | gate waits | unreliable |
|---|---|---:|---|
| v3-t1-map | 30,30,30,30 | 120 | True |
| v3-t1-score | 30,30,30,30 | 120 | True |
| v3-t8-map | 30,30,30,30 | 120 | True |
| v3-t8-score | 30,30,30,30 | 120 | True |
| v4x-t1-map | 30,30,30,30 | 120 | True |
| v4x-t1-score | 30,30,30,30 | 120 | True |
| v4x-t8-map | 30,30,30,30 | 120 | True |
| v4x-t8-score | 30,30,30,30 | 120 | True |

Final post-directive executable SHA-256: `1eb6419af3491ecb7dc573b9c2a64f55a8cd8f054861bb5d9cd5825f89836f21`.

Bake hashes (SHA-256):

```text
c87e82fee12e147f715aba798699142cf6912ba8f56ed2559924bd5b6ee8863d  by_v2fy.r3.bin
dda9f713c34c609e87bcd9835bcd8efa2338942d0a0e8d2c754cd19e0cb09bb1  by_v2fy.r4.bin
1324bc8548f4a1f829f076ddb9a29d058167d5770f67f5e15c8c7e7604c974c2  by_v2fy.r5.bin
ec2d1e124c8ebe7107ec491613d23dc0ad1458f99b8183e2e6b1d3c01d16eda6  v2basic.r3.bin
8a6d244ffc6b8f9a4774964283aa8e03e52fde0e1390616125769289c6747b09  v2basic.r4.bin
885e59ff3186ace1323bc402036de4a5294dc209a9624fa763ada6bed37ed054  v2basic.r5.bin
```
