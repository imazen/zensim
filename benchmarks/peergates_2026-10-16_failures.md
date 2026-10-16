# Peer steering failures — 2026-10-16

Engineering identifiers describe content provenance; no semantic class is inferred.
Candidate misses: M2<.99; listed M3f values all pass .70. Peer misses: own-map rho<.70; M2=N/A.
Peers: butteraugli0.9.3 pnorm3, SSIMULACRA2 native-map54fabca9.
All51 SSIMULACRA2 failures retained despite its lower task priority.

| Model | Case / block | Source distortion / geometry | M2 | M3f or peer own-map rho |
|---|---|---|---:|---:|
| diff-410 | broad-10-b32 / 32 | 2010-scalar-10.png; 205x256 | 0.9650540851 | 0.9650199082 |
| diff-410 | broad-10-b64 / 64 | 2010-scalar-10.png; 205x256 | 0.9382352941 | 0.9352941176 |
| diff-410 | broad-76-b64 / 64 | 6068-scalar-13.png; 196x256 | 0.9882352941 | 0.9970588235 |
| diff-410 | broad-136-b64 / 64 | 7066-scalar-10.png; 256x256 | 0.9676470588 | 0.9794117647 |
| diff+frag-820 | broad-31-b64 / 64 | 1054-scalar-10.png; 192x256 | 0.7272727273 | 0.7272727273 |
| diff+frag-820 | broad-34-b32 / 32 | 1054-scalar-13.png; 192x256 | 0.9518019974 | 0.9468085106 |
| diff+frag-820 | broad-34-b64 / 64 | 1054-scalar-13.png; 192x256 | 0.9720279720 | 0.9720279720 |
| diff+frag-820 | broad-202-b32 / 32 | 8206-scalar-13.png; 256x144 | 0.9666041276 | 0.9660412758 |
| diff+frag-820 | broad-202-b64 / 64 | 8206-scalar-13.png; 256x144 | 0.8601398601 | 0.9370629371 |
| W-C seed0 | broad-13-b64 / 64 | 2010-scalar-13.png; 205x256 | 0.8676470588 | 0.8676470588 |
| W-C seed0 | broad-202-b64 / 64 | 8206-scalar-13.png; 256x144 | 0.9790209790 | 0.9860139860 |
| butter | broad-199-b64 / 64 | 8206-scalar-10.png; 256x144 | N/A | 0.5034965035 |
| butter | broad-202-b64 / 64 | 8206-scalar-13.png; 256x144 | N/A | 0.3566433566 |
| butter | broad-206-b64 / 64 | 8206-scalar-17.png; 256x144 | N/A | 0.3776223776 |
| butter | broad-223-b64 / 64 | 8384-scalar-13.png; 192x256 | N/A | 0.6690027773 |
| ssim2 | broad-31-b8 / 8 | 1054-scalar-10.png; 192x256 | N/A | 0.2360039050 |
| ssim2 | broad-31-b16 / 16 | 1054-scalar-10.png; 192x256 | N/A | 0.6254459078 |
| ssim2 | broad-34-b8 / 8 | 1054-scalar-13.png; 192x256 | N/A | 0.6341278973 |
| ssim2 | broad-10-b8 / 8 | 2010-scalar-10.png; 205x256 | N/A | 0.3187165522 |
| ssim2 | broad-10-b16 / 16 | 2010-scalar-10.png; 205x256 | N/A | 0.4367348898 |
| ssim2 | broad-13-b8 / 8 | 2010-scalar-13.png; 205x256 | N/A | 0.5730826071 |
| ssim2 | broad-73-b8 / 8 | 6068-scalar-10.png; 196x256 | N/A | 0.2388414223 |
| ssim2 | broad-73-b16 / 16 | 6068-scalar-10.png; 196x256 | N/A | 0.2334676873 |
| ssim2 | broad-73-b32 / 32 | 6068-scalar-10.png; 196x256 | N/A | 0.4951212427 |
| ssim2 | broad-76-b8 / 8 | 6068-scalar-13.png; 196x256 | N/A | 0.0929069804 |
| ssim2 | broad-76-b16 / 16 | 6068-scalar-13.png; 196x256 | N/A | 0.4563845395 |
| ssim2 | broad-94-b8 / 8 | 6610-scalar-10.png; 256x202 | N/A | 0.2368357854 |
| ssim2 | broad-94-b16 / 16 | 6610-scalar-10.png; 256x202 | N/A | 0.4561723555 |
| ssim2 | broad-97-b8 / 8 | 6610-scalar-13.png; 256x202 | N/A | 0.5034774046 |
| ssim2 | broad-136-b8 / 8 | 7066-scalar-10.png; 256x256 | N/A | 0.0456187862 |
| ssim2 | broad-136-b16 / 16 | 7066-scalar-10.png; 256x256 | N/A | 0.1163953422 |
| ssim2 | broad-136-b32 / 32 | 7066-scalar-10.png; 256x256 | N/A | 0.4807692308 |
| ssim2 | broad-139-b8 / 8 | 7066-scalar-13.png; 256x256 | N/A | 0.1683143083 |
| ssim2 | broad-139-b16 / 16 | 7066-scalar-13.png; 256x256 | N/A | 0.0911550126 |
| ssim2 | broad-139-b32 / 32 | 7066-scalar-13.png; 256x256 | N/A | 0.1562271062 |
| ssim2 | broad-143-b8 / 8 | 7066-scalar-17.png; 256x256 | N/A | 0.2519262079 |
| ssim2 | broad-143-b16 / 16 | 7066-scalar-17.png; 256x256 | N/A | -0.0262846189 |
| ssim2 | broad-143-b32 / 32 | 7066-scalar-17.png; 256x256 | N/A | -0.0663003663 |
| ssim2 | broad-199-b8 / 8 | 8206-scalar-10.png; 256x144 | N/A | 0.0332180243 |
| ssim2 | broad-199-b16 / 16 | 8206-scalar-10.png; 256x144 | N/A | 0.0761728770 |
| ssim2 | broad-199-b32 / 32 | 8206-scalar-10.png; 256x144 | N/A | 0.1273518337 |
| ssim2 | broad-199-b64 / 64 | 8206-scalar-10.png; 256x144 | N/A | 0.5874125874 |
| ssim2 | broad-202-b8 / 8 | 8206-scalar-13.png; 256x144 | N/A | 0.2569359263 |
| ssim2 | broad-202-b16 / 16 | 8206-scalar-13.png; 256x144 | N/A | 0.4029807477 |
| ssim2 | broad-206-b8 / 8 | 8206-scalar-17.png; 256x144 | N/A | 0.5593987934 |
| ssim2 | broad-220-b8 / 8 | 8384-scalar-10.png; 192x256 | N/A | 0.1208726310 |
| ssim2 | broad-220-b16 / 16 | 8384-scalar-10.png; 192x256 | N/A | 0.1617715267 |
| ssim2 | broad-220-b32 / 32 | 8384-scalar-10.png; 192x256 | N/A | 0.4511926237 |
| ssim2 | broad-220-b64 / 64 | 8384-scalar-10.png; 192x256 | N/A | 0.4518395721 |
| ssim2 | broad-223-b8 / 8 | 8384-scalar-13.png; 192x256 | N/A | -0.0497083650 |
| ssim2 | broad-223-b16 / 16 | 8384-scalar-13.png; 192x256 | N/A | 0.0113410887 |
| ssim2 | broad-223-b32 / 32 | 8384-scalar-13.png; 192x256 | N/A | 0.5627326043 |
| ssim2 | broad-227-b8 / 8 | 8384-scalar-17.png; 192x256 | N/A | 0.6166622343 |
| ssim2 | broad-157-b8 / 8 | 9380-scalar-10.png; 171x256 | N/A | 0.1152114306 |
| ssim2 | broad-157-b16 / 16 | 9380-scalar-10.png; 171x256 | N/A | 0.3621379709 |
| ssim2 | broad-157-b32 / 32 | 9380-scalar-10.png; 171x256 | N/A | 0.5651297514 |
| ssim2 | broad-160-b8 / 8 | 9380-scalar-13.png; 171x256 | N/A | 0.6438618023 |
| ssim2 | I01_10_03 / 8 | I01_10_03.png; 512x384 | N/A | 0.6300551325 |
| ssim2 | I01_10_05 / 8 | I01_10_05.png; 512x384 | N/A | 0.6261000825 |
| ssim2 | I21_10_03 / 8 | I21_10_03.png; 512x384 | N/A | 0.6866059725 |
| ssim2 | I21_10_05 / 8 | I21_10_05.png; 512x384 | N/A | 0.6274786193 |
| ssim2 | I41_10_03 / 8 | I41_10_03.png; 512x384 | N/A | 0.6764031095 |
| ssim2 | I41_10_05 / 8 | I41_10_05.png; 512x384 | N/A | 0.6802794607 |
| ssim2 | I61_10_03 / 8 | I61_10_03.png; 512x384 | N/A | 0.6538887973 |
| ssim2 | I61_10_05 / 8 | I61_10_05.png; 512x384 | N/A | 0.5876663507 |
| ssim2 | city-384-50 / 32 | city_384_q50.jpg; 384x384 | N/A | 0.6814966643 |

Full per-case/block cross-judge values, including every cross-judge miss:
`/mnt/v/output/zensim/peergates-2026-10-16/{steering,steering-remaining}/results.jsonl`.
Content metadata SHA256: `47d04ecd9985994e89d4595c8bd3656af31baf2f3d94f170767ec6aa92845334`.
Original candidate values: [D12](e34_d12_2026-10-15.json);
aggregate findings and pins: [peer gates](peergates_2026-10-16.md).
