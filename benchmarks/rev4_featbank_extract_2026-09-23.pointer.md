# Rev4 feature bank — artifact pointer (2026-09-23)

Bank bytes are too large for git; this is the tracked record of where they are and what produced them. Report: [`rev4_featbank_extract_2026-09-23.md`](rev4_featbank_extract_2026-09-23.md) · machine-readable: [`rev4_featbank_extract_2026-09-23.json`](rev4_featbank_extract_2026-09-23.json)

**Bank:** `/var/tmp/rev4-featbank/bank/` (696 MB, 18 sets). **Sealed source replicas:** `/var/tmp/rev4-featbank/_sealed/` (239 MB; held-out `human_score`-bearing extraction inputs, review correction 4a). Extractor `extract-native-admission` sha256 `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`.

## bank/ — every file sha256

| file | bytes | sha256 |
|---|---|---|
| `bank/_MANIFEST.json` | 13,516 | `813cd9204912df55aedf9967f45fcb6a368a6ba9867271c00a68c3ad529268c3` |
| `bank/aic3/_MANIFEST.json` | 9,078 | `e73441296ec64433198a537b5fe0dd340003a3bced446848a8b38c4ba22dc389` |
| `bank/aic3/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 1,906,570 | `8ef09a4fb0e4bd82c3bd5b19c1c556dc043524f87a8aeb552e2d0368a2c5b9be` |
| `bank/aic3/keys.parquet` | 53,740 | `f61295406c6090d34f8abfc0b42c2cf7a5160060a07e2f466316916bf5ea801a` |
| `bank/aic3/labels__human.parquet` | 25,198 | `e7f3953cbdea8268efd4ba7d2dcade5e1deeba3d2b0e42130a41c272addceb6b` |
| `bank/aic4/_MANIFEST.json` | 9,003 | `d1d404093f8c75397dab20f966fbd9d4366df43a6e685c254abb050e1329bd24` |
| `bank/aic4/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 1,069,596 | `cf91793f6c46c10b932913a6e549f21622e432b13e2f4b000bcfe0e05a9c5a23` |
| `bank/aic4/keys.parquet` | 28,957 | `b0980834232f14bbacc20b5cd890fe93b86be3a8270ffd2f92908a10f4e5ee05` |
| `bank/cid22_a25/_MANIFEST.json` | 9,215 | `38474bc10a3ad6d89965dcf3cabdc10e7fb677b0ce76606ccd9e73c4ef9009f4` |
| `bank/cid22_a25/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 6,374,135 | `b9699a3659a6016e5be91ee6c774fa6797edd2986534c153e7bab2601b1e4d51` |
| `bank/cid22_a25/keys.parquet` | 203,749 | `45decdcace7e1b054ad31294ee75e9d3c15d7ee3575524a33fc87837b4edc9f0` |
| `bank/cid22_a25/labels__human.parquet` | 117,053 | `e0d955c7a5e80c893ceb4593f70749288fb88685875239e1e7f8180525a2eb12` |
| `bank/cid22_b/_MANIFEST.json` | 9,021 | `ddb6cffa577a58adb8513023130a07e44df32fdccca146cde4ddc5132e6d1973` |
| `bank/cid22_b/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 6,160,266 | `e97013f385573752b1819891894e51d60c03e1137cab45ebf6f30ed872be25a8` |
| `bank/cid22_b/keys.parquet` | 195,258 | `fa1fce412e8a4ffabc43e809216e3eaf0d4114b04307643cfd517e59d959b003` |
| `bank/cid22_train/_MANIFEST.json` | 8,966 | `07bf192a129ddb06c0408f2dd1a43ae42b865b6b02a9b2456a262069915c68d5` |
| `bank/cid22_train/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 48,787,181 | `7da5bd82542ae6d389b60b1ea2efeae14741cd331f054ba92deb140c23bf3bd1` |
| `bank/cid22_train/keys.parquet` | 1,441,691 | `c99a0887705b17bff05bdbfc55b89f9b4f060bc94a009302543f15d9a0127219` |
| `bank/cid22_train/labels__ssim2_oracle.parquet` | 878,408 | `df7a259b0403dfb0e251fa654f3c9c792682005ab3de04f7a777ae691f1ff4fc` |
| `bank/csiq/_MANIFEST.json` | 8,995 | `d9ce0601267c1cc4bd0e72afdd88da8b4531262bd544cdff2da1d70d6f256596` |
| `bank/csiq/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 2,766,628 | `abc633d6c233045d165d78865673ea5bcdbabb102e4bb635b8a7c0ea1eb03610` |
| `bank/csiq/keys.parquet` | 75,358 | `72512f764a4706333e3f7a20708d613896f61a85e624cdd1b3f7c64ac88e8b8d` |
| `bank/kadid_select/_MANIFEST.json` | 9,106 | `c8807a25089e0d354c8cff068477edca410b33cc82085ddc2ec86930af24183b` |
| `bank/kadid_select/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 9,030,333 | `c2b7800bbc357e0f1308e00d05fa1ae15591cd78a6bb725743ed568dad71dd8a` |
| `bank/kadid_select/keys.parquet` | 265,899 | `45ff2ab4eb50572ca3c133330dca036f7ea63e97f42b3bf9be0aa1b0fae7f6ec` |
| `bank/kadid_select/labels__human.parquet` | 137,650 | `e7b53b0f828f39ba069a03e54e82e207fa80377e292794b56e7a43855a52b7f2` |
| `bank/kadid_terminal/_MANIFEST.json` | 9,104 | `658203fac16ab38fc3cd92feeafa797d870768e8fca66f7473be743557e5bb95` |
| `bank/kadid_terminal/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 5,914,422 | `6a31b3b4e759091d9e72f1fa789dfa87743026933c7424297a59cbe3412c51da` |
| `bank/kadid_terminal/keys.parquet` | 169,853 | `f47118a44f09574af4ebdbdfab37fc5a178e5a2b6378c45e694f9e0ea1bc1889` |
| `bank/kadid_train/_MANIFEST.json` | 9,055 | `6976a3ce05fa52552883ac380c85bdfd4cbeb800be23077299069d2b78769b30` |
| `bank/kadid_train/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 14,305,718 | `74a1505b3909bf628cc1a295a3932cccbbaefd609733fbb335c611937042a764` |
| `bank/kadid_train/keys.parquet` | 394,207 | `9c334ce059693e1c53210eeb40311a9d811a55719172fca5616111e38b458161` |
| `bank/kadid_train/labels__human.parquet` | 200,989 | `9402b1819e377e60977048f04056790030edf2b04d51c56c107a45423bd187c5` |
| `bank/konfig_train/_MANIFEST.json` | 9,118 | `2a48a0c3f3686dce24ef9e1288cec0907df17531df32f345d91eda0b1ad8e405` |
| `bank/konfig_train/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 1,161,903 | `109f59a2025e089d71eca6a7b7fd89a6ade4d240677b5535c6def24349e35546` |
| `bank/konfig_train/keys.parquet` | 31,233 | `b66cb8f70b2773bcf4203441b3f03690820c1f027d3f0f121de6ce8c19056f81` |
| `bank/konfig_train/labels__human.parquet` | 14,820 | `c4291d7d6818390b04c3d66d24bedc7c3a2f9c59e9b7c7d7b096ca67a1b39530` |
| `bank/konfig_val/_MANIFEST.json` | 9,122 | `b2fc90e103312ec86968ede3fc0d820b4dc45a48df7e37c3bebd4efff945605e` |
| `bank/konfig_val/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 1,470,078 | `663f9026fcd584002eed0de712278b7ecbb349f932d45630174636dd1321fb2d` |
| `bank/konfig_val/keys.parquet` | 39,810 | `9412dec44eb280662d0109b1d440d929a5d3577efbd3dd31c97eb9d7a635a5f1` |
| `bank/konfig_val/labels__human.parquet` | 19,153 | `634d58a4c6cc6ec83561421c3247f0f5cd424d2dda6801538d5968264889b008` |
| `bank/konjnd_bpg_train/_MANIFEST.json` | 9,164 | `f0badfe8d26e69fb79839458749c385b32d04734e36116bd5c40f25749046292` |
| `bank/konjnd_bpg_train/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 22,577,190 | `f91c4634b683e270e7359e0faa610872eda4fd067dc8e6301d643aed91ad3524` |
| `bank/konjnd_bpg_train/keys.parquet` | 666,733 | `c0278fd07d55cfea7de482b1cea8f9438ea2bd3e8ca61da2bbf875ccdbe8756b` |
| `bank/konjnd_bpg_train/labels__ssim2_oracle.parquet` | 401,840 | `eafd8a50e5e638960ba2c402c166c335d7a321cfbed6ededce1bf56e7903b7d4` |
| `bank/konjnd_bpg_val/_MANIFEST.json` | 9,176 | `59f7099930cde8b2899ad3f76f826a7e12f3615938356fa370ec0ad79d9c2cf9` |
| `bank/konjnd_bpg_val/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 5,864,353 | `9f208bd952c1d1f3021cb2eaab3f249e41f3e805f6ff086fbfe63a7ceea4e94a` |
| `bank/konjnd_bpg_val/keys.parquet` | 180,076 | `e288cff0e14cf8014ad1489496769eced3cc09a771fa6fb04f5778eb2687a731` |
| `bank/konjnd_bpg_val/labels__ssim2_oracle.parquet` | 105,937 | `fff249f9a45aa1709713c199bacf6a6702ec85d154bd996a9fa9b45ab1e6c48a` |
| `bank/konjnd_jpeg_select/_MANIFEST.json` | 9,056 | `300de6b95507ebeb3306d3cda57ce38633b0e4fd12fe9232a2f8bd460b7b0f5d` |
| `bank/konjnd_jpeg_select/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 1,430,119 | `9b6a771ace5632e0087506993103f421b6094908165ad156e07b251362cddf4f` |
| `bank/konjnd_jpeg_select/keys.parquet` | 52,981 | `7f0fa3eebbb2791ef7e69521b25956d48c6742dc7a2e0381355f0617a2aaeb79` |
| `bank/konjnd_jpeg_terminal/_MANIFEST.json` | 9,065 | `81d07b8397219e2addd3d81fe61f598aee3c0a847c8db1a7d601571b3016c545` |
| `bank/konjnd_jpeg_terminal/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 519,056 | `4f87b1173406737f3681cf5b7968fdc260dc81eff8e12d3546cfe08d8cb924d5` |
| `bank/konjnd_jpeg_terminal/keys.parquet` | 17,190 | `136b17ff2c47cab0a35e353e2fe6e643d205ed3575684fbc61b68912b56a9a7c` |
| `bank/mcljci/_MANIFEST.json` | 9,190 | `961bc9289bb491d99bacab4488fdbd7f80b8c88cc5a38ce3c55830cc6c50eed6` |
| `bank/mcljci/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 13,615,413 | `4dc6ef8af2c356b9a26f00158cd48e4107a03a3eaa477a2f761f03c96f31e0c7` |
| `bank/mcljci/keys.parquet` | 406,875 | `6a6e7ab20b50dc6995617426559b7fb72bc0f4ab5a5a687cd9a74ab8db1432e3` |
| `bank/safesyn/_MANIFEST.json` | 8,969 | `44625355c6c5891876b84372e103a653dac3d5acbcc841ce2a2d9088eed519c9` |
| `bank/safesyn/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 544,214,215 | `b2915b4dfb03204f2ba1be72394f05193fedc4e465a360f7eca24edee452cf2c` |
| `bank/safesyn/keys.parquet` | 15,407,413 | `12d48d7fc02afd5026067a348ea20a3896ea6de9a94bd99a25ec2db071c2b28f` |
| `bank/safesyn/labels__ssim2_oracle.parquet` | 11,012,246 | `4c56180ce673de063a2aefb67d3edaf9688e637fad185fc4a5d6b0404d14392a` |
| `bank/tid2013/_MANIFEST.json` | 9,041 | `ec305b5e610dc36815a91c7faf533dd1000cdba5f4faa59a14f4ab924bb69c1d` |
| `bank/tid2013/features__basic+peaks+masked+iw+v2+append+append2__ceiling_rev3__b782e349.parquet` | 8,912,657 | `6fd2b233aa45e6fb76222c85c800d0b857b4f52a28339c9e428e09f71e5c7cdf` |
| `bank/tid2013/keys.parquet` | 262,446 | `f33e47bf643f9b2234babd52420b1fa41676b2d38a2a945b25087ecb9cd72cab` |
| `bank/tid2013/labels__human.parquet` | 141,223 | `3bab8ba7201aae919b305b174caf4a24d01aaf3cf1aa83bf879657ff01c0c73f` |

## _sealed/ — every file sha256

| file | bytes | sha256 |
|---|---|---|
| `_sealed/README.md` | 2,189 | `b4deddc59b6fa761375a5f7b7a0b3bed3a46a72b2007b0e441366b7b0899e3d9` |
| `_sealed/pairs/aic4.tsv` | 60,786 | `4f5d3ec4f059aec92e38125cc393e74f39705e59cd8edffe022c9dd23f09d977` |
| `_sealed/pairs/cid22val.tsv` | 696,095 | `0f225575ddd45d568303476b024d08deb7ffc7d3cb29105591f8f333dc179d17` |
| `_sealed/pairs/csiq.tsv` | 90,665 | `f8c0b24fa2c0f9ab0011f9147be5f32315bad11a23872e66a58a451c38998840` |
| `_sealed/pairs/kadid_terminal.tsv` | 201,983 | `3fb0c946ed58894d9cac2f408c8cf938178d4e9ee219b12a8c1f25762ff4b569` |
| `_sealed/pairs/konjnd_jpeg_val.tsv` | 67,393 | `bd1cf3ddd55152326b90b0ccf66d9beea4f82572bac30f747d4218cfd950480b` |
| `_sealed/pairs/mcljci.tsv` | 973,928 | `b7f4667d52e5b58ee5502d025d3c2c7c5867ef202ae77f7fbb4c43dc652c91b1` |
| `_sealed/raw/aic4.audit.jsonl` | 250,047 | `76b8b3522d4095f89431e2cd3ce46422d48fa4550ad37db5f48d7f08165efaae` |
| `_sealed/raw/aic4.feats.csv` | 5,638,722 | `73c79ac43f795b40b30b21d4e22825d018abadf2aed6a4469436efbf1c3d61ee` |
| `_sealed/raw/aic4.feats.csv.manifest.json` | 8,636 | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `_sealed/raw/aic4.feats.csv.producer.bin` | 15,231 | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `_sealed/raw/cid22val.audit.jsonl` | 3,370,371 | `ba9f70d2f285c2a45bacce861ab50678e2749cf8be953ee51ff8882c557e2267` |
| `_sealed/raw/cid22val.feats.csv` | 79,875,556 | `9b22251eab31a7cd2da0ab9934eda2bf1cbc6a932719f36f95a0a4ca9f06d524` |
| `_sealed/raw/cid22val.feats.csv.manifest.json` | 8,636 | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `_sealed/raw/cid22val.feats.csv.producer.bin` | 15,231 | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `_sealed/raw/csiq.audit.jsonl` | 629,110 | `8a56918d128167c66b78227118b40b063bf11e359f7e99f06b93a065e29d66e8` |
| `_sealed/raw/csiq.feats.csv` | 15,813,841 | `60dc32906adca2c4c74519a3edf87481d11e3d6831185238b93cce02cffea709` |
| `_sealed/raw/csiq.feats.csv.manifest.json` | 8,636 | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `_sealed/raw/csiq.feats.csv.producer.bin` | 15,231 | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `_sealed/raw/kadid_terminal.audit.jsonl` | 1,435,880 | `a253cba97a060f1d8c351a51dd7cc6b46bb9d9ca2379bc3cf9945f45f396b173` |
| `_sealed/raw/kadid_terminal.feats.csv` | 35,239,672 | `07e0b63fe27b8b04b1b543f6ec0907752e4ab0541f00cb6631a9bca6b51cb485` |
| `_sealed/raw/kadid_terminal.feats.csv.manifest.json` | 8,636 | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `_sealed/raw/kadid_terminal.feats.csv.producer.bin` | 15,231 | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `_sealed/raw/konjnd_jpeg_val.audit.jsonl` | 380,338 | `9c799355661121ad4b539169ed318ec757e343dc0a203530c195fa252487c004` |
| `_sealed/raw/konjnd_jpeg_val.feats.csv` | 9,279,165 | `643704f7aa191063b037ffa6811672f06cb21f67b0196167c0bb6666d6d68e1b` |
| `_sealed/raw/konjnd_jpeg_val.feats.csv.manifest.json` | 8,636 | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `_sealed/raw/konjnd_jpeg_val.feats.csv.producer.bin` | 15,231 | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `_sealed/raw/mcljci.audit.jsonl` | 4,113,105 | `99cc0c83f3aa4ed0c6dc58d6851383bcc74d005f3fcf381a827c6c23354dd80b` |
| `_sealed/raw/mcljci.feats.csv` | 91,989,410 | `d3cd1e4b1304f758ce0a7b731a7ea65e7fdd9116d058eb12926c6a6ef7db74ea` |
| `_sealed/raw/mcljci.feats.csv.manifest.json` | 8,636 | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `_sealed/raw/mcljci.feats.csv.producer.bin` | 15,231 | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
