# restore-cuts extraction log (per set, verbatim result lines)

### konjnd_jpeg_terminal

UTC 2026-09-25T02:05:02Z-2026-09-25T03:13:56Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh konjnd_jpeg_terminal`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/konjnd_jpeg_terminal.log`:

```text
RESTORE_START set=konjnd_jpeg_terminal utc=2026-09-25T03:13:55Z
Loaded 100 pairs from pairs-tsv
scored 100/100 pairs in 1.5s (0 failed)
Wrote 100 rows × 1825 features to /var/tmp/restore-cuts/raw/konjnd_jpeg_terminal.csv
WALL_SECONDS=1.52 MAXRSS_KB=348696
RESTORE_CHECK set=konjnd_jpeg_terminal family=mapdev rows=100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_jpeg_terminal family=z1max rows=100 finite=1 unique=1 identity_violations=0 dead_columns=[1574, 1612, 1631]
RESTORE_CHECK set=konjnd_jpeg_terminal family=gmsnative rows=100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_jpeg_terminal family=dvifmgate rows=100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=konjnd_jpeg_terminal rows=100 manifest_sha256=a5f86f6d0347ba87b45d9ea9e6592c15c5da4f06453c117b8387909722d02b73 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
62d22da436fa63db28082ee808422a7148ba384a85fed04fc35a780eccb2edfe  /var/tmp/restore-cuts/pairs/konjnd_jpeg_terminal.tsv
019c3ac135cbff9b3e126576f15091295e318c2df54d0f3f450aa35ccfb74017  /var/tmp/restore-cuts/raw/konjnd_jpeg_terminal.csv
92c149fec422c6fb0d21248cbb2c3416c0e2830cb02f0e7308476d536e75b8fb  /var/tmp/restore-cuts/raw/konjnd_jpeg_terminal.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/konjnd_jpeg_terminal.csv.manifest.json
RESTORE_END set=konjnd_jpeg_terminal utc=2026-09-25T03:13:56Z
```
Log sha256: `83e7769ca495f8c26e4bd0c69919a509a08bc61b007363381cb3bf61ff726ed7`.

### aic4

UTC 2026-09-25T03:13:56Z-2026-09-25T04:57:53Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh aic4`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/aic4.log`:

```text
RESTORE_START set=aic4 utc=2026-09-25T04:57:45Z
Loaded 300 pairs from pairs-tsv
scored 300/300 pairs in 7.6s (0 failed)
Wrote 300 rows × 1825 features to /var/tmp/restore-cuts/raw/aic4.csv
WALL_SECONDS=7.68 MAXRSS_KB=463380
RESTORE_CHECK set=aic4 family=mapdev rows=300 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=aic4 family=z1max rows=300 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=aic4 family=gmsnative rows=300 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=aic4 family=dvifmgate rows=300 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=aic4 rows=300 manifest_sha256=6985469a2d91a7032d84045fe64f7675b519b77f1ca11c119681a5adc44385c5 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
c14922665a0d0ead8128e5fbe1d79a266fc47d91ce9c28d607dd7a12ab6cdf99  /var/tmp/restore-cuts/pairs/aic4.tsv
9e7d0a147ba46e1488e9088ce60bcacc15e6e3c3b4068aa96964e625dc04721e  /var/tmp/restore-cuts/raw/aic4.csv
ddb6d7a64f5f2683a8e3eeb4445ab4e4820662bbf1179cf430826c352762d4a4  /var/tmp/restore-cuts/raw/aic4.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/aic4.csv.manifest.json
RESTORE_END set=aic4 utc=2026-09-25T04:57:53Z
```
Log sha256: `cc392355eff3495d378504b1008894f8245edce46f8ae3dc6c4323c09facda38`.

### konfig_train

UTC 2026-09-25T04:57:53Z-2026-09-25T06:26:48Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh konfig_train`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/konfig_train.log`:

```text
RESTORE_START set=konfig_train utc=2026-09-25T06:26:44Z
Loaded 327 pairs from pairs-tsv
scored 327/327 pairs in 3.2s (0 failed)
Wrote 327 rows × 1825 features to /var/tmp/restore-cuts/raw/konfig_train.csv
WALL_SECONDS=3.27 MAXRSS_KB=218164
RESTORE_CHECK set=konfig_train family=mapdev rows=327 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konfig_train family=z1max rows=327 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konfig_train family=gmsnative rows=327 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konfig_train family=dvifmgate rows=327 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=konfig_train rows=327 manifest_sha256=ae7b3a4bd38e0dfb86461f7d4a1ecae543babd1681927026920cc1e6dc4b6235 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
57b9f1c13ea72a4194983a6356e57d21cb8980f1f7948c0f05456ac834b5a155  /var/tmp/restore-cuts/pairs/konfig_train.tsv
d961cf3406d8d780a1bab9f61323dc72d027ddb421ae99e35036c9f083f65e26  /var/tmp/restore-cuts/raw/konfig_train.csv
a271a7f5a309941c36a4cfc94a91581aa470cfb04f0d807ed7e74ec8c4db4057  /var/tmp/restore-cuts/raw/konfig_train.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/konfig_train.csv.manifest.json
RESTORE_END set=konfig_train utc=2026-09-25T06:26:48Z
```
Log sha256: `c0f7d698f67e00dc309dd3f61592667132cdd95c422be2aeb668102e4a164acf`.

### konfig_val

UTC 2026-09-25T06:26:48Z-2026-09-25T07:29:29Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh konfig_val`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/konfig_val.log`:

```text
RESTORE_START set=konfig_val utc=2026-09-25T07:29:24Z
Loaded 436 pairs from pairs-tsv
scored 436/436 pairs in 4.2s (0 failed)
Wrote 436 rows × 1825 features to /var/tmp/restore-cuts/raw/konfig_val.csv
WALL_SECONDS=4.30 MAXRSS_KB=203624
RESTORE_CHECK set=konfig_val family=mapdev rows=436 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konfig_val family=z1max rows=436 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konfig_val family=gmsnative rows=436 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konfig_val family=dvifmgate rows=436 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=konfig_val rows=436 manifest_sha256=495b6d5350291031c86098f3631c695d6cb9782b7f62e6001834d6ca8a380f6f feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
bdd9c7de630767bfcab2381b34ca1a0b8757a11b5beab87f744eafd317c952b5  /var/tmp/restore-cuts/pairs/konfig_val.tsv
f1c37dbc6e80e17e1c4eefff6505f45db54dd4bd10279ad86c5e10a998d20355  /var/tmp/restore-cuts/raw/konfig_val.csv
afebf5eb48c090f0cd5952a3cce144fc8aa40590fe4e3ee9fe255465bdd274a5  /var/tmp/restore-cuts/raw/konfig_val.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/konfig_val.csv.manifest.json
RESTORE_END set=konfig_val utc=2026-09-25T07:29:29Z
```
Log sha256: `b2b426b7a3cd4b0d59297cec298a8614f0d773c3d80edede41a603474c958bbe`.

### aic3

UTC 2026-09-25T07:29:29Z-2026-09-25T08:28:05Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh aic3`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/aic3.log`:

```text
RESTORE_START set=aic3 utc=2026-09-25T08:26:50Z
Loaded 600 pairs from pairs-tsv
scored 600/600 pairs in 73.8s (0 failed)
Wrote 600 rows × 1825 features to /var/tmp/restore-cuts/raw/aic3.csv
WALL_SECONDS=73.93 MAXRSS_KB=3026436
RESTORE_CHECK set=aic3 family=mapdev rows=600 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=aic3 family=z1max rows=600 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=aic3 family=gmsnative rows=600 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=aic3 family=dvifmgate rows=600 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=aic3 rows=600 manifest_sha256=55bc92726289246c61827decd8e71aba8fa950f73b9b849e5b1155044a89b4f4 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
ffda1b0d34b09ebe9842722c7cb0975d0613ab576d62644e193f401decd4f7bd  /var/tmp/restore-cuts/pairs/aic3.tsv
11a206152a8950652af491521b6ff5ee852edac2ac39041535b3bb7b12790436  /var/tmp/restore-cuts/raw/aic3.csv
290fb8bdfd1e5143376e580430a369abbf5ab8ae4cd9f29a3a3e73375bc29814  /var/tmp/restore-cuts/raw/aic3.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/aic3.csv.manifest.json
RESTORE_END set=aic3 utc=2026-09-25T08:28:05Z
```
Log sha256: `7d33ddfb2de43b4450965ae8e0ee48972b7eee970a44d9e0331d2296fb59792c`.

### konjnd_jpeg_select

UTC 2026-09-25T08:28:05Z-2026-09-25T09:40:14Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh konjnd_jpeg_select`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/konjnd_jpeg_select.log`:

```text
RESTORE_START set=konjnd_jpeg_select utc=2026-09-25T09:40:07Z
Loaded 404 pairs from pairs-tsv
scored 404/404 pairs in 5.8s (0 failed)
Wrote 404 rows × 1825 features to /var/tmp/restore-cuts/raw/konjnd_jpeg_select.csv
WALL_SECONDS=5.91 MAXRSS_KB=349628
RESTORE_CHECK set=konjnd_jpeg_select family=mapdev rows=404 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_jpeg_select family=z1max rows=404 finite=1 unique=1 identity_violations=0 dead_columns=[1574]
RESTORE_CHECK set=konjnd_jpeg_select family=gmsnative rows=404 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_jpeg_select family=dvifmgate rows=404 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=konjnd_jpeg_select rows=404 manifest_sha256=c43347557cea0688a3c4685706ad981919dd0471ed8f35e522856dcf19054686 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
9d80a0d762a31e8087cab6504a6a956dce05d983e705fb1e82e355c7b393d715  /var/tmp/restore-cuts/pairs/konjnd_jpeg_select.tsv
82412b590c3d37447aff633aed9281dd832817bc4a78ea51f894064369526531  /var/tmp/restore-cuts/raw/konjnd_jpeg_select.csv
2198d7c9f85a48abbdc1cc5d552a76df6c85cdeb4c08c29fae40346782143b00  /var/tmp/restore-cuts/raw/konjnd_jpeg_select.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/konjnd_jpeg_select.csv.manifest.json
RESTORE_END set=konjnd_jpeg_select utc=2026-09-25T09:40:14Z
```
Log sha256: `1c0598171f724ae3320cd5f9c9bb47b454e841aacdcc11d1717da1e4297be95b`.

### csiq

UTC 2026-09-25T09:40:14Z-2026-09-25T10:33:06Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh csiq`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/csiq.log`:

```text
RESTORE_START set=csiq utc=2026-09-25T10:32:54Z
Loaded 865 pairs from pairs-tsv
scored 865/865 pairs in 11.2s (0 failed)
Wrote 865 rows × 1825 features to /var/tmp/restore-cuts/raw/csiq.csv
WALL_SECONDS=11.30 MAXRSS_KB=300476
RESTORE_CHECK set=csiq family=mapdev rows=865 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=csiq family=z1max rows=865 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=csiq family=gmsnative rows=865 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=csiq family=dvifmgate rows=865 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=csiq rows=865 manifest_sha256=1a4a23530258eec6f4a45bf0b695aacbe326ea8bf3451c2042bd4c052016c726 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
49400c603b7dd22ef559f935ed506e6c283bd3a1d9c16de2f162799436d9f9b1  /var/tmp/restore-cuts/pairs/csiq.tsv
f239b1109c1415b99f73fe720fec41805d60a4d56a04662bcaaa90e55b3ccc4a  /var/tmp/restore-cuts/raw/csiq.csv
c09033b5e34504bcd7ccdf2e567b8bdffdb5db870d888350e7e9bfb772c1fae3  /var/tmp/restore-cuts/raw/csiq.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/csiq.csv.manifest.json
RESTORE_END set=csiq utc=2026-09-25T10:33:06Z
```
Log sha256: `d295597bdcd17eb48721b5928357bca006721d8ef6b383a56832fd034ddb80fa`.

### cid22_b

UTC 2026-09-25T10:33:06Z-2026-09-25T10:34:17Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh cid22_b`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/cid22_b.log`:

```text
RESTORE_START set=cid22_b utc=2026-09-25T10:33:49Z
Loaded 2100 pairs from pairs-tsv
scored 2100/2100 pairs in 27.1s (0 failed)
Wrote 2100 rows × 1825 features to /var/tmp/restore-cuts/raw/cid22_b.csv
WALL_SECONDS=27.33 MAXRSS_KB=302956
RESTORE_CHECK set=cid22_b family=mapdev rows=2100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_b family=z1max rows=2100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_b family=gmsnative rows=2100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_b family=dvifmgate rows=2100 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=cid22_b rows=2100 manifest_sha256=433c2fb2d6979edc9f6c4373c947c09725393ca6155787ddf838010c71df9cb5 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
6be3d64d36720c4d5b0953a2839345d70b6e91f094f414e0f9a709d772d990e6  /var/tmp/restore-cuts/pairs/cid22_b.tsv
20ce07022741594151d3bc5302804b2da34bd98431da9a2b96df827822071113  /var/tmp/restore-cuts/raw/cid22_b.csv
f4a32541f1cd556bb5f3463781c53c06f9e6598c73840946eb49a79df47c92b9  /var/tmp/restore-cuts/raw/cid22_b.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/cid22_b.csv.manifest.json
RESTORE_END set=cid22_b utc=2026-09-25T10:34:17Z
```
Log sha256: `143b91a7cf0f59e982679ef1d93b54d2d21b43a9fc6a2b268b9973cd38800520`.

### cid22_a25

UTC 2026-09-25T10:34:17Z-2026-09-25T10:35:13Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh cid22_a25`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/cid22_a25.log`:

```text
RESTORE_START set=cid22_a25 utc=2026-09-25T10:34:43Z
Loaded 2192 pairs from pairs-tsv
scored 2192/2192 pairs in 28.6s (0 failed)
Wrote 2192 rows × 1825 features to /var/tmp/restore-cuts/raw/cid22_a25.csv
WALL_SECONDS=28.81 MAXRSS_KB=303036
RESTORE_CHECK set=cid22_a25 family=mapdev rows=2192 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_a25 family=z1max rows=2192 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_a25 family=gmsnative rows=2192 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_a25 family=dvifmgate rows=2192 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=cid22_a25 rows=2192 manifest_sha256=a722b1e664f987b496474ddfa2248d827e1c49cf8cdef100c654b858d15c3caa feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
60d4ddf358a6099644e19022bbc5c811cd76be025a50f944e60acf26b6f2115e  /var/tmp/restore-cuts/pairs/cid22_a25.tsv
f7e9f24a28fa935aab4d9411d94be11d20798faf99a7f993eec8a0d8cd1a8842  /var/tmp/restore-cuts/raw/cid22_a25.csv
478d1859fc88aae61ff705f79d93dcb208953dcc4bbe67337c807b54bba82ed6  /var/tmp/restore-cuts/raw/cid22_a25.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/cid22_a25.csv.manifest.json
RESTORE_END set=cid22_a25 utc=2026-09-25T10:35:13Z
```
Log sha256: `910d8050d1b435ea969d777b9f8086bb1e028d174f6727eb7e617a0788a1676f`.

### tid2013

UTC 2026-09-25T10:35:13Z-2026-09-25T10:37:09Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh tid2013`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/tid2013.log`:

```text
RESTORE_START set=tid2013 utc=2026-09-25T10:36:38Z
Loaded 3000 pairs from pairs-tsv
scored 3000/3000 pairs in 29.1s (0 failed)
Wrote 3000 rows × 1825 features to /var/tmp/restore-cuts/raw/tid2013.csv
WALL_SECONDS=29.45 MAXRSS_KB=296884
RESTORE_CHECK set=tid2013 family=mapdev rows=3000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=tid2013 family=z1max rows=3000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=tid2013 family=gmsnative rows=3000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=tid2013 family=dvifmgate rows=3000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=tid2013 rows=3000 manifest_sha256=fb0088906fc7dc14e1afb228a6f1ddd62c325861b3ac5c1b1a3613ba0e92bbd1 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
319137c5036520f9661fa110d845ef09ab087e5140f48a2b02b1fac05834dab1  /var/tmp/restore-cuts/pairs/tid2013.tsv
5734785006e88a61f871131e876267f9b95977dafed56330c10096902eb5fbce  /var/tmp/restore-cuts/raw/tid2013.csv
2f68035937b7e6daf1198eb75ee6fb52eb092df93c0554874b6c3d6e3eb9f282  /var/tmp/restore-cuts/raw/tid2013.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/tid2013.csv.manifest.json
RESTORE_END set=tid2013 utc=2026-09-25T10:37:09Z
```
Log sha256: `88d3526e8c257b701d61a60246a625b6bf000d3c204f187036ba767687bc7dba`.

### kadid_terminal

UTC 2026-09-25T10:37:09Z-2026-09-25T10:37:47Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh kadid_terminal`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/kadid_terminal.log`:

```text
RESTORE_START set=kadid_terminal utc=2026-09-25T10:37:27Z
Loaded 1952 pairs from pairs-tsv
scored 1952/1952 pairs in 18.5s (0 failed)
Wrote 1952 rows × 1825 features to /var/tmp/restore-cuts/raw/kadid_terminal.csv
WALL_SECONDS=18.69 MAXRSS_KB=284508
RESTORE_CHECK set=kadid_terminal family=mapdev rows=1952 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_terminal family=z1max rows=1952 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_terminal family=gmsnative rows=1952 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_terminal family=dvifmgate rows=1952 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=kadid_terminal rows=1952 manifest_sha256=d55057ea706a5d76ffada7156e1256565c265986e6db53263c99d685c26cfb81 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
64c037d64ad8f20272f5c792ef561f2a167717da65e46063cbe80b6332ff5cbc  /var/tmp/restore-cuts/pairs/kadid_terminal.tsv
815a7ee7bafe7be40aceeecc683da37b1df6632555a4077a79864f43a529287d  /var/tmp/restore-cuts/raw/kadid_terminal.csv
8446bce9ba2b64f31cd153290597320e4b12ce5261dec7cc9e3ad052879082b8  /var/tmp/restore-cuts/raw/kadid_terminal.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/kadid_terminal.csv.manifest.json
RESTORE_END set=kadid_terminal utc=2026-09-25T10:37:47Z
```
Log sha256: `64f76f69a71ac63d12a650962673724c0bc49636c7f8fbce9b3085e4056c0ffe`.

### kadid_select

UTC 2026-09-25T10:37:47Z-2026-09-25T10:39:01Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh kadid_select`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/kadid_select.log`:

```text
RESTORE_START set=kadid_select utc=2026-09-25T10:38:30Z
Loaded 3050 pairs from pairs-tsv
scored 3050/3050 pairs in 28.7s (0 failed)
Wrote 3050 rows × 1825 features to /var/tmp/restore-cuts/raw/kadid_select.csv
WALL_SECONDS=29.02 MAXRSS_KB=289440
RESTORE_CHECK set=kadid_select family=mapdev rows=3050 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_select family=z1max rows=3050 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_select family=gmsnative rows=3050 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_select family=dvifmgate rows=3050 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=kadid_select rows=3050 manifest_sha256=733fdd9fc6d2deca83c2da942cf37fe37cf6d185f3ceca27623b3f35efd35c38 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
304a9498c4aed99b16473eafeec873738edce2fad70bcb50f964ce072d35285d  /var/tmp/restore-cuts/pairs/kadid_select.tsv
fccc4d72a9c5f4ee6af057bc1143014e1dff824e02c83aad594a2691ba95b461  /var/tmp/restore-cuts/raw/kadid_select.csv
a7e976f459255b09a9ab9191048a5ab47acda6a317871fac6c812cfa3c857e46  /var/tmp/restore-cuts/raw/kadid_select.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/kadid_select.csv.manifest.json
RESTORE_END set=kadid_select utc=2026-09-25T10:39:01Z
```
Log sha256: `b9ad7121f93720dd63fc1ef49bc8a695c66a213df923db97d17e1c9c05079b26`.

### kadid_train

UTC 2026-09-25T10:39:01Z-2026-09-25T10:40:20Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh kadid_train`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/kadid_train.log`:

```text
RESTORE_START set=kadid_train utc=2026-09-25T10:39:33Z
Loaded 4880 pairs from pairs-tsv
scored 4880/4880 pairs in 45.5s (0 failed)
Wrote 4880 rows × 1825 features to /var/tmp/restore-cuts/raw/kadid_train.csv
WALL_SECONDS=45.99 MAXRSS_KB=330420
RESTORE_CHECK set=kadid_train family=mapdev rows=4880 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_train family=z1max rows=4880 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_train family=gmsnative rows=4880 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=kadid_train family=dvifmgate rows=4880 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=kadid_train rows=4880 manifest_sha256=585ecc2518958119f42085d76466938f6cda5e1d3ec96c3673936e19d20265dc feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
88fdf3ed5bcd7f516a2babc30be8aa3c40569e4fcc4a46c86a5b5a89c6d86f21  /var/tmp/restore-cuts/pairs/kadid_train.tsv
27c3f945a0b1b1cea6690d758fdf115d791cb420959cb292319296f49a6411be  /var/tmp/restore-cuts/raw/kadid_train.csv
a735d0b9cd2f77b0e7b2f22ef75db698c385f0e1823f2ac3f3ba4b948968487a  /var/tmp/restore-cuts/raw/kadid_train.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/kadid_train.csv.manifest.json
RESTORE_END set=kadid_train utc=2026-09-25T10:40:20Z
```
Log sha256: `9689aead0a6b9bde109260439183208aa0ae307ea1344aa8aad64656ace3cd57`.

### konjnd_bpg_val

UTC 2026-09-25T10:40:20Z-2026-09-25T10:41:11Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh konjnd_bpg_val`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/konjnd_bpg_val.log`:

```text
RESTORE_START set=konjnd_bpg_val utc=2026-09-25T10:40:40Z
Loaded 2020 pairs from pairs-tsv
scored 2020/2020 pairs in 29.9s (0 failed)
Wrote 2020 rows × 1825 features to /var/tmp/restore-cuts/raw/konjnd_bpg_val.csv
WALL_SECONDS=30.15 MAXRSS_KB=361320
RESTORE_CHECK set=konjnd_bpg_val family=mapdev rows=2020 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_bpg_val family=z1max rows=2020 finite=1 unique=1 identity_violations=0 dead_columns=[1574]
RESTORE_CHECK set=konjnd_bpg_val family=gmsnative rows=2020 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_bpg_val family=dvifmgate rows=2020 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=konjnd_bpg_val rows=2020 manifest_sha256=614436013352ee6f784ccbc80b0df66e45dcc0806ec3844ed0db6ba81488070f feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
139446322bb1a4913c8b7352d16c9f03b73bf89d070884ba27a701df9c724690  /var/tmp/restore-cuts/pairs/konjnd_bpg_val.tsv
495ab1ef431feda6f1c9a02c279f8042d02ed4f3dfd9629be95cb5bf5166225c  /var/tmp/restore-cuts/raw/konjnd_bpg_val.csv
8408d8c2b80e0a1908ba799cbe6c6c20e792b9f23823d7672472e333fe5048a4  /var/tmp/restore-cuts/raw/konjnd_bpg_val.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/konjnd_bpg_val.csv.manifest.json
RESTORE_END set=konjnd_bpg_val utc=2026-09-25T10:41:11Z
```
Log sha256: `0bc0ad229a100bc86641d75339bf31caeb0f80d36ddd8cdba244f88deef2858b`.

### konjnd_bpg_train

UTC 2026-09-25T10:41:11Z-2026-09-25T10:44:11Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh konjnd_bpg_train`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/konjnd_bpg_train.log`:

```text
RESTORE_START set=konjnd_bpg_train utc=2026-09-25T10:42:09Z
Loaded 8060 pairs from pairs-tsv
scored 8060/8060 pairs in 118.5s (0 failed)
Wrote 8060 rows × 1825 features to /var/tmp/restore-cuts/raw/konjnd_bpg_train.csv
WALL_SECONDS=119.33 MAXRSS_KB=479672
RESTORE_CHECK set=konjnd_bpg_train family=mapdev rows=8060 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_bpg_train family=z1max rows=8060 finite=1 unique=1 identity_violations=0 dead_columns=[1574]
RESTORE_CHECK set=konjnd_bpg_train family=gmsnative rows=8060 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=konjnd_bpg_train family=dvifmgate rows=8060 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=konjnd_bpg_train rows=8060 manifest_sha256=25bec47bab50fa2403c6805b1b12a2d46b3d68e3468a9f2d6dc24d19958e6956 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
f3e1224595ca62da8ce9350863529dd4387fc0bf9252f5b4ef1d8bfc4b4397cb  /var/tmp/restore-cuts/pairs/konjnd_bpg_train.tsv
0714a3aa46ead9fef47767efcfc7df3974d1b677d7e883d5f57c2ab4443a9cf4  /var/tmp/restore-cuts/raw/konjnd_bpg_train.csv
89fd6a0779e3e5a2cde2cd2ea0233185ffdb6e78b2b219849d61e8a375e5e456  /var/tmp/restore-cuts/raw/konjnd_bpg_train.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/konjnd_bpg_train.csv.manifest.json
RESTORE_END set=konjnd_bpg_train utc=2026-09-25T10:44:11Z
```
Log sha256: `4570bf6bd8f7867177b48876f4942b5a3d890ffc6273036dd0135678563340f1`.

### mcljci

UTC 2026-09-25T10:44:11Z-2026-09-25T12:08:26Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh mcljci`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/mcljci.log`:

```text
RESTORE_START set=mcljci utc=2026-09-25T12:00:09Z
Loaded 5000 pairs from pairs-tsv
scored 5000/5000 pairs in 494.1s (0 failed)
Wrote 5000 rows × 1825 features to /var/tmp/restore-cuts/raw/mcljci.csv
WALL_SECONDS=494.60 MAXRSS_KB=1687632
RESTORE_CHECK set=mcljci family=mapdev rows=5000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=mcljci family=z1max rows=5000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=mcljci family=gmsnative rows=5000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=mcljci family=dvifmgate rows=5000 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=mcljci rows=5000 manifest_sha256=fa90f27767feb218a377ddfd6211cbf2ca6012acfcc0ec9a65d6bcca32bd5da7 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
713cd5db5b78a38b87fa8dcf47585f7edb61b8946fcd7dba3e0fc30633920527  /var/tmp/restore-cuts/pairs/mcljci.tsv
9dc0619563a263f162741780647c2e246ee5b0dbd097463f64d8e1edee8bb50f  /var/tmp/restore-cuts/raw/mcljci.csv
13eb73b0021758a92f4ce36af22a1bd1eaa7d8738a2291442562a87db837c91c  /var/tmp/restore-cuts/raw/mcljci.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/mcljci.csv.manifest.json
RESTORE_END set=mcljci utc=2026-09-25T12:08:26Z
```
Log sha256: `3bf259179b250b63c56d9e00f9364b99084aaa546c56e1d8236f6e499090283b`.

(continued in restore-cuts_extraction_log_2026-09-25b.md: cid22_train, safesyn)
