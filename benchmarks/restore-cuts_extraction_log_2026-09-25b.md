# restore-cuts extraction log, part b

### cid22_train

UTC 2026-09-25T12:08:26Z-2026-09-25T12:15:59Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh cid22_train`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/cid22_train.log`:

```text
RESTORE_START set=cid22_train utc=2026-09-25T12:12:06Z
Loaded 17611 pairs from pairs-tsv
scored 17611/17611 pairs in 225.8s (0 failed)
Wrote 17611 rows × 1825 features to /var/tmp/restore-cuts/raw/cid22_train.csv
WALL_SECONDS=227.69 MAXRSS_KB=597192
RESTORE_CHECK set=cid22_train family=mapdev rows=17611 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_train family=z1max rows=17611 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_train family=gmsnative rows=17611 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=cid22_train family=dvifmgate rows=17611 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=cid22_train rows=17611 manifest_sha256=1cbe99f3e5a5da1d3fbb1c037a8f28b56497043b46a5acc2a6e3216ad7b140cf feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
0ec4576a41415424b99dcacd0f45ed5a543a72d32410f4e88f289b392da3bc2b  /var/tmp/restore-cuts/pairs/cid22_train.tsv
7ed361da833c281ebefcb1a7a809499483c7a90c0cc6447e1c182457601aedc0  /var/tmp/restore-cuts/raw/cid22_train.csv
fb23e365b91c4bfa947f21756d62e59c39a3418af1e2c283f22855d2877d91ad  /var/tmp/restore-cuts/raw/cid22_train.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/cid22_train.csv.manifest.json
RESTORE_END set=cid22_train utc=2026-09-25T12:15:59Z
```
Log sha256: `c2ae7d5cc523c4b56190814538c013e1d64f457d39e32cecfed222eb17f1ad11`.

### safesyn

UTC 2026-09-25T12:15:59Z-2026-09-25T14:11:47Z; cwd `/home/lilith/work/zen/zensim--restore-cuts`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh safesyn`; exit 0.

Exact result lines from `/var/tmp/restore-cuts/logs/safesyn.log`:

```text
RESTORE_START set=safesyn utc=2026-09-25T12:37:15Z
Loaded 196086 pairs from pairs-tsv
scored 196086/196086 pairs in 5599.7s (0 failed)
Wrote 196086 rows × 1825 features to /var/tmp/restore-cuts/raw/safesyn.csv
WALL_SECONDS=5618.79 MAXRSS_KB=4340404
RESTORE_CHECK set=safesyn family=mapdev rows=196086 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=safesyn family=z1max rows=196086 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=safesyn family=gmsnative rows=196086 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_CHECK set=safesyn family=dvifmgate rows=196086 finite=1 unique=1 identity_violations=0 dead_columns=[]
RESTORE_WRITTEN set=safesyn rows=196086 manifest_sha256=8e257ec13c1883e27e949ef965feb6ef5694bf630314424cbec187b664c37c86 feature_set_id=basic+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/unknown#eb217aec
e3761f42784a21b8f42d6371831571c1f14cd867058f4d0c475f4cdd334f69d4  /var/tmp/restore-cuts/pairs/safesyn.tsv
9c60bfb43facd32a4ea1d065020e724a202c29de6c4da20b34b8b0755993182c  /var/tmp/restore-cuts/raw/safesyn.csv
701ddb7d16134d62aebed91b5e38922bc252797b448aa97ebe1dfcea63e934f8  /var/tmp/restore-cuts/raw/safesyn.audit.jsonl
82b1526b04108eceb8a134cef4c4976934285bad33f33a7ccbf34865afd72f31  /var/tmp/restore-cuts/raw/safesyn.csv.manifest.json
RESTORE_END set=safesyn utc=2026-09-25T14:11:47Z
```
Log sha256: `e4af97a9c40cc41f17ad22c443fbb46fabedeed6e3099496cda7cbb14de94730`.

