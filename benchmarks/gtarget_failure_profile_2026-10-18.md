# Scalar target failure profile

Post-run descriptive diagnostic; no model, controller, gate or verdict changes. Among910 panel-specific TRAIN-curve K3 targets,187 have absolute error>1. Of those,43 have observed within1 witnesses only at a native knob endpoint;19 include repeated encoded bitstreams among their actual probes;12 meet both descriptions. These are correlated panel cases, not187 independent source images.

The shared WebP endpoint example is one measured mechanism, not a complete explanation. Most misses do not have either of these two recorded patterns. All candidate and peer budget gates remain FAIL.

| Dial | Codec | Reference | K3 targets | Error>1 | Only endpoint witnesses | Repeated bitstreams | Both |
|---|---|---|---:|---:|---:|---:|---:|
| zensim | jpeg | diff-frag-820 | 54 | 12 | 5 | 0 | 0 |
| zensim | jpeg | w-c | 50 | 13 | 4 | 0 | 0 |
| zensim | webp | diff-frag-820 | 51 | 14 | 3 | 1 | 1 |
| zensim | webp | w-c | 53 | 9 | 3 | 1 | 1 |
| zensim | avif | diff-frag-820 | 52 | 9 | 2 | 1 | 1 |
| zensim | avif | w-c | 50 | 11 | 1 | 2 | 0 |
| ssim2 | jpeg | diff-frag-820 | 53 | 7 | 3 | 0 | 0 |
| ssim2 | jpeg | w-c | 53 | 8 | 3 | 0 | 0 |
| ssim2 | webp | diff-frag-820 | 51 | 10 | 3 | 2 | 2 |
| ssim2 | webp | w-c | 51 | 11 | 3 | 2 | 2 |
| ssim2 | avif | diff-frag-820 | 54 | 16 | 2 | 4 | 1 |
| ssim2 | avif | w-c | 47 | 12 | 2 | 2 | 1 |
| butter3 | jpeg | diff-frag-820 | 50 | 8 | 2 | 0 | 0 |
| butter3 | jpeg | w-c | 50 | 9 | 2 | 0 | 0 |
| butter3 | webp | diff-frag-820 | 51 | 10 | 1 | 0 | 0 |
| butter3 | webp | w-c | 49 | 9 | 1 | 0 | 0 |
| butter3 | avif | diff-frag-820 | 43 | 9 | 2 | 2 | 2 |
| butter3 | avif | w-c | 48 | 10 | 1 | 2 | 1 |

An endpoint case has at least one observed witness within1, and every such witness is at the minimum or maximum sampled native knob. This does not prove an unsampled interior knob cannot hit the request. Repeated bitstreams mean fewer unique encoded hashes than actual full encodes; that is evidence of repeated output, not a causal assignment. These categories may overlap.

[Exact derived counts](gtarget_failure_profile_2026-10-18.json), SHA256 `fe1af8da1c43a93c5f4dddd12bcbea57e9d4fbd7f9253b28fd48d67a82d8325a`. Inputs are the immutable `bounds.jsonl` and `measurements.jsonl` files in [the evidence archive](gtarget_2026-10-18.pointer.md).

Reproduction (metadata only; no new encodes or metric scores):

```python
from pathlib import Path
import json
root = Path("/mnt/v/output/zensim/gtarget-2026-10-18-svt-corrected")
for dial, base in [("zensim", root/"scalar-v1"), ("ssim2", root/"peers-v1/ssim2"),
                   ("butter3", root/"peers-v1/butter")]:
    for codec in ("jpeg", "webp", "avif"):
        for model in ("diff-frag-820", "w-c"):
            p = base/codec/model/"eval"
            data = [json.loads(s) for s in (p/"measurements.jsonl").read_text().splitlines()]
            bounds = [json.loads(s)["bound"] for s in (p/"bounds.jsonl").read_text().splitlines()]
            selected = [r for r in data if r["policy"] == "train_curve" and r["budget"] == 3]
            misses = [r for r in selected if abs(r["signed_error"]) > 1]
            endpoint = repeat = both = 0
            for r in misses:
                b = [b for b in bounds if b["origin"] == r["origin"]]
                w = [b for b in b if abs(b["score"]-r["target"]) <= 1]
                assert w
                edges = {min(b["knob"] for b in b), max(b["knob"] for b in b)}
                e = all(b["knob"] in edges for b in w)
                d = len({v["encoded_sha256"] for v in r["native_work"]}) < r["full_encodes"]
                endpoint += e; repeat += d; both += e and d
            print(dial, codec, model, len(selected), len(misses), endpoint, repeat, both)
```
