#!/usr/bin/env python3
"""transplant lane: emit the extraction-input TSVs.

Every TSV is `ref_path, dist_path, human_score, row_id` — `row_id` is the
pair's index in pairs/pairs_core.tsv, carried through the extractor CSV as
an extra target so assembly joins are exact (no basename/score matching).

  extract/pilot.tsv      — seeded 6k-row stratum across all cohorts (c0 pilot)
  extract/v2fresh.tsv    — cohort=v2fresh rows (full-986 + transplant pass)
  extract/rest.tsv       — cohort v1 + v2reused rows (transplant-only pass)
  extract/all.tsv        — every row (Cb / Cr dvifm-only passes)

Dev legs reuse the verdict lane's pair TSVs verbatim — verified positional
match against the frozen *_development.parquet rows.
"""
import csv, os, random, sys

V2 = '/mnt/v/output/zensim/joint-core-v2'
OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20/extract'
os.makedirs(OUT, exist_ok=True)

rows = []
with open(f'{V2}/pairs/pairs_core.tsv') as f:
    for i, r in enumerate(csv.DictReader(f, delimiter='\t')):
        r['row_id'] = i
        rows.append(r)
print('pairs_core rows:', len(rows))

def emit(name, sel):
    with open(f'{OUT}/{name}.tsv', 'w') as f:
        f.write('ref_path\tdist_path\thuman_score\trow_id\n')
        for r in sel:
            f.write(f"{r['ref_path']}\t{r['dist_path']}\t{r['human_score']}\t{r['row_id']}\n")
    print(name, len(sel))

# pilot: seeded, stratified by (cohort, leg) — 6k rows for the c0 histograms
rng = random.Random(770031)
by = {}
for r in rows:
    by.setdefault((r['cohort'], r['leg']), []).append(r)
pilot = []
want = 6000
tot = len(rows)
for k, g in sorted(by.items()):
    n = max(60, round(want * len(g) / tot)) if len(g) >= 60 else len(g)
    pilot.extend(rng.sample(g, min(n, len(g))))
# HDR rows are PQ-regime: the SDR research path cannot plan their
# PQ->sRGB conversion (cms_lite needs a peak). They are not a train leg —
# their appended cols are zero-filled at table assembly instead.
def sdr(r):
    return r['leg'] != 'hdr'


emit('pilot', [r for r in pilot if sdr(r)])
emit('v2fresh', [r for r in rows if r['cohort'] == 'v2fresh' and sdr(r)])
emit('rest', [r for r in rows if r['cohort'] != 'v2fresh' and sdr(r)])
emit('all', [r for r in rows if sdr(r)])
# era-parity audit: seeded v1-cohort rows re-extracted full-986, then
# compared bitwise against the canonical f0..f943 sources.
rest = [r for r in rows if r['cohort'] != 'v2fresh' and sdr(r)]
emit('verify', rng.sample(rest, min(1024, len(rest))))
