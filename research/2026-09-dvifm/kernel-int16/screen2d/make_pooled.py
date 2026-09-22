#!/usr/bin/env python3
"""Build the pooled-domain pair TSVs by concatenating the source TSVs in the
exact order the caches will be concatenated."""
import csv

BASE = '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/pairs/'


def rows(p):
    with open(BASE + p) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(name, srcs):
    with open(BASE + name, 'w') as out:
        out.write('ref_path\tdist_path\thuman_score\n')
        n = 0
        for s in srcs:
            for r in rows(s):
                out.write('%s\t%s\t%s\n' % (r['ref_path'], r['dist_path'],
                                            r['human_score']))
                n += 1
    print(name, n)


write('tidkadid.tsv', ['tid_jp2kjpeg.tsv', 'kadid_train.tsv'])
write('pooled_cid22a_tidkadid.tsv',
      ['cid22a.tsv', 'tid_jp2kjpeg.tsv', 'kadid_train.tsv'])
