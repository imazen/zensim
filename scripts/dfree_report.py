#!/usr/bin/env python3
"""Print one bake_verdict --full-json's G-ADDR + rank line for the D+free lane.
Reads only; computes no statistic (every number is what bake_verdict produced)."""
import json, sys
d = json.load(open(sys.argv[1]))
dl = d.get('dial', {}); a = dl.get('addressability', {})
if a:
    print('  ', a.get('headline'))
    print('   reach %.3f min %.3f max %.3f p5 %.3f p95 %.3f DR %.3f mono %.4f tied %.4f' % (
        dl['reach'], dl['min'], dl['max'], dl['p5'], dl['p95'],
        dl['dynamic_range'], dl['mono_pct'], dl['tied_pct']))
    m = a.get('measured', {})
    if m.get('negtail'):
        n = m['negtail']
        print('   negtail min %.3f p1 %.3f frac<0 %.4f' % (n['min'], n['p1'], n['frac_below_zero']))
    if m.get('identity'):
        i = m['identity']
        print('   identity min/med/max %.4f / %.4f / %.4f  outside-band %d  above-identity %d/%d' % (
            i['dial_min'], i['dial_median'], i['dial_max'],
            i['n_outside_band'], i['n_above_identity'], i['n_grid_cells_total']))
    print('   fails:', [c['id'] for c in a.get('checks', []) if c['state'] == 'fail'],
          '| not measured:', [c['id'] for c in a.get('checks', []) if c['state'] == 'not_measured'])
r = d.get('rank', {})
ks = [k for k in ('cid22','konjnd','aic3','aic4','tid','kadid','csiq','live','sdr25',
                  'imazen26','nonphoto','hfnlproxy') if k in r]
print('   ' + '  '.join('%s %.5f' % (k, r[k]['srocc_signed']) for k in ks))
if 'product_composite' in d: print('   product_composite %.10f' % d['product_composite'])
