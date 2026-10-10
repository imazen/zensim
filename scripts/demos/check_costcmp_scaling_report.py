"""Exercise report refusal against complete, immutable COSTCMP evidence."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import costcmp_scaling_report as reporter
import speedq_run as owner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--budget-grid', action='store_true')
    ap.add_argument('--rss-only', action='store_true')
    ap.add_argument('--floor-arm', action='store_true')
    args = ap.parse_args()
    assert args.budget_grid or not args.floor_arm
    args.dest.mkdir(parents=True, exist_ok=False)
    grid = 'budget' if args.budget_grid else 'scaling'
    first = args.raw_dir / (grid+'/timing/'+('v4x-t8-1024x1024' if args.budget_grid else 'v4x-t1-1024x1024'))
    reads = Path.read_text
    controls = [
        ('before_binary_pin', first / 'header.json',
         lambda value: value['worker_binary_sha256'].__setitem__('by_v2fy_r5_before', '0'*64)),
        ('shared_round_alignment', first / 'COMPLETE.json',
         lambda value: value['retained_indices'].reverse()),
        ('complete_requested_grid', args.raw_dir / grid / ('parity-v2' if args.floor_arm else 'parity') / 'PREFLIGHT_PASS.json',
         lambda value: value['records'].pop()),
        ('measured_rss_log', args.raw_dir / (grid+'/rss/'+('v4x-t8-1024x1024-by_v2fy_r5_b64.log' if args.budget_grid else 'v4x-t1-1024x1024-by_v2fy_r5.log')), None),
        ('exact_statistics_replay', None, None),
    ]
    if args.floor_arm:
        controls.append(('floor_binary_pin', first / 'header.json',
                         lambda value: value['worker_binary_sha256'].__setitem__('by_v2fy_r5_floor3', '0'*64)))
    if args.rss_only:
        assert args.budget_grid and not args.floor_arm
        rss = args.raw_dir/'budget/rss/v4x-t8-1024x1024-by_v2fy_r5_b64'
        controls = [
            ('complete_requested_grid', args.raw_dir/'budget/parity/PREFLIGHT_PASS.json', lambda v:v['records'].pop()),
            ('rss_binary_pin', rss.with_suffix('.json'), lambda v:v.update(binary_sha256='0'*64)),
            ('rss_authorization_scope', rss.with_suffix('.json'), lambda v:v.update(rss_policy='timing under load')),
            ('measured_rss_log', rss.with_suffix('.log'), None),
            ('actual_byte_accounting', args.raw_dir/'provenance/byte-accounting.log', None),
        ]
    refused = []
    for label, target, mutate in controls:
        out = SimpleNamespace(raw_dir=args.raw_dir, out_json=args.dest / (label+'.json'),
                              out_md=args.dest / (label+'.md'), budget_grid=args.budget_grid,rss_only=args.rss_only,
                              floor_arm=args.floor_arm)
        def altered(path, *argv, **kwargs):
            text = reads(path, *argv, **kwargs)
            if path == target:
                if mutate is None:
                    if label == 'actual_byte_accounting':
                        return text.replace('"planes":11', '"planes":10')
                    return '\n'.join('Maximum resident set size (kbytes): 1' if
                                     'Maximum resident set size (kbytes)' in line else line
                                     for line in text.splitlines())
                value = json.loads(text)
                mutate(value)
                return json.dumps(value)
            return text
        context = (patch.object(reporter.subprocess, 'check_output', return_value='[]')
                   if target is None else patch.object(Path, 'read_text', altered))
        with context:
            try:
                reporter.report(out)
            except AssertionError:
                refused.append(label)
            else:
                raise RuntimeError(f'report admitted corrupted evidence: {label}')
        assert not out.out_json.exists() and not out.out_md.exists()
    owner.write(args.dest / 'PASS.json', dict(status='PASS', refused=refused,
                mutation_scope='in-memory reads; original evidence files unchanged'))
    print(json.dumps(dict(status='PASS', refused=refused)), flush=True)


if __name__ == '__main__':
    main()
