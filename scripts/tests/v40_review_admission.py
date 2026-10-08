"""Independent V40 actual-binary admission probes (synthetic payloads).

Oracle: strace of every open/openat/openat2 by the trainer process tree. A "payload open" is any
non-O_PATH open of a group table path (exact spelling or resolved), or of any non-key *.parquet under
the fixture. Every negative case must exit nonzero, open zero payloads, write no model and fail for its
intended reason; every family has a positive control that does reach payload access.
Real inputs used: only label-free keys/manifests/dispositions already approved for TRAIN.
"""
import argparse, copy, hashlib, json, os, re, shutil, subprocess, sys
from unittest.mock import patch
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--dest', type=Path, required=True)
parser.add_argument('--family', choices=('e29', 'e32', 'd1'), action='append')
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
OUT = args.dest; OUT.mkdir(parents=True, exist_ok=False)
BIN = args.binary
UPIQ = Path('/mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07')
E29_SOURCE = Path('/mnt/v/output/zensim/e29-2026-10-07/v2e29/wide/main/real')
DISPOSITION = REPO / 'benchmarks/e31_upiq_owner_disposition_2026-10-07.json'
sys.path[:0] = [str(REPO / 'scripts/rev4_featpot'), str(REPO / 'scripts/tests'), str(REPO / 'scripts')]
from test_e32_palette_training import PaletteTraining
from test_shippath2_admission import RecipeAdmissionTests
import e32_palette as palette
import e31_training as upiq
import v2_common as common
import v2_d1_prepare as prepare
import v2_human_role as roles
import v2_teacher as teacher
from e21_cheap_recipe import columns


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert BIN.is_file()
results = []


def run(family, name, argv, payloads, expect, reason=None, keys=()):
    """expect: 'refuse' | 'reach' | 'train'."""
    d = OUT / family / name; d.mkdir(parents=True)
    model = d / 'model.bin'
    argv = [str(a) for a in argv]
    argv = [model if a == '@OUT' else a for a in argv]
    argv = [str(a) for a in argv]
    trace = d / 'open.strace'
    p = subprocess.run(['strace', '-qq', '-f', '-e', 'trace=open,openat,openat2', '-o', str(trace), *argv],
                       capture_output=True, text=True)
    (d / 'stdout.log').write_text(p.stdout); (d / 'stderr.log').write_text(p.stderr)
    (d / 'argv.json').write_text(json.dumps(argv, indent=1) + '\n')
    lines = [l for l in trace.read_text().splitlines() if 'O_PATH' not in l and 'O_DIRECTORY' not in l and ' = -1 ' not in l]
    spell = {str(x) for x in payloads} | {str(Path(x).resolve()) for x in payloads if Path(x).exists()}
    exact = sum(any(f'"{s}"' in l for s in spell) for l in lines)
    anyp = sum(1 for l in lines for m in re.findall(r'"([^"]+\.parquet)"', l) if not m.endswith('.keys.parquet'))
    keyo = sum(any(f'"{k}"' in l for k in keys) for l in lines)
    r = dict(family=family, case=name, rc=p.returncode, payload_opens=exact, any_parquet_payload_opens=anyp,
             key_opens=keyo, model_written=model.exists(), reason=p.stderr.strip().splitlines()[-1][:220] if p.stderr.strip() else '')
    if expect == 'refuse':
        ok = p.returncode != 0 and exact == 0 and anyp == 0 and not model.exists() and (reason is None or reason in p.stderr)
    elif expect == 'reach':
        ok = exact > 0
    else:  # train
        ok = p.returncode == 0 and exact > 0 and model.exists()
    r.update(expect=expect, expected_reason=reason, ok=ok)
    results.append(r); print(json.dumps(r), flush=True)
    return r


# ---------------------------------------------------------------- E29 (E29R2 reviewer shapes)
def e29_family():
    fam = 'e29'
    cases = [('hb4-VAL', 'hb4', 'val', None, BIN, 'refuse'),
             ('hc4-VAL', 'hc4', 'val', None, BIN, 'refuse'), ('hb4-TRAIN-positive', 'hb4', 'train', None, BIN, 'reach'),
             ('hc4-TRAIN-positive', 'hc4', 'train', None, BIN, 'reach'), ('hc4-other-CLI', 'hc4', 'train', 'cli', BIN, 'refuse'),
             ('hc4-other-group', 'hc4', 'train', 'group', BIN, 'refuse'), ('hc4-traversal', 'hc4', 'train', 'traversal', BIN, 'refuse'),
             ('hc4-pair-digest', 'hc4', 'train', 'digest', BIN, 'refuse'), ('hc4-no-CLI', 'hc4', 'train', 'missing', BIN, 'refuse'),
             ('hc4-multiple-CLI', 'hc4', 'train', 'multiple', BIN, 'refuse'), ('hb4-pair-CLI', 'hb4', 'train', 'extra', BIN, 'refuse'),
             ('hb4-row-digest', 'hb4', 'train', 'rowdigest', BIN, 'refuse'), ('hb4-duplicate-IDs', 'hb4', 'train', 'duplicate', BIN, 'refuse'),
             ('hb4-no-agreement', 'hb4', 'train', 'agree', BIN, 'refuse'), ('hb4-missing-source', 'hb4', 'train', 'source', BIN, 'refuse'),
             ('hb4-dev-weight', 'hb4', 'train', 'devweight', BIN, 'refuse'), ('hb4-withinref', 'hb4', 'train', 'withinref', BIN, 'refuse')]
    for tag, arm, role, mutation, binary, expect in cases:
        dest = OUT / fam / ('fixture-' + tag); dest.mkdir(parents=True)
        table = dest / 'native.parquet'; table.write_bytes(b'INDEPENDENT V40 REVIEW SYNTHETIC BAD PARQUET FOOTER')
        pairs = dest / 'pairs.json'; pairs.write_text('[[0,1]]\n')
        alternate = dest / 'other.json'; alternate.write_text('[[0,2]]\n')
        keys = pa.table({'row_id': pa.array(list(range(7390)), type=pa.int64()), 'role': [role] * 7390,
                         'agree': [mutation != 'agree'] * 7390,
                         'ref_basename': ['synthetic://reference/' + str(i % 17) for i in range(7390)]})
        if mutation == 'duplicate': keys = keys.set_column(0, 'row_id', pa.array([0] * 7390, type=pa.int64()))
        kp = table.with_suffix('.keys.parquet'); pq.write_table(keys, kp, compression='zstd')
        d = json.loads((E29_SOURCE / f'hdr_{arm}.parquet.manifest.json').read_text())
        d.update(table_sha256=sha(table), keys_sha256=sha(kp), row_keys_sha256=teacher.row_keys_sha(keys))
        if arm == 'hc4': d.update(pair_list=pairs.name, pair_list_sha256=sha(pairs))
        if mutation == 'traversal': d['pair_list'] = '../pairs.json'
        if mutation == 'digest': d['pair_list_sha256'] = '0' * 64
        if mutation == 'rowdigest': d['row_keys_sha256'] = '0' * 64
        if mutation == 'source': d.pop('source_keys_sha256', None)
        Path(str(table) + '.manifest.json').write_text(json.dumps(d, indent=2) + '\n')
        keep = dest / 'keep.txt'; keep.write_text('\n'.join(map(str, columns('by_v2fy'))) + '\n')
        tw, vw, mode = ('0', '1', 'rank') if mutation == 'devweight' else ('1', '0', 'withinref,rank' if mutation == 'withinref' else 'rank')
        argv = [binary, '--group', f'hdr:{table}:{tw}:{vw}:{mode}', '--hdr-consensus-research', '--nonneg-distance',
                '--target-column', 'human_score', '--target-scale', '1', '--keep-features', keep, '--max-features', '1825',
                '--out', '@OUT', '--no-auto-eval', '--epochs', '1', '--pairs-per-epoch', '1']
        if (arm == 'hc4' and mutation != 'missing') or mutation == 'extra':
            group = 'wrong' if mutation == 'group' else 'hdr'
            pth = alternate if mutation == 'cli' else pairs
            argv += ['--rank-pair-list', f'{group}:{pth}', '--no-sample-coverage']
            if mutation == 'multiple': argv += ['--rank-pair-list', f'hdr:{alternate}']
        run(fam, tag, argv, [table], expect, keys=[kp])


# ---------------------------------------------------------------- E32 (E32 reviewer shapes)
def e32_family():
    fam = 'e32'
    fixture = PaletteTraining(); fixture.setUp()
    try:
        payload = fixture.path; kp = teacher.key_path(payload); original_keys = kp.read_bytes(); base = copy.deepcopy(fixture.d)
        cases = [('valid-positive', None, 'train', None), ('VAL-keys', 'val', 'refuse', None),
                 ('development-trained', 'development', 'refuse', None), ('false-row-key-pin', 'rowpin', 'refuse', 'ordered observation'),
                 ('missing-pair-identity', 'missing', 'refuse', 'missing observation identity'), ('wrong-producer-pin', 'producer', 'refuse', None),
                 ('reversed-cli-IDs-canonicalized', 'ids', 'train', None), ('subset-IDs', 'subset', 'refuse', None),
                 ('aic-member', 'aic', 'refuse', 'unapproved/AIC observation population member'),
                 ('row-count', 'rows', 'refuse', 'ordered observation')]
        for tag, mutation, expect, reason in cases:
            d = copy.deepcopy(base); kp.write_bytes(original_keys)
            keys = pq.read_table(kp)
            if mutation == 'val':
                keys = keys.append_column('role', pa.array(['val'] * len(keys)))
            elif mutation == 'missing':
                keys = pa.table({'member_set': ['safesyn']}); d['rows'] = 1
            elif mutation == 'aic':
                ms = keys.column('member_set').to_pylist(); ms[0] = 'aic3'
                keys = keys.set_column(keys.schema.get_field_index('member_set'), 'member_set', pa.array(ms))
            if mutation in ('val', 'missing', 'aic'):
                pq.write_table(keys, kp, compression='zstd')
                d['keys_sha256'] = sha(kp); d['row_keys_sha256'] = teacher.row_keys_sha(pq.read_table(kp))
            if mutation == 'development': d['research_palette']['role'] = 'TRAIN-oracle-development'
            if mutation == 'rowpin': d['row_keys_sha256'] = '0' * 64
            if mutation == 'producer': d['research_palette']['producer_binary_sha256'] = '0' * 64
            if mutation == 'rows': d['rows'] = len(keys) + 1
            fixture.save(d)
            ids = OUT / fam / f'{tag}.keep.txt'; ids.parent.mkdir(parents=True, exist_ok=True)
            ids.write_text('\n'.join(map(str, reversed(palette.ARM_IDS) if mutation == 'ids' else palette.ARM_IDS[:-1] if mutation == 'subset' else palette.ARM_IDS)) + '\n')
            argv = [BIN, '--group', f'synthetic:{payload}:1:0:withinref,both', '--target-column', 'human_score', '--target-scale', '1',
                    '--hidden', '128', '--epochs', '2', '--pairs-per-epoch', '128', '--init-seed', '1101', '--sample-seed', '101',
                    '--pair-sampling', 'uniform', '--max-features', '1867', '--keep-features', ids, '--mse-weight', '1',
                    '--early-stop-patience', '0', '--out-dtype', 'f32', '--no-auto-eval', '--nonneg-distance', '--out', '@OUT']
            run(fam, tag, argv, [payload], expect, reason, keys=[kp])
    finally:
        fixture.doCleanups()


# ---------------------------------------------------------------- D1 seven-group fixture (+ E31 / E29 late)
def d1_fixture(dense=False):
    f = RecipeAdmissionTests()
    def densify(path):
        table = pq.read_table(path)
        columns = {n: table[n] for n in table.column_names if not n.startswith('f') or not n[1:].isdigit()}
        for index in range(1853):
            name = f'f{index}'
            columns[name] = table[name] if name in table.column_names else pa.array([float('nan')] * len(table), type=pa.float32())
        pq.write_table(pa.table(columns), path, compression='zstd')
    # Keep the positive training assertion. The metadata-test owner's sparse
    # f0/f719 fixture is replaced by a loadable, contiguous synthetic table
    # before its receipts are admitted; no negative expectations are changed.
    if dense:
        original_table = f.table
        def table(stem, frame):
            record = original_table(stem, frame)
            path = f.source / record['rel']
            densify(path)
            record['sha256'] = sha(path)
            return record
        f.table = table
    f.setUp()
    if dense:
        pool = f.source / 'e15/coverage_pool.parquet'
        densify(pool)
        manifest = pool.with_name('coverage_pool.manifest.json')
        record = json.loads(manifest.read_text()); record['sha256'] = sha(pool)
        manifest.write_text(json.dumps(record))
        f.stack.enter_context(patch.object(teacher, 'POOL_SHA_REV5', sha(pool)))
    f.admit()
    approval = f.root / 'decision.json'
    f.json(approval, dict(schema='shippath-human-role-decision-v1', decision_id='SHIPPATH-human-production-role', state='approved',
                          decided_by='TEST', allowed_use='qualified-recipe-training', sources=list(roles.PRODUCTION_SOURCES),
                          ledger_commit=roles.LEDGER_COMMIT, source_receipt_sha256=common.sha(f.wide / 'receipt.json'),
                          source_frozen_sha256=common.sha(f.source / 'wide/frozen.json')))
    root = f.root / 'd1'
    prepare.prepare(f.out, f.source, f.bank, f.root / 'stage', root, approval, f.root / 'logical')
    coverage, _ = teacher.coverage_leg(0x98, f.root, admitted_root=root)
    base = root / 'wide/main/real'
    groups = [(n, base / f'{stem}.parquet', tw, vw, 'withinref,rank') for n, stem, tw, vw in [
        ('safesyn', 'safesyn_fit', 1, 0), ('safesyn_development', 'safesyn_dev', 0, 1), ('cid22', 'cid22_fit', 16, 0),
        ('cid22_development', 'cid22_dev', 0, 1), ('human', 'human_without_kadid_fit', 32, 0),
        ('human_development', 'human_without_kadid_dev', 0, 1)]]
    groups.append(('coverage', coverage, 1, 0, 'withinref,rank'))
    return f, root, base, groups


def d1_argv(groups, ids, width, flags=()):
    argv = [BIN, '--target-column', 'human_score', '--target-scale', '1', '--keep-features', ','.join(map(str, ids)),
            '--max-features', str(width), '--epochs', '2', '--pairs-per-epoch', '128', '--hidden', '128',
            '--early-stop-patience', '0', '--no-auto-eval', '--nonneg-distance', '--out', '@OUT', *flags]
    for g in groups: argv += ['--group', ':'.join(map(str, g))]
    return argv


def rebind(base, stem, d=None, keys=None):
    """Rewrite a group's manifest/keys and re-pin the receipt so ONLY the intended property changes."""
    sp = base / f'{stem}.parquet.manifest.json'; kp = base / f'{stem}.keys.parquet'
    d = json.loads(sp.read_text()) if d is None else d
    if keys is not None:
        pq.write_table(keys, kp, compression='zstd'); d['keys_sha256'] = sha(kp); d['row_keys_sha256'] = teacher.row_keys_sha(pq.read_table(kp))
    sp.write_text(json.dumps(d))
    rp = base / 'receipt.json'; r = json.loads(rp.read_text())
    for leg in r['legs'].values():
        for split in ('fit', 'dev', 'full'):
            if isinstance(leg.get(split), dict) and Path(leg[split].get('rel', '')).name == f'{stem}.parquet':
                leg[split]['manifest_sha256'] = sha(sp)
    rp.write_text(json.dumps(r))
    return d


def d1_family():
    fam = 'd1-e31'
    ids420 = upiq.columns('by_v2fy')

    def case(tag, mutate, expect, reason=None, e31=False, e29_late=False):
        f, root, base, groups = d1_fixture(dense=expect == 'train')
        try:
            flags = []
            extra_payloads = []
            if e31:
                native = f.root / 'upiq.parquet'; native.write_bytes(b'SYNTHETIC V40 REVIEW UPIQ PAYLOAD (invalid parquet)')
                Path(f'{native}.manifest.json').write_bytes((UPIQ / 'upiq380_fit.parquet.manifest.json').read_bytes())
                shutil.copyfile(UPIQ / 'upiq380_fit.keys.parquet', teacher.key_path(native))  # label-free TRAIN keys
                groups.append(('upiq380', native, '4.34410740924913', 0, 'rank')); extra_payloads.append(native)
                flags += ['--upiq-label-disposition', str(DISPOSITION)]
            if e29_late:
                hdr = f.root / 'hdr.parquet'; hdr.write_bytes(b'SYNTHETIC V40 REVIEW HDR PAYLOAD')
                k = pa.table({'row_id': pa.array(list(range(7390)), type=pa.int64()), 'role': ['val'] * 7390, 'agree': [True] * 7390,
                              'ref_basename': ['synthetic://reference/' + str(i % 17) for i in range(7390)]})
                kp = teacher.key_path(hdr); pq.write_table(k, kp, compression='zstd')
                dd = json.loads((E29_SOURCE / 'hdr_hb4.parquet.manifest.json').read_text())
                dd.update(table_sha256=sha(hdr), keys_sha256=sha(kp), row_keys_sha256=teacher.row_keys_sha(k))
                Path(f'{hdr}.manifest.json').write_text(json.dumps(dd))
                groups.append(('hdr', hdr, 1, 0, 'rank')); extra_payloads.append(hdr); flags += ['--hdr-consensus-research']
            ids, width = ids420, 1853
            out = mutate(f, root, base, groups, flags) if mutate else None
            if isinstance(out, tuple): ids, width = out
            payloads = [g[1] for g in groups] + extra_payloads
            run(fam, tag, d1_argv(groups, ids, width, flags), payloads, expect, reason)
        finally:
            f.doCleanups()

    H = 'human_without_kadid_fit'
    def m_role(field, value):
        def m(f, root, base, groups, flags):
            d = json.loads((base / f'{H}.parquet.manifest.json').read_text()); d[field] = value; rebind(base, H, d)
        return m
    def m_weights(index, tw, vw):
        def m(f, root, base, groups, flags):
            g = groups[index]; groups[index] = (g[0], g[1], tw, vw, g[4])
        return m
    def m_keys(fn, keep_digest=False):
        def m(f, root, base, groups, flags):
            kp = base / f'{H}.keys.parquet'; k = fn(pq.read_table(kp))
            if keep_digest:
                d = json.loads((base / f'{H}.parquet.manifest.json').read_text()); old = d['row_keys_sha256']
                pq.write_table(k, kp, compression='zstd'); d['keys_sha256'] = sha(kp); d['row_keys_sha256'] = old; rebind(base, H, d)
            else:
                rebind(base, H, keys=k)
        return m
    def swap01(k):
        idx = list(range(len(k))); idx[0], idx[1] = 1, 0
        return k.take(pa.array(idx))
    def setcol(name, fn):
        def g(k):
            v = fn(k.column(name).to_pylist())
            return k.set_column(k.schema.get_field_index(name), name, pa.array(v))
        return g
    def m_decision(field, value):
        def m(f, root, base, groups, flags):
            p = root / 'human_role_decision.json'; d = json.loads(p.read_text()); d[field] = value; p.write_text(json.dumps(d))
        return m
    def m_late(f, root, base, groups, flags):
        late = base / 'late_val.parquet'; shutil.copyfile(base / f'{H}.parquet', late)
        shutil.copyfile(base / f'{H}.keys.parquet', teacher.key_path(late))
        d = json.loads((base / f'{H}.parquet.manifest.json').read_text()); d['role'] = 'val'
        Path(f'{late}.manifest.json').write_text(json.dumps(d))
        groups.append(('late', late, 0, 1, 'withinref,rank'))
    def m_protected(f, root, base, groups, flags):
        prot = f.root / 'kadid_terminal'; prot.mkdir()
        for suffix in ('.parquet', '.keys.parquet', '.parquet.manifest.json'):
            shutil.copyfile(base / f'{H}{suffix}', prot / f'{H}{suffix}')
        g = groups[4]; groups[4] = (g[0], prot / f'{H}.parquet', g[2], g[3], g[4])
    def m_ids_width(f, root, base, groups, flags):
        return ids420 + [1853], 1853
    def m_manifest_route(f, root, base, groups, flags):
        d = json.loads((base / f'{H}.parquet.manifest.json').read_text()); d['role'] = 'val'; rebind(base, H, d)
        toml = f.root / 'train_manifest.toml'
        toml.write_text(f'[inputs.human]\npath = "{base / (H + ".parquet")}"\nsha256 = "{sha(base / (H + ".parquet"))}"\n')
        flags += ['--manifest', str(toml)]
    def m_native_keys_val(f, root, base, groups, flags):
        native = groups[-1][1]; pq.write_table(pa.table({'row_id': list(range(330)), 'role': ['val'] * 330,
                                                          'source': ['UPIQ-380'] * 330, 'split': ['development'] * 330}), teacher.key_path(native))
    def m_extra_hdr_group(f, root, base, groups, flags):
        g = groups[4]
        d = json.loads((base / f'{H}.parquet.manifest.json').read_text()); d.pop('human_sources', None); d['source'] = 'HDR teacher'
        rebind(base, H, d); groups[4] = ('hdr', g[1], g[2], g[3], g[4])
    def m_native_manifest(field, value):
        def m(f, root, base, groups, flags):
            native = groups[-1][1]; sp = Path(f'{native}.manifest.json'); d = json.loads(sp.read_text()); d[field] = value; sp.write_text(json.dumps(d))
        return m

    case('positive-control-7-groups', None, 'train')
    case('ordinary-role-val', m_role('role', 'val'), 'refuse', 'unapproved role or training/development weight')
    case('ordinary-split-terminal', m_role('split', 'terminal'), 'refuse', 'unapproved role or training/development weight')
    case('ordinary-tier-T0', m_role('tier', 'T0'), 'refuse', 'unapproved role or training/development weight')
    case('human-development-as-training', m_weights(5, 1, 0), 'refuse', 'fit/development role disagrees with training/validation weights')
    case('human-fit-as-development', m_weights(4, 0, 1), 'refuse', 'fit/development role disagrees with training/validation weights')
    case('teacher-development-as-training', m_weights(1, 1, 0), 'refuse', 'fit/development role disagrees with training/validation weights')
    case('coverage-as-development', m_weights(6, 0, 1), 'refuse', 'fit/development role disagrees with training/validation weights')
    case('ordinary-VAL-role-key-column', m_keys(lambda k: k.append_column('role', pa.array(['val'] * len(k)))), 'refuse', 'label-bearing key schema')
    case('ordered-key-digest-zero', m_role('row_keys_sha256', '0' * 64), 'refuse', 'ordered observation count/order/file pin changed')
    case('keys-reordered-old-digest', m_keys(swap01, keep_digest=True), 'refuse', 'ordered observation count/order/file pin changed')
    case('missing-pair-key-identity', m_keys(lambda k: k.drop(['pair_key'])), 'refuse', 'missing observation identity')
    case('aic-member-in-keys', m_keys(setcol('member_set', lambda v: ['aic3'] + v[1:])), 'refuse', 'unapproved/AIC observation population member')
    case('human-decision-pending', m_decision('state', 'pending'), 'refuse', 'receipt-bound four-source D1 human decision required')
    case('human-decision-adds-aic3', m_decision('sources', list(roles.PRODUCTION_SOURCES) + ['aic3']), 'refuse', 'receipt-bound four-source D1 human decision required')
    case('ids-beyond-width', m_ids_width, 'refuse')
    case('late-VAL-group-last', m_late, 'refuse', 'unapproved role or training/development weight')
    case('protected-ancestry', m_protected, 'refuse', 'protected training input ancestry')
    case('manifest-route-val-group', m_manifest_route, 'refuse', 'unapproved role or training/development weight')
    def m_valid_manifest(f, root, base, groups, flags):
        payload = groups[4][1]
        toml = f.root / 'valid.toml'
        toml.write_text(f'[inputs.human]\npath = "{payload}"\nsha256 = "{sha(payload)}"\n')
        flags += ['--manifest', str(toml)]
    case('manifest-positive-control', m_valid_manifest, 'reach')

    # E31: eight groups (registered inventory) with the real label-free TRAIN keys and owner disposition.
    case('e31-positive-control', None, 'reach', e31=True)
    case('e31-native-VAL-keys', m_native_keys_val, 'refuse', 'E31 native key pin changed', e31=True)
    case('e31-ordinary-role-val-terminal-T0', lambda *a: [m_role(k, v)(*a) for k, v in (('role', 'val'), ('split', 'terminal'), ('tier', 'T0'))], 'refuse', 'unapproved role or training/development weight', e31=True)
    case('e31-extra-hdr-companion', m_extra_hdr_group, 'refuse', 'E31 requires exactly the inherited seven SDR groups and upiq380', e31=True)
    case('e31-native-role-val', m_native_manifest('role', 'val'), 'refuse', None, e31=True)
    case('e31-native-split-development', m_native_manifest('split', 'development'), 'refuse', None, e31=True)
    case('e31-native-table-pin', m_native_manifest('table_sha256', '0' * 64), 'refuse', None, e31=True)
    case('e31-native-ids', m_native_manifest('requested_ids', list(range(420))), 'refuse', None, e31=True)
    case('e31-human-development-as-training', m_weights(5, 1, 0), 'refuse', 'fit/development role disagrees with training/validation weights', e31=True)
    # E29 HDR group appended LAST with VAL keys after seven valid SDR groups: no SDR payload may open first.
    case('e29-late-hdr-VAL-keys', None, 'refuse', None, e29_late=True)


if __name__ == '__main__':
    os.environ.setdefault('TMPDIR', str(Path.home() / 'tmp/v40r2'))
    for fam in args.family or ['e29', 'e32', 'd1']:
        {'e29': e29_family, 'e32': e32_family, 'd1': d1_family}[fam]()
    bad = [r for r in results if not r['ok']]
    (OUT / 'RESULT.json').write_text(json.dumps({'trainer_sha256': sha(BIN), 'cases': results, 'not_ok': bad}, indent=1) + '\n')
    print(f'cases {len(results)} ok {len(results) - len(bad)} not_ok {len(bad)}')

    if bad:
        raise SystemExit(1)
