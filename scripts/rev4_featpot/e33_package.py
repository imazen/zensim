"""E33 fleet packet: fit specs, jobset caps and a filler-placement rehearsal (registration section 10.1).

Local artifacts only. Nothing here queues work, uploads, pushes an image or grants approval; launch goes through the
coordinator with its own authorization record. Cell argv follow the V40 owners (`v40_package.specs`,
`e30_four_source.grid`): one data archive (the V40 control's, data_sha 9c3eff1b...) for every jobset.
"""

import argparse
import json
from pathlib import Path
import subprocess

from v2_common import FX1_SHA256, fx1_declaration, selection_id, sha
from v2_human_role import PRODUCTION_SOURCES

DATE = "20261010"
DATA_SHA = "9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008"
ROOT = "/var/tmp/rev4-featpot/v2e29"
RECIPE = "@h32:H128:cv16:cf98"
# Registered per-cell wall caps (registration section 10.1, owner-approved): 3 x the 1,399 s V40 control maximum for
# control/A; 3 x the C maximum the r3500 first-epoch smokes imply (1,399 x 9.5/5.3, rounded up) for C. Carried in the
# harvest package and enforced per cell by the program's fit_cell_exec (stop and diagnose, never retry).
WALL_CAPS = {"control": 4200, "a": 4200, "c": 7600, "full-a": 4200, "full-c": 7600}
CONTROL_COLUMNS = (list(range(13, 26)) + list(range(39, 156)) + list(range(401, 430))
                   + list(range(459, 720)))


def arms():
    """Registered E33 arms: (name, spec, columns, fx1)."""
    decl = fx1_declaration()
    direct = decl["direct"]
    assert selection_id(CONTROL_COLUMNS) == "59f0bbc2f290"
    assert selection_id(direct) == "3b7bd5ebe929"
    return {
        "control": (f"sel:59f0bbc2f290{RECIPE}", CONTROL_COLUMNS),
        "a": (f"sel:3b7bd5ebe929{RECIPE}", direct),
        "c": (f"sel:3b7bd5ebe929{RECIPE}:fx1", direct),
    }


def jobset(name):
    return f"fite33-{name}-{DATE}"


def cell_argv(script, spec, columns, seed, dest, extra):
    return [script, "--spec", spec, "--head", "N", "--seed-index", str(seed), "--root", ROOT,
            "--columns", ",".join(map(str, columns)), "--strict-admission", "--train-only",
            "--data-role-decision", ROOT + "/human_role_decision.json", "--dest", dest, *extra]


def specs(bundle, program, data, ctl):
    """Write the four registered fit specs and declare them through the canonical fleet owner."""
    if sha(data) != DATA_SHA:
        raise ValueError(f"{data}: not the registered V40 control data archive")
    by_arm = arms()
    plans = {}
    for name in ("control", "a", "c"):
        spec, columns = by_arm[name]
        cells = []
        for fold in PRODUCTION_SOURCES:
            for seed in range(10):
                cell = f"{spec}__N/without_{fold}_s{seed}"
                cells.append(dict(name=cell, argv=cell_argv(
                    "v2_lodo_mlp.py", spec, columns, seed,
                    f"/var/tmp/rev4-featpot/e33-{name}-results/cells/{cell}", ["--heldout", fold])))
        plans[jobset(name)] = cells
    full = []
    for name in ("a", "c"):
        spec, columns = by_arm[name]
        for seed in range(3):
            cell = f"{spec}__N/full_s{seed}"
            full.append(dict(name=cell, argv=cell_argv(
                "v2_confirm_fit.py", spec, columns, seed,
                f"/var/tmp/rev4-featpot/e33-full-results/confirm/cells/{cell}",
                ["--pack-production", "--e33-output-stage"])))
    plans[jobset("full")] = full
    assert [len(v) for v in plans.values()] == [40, 40, 40, 6]
    for js, cells in plans.items():
        path = bundle / f"fit-spec-{js}.json"
        path.write_text(json.dumps(dict(program_sha=sha(program), data_sha=DATA_SHA, cells=cells,
                                        e33_fx1_sha256=FX1_SHA256), indent=2) + "\n")
        subprocess.run([str(ctl), "declare-fits", "--spec", str(path),
                        "--out", str(bundle / f"fit-manifest-{js}.json")], check=True)
    return plans


def smoke_routes():
    """Every (variant, route) the packet declares: three LODO arms x four folds, A/C production."""
    out = [(arm, fold) for arm in ("control", "a", "c") for fold in PRODUCTION_SOURCES]
    return out + [("full-a", "production"), ("full-c", "production")]


def smokes(dest, root, bin_dir, budget_lodo="2:128", budget_full="1:49999"):
    """Bounded local fits through the real strict owners, one per declared route (training-only, TRAIN legs).

    They exist to derive the harvest contract from what the program actually admits and trains on; they are
    never registered fits (`execution_contract` = local-smoke).
    """
    import os
    import sys
    by_arm = arms()
    env = dict(os.environ, REV4_V2_BIN_DIR=str(bin_dir), ZENSIM_MAX_TIER="v3", RAYON_NUM_THREADS="1",
               OMP_NUM_THREADS="1")
    here = Path(__file__).resolve().parent
    for variant, route in smoke_routes():
        arm = variant.removeprefix("full-")
        spec, columns = by_arm[arm]
        cell = dest / variant / route
        if (cell / "result.json").is_file():
            continue
        if route == "production":
            argv = cell_argv("v2_confirm_fit.py", spec, columns, 0, str(cell),
                             ["--pack-production", "--e33-output-stage", "--local-smoke-budget", budget_full])
        else:
            argv = cell_argv("v2_lodo_mlp.py", spec, columns, 0, str(cell),
                             ["--heldout", route, "--local-smoke-budget", budget_lodo])
        argv[argv.index("--root") + 1] = str(root)
        argv[argv.index("--data-role-decision") + 1] = str(root / "human_role_decision.json")
        cell.parent.mkdir(parents=True, exist_ok=True)
        with (cell.parent / f"{route}.log").open("w") as log:
            subprocess.run([sys.executable, str(here / argv[0]), *argv[1:]], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


def contracts(bundle, v40_contract, smoke_dir, inspector):
    """The E33 harvest package: the V40 control variant (same data, tables, seeds, budget) per registered arm,
    with production inputs and every arm's input roles taken from the bounded local fits. Each LODO route's
    admitted tables and weights must equal the V40 template; anything else refuses."""
    import copy
    v40 = json.loads(v40_contract.read_text())
    if v40.get("schema") != "v40-research-fit-package-v1":
        raise ValueError("not the V40 package")
    template = next(v for v in v40["variants"] if v["name"] == "control")
    if template["data_sha"] != DATA_SHA or template["root"] != ROOT:
        raise ValueError("V40 control variant does not bind the registered data/root")
    by_arm = arms()
    variants = []
    for variant in ("control", "a", "c", "full-a", "full-c"):
        arm = variant.removeprefix("full-")
        spec, columns = by_arm[arm]
        c = copy.deepcopy(template)
        c.update(schema="e33-research-fit-contract-v1", name=variant, spec=spec, specs=[spec],
                 columns=list(columns), launchable=True, research_hdr=False, wall_cap_sec=WALL_CAPS[variant],
                 e33_fx1_sha256=FX1_SHA256 if arm == "c" else None)
        decl = fx1_declaration()
        # The trainer admits tables for every id the model READS: the read set for C (products' fragility
        # factors included), the kept columns otherwise. Harvest compares decoded `requested_ids` to this.
        c["requested_ids"] = (sorted(set(decl["direct"]) | {b for _, b in decl["products"]})
                              if arm == "c" else list(columns))
        c.pop("hdr_declarations", None)
        c.pop("hdr_tables", None)
        routes = ["production"] if variant.startswith("full-") else list(PRODUCTION_SOURCES)
        c["routes"] = {r: template["routes"][r] for r in routes}
        c["input_contracts"], c["train_weights"] = {}, {}
        for route in routes:
            cell = smoke_dir / variant / route
            result = json.loads((cell / "result.json").read_text())
            tables = sorted(result["selection"]["strict_table_admission"], key=lambda t: t["name"])
            if tables != c["routes"][route]:
                raise ValueError(f"{variant}/{route}: admitted tables differ from the V40 control template")
            if route != "production" and result["train_weights"] != template["train_weights"][route]:
                raise ValueError(f"{variant}/{route}: train weights differ from the V40 control template")
            bake = cell / "refit/last.bin"
            decoded = json.loads(subprocess.check_output([str(inspector), str(bake)], text=True))
            c["input_contracts"][route] = {t["name"]: {k: t.get(k) for k in
                                           ("loss_mode", "n_features", "rows", "train_w", "val_w", "within_ref")}
                                           for t in decoded["repro"]["inputs"]}
            c["train_weights"][route] = result["train_weights"]
            recorded = {tuple(t.get("requested_ids") or ()) for t in decoded["repro"]["table_admission"]["tables"]}
            if recorded != {tuple(c["requested_ids"])}:
                raise ValueError(f"{variant}/{route}: decoded requested ids differ from the contract")
            if arm != "c" and route != "production" and c["input_contracts"][route] != template["input_contracts"][route]:
                raise ValueError(f"{variant}/{route}: input roles differ from the V40 control template")
        variants.append(c)
    out = bundle / "e33-fit-contract.json"
    out.write_text(json.dumps(dict(schema="e33-research-fit-package-v1", variants=variants),
                              sort_keys=True, indent=2) + "\n")
    return out


def caps(bundle, approved, memory_c):
    """Caps for every E33 jobset from the approved V40 capsfix standard map (hosts listed, never empty)."""
    source = json.loads(approved.read_text())
    standard = source["fitv40-control-20261007"]
    if not isinstance(standard.get("hosts"), dict) or not standard["hosts"]:
        raise ValueError("approved standard map has no hosts")
    out = {}
    for name in ("control", "a", "c", "full"):
        out[jobset(name)] = dict(
            hosts=dict(standard["hosts"]),
            memory=memory_c if name in ("c", "full") else standard["memory"],
            reason=f"E33 {name}; approved V40 capsfix standard map; one CPU per cell, no swap; "
                   "coordinator authorization required",
        )
    (bundle / "jobset_caps.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


ENVELOPE_MARK = ("<<'PY_CAP'\n", "PY_CAP\n")


def envelope_source(launcher):
    """The exact resource-envelope program the filler runs (launch_v2.sh's PY_CAP block)."""
    text = launcher.read_text()
    start = text.index(ENVELOPE_MARK[0]) + len(ENVELOPE_MARK[0])
    end = text.index(ENVELOPE_MARK[1], start)
    return text[start:end]


def rehearse(bundle, launcher, fleet_hosts, want=8):
    """Run the filler's envelope for every (jobset, host) with live counts stubbed to zero active cells.

    The V40 failure mode (empty `hosts`, 147 refusals, 0/40 cells) refuses here. No SSH, no docker, no queue.
    """
    import contextlib
    import io
    import subprocess as sp
    from unittest.mock import patch

    from v40_launch import placement

    program = envelope_source(launcher)
    caps_path = bundle / "jobset_caps.json"
    caps_doc = json.loads(caps_path.read_text())
    report = dict(schema="e33-placement-rehearsal-v1", launcher=str(launcher), launcher_sha256=sha(launcher),
                  caps_sha256=sha(caps_path), fleet_hosts=sorted(fleet_hosts), jobsets={})
    for js, entry in caps_doc.items():
        placement(entry, fleet_hosts)  # the launch gate's own check
        rows = {}
        for host in entry["hosts"]:
            fake = sp.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            out = io.StringIO()
            with patch("subprocess.run", return_value=fake), contextlib.redirect_stdout(out):
                with patch("sys.argv", ["envelope", str(caps_path), js, host, str(want)]):
                    try:
                        exec(compile(program, "launch_v2.sh:PY_CAP", "exec"), {"__name__": "__main__"})
                    except SystemExit as refusal:
                        rows[host] = dict(refused=str(refusal))
                        continue
            memory, launches = out.getvalue().split()
            rows[host] = dict(memory=memory, launches=int(launches), slots=entry["hosts"][host])
        refused = [h for h, r in rows.items() if "refused" in r or r["launches"] <= 0]
        report["jobsets"][js] = dict(hosts=rows, refused=refused,
                                     total_first_wave=sum(r.get("launches", 0) for r in rows.values()))
        if refused:
            raise ValueError(f"{js}: placement refused on {refused}")
    # The envelope counts only containers of the SAME jobset, so jobsets launched together stack on a host.
    hosts = sorted({h for e in caps_doc.values() for h in e["hosts"]})
    report["all_jobsets_concurrent"] = {
        h: dict(cells=sum(e["hosts"].get(h, 0) for e in caps_doc.values()),
                nominal_gib=sum(e["hosts"].get(h, 0) * int(str(e["memory"]).rstrip("g")) for e in caps_doc.values()))
        for h in hosts}
    report["launch_order_note"] = ("each jobset's envelope ignores the others; launching all four together can place "
                                   "the per-host sums above (tower's standing cap is 40g with >= 12 GiB free), so "
                                   "launch jobsets in sequence or lower per-host slots")
    (bundle / "PLACEMENT_REHEARSAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


FROZEN = ("BUILD_LOG.txt", "PLACEMENT_REHEARSAL.json", "build-meta.packer-input.json", "e33-fit-contract.json",
          "jobset_caps.json", "program.tar.gz")


def freeze(bundle, superseded, status):
    """Write PACKET.json, the packet's freeze inventory, from the bundle's own files.

    Refuses unless every jobset has a declared manifest and a PASS executor smoke whose output the program's harvest
    owner VERIFIED, and unless the image tag names this program archive.
    """
    meta = json.loads((bundle / "build-meta.packer-input.json").read_text())
    program_sha = sha(bundle / "program.tar.gz")
    image = (bundle / "IMAGE_TAG.txt").read_text().strip()
    if f"fit-e33-{program_sha[:12]}-" not in image:
        raise ValueError(f"image tag {image} does not name program {program_sha[:12]}")
    files = {name: sha(bundle / name) for name in FROZEN}
    jobsets, smokes_seen = {}, {}
    for js in ("control", "a", "c", "full"):
        for kind in ("spec", "manifest"):
            path = next(bundle.glob(f"fit-{kind}-fite33-{js}-*.json"))
            files[path.name] = sha(path)
        manifest = json.loads(next(bundle.glob(f"fit-manifest-fite33-{js}-*.json")).read_text())
        jobsets[f"fite33-{js}-{DATE}"] = len(manifest)
        runs = sorted(bundle.glob(f"smoke-{js}-[0-9]*-[0-9]*"), key=lambda d: d.stat().st_mtime)
        if not runs:
            raise ValueError(f"no executor smoke for jobset {js}")
        run = runs[-1]
        receipt = next(run.glob("*_PATH_PASS.json"))
        record = json.loads(receipt.read_text())
        if record.get("status") != "PASS" or record.get("harvest", {}).get("status") != "VERIFIED":
            raise ValueError(f"{run.name}: smoke not PASS/VERIFIED")
        if record.get("program_sha") != program_sha or record.get("image") != image:
            raise ValueError(f"{run.name}: smoke ran another program or image")
        smokes_seen[run.name] = dict(cell=record["cell"], status=record["status"], harvest=record["harvest"],
                                     memory_peak_bytes=record.get("memory_peak_bytes"),
                                     receipt=str(receipt.relative_to(bundle)), sha256=sha(receipt))
    caps_doc = json.loads((bundle / "jobset_caps.json").read_text())
    contract = json.loads((bundle / "e33-fit-contract.json").read_text())
    image_id = (bundle / "IMAGE_ID.txt").read_text().strip()
    packet = dict(
        schema="e33-fleet-packet-v1", registration="benchmarks/e33_registration_2026-10-09.md", status=status,
        zensim_commit=meta["zensim_source_commit"], binaries_build_commit=meta["build_commit"],
        zenmetrics_commit=meta["zenmetrics_lane_commit"], program_sha=program_sha, image=image, image_id=image_id,
        image_recipe_sha256=meta["image_recipe_sha256"], data_sha=DATA_SHA,
        data_file="e33-fit-data.tar.gz (hardlink of the V40 control archive)", jobsets=jobsets,
        memory_caps={js.split("-")[1]: v.get("memory") for js, v in caps_doc.items() if js.startswith("fite33-")},
        wall_caps_seconds={v["name"]: v["wall_cap_sec"] for v in contract["variants"]},
        wall_cap_owner="program fit_cell_exec: per-cell kill at wall_cap_sec, deterministic failure, never retried",
        executor_smokes=smokes_seen, files=files, superseded=list(superseded))
    if sum(jobsets.values()) != 126 or packet["wall_caps_seconds"] != WALL_CAPS:
        raise ValueError(f"packet shape differs from the registration: {jobsets} {packet['wall_caps_seconds']}")
    (bundle / "PACKET.json").write_text(json.dumps(packet, indent=1, sort_keys=True) + "\n")
    return packet


# E34 uses the same declaration, archive and placement owners as E33. Conditional
# K branches are pinned before launch; exactly one branch per family is released.
def e34_variants(template_path, wall_caps):
    import copy
    old = json.loads(template_path.read_text())
    variants = []
    for family in ('a', 'c'):
        for arm, hidden in [('control', 128), ('w', 32), *([('l', 1)] if family == 'a' else []),
                            ('k01', 128), ('k1', 128), ('k10', 128)]:
            lam = {'k01':'0.1', 'k1':'1.0', 'k10':'10.0'}.get(arm)
            name = f'{arm}{family}'
            flags = {'--monotone-cbc': arm == 'l',
                     '--steer-pairs': '/opt/fleet-fits/program/data/e34/augmentation.parquet' if lam else None,
                     '--steer-lambda': lam}
            spec = f'sel:3b7bd5ebe929@h32:H{hidden}:cv16:cf98' + (':fx1' if family == 'c' else '')
            for full in (False, True):
                c = copy.deepcopy(next(v for v in old['variants'] if v['name'] == ('full-' if full else '') + family))
                c.update(schema='e34-research-fit-contract-v1', name=('full-' if full else '') + name,
                         spec=spec, specs=[spec], hidden=hidden, e34_flags=flags,
                         wall_cap_sec=wall_caps[name])
                variants.append(c)
    return variants


def e34_specs(bundle, program, data, ctl, contract):
    if sha(data) != DATA_SHA:
        raise ValueError('E34 rank archive differs from registered E33 bytes')
    variants = json.loads(contract.read_text())['variants']
    plans, unconditional, branches = {}, [], {}
    for c in variants:
        full = c['name'].startswith('full-')
        name = c['name'].removeprefix('full-')
        is_k = name.startswith('k')
        js = f'fite34-{name}-' + ('pilot' if full and is_k else 'full' if full else 'lodo') + '-20261015'
        seeds = [0] if full and is_k else range(3) if full else range(10)
        cells = []
        for route in (['production'] if full else PRODUCTION_SOURCES):
            for seed in seeds:
                cell = f"{c['spec']}__N/" + (f'full_s{seed}' if full else f'without_{route}_s{seed}')
                extra = ['--pack-production','--e33-output-stage'] if full else ['--heldout', route]
                for flag, value in c['e34_flags'].items():
                    if value is True: extra.append(flag)
                    elif isinstance(value, str): extra += [flag, value]
                dest = f'/var/tmp/rev4-featpot/e34-{name}-results/' + ('confirm/' if full else '') + f'cells/{cell}'
                cells.append(dict(name=cell, argv=cell_argv('v2_confirm_fit.py' if full else 'v2_lodo_mlp.py',
                                                          c['spec'], c['columns'], seed, dest, extra)))
        plans[js] = cells
        if not is_k or full:
            unconditional.append(js)
        else:
            # Full seeds 1/2 join each conditional branch. Seed 0 is the pilot.
            full_c = next(v for v in variants if v['name'] == 'full-' + name)
            for seed in (1, 2):
                cell = f"{c['spec']}__N/full_s{seed}"
                extra = ['--pack-production','--e33-output-stage']
                for flag,value in full_c['e34_flags'].items():
                    if value is True: extra.append(flag)
                    elif isinstance(value,str): extra += [flag,value]
                cells.append(dict(name=cell, argv=cell_argv('v2_confirm_fit.py',c['spec'],c['columns'],seed,
                                  f'/var/tmp/rev4-featpot/e34-{name}-results/confirm/cells/{cell}',extra)))
            branches[name] = js
    for js,cells in plans.items():
        path = bundle / f'fit-spec-{js}.json'
        path.write_text(json.dumps(dict(program_sha=sha(program),data_sha=DATA_SHA,cells=cells),indent=2)+'\n')
        subprocess.run([str(ctl),'declare-fits','--spec',str(path),'--out',str(bundle/f'fit-manifest-{js}.json')],check=True)
    initial = sum(len(plans[js]) for js in unconditional)
    assert initial == 221 and all(len(plans[js]) == 42 for js in branches.values())
    plan = dict(schema='e34-conditional-305-cell-plan-v1', unconditional=unconditional,
                conditional=branches, initial_cells=initial, selected_main_cells=84,total_executed_cells=305,
                rule='For each family maximize dev96 joint(M2>=.99,M3f>=.70); ties choose smaller lambda; reuse selected pilot as full seed0',
                branches_per_family=1, margin_a=0.0065,margin_c=0.0065,
                manifests={js:sha(bundle/f'fit-manifest-{js}.json') for js in plans})
    (bundle/'E34_JOB_PLAN.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    return plan


def e34_freeze(bundle):
    """Freeze the existing 305-cell conditional plan after real-entry gates."""
    plan=json.loads((bundle/'E34_JOB_PLAN.json').read_text())
    cap=json.loads((bundle/'FIRST_EPOCH_CAPS.json').read_text())
    parity=json.loads((bundle/'CONTROL_PARITY.json').read_text())
    if not parity['weights_bit_identical']:
        raise ValueError('Control-A parity pre-check is not bit-identical')
    image=(bundle/'IMAGE_TAG.txt').read_text().strip()
    image_id=(bundle/'IMAGE_ID.txt').read_text().strip()
    digest=(bundle/'IMAGE_DIGEST.txt').read_text().strip()
    if not digest.startswith('sha256:') or len(digest)!=71:
        raise ValueError('registry image digest missing')
    if sha(bundle/'e33-fit-data.tar.gz')!=DATA_SHA or plan['total_executed_cells']!=305:
        raise ValueError('registered data/grid differs')
    program_sha=sha(bundle/'program.tar.gz')
    meta=json.loads((bundle/'build-meta.packer-input.json').read_text())
    contract=json.loads((bundle/'e34-fit-contract.json').read_text())
    caps_doc=json.loads((bundle/'jobset_caps.json').read_text())
    allowed={'tower','r3500','r3800x','i270','i134','r5600g'}
    if set(caps_doc)!=set(plan['manifests']) or any(set(c['hosts'])!=allowed for c in caps_doc.values()):
        raise ValueError('only six approved hosts may have slots')
    receipts={}
    for variant in contract['variants']:
        full=variant['name'].startswith('full-')
        arm=variant['name'].removeprefix('full-')
        route='pilot' if full and arm.startswith('k') else 'full' if full else 'lodo'
        runs=list(bundle.glob(f'smoke-{arm}-{route}-0-201/*_PATH_PASS.json'))
        if len(runs)!=1:raise ValueError(f'missing smoke for {variant["name"]}')
        path=runs[0];r=json.loads(path.read_text())
        if (r['status']!='PASS' or r.get('harvest',{}).get('status')!='VERIFIED'
                or r['program_sha']!=program_sha or r['image']!=image
                or r['harvest']['execution_contract']!='local-smoke'):
            raise ValueError(f'smoke not PASS/VERIFIED: {path}')
        receipts[variant['name']]=dict(receipt=str(path.relative_to(bundle)),sha256=sha(path),
                                      memory_peak_bytes=r['memory_peak_bytes'])
        if variant['wall_cap_sec']!=cap['wall_cap_sec']:
            raise ValueError('wall cap differs from registered smoke rule')
    for js,pin in plan['manifests'].items():
        if sha(bundle/f'fit-manifest-{js}.json')!=pin:
            raise ValueError('declared conditional manifest changed')
    frozen=['program.tar.gz','build-meta.packer-input.json','source-pins.json','e34-fit-contract.json',
            'E34_JOB_PLAN.json','jobset_caps.json','PLACEMENT_REHEARSAL.json','FIRST_EPOCH_CAPS.json',
            'CONTROL_PARITY.json','DEVELOPMENT_BINDING.json','LAUNCH_CONTROL.json','BUILD_LOG.txt']
    frozen += [p.name for p in bundle.glob('fit-spec-*.json')]+[p.name for p in bundle.glob('fit-manifest-*.json')]
    frozen += [r['receipt'] for r in receipts.values()]
    frozen += [str(p.relative_to(bundle)) for p in bundle.glob('smoke-*-0-201/*_TIME.txt')]
    costs={}
    for r in cap['records']:
        name=Path(r['receipt']).parts[0].split('-')[1]
        costs[name]=max(costs.get(name,0),r['epoch_seconds_upper'])
    for family in ['a','c']:
        for k in ['k01','k1']:costs[k+family]=costs['k10'+family]
    doc=dict(schema='e34-fleet-packet-v1',registration='benchmarks/e34_registration_freeze_draft_2026-10-15.md',
             revision=5,margin_a=0.0065,margin_c=0.0065,zensim_commit=meta['zensim_source_commit'],
             zenmetrics_commit=meta['zenmetrics_lane_commit'],program_sha=program_sha,data_sha=DATA_SHA,
             image=image,image_id=image_id,image_digest=digest,image_ref=image+'@'+digest,
             image_recipe_sha256=meta['image_recipe_sha256'],
             jobsets={js:len(json.loads((bundle/f'fit-manifest-{js}.json').read_text())) for js in plan['manifests']},
             scientific_cells=305,conditional_plan_sha256=sha(bundle/'E34_JOB_PLAN.json'),
             wall_caps_seconds={v['name']:v['wall_cap_sec'] for v in contract['variants']},
             executor_smokes=receipts,smoke_epoch_seconds=costs,
             persistence=dict(hosts=['i134','r5600g','i270'],interval_seconds=60,optimizer_resume=False,
                 shorter_first=True,no_cutoff=True,
                 snapshot_root='/mnt/tower/output/zensim/e34-launch-2026-10-15/loan-persistence'),
             files={n:sha(bundle/n) for n in sorted(frozen)})
    (bundle/'PACKET.json').write_text(json.dumps(doc,indent=1,sort_keys=True)+'\n')
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smokes", "contracts", "specs", "caps", "rehearse", "freeze", "e34-contracts", "e34-specs", "e34-freeze"))
    p.add_argument("--template", type=Path)
    p.add_argument("--wall-caps", type=Path)
    p.add_argument("--smoke-dir", type=Path)
    p.add_argument("--root", type=Path, help="local copy of the registered data root (smokes)")
    p.add_argument("--bin-dir", type=Path)
    p.add_argument("--v40-contract", type=Path)
    p.add_argument("--inspector", type=Path)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--program", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--ctl", type=Path)
    p.add_argument("--approved-caps", type=Path, help="V40 capsfix jobset_caps.json (the approved standard map)")
    p.add_argument("--memory-c", help="Arm C / full-data memory cap from the first-epoch smokes, e.g. 6g")
    p.add_argument("--launcher", type=Path, default=Path("/var/tmp/fitv2/launch_v2.sh"))
    p.add_argument("--fleet-hosts", type=Path, help="CAPS_FIX.json carrying the registered placement keys")
    p.add_argument("--superseded", action="append", default=[], help="freeze: earlier packet attempt and why")
    p.add_argument("--status", default="READY FOR COORDINATOR REVIEW; not launched, image not pushed")
    a = p.parse_args()
    if a.mode == "e34-freeze":
        print(json.dumps(e34_freeze(a.bundle),indent=1))
    elif a.mode == "e34-contracts":
        variants=e34_variants(a.template,json.loads(a.wall_caps.read_text()))
        (a.bundle/"e34-fit-contract.json").write_text(json.dumps(dict(schema="e34-research-fit-package-v1",variants=variants),indent=2,sort_keys=True)+"\n")
    elif a.mode == "e34-specs":
        e34_specs(a.bundle,a.program,a.data,a.ctl,a.bundle/"e34-fit-contract.json")
    elif a.mode == "smokes":
        if not (a.smoke_dir and a.root and a.bin_dir):
            p.error("smokes requires --smoke-dir, --root and --bin-dir")
        smokes(a.smoke_dir, a.root, a.bin_dir)
    elif a.mode == "contracts":
        if not (a.smoke_dir and a.v40_contract and a.inspector):
            p.error("contracts requires --smoke-dir, --v40-contract and --inspector")
        print(contracts(a.bundle, a.v40_contract, a.smoke_dir, a.inspector))
    elif a.mode == "specs":
        if not (a.program and a.data and a.ctl):
            p.error("specs requires --program, --data and --ctl")
        specs(a.bundle, a.program, a.data, a.ctl)
    elif a.mode == "caps":
        if not (a.approved_caps and a.memory_c):
            p.error("caps requires --approved-caps and --memory-c")
        caps(a.bundle, a.approved_caps, a.memory_c)
    elif a.mode == "freeze":
        packet = freeze(a.bundle, a.superseded, a.status)
        print(json.dumps({k: v for k, v in packet.items() if k not in ("files", "executor_smokes")}, indent=1))
    else:
        if not a.fleet_hosts:
            p.error("rehearse requires --fleet-hosts")
        hosts = json.loads(a.fleet_hosts.read_text())["fleet_hosts"]
        print(json.dumps(rehearse(a.bundle, a.launcher, hosts), indent=2))


if __name__ == "__main__":
    main()
