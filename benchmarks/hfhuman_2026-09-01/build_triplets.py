#!/usr/bin/env python3
"""Forced-choice (2AFC) agreement tables for the JPEG-AIC study family.

Turns 515,250 raw human triplet responses into `panel --pairwise` input, one
rows-file per (arm, subset, scorer) with an IDENTICAL group ordering across
scorers so the cluster bootstrap is exactly paired, then drives the owner.

No statistic is computed in this file. Every number in the output TSVs comes
from `panel --pairwise` = `zensim_validate::pairwise::agreement`. The RNG for
the bootstrap lives here, per the `--batch` contract (caller owns the RNG,
owner owns the arithmetic).

Response semantics, established from the data rather than assumed
(`anatomy.py` G4 + the trap check): the pivot is ALWAYS the original, and
`response` names the side the worker judged MORE DIFFERENT from it — 76.5 %
of trap responses (original vs dlevel 10) name the distorted side.

Subsets:
  vs_original   one side IS the original -> the visually-lossless / JND
                threshold question, the one a near-lossless product lives on
  same_codec    both sides distorted by the SAME codec -> pure ladder order
  cross_codec   both sides distorted by DIFFERENT codecs -> the cross-codec
                comparability a picker and an RD study depend on
  all           the union of the three
Trap and bias rows are EXCLUDED from all four and reported separately: bias
rows (left and right are the same file) have no ground truth and measure the
observer noise floor; trap rows measure attention, not quality.
"""
from __future__ import annotations
import argparse, csv, json, random, subprocess, sys
from pathlib import Path

STUDIES = {
    "aic3_btc": ("/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_BTC_final_response_data_2024.01.10.csv", "BTC"),
    "sdr25_btc": ("/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_BTC_JPEG_AI_responses_2025.02.28_v1.csv", "BTC"),
    "sdr25_ptc": ("/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_PTC_JPEG_AI_responses_2025.02.28_v1.csv", "PTC"),
    "aic3_iptc": ("/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_IPTC_final_response_data_2024_06_28 (1).csv", "IPTC"),
}
ARM_STUDIES = {"ptc_native": ["sdr25_ptc"],
               "btc_displayed": ["aic3_btc", "sdr25_btc"],
               "btc_native": ["aic3_btc", "sdr25_btc"],
               # the recovered AIC-3 interactive-PTC campaign (build_stimuli G8)
               "iptc_native": ["aic3_iptc"],
               # ... and its mis-mapped negative controls, same responses
               "iptc_ctl_levelshift": ["aic3_iptc"],
               "iptc_ctl_levelrev": ["aic3_iptc"],
               "iptc_ctl_codecrot": ["aic3_iptc"],
               "iptc_ctl_imgrot": ["aic3_iptc"],
               # both native legs pooled -- the axis §2.7 said recovery would grow
               "native_all": ["sdr25_ptc", "aic3_iptc"]}
UNDECIDED = {"notsure", "notSure", "skip", ""}


def load_responses():
    out = {}
    for key, (p, fam) in STUDIES.items():
        rows = []
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        out[key] = rows
    return out


def classify(r) -> str:
    cl, cr = r["codec_left"], r["codec_right"]
    if cl == "0" or cr == "0":
        return "vs_original"
    return "same_codec" if cl == cr else "cross_codec"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--panel-bin", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--arms", default="ptc_native,btc_displayed,btc_native",
                    help="comma list; the default reproduces the original three-arm run byte-for-byte")
    ap.add_argument("--out-name", default="pairwise_results.tsv")
    ap.add_argument("--diag-name", default="pairwise_diagnostics.json")
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args()
    d = Path(a.dir); scratch = Path(a.scratch); scratch.mkdir(parents=True, exist_ok=True)
    resp = load_responses()
    results = []
    diag = {}

    want = [x for x in a.arms.split(",") if x]
    for arm in want:
        assert arm in ARM_STUDIES, f"unknown arm {arm}"
    for arm, studies in ((k, ARM_STUDIES[k]) for k in want):
        if arm == "native_all":
            srows = []
            for src in ("ptc_native", "iptc_native"):
                srows += list(csv.DictReader(open(d / f"{src}_scores.tsv"), delimiter="\t"))
        else:
            srows = list(csv.DictReader(open(d / f"{arm}_scores.tsv"), delimiter="\t"))
        score = {r["stimulus"]: r for r in srows}
        assert len(score) == len(srows), f"{arm}: duplicate stimulus keys in the score table"
        scorers = [c for c in srows[0] if c not in ("row", "stimulus")]
        # ---- assemble decided responses, keyed by triplet -----------------
        # counts[(subset, group)] -> {"left": w, "right": w, "L": name, "R": name}
        counts: dict[tuple[str, str], dict] = {}
        diag_arm = {"n_rows": 0, "undecided": 0, "trap": 0, "bias": 0,
                    "bias_left_frac": None, "trap_picked_distorted_frac": None}
        bias_l = bias_n = trap_d = trap_n = 0
        for st in studies:
            for r in resp[st]:
                diag_arm["n_rows"] += 1
                if r["response"] in UNDECIDED:
                    diag_arm["undecided"] += 1
                    continue
                if r["is_bias"] == "1":
                    diag_arm["bias"] += 1
                    bias_n += 1; bias_l += (r["response"] == "left")
                    continue
                if r["is_trap"] == "1":
                    diag_arm["trap"] += 1
                    orig = "left" if r["dlevel_left"] == "0" else "right"
                    trap_n += 1; trap_d += (r["response"] != orig)
                    continue
                sub = classify(r)
                g = f"{st}:{r['question_id']}"
                e = counts.setdefault((sub, g), {"left": 0.0, "right": 0.0,
                                                 "L": r["img_left"], "R": r["img_right"],
                                                 "img": int(r["img_num"]), "study": st})
                assert e["L"] == r["img_left"] and e["R"] == r["img_right"], f"{g}: stimuli differ within a question_id"
                e[r["response"]] += 1.0
        diag_arm["bias_left_frac"] = round(bias_l / bias_n, 6) if bias_n else None
        diag_arm["trap_picked_distorted_frac"] = round(trap_d / trap_n, 6) if trap_n else None
        diag[arm] = diag_arm

        subsets = ["all", "vs_original", "same_codec", "cross_codec"] + [f"study:{s}" for s in studies]

        def members(sub: str):
            if sub == "all":
                return [k for k in counts]
            if sub.startswith("study:"):
                return [k for k in counts if counts[k]["study"] == sub.split(":", 1)[1]]
            return [k for k in counts if k[0] == sub]

        for sub in subsets:
            keys = sorted(members(sub))
            if not keys:
                continue
            gnames = [k[1] for k in keys]
            gimg = [counts[k]["img"] for k in keys]
            # resample manifests (shared by every scorer -> exactly paired)
            rng = random.Random(a.seed)
            man_q = scratch / f"{arm}__{sub.replace(':','_')}__q.tsv"
            with open(man_q, "w") as f:
                f.write("POINT\t*\n")
                n = len(keys)
                for b in range(a.boot):
                    f.write(f"B{b}\t" + ",".join(str(rng.randrange(n)) for _ in range(n)) + "\n")
            imgs = sorted(set(gimg))
            by_img = {i: [j for j, v in enumerate(gimg) if v == i] for i in imgs}
            rng2 = random.Random(a.seed + 1)
            man_i = scratch / f"{arm}__{sub.replace(':','_')}__img.tsv"
            with open(man_i, "w") as f:
                f.write("POINT\t*\n")
                for b in range(a.boot):
                    pick = []
                    for _ in range(len(imgs)):
                        pick.extend(by_img[imgs[rng2.randrange(len(imgs))]])
                    f.write(f"B{b}\t" + ",".join(map(str, pick)) + "\n")

            per_scorer = {}
            for sc in scorers:
                rf = scratch / f"{arm}__{sub.replace(':','_')}__{sc.replace('/','_')}.tsv"
                with open(rf, "w") as f:
                    f.write("group\ts_left\ts_right\tchoice\tweight\n")
                    for k in keys:
                        e = counts[k]
                        sl, sr = score[e["L"]][sc], score[e["R"]][sc]
                        for ch in ("left", "right"):
                            if e[ch] > 0:
                                f.write(f"{k[1]}\t{sl}\t{sr}\t{ch}\t{e[ch]:g}\n")
                out = {}
                for tag, man in (("question", man_q), ("image", man_i)):
                    p = subprocess.run([a.panel_bin, "--pairwise", str(rf), "--resample", str(man)],
                                       capture_output=True, text=True)
                    if p.returncode != 0:
                        raise SystemExit(f"panel --pairwise failed:\n{p.stderr[-3000:]}")
                    lines = p.stdout.strip().split("\n")
                    hdr = lines[0].split("\t")
                    rows = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
                    out[tag] = rows
                per_scorer[sc] = out
                man_q_rows = out["question"]
                pt = man_q_rows[0]
                assert pt["label"] == "POINT"
            # ---- assemble the comparison rows ---------------------------
            base = per_scorer["peer_ssim2"]
            for sc in scorers:
                pt = per_scorer[sc]["question"][0]
                rec = {"arm": arm, "subset": sub, "scorer": sc,
                       "n_groups": int(pt["n_groups"]), "n_responses": float(pt["n_responses"]),
                       "acc": float(pt["acc_response"]), "tie_rate": float(pt["tie_rate"]),
                       "ceiling": float(pt["ceiling_response"]),
                       "acc_norm": float(pt["acc_norm"]),
                       "acc_group_majority": float(pt["acc_group_majority"]),
                       "n_groups_majority": int(pt["n_groups_majority"])}
                for tag in ("question", "image"):
                    d_pt = float(per_scorer[sc][tag][0]["acc_response"]) - float(base[tag][0]["acc_response"])
                    deltas = sorted(float(x["acc_response"]) - float(y["acc_response"])
                                    for x, y in zip(per_scorer[sc][tag][1:], base[tag][1:]))
                    n = len(deltas)
                    lo, hi = deltas[int(0.025 * n)], deltas[min(n - 1, int(0.975 * n))]
                    rec[f"d_ssim2_{tag}"] = d_pt
                    rec[f"d_ssim2_{tag}_lo"] = lo
                    rec[f"d_ssim2_{tag}_hi"] = hi
                    rec[f"d_ssim2_{tag}_p_gt0"] = sum(1 for v in deltas if v > 0) / n
                results.append(rec)
            print(f"{arm}/{sub}: {len(keys)} groups, "
                  f"ssim2 acc={float(base['question'][0]['acc_response']):.4f} "
                  f"ceiling={float(base['question'][0]['ceiling_response']):.4f}")

    outp = d / a.out_name
    keys = list(results[0])
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in results:
            w.writerow(r)
    (d / a.diag_name).write_text(json.dumps(diag, indent=2, sort_keys=True))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
