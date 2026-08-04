#!/usr/bin/env python3
"""cycle_audit.py — measure where an R&D campaign's wall-clock and tokens went.

THE OWNER for "why was that campaign slow / expensive". Run it after any
multi-wave day instead of rebuilding the analysis by hand; the first pass
(2026-08-04) took an hour of ad-hoc scripting, and the answer it produced --
benchmarks/rnd_cycle_audit_2026-08-04.md -- is the reason CLAUDE.md now has a
"Latency + token discipline" section and scripts/ has await_artifacts.sh +
harvest_bakes.sh.

Three measurements, three sub-commands:

  tokens   Per-turn token spend from the Claude Code session transcripts, and
           specifically the cost of IDLE WAITING. Prompt-cache entries are
           `ephemeral_5m`: read at 0.1x base, written at 1.25x. So any gap over
           five minutes converts the whole cached prefix from a 0.1x read into
           a 1.25x write -- a 12.5x multiplier on 500-800k tokens. This prices
           exactly that. (2026-08-03/04: $395.24, 13.9% of the session.)

  idle     Whole-session idle windows (no assistant turn from ANY agent),
           each split into compute-bound vs DEAD -- dead being time after the
           last campaign artifact write, i.e. finished work nobody looked at.
           (2026-08-03/04: 14.80 h idle, 6.77 h dead.)

  builds   Total `cargo` wall-clock across every wave log. Answers "would a
           shared build cache help?" with a number. (2026-08-03/04: 23.0 min
           all day -- it would not.)

Usage:
    python3 scripts/cycle_audit.py tokens --since 2026-08-03
    python3 scripts/cycle_audit.py idle   --since 2026-08-03 --waves wave4,wave5
    python3 scripts/cycle_audit.py builds --since 2026-08-03

`--waves` scopes the artifact stream. It matters: ~/tmp is shared with
unrelated projects, and an unscoped first cut of the idle table credited an
`fkdlocal_*.pth` training run as zensim compute, halving the apparent dead
time. Scope, or the answer is wrong in the flattering direction.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import defaultdict

UTC = dt.timezone.utc
TRANSCRIPTS = os.path.expanduser("~/.claude/projects/-home-lilith-work-zen-zensim")
DEFAULT_WAVES = (
    "sota944,cohend,wave4,wave5,wave6,wave7,balanced,shippack,hygiene,"
    "contrib,echarts,consolidate"
)
CAMPAIGN_GLOBS = (
    "/mnt/v/output/zensim/bakes/sota944/bakes/*",
    "/mnt/v/output/zensim/bakes/sota944/verdicts/*",
    "/mnt/v/output/zensim/reports/fulleval/*.json",
)
# Opus list price, USD per token. cache write (5m) = 1.25x base, read = 0.1x.
P_IN, P_WRITE, P_READ, P_OUT = 15 / 1e6, 18.75 / 1e6, 1.50 / 1e6, 75 / 1e6
CACHE_TTL_S = 300


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def transcript_files() -> list[str]:
    return glob.glob(os.path.join(TRANSCRIPTS, "*.jsonl")) + glob.glob(
        os.path.join(TRANSCRIPTS, "*", "subagents", "*.jsonl")
    )


def read_turns(path: str, since: dt.datetime) -> list[dict]:
    """Assistant turns with a usage block, deduped by requestId.

    Streaming emits several assistant events per request; counting them all
    would multiply every token figure.
    """
    out, seen = [], set()
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "assistant" or not d.get("timestamp"):
                continue
            t = parse_ts(d["timestamp"])
            if t < since:
                continue
            msg = d.get("message") or {}
            usage = msg.get("usage") or {}
            if not usage:
                continue
            rid = d.get("requestId")
            if rid and rid in seen:
                continue
            if rid:
                seen.add(rid)
            tools = []
            for blk in msg.get("content") or []:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    inp = blk.get("input") or {}
                    tools.append((blk.get("name", "?"), str(inp.get("command", ""))[:400]))
            out.append(
                dict(
                    ts=t,
                    inp=usage.get("input_tokens", 0),
                    cc=usage.get("cache_creation_input_tokens", 0),
                    cr=usage.get("cache_read_input_tokens", 0),
                    out=usage.get("output_tokens", 0),
                    tools=tools,
                )
            )
    return out


def cmd_tokens(args) -> int:
    since = parse_ts(args.since + "T00:00:00Z")
    per_agent, rows = {}, []
    for f in transcript_files():
        turns = read_turns(f, since)
        if not turns:
            continue
        name = os.path.basename(f).replace(".jsonl", "")
        per_agent[name] = turns
        for i, r in enumerate(turns):
            r["gap"] = (r["ts"] - turns[i - 1]["ts"]).total_seconds() if i else 0.0
            r["agent"] = name
            rows.append(r)
    if not rows:
        print("no turns in range", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["ts"])

    def cost(rs):
        return sum(
            r["cc"] * P_WRITE + r["cr"] * P_READ + r["out"] * P_OUT + r["inp"] * P_IN
            for r in rs
        )

    tot_cc = sum(r["cc"] for r in rows)
    tot_cr = sum(r["cr"] for r in rows)
    print(f"turns={len(rows)}  agents={len(per_agent)}  span_from={args.since}")
    print(f"cache_create={tot_cc:,}  cache_read={tot_cr:,}  "
          f"output={sum(r['out'] for r in rows):,}")
    print(f"TOTAL COST (opus list): ${cost(rows):,.2f}\n")

    print(f"=== cache re-creation by idle gap ({CACHE_TTL_S}s TTL) ===")
    bands = [(0, CACHE_TTL_S), (CACHE_TTL_S, 600), (600, 1800), (1800, 3600), (3600, 10**9)]
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["gap"] < hi and r["gap"] > 0]
        lbl = f"{lo}-{hi}s" if hi < 10**9 else f">{lo}s"
        print(f"{lbl:>14} turns={len(sel):>5} cache_create={sum(r['cc'] for r in sel):>12,}")
    stale = [r for r in rows if r["gap"] > CACHE_TTL_S]
    recharged = sum(r["cc"] for r in stale)
    waste = recharged * (P_WRITE - P_READ)
    pct_cc = 100 * recharged / tot_cc if tot_cc else 0.0
    print(f"\nRE-CHARGED prefix (gap>{CACHE_TTL_S}s): {recharged:,} tokens over "
          f"{len(stale)} turns ({100*len(stale)/len(rows):.1f}% of turns, "
          f"{pct_cc:.1f}% of all cache writes)")
    print(f"  billed as writes ${recharged*P_WRITE:,.2f}  vs as reads "
          f"${recharged*P_READ:,.2f}")
    print(f"  >>> IDLE-EXPIRY WASTE = ${waste:,.2f} "
          f"({100*waste/cost(rows):.1f}% of session cost)")

    print("\n=== per agent (by billed prefix) ===")
    print(f"{'agent':<46}{'turns':>6}{'cc':>13}{'cr':>14}{'span':>8}{'idle>5m':>9}")
    stats = []
    for name, turns in per_agent.items():
        span = (turns[-1]["ts"] - turns[0]["ts"]).total_seconds() / 3600
        idle = sum(r.get("gap", 0) for r in turns if r.get("gap", 0) > CACHE_TTL_S) / 3600
        cc = sum(r["cc"] for r in turns)
        cr = sum(r["cr"] for r in turns)
        stats.append((cc + cr, name, len(turns), cc, cr, span, idle))
    for _, name, n, cc, cr, span, idle in sorted(stats, reverse=True)[: args.top]:
        print(f"{name[:46]:<46}{n:>6}{cc:>13,}{cr:>14,}{span:>7.2f}h{idle:>8.2f}h")

    print("\n=== costliest post-idle turns ===")
    for r in sorted(stale, key=lambda r: -r["cc"])[: args.top]:
        tl = ",".join(t[0] for t in r["tools"]) or "(text)"
        print(f"{r['ts'].strftime('%m-%d %H:%M:%SZ')} gap={r['gap']/60:6.1f}m "
              f"cc={r['cc']:>9,} cr={r['cr']:>9,} {tl[:24]:<24} {r['agent'][:32]}")
    return 0


def cmd_idle(args) -> int:
    since = parse_ts(args.since + "T00:00:00Z")
    awake = set()
    for f in transcript_files():
        with open(f, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") == "assistant" and d.get("timestamp"):
                    t = parse_ts(d["timestamp"])
                    if t >= since:
                        awake.add(t)
    awake = sorted(awake)
    if len(awake) < 2:
        print("not enough turns in range", file=sys.stderr)
        return 1

    art = []

    def add(path, tag):
        try:
            st = os.stat(path)
        except OSError:
            return
        t = dt.datetime.fromtimestamp(st.st_mtime, UTC)
        if t >= since:
            art.append((t, tag, os.path.basename(path)))

    for pat in CAMPAIGN_GLOBS:
        for p in glob.glob(pat):
            if os.path.isfile(p):
                add(p, "campaign")
    for w in args.waves.split(","):
        w = w.strip()
        if not w:
            continue
        for p in glob.glob(os.path.expanduser(f"~/tmp/{w}/*")):
            if os.path.isfile(p):
                add(p, "wavelog")
    for spec in args.remote or []:
        ts, _, label = spec.partition("=")
        art.append((parse_ts(ts), "remote", label or "remote-artifact"))
    art.sort()

    print(f"artifacts={len(art)}  agent_turns={len(awake)}  "
          f"min_window={args.min_minutes}m")
    print(f"\n{'window (UTC)':>28}{'total':>9}{'compute':>9}{'DEAD':>9}  last campaign artifact")
    tot = dead = comp = 0.0
    for a, b in zip(awake, awake[1:]):
        gap = (b - a).total_seconds()
        if gap <= args.min_minutes * 60:
            continue
        inside = [x for x in art if a < x[0] < b]
        if inside:
            last = inside[-1]
            d = (b - last[0]).total_seconds()
            note = f"{last[0].strftime('%H:%M:%SZ')} {last[1]}:{last[2][:32]}"
        else:
            d, last, note = gap, None, "(nothing was computing)"
        c = gap - d
        tot += gap
        dead += d
        comp += c
        print(f"{a.strftime('%m-%d %H:%M:%SZ')} -> {b.strftime('%H:%M:%SZ'):>9}"
              f"{gap/60:8.1f}m{c/60:8.1f}m{d/60:8.1f}m  {note}")
    print(f"\nTOTALS idle={tot/3600:.2f}h  compute-bound={comp/3600:.2f}h  "
          f"DEAD={dead/3600:.2f}h ({100*dead/tot if tot else 0:.0f}% of idle)")
    return 0


FINISHED_RE = re.compile(r"Finished `?(?:release|dev)`? profile.*?in (?:(\d+)m )?([\d.]+)s")


def cmd_builds(args) -> int:
    waves = {w.strip() for w in args.waves.split(",") if w.strip()}
    total, n, per = 0.0, 0, defaultdict(float)
    for f in glob.glob(os.path.expanduser("~/tmp/*/*.log")):
        wave = os.path.basename(os.path.dirname(f))
        if wave not in waves:
            continue
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        for m in FINISHED_RE.finditer(text):
            secs = int(m.group(1) or 0) * 60 + float(m.group(2))
            total += secs
            per[wave] += secs
            n += 1
    for wave, secs in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"{wave:<14}{secs:8.0f}s")
    print(f"\nTOTAL cargo wall-clock: {n} builds, {total:.0f}s = {total/60:.1f} min")
    print("Compare against the lock cost of a SHARED CARGO_TARGET_DIR before "
          "proposing one:\n  a second concurrent `cargo` blocks on the build "
          "lock (measured 31.8s, 2026-08-04).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tokens", "idle", "builds"):
        p = sub.add_parser(name)
        p.add_argument("--since", default=dt.datetime.now(UTC).strftime("%Y-%m-%d"),
                       help="YYYY-MM-DD (UTC), inclusive")
        p.add_argument("--waves", default=DEFAULT_WAVES,
                       help="comma-separated ~/tmp/<dir> scratch dirs to scope to")
        if name == "tokens":
            p.add_argument("--top", type=int, default=20)
        if name == "idle":
            p.add_argument("--min-minutes", type=float, default=20.0)
            p.add_argument("--remote", action="append",
                           help="ISO8601Z=label for a remote-lane finish time "
                                "(rsync -a preserves it; ls on the remote shows it)")
    args = ap.parse_args()
    return {"tokens": cmd_tokens, "idle": cmd_idle, "builds": cmd_builds}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
