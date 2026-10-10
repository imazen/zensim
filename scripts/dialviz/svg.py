"""Inline SVG charts drawn to scale from extracted values.

Marks follow the dataviz reference: thin bars with rounded data ends, 2px
lines, recessive grid, direct labels, per-mark tooltips (`data-tip`), one axis.
Every function takes values that a reader extracted; none derives a statistic.
"""
from __future__ import annotations

import math

from .htmlkit import esc

STATUS_FILL = {"pass": "var(--good)", "fail": "var(--crit)", "blocked": "var(--warn)", "ready": "var(--accent)",
               "done": "var(--good)", "na": "var(--na)", "pending": "var(--na)", "open": "var(--crit)",
               "fixed": "var(--good)", "info": "var(--na)"}


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 12))
        v += step
    return ticks


def _fmt(v: float, digits: int = 4) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100 or v == int(v) and abs(v) < 1e6:
        return f"{v:g}"
    return f"{v:.{digits}g}"


def hbars(items, *, xmin=None, xmax=None, thresholds=(), width=760, row=22, label_w=170, unit="", title=""):
    """Horizontal bars from a common zero/xmin baseline.

    items: list of dict(label, value, tip?, color?)  (value None -> 'not measured' row)
    thresholds: list of (value, label) drawn as dashed verticals (rule bars).
    """
    vals = [i["value"] for i in items if i.get("value") is not None] + [t[0] for t in thresholds]
    lo = min([0.0] + vals) if xmin is None else xmin
    hi = max([0.0] + vals) if xmax is None else xmax
    if hi == lo:
        hi = lo + 1
    plot_w = width - label_w - 56
    top = 18 + 11 * max(0, len(thresholds) - 1)
    h = top + row * len(items) + 26
    sx = lambda v: label_w + (v - lo) / (hi - lo) * plot_w
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    for t in _nice_ticks(lo, hi):
        x = sx(t)
        out.append(f'<line class="gridl" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{h - 22}"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle">{esc(_fmt(t))}{esc(unit)}</text>')
    zx = sx(0) if lo <= 0 <= hi else sx(lo)
    out.append(f'<line class="axis" x1="{zx:.1f}" x2="{zx:.1f}" y1="{top - 4}" y2="{h - 22}"/>')
    for i, it in enumerate(items):
        y = top + i * row
        out.append(f'<text x="{label_w - 8}" y="{y + row / 2 + 4:.1f}" text-anchor="end">{esc(it["label"])}</text>')
        v = it.get("value")
        tip = it.get("tip") or f'{it["label"]}\n{_fmt(v) if v is not None else "not measured"}{unit}'
        if v is None:
            out.append(f'<text x="{zx + 6:.1f}" y="{y + row / 2 + 4:.1f}" style="fill:var(--muted)">not measured</text>')
            continue
        x0, x1 = sorted((zx, sx(v)))
        color = it.get("color", "var(--s1)")
        out.append(f'<rect class="bar" x="{x0:.1f}" y="{y + 4}" width="{max(x1 - x0, 1.5):.1f}" height="{row - 8}" '
                   f'fill="{color}" data-tip="{esc(tip)}"/>')
        tx = x1 + 5 if v >= 0 or x0 == zx else x0 - 5
        anchor = "start" if (v >= 0 or x0 == zx) else "end"
        out.append(f'<text x="{tx:.1f}" y="{y + row / 2 + 4:.1f}" text-anchor="{anchor}">{esc(_fmt(v))}</text>')
    for k, (tv, tl) in enumerate(sorted(thresholds)):
        x = sx(tv)
        ly = top - 8 - 11 * (len(thresholds) - 1 - k)
        # rightmost threshold labels to the right of its line, the others to the left
        anchor = "start" if k == len(thresholds) - 1 else "end"
        tx = x - 3 if anchor == "end" else x + 3
        out.append(f'<line class="thr" x1="{x:.1f}" x2="{x:.1f}" y1="{ly + 2}" y2="{h - 22}"><title>{esc(tl)}</title></line>')
        out.append(f'<text x="{tx:.1f}" y="{ly}" text-anchor="{anchor}" style="fill:var(--ink)">{esc(tl)}</text>')
    out.append("</svg>")
    return "".join(out)


def dots_ci(items, *, thresholds=(), width=760, row=24, label_w=150, title="", xmin=None, xmax=None):
    """Point estimates with ±k·SE whiskers against rule thresholds.

    items: list of dict(label, value, lo?, hi?, tip?, state?)
    """
    pts = []
    for i in items:
        for k in ("value", "lo", "hi"):
            if i.get(k) is not None:
                pts.append(i[k])
    pts += [t[0] for t in thresholds] + [0.0]
    lo = min(pts) if xmin is None else xmin
    hi = max(pts) if xmax is None else xmax
    if xmin is None or xmax is None:
        pad = (hi - lo) * 0.06 or 0.001
        lo, hi = (lo - pad if xmin is None else lo), (hi + pad if xmax is None else hi)
    plot_w = width - label_w - 20
    top = 22
    h = top + row * len(items) + 26
    sx = lambda v: label_w + (v - lo) / (hi - lo) * plot_w
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    for t in _nice_ticks(lo, hi):
        x = sx(t)
        out.append(f'<line class="gridl" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{h - 22}"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle">{esc(_fmt(t, 3))}</text>')
    out.append(f'<line class="zero" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top - 4}" y2="{h - 22}"/>')
    for k, (tv, tl) in enumerate(thresholds):
        x = sx(tv)
        out.append(f'<line class="thr" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 10}" y2="{h - 22}"/>')
        anchor = "end" if x > width - 90 else "start"
        out.append(f'<text x="{x - 3 if anchor == "end" else x + 3:.1f}" y="{top - 12}" text-anchor="{anchor}" style="fill:var(--ink)">{esc(tl)}</text>')
    for i, it in enumerate(items):
        y = top + i * row + row / 2
        out.append(f'<text x="{label_w - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(it["label"])}</text>')
        v = it.get("value")
        if v is None:
            out.append(f'<text x="{label_w + 4}" y="{y + 4:.1f}" style="fill:var(--muted)">not measured</text>')
            continue
        if it.get("lo") is not None and it.get("hi") is not None and it["hi"] >= lo and it["lo"] <= hi:
            out.append(f'<line x1="{sx(max(it["lo"], lo)):.1f}" x2="{sx(min(it["hi"], hi)):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                       f'stroke="var(--ink-2)" stroke-width="2"/>')
        fill = STATUS_FILL.get(it.get("state"), "var(--s1)")
        tip = it.get("tip") or f'{it["label"]}\n{_fmt(v)}'
        if v < lo or v > hi:
            # off-scale: a labelled arrow at the axis edge instead of stretching the axis
            edge = label_w + 2 if v < lo else label_w + plot_w - 2
            arrow = "◀" if v < lo else "▶"
            anchor = "start" if v < lo else "end"
            out.append(f'<text x="{edge:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" style="fill:var(--ink);font-weight:600" '
                       f'data-tip="{esc(tip)}">{arrow} {esc(_fmt(v, 3))} (off scale)</text>')
            continue
        out.append(f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="var(--surface)" stroke-width="2" '
                   f'data-tip="{esc(tip)}"/>')
        out.append(f'<circle class="hit" cx="{sx(v):.1f}" cy="{y:.1f}" r="11" data-tip="{esc(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def status_strip(counts: dict, *, width=640, height=26, title=""):
    """One stacked bar of state counts (pass/fail/blocked/...), labelled below via legend."""
    total = sum(counts.values()) or 1
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    x = 0.0
    for state, n in counts.items():
        if not n:
            continue
        w = n / total * width
        out.append(f'<rect x="{x + 1:.1f}" y="2" width="{max(w - 2, 1):.1f}" height="{height - 4}" rx="4" '
                   f'fill="{STATUS_FILL.get(state, "var(--na)")}" data-tip="{esc(state)}\n{n} of {total}"/>')
        if w > 28:
            out.append(f'<text x="{x + w / 2:.1f}" y="{height / 2 + 4:.1f}" text-anchor="middle" '
                       f'style="fill:#0b0b0b;font-weight:600">{n}</text>')
        x += w
    out.append("</svg>")
    return "".join(out)


def id_layout(ids_on: set[int], total: int, *, width=640, cols=None, cell=None, title="", tips=None):
    """A to-scale map of feature IDs 0..total-1, highlighting members of a set."""
    cols = cols or 64
    cell = cell or max(4.0, (width - 40) / cols)
    rows = math.ceil(total / cols)
    h = rows * cell + 4
    out = [f'<svg class="chart" viewBox="0 0 {40 + cols * cell:.0f} {h:.0f}" style="max-width:{40 + cols * cell:.0f}px" role="img" aria-label="{esc(title)}">']
    for r in range(rows):
        if r % 2 == 0:
            out.append(f'<text x="34" y="{r * cell + cell - 1:.1f}" text-anchor="end">{r * cols}</text>')
    for i in range(total):
        r, c = divmod(i, cols)
        on = i in ids_on
        tip = tips.get(i) if tips else None
        t = f' data-tip="{esc(tip)}"' if tip else ""
        out.append(f'<rect x="{40 + c * cell:.1f}" y="{r * cell:.1f}" width="{cell - 1:.1f}" height="{cell - 1:.1f}" '
                   f'fill="{"var(--s1)" if on else "var(--grid)"}"{t}/>')
    out.append("</svg>")
    return "".join(out)


def legend(entries):
    """entries: list of (css color, label)."""
    return '<div class="bar-legend">' + "".join(
        f'<span><span class="swatch" style="background:{c}"></span>{esc(l)}</span>' for c, l in entries) + "</div>"


def strips(rows, *, width=760, row=26, label_w=120, title="", xmin=None, xmax=None):
    """One row of points per group (every recorded value drawn, no sampling).

    rows: list of (label, values, css color)
    """
    vals = [v for _, vs, _ in rows for v in vs]
    lo = min(vals) if xmin is None else xmin
    hi = max(vals) if xmax is None else xmax
    if hi == lo:
        hi = lo + 1e-6
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    plot_w = width - label_w - 20
    top = 8
    h = top + row * len(rows) + 26
    sx = lambda v: label_w + (v - lo) / (hi - lo) * plot_w
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    for t in _nice_ticks(lo, hi):
        x = sx(t)
        out.append(f'<line class="gridl" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{h - 22}"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle">{esc(_fmt(t, 3))}</text>')
    for i, (label, vs, color) in enumerate(rows):
        y = top + i * row + row / 2
        out.append(f'<text x="{label_w - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        for v in vs:
            out.append(f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="3.5" fill="{color}" fill-opacity="0.55" '
                       f'data-tip="{esc(label)}\n{v:.5f}"/>')
    out.append("</svg>")
    return "".join(out)
