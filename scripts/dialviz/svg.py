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


STATE_STYLE = {
    # state: (fill colour, icon, label)
    "pass": ("var(--good)", "✓", "pass"),
    "fail": ("var(--crit)", "✗", "fail"),
    "blocked": ("var(--warn)", "■", "blocked"),
    "rule_pending": ("var(--serious)", "◆", "rule pending"),
    "not_measured": ("var(--na)", "○", "not measured"),
    "forbidden": ("var(--na)", "⊘", "forbidden here"),
    "outstanding": ("var(--na)", "…", "outstanding"),
}


def _wrap(text: str, n: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def tile_board(groups, *, width=760, cols=4, tile_h=52, gap=6, head=22, title=""):
    """Readiness board: one tile per gate, grouped in labelled bands, coloured and iconed by state.

    groups: list of (band label, [dict(label, state, tip, href?)]). State is a STATE_STYLE key.
    """
    tile_w = (width - gap * (cols - 1)) / cols
    chars = max(10, int(tile_w / 6.4))
    y = 0.0
    parts = []
    for band, tiles in groups:
        parts.append(f'<text x="0" y="{y + 15:.1f}" style="font-weight:600;fill:var(--ink)">{esc(band)} '
                     f'<tspan style="fill:var(--muted);font-weight:400">({len(tiles)})</tspan></text>')
        y += head
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            x0, y0 = c * (tile_w + gap), y + r * (tile_h + gap)
            fill, icon, slabel = STATE_STYLE[t["state"]]
            lines = _wrap(t["label"], chars)[:2]
            if len(_wrap(t["label"], chars)) > 2:
                lines[-1] = lines[-1][: chars - 1] + "…"
            body = (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{tile_w:.1f}" height="{tile_h}" rx="6" fill="{fill}" '
                    f'fill-opacity="0.16" stroke="{fill}" stroke-width="1.5"/>'
                    f'<rect x="{x0:.1f}" y="{y0:.1f}" width="5" height="{tile_h}" rx="2" fill="{fill}"/>'
                    f'<text x="{x0 + 12:.1f}" y="{y0 + 16:.1f}" style="font-weight:600;fill:var(--ink)">{esc(icon)} {esc(slabel)}</text>')
            for k, ln in enumerate(lines):
                body += f'<text x="{x0 + 12:.1f}" y="{y0 + 31 + k * 13:.1f}" style="fill:var(--ink-2)">{esc(ln)}</text>'
            g = f'<g data-tip="{esc(t.get("tip") or t["label"])}">{body}</g>'
            if t.get("href"):
                g = f'<a href="{esc(t["href"])}">{g}</a>'
            parts.append(g)
        rows = (len(tiles) + cols - 1) // cols
        y += rows * (tile_h + gap) + 8
    return (f'<svg class="chart" viewBox="0 0 {width} {y:.0f}" style="max-width:{width}px" role="img" '
            f'aria-label="{esc(title)}">' + "".join(parts) + "</svg>")


def bullets(rows, *, models, width=760, row=40, label_w=220, right_w=150, title="", bar_text=True):
    """Bullet rows: each check on its own axis, the pass side shaded, the bar as a line, one marker per model.

    rows: list of dict(label, bar, dir ('le'|'ge'), unit, values {model: v}, states {model: state}, tip?)
    models: list of (key, colour, short label, shape) with shape in circle|diamond|square.
    """
    plot_x0, plot_x1 = label_w, width - right_w
    h = row * len(rows) + 8
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    for i, r in enumerate(rows):
        y = 4 + i * row
        cy = y + row / 2
        vals = list(r["values"].values()) + [r["bar"]]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or abs(hi) or 1.0
        lo, hi = lo - span * 0.25, hi + span * 0.25
        if min(vals) >= 0 and lo < 0:
            lo = 0.0
        sx = lambda v: plot_x0 + (v - lo) / (hi - lo) * (plot_x1 - plot_x0)
        bx = sx(r["bar"])
        good_x0, good_x1 = (plot_x0, bx) if r["dir"] == "le" else (bx, plot_x1)
        bad_x0, bad_x1 = (bx, plot_x1) if r["dir"] == "le" else (plot_x0, bx)
        lab = _wrap(r["label"], max(8, int((label_w - 10) / 6.2)))[:2]
        for k, ln in enumerate(lab):
            ly = cy + 4 + (k - (len(lab) - 1) / 2) * 12
            out.append(f'<text x="{label_w - 10}" y="{ly:.1f}" text-anchor="end">{esc(ln)}</text>')
        out.append(f'<rect x="{good_x0:.1f}" y="{cy - 9:.1f}" width="{max(good_x1 - good_x0, 0):.1f}" height="18" '
                   f'fill="var(--good)" fill-opacity="0.13"/>')
        out.append(f'<rect x="{bad_x0:.1f}" y="{cy - 9:.1f}" width="{max(bad_x1 - bad_x0, 0):.1f}" height="18" '
                   f'fill="var(--crit)" fill-opacity="0.10"/>')
        out.append(f'<line class="thr" x1="{bx:.1f}" x2="{bx:.1f}" y1="{cy - 13:.1f}" y2="{cy + 13:.1f}">'
                   f'<title>bar {esc(_fmt(r["bar"]))}{esc(r["unit"])}</title></line>')
        out.append(f'<text x="{plot_x0:.1f}" y="{cy + 20:.1f}" style="fill:var(--muted);font-size:10px">{esc(_fmt(lo))}</text>')
        out.append(f'<text x="{plot_x1:.1f}" y="{cy + 20:.1f}" text-anchor="end" style="fill:var(--muted);font-size:10px">'
                   f'{esc(_fmt(hi))}</text>')
        for k, (mk, color, short, shape) in enumerate(models):
            v = r["values"][mk]
            x, yy = sx(v), cy - 5 + k * 5
            tip = f'{r["label"]}\n{short}: {_fmt(v)}{r["unit"]} ({r["states"][mk]})\nbar {"≤" if r["dir"] == "le" else "≥"} {_fmt(r["bar"])}{r["unit"]}'
            if shape == "diamond":
                m = f'<path d="M{x:.1f},{yy - 5:.1f} L{x + 5:.1f},{yy:.1f} L{x:.1f},{yy + 5:.1f} L{x - 5:.1f},{yy:.1f} Z"'
            elif shape == "square":
                m = f'<rect x="{x - 4:.1f}" y="{yy - 4:.1f}" width="8" height="8"'
            else:
                m = f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="4.5"'
            out.append(f'{m} fill="{color}" stroke="var(--surface)" stroke-width="1.5" data-tip="{esc(tip)}"/>')
        a = models[0][0]
        st = r["states"][a]
        out.append(f'<text x="{plot_x1 + 10:.1f}" y="{cy + 4:.1f}" style="fill:var(--ink)">'
                   f'{"✓" if st == "pass" else "✗"} {esc(_fmt(r["values"][a]))}{esc(r["unit"])} '
                   + (f'<tspan style="fill:var(--muted)">{"≤" if r["dir"] == "le" else "≥"} {esc(_fmt(r["bar"]))}</tspan>' if bar_text else "")
                   + '</text>')
    out.append("</svg>")
    return "".join(out)


def bipartite(left, right, edges, *, width=760, row=30, title="", lx=300, rx=470, label_chars=40):
    """Gates on the left, root causes on the right, one curve per (gate, cause).

    left: list of dict(id, label, state, tip?, href?); right: list of dict(id, label, color);
    edges: list of (left id, right id, tip).
    """
    h = max(len(left), len(right)) * row + 10
    ly = {n["id"]: 8 + i * row + row / 2 for i, n in enumerate(left)}
    step = (h - 10) / max(len(right), 1)
    ry = {n["id"]: 5 + i * step + step / 2 for i, n in enumerate(right)}
    color = {n["id"]: n["color"] for n in right}
    count = {n["id"]: 0 for n in right}
    for _, rid, _ in edges:
        count[rid] += 1
    out = [f'<svg class="chart" viewBox="0 0 {width} {h:.0f}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    for lid, rid, tip in edges:
        y0, y1 = ly[lid], ry[rid]
        bend = (rx - lx) / 2
        out.append(f'<path d="M{lx:.1f},{y0:.1f} C{lx + bend:.1f},{y0:.1f} {rx - bend:.1f},{y1:.1f} {rx:.1f},{y1:.1f}" '
                   f'fill="none" stroke="{color[rid]}" stroke-width="2" stroke-opacity="0.75" data-tip="{esc(tip)}"/>')
    for n in left:
        y = ly[n["id"]]
        fill, icon, slabel = STATE_STYLE[n["state"]]
        lab = n["label"] if len(n["label"]) <= label_chars else n["label"][: label_chars - 1] + "…"
        t = f'<text x="{lx - 14}" y="{y + 4:.1f}" text-anchor="end" data-tip="{esc(n.get("tip") or n["label"])}">{esc(lab)}</text>'
        if n.get("href"):
            t = f'<a href="{esc(n["href"])}">{t}</a>'
        out.append(t)
        out.append(f'<circle cx="{lx - 5}" cy="{y:.1f}" r="4.5" fill="{fill}"><title>{esc(slabel)}</title></circle>')
    for n in right:
        y = ry[n["id"]]
        out.append(f'<rect x="{rx}" y="{y - 11:.1f}" width="8" height="22" rx="3" fill="{n["color"]}"/>')
        out.append(f'<text x="{rx + 16}" y="{y + 4:.1f}" style="fill:var(--ink);font-weight:600">{esc(n["label"])} '
                   f'<tspan style="fill:var(--muted);font-weight:400">{count[n["id"]]}</tspan></text>')
    out.append("</svg>")
    return "".join(out)


def state_grid(rows, cols, *, width=760, row=22, label_w=None, cell_w=92, title=""):
    """A compact gate x model grid: one coloured, iconed cell per (row, column).

    rows: list of dict(label, cells {col key: dict(state, tip)}, href?); cols: list of (key, label).
    """
    label_w = label_w or width - cell_w * len(cols) - 8
    h = 26 + row * len(rows)
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" style="max-width:{width}px" role="img" aria-label="{esc(title)}">']
    for j, (_, cl) in enumerate(cols):
        out.append(f'<text x="{label_w + j * cell_w + cell_w / 2:.1f}" y="14" text-anchor="middle" '
                   f'style="font-weight:600;fill:var(--ink)">{esc(cl)}</text>')
    chars = int(label_w / 6.2)
    for i, r in enumerate(rows):
        y = 22 + i * row
        lab = r["label"] if len(r["label"]) <= chars else r["label"][: chars - 1] + "…"
        t = f'<text x="{label_w - 8}" y="{y + row / 2 + 4:.1f}" text-anchor="end" data-tip="{esc(r["label"])}">{esc(lab)}</text>'
        out.append(f'<a href="{esc(r["href"])}">{t}</a>' if r.get("href") else t)
        for j, (ck, _) in enumerate(cols):
            c = r["cells"][ck]
            fill, icon, slabel = STATE_STYLE[c["state"]]
            x = label_w + j * cell_w
            out.append(f'<g data-tip="{esc(c.get("tip") or slabel)}"><rect x="{x + 3:.1f}" y="{y + 2:.1f}" width="{cell_w - 6}" '
                       f'height="{row - 4}" rx="4" fill="{fill}" fill-opacity="0.18" stroke="{fill}"/>'
                       f'<text x="{x + cell_w / 2:.1f}" y="{y + row / 2 + 4:.1f}" text-anchor="middle" style="fill:var(--ink)">'
                       f'{esc(icon)} {esc(slabel)}</text></g>')
    out.append("</svg>")
    return "".join(out)
