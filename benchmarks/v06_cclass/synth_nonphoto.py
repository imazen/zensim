#!/usr/bin/env python3
"""Synthesize non-photo training references for V0_6 rebalance.

Generates ~1500-2000 reference images across 5 categories:
  - screenshots (UI mockups, ~500)
  - documents (text pages, ~500)
  - charts (matplotlib plots, ~300)
  - line-art (synthetic geometric, ~300)
  - mixed (text-on-photo overlays, ~200)

All content is procedurally generated → CC0. No copyrighted assets.
Output PNGs land in /mnt/v/input/zensim/sources/ with filename prefixes
that the cclass classifier can detect:
  gen-screen__<id>_<size>.png
  gen-doc__<id>_<size>.png
  gen-chart__<id>_<size>.png
  gen-line__<id>_<size>.png
  gen-mixed__<id>_<size>.png

Sizes: 512sq + 1024sq variants per source so that the safe-synthetic
6-bucket size augmentation pipeline downstream (used by
generate_zensim_training) has enough resolution to crop+resample.

Sidecar TSV:
  /mnt/v/output/zensim/v06-rebalance/synth_sources.tsv
with columns: source_path, content_class, subset, seed, license.
"""
from __future__ import annotations

import argparse
import os
import random
import string
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/mnt/v/input/zensim/sources")
SIDECAR = Path("/mnt/v/output/zensim/v06-rebalance/synth_sources.tsv")
LICENSE = "CC0-1.0"

# --- Font discovery -------------------------------------------------------

MONO_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]
SANS_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]
SERIF_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf",
]


def first_existing(paths: list[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise RuntimeError(f"no font found in {paths}")


MONO_FONT = first_existing(MONO_FONT_CANDIDATES)
SANS_FONT = first_existing(SANS_FONT_CANDIDATES)
SERIF_FONT = first_existing(SERIF_FONT_CANDIDATES)


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


# --- Lorem-style word list (no copyrighted text) --------------------------

WORDS_EN_COMMON = (
    "the of and to a in is it that for as with on this be by are have not but at "
    "from or had has was were they you we he she them their which when what about "
    "into other after before than over under between through during without these "
    "those some many much such where here there because while until against above "
    "below within across each every another however therefore moreover furthermore "
    "indeed example perhaps maybe always never sometimes often rarely usually "
    "system value memory buffer index offset length kernel thread process queue "
    "module function pointer struct vector matrix tensor packet stream pipeline "
    "decoder encoder filter quality lossy lossless bitrate entropy compression "
    "frequency wavelet transform residual coefficient quantization prediction "
    "reference codec format version commit branch release benchmark profile data"
).split()


def make_paragraph(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(WORDS_EN_COMMON) for _ in range(n_words)]
    text = " ".join(words)
    # Occasional capitalization, periods
    out = []
    sentence_len = 0
    for w in words:
        if sentence_len == 0:
            w = w.capitalize()
        out.append(w)
        sentence_len += 1
        if sentence_len >= rng.randint(6, 18):
            out[-1] = out[-1] + "."
            sentence_len = 0
    return " ".join(out)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    lines = []
    cur = []
    cur_w = 0
    space_w = font.getlength(" ")
    for w in words:
        ww = font.getlength(w)
        if cur and cur_w + space_w + ww > max_w:
            lines.append(" ".join(cur))
            cur = [w]
            cur_w = ww
        else:
            if cur:
                cur_w += space_w
            cur.append(w)
            cur_w += ww
    if cur:
        lines.append(" ".join(cur))
    return lines


# --- Generators -----------------------------------------------------------

def gen_screenshot(rng: random.Random, size: int) -> Image.Image:
    """Faux UI: titlebar, sidebar, content area, buttons."""
    bg_palettes = [
        ((245, 245, 245), (220, 220, 220), (40, 40, 40)),  # light
        ((30, 30, 35), (50, 50, 60), (235, 235, 240)),  # dark
        ((255, 255, 255), (205, 220, 255), (10, 10, 30)),  # blue-tint
        ((250, 248, 240), (220, 200, 170), (60, 40, 20)),  # warm
    ]
    bg, sidebar, fg = rng.choice(bg_palettes)
    accent = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))

    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)

    # Titlebar
    tb_h = max(24, size // 22)
    d.rectangle([0, 0, size, tb_h], fill=tuple(min(255, c + 10) for c in sidebar))
    # Traffic-light buttons
    for i, col in enumerate([(255, 100, 100), (255, 200, 60), (100, 220, 100)]):
        cx = 12 + i * (tb_h - 6)
        cy = tb_h // 2
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)
    # Title text
    f_title = load_font(SANS_FONT, max(11, tb_h // 2))
    d.text((tb_h * 4, tb_h // 4), rng.choice(["Document Editor", "Project Console", "Settings", "Mail Client", "Terminal", "Browser", "File Manager"]), fill=fg, font=f_title)

    # Sidebar
    sb_w = size // 5
    d.rectangle([0, tb_h, sb_w, size], fill=sidebar)
    f_side = load_font(SANS_FONT, max(11, size // 50))
    sidebar_items = ["Inbox", "Sent", "Drafts", "Spam", "Trash", "Archive", "Starred",
                     "Projects", "Notes", "Tasks", "Calendar", "Files"]
    rng.shuffle(sidebar_items)
    y = tb_h + 12
    for item in sidebar_items[: rng.randint(5, 10)]:
        if y > size - 30:
            break
        if rng.random() < 0.15:
            d.rectangle([4, y - 2, sb_w - 4, y + 22], fill=accent)
            color = (255, 255, 255)
        else:
            color = fg
        d.text((16, y), item, fill=color, font=f_side)
        y += 30

    # Toolbar in content area
    cx0 = sb_w + 8
    d.rectangle([cx0, tb_h, size, tb_h + tb_h], fill=tuple(min(255, c + 5) for c in bg))
    btn_w = (size - cx0 - 16) // 6
    for i in range(rng.randint(3, 6)):
        x = cx0 + 8 + i * (btn_w + 4)
        d.rectangle([x, tb_h + 6, x + btn_w, tb_h + tb_h - 6], outline=fg, width=1, fill=tuple(min(255, c + 3) for c in bg))
        f_btn = load_font(SANS_FONT, max(10, tb_h // 3))
        d.text((x + 8, tb_h + tb_h // 3), rng.choice(["New", "Open", "Save", "Edit", "View", "Run", "Build", "Send", "Reply", "Forward"]), fill=fg, font=f_btn)

    # Content text rows
    f_body = load_font(rng.choice([SANS_FONT, SERIF_FONT, MONO_FONT]), max(12, size // 45))
    para = make_paragraph(rng, 200)
    lines = wrap_text(para, f_body, size - cx0 - 16)
    y = tb_h * 2 + 16
    for ln in lines:
        if y > size - 40:
            break
        d.text((cx0 + 8, y), ln, fill=fg, font=f_body)
        y += int(f_body.size * 1.4)

        # Random blockquote-style boxes
        if rng.random() < 0.05:
            d.rectangle([cx0 + 8, y, size - 8, y + 40], outline=accent, width=2)
            y += 50

    # Status bar
    d.rectangle([0, size - 24, size, size], fill=sidebar)
    d.text((10, size - 22), rng.choice(["Ready", "Loading…", "Connected", "12 items", "All synced"]), fill=fg, font=load_font(SANS_FONT, 12))
    return img


def gen_document(rng: random.Random, size: int) -> Image.Image:
    """Document page with paragraphs of text, optional headings."""
    paper_palettes = [
        ((252, 250, 245), (10, 10, 10)),  # cream/black
        ((255, 255, 255), (15, 20, 30)),
        ((250, 240, 220), (40, 25, 10)),
        ((20, 22, 28), (220, 220, 220)),  # dark mode "ebook"
    ]
    bg, fg = rng.choice(paper_palettes)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    margin = max(40, size // 12)

    title_font = load_font(rng.choice([SERIF_FONT, SANS_FONT]), max(28, size // 18))
    body_font = load_font(rng.choice([SERIF_FONT, SANS_FONT, MONO_FONT]),
                          max(14, size // 38))
    heading_font = load_font(rng.choice([SERIF_FONT, SANS_FONT]), max(20, size // 28))

    # Title
    title = " ".join(rng.choice(WORDS_EN_COMMON).capitalize()
                     for _ in range(rng.randint(2, 6)))
    d.text((margin, margin), title, fill=fg, font=title_font)
    y = margin + int(title_font.size * 1.6)

    # Optional subtitle
    if rng.random() < 0.5:
        sub = make_paragraph(rng, rng.randint(8, 14))
        sub_font = load_font(SANS_FONT, max(12, size // 45))
        for ln in wrap_text(sub, sub_font, size - 2 * margin):
            d.text((margin, y), ln, fill=fg, font=sub_font)
            y += int(sub_font.size * 1.4)
        y += 16

    # Paragraphs with optional headings, code blocks, bullet lists
    while y < size - margin - body_font.size * 4:
        kind = rng.choices(["para", "heading", "code", "bullets"],
                           weights=[6, 1, 1, 1])[0]
        if kind == "heading":
            h = " ".join(rng.choice(WORDS_EN_COMMON).capitalize()
                         for _ in range(rng.randint(2, 5)))
            d.text((margin, y), h, fill=fg, font=heading_font)
            y += int(heading_font.size * 1.6)
        elif kind == "code":
            code_lines = [
                "fn " + ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10))) + "(x: i32) -> i32 {",
                "    let v = " + str(rng.randint(0, 99)) + " * x;",
                "    v + " + str(rng.randint(0, 99)) + "",
                "}",
            ]
            code_font = load_font(MONO_FONT, max(13, size // 42))
            box_h = int(code_font.size * 1.5) * len(code_lines) + 16
            d.rectangle([margin, y, size - margin, y + box_h],
                        fill=tuple(max(0, min(255, c + (-15 if sum(bg) > 400 else 25))) for c in bg))
            cy = y + 8
            for cl in code_lines:
                d.text((margin + 8, cy), cl, fill=fg, font=code_font)
                cy += int(code_font.size * 1.5)
            y += box_h + 12
        elif kind == "bullets":
            for _ in range(rng.randint(2, 5)):
                if y > size - margin - body_font.size * 2:
                    break
                bullet_text = "• " + make_paragraph(rng, rng.randint(6, 14))
                for ln in wrap_text(bullet_text, body_font, size - 2 * margin - 24)[:2]:
                    d.text((margin + 16, y), ln, fill=fg, font=body_font)
                    y += int(body_font.size * 1.35)
            y += 8
        else:
            para = make_paragraph(rng, rng.randint(40, 110))
            for ln in wrap_text(para, body_font, size - 2 * margin):
                if y > size - margin - body_font.size:
                    break
                d.text((margin, y), ln, fill=fg, font=body_font)
                y += int(body_font.size * 1.4)
            y += 12

    # Page number
    pg = str(rng.randint(1, 200))
    d.text((size // 2 - 8, size - margin // 2), pg, fill=fg, font=body_font)
    return img


def gen_chart(rng: random.Random, size: int) -> Image.Image:
    """Matplotlib chart at the requested resolution."""
    dpi = 100
    fig_in = size / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)

    chart_kind = rng.choice(["line", "bar", "scatter", "heatmap", "area", "stack"])
    n = rng.randint(8, 40)
    style = rng.choice(["seaborn-v0_8-whitegrid", "ggplot", "default", "seaborn-v0_8-darkgrid", "Solarize_Light2"])
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)

    if chart_kind == "line":
        for s in range(rng.randint(2, 5)):
            xs = np.arange(n)
            steps = np.array([rng.gauss(0, 1) for _ in range(n)])
            ys = np.cumsum(steps)
            ax.plot(xs, ys, marker=rng.choice(["o", "s", "^", "x", ".", None]),
                    label=f"series {s + 1}")
        ax.legend(loc="best")
    elif chart_kind == "bar":
        xs = np.arange(n)
        ys = np.array([rng.uniform(0.1, 1.0) for _ in range(n)])
        ax.bar(xs, ys, color=plt.cm.viridis(np.linspace(0, 1, n)))
    elif chart_kind == "scatter":
        xs = np.array([rng.gauss(0, 1) for _ in range(n * 5)])
        ys = np.array([rng.gauss(0, 1) for _ in range(n * 5)])
        ax.scatter(xs, ys, c=np.arange(n * 5), cmap="plasma", alpha=0.7)
    elif chart_kind == "heatmap":
        m = np.array([[rng.uniform(0, 1) for _ in range(n)] for _ in range(n)])
        ax.imshow(m, cmap=rng.choice(["viridis", "magma", "plasma", "inferno", "coolwarm"]),
                  aspect="auto")
    elif chart_kind == "area":
        xs = np.arange(n)
        steps = np.array([rng.gauss(0, 1) for _ in range(n)])
        ys = np.cumsum(steps)
        ax.fill_between(xs, ys, alpha=0.5)
        ax.plot(xs, ys)
    elif chart_kind == "stack":
        xs = np.arange(n)
        ys_list = [np.array([rng.uniform(0, 1) for _ in range(n)])
                   for _ in range(rng.randint(3, 6))]
        ax.stackplot(xs, *ys_list, labels=[f"s{i + 1}" for i in range(len(ys_list))])
        ax.legend(loc="best")

    # Random labels
    title = " ".join(rng.choice(WORDS_EN_COMMON).capitalize()
                     for _ in range(rng.randint(2, 5)))
    ax.set_title(title)
    ax.set_xlabel(rng.choice(WORDS_EN_COMMON).capitalize())
    ax.set_ylabel(rng.choice(WORDS_EN_COMMON).capitalize())

    fig.tight_layout()
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgb = arr[..., 1:4]
    plt.close(fig)
    img = Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS)
    return img


def gen_lineart(rng: random.Random, size: int) -> Image.Image:
    """Hard-edged synthetic patterns: tilings, strokes, technical drawings."""
    bg_choices = [(255, 255, 255), (250, 250, 248), (15, 18, 22), (240, 235, 225)]
    bg = rng.choice(bg_choices)
    fg = (255, 255, 255) if sum(bg) < 380 else (10, 10, 10)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)

    kind = rng.choice(["polygons", "tiling", "voronoi-ish", "concentric", "grid-pattern", "stars-burst"])

    if kind == "polygons":
        for _ in range(rng.randint(8, 40)):
            cx, cy = rng.randint(0, size), rng.randint(0, size)
            r = rng.randint(20, size // 3)
            sides = rng.randint(3, 9)
            phase = rng.uniform(0, 6.28)
            pts = []
            for k in range(sides):
                a = phase + 2 * 3.14159 * k / sides
                pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) if rng.random() < 0.4 else fg
            if rng.random() < 0.5:
                d.polygon(pts, outline=color, width=rng.randint(1, 4))
            else:
                d.polygon(pts, fill=color, outline=fg)
    elif kind == "tiling":
        cell = rng.randint(16, 64)
        for x in range(0, size, cell):
            for y in range(0, size, cell):
                shape = rng.choice(["box", "diag1", "diag2", "circle", "tri"])
                color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) if rng.random() < 0.3 else fg
                if shape == "box":
                    d.rectangle([x, y, x + cell - 2, y + cell - 2], outline=color, width=1)
                elif shape == "diag1":
                    d.line([x, y, x + cell, y + cell], fill=color, width=2)
                elif shape == "diag2":
                    d.line([x + cell, y, x, y + cell], fill=color, width=2)
                elif shape == "circle":
                    d.ellipse([x + 2, y + 2, x + cell - 2, y + cell - 2], outline=color, width=1)
                else:
                    d.polygon([(x + cell // 2, y), (x, y + cell), (x + cell, y + cell)], outline=color, width=1)
    elif kind == "voronoi-ish":
        # Random irregular triangulation (cheap voronoi feel)
        pts = [(rng.randint(0, size), rng.randint(0, size)) for _ in range(rng.randint(12, 50))]
        for i, p in enumerate(pts):
            for q in pts[i + 1: i + 4]:
                d.line([p, q], fill=fg, width=1)
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=fg)
    elif kind == "concentric":
        cx, cy = size // 2, size // 2
        for r in range(8, size // 2, rng.randint(6, 22)):
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) if rng.random() < 0.3 else fg
            shape = rng.choice(["ellipse", "rect", "polygon"])
            if shape == "ellipse":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=rng.randint(1, 3))
            elif shape == "rect":
                d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=color, width=rng.randint(1, 3))
            else:
                sides = 6
                phase = rng.uniform(0, 6.28)
                pts = [(cx + r * np.cos(phase + 2 * 3.14159 * k / sides),
                        cy + r * np.sin(phase + 2 * 3.14159 * k / sides))
                       for k in range(sides)]
                d.polygon(pts, outline=color, width=rng.randint(1, 3))
    elif kind == "grid-pattern":
        spacing = rng.randint(8, 32)
        for i in range(0, size, spacing):
            d.line([(i, 0), (i, size)], fill=fg, width=1)
            d.line([(0, i), (size, i)], fill=fg, width=1)
        # Highlight cells
        for _ in range(rng.randint(20, 80)):
            cx = rng.randint(0, size // spacing) * spacing
            cy = rng.randint(0, size // spacing) * spacing
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            d.rectangle([cx, cy, cx + spacing, cy + spacing], fill=color)
    else:  # stars-burst
        cx, cy = rng.randint(size // 4, 3 * size // 4), rng.randint(size // 4, 3 * size // 4)
        for k in range(rng.randint(40, 200)):
            ang = 2 * 3.14159 * k / rng.randint(40, 200)
            r = rng.randint(20, size // 2)
            x2 = cx + r * np.cos(ang)
            y2 = cy + r * np.sin(ang)
            d.line([(cx, cy), (x2, y2)], fill=fg, width=1)

    return img


def gen_mixed(rng: random.Random, size: int, photo_pool: list[Path]) -> Image.Image:
    """Photo with overlaid text or UI chrome."""
    if not photo_pool:
        # Fallback: noise base
        arr = np.random.randint(40, 200, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        p = rng.choice(photo_pool)
        try:
            img = Image.open(p).convert("RGB")
            # Random crop to square then resize
            w, h = img.size
            s = min(w, h)
            x0 = rng.randint(0, w - s)
            y0 = rng.randint(0, h - s)
            img = img.crop((x0, y0, x0 + s, y0 + s)).resize(
                (size, size), Image.Resampling.LANCZOS
            )
        except Exception:
            arr = np.random.randint(40, 200, (size, size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    style = rng.choice(["caption", "subtitle-strip", "watermark", "ui-chrome"])
    if style == "caption":
        f = load_font(SANS_FONT, max(20, size // 22))
        text = " ".join(rng.choice(WORDS_EN_COMMON).capitalize()
                        for _ in range(rng.randint(3, 7)))
        margin = size // 20
        # Box
        bbox = d.textbbox((margin, size - margin - f.size * 1.4), text, font=f)
        d.rectangle((bbox[0] - 8, bbox[1] - 4, bbox[2] + 8, bbox[3] + 4),
                    fill=(0, 0, 0, 160))
        d.text((margin, size - margin - f.size * 1.4), text, fill=(255, 255, 255, 255), font=f)
    elif style == "subtitle-strip":
        f = load_font(SANS_FONT, max(18, size // 26))
        strip_y = size - size // 9
        d.rectangle((0, strip_y, size, size), fill=(0, 0, 0, 180))
        text = make_paragraph(rng, rng.randint(8, 14))
        for line in wrap_text(text, f, size - 40)[:2]:
            d.text((20, strip_y + 10), line, fill=(255, 255, 255, 255), font=f)
            strip_y += int(f.size * 1.35)
    elif style == "watermark":
        f = load_font(SERIF_FONT, max(40, size // 8))
        mark = rng.choice(["DRAFT", "SAMPLE", "PREVIEW", "INTERNAL", "DO NOT COPY"])
        d.text((size // 4, size // 2), mark, fill=(255, 255, 255, 80), font=f)
    elif style == "ui-chrome":
        # Top bar + bottom dock
        d.rectangle((0, 0, size, max(28, size // 22)), fill=(20, 20, 25, 220))
        d.rectangle((0, size - max(40, size // 14), size, size), fill=(20, 20, 25, 220))
        f_small = load_font(SANS_FONT, max(12, size // 50))
        d.text((10, 6), rng.choice(["Camera", "Gallery", "Live", "Stream"]), fill=(255, 255, 255, 255), font=f_small)
        for i in range(5):
            cx = size // 6 + i * size // 6
            cy = size - size // 22
            d.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=(255, 255, 255, 180))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


# --- Driver ---------------------------------------------------------------

CATEGORY_TO_PREFIX = {
    "screen": "gen-screen",
    "document": "gen-doc",
    "chart": "gen-chart",
    "lineart": "gen-line",
    "mixed": "gen-mixed",
}

CATEGORY_TO_GEN = {
    "screen": gen_screenshot,
    "document": gen_document,
    "chart": gen_chart,
    "lineart": gen_lineart,
    # mixed handled specially due to photo_pool
}

DEFAULT_BUDGET = {
    "screen": 500,
    "document": 500,
    "chart": 300,
    "lineart": 300,
    "mixed": 200,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument("--screen", type=int, default=DEFAULT_BUDGET["screen"])
    ap.add_argument("--document", type=int, default=DEFAULT_BUDGET["document"])
    ap.add_argument("--chart", type=int, default=DEFAULT_BUDGET["chart"])
    ap.add_argument("--lineart", type=int, default=DEFAULT_BUDGET["lineart"])
    ap.add_argument("--mixed", type=int, default=DEFAULT_BUDGET["mixed"])
    ap.add_argument("--sizes", default="512,1024",
                    help="comma-separated PNG sizes to generate per source seed")
    ap.add_argument("--photo-pool-glob",
                    default="/mnt/v/input/zensim/sources/*1024sq.png",
                    help="glob for mixed-category photo bases")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"sizes per source: {sizes}")

    # Photo pool for mixed category
    import glob
    photo_pool = [Path(p) for p in glob.glob(args.photo_pool_glob)]
    rng.shuffle(photo_pool)
    print(f"mixed photo pool: {len(photo_pool)} images")

    budgets = {
        "screen": args.screen,
        "document": args.document,
        "chart": args.chart,
        "lineart": args.lineart,
        "mixed": args.mixed,
    }
    total = sum(budgets.values()) * len(sizes)
    print(f"total target images: {total}")

    rows = []  # (source_path, content_class, subset, seed, license)
    n_done = 0

    for cat, count in budgets.items():
        prefix = CATEGORY_TO_PREFIX[cat]
        for i in range(count):
            sub_seed = rng.randint(0, 2**31 - 1)
            sub_rng = random.Random(sub_seed)
            for sz in sizes:
                fname = f"{prefix}__{i:05d}_s{sub_seed:08x}_{sz}sq.png"
                out = OUT_DIR / fname
                if out.exists():
                    rows.append((str(out), cat, prefix, sub_seed, LICENSE))
                    continue
                try:
                    if cat == "mixed":
                        img = gen_mixed(sub_rng, sz, photo_pool)
                    else:
                        img = CATEGORY_TO_GEN[cat](sub_rng, sz)
                    img.save(out, optimize=True, compress_level=6)
                    rows.append((str(out), cat, prefix, sub_seed, LICENSE))
                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"  {n_done}/{total}: {fname}")
                except Exception as e:
                    print(f"  FAILED {cat}/{i} sz={sz}: {e}", file=sys.stderr)

    # Write sidecar TSV
    with SIDECAR.open("w") as f:
        f.write("source_path\tcontent_class\tsubset\tseed\tlicense\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote sidecar {SIDECAR} ({len(rows)} rows)")
    print(f"new images written: {n_done}")
    # Per-class summary
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("class totals (across all sizes):")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
