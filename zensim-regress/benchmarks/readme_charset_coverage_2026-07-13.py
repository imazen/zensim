#!/usr/bin/env python3
"""Coverage audit: which characters in real README.md files can
console-lean NOT render? Ranks misses by file count, splits by
in-DejaVu (addable) vs not-in-font (fallback policy needed)."""
import unicodedata
from collections import Counter
from fontTools.ttLib import TTFont

TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
cmap = set(TTFont(TTF).getBestCmap().keys())

def rng(*p):
    o = set()
    for a, b in p:
        o.update(range(a, b + 1))
    return o

EXTB_KEEP = rng((0x01CD, 0x01DC), (0x0218, 0x021B))
EXTADD_KEEP = rng((0x1E80, 0x1E85), (0x1E9E, 0x1E9E), (0x1EF2, 0x1EF3))
MATH_KEEP = {0x2202, 0x2205, 0x2208, 0x2209, 0x220F, 0x2211, 0x2212, 0x2215,
             0x221A, 0x221E, 0x2227, 0x2228, 0x2229, 0x222A, 0x222B, 0x2248,
             0x2260, 0x2261, 0x2264, 0x2265, 0x22C5}
TECH_KEEP = {0x2302, 0x2318, 0x2325, 0x2326, 0x232B, 0x23CE}
CONSOLE_LEAN = ((rng((0x20, 0x7E), (0xA0, 0xFF), (0x100, 0x17F),
                     (0x2000, 0x206F), (0x2070, 0x209F), (0x20A0, 0x20BF),
                     (0x2500, 0x257F), (0x2580, 0x259F), (0x25A0, 0x25FF),
                     (0x2190, 0x21FF), (0x2100, 0x214F), (0x2400, 0x243F))
                | EXTB_KEEP | EXTADD_KEEP | MATH_KEEP | TECH_KEEP) & cmap)
WHITESPACE = {0x09, 0x0A, 0x0D}

occurrences = Counter()
file_hits = Counter()
total_chars = 0
n_files = 0
for path in open("readme_list.txt"):
    path = path.strip()
    try:
        data = open(path, "rb").read(5_000_000)
    except OSError:
        continue
    text = data.decode("utf-8", errors="ignore")
    n_files += 1
    seen = set()
    for ch in text:
        cp = ord(ch)
        if cp in WHITESPACE or cp < 0x20:
            continue
        total_chars += 1
        if cp not in CONSOLE_LEAN:
            occurrences[cp] += 1
            seen.add(cp)
    for cp in seen:
        file_hits[cp] += 1

missing_occ = sum(occurrences.values())
print(f"files: {n_files}   glyph chars: {total_chars:,}   "
      f"missing occurrences: {missing_occ:,} ({100*missing_occ/total_chars:.3f}%)")
print(f"distinct missing codepoints: {len(occurrences)}")
addable = {cp for cp in occurrences if cp in cmap}
print(f"  of which in DejaVu Mono (addable): {len(addable)}; "
      f"not in font (fallback needed): {len(occurrences) - len(addable)}\n")

def name(cp):
    try:
        return unicodedata.name(chr(cp))
    except ValueError:
        return "<unnamed>"

print(f"{'char':>4} {'cp':>7} {'files':>6} {'occurs':>8} {'font':>4}  name")
for cp, nf in file_hits.most_common(45):
    ch = chr(cp) if unicodedata.category(chr(cp))[0] != "C" else "?"
    print(f"{ch:>4} U+{cp:05X} {nf:>6} {occurrences[cp]:>8} "
          f"{'yes' if cp in cmap else 'NO':>4}  {name(cp)[:52]}")

# Block rollup of ALL misses (by occurrences).
BLOCKS = [(0x180, 0x24F, "latin-ext-B (dropped part)"), (0x250, 0x2FF, "IPA/modifiers (dropped)"),
          (0x300, 0x36F, "combining (dropped)"), (0x370, 0x3FF, "greek"),
          (0x400, 0x52F, "cyrillic"), (0x1E00, 0x1EFF, "latin-ext-add (dropped part)"),
          (0x2200, 0x22FF, "math beyond keep-list"), (0x2300, 0x23FF, "misc-technical beyond keep"),
          (0x2600, 0x26FF, "misc-symbols"), (0x2700, 0x27BF, "dingbats"),
          (0x27C0, 0x2BFF, "arrows/math suppl"), (0x2E80, 0x9FFF, "CJK"),
          (0xAC00, 0xD7FF, "hangul"), (0x1F000, 0x1FAFF, "emoji"),
          (0xFE00, 0xFE0F, "variation selectors"), (0x200B, 0x200D, "zero-width")]
roll = Counter()
for cp, n in occurrences.items():
    for a, b, label in BLOCKS:
        if a <= cp <= b:
            roll[label] += n
            break
    else:
        roll["other"] += n
print("\nmisses by block (occurrences):")
for label, n in roll.most_common():
    print(f"  {label:>34}: {n:,}")

# Proposed additions: in font, appearing in >= 15 files.
adds = sorted(cp for cp, nf in file_hits.items() if nf >= 15 and cp in cmap)
print(f"\nproposed additions (in-font, >=15 files): {len(adds)} glyphs")
print("  " + " ".join(chr(c) for c in adds))
print(f"  arithmetic cost: ~{len(adds)*240:,} B raw4 (tight-crop avg)")
