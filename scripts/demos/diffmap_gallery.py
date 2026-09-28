#!/usr/bin/env python3
"""Build the zensim spatial-diffmap demo gallery.

WHAT THIS IS. `zensim` has had a spatial diffmap for months and no way to
*look* at one: every consumer so far reads the `f32` map numerically. This
builds the first viewable artifact -- for a handful of real corpus sources,
encode a JPEG ladder, render `zensim`'s diffmap as a colour heatmap and as an
overlay on the distorted image, and lay it out as one self-contained page.

It is a DEMO, not evidence. It ranks nothing, qualifies nothing, and its
numbers are a single un-replicated pass. Its only job is to answer "is the
spatial diffmap worth showing to a human".

PIPELINE, all imazen codecs, no ImageMagick / ffmpeg / libjpeg anywhere:

    source PNG  --zenresize Lanczos-->  reference PNG (long edge <= MAX_DIM)
                --zenjpeg 4:2:0 q{20,50,80}-->  distorted JPEG
    (reference PNG, distorted JPEG) --zensim compute_with_diffmap--> heat +
                                                                     overlay

    Both stages are zensim examples: `gen_jpeg_distortion` owns the
    resize+encode pair build, `diffmap_heatmap` owns the render. The JPEG is
    decoded by zenjpeg inside `diffmap_heatmap` (that is what gets scored);
    the page shows the JPEG bytes themselves in the "distorted" column, since
    the browser renders those without any decoder of ours in the loop.

DATA. Sources are imazen-26 (`png-v3`, the canonical 4-digit-id estate) and
every one is TRAIN-split, verified against `imazen26_manifest.tsv` at build
time -- no CID22 reference, no T0 holdout content, nothing from a secret
holdout. The chosen paths and their sha256 land in `manifest.json` next to
the page.

    python3 scripts/demos/diffmap_gallery.py [--out DIR] [--max-dim N]
                                             [--profile b|d] [--scale-max F]

    just demo-diffmap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V3 = Path("/mnt/v/output/imazen-26-png-v3")
MANIFEST_TSV = Path("/mnt/v/output/imazen-26-features/imazen26_manifest.tsv")
DEFAULT_OUT = Path("/mnt/v/output/zensim/demos/diffmap-heatmap-2026-09-18")
# `/mnt/v/output/` is served at this prefix by the mntv-gallery service.
SERVE_ROOT = Path("/mnt/v/output")
SERVE_URL = "http://localhost:3300"

QUALITIES = (20, 50, 80)

# Six sources spanning photo / screen / line-art-and-graphic, chosen for
# content class rather than for how they score. Every id ends in an even
# digit, which is the canonical TRAIN bucket (DATA_SPLITS §2a); `load_sources`
# re-checks each one against the manifest's own `split` column and refuses
# anything that is not TRAIN, so this list cannot silently drift into eval
# content.
SOURCES = [
    (
        "1000",
        "photo",
        "1000-lilith-photos-general/"
        "1000_general_red-convertible-car_mission-hills-san-diego-california"
        "_note9_iso50-f2p4_20190514-100811_4032x3024.sdr.png",
    ),
    (
        "1402",
        "photo (nature, dense texture)",
        "1400-lilith-nature/"
        "1402_nature_rocky-ocean-coast_spitting-caves-east-honolulu"
        "_s21u_iso16-f1p8_20210605-110609_4000x3000.sdr.png",
    ),
    (
        "8160",
        "web screenshot (text + UI chrome)",
        "8100-lilith-web-screenshots/1920x1080/"
        "8160_web-screenshots_archives-exhibits_dpr1_page1_1920x1080.sdr.png",
    ),
    (
        "7106",
        "chart (flat fills, thin rules)",
        "7000-lilith-plots/real-charts/7106_plots_chart-bar-01-light_1024x1024.sdr.png",
    ),
    (
        "9000",
        "clipart (flat colour, hard edges)",
        "9000-lilith-ai-clipart/9000_gen_clipart_avocado-half_1024x1536.sdr.png",
    ),
    (
        "6110",
        "grayscale document scan (line art + text)",
        "6000-lilith-scans-public-patents/yvonne_brill_us3807657_printrescangray300/"
        "6110_scans-patents_yvonne-brill-us3807657-rescan-gray_p002_2479x3230.sdr.png",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources() -> list[dict]:
    """Resolve the source list and HARD-FAIL on anything not TRAIN.

    The split column is read from the manifest rather than inferred from the
    id, because the last-significant-digit rule is the *derivation* of the
    split and the manifest is the record of it. A demo is not a reason to
    guess at split membership.
    """
    if not MANIFEST_TSV.exists():
        sys.exit(f"missing imazen-26 manifest: {MANIFEST_TSV}")
    splits: dict[str, str] = {}
    with MANIFEST_TSV.open() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                splits[parts[0]] = parts[1]

    out = []
    for stem, label, rel in SOURCES:
        path = V3 / rel
        if not path.exists():
            sys.exit(f"missing source image: {path}")
        split = splits.get(stem)
        if split != "train":
            sys.exit(f"source {stem} has split={split!r}, refusing (TRAIN only)")
        out.append(
            {
                "id": stem,
                "label": label,
                "path": str(path),
                "split": split,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return out


def build_examples() -> Path:
    """Build the two zensim examples this driver shells to; return their dir."""
    run_heavy = Path.home() / "work/zen/scripts/run-heavy"
    cargo = [
        "cargo",
        "build",
        "--release",
        "-p",
        "zensim",
        "--example",
        "gen_jpeg_distortion",
        "--example",
        "diffmap_heatmap",
    ]
    cmd = (
        [str(run_heavy), "--mem", "16G", "--jobs", "8", "--"] + cargo
        if run_heavy.exists()
        else cargo
    )
    subprocess.run(cmd, cwd=REPO, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))
    return target / "release" / "examples"


def serve_url(path: Path) -> str:
    """Browser URL for a path under the served root, or the bare path if not.

    `--out` may point anywhere; only paths under `/mnt/v/output` are reachable
    through the local gallery service, and saying so beats printing a URL that
    404s.
    """
    try:
        return f"{SERVE_URL}/{path.relative_to(SERVE_ROOT).as_posix()}"
    except ValueError:
        return f"file://{path}  (outside {SERVE_ROOT}, not served)"


CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #16181d; --muted: #5d636e;
  --card: #f4f5f7; --line: #d9dce1; --accent: #b3431f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e8eaee; --muted: #9aa1ac;
    --card: #1d2026; --line: #2f343c; --accent: #f7a23d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px 20px 64px;
  background: var(--bg); color: var(--fg);
  font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
h1 { font-size: 22px; margin: 0 0 6px; }
h2 { font-size: 17px; margin: 40px 0 4px; }
p, li { max-width: 64rem; color: var(--muted); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
a { color: var(--accent); }
.note { border-left: 3px solid var(--line); padding: 2px 0 2px 12px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; }
th, td {
  border-top: 1px solid var(--line); padding: 8px 8px; vertical-align: top;
  text-align: left;
}
th { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
td.meta { white-space: nowrap; font-variant-numeric: tabular-nums; }
td img { width: 100%; height: auto; display: block; border-radius: 4px; background: var(--card); }
td.cell { width: 24%; }
.score { font-size: 19px; font-weight: 650; color: var(--fg); }
.stats { font-size: 12px; color: var(--muted); line-height: 1.7; }
.ramp {
  height: 14px; border-radius: 3px; margin: 6px 0 4px; max-width: 34rem;
  background: linear-gradient(90deg, RAMP_STOPS);
}
.ramplabels {
  display: flex; justify-content: space-between; max-width: 34rem;
  font-size: 12px; color: var(--muted);
}
"""

INFERNO = [
    "#000004",
    "#1b0c41",
    "#4a0c6b",
    "#781c6d",
    "#a52c60",
    "#cf4446",
    "#ed6925",
    "#fb9b06",
    "#f7d13d",
    "#fcffa4",
]


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def render_html(out: Path, sources: list[dict], rows: list[dict], cfg: dict) -> str:
    stops = ", ".join(
        f"{c} {i * 100 / (len(INFERNO) - 1):.1f}%" for i, c in enumerate(INFERNO)
    )
    css = CSS.replace("RAMP_STOPS", stops)

    def rel(p: str) -> str:
        return html_escape(Path(p).name)

    parts: list[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>zensim spatial diffmap - JPEG ladder</title>",
        f"<style>{css}</style></head><body>",
        "<h1>zensim spatial diffmap &mdash; JPEG quality ladder</h1>",
        "<p>Each row compares a reference against a zenjpeg 4:2:0 encode of it. "
        "<strong>heat</strong> is <code>zensim</code>&rsquo;s per-pixel diffmap on a "
        "<em>fixed absolute</em> colour scale, shared by every image on this page, so "
        "brightness is comparable across rows &mdash; a q20 encode is <em>supposed</em> to "
        "look hotter than a q80 one. <strong>overlay</strong> lays the same map over a "
        "desaturated copy of the distorted image.</p>",
        "<div class=\"ramp\"></div><div class=\"ramplabels\"><span>0.000 "
        "(perceptually identical)</span><span>&ge; "
        f"{cfg['scale_max']:.3f} (clipped)</span></div>",
        "<div class=\"note\"><p>"
        f"profile <code>{html_escape(cfg['profile'])}</code> &middot; "
        f"<code>DiffmapOptions::default()</code> (<code>Trained</code> channel weighting, "
        "no masking, no sqrt, SSIM-error only) &middot; reference long edge "
        f"{cfg['max_dim']} px (zenresize Lanczos) &middot; diffmap is at full image "
        "resolution &mdash; nothing is upsampled by the renderer.</p>"
        "<p>The <em>distorted</em> column shows the JPEG bytes the browser renders; "
        "<code>zensim</code> scored the zenjpeg decode of those same bytes. This page is a "
        "demo, not evidence: one pass, no replication, no ranking claim.</p></div>",
        "<h2>Sources</h2>",
        "<p>imazen-26 <code>png-v3</code>, all TRAIN split (verified against "
        "<code>imazen26_manifest.tsv</code>). Paths + sha256 in "
        "<a href=\"manifest.json\">manifest.json</a>.</p><table>",
        "<tr><th>id</th><th>content</th><th>source file</th></tr>",
    ]
    for s in sources:
        parts.append(
            f"<tr><td class=\"meta\">{html_escape(s['id'])}</td>"
            f"<td>{html_escape(s['label'])}</td>"
            f"<td><code>{html_escape(Path(s['path']).name)}</code></td></tr>"
        )
    parts.append("</table>")

    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source_id"], []).append(r)

    for s in sources:
        group = by_src.get(s["id"], [])
        if not group:
            continue
        first = group[0]
        parts.append(
            f"<h2>{html_escape(s['id'])} &mdash; {html_escape(s['label'])}</h2>"
            f"<p>{first['width']}&times;{first['height']} &middot; reference PNG "
            f"{first['ref_bytes'] / 1024:.0f} KiB</p><table>"
            "<tr><th>encode</th><th class=\"cell\">reference</th>"
            "<th class=\"cell\">distorted</th><th class=\"cell\">heat</th>"
            "<th class=\"cell\">overlay</th></tr>"
        )
        for r in sorted(group, key=lambda r: r["quality"]):
            parts.append(
                "<tr>"
                f"<td class=\"meta\"><div class=\"score\">{r['score']:.2f}</div>"
                f"<div class=\"stats\">zensim score<br>"
                f"JPEG q{r['quality']}<br>"
                f"{r['jpeg_bytes'] / 1024:.1f} KiB "
                f"({r['bpp']:.3f} bpp)<br><br>"
                f"map p50 {r['map_p50']:.4f}<br>"
                f"map p99 {r['map_p99']:.4f}<br>"
                f"map max {r['map_max']:.4f}</div></td>"
                f"<td><img loading=\"lazy\" src=\"{rel(r['ref_png'])}\" alt=\"reference\"></td>"
                f"<td><img loading=\"lazy\" src=\"{rel(r['jpeg'])}\" alt=\"JPEG q{r['quality']}\"></td>"
                f"<td><img loading=\"lazy\" src=\"{rel(r['heat_png'])}\" alt=\"diffmap heat\"></td>"
                f"<td><img loading=\"lazy\" src=\"{rel(r['overlay_png'])}\" alt=\"diffmap overlay\"></td>"
                "</tr>"
            )
        parts.append("</table>")

    parts.append(
        "<h2>Reproduce</h2><p><code>just demo-diffmap</code>, or "
        "<code>python3 scripts/demos/diffmap_gallery.py</code> from the zensim repo root.</p>"
        "</body></html>"
    )
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-dim", type=int, default=900)
    ap.add_argument("--profile", default="b", choices=["b", "d"])
    ap.add_argument(
        "--scale-max",
        type=float,
        default=None,
        help="absolute diffmap value at the top of the colour ramp "
        "(default: the example's own measured default)",
    )
    args = ap.parse_args()

    sources = load_sources()
    bindir = build_examples()
    gen = bindir / "gen_jpeg_distortion"
    heat = bindir / "diffmap_heatmap"
    for b in (gen, heat):
        if not b.exists():
            sys.exit(f"example binary not built: {b}")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows: list[dict] = []
    scale_max = None
    for s in sources:
        ref_png = out / f"{s['id']}_ref.png"
        for q in QUALITIES:
            jpeg = out / f"{s['id']}_q{q}.jpg"
            cmd = [str(gen), s["path"], str(q), str(jpeg), "--max-dim", str(args.max_dim)]
            if not ref_png.exists():
                cmd += ["--ref-out", str(ref_png)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            hcmd = [
                str(heat),
                str(ref_png),
                str(jpeg),
                str(out / f"{s['id']}_q{q}"),
                "--profile",
                args.profile,
            ]
            if args.scale_max is not None:
                hcmd += ["--scale-max", str(args.scale_max)]
            res = subprocess.run(hcmd, check=True, capture_output=True, text=True)
            rec = json.loads(res.stdout.strip().splitlines()[-1])
            scale_max = rec["scale_max"]
            rec.update(
                source_id=s["id"],
                quality=q,
                jpeg=str(jpeg),
                jpeg_bytes=jpeg.stat().st_size,
                ref_png=str(ref_png),
                ref_bytes=ref_png.stat().st_size,
                bpp=jpeg.stat().st_size * 8 / (rec["width"] * rec["height"]),
            )
            rows.append(rec)
            print(
                f"  {s['id']} q{q:<3d} score {rec['score']:6.2f}  "
                f"{rec['jpeg_bytes'] / 1024:7.1f} KiB  "
                f"map p99 {rec['map_p99']:.4f} max {rec['map_max']:.4f}",
                flush=True,
            )

    cfg = {
        "profile": rows[0]["profile"],
        "scale_max": scale_max,
        "max_dim": args.max_dim,
        "qualities": list(QUALITIES),
        "diffmap_options": "DiffmapOptions::default() (Trained weighting, SSIM error only)",
    }
    manifest = {
        "generated_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "config": cfg,
        "pipeline": [
            "zenresize Lanczos downscale to max-dim",
            "zenpng encode -> reference PNG",
            "zenjpeg encode 4:2:0 at q in {20,50,80}",
            "zenjpeg decode + zensim compute_with_diffmap",
        ],
        "sources": sources,
        "rows": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "index.html").write_text(render_html(out, sources, rows, cfg))

    print(f"\nwrote {out}/index.html")
    print(f"view: {serve_url(out / 'index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
