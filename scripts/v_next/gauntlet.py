#!/usr/bin/env python3
"""Summer-gauntlet INTERACTIVE dashboard builder — the ``--fulleval-dir`` mode of
``bandwise_dashboard.py`` (user 2026-07-26: "interactively compare all the summer's best bakes,
with correlation scatterplots for every reference — MOS, JND, ssim2, butteraugli, cvvdp").

It reads the per-bake ``*.fulleval.json`` files (schema + fixtures: ``make_stub_fulleval.py``)
and emits ONE self-contained, offline HTML page with:

  * a toggle bar (checkbox per bake — hide/show it across EVERY chart; stable color per bake),
  * a SORTABLE scoreboard table (click any header to sort by CID22 / KonJND / dial-mono / M3 /
    corruption-detection / composite / …),
  * the correlation SCATTER MATRIX — for the selected reference, one clean scatter per
    (bake x corpus) with an OLS fit line + canonical SROCC/PLCC annotated, faceted so bakes sit
    side by side per corpus,
  * a cross-corpus SROCC heatmap and a CID22-vs-{nonphoto,KonJND} operating-point trade map,
  * the JXL loop-targeting panel (2026-08-01): 2-shot/3-shot within-±2 scoreboard columns
    (emit-best, bakes mapped via ``LOOP_BAKE_MAP``) + a section table of every loop model
    (emit-last detail, outer arms, ssim2), fed verbatim by the jxl-encoder sweep summary
    JSON (``--loop-targeting``; counts/medians are READ, never re-derived here).

ENSEMBLE rows (2026-08-04): a fulleval JSON carrying ``model.kind == "ensemble"`` (stamped by
``scripts/promote_ensemble_fulleval.py``) renders an ``ens×k`` marker everywhere the bake is
named, and its Model-details card leads with a warning that the architecture/repro shown is the
ANCHOR member. An ensemble is an evaluation FUNCTION, not a shippable artifact — its rank/dial/
corruption numbers come from the identical verdict invocation as every single-bake row and are
directly comparable, but ``m3_coherence``/``m3a_coherence`` are **null** because the coherence
instrument loads one ZNPR. Null renders as an em-dash (NOT MEASURED) and is excluded from column
shading and min/max — it is never displayed or shaded as a measured zero.

NO external requests: all CSS/JS/data are inlined (no CDN, no web fonts) so the file opens
offline. NO hand-rolled statistics: every SROCC/PLCC comes from the canonical ``panel`` (via the
fulleval JSON's precomputed ``scatter`` block, or computed at build through
``scripts/lib/zen_stats.panel`` when a JSON omits it). Only OLS fit-line endpoints are computed
here (numpy polyfit — a display aid, like bake_report.py), never an IQA stat.

Colors follow the dataviz skill's validated categorical palette (see the palette validator run in
the commit); identity is never color-alone — every series is labeled and the table view exists.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# dataviz validated categorical palette (8 hues, light/dark), + chart ink. Validated by
# scripts/validate_palette.js (light: all CVD/normal gates pass, 3 slots need the relief rule =
# labels+table, both provided; dark: all gates pass incl. contrast).
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
REFERENCES = ["mos", "jnd", "ssim2", "butter", "cvvdp"]
REF_LABELS = {"mos": "MOS (human)", "jnd": "JND (human)", "ssim2": "SSIMULACRA2",
              "butter": "butteraugli (↑=better)", "cvvdp": "ColorVideoVDP"}
# scoreboard columns beyond CID22: (key, header, higher_is_better, fmt)
CORP_ORDER = ["cid22", "nonphoto", "konjnd", "aic3", "aic4", "live", "csiq", "kadid", "tid"]
SCATTER_MAX = 500  # subsample dense per_pair for embedding — keeps the offline file responsive

# ---- JXL loop-targeting (2/3-shot) summary — produced by the jxl-encoder repo's exact
# 2/3-shot sweep (raw cells: benchmarks/zensim_loop_23shot_2026-08-01.tsv there; doc:
# benchmarks/zensim_loop_23shot_2026-08-01.md). The dashboard READS the machine summary
# JSON — counts/medians are never re-derived here (no-duplication rule; the jxl analyze
# script is the owner). Loop-model keys -> gauntlet bake names (fulleval `name`): models
# not in this map (the outer arms + ssim2, which are not bakes) render only in the
# section's own table; bakes without loop data render an em-dash.
DEFAULT_LOOP_TARGETING = (
    "/home/lilith/work/zen/jxl-encoder/benchmarks/zensim_loop_23shot_summary_2026-08-01.json"
)
LOOP_BAKE_MAP = {
    # loop-model key (summary JSON `models` key, = the sweep TSV run prefix)
    #   ->  bake `name` on the gauntlet board (fulleval JSON `name`).
    # Order matters: the FIRST model mapping to a bake is that bake's scoreboard
    # primary (v47A_base before the h3 variant and the outer arm).
    "v47A_base": "v47_strict_QAT_native",
    "v47A_h3g20c135": "v47_strict_QAT_native",
    "B_base": "b_sdr_linear_cid80_inclwinsor_dense_dial",
    "bvls_base": "v02_bvls_NO_shaping",
    "outer_zensimA": "v47_strict_QAT_native",
    # blend2L_base's bake (mlp_2L_diverse_H128) has no fulleval JSON on the board —
    # its row shows the bake filename from the summary JSON; map it when one lands.
}


def load_loop_targeting(path=None):
    """Read the Part-A machine summary JSON (jxl-encoder sweep). Returns the embed dict
    {meta, models, bakeMap, modelBake} or None (missing file -> section omitted, loud note).
    Counts/medians are READ verbatim, never re-derived here."""
    p = Path(path or DEFAULT_LOOP_TARGETING)
    if not p.exists():
        print(f"NOTE: loop-targeting summary not found at {p} — JXL loop panel omitted",
              file=sys.stderr)
        return None
    o = json.loads(p.read_text())
    models = o.get("models") or {}
    bake_map = {}     # bake name -> PRIMARY loop-model key (first map hit wins = baseline arm)
    model_bake = {}   # loop-model key -> bake name (for the section table's bake column)
    for mk, bake in LOOP_BAKE_MAP.items():
        if mk in models:
            model_bake[mk] = bake
            if bake not in bake_map:
                bake_map[bake] = mk
    return {"meta": {k: o.get(k) for k in ("date", "matrix", "notes", "source") if k in o},
            "models": models, "bakeMap": bake_map, "modelBake": model_bake}


def _fit_line(x, y):
    """OLS endpoints [x0,y0,x1,y1] for the display trend (a fit aid, not a stat)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    a = np.polyfit(x, y, 1)
    x0, x1 = float(np.min(x)), float(np.max(x))
    return [round(x0, 4), round(float(a[0] * x0 + a[1]), 4),
            round(x1, 4), round(float(a[0] * x1 + a[1]), 4)]


def _panel_srocc_plcc(pred, ref):
    """Canonical SROCC/PLCC via the Rust panel shim (fallback when a JSON omits `scatter`)."""
    from lib.zen_stats import panel
    p = panel(list(map(float, pred)), list(map(float, ref)))
    return {"srocc": round(abs(p["srocc"]), 4), "plcc": round(p["plcc"], 4), "n": int(p["n"])}


def _composite(rank):
    """FALLBACK ONLY (pre-2026-07-26 JSONs). The canonical composite is the Rust
    `product_composite`, emitted as `composite` in the fulleval JSON; `load_fulleval`
    READS that and only calls this when the field is absent, so there is one source
    of truth (stats review Rec-7). Goal-aware ranking scalar (reuses blend_lib.composite
    — the owner — when importable, else a transparent documented fallback). rank:
    {corpus: {srocc,...}} with srocc already
    polarity-corrected (abs for JND corpora), per the fulleval schema."""
    def g(c):
        v = rank.get(c, {}).get("srocc")
        return float(v) if v is not None and np.isfinite(v) else 0.0
    try:
        import blend_lib as B
        res = {}
        for c in B.VAL_CORPORA:
            v = g(c)
            res[c] = {"srocc": v, "srocc_abs": v}
        score, reject = B.composite(res)
        return round(float(score), 4), bool(reject)
    except Exception:
        # documented fallback (same weights as blend_lib.composite)
        score = g("cid22") + 0.30 * g("nonphoto") + 0.20 * g("konjnd") + 0.10 * g("aic3") + 0.05 * g("aic4")
        reject = (g("cid22") < 0.84) or (g("nonphoto") < 0.80)
        return round(score, 4), bool(reject)


def load_fulleval(fulleval_dir, best_per_day=None):
    """Read every *.fulleval.json; order by best_per_day date when available. Returns the list of
    bake dicts prepared for embedding (subsampled scatter points + fit lines + composite)."""
    fulleval_dir = Path(fulleval_dir)
    files = sorted(fulleval_dir.glob("*.fulleval.json"))
    if not files:
        raise SystemExit(f"no *.fulleval.json in {fulleval_dir} — run make_stub_fulleval.py to stub them")
    order = {}
    bpd = Path(best_per_day) if best_per_day else fulleval_dir.parent / "best_per_day.json"
    if bpd.exists():
        try:
            for i, r in enumerate(json.loads(bpd.read_text())):
                order[r.get("name")] = (r.get("date", ""), i)
        except Exception:
            pass
    raw = [json.loads(f.read_text()) for f in files]
    raw.sort(key=lambda o: order.get(o.get("name"), (o.get("date", ""), 99)))

    rng = np.random.RandomState(0)
    bakes = []
    for ci, o in enumerate(raw):
        rank = o.get("rank", {})
        # Prefer the Rust-emitted `composite` (product_composite is the single
        # source — stats review Rec-7); the dashboard READS it rather than
        # re-deriving a divergent one. `_composite` stays only as the fallback
        # for pre-2026-07-26 JSONs that predate the field. The reject gate is a
        # dashboard concern (CID22<0.84 or nonphoto<0.80), computed either way.
        emitted = o.get("composite")
        if emitted is not None:
            comp = round(float(emitted), 4)
            cid = rank.get("cid22", {}).get("srocc")
            nph = rank.get("nonphoto", {}).get("srocc")
            reject = (cid is None or abs(cid) < 0.84) or (nph is not None and abs(nph) < 0.80)
        else:
            comp, reject = _composite(rank)
        scatter_out = {}
        pp = o.get("per_pair", {})
        sc_json = o.get("scatter", {})
        for corp, cols in pp.items():
            pred = cols.get("pred")
            if not pred:
                continue
            pred = np.asarray(pred, float)
            n = len(pred)
            idx = np.arange(n) if n <= SCATTER_MAX else rng.permutation(n)[:SCATTER_MAX]
            cell = {}
            for ref in REFERENCES:
                if ref not in cols:
                    continue
                rv = np.asarray(cols[ref], float)
                pts = [[round(float(pred[i]), 4), round(float(rv[i]), 4)] for i in idx
                       if np.isfinite(pred[i]) and np.isfinite(rv[i])]
                stats = sc_json.get(corp, {}).get(ref)
                if not stats:                      # JSON omitted it -> canonical panel at build
                    stats = _panel_srocc_plcc(pred, rv)
                cell[ref] = {"pts": pts, "fit": _fit_line(pred, rv),
                             "srocc": stats.get("srocc"), "plcc": stats.get("plcc"),
                             "n": stats.get("n", len(pts))}
            if cell:
                scatter_out[corp] = cell
        bakes.append({
            "name": o.get("name", f"bake{ci}"), "regime": o.get("regime", "?"),
            "date": o.get("date", ""), "colorIndex": ci,
            "rank": rank, "dial": o.get("dial", {}), "m3": o.get("m3_coherence"),
            "m3a": o.get("m3a_coherence"),
            "corruption": o.get("corruption", {}), "composite": comp, "reject": reject,
            "m3_dropped_mass": o.get("m3_dropped_mass_pct"),
            "gates": o.get("gates") or {},
            "model": o.get("model") or {},
            "repro": o.get("repro"),
            "scatter": scatter_out, "is_stub": bool(o.get("_stub")),
        })
    return bakes


# ------------------------------------------------------------------ HTML assembly ------------
_CSS = """
:root{color-scheme:light dark}
.viz-root{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --seq-lo:#cde2fb; --seq-hi:#104281;
  color:var(--text-primary); background:var(--plane);
  font:13px system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --seq-lo:#0d366b; --seq-hi:#cde2fb;
}}
:root[data-theme="dark"] .viz-root{
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --seq-lo:#0d366b; --seq-hi:#cde2fb;
}
.viz-root{margin:0;padding:1.1rem 1.3rem 4rem}
h1{font-size:1.32rem;margin:.1rem 0 .2rem}
h2{font-size:1.03rem;margin:1.5rem 0 .5rem;border-top:1px solid var(--axis);padding-top:.55rem}
.sub{color:var(--text-secondary);max-width:70rem;line-height:1.45}
a{color:var(--seq-hi)}
code{background:var(--surface-1);border:1px solid var(--border);padding:.05rem .3rem;border-radius:3px;font-size:11px}
.bar{position:sticky;top:0;z-index:5;background:var(--plane);border-bottom:1px solid var(--border);
     padding:.5rem 0 .55rem;margin-bottom:.4rem;display:flex;flex-wrap:wrap;gap:.55rem;align-items:center}
.chip{display:inline-flex;align-items:center;gap:.4rem;padding:.24rem .55rem;border:1px solid var(--border);
      border-radius:1rem;cursor:pointer;user-select:none;background:var(--surface-1);font-size:12px;white-space:nowrap}
.chip input{margin:0;cursor:pointer}
.chip.off{opacity:.4}
.sw{width:11px;height:11px;border-radius:50%;flex:0 0 auto;border:1px solid var(--border)}
.btn{padding:.24rem .6rem;border:1px solid var(--border);border-radius:.35rem;background:var(--surface-1);
     color:var(--text-primary);cursor:pointer;font-size:12px}
.btn:hover{border-color:var(--muted)}
.btn.active{background:var(--seq-hi);color:#fff;border-color:var(--seq-hi)}
.tabs{display:flex;gap:.3rem;flex-wrap:wrap}
table{border-collapse:collapse;margin:.4rem 0;font-size:11.5px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--border);padding:3px 7px;text-align:right;white-space:nowrap}
th{cursor:pointer;background:var(--surface-1);position:relative;text-align:right}
th:hover{color:var(--seq-hi)}
th.sorted::after{content:" \\2193";font-size:9px}
th.sorted.asc::after{content:" \\2191"}
td.lbl,th.lbl{text-align:left;font-weight:600}
tr.reject td{opacity:.55}
.grid{display:flex;flex-wrap:wrap;gap:.5rem;align-items:flex-start}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:6px;padding:.35rem .4rem}
.cap{font-size:10.5px;color:var(--muted)}
.scrow{display:flex;flex-wrap:wrap;gap:.5rem;align-items:flex-start;margin:.35rem 0 .6rem}
.corpttl{font-weight:600;margin:.7rem 0 .1rem;font-size:12.5px}
.corpttl .badge{font-weight:600;font-size:9.5px;padding:.05rem .35rem;border-radius:.25rem;color:#fff;margin-left:.4rem}
svg{display:block;max-width:100%;height:auto}
.tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--muted);
    border-radius:4px;padding:.25rem .45rem;font-size:11px;z-index:20;opacity:0;transition:opacity .08s;
    box-shadow:0 2px 8px rgba(0,0,0,.18);white-space:nowrap}
.stub{color:var(--serious);font-weight:600}
"""


def build_html(bakes, out_path, title="zensim summer gauntlet", loop_targeting=None):
    data = {"bakes": bakes, "palette": PALETTE, "references": REFERENCES,
            "refLabels": REF_LABELS, "corpOrder": CORP_ORDER,
            "loopTargeting": loop_targeting}
    any_stub = any(b.get("is_stub") for b in bakes)
    stub_note = ("<span class='stub'>STUB DATA</span> — synthesized fixtures "
                 "(<code>make_stub_fulleval.py</code>); drop the eval agent's real "
                 "<code>*.fulleval.json</code> in and re-run. " if any_stub else "")
    head = (
        "<h1>" + title + "</h1>"
        "<p class='sub'>" + stub_note +
        "Toggle bakes below to compare them across every chart; click a table header to sort. "
        "The scatter matrix shows <b>predicted vs each reference</b> (MOS, JND, SSIMULACRA2, "
        "butteraugli, ColorVideoVDP) per corpus, with an OLS fit and canonical SROCC/PLCC. "
        "All data, styles and scripts are inlined — this page opens offline. "
        "By <code>scripts/v_next/bandwise_dashboard.py --fulleval-dir</code>.</p>"
    )
    payload = json.dumps(data, separators=(",", ":"))
    html = (
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>" + title + "</title>"
        "<style>" + _CSS + "</style>"
        "<div class='viz-root'>" + head +
        "<div id='bar' class='bar'></div>"
        "<div id='panels'></div>"
        "<div id='tt' class='tt'></div>"
        "</div>"
        "<script>\nconst DATA=" + payload + ";\n" + _JS + "\n</script>"
    )
    Path(out_path).write_text(html)
    return out_path, len(html)


# ------------------------------------------------------------------ client JS ----------------
_JS = r"""
'use strict';
const $=(s,r=document)=>r.querySelector(s);
const el=(t,a={},kids=[])=>{const e=document.createElementNS(t.startsWith('svg:')?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml',t.replace('svg:',''));
  for(const k in a){if(k==='text')e.textContent=a[k];else if(k==='html')e.innerHTML=a[k];else e.setAttribute(k,a[k]);}
  (Array.isArray(kids)?kids:[kids]).forEach(c=>c&&e.appendChild(c));return e;};
const S=(t,a={},k=[])=>el('svg:'+t,a,k);

const state={visible:new Set(DATA.bakes.map(b=>b.name)), ref:null, sortKey:'composite', sortDir:-1, mcorp:null};
function effTheme(){const dt=document.documentElement.getAttribute('data-theme');
  return dt||((window.matchMedia&&matchMedia('(prefers-color-scheme:dark)').matches)?'dark':'light');}
const pal=()=>DATA.palette[effTheme()==='dark'?'dark':'light'];
const color=b=>pal()[b.colorIndex%8];
const cssv=n=>getComputedStyle($('.viz-root')).getPropertyValue(n).trim()||'#888';
const visBakes=()=>DATA.bakes.filter(b=>state.visible.has(b.name));

// pick default reference = first one that any bake carries
function initRef(){const have=new Set();DATA.bakes.forEach(b=>Object.values(b.scatter).forEach(c=>Object.keys(c).forEach(r=>have.add(r))));
  state.ref=DATA.references.find(r=>have.has(r))||'mos';}

// ---- number helpers
const f3=v=>v==null||!isFinite(v)?'—':(+v).toFixed(3);
const f2=v=>v==null||!isFinite(v)?'—':(+v).toFixed(2);
const pct=v=>v==null||!isFinite(v)?'—':(v*100).toFixed(1)+'%';

// ---- ENSEMBLE marker. An equal-weight ensemble of k bakes is an evaluation
// FUNCTION, not a shippable artifact: there is no single ZNPR, so M3/M3a are
// not computable (they render as an em-dash = NOT MEASURED, never a low score)
// and the model-details card describes the ANCHOR member only. Flag set by
// scripts/promote_ensemble_fulleval.py (model.kind / model.members).
const isEns=b=>!!(b.model&&b.model.kind==='ensemble');
const ensK=b=>(b.model&&b.model.members)||null;
const ensTag=b=>isEns(b)?' ens×'+ensK(b):'';
const ensBadge=b=>isEns(b)?el('span',{style:'font-size:9px;font-weight:700;letter-spacing:.03em;'
  +'padding:0 4px;margin-left:5px;border-radius:7px;vertical-align:1px;white-space:nowrap;'
  +'background:color-mix(in srgb, var(--warn) 34%, var(--surface-1));border:1px solid var(--border)',
  title:'equal-weight ensemble of '+ensK(b)+' bakes — an evaluation function, not a single '
    +'shippable bake; M3/M3a not computable',text:'ens×'+ensK(b)}):null;
// swatch + name (+ ens badge) cell content, shared by every table that names a bake
function nameInto(node,b,suffix){
  node.append(el('span',{class:'sw',style:'display:inline-block;margin-right:5px;background:'+color(b)}),
    document.createTextNode(b.name+(suffix||'')));
  const bd=ensBadge(b);if(bd)node.append(bd);
  return node;
}

// ---- tooltip
const tt=$('#tt');
function showTip(html,ev){tt.innerHTML=html;tt.style.opacity=1;
  let x=ev.clientX+12,y=ev.clientY+12;if(x>innerWidth-160)x=ev.clientX-tt.offsetWidth-12;
  tt.style.left=x+'px';tt.style.top=y+'px';}
function hideTip(){tt.style.opacity=0;}

// ---- CONTROL BAR: bake chips + select buttons + reference tabs + theme
function renderBar(){
  const bar=$('#bar');bar.innerHTML='';
  DATA.bakes.forEach(b=>{
    const on=state.visible.has(b.name);
    const chip=el('label',{class:'chip'+(on?'':' off'),
      title:b.regime+(b.is_stub?' (stub)':'')+(isEns(b)?' · ensemble of '+ensK(b)+' bakes':'')});
    const cb=el('input',{type:'checkbox'});cb.checked=on;
    cb.onchange=()=>{on?state.visible.delete(b.name):state.visible.add(b.name);rerender();renderBar();};
    chip.append(cb, el('span',{class:'sw',style:'background:'+color(b)}),
      el('span',{text:b.name}), el('span',{class:'cap',text:b.regime+ensTag(b)}));
    bar.appendChild(chip);
  });
  const mk=(t,fn)=>{const x=el('button',{class:'btn',text:t});x.onclick=fn;return x;};
  bar.append(
    mk('all',()=>{DATA.bakes.forEach(b=>state.visible.add(b.name));rerender();renderBar();}),
    mk('none',()=>{state.visible.clear();rerender();renderBar();}),
    mk('top 6',()=>{const s=[...DATA.bakes].sort((a,b)=>b.composite-a.composite).slice(0,6).map(b=>b.name);
      state.visible=new Set(s);rerender();renderBar();}));
  // reference tabs
  const tabs=el('span',{class:'tabs'});
  tabs.append(el('span',{class:'cap',text:'reference:',style:'align-self:center'}));
  DATA.references.forEach(r=>{
    const has=DATA.bakes.some(b=>Object.values(b.scatter).some(c=>r in c));
    if(!has)return;
    const x=el('button',{class:'btn'+(state.ref===r?' active':''),text:DATA.refLabels[r]||r});
    x.onclick=()=>{state.ref=r;renderBar();renderScatter();};
    tabs.appendChild(x);
  });
  bar.appendChild(tabs);
  const th=el('button',{class:'btn',text:'◐ theme'});
  th.onclick=()=>{const cur=document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme',cur==='dark'?'light':(cur==='light'?'dark':'light'));rerender();};
  bar.appendChild(th);
}

// ---- SCOREBOARD TABLE (sortable). columns = derived metrics per bake.
const COLS=[
  ['name','bake',true,b=>b.name],
  ['regime','regime',true,b=>b.regime],
  ['composite','composite',false,b=>b.composite],
  ['cid22','CID22',false,b=>rs(b,'cid22')],
  ['nonphoto','nonphoto',false,b=>rs(b,'nonphoto')],
  ['konjnd','KonJND',false,b=>rs(b,'konjnd')],
  ['aic3','AIC-3',false,b=>rs(b,'aic3')],
  ['live','LIVE',false,b=>rs(b,'live')],
  ['csiq','CSIQ',false,b=>rs(b,'csiq')],
  ['dial_mono','dial-mono',false,b=>b.dial.mono_pct],
  ['dial_tied','tied',false,b=>b.dial.tied_pct],
  ['m3a','M3a-attr',false,b=>b.m3a],
  ['m3','M3-coh',false,b=>b.m3],
  ['m3_mass','M3 drop%',false,b=>b.m3_dropped_mass],
  ['corr','corr-passq20',false,b=>b.corruption&&b.corruption.pass_q20!=null?b.corruption.pass_q20:null],
  ['cid22_ci','CID22 95%CI±',false,b=>{const r=b.rank.cid22;return r&&r.srocc_ci?(r.srocc_ci[1]-r.srocc_ci[0])/2:null;}],
  ['cid22_bwd','CID22 %bwd',false,b=>{const r=b.rank.cid22;return r&&r.frac_negative!=null?r.frac_negative:null;}],
];
const rs=(b,c)=>{const r=b.rank[c];return r?r.srocc:null;};
// ---- JXL loop-targeting join (2/3-shot). LT is the jxl-encoder sweep summary (READ, not
// re-derived). Scoreboard shows the mapped bake's emit-best cells; full detail (emit-last,
// outer arms, ssim2) lives in the JXL loop targeting section.
const LT=DATA.loopTargeting||null;
const ltN=()=>(LT&&LT.meta&&LT.meta.matrix&&LT.meta.matrix.n_cells)||27;
const ltCell=(b,mode)=>{if(!LT)return null;const mk=LT.bakeMap[b.name];if(!mk)return null;
  const m=LT.models[mk];return (m&&m.cells&&m.cells[mode])||null;};
if(LT){COLS.push(
  ['loop2','2shot ±2',false,b=>{const c=ltCell(b,'k2_emit_best');return c?c.within2:null;}],
  ['loop3','3shot ±2',false,b=>{const c=ltCell(b,'k3_emit_best');return c?c.within2:null;}],
  ['loop3err','3shot med|err|',false,b=>{const c=ltCell(b,'k3_emit_best');return c!=null&&c.med_abs_err!=null?c.med_abs_err:null;}]);}
function fmtCell(key,v){
  if(key==='name'||key==='regime')return v;
  if(key==='cid22_bwd')return v==null?'—':pct(v);
  if(key==='dial_tied')return pct(v);
  if(key==='corr'||key==='dial_mono')return pct(v);
  if(key==='m3_mass')return v==null?'—':v.toFixed(1)+'%';
  if(key==='cid22_ci')return v==null?'—':'±'+v.toFixed(3);
  if(key==='loop2'||key==='loop3')return v==null?'—':v+'/'+ltN();
  if(key==='loop3err')return v==null?'—':(+v).toFixed(2);
  return f3(v);
}
function renderTable(){
  const wrap=el('div',{});
  const cap=el('div',{class:'cap',html:'Sortable scoreboard — click a header. SROCC is polarity-corrected '
    +'(|SROCC| for JND corpora). <b>composite</b> = the Rust <code>product_composite</code> (CID22·1.0 + '
    +'imazen26·0.5 + nonphoto·0.3 + KonJND·0.2 + AIC·0.15; KADID/TID excluded, train==val), READ from the JSON '
    +'not re-derived. <b>CID22 95%CI±</b> = bootstrap half-width; bakes with overlapping CIs are a statistical '
    +'TIE, not an ordering. <b>CID22 %bwd</b> = share of reference ladders ranked BACKWARDS (no pooled stat sees '
    +'it). <b>M3a-attr</b> = the DEPLOYABLE attribution-density steering map vs \u0394S (exact integrands + SAT, task #67 \u2014 the map codecs query); <b>M3-coh</b> = the legacy signal fold, kept for the before/after story (the 128px fold inversion the attribution map cures). <b>M3 drop%</b> = f156-371 mass the FOLD cannot spatialize — read a low M3 against it (high drop% '
    +'= M3 structurally capped, not incoherent). An <b>em-dash in any cell means NOT MEASURED</b> — never a '
    +'measured zero. Greyed row = reject-gate (CID22&lt;0.84 or nonphoto&lt;0.80). '
    +'<b>ens×k</b> = an equal-weight ENSEMBLE of k bakes, scored through the identical verdict invocation '
    +'as every single-bake row: rank/dial/corruption numbers are directly comparable, but an ensemble is an '
    +'<b>evaluation function, not a shippable artifact</b> — there is no single ZNPR, so <b>M3a/M3 are not '
    +'computable for it</b> (the coherence instrument loads one bake) and its Model-details card describes '
    +'the ANCHOR member only. Distillation to a single bake is pending.'
    +(LT?' <b>2shot/3shot ±2</b> = JXL loop targeting: cells (of '+ltN()+') where the DECODED-judged score lands '
    +'within ±2 of target in the bake’s own units at encode budget k=2/3, emit-best (emit-last + outer arms: '
    +'see the JXL loop targeting section).':'')});
  const tbl=el('table',{});
  const thead=el('tr',{});
  COLS.forEach(c=>{const th=el('th',{class:(c[0]==='name'||c[0]==='regime'?'lbl':'')
      +(state.sortKey===c[0]?' sorted'+(state.sortDir>0?' asc':''):''),text:c[1]});
    th.onclick=()=>{if(state.sortKey===c[0])state.sortDir*=-1;else{state.sortKey=c[0];state.sortDir=c[2]?1:-1;}renderTable();};
    thead.appendChild(th);});
  tbl.appendChild(el('thead',{},thead));
  // column min/max for shading (numeric cols only), across ALL bakes
  const ranges={};
  COLS.forEach(c=>{if(c[0]==='name'||c[0]==='regime')return;
    const vs=DATA.bakes.map(c[3]).filter(v=>v!=null&&isFinite(v));
    ranges[c[0]]=vs.length?[Math.min(...vs),Math.max(...vs)]:[0,1];});
  const col=COLS.find(c=>c[0]===state.sortKey)||COLS[2];
  const rows=[...DATA.bakes].sort((a,b)=>{let x=col[3](a),y=col[3](b);
    if(typeof x==='string')return state.sortDir*x.localeCompare(y);
    x=x==null?-1e9:x;y=y==null?-1e9:y;return state.sortDir*(x-y);});
  const tb=el('tbody',{});
  rows.forEach(b=>{
    const tr=el('tr',{class:b.reject?'reject':''});
    if(!state.visible.has(b.name))tr.style.opacity=.45;
    COLS.forEach(c=>{
      const v=c[3](b);
      const td=el('td',{class:(c[0]==='name'||c[0]==='regime')?'lbl':'',text:fmtCell(c[0],v)});
      if(c[0]==='name'){td.textContent='';nameInto(td,b,b.is_stub?' ✳':'');}
      if(c[0]!=='name'&&c[0]!=='regime'&&v!=null&&isFinite(v)){
        const[lo,hi]=ranges[c[0]];let t=hi===lo?.5:(v-lo)/(hi-lo);
        // invert shading where lower is better (tied dead-zone, CI width,
        // backwards-ref share, dropped-mass — all "smaller is better")
        if(c[0]==='dial_tied'||c[0]==='cid22_ci'||c[0]==='cid22_bwd'||c[0]==='m3_mass'||c[0]==='loop3err')t=1-t;
        td.style.background='color-mix(in srgb, var(--seq-hi) '+Math.round(t*62)+'%, var(--surface-1))';
        if(t>.6)td.style.color='#fff';
      }
      tr.appendChild(td);
    });
    tr.onclick=()=>{state.visible.has(b.name)?state.visible.delete(b.name):state.visible.add(b.name);rerender();renderBar();};
    tr.style.cursor='pointer';
    tb.appendChild(tr);
  });
  tbl.appendChild(tb);
  wrap.append(el('h2',{text:'Scoreboard'}),cap,tbl);
  return wrap;
}

// ---- generic SVG scatter cell: pred (x) vs reference (y)
function scatterCell(b,corp,ref){
  const cell=b.scatter[corp]&&b.scatter[corp][ref];
  const W=214,H=200,mL=34,mR=8,mT=34,mB=26;
  const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,
    style:'background:var(--surface-1);border:1px solid var(--border);border-radius:5px'});
  if(!cell||!cell.pts.length){svg.append(S('text',{x:W/2,y:H/2,'text-anchor':'middle',
    fill:cssv('--muted'),'font-size':10,text:'no '+ref}));return svg;}
  const pts=cell.pts;const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const px=(x1-x0)*.04||1,py=(y1-y0)*.04||1;x0-=px;x1+=px;y0-=py;y1+=py;
  const SX=v=>mL+(v-x0)/(x1-x0)*(W-mL-mR),SY=v=>H-mB-(v-y0)/(y1-y0)*(H-mT-mB);
  // grid + axes
  const g=cssv('--grid'),ax=cssv('--axis'),mu=cssv('--muted');
  for(let i=0;i<=3;i++){const gy=mT+i/3*(H-mT-mB);svg.append(S('line',{x1:mL,y1:gy,x2:W-mR,y2:gy,stroke:g,'stroke-width':.6}));}
  svg.append(S('line',{x1:mL,y1:mT,x2:mL,y2:H-mB,stroke:ax,'stroke-width':1}));
  svg.append(S('line',{x1:mL,y1:H-mB,x2:W-mR,y2:H-mB,stroke:ax,'stroke-width':1}));
  // y ticks (ref) + x ticks (pred)
  [y0,(y0+y1)/2,y1].forEach(v=>svg.append(S('text',{x:mL-3,y:SY(v)+3,'text-anchor':'end','font-size':8,fill:mu,text:(Math.abs(v)>=100?v.toFixed(0):v.toFixed(1))})));
  [x0,x1].forEach(v=>svg.append(S('text',{x:SX(v),y:H-mB+10,'text-anchor':'middle','font-size':8,fill:mu,text:v.toFixed(1)})));
  // points
  const c=color(b);
  pts.forEach(p=>{const cx=SX(p[0]),cy=SY(p[1]);
    const dot=S('circle',{cx,cy,r:2.2,fill:c,'fill-opacity':.5,stroke:cssv('--surface-1'),'stroke-width':.4});
    dot.addEventListener('mousemove',ev=>showTip('pred <b>'+p[0].toFixed(3)+'</b><br>'+ref+' <b>'+p[1].toFixed(3)+'</b>',ev));
    dot.addEventListener('mouseleave',hideTip);svg.append(dot);});
  // fit line
  if(cell.fit){const[fx0,fy0,fx1,fy1]=cell.fit;
    svg.append(S('line',{x1:SX(fx0),y1:SY(fy0),x2:SX(fx1),y2:SY(fy1),stroke:c,'stroke-width':2,'stroke-opacity':.9,'stroke-linecap':'round'}));}
  // title (own line) + stats (own line) — two lines so a long bake name never collides with ρ/r
  svg.append(S('text',{x:mL,y:12,'font-size':10.5,'font-weight':600,fill:cssv('--text-primary'),
    text:(b.name.length>29?b.name.slice(0,28)+'…':b.name)+ensTag(b)}));
  svg.append(S('text',{x:mL,y:25,'font-size':9.5,fill:cssv('--text-secondary'),
    text:'ρ '+f3(cell.srocc)+'   r '+f3(cell.plcc)+'   n='+cell.n}));
  return svg;
}

function renderScatter(){
  const host=$('#scatter');if(!host)return;host.innerHTML='';
  const ref=state.ref;const bs=visBakes();
  host.append(el('h2',{text:'Correlation scatter matrix — predicted vs '+(DATA.refLabels[ref]||ref)}));
  host.append(el('div',{class:'cap',html:'One clean scatter per (bake × corpus) for the selected reference; '
    +'bakes sit side by side per corpus so you can compare fits. ρ = canonical SROCC, r = PLCC. '
    +'Switch reference in the bar above; toggle bakes to add/remove columns.'}));
  if(!bs.length){host.append(el('p',{class:'sub',text:'no bakes selected.'}));return;}
  // corpora that carry this reference for any visible bake
  const corps=DATA.corpOrder.filter(c=>bs.some(b=>b.scatter[c]&&b.scatter[c][ref]));
  const extra=[...new Set(bs.flatMap(b=>Object.keys(b.scatter)))].filter(c=>!DATA.corpOrder.includes(c)&&bs.some(b=>b.scatter[c]&&b.scatter[c][ref]));
  [...corps,...extra].forEach(corp=>{
    host.append(el('div',{class:'corpttl',text:corp}));
    const row=el('div',{class:'scrow'});
    bs.forEach(b=>{if(b.scatter[corp]&&b.scatter[corp][ref])row.appendChild(scatterCell(b,corp,ref));});
    if(!row.children.length)row.appendChild(el('div',{class:'cap',text:'(no visible bake has '+ref+' here)'}));
    host.appendChild(row);
  });
}

// ---- cross-corpus SROCC heatmap (sequential ramp; visible bakes only)
function renderHeat(){
  const host=$('#heat');if(!host)return;host.innerHTML='';
  const bs=visBakes();if(!bs.length){return;}
  const corps=DATA.corpOrder.filter(c=>bs.some(b=>b.rank[c]));
  // train==val corpora (KADID/TID) read from the Rust-emitted flag — mark them so
  // their SROCC is not read as held-out skill (stats review Rec-6).
  const TVSET=new Set();
  DATA.bakes.forEach(b=>Object.entries(b.rank||{}).forEach(([c,r])=>{if(r&&r.train_eq_val)TVSET.add(c);}));
  const cw=62,rh=22,mL=140,mT=52;const W=mL+corps.length*cw+8,Ht=mT+bs.length*rh+8;
  const svg=S('svg',{viewBox:`0 0 ${W} ${Ht}`,width:W,height:Ht});
  corps.forEach((c,j)=>svg.append(S('text',{x:mL+j*cw+cw/2,y:mT-6,'text-anchor':'end','font-size':9.5,
    fill:TVSET.has(c)?cssv('--warn'):cssv('--text-secondary'),transform:`rotate(-32 ${mL+j*cw+cw/2} ${mT-6})`,
    text:TVSET.has(c)?c+' ⚠':c})));
  bs.forEach((b,i)=>{
    svg.append(S('text',{x:mL-6,y:mT+i*rh+rh/2+3,'text-anchor':'end','font-size':10,fill:cssv('--text-primary'),text:b.name+ensTag(b)}));
    svg.append(S('rect',{x:mL-16,y:mT+i*rh+rh/2-5,width:10,height:10,rx:2,fill:color(b)}));
    corps.forEach((c,j)=>{
      const r=b.rank[c];const v=r?r.srocc:null;
      const x=mL+j*cw,y=mT+i*rh;
      let fill=cssv('--surface-1');
      if(v!=null&&isFinite(v)){const t=Math.max(0,Math.min(1,(v-.4)/.6));
        fill='color-mix(in srgb, var(--seq-hi) '+Math.round(t*100)+'%, var(--seq-lo))';}
      const rect=S('rect',{x:x+1,y:y+1,width:cw-2,height:rh-2,rx:3,fill});
      rect.addEventListener('mousemove',ev=>showTip('<b>'+b.name+'</b> × '+c+'<br>SROCC <b>'+f3(v)+'</b>'+(r?' · n='+r.n:''),ev));
      rect.addEventListener('mouseleave',hideTip);svg.append(rect);
      svg.append(S('text',{x:x+cw/2,y:y+rh/2+3,'text-anchor':'middle','font-size':9,
        fill:(v!=null&&isFinite(v)&&(v-.4)/.6>.55)?'#fff':cssv('--text-secondary'),text:v==null?'':f3(v).replace('0.','.')}));
    });
  });
  host.append(el('h2',{text:'Cross-corpus SROCC'}),
    el('div',{class:'cap',html:'Bake × corpus, SROCC (|SROCC| for JND corpora). Sequential blue: darker = higher. '
      +'<b>⚠</b> (amber header) = KADID/TID, train==val — SROCC rewards memorization, not held-out generalization.'}),svg);
}

// ---- operating-point trade map: CID22 vs nonphoto and vs KonJND (labeled points)
function tradePanel(xc,yc,xl,yl){
  const bs=visBakes();const W=340,H=270,mL=44,mR=12,mT=16,mB=34;
  const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'background:var(--surface-1);border:1px solid var(--border);border-radius:6px'});
  const pts=bs.map(b=>({b,x:rs(b,xc),y:rs(b,yc)})).filter(p=>p.x!=null&&p.y!=null&&isFinite(p.x)&&isFinite(p.y));
  if(!pts.length){svg.append(S('text',{x:W/2,y:H/2,'text-anchor':'middle',fill:cssv('--muted'),'font-size':11,text:'no data'}));return svg;}
  const xr=[Math.min(...pts.map(p=>p.x)),Math.max(...pts.map(p=>p.x))];
  const yr=[Math.min(...pts.map(p=>p.y)),Math.max(...pts.map(p=>p.y))];
  const pad=(r)=>{const d=(r[1]-r[0])*.12||.02;return [r[0]-d,r[1]+d];};
  const[X0,X1]=pad(xr),[Y0,Y1]=pad(yr);
  const SX=v=>mL+(v-X0)/(X1-X0)*(W-mL-mR),SY=v=>H-mB-(v-Y0)/(Y1-Y0)*(H-mT-mB);
  const g=cssv('--grid'),ax=cssv('--axis'),mu=cssv('--muted');
  for(let i=0;i<=4;i++){const gx=mL+i/4*(W-mL-mR),gy=mT+i/4*(H-mT-mB);
    svg.append(S('line',{x1:gx,y1:mT,x2:gx,y2:H-mB,stroke:g,'stroke-width':.5}));
    svg.append(S('line',{x1:mL,y1:gy,x2:W-mR,y2:gy,stroke:g,'stroke-width':.5}));}
  svg.append(S('line',{x1:mL,y1:mT,x2:mL,y2:H-mB,stroke:ax,'stroke-width':1}));
  svg.append(S('line',{x1:mL,y1:H-mB,x2:W-mR,y2:H-mB,stroke:ax,'stroke-width':1}));
  [X0,X1].forEach(v=>svg.append(S('text',{x:SX(v),y:H-mB+12,'text-anchor':'middle','font-size':9,fill:mu,text:f3(v)})));
  [Y0,Y1].forEach(v=>svg.append(S('text',{x:mL-4,y:SY(v)+3,'text-anchor':'end','font-size':9,fill:mu,text:f3(v)})));
  svg.append(S('text',{x:(mL+W-mR)/2,y:H-4,'text-anchor':'middle','font-size':10,fill:cssv('--text-secondary'),text:xl}));
  svg.append(S('text',{x:12,y:(mT+H-mB)/2,'text-anchor':'middle','font-size':10,fill:cssv('--text-secondary'),transform:`rotate(-90 12 ${(mT+H-mB)/2})`,text:yl}));
  pts.forEach(p=>{const cx=SX(p.x),cy=SY(p.y);
    svg.append(S('circle',{cx,cy,r:5,fill:color(p.b),stroke:cssv('--surface-1'),'stroke-width':1.2}));
    const right=cx>mL+(W-mL-mR)*0.6;                       // flip label left near the right edge so it doesn't clip
    svg.append(S('text',{x:right?cx-8:cx+8,y:cy+3,'text-anchor':right?'end':'start','font-size':9.5,fill:cssv('--text-primary'),text:p.b.name+ensTag(p.b)}));});
  return svg;
}
function renderTrade(){
  const host=$('#trade');if(!host)return;host.innerHTML='';
  if(!visBakes().length)return;
  host.append(el('h2',{text:'Operating-point trade map'}),
    el('div',{class:'cap',text:'Upper-right = better on both. Points are directly labeled (identity is never color-alone).'}));
  const grid=el('div',{class:'grid'});
  grid.append(tradePanel('cid22','nonphoto','CID22 SROCC','non-photo SROCC'),
              tradePanel('cid22','konjnd','CID22 SROCC','KonJND |SROCC|'));
  host.appendChild(grid);
}

// ---- FULL MOHAMMADI PANEL (all six stats per corpus, per visible bake)
function renderMPanel(){
  const host=$('#mpanel');if(!host)return;host.innerHTML='';
  const bs=visBakes();if(!bs.length)return;
  const corps=DATA.corpOrder.filter(c=>DATA.bakes.some(b=>b.rank[c]));
  if(!state.mcorp||!corps.includes(state.mcorp))state.mcorp=corps[0];
  const TV=new Set();
  DATA.bakes.forEach(b=>Object.entries(b.rank||{}).forEach(([c,r])=>{if(r&&r.train_eq_val)TV.add(c);}));
  host.append(el('h2',{text:'Full Mohammadi panel'}));
  host.append(el('div',{class:'cap',html:'All six stats (Mohammadi 2025): SROCC/KROCC on raw ranks; '
    +'PLCC, OR, PWRC, Z-RMSE on the 4-param-logistic-rescaled prediction. <b>OR + Z-RMSE: lower is '
    +'better</b>; OR is a catastrophe gate, not a ranker. <b>SROCC</b> is signed (a negative = globally '
    +'inverted bake) with the bootstrap 95% CI half-width. <b>per-ref / %bwd</b> = within-image mean SROCC '
    +'and share of reference ladders ranked backwards (— when the corpus carries no ref identity). '
    +'⚠ = train==val (KADID/TID: memorization, not held-out skill).'}));
  const sel=el('div',{class:'bar',style:'margin:6px 0 10px'});
  corps.forEach(c=>{
    const b=el('button',{class:'btn',text:(TV.has(c)?c+' ⚠':c)});
    if(state.mcorp===c)b.style.cssText='font-weight:700;outline:2px solid var(--seq-hi)';
    b.onclick=()=>{state.mcorp=c;renderMPanel();};
    sel.append(b);
  });
  host.append(sel);
  const c=state.mcorp;
  const tbl=el('table',{});
  const thead=el('tr',{});
  ['bake','n','SROCC ±CI','PLCC','KROCC','OR','PWRC','Z-RMSE','per-ref','%bwd'].forEach((h,i)=>
    thead.append(el('th',{class:i===0?'lbl':'',text:h})));
  tbl.append(el('thead',{},thead));
  const tb=el('tbody',{});
  const rows=bs.filter(b=>b.rank[c]).sort((a,b)=>(b.rank[c].srocc||0)-(a.rank[c].srocc||0));
  rows.forEach(b=>{
    const r=b.rank[c];
    const sroccs=(r.srocc_signed!=null?r.srocc_signed:r.srocc);
    const ciw=r.srocc_ci?(r.srocc_ci[1]-r.srocc_ci[0])/2:null;
    const tr=el('tr',{});
    tr.append(nameInto(el('td',{class:'lbl'}),b));
    const cells=[
      r.n!=null?String(r.n):'—',
      (sroccs!=null?(sroccs>=0?'+':'')+sroccs.toFixed(4):'—')+(ciw!=null?' ±'+ciw.toFixed(3):''),
      f3(r.plcc), f3(r.krocc), r.or!=null?r.or.toFixed(4):'—', f3(r.pwrc),
      r.z_rmse!=null?r.z_rmse.toFixed(3):'—',
      r.per_ref_mean!=null?(r.per_ref_mean>=0?'+':'')+r.per_ref_mean.toFixed(4):'—',
      r.frac_negative!=null?pct(r.frac_negative):'—'];
    cells.forEach(v=>tr.append(el('td',{text:v})));
    if(sroccs!=null&&sroccs<0)tr.style.background='color-mix(in srgb, var(--serious) 18%, transparent)';
    tb.append(tr);
  });
  tbl.append(tb);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
  // 10-band SROCC grouped bars for the selected corpus (when the corpus is banded).
  const banded=rows.filter(b=>b.rank[c]&&b.rank[c].bands);
  if(banded.length){
    host.append(el('h3',{text:'10-band SROCC — '+c}));
    host.append(el('div',{class:'cap',html:'Per-band SROCC across the quality range (B0 worst → B9 best). '
      +'Dimmed = n&lt;30 (noisy; CI &gt; ±0.3 — do not rank bakes on those bands). Band SROCC is '
      +'range-restricted — B0/B9 values run low by construction; compare bakes, not bands.'}));
    const bands=banded[0].rank[c].bands.map(x=>x.band);
    const W=Math.max(560,bands.length*(banded.length*9+16)+70),H=210,mL=38,mB=26,mT=10;
    const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'max-width:100%'});
    const Y0=-0.2,Y1=1.0,SY=v=>mT+(Y1-Math.max(Y0,Math.min(Y1,v)))/(Y1-Y0)*(H-mT-mB);
    [0,0.25,0.5,0.75,1].forEach(g=>{svg.append(S('line',{x1:mL,y1:SY(g),x2:W-6,y2:SY(g),stroke:cssv('--grid'),'stroke-width':.5}));
      svg.append(S('text',{x:mL-4,y:SY(g)+3,'text-anchor':'end','font-size':8.5,fill:cssv('--muted'),text:g.toFixed(2)}));});
    svg.append(S('line',{x1:mL,y1:SY(0),x2:W-6,y2:SY(0),stroke:cssv('--axis'),'stroke-width':1}));
    bands.forEach((bn,bi)=>{
      const gx=mL+8+bi*(banded.length*9+16);
      banded.forEach((b,k)=>{
        const row=b.rank[c].bands[bi];if(!row||row.srocc==null)return;
        const x=gx+k*9,y=SY(Math.max(0,row.srocc)),y0=SY(0);
        const r=S('rect',{x,y:Math.min(y,y0),width:7,height:Math.abs(y0-y)||1,fill:color(b),opacity:row.n<30?0.35:0.95});
        r.addEventListener('mousemove',ev=>showTip('<b>'+b.name+'</b> '+bn+' n='+row.n+'<br>SROCC <b>'+f3(row.srocc)+'</b> · PLCC '+f3(row.plcc)+' · PWRC '+f3(row.pwrc)+' · Z-RMSE '+(row.z_rmse!=null?row.z_rmse.toFixed(2):'—'),ev));
        r.addEventListener('mouseleave',hideTip);svg.append(r);
      });
      svg.append(S('text',{x:gx+(banded.length*9)/2,y:H-8,'text-anchor':'middle','font-size':9,fill:cssv('--text-secondary'),text:bn}));
    });
    const bw=el('div',{style:'overflow-x:auto'});bw.append(svg);host.append(bw);
    // ---- the NUMBERS behind those bars: cross-bake per-band SROCC table.
    // Columns = bands populated somewhere in this corpus (n>0); on CID22 that drops the
    // structurally-empty B0/B1. Values come straight from rank.<corpus>.bands[] — nothing
    // is recomputed here (the fulleval JSON, i.e. zenstats, owns every statistic).
    const bandN=i=>Math.max(...banded.map(b=>{const r=b.rank[c].bands[i];return r&&r.n!=null?r.n:0;}));
    const bandS=(b,i)=>{const r=b.rank[c].bands[i];return r&&r.srocc!=null?r.srocc:null;};
    const cols=bands.map((_,i)=>i).filter(i=>bandN(i)>0);
    const scored=cols.filter(i=>banded.some(b=>bandS(b,i)!=null));
    if(cols.length&&scored.length){
      // band-profile summary: who leads the top band vs the bottom band (the finding —
      // a bake can own the near-lossless end and trail at the low end, or vice versa).
      const lead=i=>{let best=null;banded.forEach(b=>{const v=bandS(b,i);
        if(v!=null&&(best===null||v>best.v))best={nm:b.name,v};});return best;};
      const span=i=>{const vs=banded.map(b=>bandS(b,i)).filter(v=>v!=null);
        return [Math.min.apply(null,vs),Math.max.apply(null,vs)];};
      const loI=scored[0],hiI=scored[scored.length-1];
      const loL=lead(loI),hiL=lead(hiI),loS=span(loI),hiS=span(hiI);
      const sum=el('div',{class:'cap',style:'margin:8px 0 3px'});
      const put=(t,b)=>sum.append(b?el('b',{text:t}):document.createTextNode(t));
      put('Band profile ('+banded.length+' bake'+(banded.length===1?'':'s')+' shown) — ');
      put('top band '+bands[hiI]+' (n='+bandN(hiI)+')',1);
      put(' spans '+f3(hiS[0])+' → '+f3(hiS[1])+', led by ');
      put(hiL.nm+' '+f3(hiL.v),1);
      put('.  ');
      put('bottom band '+bands[loI]+' (n='+bandN(loI)+')',1);
      put(' spans '+f3(loS[0])+' → '+f3(loS[1])+', led by ');
      put(loL.nm+' '+f3(loL.v),1);
      put('. ');
      put(hiL.nm===loL.nm
        ? 'Same bake leads both ends.'
        : 'Different leaders at the two ends — this is a band PROFILE, not one ranking.');
      host.append(sum);
      const bt=el('table',{});
      const bh=el('tr',{});
      bh.append(el('th',{class:'lbl',text:'bake'}));
      cols.forEach(i=>{const t=el('th',{text:bands[i]});
        t.append(el('div',{style:'font-weight:400;font-size:9px;color:var(--muted)',text:'n='+bandN(i)}));
        bh.append(t);});
      bt.append(el('thead',{},bh));
      const bb=el('tbody',{});
      banded.forEach(b=>{
        const tr=el('tr',{});
        tr.append(nameInto(el('td',{class:'lbl'}),b));
        cols.forEach(i=>{
          const r=b.rank[c].bands[i],v=bandS(b,i),noisy=!r||r.n==null||r.n<30;
          const td=el('td',{text:v==null?'—':(noisy?'('+v.toFixed(3)+')':v.toFixed(3))});
          if(noisy)td.style.color='var(--muted)';
          tr.append(td);
        });
        bb.append(tr);
      });
      bt.append(bb);
      const btw=el('div',{style:'overflow-x:auto'});btw.append(bt);host.append(btw);
      host.append(el('div',{class:'cap',style:'margin-top:3px',html:
        'Read DOWN a column (which bake wins that band), never ACROSS one: band SROCC is '
        +'range-restricted, so every value runs low by construction and bands are not comparable '
        +'to each other. Parenthesized + dimmed = n&lt;30, which is noise (CI &gt; ±0.3) — do not '
        +'rank on it. Empty bands are omitted; on CID22 the low bands are structurally near-empty '
        +'(B0/B1 hold no pairs at all), so the low-end signal rests on a few dozen pairs.'}));
    }
  }
  // Calibration curve (binned pred → mean target) for MOS corpora from per_pair.
  const mosRows=rows.filter(b=>b.scatter[c]&&b.scatter[c].mos&&b.scatter[c].mos.pts.length>30);
  if(mosRows.length){
    host.append(el('h3',{text:'Calibration — '+c}));
    host.append(el('div',{class:'cap',text:'Binned mean MOS per predicted-score bin (15 bins). A straight rising line = well-calibrated dial; plateaus = dead zones; non-monotone = mis-calibration.'}));
    const W=430,H=250,mL=40,mB=30,mT=10,mR=10;
    const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'background:var(--surface-1);border:1px solid var(--border);border-radius:6px'});
    let xmin=1e9,xmax=-1e9;mosRows.forEach(b=>b.scatter[c].mos.pts.forEach(p=>{if(p[0]<xmin)xmin=p[0];if(p[0]>xmax)xmax=p[0];}));
    const SX=v=>mL+(v-xmin)/(xmax-xmin||1)*(W-mL-mR),SY=v=>mT+(1-v)*(H-mT-mB);
    [0,0.5,1].forEach(g=>{svg.append(S('line',{x1:mL,y1:SY(g),x2:W-mR,y2:SY(g),stroke:cssv('--grid'),'stroke-width':.5}));
      svg.append(S('text',{x:mL-4,y:SY(g)+3,'text-anchor':'end','font-size':8.5,fill:cssv('--muted'),text:g.toFixed(1)}));});
    svg.append(S('text',{x:(mL+W-mR)/2,y:H-6,'text-anchor':'middle','font-size':9.5,fill:cssv('--text-secondary'),text:'predicted score'}));
    mosRows.forEach(b=>{
      const pts=b.scatter[c].mos.pts,NB=15,acc=Array.from({length:NB},()=>[0,0]);
      pts.forEach(([x,y])=>{let i=Math.min(NB-1,Math.floor((x-xmin)/(xmax-xmin||1)*NB));acc[i][0]+=y;acc[i][1]++;});
      const line=acc.map((a,i)=>a[1]>=3?[SX(xmin+(i+0.5)/NB*(xmax-xmin)),SY(a[0]/a[1])]:null).filter(Boolean);
      if(line.length>1)svg.append(S('polyline',{points:line.map(p=>p.join(',')).join(' '),fill:'none',stroke:color(b),'stroke-width':1.8,opacity:.9}));
    });
    host.append(svg);
  }
}

// ---- PER-CODEC DIAL CURVES (median dial score vs q per codec, per visible bake)
function renderDial(){
  const host=$('#dialsec');if(!host)return;host.innerHTML='';
  const bs=visBakes().filter(b=>b.dial&&b.dial.curves&&Object.keys(b.dial.curves).length);
  if(!bs.length)return;
  host.append(el('h2',{text:'Per-codec dial curves'}));
  host.append(el('div',{class:'cap',html:'Median dial score vs grid quality per codec family (across each family\u2019s '
    +'image ladders on the densified grid; jxl x-axis = butteraugli-distance mapped to q-equiv). A good dial rises '
    +'monotonically and spans low→high. The per-codec mono% is in the tooltip — a family can be broken while the '
    +'pooled headline stays green.'}));
  const codecs=[...new Set(bs.flatMap(b=>Object.keys(b.dial.curves)))].sort();
  const grid=el('div',{style:'display:flex;flex-wrap:wrap;gap:10px'});
  codecs.forEach(cd=>{
    const W=330,H=240,mL=38,mB=28,mT=24,mR=8;
    const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'background:var(--surface-1);border:1px solid var(--border);border-radius:6px'});
    let xmin=1e9,xmax=-1e9,ymin=0,ymax=100;
    bs.forEach(b=>{const cv=b.dial.curves[cd];if(cv)cv.forEach(p=>{if(p[0]<xmin)xmin=p[0];if(p[0]>xmax)xmax=p[0];});});
    if(xmax<=xmin)return;
    const SX=v=>mL+(v-xmin)/(xmax-xmin)*(W-mL-mR),SY=v=>mT+(ymax-Math.max(ymin,Math.min(ymax,v)))/(ymax-ymin)*(H-mT-mB);
    [0,25,50,75,100].forEach(g=>{svg.append(S('line',{x1:mL,y1:SY(g),x2:W-mR,y2:SY(g),stroke:cssv('--grid'),'stroke-width':.5}));
      svg.append(S('text',{x:mL-4,y:SY(g)+3,'text-anchor':'end','font-size':8.5,fill:cssv('--muted'),text:String(g)}));});
    svg.append(S('text',{x:mL+4,y:14,'font-size':11,'font-weight':700,fill:cssv('--text-primary'),text:cd}));
    [xmin,xmax].forEach(v=>svg.append(S('text',{x:SX(v),y:H-8,'text-anchor':'middle','font-size':8.5,fill:cssv('--muted'),text:v.toFixed(0)})));
    bs.forEach(b=>{
      const cv=b.dial.curves[cd];if(!cv||cv.length<2)return;
      const pc=(b.dial.per_codec||[]).find(x=>x.codec===cd);
      const pl=S('polyline',{points:cv.map(p=>SX(p[0])+','+SY(p[2])).join(' '),fill:'none',stroke:color(b),'stroke-width':1.7,opacity:.9});
      pl.addEventListener('mousemove',ev=>showTip('<b>'+b.name+'</b> × '+cd+(pc?'<br>mono <b>'+pct(pc.mono)+'</b> · tied '+pct(pc.tied)+' · '+pc.n_curves+' ladders':''),ev));
      pl.addEventListener('mouseleave',hideTip);svg.append(pl);
    });
    grid.append(svg);
  });
  host.append(grid);
}

// ---- GATE SCORECARD (CODEC_TARGET_GOALS soft-gates per bake)
function renderGates(){
  const host=$('#gates');if(!host)return;host.innerHTML='';
  const bs=visBakes().filter(b=>b.gates&&Object.keys(b.gates).length);
  if(!bs.length)return;
  host.append(el('h2',{text:'Gate scorecard'}));
  host.append(el('div',{class:'cap',html:'CODEC_TARGET_GOALS soft-gates (1.00 = full pass). <b>weighted</b> = the '
    +'shippability gate (G1·3 + G8·2.5 + G5·1.5 + G9·1 + G-IM26·1 + G-NP·1 + G7·0.5 + G-OR·0.5) — a DIFFERENT '
    +'question from the ranking composite. G-OR is the catastrophe floor (worst-corpus outlier ratio).'}));
  const KEYS=[['g1_dynamic_range','G1 range'],['g5_hf_rank','G5 HF'],['g7_cid22','G7 CID22'],['g8_zrmse','G8 Z-RMSE'],
    ['g9_ds_auc','G9 DS-AUC'],['g_np_nonphoto','G-NP'],['g_im26_realcodec','G-IM26'],['g_or_catastrophe','G-OR'],['weighted_goal','weighted']];
  const tbl=el('table',{});
  const thead=el('tr',{});thead.append(el('th',{class:'lbl',text:'bake'}));
  KEYS.forEach(([,h])=>thead.append(el('th',{text:h})));
  tbl.append(el('thead',{},thead));
  const tb=el('tbody',{});
  bs.sort((a,b)=>(b.gates.weighted_goal||0)-(a.gates.weighted_goal||0)).forEach(b=>{
    const tr=el('tr',{});
    tr.append(nameInto(el('td',{class:'lbl'}),b));
    KEYS.forEach(([k])=>{
      const v=b.gates[k];
      const td=el('td',{text:v!=null?v.toFixed(2):'—'});
      if(v!=null){const t=Math.max(0,Math.min(1,v));
        td.style.background='color-mix(in srgb, var(--seq-hi) '+Math.round(t*55)+'%, var(--surface-1))';
        if(t>.65)td.style.color='#fff';}
      tr.append(td);
    });
    tb.append(tr);
  });
  tbl.append(tb);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
}

// ---- JXL LOOP TARGETING (2-shot / 3-shot) — fed by the jxl-encoder exact-sweep summary
// JSON (READ verbatim; the jxl-encoder analyze script is the stats owner). Shows EVERY
// loop model, including the outer arms + ssim2 which are not bakes on this board.
function renderLoop(){
  const host=$('#looptgt');if(!host)return;host.innerHTML='';
  if(!LT||!LT.models||!Object.keys(LT.models).length)return;
  const meta=LT.meta||{};const mx=meta.matrix||{};const N=ltN();
  host.append(el('h2',{text:'JXL loop targeting — 2-shot / 3-shot'}));
  host.append(el('div',{class:'cap',html:'Which model, driving the jxl-encoder zensim loop, HITS a requested target '
    +'in k encodes? '+(mx.refs||9)+' refs × targets {'+((mx.targets||[70,80,88]).join(', '))+'} = '+N+' cells; a cell '
    +'scores when the DECODED-judged result lands within ±'+(mx.within_tol!=null?mx.within_tol:2)+' of target '
    +'<b>in the arm’s OWN metric units</b> — rows are NOT unit-comparable across metrics. <b>k2/k3</b> = inner-loop '
    +'budget of 2/3 encodes; <b>emit-best</b> = best-scoring iterate kept (primary, what the scoreboard columns '
    +'show); <b>emit-last</b> = final iterate. Outer arms (<b>j2/j3</b>, marked °) re-encode outside the inner loop, '
    +'judged at outer_iter ≤ 2/3, and sit in the k2/k3 emit-last columns (an outer iterate IS its last emit). '
    +'Hover a cell for median |err|, median bytes and provenance (fresh run vs derived from a committed TSV). '
    +'Source: <code>'+(meta.source||'jxl-encoder benchmarks/zensim_loop_23shot_summary_2026-08-01.json')+'</code>.'}));
  const MODES=[['k2_emit_best','k2 emit-best'],['k3_emit_best','k3 emit-best'],['k2_emit_last','k2 emit-last'],['k3_emit_last','k3 emit-last']];
  const OUTER={k2_emit_last:'j2',k3_emit_last:'j3'};
  const cellOf=(m,mode)=>{const cs=m.cells||{};if(cs[mode])return cs[mode];
    if(m.kind==='outer'&&OUTER[mode]&&cs[OUTER[mode]])return cs[OUTER[mode]];return null;};
  const tbl=el('table',{});
  const h1=el('tr',{});
  h1.append(el('th',{class:'lbl',text:'loop model'}),el('th',{class:'lbl',text:'bake row'}));
  MODES.forEach(([,lab])=>{h1.append(el('th',{text:lab+' ±2'}),el('th',{text:'med|err|'}));});
  h1.append(el('th',{text:'med bytes (k3 best)'}));
  tbl.append(el('thead',{},h1));
  const tb=el('tbody',{});
  Object.entries(LT.models).forEach(([mk,m])=>{
    const tr=el('tr',{});
    const bakeName=(LT.modelBake&&LT.modelBake[mk])||null;
    const bk=bakeName?DATA.bakes.find(x=>x.name===bakeName):null;
    const nameTd=el('td',{class:'lbl'});
    if(bk)nameTd.append(el('span',{class:'sw',style:'display:inline-block;margin-right:5px;background:'+color(bk)}));
    nameTd.append(document.createTextNode(mk));
    tr.append(nameTd);
    tr.append(el('td',{class:'lbl',text:bakeName?bakeName:(m.bake?m.bake+' (bake not on board)':'(not a bake)')}));
    MODES.forEach(([mode])=>{
      const c=cellOf(m,mode);
      const outer=!!(m.kind==='outer'&&!(m.cells||{})[mode]&&c);
      const tdA=el('td',{text:c&&c.within2!=null?(c.within2+'/'+(c.n_cells||N)+(outer?'°':'')):'—'});
      if(c&&c.within2!=null){const t=Math.max(0,Math.min(1,c.within2/(c.n_cells||N)));
        tdA.style.background='color-mix(in srgb, var(--seq-hi) '+Math.round(t*62)+'%, var(--surface-1))';
        if(t>.6)tdA.style.color='#fff';}
      if(c){const tip='<b>'+mk+'</b> '+mode+(outer?' (outer '+OUTER[mode]+')':'')
        +'<br>within ±2: <b>'+c.within2+'/'+(c.n_cells||N)+'</b>'
        +'<br>med|err| '+(c.med_abs_err!=null?(+c.med_abs_err).toFixed(2):'—')
        +' · med bytes '+(c.med_bytes!=null?Math.round(c.med_bytes/1024)+' KB':'—')
        +(c.provenance?'<br>'+c.provenance:'');
        tdA.addEventListener('mousemove',ev=>showTip(tip,ev));tdA.addEventListener('mouseleave',hideTip);}
      tr.append(tdA);
      tr.append(el('td',{text:c&&c.med_abs_err!=null?(+c.med_abs_err).toFixed(2):'—'}));
    });
    const c3=cellOf(m,'k3_emit_best')||cellOf(m,'k3_emit_last');
    tr.append(el('td',{text:c3&&c3.med_bytes!=null?Math.round(c3.med_bytes/1024)+' KB':'—'}));
    tb.append(tr);
  });
  tbl.append(tb);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
  if(meta.notes)host.append(el('div',{class:'cap',text:'notes: '+meta.notes}));
}

// ---- MODEL DETAILS (architecture + in/out modifiers per bake, from the ZNPR itself)
function renderModels(){
  const host=$('#models');if(!host)return;host.innerHTML='';
  const bs=visBakes().filter(b=>b.model&&b.model.layers);
  if(!bs.length)return;
  host.append(el('h2',{text:'Model details'}));
  host.append(el('div',{class:'cap',html:'Read from each bake\u2019s ZNPR (structured <code>zenpredict inspect</code>): '
    +'architecture, weight dtype, INPUT modifiers (per-feature transforms + winsor guard count + scaler), and the '
    +'OUTPUT modifier — the dial calibration spline (plotted raw→dial; the top cap at 100 and any negative-tail '
    +'extension are visible in the knots). Hover a transform chip for its params.'}));
  const grid=el('div',{style:'display:flex;flex-wrap:wrap;gap:12px;align-items:stretch'});
  bs.forEach(b=>{
    const m=b.model;
    const card=el('div',{style:'border:1px solid var(--border);border-radius:8px;padding:10px 12px;'
      +'background:var(--surface-1);min-width:300px;max-width:360px;flex:1 1 300px'});
    const hd=el('div',{style:'display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap'});
    hd.append(el('span',{class:'sw',style:'display:inline-block;background:'+color(b)}),
      el('b',{text:b.name}));
    const hbd=ensBadge(b);if(hbd)hd.append(hbd);
    card.append(hd);
    // An ensemble has no single ZNPR: everything below (arch, size, transforms,
    // repro, spline) is the ANCHOR member. Say so before the numbers, not after.
    if(isEns(b)){
      const mem=(m.member_names||[]);
      const note=el('div',{style:'font-size:10px;line-height:1.4;margin:-2px 0 7px;padding:5px 7px;'
        +'border-radius:5px;background:color-mix(in srgb, var(--warn) 16%, var(--surface-1));'
        +'border:1px solid var(--border)'});
      note.append(el('b',{text:'Equal-weight ensemble of '+ensK(b)+' bakes.'}),
        document.createTextNode(' Not a shippable artifact — the fields below describe the '
          +'ANCHOR member '+(m.anchor||'?')+' only, and M3/M3a are NOT COMPUTABLE for an ensemble '
          +'(the coherence instrument loads one ZNPR). Distillation to a single bake is pending.'));
      if(mem.length){
        const det=el('details',{style:'margin-top:4px'});
        det.append(el('summary',{style:'font-size:9.5px;cursor:pointer;opacity:.75',
          text:'members ('+mem.length+')'}),
          el('div',{style:'font-size:9px;word-break:break-all;opacity:.85',text:mem.join(', ')}));
        note.append(det);
      }
      card.append(note);
    }
    // Full dim chain with hidden sizes: "720 → 128 (LeakyRelu) → 1", trainer-log style.
    // Identity on the last layer is the plain linear output head — omit the label.
    const L=m.layers||[];
    let arch=L.length?String(L[0].in):'—';
    let nparams=0;
    L.forEach((l,i)=>{
      const act=(l.activation==='Identity'&&i===L.length-1)?'':' ('+l.activation+(l.dtype!=='f32'?' '+l.dtype:'')+')';
      arch+=' → '+l.out+act;
      nparams+=l.in*l.out+l.out;
    });
    const kb=m.file_bytes?(m.file_bytes/1024).toFixed(1)+' KB':'—';
    const lines=[
      ['arch', arch+(nparams?'  ·  '+(nparams>=1000?(nparams/1000).toFixed(1)+'k':nparams)+' params':'')],
      ['size / ZNPR', kb+' · v'+(m.znpr_version||'?')],
      ['inputs', m.n_inputs+' feats · scaler '+(m.scaler&&m.scaler.present?('z-norm ('+m.scaler.n+')'):'none')],
      ['in-mods', (m.feature_transforms&&m.feature_transforms.length?m.feature_transforms.length+' transforms':'none')
        +' · '+(m.n_feature_bounds||0)+' winsor bounds'],
      ['heads', ['per_sample_alpha','hybrid','minmax'].filter(k=>m.heads&&m.heads[k]).join(', ')
        +((m.heads&&m.heads.tanh_pin_scale!=null)?' tanh-pin '+m.heads.tanh_pin_scale:'')||'none'],
      ['out-mods', (m.output_spline?('spline '+m.output_spline.n_knots+' knots'):'no spline')
        +(m.n_output_specs?(' · '+m.n_output_specs+' output_specs'):'')
        +(m.n_discrete_sets?(' · '+m.n_discrete_sets+' discrete'):'')],
    ];
    const tb=el('table',{style:'font-size:11px;width:100%'});
    lines.forEach(([k,v])=>{const tr=el('tr',{});
      tr.append(el('td',{class:'lbl',style:'opacity:.65;padding-right:8px;white-space:nowrap',text:k}),
        el('td',{text:v}));tb.append(tr);});
    card.append(tb);
    // transform chips (hover = kind + params)
    if(m.feature_transforms&&m.feature_transforms.length){
      const chips=el('div',{style:'display:flex;flex-wrap:wrap;gap:3px;margin-top:6px'});
      m.feature_transforms.slice(0,48).forEach(t=>{
        const ch=el('span',{style:'font-size:9.5px;padding:1px 5px;border-radius:8px;'
          +'background:color-mix(in srgb, var(--seq-hi) 18%, var(--surface-1));border:1px solid var(--border)',
          text:'f'+t.idx});
        ch.addEventListener('mousemove',ev=>showTip('<b>f'+t.idx+'</b> '+t.kind+'<br>params ['+(t.params||[]).map(p=>(+p).toPrecision(4)).join(', ')+']',ev));
        ch.addEventListener('mouseleave',hideTip);
        chips.append(ch);
      });
      if(m.feature_transforms.length>48)chips.append(el('span',{style:'font-size:9.5px;opacity:.6',text:'+'+(m.feature_transforms.length-48)+' more'}));
      card.append(chips);
    }
    // Reproduction provenance: source badge + seed/commit + input-parquet
    // chips (hover = path + sha256 prefix + rows) + argv in a collapsible.
    {
      const r=b.repro;
      const rep=el('div',{style:'margin-top:7px;border-top:1px dashed var(--border);padding-top:6px'});
      const badge=(txt,tone)=>el('span',{style:'font-size:9px;font-weight:700;letter-spacing:.04em;'
        +'padding:1px 6px;border-radius:8px;margin-right:6px;background:'+tone,text:txt});
      if(!r){
        rep.append(badge('NO REPRO','color-mix(in srgb, var(--serious) 30%, var(--surface-1))'),
          el('span',{style:'font-size:10px;opacity:.7',text:'no embedded zentrain.repro, no .spec.json — irreproducible without archaeology'}));
      }else{
        const emb=r.source==='embedded';
        rep.append(badge(emb?'REPRO: EMBEDDED':'REPRO: SIDECAR',
          emb?'color-mix(in srgb, var(--good) 25%, var(--surface-1))':'color-mix(in srgb, var(--warn) 25%, var(--surface-1))'));
        const bits=[];
        if(r.seed!=null)bits.push('seed '+r.seed);
        if(r.epochs!=null)bits.push(r.epochs+' ep');
        if(r.trainer_head_at_train)bits.push('@'+r.trainer_head_at_train);
        if(r.timestamp_epoch)bits.push(new Date(r.timestamp_epoch*1000).toISOString().slice(0,10));
        rep.append(el('span',{style:'font-size:10px;opacity:.85',text:bits.join(' · ')}));
        const ins=r.inputs||[];
        if(ins.length){
          const chips=el('div',{style:'display:flex;flex-wrap:wrap;gap:3px;margin-top:4px'});
          ins.forEach(inp=>{
            const ch=el('span',{style:'font-size:9.5px;padding:1px 6px;border-radius:8px;'
              +'background:color-mix(in srgb, var(--good) 12%, var(--surface-1));border:1px solid var(--border)',
              text:inp.name+(inp.rows?' ('+(inp.rows>=1000?Math.round(inp.rows/1000)+'k':inp.rows)+')':'')});
            ch.addEventListener('mousemove',ev=>showTip('<b>'+inp.name+'</b><br>'+(inp.path||'?')
              +'<br>sha256 '+String(inp.sha256||'?').slice(0,16)+'… · '+(inp.rows||'?')+' rows',ev));
            ch.addEventListener('mouseleave',hideTip);
            chips.append(ch);
          });
          rep.append(chips);
        }
        if(r.argv&&r.argv.length){
          const det=el('details',{style:'margin-top:4px'});
          det.append(el('summary',{style:'font-size:9.5px;cursor:pointer;opacity:.7',text:'reproduction command (argv)'}),
            el('pre',{style:'font-size:9px;white-space:pre-wrap;word-break:break-all;max-height:120px;'
              +'overflow-y:auto;background:var(--plane);padding:5px;border-radius:4px',text:r.argv.join(' ')}));
          rep.append(det);
        }
      }
      card.append(rep);
    }
    // spline mini-plot: raw pred (x) -> dial score (y)
    if(m.output_spline&&m.output_spline.xs&&m.output_spline.xs.length>1){
      const xs=m.output_spline.xs,ys=m.output_spline.ys;
      const W=300,H=110,mL=30,mB=18,mT=6,mR=6;
      const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(0,...ys),y1=Math.max(100,...ys);
      const SX=v=>mL+(v-x0)/(x1-x0||1)*(W-mL-mR),SY=v=>mT+(y1-v)/(y1-y0||1)*(H-mT-mB);
      const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'margin-top:6px;background:var(--plane);border-radius:5px;max-width:100%'});
      [0,50,100].forEach(g=>{if(g>=y0&&g<=y1){svg.append(S('line',{x1:mL,y1:SY(g),x2:W-mR,y2:SY(g),stroke:cssv('--grid'),'stroke-width':.5}));
        svg.append(S('text',{x:mL-3,y:SY(g)+3,'text-anchor':'end','font-size':7.5,fill:cssv('--muted'),text:String(g)}));}});
      svg.append(S('polyline',{points:xs.map((x,i)=>SX(x)+','+SY(ys[i])).join(' '),fill:'none',stroke:color(b),'stroke-width':1.6}));
      xs.forEach((x,i)=>svg.append(S('circle',{cx:SX(x),cy:SY(ys[i]),r:1.6,fill:color(b)})));
      svg.append(S('text',{x:(mL+W-mR)/2,y:H-4,'text-anchor':'middle','font-size':7.5,fill:cssv('--muted'),text:'output spline: raw → dial'}));
      card.append(svg);
    }
    grid.append(card);
  });
  host.append(grid);
}

// ---- layout + orchestration
function layout(){
  const p=$('#panels');p.innerHTML='';
  p.append(el('div',{id:'table'}),el('div',{id:'heat'}),el('div',{id:'mpanel'}),el('div',{id:'dialsec'}),el('div',{id:'looptgt'}),el('div',{id:'gates'}),el('div',{id:'models'}),el('div',{id:'trade'}),el('div',{id:'scatter'}));
}
// renderTable() returns a wrapper without an id; mountTable tags it and swaps it in.
function mountTable(){const w=renderTable();w.id='table';const cur=$('#table');cur?cur.replaceWith(w):$('#panels').prepend(w);}
function rerender(){mountTable();renderHeat();renderMPanel();renderDial();renderLoop();renderGates();renderModels();renderTrade();renderScatter();}

initRef();layout();renderBar();rerender();
if(window.matchMedia)matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>{if(!document.documentElement.getAttribute('data-theme'))rerender();});
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--best-per-day", default=None)
    ap.add_argument("--loop-targeting", default=DEFAULT_LOOP_TARGETING,
                    help="jxl-encoder 2/3-shot loop-targeting summary JSON (section omitted if absent)")
    ap.add_argument("--out", default="/mnt/v/output/zensim/reports/summer_gauntlet.html")
    a = ap.parse_args()
    bakes = load_fulleval(a.fulleval_dir, a.best_per_day)
    out, size = build_html(bakes, a.out, loop_targeting=load_loop_targeting(a.loop_targeting))
    print(f"wrote {out}  ({size // 1024} KB)  {len(bakes)} bakes")
