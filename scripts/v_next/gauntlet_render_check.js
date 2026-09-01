#!/usr/bin/env node
'use strict';
/* DOM-shim render harness for the gauntlet offline HTML (gate 2 of the regen pipeline;
 * gate 1 is `node --check` on the extracted inline JS — both are wired in
 * gauntlet_gates.sh, which the regen MUST run before shipping the HTML).
 *
 * Usage: node gauntlet_render_check.js <summer_gauntlet.html> [--dump-row <bake-name>]
 *
 * `--dump-row` prints the RENDERED scoreboard row (header -> displayed cell text) for one
 * bake after the assertions pass. It exists so a reviewer can spot-check what the page
 * actually shows against the source verdict JSON without opening a browser — the numbers on
 * the board and the numbers in the verdict must be the same numbers.
 *
 * Why: the client JS lives inside a RAW Python string template in gauntlet.py. A raw
 * string turns \' into literal backslash+quote and one bad escape kills the entire
 * <script> parse -> silently blank page (shipped once, 2026-07-29, commit e7f929ca).
 * `node --check` catches the parse class; THIS harness catches the render class —
 * it executes the page script against a minimal hand-rolled DOM (no jsdom dependency)
 * and asserts the page actually populated: control bar chips, scoreboard rows for every
 * bake, section panels, and (when the payload carries loopTargeting) the JXL
 * loop-targeting section with one row per loop model.
 */
const fs = require('fs');
const vm = require('vm');

const htmlPath = process.argv[2];
if (!htmlPath) { console.error('usage: gauntlet_render_check.js <summer_gauntlet.html>'); process.exit(2); }
const html = fs.readFileSync(htmlPath, 'utf8');
// The page carries multiple <script> blocks since the ECharts migration (vendored
// bundle first, app script second). The harness executes only the APP block — the
// vendor bundle is parse-checked by gate 1, and the app's echarts init is guarded, so
// running it without the vendor (HAS_ECH false) exercises exactly the canvas-less path.
const blocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(x => x[1]);
if (!blocks.length) { console.error('FAIL: no <script> block in ' + htmlPath); process.exit(1); }
const js = blocks.find(b => b.includes('const DATA='));
if (!js) { console.error('FAIL: no app <script> block (const DATA=) in ' + htmlPath); process.exit(1); }

// ---------------------------------------------------------------- minimal DOM shim ----
const registry = []; // every element ever created, in creation order (query scans this)

function classesOf(e) {
  const c = (e.attrs && e.attrs.class) || e.className || '';
  return String(c).split(/\s+/).filter(Boolean);
}

// Tree-FIRST lookup (2026-08-04): a real document.querySelector only sees the attached
// tree, and the sort fix depends on that — mountTable() replaces the CURRENT #table
// wrapper, so resolving #table to the first-ever-created (now detached) node would
// re-mount into nowhere and falsely fail the click test. Walk the attached tree from
// the root; fall back to the registry for detached lookups.
let rootEl = null; // assigned after the skeleton is built below
function treeFind(pred) {
  let out = null;
  (function walk(e) {
    if (out || !e || !e.children) return;
    if (pred(e)) { out = e; return; }
    e.children.forEach(walk);
  })(rootEl);
  return out;
}
function query(sel) {
  if (sel.startsWith('#')) {
    const id = sel.slice(1);
    const pred = e => e.id === id || (e.attrs && e.attrs.id === id);
    return treeFind(pred) || registry.find(pred) || null;
  }
  if (sel.startsWith('.')) {
    const cl = sel.slice(1);
    const pred = e => classesOf(e).includes(cl);
    return treeFind(pred) || registry.find(pred) || null;
  }
  const tag = sel.toUpperCase();
  const pred = e => e.tagName === tag;
  return treeFind(pred) || registry.find(pred) || null;
}

function mkEl(tag) {
  const e = {
    tagName: String(tag).toUpperCase(),
    children: [], childNodes: [], attrs: {}, style: {}, _listeners: {},
    id: undefined, className: '', textContent: '', _innerHTML: '', parentNode: null,
    offsetWidth: 100,
    // canvas-less environment: a canvas element exists but yields NO 2d context —
    // the app's CANVAS_OK probe stays false (echarts.init skipped) and zrender's
    // measureText takes its estimation fallback during the SSR option check.
    getContext() { return null; },
    get innerHTML() { return this._innerHTML; },
    set innerHTML(v) { this._innerHTML = String(v); this.children.length = 0; this.childNodes.length = 0; },
    setAttribute(k, v) {
      this.attrs[k] = String(v);
      if (k === 'id') this.id = String(v);
      if (k === 'class') this.className = String(v);
    },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { if (c == null) return c; c.parentNode = this; this.children.push(c); this.childNodes.push(c); return c; },
    append(...cs) { cs.forEach(c => { if (c == null) return; if (typeof c === 'string') { this.childNodes.push({ nodeType: 3, textContent: c }); return; } this.appendChild(c); }); },
    prepend(...cs) { for (let i = cs.length - 1; i >= 0; i--) { const c = cs[i]; if (c == null) continue; c.parentNode = this; this.children.unshift(c); this.childNodes.unshift(c); } },
    replaceWith(n) {
      const p = this.parentNode; if (!p) return;
      const i = p.children.indexOf(this);
      if (i >= 0) { p.children[i] = n; n.parentNode = p; }
      const j = p.childNodes.indexOf(this);
      if (j >= 0) { p.childNodes[j] = n; }
    },
    addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); },
    querySelector(s) { return query(s); },
  };
  registry.push(e);
  return e;
}

const documentShim = {
  documentElement: {
    _attrs: {},
    getAttribute(k) { return k in this._attrs ? this._attrs[k] : null; },
    setAttribute(k, v) { this._attrs[k] = String(v); },
  },
  createElementNS(ns, t) { return mkEl(t); },
  createElement(t) { return mkEl(t); },
  createTextNode(t) { return { nodeType: 3, textContent: String(t) }; },
  querySelector(s) { return query(s); },
};

// The static skeleton the HTML provides before the script runs.
const root = mkEl('div'); root.setAttribute('class', 'viz-root');
rootEl = root;
const barEl = mkEl('div'); barEl.setAttribute('id', 'bar'); barEl.setAttribute('class', 'bar');
const panelsEl = mkEl('div'); panelsEl.setAttribute('id', 'panels');
const ttEl = mkEl('div'); ttEl.setAttribute('id', 'tt'); ttEl.setAttribute('class', 'tt');
root.append(barEl, panelsEl, ttEl);

const matchMediaShim = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
// NOTE: only the DOM surface is injected — the vm context is a fresh realm with its own
// Object/Array/Math/Date built-ins (shadowing them with outer-realm copies invites subtle
// cross-realm bugs; Array.isArray etc. are realm-safe as provided).
const sandbox = {
  document: documentShim,
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  innerWidth: 1280,
  innerHeight: 900,
  setTimeout, clearTimeout,        // zrender scheduling (SSR flushes synchronously)
  console,
};

let failed = false;
const fail = (msg) => { console.error('FAIL: ' + msg); failed = true; };

// Run the VENDORED ECharts bundle in the same realm first (when present) so the app
// script sees the real `echarts` global. ORDER MATTERS: the bundle loads while the
// realm has NO `window`, so its environment sniff takes the Node/SSR branch (a window
// without navigator/real DOM would send it down the browser branch and crash);
// `window` is attached only afterwards, for the app script. With no canvas in this
// shim the app's CANVAS_OK guard keeps echarts.init un-called during render — the SSR
// option check below drives echarts explicitly through its Node SSR path instead.
const vendorJs = blocks.find(b => !b.includes('const DATA=') && /apache/i.test(b.slice(0, 2000)));
try {
  vm.createContext(sandbox);
  if (vendorJs) vm.runInContext(vendorJs, sandbox, { filename: 'vendor-echarts.js', timeout: 60000 });
} catch (err) {
  console.error('FAIL: vendored ECharts bundle threw at load:');
  console.error(err && err.stack || err);
  process.exit(1);
}
sandbox.window = { matchMedia: matchMediaShim };
sandbox.matchMedia = matchMediaShim;
try {
  vm.runInContext(js, sandbox, { filename: 'gauntlet-inline.js', timeout: 60000 });
} catch (err) {
  console.error('FAIL: page script threw during render:');
  console.error(err && err.stack || err);
  process.exit(1);
}

// ---------------------------------------------------------------- assertions ----------
const payload = html.match(/const DATA=(\{[\s\S]*?\});\n/);
let DATA = null;
try { DATA = payload ? JSON.parse(payload[1]) : null; } catch (e) { DATA = null; }
if (!DATA) fail('could not re-parse the embedded DATA payload');
const nBakes = DATA ? DATA.bakes.length : 0;

const countTag = (tag) => registry.filter(e => e.tagName === tag.toUpperCase()).length;
const texts = (tag) => registry.filter(e => e.tagName === tag.toUpperCase()).map(e => e.textContent);

// ---- shared table helpers (used by the sortability tests AND --dump-row) -------------
// text as DISPLAYED: td.textContent when set directly, else the concatenated
// descendant text (the name cell builds swatch + text node + optional ens badge).
const cellText = (e) => {
  if (!e) return '';
  if (e.nodeType === 3) return String(e.textContent || '');
  if (e.textContent) return String(e.textContent);
  return (e.childNodes || []).map(cellText).join('');
};
// cellText() stops at td.textContent, so it cannot see an APPENDED child — and the ⚠
// registry badge, the ens×k marker and the dominated marker are all appended <span>s.
// deepText = own text + every descendant's = what a reader actually sees. Used by the
// badge test and by --dump-row (which otherwise printed rows with every badge stripped,
// making it useless for exactly the verification it exists for).
const deepText = (e) => {
  if (!e) return '';
  if (e.nodeType === 3) return String(e.textContent || '');
  let s = String(e.textContent || '');
  (e.childNodes || e.children || []).forEach(c => { s += deepText(c); });
  return s;
};
const rowsOf = (t) => { const acc = []; (function walk(e) { if (!e || !e.children) return; if (e.tagName === 'TR') acc.push(e); e.children.forEach(walk); })(t); return acc; };
// ATTACHED tables only (walked from #panels): a detached rebuild must not count — that
// is exactly the sortability bug shape (built the sorted table, never mounted it).
const attachedTables = (headerPred) => {
  const acc = [];
  (function walk(e) {
    if (!e || !e.children) return;
    if (e.tagName === 'TABLE') {
      const rs = rowsOf(e);
      if (rs.length) {
        const h = rs[0].children.map(cellText).map(s => String(s).trim());
        if (headerPred(h)) acc.push(rs);
      }
    }
    e.children.forEach(walk);
  })(panelsEl);
  return acc;
};

if (panelsEl.children.length < 8) fail('panels not populated: ' + panelsEl.children.length + ' sections (< 8)');
const chips = registry.filter(e => e.tagName === 'LABEL' && classesOf(e).includes('chip'));
if (chips.length !== nBakes) fail('control-bar chips ' + chips.length + ' != bakes ' + nBakes);
const tables = countTag('table');
if (tables < 2) fail('expected >= 2 tables (scoreboard + Mohammadi), got ' + tables);
const rows = countTag('tr');
if (rows < nBakes + 2) fail('too few table rows: ' + rows);
if (countTag('svg') < 1) fail('no SVG mini-plots at all (model spline/calibration should remain SVG)');
const h2s = texts('h2');
if (!h2s.includes('Scoreboard')) fail('Scoreboard heading missing (h2s: ' + h2s.join(' | ') + ')');

// ---- ECharts panels (2026-08-04 migration): the five heavyweight panels mount as
// .echart divs whose OPTION is always built (pure data, stashed on ._chartOption) even
// when no canvas exists — echarts.init itself is guarded, which is exactly the path this
// canvas-less shim exercises. Assert: vendor bundle inlined, mounts exist per panel kind
// the data implies, every mount carries a series-bearing option, and both chart theme
// variants ship (light + dark), plus the data-theme MutationObserver hookup in source.
if (!/<script id=['"]vendor-echarts['"]>/.test(html)) fail('vendored ECharts <script id=vendor-echarts> block missing');
if (!vendorJs || vendorJs.length < 500000) fail('vendored ECharts bundle missing or implausibly small');
if (DATA) {
  const mounts = registry.filter(e => classesOf(e).includes('echart'));
  if (!mounts.length) fail('no ECharts mounts (.echart) rendered');
  const badOpt = mounts.filter(e => !e._chartOption || !Array.isArray(e._chartOption.series)
    || !e._chartOption.series.length);
  if (badOpt.length) fail(badOpt.length + ' .echart mounts lack a built option with series');
  const kinds = {};
  mounts.forEach(e => { const k = (e.attrs && e.attrs['data-kind']) || '?'; kinds[k] = (kinds[k] || 0) + 1; });
  const visible = DATA.bakes.filter(b => !DATA.bakes.some(x => x.curated) || b.curated);
  const expect = [];
  if (visible.some(b => b.rank && Object.keys(b.rank).length)) expect.push('heat', 'trade');
  if (visible.some(b => b.dial && b.dial.curves && Object.keys(b.dial.curves).length)) expect.push('dial');
  if (visible.some(b => b.rank && Object.values(b.rank).some(r => r && r.bands))) expect.push('band');
  if (visible.some(b => b.scatter && Object.keys(b.scatter).length)) expect.push('scatter');
  expect.forEach(k => { if (!kinds[k]) fail('expected an ECharts "' + k + '" mount, none rendered (kinds: ' + JSON.stringify(kinds) + ')'); });
  if (kinds.trade && kinds.trade !== 2) fail('expected 2 trade-map charts, got ' + kinds.trade);
  const th = DATA.chartThemes;
  if (!th || !th.light || !th.dark) fail('DATA.chartThemes must carry light + dark variants');
  else ['surface-1', 'text-primary', 'seq-lo', 'seq-hi'].forEach(k => {
    if (!th.light[k] || !th.dark[k]) fail('chartThemes.' + k + ' missing in a variant');
  });
  if (!js.includes('MutationObserver') || !js.includes('data-theme'))
    fail('data-theme MutationObserver hookup missing from the app script');

  // ---- SSR option check: the shim proves options are BUILT; this proves the real
  // ECharts ACCEPTS them. One option per panel kind is rendered through echarts' Node
  // SSR path (svg renderer, no canvas needed) — a malformed option (bad series type,
  // broken dataZoom/visualMap config) throws or yields a trivial document here while
  // every purely-structural assert above would still pass.
  (function ssrCheck() {
    let ech = null;
    try { ech = vm.runInContext('typeof echarts!=="undefined"?echarts:null', sandbox); } catch (e) { ech = null; }
    if (!ech || typeof ech.init !== 'function') { fail('SSR check: no echarts global after vendor load'); return; }
    const seen = new Set(); const ok = [];
    mounts.forEach(m => {
      const k = (m.attrs && m.attrs['data-kind']) || '?';
      if (seen.has(k) || !m._chartOption) return;
      seen.add(k);
      try {
        const chart = ech.init(null, null, { renderer: 'svg', ssr: true, width: 520, height: 400 });
        chart.setOption(m._chartOption);
        const svg = chart.renderToSVGString();
        chart.dispose();
        if (!svg || svg.length < 2000 || svg.indexOf('<svg') < 0)
          fail('SSR check: kind "' + k + '" rendered a trivial document (' + (svg ? svg.length : 0) + ' bytes)');
        else ok.push(k);
      } catch (err) {
        fail('SSR check: echarts rejected the "' + k + '" option: ' + (err && err.message || err));
      }
    });
    if (!seen.size) fail('SSR check: no chart options reached echarts');
    else if (ok.length === seen.size)
      console.log('SSR check OK: ' + ok.sort().join(', ') + ' options render through ECharts');
  })();
}

// board curation (2026-08-04): family toggles must render when bakes carry families,
// and the registered size rule holds — non-curated cells must NOT embed scatter points.
if (DATA && DATA.bakes.some(b => b.family)) {
  const gchips = registry.filter(e => e.tagName === 'SPAN' && classesOf(e).includes('gchip'));
  if (!gchips.length) fail('bakes carry family but no family toggles (.gchip) rendered');
}
if (DATA && DATA.bakes.some(b => b.curated)) {
  const leak = DATA.bakes.filter(b => !b.curated && b.scatter && Object.keys(b.scatter).length);
  if (leak.length) fail('size rule violated — non-curated bakes embed scatter: '
    + leak.slice(0, 5).map(b => b.name).join(', ') + (leak.length > 5 ? ' …' : ''));
  const nVis = DATA.bakes.filter(b => b.curated).length;
  if (!nVis) fail('curated flags present but zero curated bakes');
}

// loop-targeting section: required iff the payload carries it
if (DATA && DATA.loopTargeting && DATA.loopTargeting.models && Object.keys(DATA.loopTargeting.models).length) {
  const nModels = Object.keys(DATA.loopTargeting.models).length;
  const h = h2s.find(t => t.startsWith('JXL loop targeting'));
  if (!h) fail('loopTargeting payload present but the JXL loop targeting section did not render');
  const loopHost = query('#looptgt');
  const loopRows = [];
  (function walk(e) { if (!e || !e.children) return; if (e.tagName === 'TR') loopRows.push(e); e.children.forEach(walk); })(loopHost);
  if (loopRows.length !== nModels + 1) fail('loop table rows ' + loopRows.length + ' != models+header ' + (nModels + 1));
  // scoreboard must carry the loop columns
  const ths = texts('th');
  if (!ths.includes('2shot ±2') || !ths.includes('3shot ±2')) fail('scoreboard loop columns (2shot/3shot ±2) missing');
} else if (DATA) {
  const h = h2s.find(t => t.startsWith('JXL loop targeting'));
  if (h) fail('no loopTargeting payload but the section rendered anyway');
}

// ---- SORTABILITY (2026-08-04 regression tests) ---------------------------------------
// User report 2026-08-04: "scoreboard is also not sortable". Root cause: th.onclick
// called renderTable() — which RETURNS a detached wrapper — instead of mountTable(),
// so the sorted table was built and thrown away (bug present since 62404415; nothing
// ever click-tested a header). These tests dispatch real header clicks and assert the
// ATTACHED tables re-order. All stat tables must sort: the scoreboard (state-based
// full re-render) and the makeSortable tables (Mohammadi panel, band, gates, loop).
const nondecr = a => a.every((v, i) => i === 0 || a[i - 1] <= v);
const nonincr = a => a.every((v, i) => i === 0 || a[i - 1] >= v);
const namesByLen = DATA ? DATA.bakes.map(b => b.name).sort((a, b) => b.length - a.length) : [];
const bakeOfRow = (tr) => {
  const t = cellText(tr.children[0]).trim();
  return namesByLen.find(n => t.startsWith(n)) || null;
};

if (DATA) (function scoreboardSortTest() {
  const isBoard = h => h.includes('bake') && h.includes('composite');
  const boards = attachedTables(isBoard);
  if (boards.length !== 1) { fail('sort test: expected exactly 1 attached scoreboard, got ' + boards.length); return; }
  // NULL composite is a real state (a peer/reference-metric row has no product_composite
  // — bake_verdict does not run on a reference metric), and the page's rule for EVERY
  // column is "nulls last in either direction". The old -1e9 sentinel encoded the
  // opposite (nulls first when ascending) and only passed because no cell was null; it
  // also never checked WHERE nulls landed. Keep them null here and assert both halves.
  const compsOf = rs => rs.slice(1).map(tr => {
    const b = DATA.bakes.find(x => x.name === bakeOfRow(tr));
    return b && b.composite != null && isFinite(b.composite) ? b.composite : null;
  });
  // nulls must form a suffix; the non-null prefix must obey `ord`.
  const nullsLast = a => a.findIndex(v => v === null) < 0
    || a.slice(a.findIndex(v => v === null)).every(v => v === null);
  const defined = a => a.filter(v => v !== null);
  const head0 = boards[0][0].children.map(cellText).map(s => String(s).trim());
  const ci = head0.indexOf('composite');
  const before = compsOf(boards[0]);
  if (before.length < 3) { fail('sort test: too few scoreboard rows to test (' + before.length + ')'); return; }
  if (!nonincr(defined(before)) || !nullsLast(before))
    fail('sort test: default scoreboard order is not composite-descending with nulls last');
  const th1 = boards[0][0].children[ci];
  if (typeof th1.onclick !== 'function') { fail('SORT REGRESSION: scoreboard header has no click handler'); return; }
  th1.onclick();     // composite is the default sort key -> this click flips to ascending
  const after1 = attachedTables(isBoard);
  if (after1.length !== 1) { fail('sort test: after click, attached scoreboards = ' + after1.length); return; }
  const a1 = compsOf(after1[0]);
  const d1 = defined(a1);
  if (!nondecr(d1) || !nullsLast(a1) || (new Set(d1).size > 1 && nonincr(d1)))
    fail('SORT REGRESSION: clicking the composite header did not re-sort the ATTACHED scoreboard '
      + '(expected ascending with nulls last; first=' + a1[0] + ' last=' + a1[a1.length - 1]
      + ' nulls=' + a1.filter(v => v === null).length + ')');
  const th2 = after1[0][0].children[ci];
  if (typeof th2.onclick === 'function') th2.onclick();  // flip back to descending
  const after2 = attachedTables(isBoard);
  const a2 = after2.length ? compsOf(after2[0]) : [];
  if (!a2.length || !nonincr(defined(a2)) || !nullsLast(a2))
    fail('SORT REGRESSION: second composite click did not restore descending order (nulls last)');
})();

if (DATA) (function statTableSortTest() {
  // Mohammadi panel (PLCC column) + gate scorecard (weighted column) exercise the
  // shared makeSortable path; the band + loop tables use the identical helper.
  const cases = [];
  if (DATA.bakes.some(b => b.rank && Object.keys(b.rank).length))
    cases.push({ label: 'Mohammadi panel', pred: h => h.includes('PLCC') && h.includes('KROCC'), col: 'PLCC' });
  if (DATA.bakes.some(b => b.gates && Object.keys(b.gates).length))
    cases.push({ label: 'gate scorecard', pred: h => h.includes('weighted') && h.includes('G1 range'), col: 'weighted' });
  cases.forEach(cs => {
    const tabs = attachedTables(cs.pred);
    if (!tabs.length) { fail('sort test: ' + cs.label + ' table not found'); return; }
    const rs = tabs[0];
    const head = rs[0].children.map(cellText).map(s => String(s).trim());
    const ci = head.indexOf(cs.col);
    const th = rs[0].children[ci];
    if (!th || typeof th.onclick !== 'function') { fail('SORT REGRESSION: ' + cs.label + ' header "' + cs.col + '" has no click handler'); return; }
    th.onclick();
    const now = attachedTables(cs.pred)[0];
    const vals = now.slice(1).map(tr => {
      const t = cellText(tr.children[ci]).trim().replace(/[%±()]/g, ' ');
      const m = t.match(/-?\d+(?:\.\d+)?/);
      return m ? parseFloat(m[0]) : null;
    }).filter(v => v != null);
    if (vals.length >= 2 && !nonincr(vals))
      fail('SORT REGRESSION: ' + cs.label + ' did not sort descending on "' + cs.col + '" after a header click');
  });
})();

if (failed) process.exit(1);

// ------------------------------------------------- registry ⚠ badge on the number -----
// The annotations registry (benchmarks/eval_annotations.json) exists so a flattered or
// superseded number is never read clean. Before 2026-08-06 the only GENERIC surface was
// the chip-picker tooltip — easy to miss on the number itself — so an entry could not
// caveat the cell it was written for without a hand-added JS rule. gauntlet.py now maps
// each scoreboard column to its fulleval dot-path (COL_FIELD) and badges any cell an
// entry's `fields` cover. This asserts that path end-to-end on real payload data: pick
// the entries that actually cover a rendered column, and require a ⚠ in that cell.
(function badgeTest() {
  if (!DATA || !DATA.annRegistry) return;                 // nothing to assert
  const COL_FIELD = { composite: 'composite', cid22: 'rank.cid22', nonphoto: 'rank.nonphoto',
    konjnd: 'rank.konjnd', aic3: 'rank.aic3', live: 'rank.live', csiq: 'rank.csiq',
    hfnl: 'rank.hfnlproxy.per_ref_mean', 'dial-mono': 'dial.mono_pct', tied: 'dial.tied_pct',
    'M3a-attr': 'm3a_coherence', 'M3-coh': 'm3_coherence' };
  const HEAD = { composite: 'composite', cid22: 'CID22', nonphoto: 'nonphoto', konjnd: 'KonJND',
    aic3: 'AIC-3', live: 'LIVE', csiq: 'CSIQ', hfnl: 'HF-NL/ref', 'dial-mono': 'dial-mono',
    tied: 'tied', 'M3a-attr': 'M3a-attr', 'M3-coh': 'M3-coh' };
  const covers = (ef, cf) => cf === ef || (cf.indexOf(ef) === 0 && cf.charAt(ef.length) === '.');
  // deepText is a shared helper (hoisted above).
  const boards = attachedTables(h => h.includes('bake') && h.includes('composite'));
  if (boards.length !== 1) { fail('badge test: expected exactly 1 attached scoreboard, got ' + boards.length); return; }
  const trs = boards[0], head = trs[0].children.map(cellText).map(s => String(s).trim());
  // Row -> bake by LONGEST-name match (namesByLen, same disambiguation the sort tests
  // use): board names nest — "W10L9_s4003" is a prefix of "W10L9_s4003_packed", and a
  // plain startsWith would test the packed twin's cell against the raw twin's entry.
  const rowByBake = Object.create(null);
  for (const tr of trs.slice(1)) {
    if (!tr.children.length) continue;
    const nm = cellText(tr.children[0]).trim();
    const hit = namesByLen.find(n => nm.startsWith(n));
    if (hit && !rowByBake[hit]) rowByBake[hit] = tr;
  }
  let checked = 0;
  for (const b of DATA.bakes) {
    for (const id of (b.annotations || [])) {
      const meta = DATA.annRegistry[id];
      if (!meta || !meta.fields) continue;
      for (const key of Object.keys(COL_FIELD)) {
        if (!meta.fields.some(ef => covers(ef, COL_FIELD[key]))) continue;
        const ci = head.indexOf(HEAD[key]);
        if (ci < 0) continue;                              // column not on this board
        const row = rowByBake[b.name];
        if (!row || !row.children[ci]) continue;           // bake not rendered / short row
        const txt = deepText(row.children[ci]).trim();
        if (!txt || txt === '—' || txt === '— (absent)') continue;  // null cells: own rule
        if (txt.indexOf('⚠') < 0) {
          fail('BADGE REGRESSION: `' + id + '` covers ' + COL_FIELD[key] + ' on bake "'
            + b.name + '" but the rendered ' + HEAD[key] + ' cell carries no ⚠ (got "' + txt + '")');
          return;
        }
        checked++;
      }
    }
  }
  if (!checked) { fail('badge test: no registry entry covered any rendered scoreboard column — '
    + 'the generic annotation-badge path is untested by this payload'); return; }
  console.log('badge check OK: ' + checked + ' registry-annotated scoreboard cells carry ⚠');
})();

// ---------------------------------------------------------------- --dump-row ----------
if (process.argv.includes('--dump-row')) {
  const want = process.argv[process.argv.indexOf('--dump-row') + 1];
  if (!want) { console.error('--dump-row needs a bake name'); process.exit(2); }
  // Locate the scoreboard by its header (helpers hoisted above). Registry scan, not
  // attachedTables: after the sort tests the board is sorted descending again, and the
  // last registry render is the live one either way.
  const boards = registry.filter(e => e.tagName === 'TABLE').map(rowsOf)
    .filter(rs => rs.length && (() => { const h = rs[0].children.map(cellText); return h.includes('bake') && h.includes('composite'); })());
  if (!boards.length) { console.error('--dump-row: scoreboard table not found'); process.exit(1); }
  const trs = boards[boards.length - 1];      // the last render wins
  const head = trs[0].children.map(cellText);
  const row = trs.slice(1).find(tr => tr.children.length && cellText(tr.children[0]).trim().startsWith(want));
  if (!row) { console.error('--dump-row: no scoreboard row starting with ' + want); process.exit(1); }
  console.log('--- rendered scoreboard row: ' + want + ' ---');
  row.children.forEach((td, i) => console.log((head[i] || ('col' + i)).padEnd(16) + '  ' + deepText(td).trim()));
}

console.log('render OK: ' + nBakes + ' bakes, ' + panelsEl.children.length + ' sections, '
  + tables + ' tables, ' + rows + ' rows, ' + countTag('svg') + ' svgs'
  + (DATA && DATA.loopTargeting ? ', loop panel: ' + Object.keys(DATA.loopTargeting.models || {}).length + ' models' : ', no loop panel'));

// ---- FAILURE PROFILE panel (2026-08-31) ----------------------------------------------
// The panel exists to say what a model gets WRONG; a panel that renders blank, or that
// draws a NOT-MEASURED cell as a zero, is worse than no panel. Assert it populated: the
// heading, one comparison row per visible bake, one card per visible bake, real finding
// rows somewhere in the set, and the NOT MEASURED / measured split matching the payload.
if (DATA) (function failurePanelTest() {
  if (!h2s.some(t => t.startsWith('Failure profile')))
    fail('Failure profile section did not render (h2s: ' + h2s.join(' | ') + ')');
  const host = query('#failures');
  if (!host || !host.children.length) fail('#failures panel is empty');
  const isFail = h => h.includes('bake') && h.includes('blockers')
    && h.some(x => x.indexOf('ladder-inv') === 0);
  const tabs = attachedTables(isFail);
  if (tabs.length !== 1) { fail('failure panel: expected 1 attached comparison table, got ' + tabs.length); return; }
  const rows = tabs[0];
  // the page's own default-visible rule: curated, not dominated, no knob-end failure
  const vis = DATA.bakes.filter(b => b.curated
    && !(b.dominated_by && b.dominated_by.length)
    && !(b.knob_end_fail && b.knob_end_fail.length));
  const nVis = vis.length || DATA.bakes.length;
  if (rows.length - 1 !== nVis)
    fail('failure comparison rows ' + (rows.length - 1) + ' != visible bakes ' + nVis);
  const hdr = rows[0].children.map(cellText).map(s => s.trim());
  const iHi = hdr.indexOf('ladder-inv q>=85');
  if (iHi < 0) fail('failure table lacks the "ladder-inv q>=85" column');
  // measured vs NOT MEASURED must follow the payload, and NOT MEASURED must never be 0%
  let nMeasured = 0, nNM = 0;
  rows.slice(1).forEach(tr => {
    const nm = bakeOfRow(tr);
    const b = DATA.bakes.find(x => x.name === nm);
    const cell = cellText(tr.children[iHi]).trim();
    const hasZones = !!(b && b.zones && b.zones.rows);
    if (hasZones) {
      nMeasured++;
      if (cell.indexOf('NOT MEASURED') >= 0)
        fail('failure table: ' + nm + ' carries zones but the cell says NOT MEASURED');
    } else {
      nNM++;
      if (cell.indexOf('NOT MEASURED') < 0)
        fail('failure table: ' + nm + ' has NO zones but the cell reads "' + cell
          + '" instead of NOT MEASURED');
    }
  });
  if (!nMeasured) fail('failure table: no visible bake carries a measured ladder-inversion split');
  // per-model cards: one per visible bake, and the visible set must produce real findings
  const cardNames = [];
  (function walk(e) {
    if (!e || !e.children) return;
    if (e.tagName === 'B' && namesByLen.indexOf(String(e.textContent || '').trim()) >= 0
        && e.parentNode && e.parentNode.parentNode
        && e.parentNode.parentNode.parentNode === host) cardNames.push(String(e.textContent).trim());
    e.children.forEach(walk);
  })(host);
  const sevSpans = [];
  (function walk(e) {
    if (!e || !e.children) return;
    const t = String(e.textContent || '').trim();
    if (e.tagName === 'SPAN' && (t === 'BLOCKER' || t === 'SERIOUS' || t === 'WATCH'))
      sevSpans.push(t);
    e.children.forEach(walk);
  })(host);
  if (!sevSpans.length)
    fail('failure panel rendered no severity-tagged finding rows at all — every visible '
      + 'model came out clean, which the board data does not support');
  if (!/what breaks, how big, where you meet it/.test(html))
    fail('failure panel heading lost its "what breaks / how big / where" framing');
  console.log('failure panel OK: ' + (rows.length - 1) + ' rows (' + nMeasured
    + ' measured, ' + nNM + ' NOT MEASURED), ' + sevSpans.length + ' findings');
})();
