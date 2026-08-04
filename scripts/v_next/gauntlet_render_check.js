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
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) { console.error('FAIL: no <script> block in ' + htmlPath); process.exit(1); }
const js = m[1];

// ---------------------------------------------------------------- minimal DOM shim ----
const registry = []; // every element ever created, in creation order (query scans this)

function classesOf(e) {
  const c = (e.attrs && e.attrs.class) || e.className || '';
  return String(c).split(/\s+/).filter(Boolean);
}

function query(sel) {
  if (sel.startsWith('#')) {
    const id = sel.slice(1);
    return registry.find(e => e.id === id || (e.attrs && e.attrs.id === id)) || null;
  }
  if (sel.startsWith('.')) {
    const cl = sel.slice(1);
    return registry.find(e => classesOf(e).includes(cl)) || null;
  }
  const tag = sel.toUpperCase();
  return registry.find(e => e.tagName === tag) || null;
}

function mkEl(tag) {
  const e = {
    tagName: String(tag).toUpperCase(),
    children: [], childNodes: [], attrs: {}, style: {}, _listeners: {},
    id: undefined, className: '', textContent: '', _innerHTML: '', parentNode: null,
    offsetWidth: 100,
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
  window: { matchMedia: matchMediaShim },
  matchMedia: matchMediaShim,
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  innerWidth: 1280,
  innerHeight: 900,
  console,
};

let failed = false;
const fail = (msg) => { console.error('FAIL: ' + msg); failed = true; };

try {
  vm.createContext(sandbox);
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

if (panelsEl.children.length < 8) fail('panels not populated: ' + panelsEl.children.length + ' sections (< 8)');
const chips = registry.filter(e => e.tagName === 'LABEL' && classesOf(e).includes('chip'));
if (chips.length !== nBakes) fail('control-bar chips ' + chips.length + ' != bakes ' + nBakes);
const tables = countTag('table');
if (tables < 2) fail('expected >= 2 tables (scoreboard + Mohammadi), got ' + tables);
const rows = countTag('tr');
if (rows < nBakes + 2) fail('too few table rows: ' + rows);
if (countTag('svg') < 3) fail('too few SVG charts: ' + countTag('svg'));
const h2s = texts('h2');
if (!h2s.includes('Scoreboard')) fail('Scoreboard heading missing (h2s: ' + h2s.join(' | ') + ')');

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

if (failed) process.exit(1);

// ---------------------------------------------------------------- --dump-row ----------
if (process.argv.includes('--dump-row')) {
  const want = process.argv[process.argv.indexOf('--dump-row') + 1];
  if (!want) { console.error('--dump-row needs a bake name'); process.exit(2); }
  // text as DISPLAYED: td.textContent when set directly, else the concatenated
  // descendant text (the name cell builds swatch + text node + optional ens badge).
  const cellText = (e) => {
    if (!e) return '';
    if (e.nodeType === 3) return String(e.textContent || '');
    if (e.textContent) return String(e.textContent);
    return (e.childNodes || []).map(cellText).join('');
  };
  // Locate the scoreboard by its header, not by #table: rerender() re-mounts the wrapper
  // and the shim's query() returns the FIRST id match (the empty placeholder from layout()).
  const rowsOf = (t) => { const acc = []; (function walk(e) { if (!e || !e.children) return; if (e.tagName === 'TR') acc.push(e); e.children.forEach(walk); })(t); return acc; };
  const boards = registry.filter(e => e.tagName === 'TABLE').map(rowsOf)
    .filter(rs => rs.length && (() => { const h = rs[0].children.map(cellText); return h.includes('bake') && h.includes('composite'); })());
  if (!boards.length) { console.error('--dump-row: scoreboard table not found'); process.exit(1); }
  const trs = boards[boards.length - 1];      // the last render wins
  const head = trs[0].children.map(cellText);
  const row = trs.slice(1).find(tr => tr.children.length && cellText(tr.children[0]).trim().startsWith(want));
  if (!row) { console.error('--dump-row: no scoreboard row starting with ' + want); process.exit(1); }
  console.log('--- rendered scoreboard row: ' + want + ' ---');
  row.children.forEach((td, i) => console.log((head[i] || ('col' + i)).padEnd(16) + '  ' + cellText(td).trim()));
}

console.log('render OK: ' + nBakes + ' bakes, ' + panelsEl.children.length + ' sections, '
  + tables + ' tables, ' + rows + ' rows, ' + countTag('svg') + ' svgs'
  + (DATA && DATA.loopTargeting ? ', loop panel: ' + Object.keys(DATA.loopTargeting.models || {}).length + ' models' : ', no loop panel'));
