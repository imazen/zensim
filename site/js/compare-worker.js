// compare-worker.js — Web Worker for DuckDB-WASM queries and stats.
//
// Responsibilities:
//   - Lazy-init DuckDB-WASM (~10 MB; one-time on first query).
//   - Register parquet HTTP-range readers for selected corpora.
//   - Run SQL to pull the (X, Y, codec, q, knob_json, feat_*) rows.
//   - If Y is a zensim variant not in the parquet, apply the 228→128→1
//     MLP forward pass against feat_* columns (using a bake .bin loaded
//     from R2 weights/, cached in `bakeCache`).
//   - Compute per-band SROCC/KROCC/PLCC/RMSE.
//   - Bin by 5-unit X-step, compute median Y per bin.
//   - Stream progress messages back to the main thread.
//
// MVP STATE (2026-05-12): skeleton with WORKING bake-load path.
// `loadBake('v0_16')` fetches `weights/v0_16.bin` and parses via mlp.js;
// a cached Model can be applied to any 228-vec via `predict()`.
// DuckDB-WASM init is sketched but not exercised against real parquets
// yet — that's the next step (needs R2 upload of codec-sweep parquets).

import { parseZnpr, predict } from "./mlp.js";

let db = null;
// dbReady: single-flight promise. Concurrent callers all await the same
// instantiate() call. Without this, the 2nd caller saw `db` already
// assigned (set before `await instantiate()` resolves) and proceeded to
// `db.connect()`, which threw "duckdb is not initialized" because the
// underlying WASM module wasn't loaded yet. Hit in production when a
// corpus-list `change` event (which triggers `list_codecs`) and a Run
// click landed within ~100 ms.
let dbReady = null;
let baseUrl = "";
// bakeCache: bakeId → parsed ZNPR Model.
const bakeCache = new Map();

// Resolve a bake id ("v0_16" / "v0_4" / ...) to a URL. Bakes ship under
// site/weights/<id>.bin so they're available on gh-pages without R2.
// Worker `fetch()` uses the page's origin as the base for relative URLs
// — but the worker is loaded via `new Worker("js/...")` so the relative
// base is the worker URL's directory. Resolve against the site root so
// `weights/<id>.bin` always lands at `<site>/weights/<id>.bin`.
function bakeUrl(bakeId) {
  return absoluteSiteUrl(`weights/${bakeId}.bin`);
}

async function loadBake(bakeId) {
  if (bakeCache.has(bakeId)) return bakeCache.get(bakeId);
  postMessage({ type: "progress", data: `loading bake ${bakeId}…` });
  const r = await fetch(bakeUrl(bakeId));
  if (!r.ok) throw new Error(`bake ${bakeId}: HTTP ${r.status}`);
  const bytes = await r.arrayBuffer();
  const model = parseZnpr(bytes);
  bakeCache.set(bakeId, model);
  return model;
}

// Map an axis metric id to a bake id, or null if the metric is a
// parquet-stored column (e.g. score_ssim2 already in the row).
function bakeIdForMetric(metric) {
  // Pattern: "score_zensim_v0_NN" → "v0_NN".
  const m = metric.match(/^score_zensim_(v0_\d+)$/);
  return m ? m[1] : null;
}

async function initDuckDB() {
  // Single-flight: all concurrent callers await the same in-flight init
  // promise. Critical because `instantiate()` is async — without this,
  // caller A sets `db = new AsyncDuckDB(...)` and then awaits `instantiate`;
  // caller B sees `db` truthy, returns immediately, then `db.connect()`
  // throws "duckdb is not initialized" because the WASM module isn't loaded.
  if (dbReady) return dbReady;
  dbReady = (async () => {
    postMessage({ type: "progress", data: "loading DuckDB-WASM…" });
    // jsDelivr ESM build; pinned version chosen at commit time.
    const duckdb = await import("https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.29.0/+esm");
    const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
    const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);
    const worker_url = URL.createObjectURL(
      new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" }),
    );
    const ddbWorker = new Worker(worker_url);
    const logger = new duckdb.ConsoleLogger();
    const localDb = new duckdb.AsyncDuckDB(logger, ddbWorker);
    await localDb.instantiate(bundle.mainModule, bundle.pthreadWorker);
    URL.revokeObjectURL(worker_url);
    db = localDb; // assign only after instantiate() resolves
    postMessage({ type: "progress", data: "DuckDB-WASM ready" });
    return db;
  })();
  return dbReady;
}

// Map corpus id → parquet URL (relative to site root). Stub manifest in
// compare.js carries the authoritative list; this map is the fallback.
const R2_BASE = "https://zentrain-r2.imazen.org/zensim-compare-site";

// DuckDB-WASM's httpfs reader needs ABSOLUTE URLs. Worker-relative paths
// like "data/parquet/x.parquet" fail with "No files found that match the
// pattern" because DuckDB resolves the URL in its own internal worker
// context. Resolve in-repo paths against the parent page's origin.
// `self.location` in a worker is the worker script URL; strip the worker
// script filename to get the site root.
const SITE_BASE = (() => {
  try {
    // self.location.href = "http://host:port/js/compare-worker.js"
    // We want "http://host:port/"
    const u = new URL(self.location.href);
    // Strip everything after the last "/js/" so site root sits one level up.
    const i = u.pathname.lastIndexOf("/js/");
    u.pathname = i >= 0 ? u.pathname.slice(0, i + 1) : "/";
    u.search = ""; u.hash = "";
    return u.toString();
  } catch (_) {
    return "/";
  }
})();
function absoluteSiteUrl(relPath) {
  if (/^https?:\/\//.test(relPath)) return relPath;
  return SITE_BASE + relPath.replace(/^\//, "");
}

const CORPUS_URLS = {
  // In-repo human-rated parquets (small, gh-pages cap-OK).
  aic3_ctc_epfl: absoluteSiteUrl("data/parquet/aic3_ctc_epfl.parquet"),
  aic4_sample:   absoluteSiteUrl("data/parquet/aic4_sample.parquet"),
  cid22:         absoluteSiteUrl("data/parquet/cid22.parquet"),
  kadid10k:      absoluteSiteUrl("data/parquet/kadid.parquet"),
  tid2013:       absoluteSiteUrl("data/parquet/tid.parquet"),
  // R2-hosted codec-sweep parquets — carry feat_0..feat_299 for JS-MLP.
  v12_zenavif: `${R2_BASE}/parquets/codec-sweeps/unified_v12_zenavif.parquet`,
  v12_zenjxl:  `${R2_BASE}/parquets/codec-sweeps/unified_v12_zenjxl.parquet`,
  v12_zenwebp: `${R2_BASE}/parquets/codec-sweeps/unified_v12_zenwebp.parquet`,
  v13_zenjpeg: `${R2_BASE}/parquets/codec-sweeps/unified_v13_zenjpeg.parquet`,
  v14_zenpng:  `${R2_BASE}/parquets/codec-sweeps/unified_v14_zenpng.parquet`,
  v15r_zenjpeg:  `${R2_BASE}/parquets/codec-sweeps/unified_v15r_zenjpeg.parquet`,
  v15rc_zenjpeg: `${R2_BASE}/parquets/codec-sweeps/unified_v15rc_zenjpeg.parquet`,
};

async function runQuery(msg) {
  const { corpora, x_metric, y_metric, codec_filter, version_filter } = msg;
  postMessage({ type: "progress", data: `query: corpora=${corpora.join(",")} X=${x_metric} Y=${y_metric}` });

  // Pre-load any bakes the X or Y axis needs (cached after first hit).
  for (const m of [x_metric, y_metric]) {
    const id = bakeIdForMetric(m);
    if (id) {
      try {
        await loadBake(id);
        postMessage({ type: "progress", data: `bake ${id} ready` });
      } catch (e) {
        postMessage({ type: "progress", data: `bake ${id} load failed: ${e.message}` });
      }
    }
  }

  // Real DuckDB-WASM path: query each selected corpus's parquet, union, project
  // X+Y columns, return rows. If a metric is a JS-MLP variant
  // (score_zensim_v0_X), we ALSO pull feat_0..feat_227 columns and apply the
  // bake per-row in this worker.
  const usable = corpora.filter((c) => CORPUS_URLS[c]);
  if (usable.length === 0) {
    postMessage({ type: "error", data: `no in-repo parquets selected (R2-hosted corpora not yet enabled). Pick AIC-3 or AIC-4.` });
    return;
  }

  await initDuckDB();
  const conn = await db.connect();

  // Per-corpus schema cache: map parquet URL → Set of column names.
  // First time a URL is seen, run DESCRIBE; cached for the worker session.
  if (!self._schemaCache) self._schemaCache = new Map();
  const schemaCache = self._schemaCache;
  async function getSchema(url) {
    if (schemaCache.has(url)) return schemaCache.get(url);
    const r = await conn.query(`DESCRIBE SELECT * FROM '${url}' LIMIT 0`);
    const cols = new Set();
    for (const row of r.toArray()) cols.add(String(row.toJSON().column_name));
    schemaCache.set(url, cols);
    return cols;
  }

  // Identifier columns we want explicitly (need exact-name dispatch in JS).
  // Every other "metric-shaped" column — anything starting with `score_`,
  // `human_`, `feat_`, or matching `bpp`/`q`/`quality_index`/`dlevel`/
  // `encoded_bytes` — is projected dynamically from the parquet schema.
  //
  // Previously this was a hard-coded `allCols` list; pre-computed score
  // columns like `score_zensim_v0_16` (added per-bake to the cid22/aic3/
  // aic4 parquets over time) were silently dropped because they weren't
  // in the list, so `r.score_zensim_v0_16` was always undefined and the
  // "kept 0 valid (x, y) rows" filter ate every row. Dynamic projection
  // means new bake columns surface automatically as the parquets are
  // updated.
  const ID_COLS = ["codec", "q", "version", "image_name", "image_path",
                   "knob_tuple_json", "ref_basename"];
  const buildAllCols = (schemaCols) => {
    const cols = new Set(ID_COLS.filter(c => schemaCols.has(c)));
    for (const c of schemaCols) {
      if (c.startsWith("score_") || c.startsWith("human_") || c.startsWith("feat_")
          || c === "bpp" || c === "quality_index" || c === "dlevel" || c === "encoded_bytes") {
        cols.add(c);
      }
    }
    return Array.from(cols);
  };

  let rows = [];
  for (const corpusId of usable) {
    const url = CORPUS_URLS[corpusId];
    postMessage({ type: "progress", data: `querying ${corpusId} (${url})…` });
    let schemaCols;
    try {
      schemaCols = await getSchema(url);
    } catch (e) {
      postMessage({ type: "progress", data: `${corpusId} schema-fetch failed: ${e.message}` });
      continue;
    }
    // Project columns the parquet actually carries (dynamic per-parquet —
    // discovers new bake-pre-computed columns automatically without code
    // changes). Numeric ones get TRY_CAST to DOUBLE; codec / version /
    // image_name stay text.
    const numericPrefix = (c) => c.startsWith("score_") || c.startsWith("human_") ||
                                  c.startsWith("feat_") ||
                                  c === "bpp" || c === "q" || c === "quality_index" ||
                                  c === "dlevel" || c === "encoded_bytes";
    const projectCols = buildAllCols(schemaCols);
    if (projectCols.length === 0) {
      postMessage({ type: "progress", data: `${corpusId} has no known columns; skipping` });
      continue;
    }
    const colList = projectCols.map((c) =>
      numericPrefix(c) ? `TRY_CAST(${c} AS DOUBLE) AS ${c}` : `${c}`
    ).join(", ");
    const sql = `SELECT ${colList} FROM '${url}'`;
    try {
      const result = await conn.query(sql);
      for (const r of result.toArray()) {
        const obj = r.toJSON();
        rows.push({ ...obj, corpus: corpusId });
      }
    } catch (e) {
      postMessage({ type: "progress", data: `${corpusId} query failed: ${e.message}` });
    }
  }
  postMessage({ type: "progress", data: `fetched ${rows.length} rows` });

  // Apply codec / version filter if set. `codec` is the only column we keep
  // as text (per `numericPrefix` filter above), so simple === works.
  if (codec_filter) {
    rows = rows.filter((r) => r.codec === codec_filter);
  }
  if (version_filter) {
    rows = rows.filter((r) => r.version === version_filter || r.knob_tuple_json === version_filter);
  }

  // Compute X and Y per row.
  const bakeId = bakeIdForMetric(y_metric);
  const bakeModel = bakeId ? bakeCache.get(bakeId) : null;

  // For JS-MLP scoring we need feat_0..feat_227 — but the AIC parquets DON'T
  // carry feature columns. Surface that.
  if (bakeModel) {
    const hasFeat = rows.length > 0 && rows[0].feat_0 !== undefined;
    if (!hasFeat) {
      postMessage({ type: "progress", data: `note: ${y_metric} needs feat_0..feat_227 from the parquet; AIC corpora don't carry them. Falling back to score_zensim.` });
    }
  }

  const dataPoints = [];
  // Track per-row curve-key + q so we can apply soft-iso BY CURVE after
  // scoring. Soft-iso is the production default for codec-sweep contexts
  // (cycle-11: drops non-mono 5.5–6.3% → 0% with SROCC cost ≤0.0008 on
  // every V_X bake measured). Only applies to bake-scored Y axes where
  // curve grouping makes sense; gracefully no-ops when curves are <2 pts
  // or knob/image are absent (e.g. AIC corpora).
  for (const r of rows) {
    const x = (x_metric in r) ? r[x_metric] : (x_metric === "q" ? r.q : null);
    let y;
    if (bakeModel && r.feat_0 !== undefined) {
      const features = new Float32Array(228);
      for (let k = 0; k < 228; k++) features[k] = r[`feat_${k}`] ?? 0;
      y = predict(bakeModel, features)[0];
    } else {
      y = (y_metric in r) ? r[y_metric] : null;
    }
    if (x != null && y != null && Number.isFinite(x) && Number.isFinite(y)) {
      const curveKey = (r.image_path ?? r.image_name ?? "?") + "|" +
                       (r.codec ?? "?") + "|" +
                       (r.knob_tuple_json ?? "");
      const q = Number.isFinite(r.q) ? r.q : Number.isFinite(r.quality_index) ? r.quality_index : null;
      dataPoints.push({ x, y, curveKey, q });
    }
  }
  await conn.close();
  postMessage({ type: "progress", data: `kept ${dataPoints.length} valid (x, y) rows` });

  // Soft-iso default-on: only for bake-scored Y axes (bakeModel != null)
  // because the bake output is the only Y we control. Reference metrics
  // (ssim2/butter/dssim/MOS) are passed through unchanged — they are
  // ground truth or paper-reported, not ours to smooth.
  if (bakeModel) {
    const beforeViol = countCurveViolations(dataPoints);
    applySoftIsoPerCurve(dataPoints);
    const afterViol = countCurveViolations(dataPoints);
    postMessage({ type: "progress",
      data: `soft-iso applied to ${y_metric}: non-mono ${beforeViol.rate.toFixed(2)}% → ${afterViol.rate.toFixed(2)}% (${beforeViol.fixed} of ${beforeViol.pairs} pairs corrected)` });
  }

  // step-5 binning over the X range (anchored to multiples of 5).
  const bins = new Map();
  for (const p of dataPoints) {
    const b = Math.floor(p.x / 5) * 5;
    if (!bins.has(b)) bins.set(b, []);
    bins.get(b).push(p.y);
  }
  const step5 = Array.from(bins.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([x, ys]) => ({ x: x + 2.5, median_y: median(ys) }));

  // Per-band SROCC. CLAUDE.md "Per-band reporting rule" (2026-05-14)
  // mandates the 10-band width-10 grid as the primary release gate; the
  // legacy 4-band CID22 Table 5 cuts are reported alongside for paper
  // comparison continuity.
  const { bands10, bandsLegacy } = computeBandSrocc(dataPoints);
  // `bands` retains the legacy 4-band shape for back-compat with the
  // existing #band-table renderer; `bands10` is the new 10-band table.
  const bands = bandsLegacy;

  // Per-band box-plot stats (p5/p25/p50/p75/p95) for the candlestick mode.
  // Bins X axis into 5-unit steps and reports Y distribution per bin.
  const boxes = computeBoxes(dataPoints, 5);

  postMessage({ type: "result", data: { rows: dataPoints, step5, bands, bands10, boxes } });
}

function quantile(sorted, q) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function computeBoxes(points, binWidth) {
  const bins = new Map();
  for (const p of points) {
    const b = Math.floor(p.x / binWidth) * binWidth;
    if (!bins.has(b)) bins.set(b, []);
    bins.get(b).push(p.y);
  }
  return Array.from(bins.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([x, ys]) => {
      const s = [...ys].sort((a, b) => a - b);
      return {
        x_lo: x,
        x_hi: x + binWidth,
        x_mid: x + binWidth / 2,
        n: s.length,
        p5:  quantile(s, 0.05),
        p25: quantile(s, 0.25),
        p50: quantile(s, 0.50),
        p75: quantile(s, 0.75),
        p95: quantile(s, 0.95),
      };
    });
}

function spearman(xs, ys) {
  if (xs.length < 5) return NaN;
  const n = xs.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  const rank = (vals) => {
    const sorted = [...idx].sort((a, b) => vals[a] - vals[b]);
    const r = new Array(n);
    let i = 0;
    while (i < n) {
      let j = i + 1;
      while (j < n && vals[sorted[j]] === vals[sorted[i]]) j++;
      const avg = (i + j - 1) / 2 + 1;
      for (let k = i; k < j; k++) r[sorted[k]] = avg;
      i = j;
    }
    return r;
  };
  const rx = rank(xs);
  const ry = rank(ys);
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += rx[i]; my += ry[i]; }
  mx /= n; my /= n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = rx[i] - mx, dy = ry[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  if (dx2 === 0 || dy2 === 0) return NaN;
  return num / Math.sqrt(dx2 * dy2);
}

// Width-10 grid mandated by zensim CLAUDE.md "Per-band reporting rule"
// (2026-05-14). Tiles 0..100 in 10 buckets so band-level pathologies in
// any 10-zq band are visible. Near-PJND (58..68) overlaps B5+B6 and is
// reported as an extra sub-band because the KonJND PJND mean lands here.
const BANDS_10 = [
  { label: "B0",        range: "[0,10)",   lo: 0,  hi: 10 },
  { label: "B1",        range: "[10,20)",  lo: 10, hi: 20 },
  { label: "B2",        range: "[20,30)",  lo: 20, hi: 30 },
  { label: "B3",        range: "[30,40)",  lo: 30, hi: 40 },
  { label: "B4",        range: "[40,50)",  lo: 40, hi: 50 },
  { label: "B5",        range: "[50,60)",  lo: 50, hi: 60 },
  { label: "B6",        range: "[60,70)",  lo: 60, hi: 70 },
  { label: "B7",        range: "[70,80)",  lo: 70, hi: 80 },
  { label: "B8",        range: "[80,90)",  lo: 80, hi: 90 },
  { label: "B9",        range: "[90,100]", lo: 90, hi: 100.0001 },
  { label: "Near-PJND", range: "[58,68]",  lo: 58, hi: 68 },
];

// Legacy 4-band CID22 Table 5 cuts (Sneyers/Ben Baruch/Vaxman 2023).
// Reported alongside the 10-band grid for paper-comparison continuity.
const BANDS_LEGACY = [
  { label: "B0 below medium",      range: "<50",       lo: -Infinity, hi: 50 },
  { label: "B1 medium",            range: "[50,65)",   lo: 50, hi: 65 },
  { label: "B2 high",              range: "[65,90)",   lo: 65, hi: 90 },
  { label: "B3 visually-lossless", range: "≥90",       lo: 90, hi: Infinity },
  { label: "Near-PJND",            range: "[58,68]",   lo: 58, hi: 68 },
];

function computeBandsForGrid(points, bands) {
  for (const b of bands) {
    const sub = points.filter((p) => p.y >= b.lo && p.y < b.hi);
    const xs = sub.map((p) => p.x);
    const ys = sub.map((p) => p.y);
    b.n = sub.length;
    b.srocc = spearman(xs, ys);
    // Pearson (PLCC) on raw values
    if (sub.length >= 5) {
      let mx = 0, my = 0;
      for (let i = 0; i < sub.length; i++) { mx += xs[i]; my += ys[i]; }
      mx /= sub.length; my /= sub.length;
      let num = 0, dx2 = 0, dy2 = 0, rmse = 0;
      for (let i = 0; i < sub.length; i++) {
        const dx = xs[i] - mx, dy = ys[i] - my;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
        const err = xs[i] - ys[i];
        rmse += err * err;
      }
      b.plcc = (dx2 === 0 || dy2 === 0) ? NaN : num / Math.sqrt(dx2 * dy2);
      b.rmse = Math.sqrt(rmse / sub.length);
    } else {
      b.plcc = NaN; b.rmse = NaN;
    }
    b.krocc = NaN; // Kendall TODO (more expensive O(n^2))
  }
  return bands;
}

function computeBandSrocc(points) {
  // Deep-clone the band definitions per call so successive computations
  // don't see stale .n/.srocc/.plcc from prior queries.
  const clone = (arr) => arr.map(b => ({ ...b }));
  const bands10 = computeBandsForGrid(points, clone(BANDS_10));
  const bandsLegacy = computeBandsForGrid(points, clone(BANDS_LEGACY));
  return { bands10, bandsLegacy };
}

function median(xs) {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

// Per-curve running-extreme projection on the Y axis. For each
// (image, codec, knob) group with ≥2 points, sort by q ascending and
// push Y to be monotonic in q. Direction (running-max vs running-min)
// is chosen per curve by which raw direction has fewer violations.
// Mutates dataPoints in place by overwriting `.y`.
function applySoftIsoPerCurve(dataPoints) {
  const groups = new Map();
  for (let i = 0; i < dataPoints.length; i++) {
    const p = dataPoints[i];
    if (!p.curveKey || !Number.isFinite(p.q)) continue;
    if (!groups.has(p.curveKey)) groups.set(p.curveKey, []);
    groups.get(p.curveKey).push(i);
  }
  for (const [, idxs] of groups) {
    if (idxs.length < 2) continue;
    idxs.sort((a, b) => dataPoints[a].q - dataPoints[b].q);
    const ys = idxs.map((i) => dataPoints[i].y);
    let viDist = 0, viScore = 0;
    for (let k = 1; k < ys.length; k++) {
      if (ys[k] > ys[k - 1]) viDist++;
      if (ys[k] < ys[k - 1]) viScore++;
    }
    const distanceMode = viDist <= viScore;
    if (distanceMode) {
      let runningMin = ys[0];
      for (let k = 1; k < ys.length; k++) {
        if (ys[k] > runningMin) ys[k] = runningMin;
        else runningMin = ys[k];
      }
    } else {
      let runningMax = ys[0];
      for (let k = 1; k < ys.length; k++) {
        if (ys[k] < runningMax) ys[k] = runningMax;
        else runningMax = ys[k];
      }
    }
    for (let k = 0; k < idxs.length; k++) dataPoints[idxs[k]].y = ys[k];
  }
}

// Count curve-level adjacent-q violations for diagnostic reporting.
// Returns { pairs, fixed, rate } where fixed is the smaller of the two
// direction counts (the "true" non-mono in the inferred sign convention).
function countCurveViolations(dataPoints) {
  const groups = new Map();
  for (const p of dataPoints) {
    if (!p.curveKey || !Number.isFinite(p.q)) continue;
    if (!groups.has(p.curveKey)) groups.set(p.curveKey, []);
    groups.get(p.curveKey).push(p);
  }
  let pairs = 0, fixed = 0;
  for (const [, arr] of groups) {
    if (arr.length < 2) continue;
    arr.sort((a, b) => a.q - b.q);
    let viDist = 0, viScore = 0;
    for (let k = 1; k < arr.length; k++) {
      if (arr[k].y > arr[k - 1].y) viDist++;
      if (arr[k].y < arr[k - 1].y) viScore++;
    }
    pairs += arr.length - 1;
    fixed += Math.min(viDist, viScore);
  }
  return { pairs, fixed, rate: pairs ? (fixed / pairs * 100) : 0 };
}

async function runLookup(msg) {
  const { corpora, x_metric, y_metric, target_y, tolerance } = msg;
  const usable = corpora.filter((c) => CORPUS_URLS[c]);
  if (usable.length === 0) {
    postMessage({ type: "lookup_result", data: { groups: [] } });
    return;
  }
  postMessage({ type: "progress", data: `lookup: Y=${y_metric} target=${target_y} ±${tolerance}` });
  await initDuckDB(); // ensure WASM is fully loaded before connect()
  const conn = await db.connect();
  // Group by (codec, knob_tuple_json), report n / median Y / median X /
  // median encoded_bytes for rows where |y - target| ≤ tolerance.
  const colExpr = (name, fallback) => `TRY_CAST(${name} AS DOUBLE)`;
  const allRows = [];
  for (const corpusId of usable) {
    const url = CORPUS_URLS[corpusId];
    // Pull only the rows we need (post-filter in JS so we can handle
    // missing columns gracefully across corpora with different schemas).
    const sql = `
      SELECT codec,
             ${colExpr(x_metric)} AS x,
             ${colExpr(y_metric)} AS y,
             ${colExpr("encoded_bytes")} AS bytes,
             TRY_CAST(knob_tuple_json AS VARCHAR) AS knob_tuple_json
        FROM '${url}'
       WHERE ${colExpr(y_metric)} IS NOT NULL
         AND ABS(${colExpr(y_metric)} - ${Number(target_y)}) <= ${Number(tolerance)}`;
    try {
      const result = await conn.query(sql);
      for (const row of result.toArray()) allRows.push(row.toJSON());
    } catch (e) {
      postMessage({ type: "progress", data: `${corpusId} lookup failed: ${e.message}` });
    }
  }
  await conn.close();

  // Group by (codec, knob_tuple_json).
  const groups = new Map();
  for (const r of allRows) {
    const key = `${r.codec || "?"}|${r.knob_tuple_json || ""}`;
    if (!groups.has(key)) groups.set(key, { codec: r.codec || "?", version: r.knob_tuple_json || "", xs: [], ys: [], bs: [] });
    const g = groups.get(key);
    if (Number.isFinite(r.x)) g.xs.push(r.x);
    if (Number.isFinite(r.y)) g.ys.push(r.y);
    if (Number.isFinite(r.bytes)) g.bs.push(r.bytes);
  }
  const sortedYs = (arr) => [...arr].sort((a, b) => a - b);
  const mid = (arr) => arr.length === 0 ? null : sortedYs(arr)[Math.floor(arr.length / 2)];
  const out = Array.from(groups.values()).map((g) => ({
    codec: g.codec, version: g.version, n: g.ys.length,
    y_median: mid(g.ys), x_median: mid(g.xs),
    bytes_median: mid(g.bs),
  })).sort((a, b) => b.n - a.n);
  postMessage({ type: "lookup_result", data: { groups: out } });
}

async function listCorpusCodecs(corpora) {
  const usable = corpora.filter((c) => CORPUS_URLS[c]);
  if (usable.length === 0) {
    postMessage({ type: "codecs", data: { codecs: [], versions: [] } });
    return;
  }
  await initDuckDB(); // single-flight; safe to call concurrently
  const conn = await db.connect();
  // Per-corpus schema cache lives on `self._schemaCache` (initialized in
  // runQuery). Reuse so we don't fetch the schema twice for the same URL.
  if (!self._schemaCache) self._schemaCache = new Map();
  const codecs = new Set();
  const versions = new Set();
  for (const corpusId of usable) {
    const url = CORPUS_URLS[corpusId];
    try {
      // Fetch schema first so we don't query columns that don't exist
      // (the AIC parquets lack knob_tuple_json + version; the prior
      // try/catch worked but still emitted Binder Error noise to console).
      let schemaCols = self._schemaCache.get(url);
      if (!schemaCols) {
        const dr = await conn.query(`DESCRIBE SELECT * FROM '${url}' LIMIT 0`);
        schemaCols = new Set();
        for (const row of dr.toArray()) schemaCols.add(String(row.toJSON().column_name));
        self._schemaCache.set(url, schemaCols);
      }
      if (schemaCols.has("codec")) {
        const r1 = await conn.query(`SELECT DISTINCT codec FROM '${url}' WHERE codec IS NOT NULL`);
        for (const row of r1.toArray()) codecs.add(String(row.toJSON().codec));
      }
      if (schemaCols.has("knob_tuple_json")) {
        const r2 = await conn.query(`SELECT DISTINCT knob_tuple_json FROM '${url}' WHERE knob_tuple_json IS NOT NULL`);
        for (const row of r2.toArray()) {
          const k = row.toJSON().knob_tuple_json;
          if (k && k !== "{}") versions.add(String(k));
        }
      } else if (schemaCols.has("version")) {
        const r2 = await conn.query(`SELECT DISTINCT version FROM '${url}' WHERE version IS NOT NULL`);
        for (const row of r2.toArray()) {
          const v = row.toJSON().version;
          if (v) versions.add(String(v));
        }
      }
    } catch (e) {
      postMessage({ type: "progress", data: `codec-enum ${corpusId} failed: ${e.message}` });
    }
  }
  await conn.close();
  postMessage({ type: "codecs", data: { codecs: Array.from(codecs).sort(), versions: Array.from(versions).sort() } });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      baseUrl = msg.base_url || "";
      // DuckDB-WASM is heavy (~10 MB); defer until first query to keep
      // page-load fast. Signal ready immediately so the UI is responsive.
      postMessage({ type: "ready" });
    } else if (msg.type === "list_codecs") {
      await listCorpusCodecs(msg.corpora || []);
    } else if (msg.type === "lookup") {
      await initDuckDB(); // single-flight; safe to call concurrently
      await runLookup(msg);
    } else if (msg.type === "query") {
      // Lazy-load DuckDB only when the user hits Run.
      await initDuckDB(); // single-flight; safe to call concurrently
      await runQuery(msg);
    }
  } catch (err) {
    postMessage({ type: "error", data: err?.message ?? String(err) });
  }
};
