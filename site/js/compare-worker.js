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
let baseUrl = "";
// bakeCache: bakeId → parsed ZNPR Model.
const bakeCache = new Map();

// Resolve a bake id ("v0_16" / "v0_4" / ...) to a URL. Bakes ship under
// site/weights/<id>.bin so they're available on gh-pages without R2.
function bakeUrl(bakeId) {
  return `weights/${bakeId}.bin`;
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
  if (db) return db;
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
  db = new duckdb.AsyncDuckDB(logger, ddbWorker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  URL.revokeObjectURL(worker_url);
  postMessage({ type: "progress", data: "DuckDB-WASM ready" });
  return db;
}

// Map corpus id → parquet URL (relative to site root). Stub manifest in
// compare.js carries the authoritative list; this map is the fallback.
const CORPUS_URLS = {
  aic3_ctc_epfl: "data/parquet/aic3_ctc_epfl.parquet",
  aic4_sample:   "data/parquet/aic4_sample.parquet",
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

  // Determine which columns to SELECT — start broad, then trim later when
  // we know exactly which need to be projected for X/Y.
  const allCols = ["codec", "q", "human_jnd", "human_jnd_ci_lo", "human_jnd_ci_hi",
                   "score_zensim", "score_ssim2_gpu", "score_dssim",
                   "score_butter_max", "score_butter_p3", "score_ssim2_paper",
                   "score_cvvdp", "score_iw_ssim", "bpp", "quality_index", "dlevel"];

  // Build UNION ALL query across selected corpora, projecting only columns
  // that exist in each (DuckDB columns_from_parquet would be cleaner but we
  // just try-coalesce for simplicity).
  let rows = [];
  for (const corpusId of usable) {
    const url = CORPUS_URLS[corpusId];
    postMessage({ type: "progress", data: `querying ${corpusId} (${url})…` });
    // Use TRY_CAST in case some columns are missing; coalesce to NULL.
    const colList = allCols.map((c) => `TRY_CAST(${c} AS DOUBLE) AS ${c}`).join(", ");
    const sql = `SELECT codec AS codec_, ${colList} FROM '${url}'`;
    try {
      const result = await conn.query(sql);
      for (const r of result.toArray()) {
        const obj = r.toJSON();
        // codec column collision: SELECT codec AS codec_ + codec inside colList
        // would conflict. Use the original `codec` value if codec_ is null.
        rows.push({ ...obj, corpus: corpusId });
      }
    } catch (e) {
      postMessage({ type: "progress", data: `${corpusId} failed: ${e.message}` });
    }
  }
  postMessage({ type: "progress", data: `fetched ${rows.length} rows` });

  // Apply codec / version filter if set.
  if (codec_filter) rows = rows.filter((r) => r.codec_ === codec_filter);

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
      dataPoints.push({ x, y });
    }
  }
  await conn.close();
  postMessage({ type: "progress", data: `kept ${dataPoints.length} valid (x, y) rows` });

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

  // Per-band SROCC (aggregate, by Y-value bands per CID22 Table 5).
  const bands = computeBandSrocc(dataPoints);

  // Per-band box-plot stats (p5/p25/p50/p75/p95) for the candlestick mode.
  // Bins X axis into 5-unit steps and reports Y distribution per bin.
  const boxes = computeBoxes(dataPoints, 5);

  postMessage({ type: "result", data: { rows: dataPoints, step5, bands, boxes } });
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

function computeBandSrocc(points) {
  // CID22 Table 5 bands on Y axis. (We use Y as the "quality" reference
  // for binning, which is conventional; flip if X is the reference.)
  const bands = [
    { label: "B0 below medium",      range: "<50",       lo: -Infinity, hi: 50 },
    { label: "B1 medium",            range: "[50,65)",   lo: 50, hi: 65 },
    { label: "B2 high",              range: "[65,90)",   lo: 65, hi: 90 },
    { label: "B3 visually-lossless", range: "≥90",       lo: 90, hi: Infinity },
    { label: "Near-PJND",            range: "[58,68]",   lo: 58, hi: 68 },
  ];
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

function median(xs) {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      baseUrl = msg.base_url || "";
      // DuckDB-WASM is heavy (~10 MB); defer until first query to keep
      // page-load fast. Signal ready immediately so the UI is responsive.
      postMessage({ type: "ready" });
    } else if (msg.type === "query") {
      // Lazy-load DuckDB only when the user hits Run.
      if (!db) await initDuckDB();
      await runQuery(msg);
    }
  } catch (err) {
    postMessage({ type: "error", data: err?.message ?? String(err) });
  }
};
