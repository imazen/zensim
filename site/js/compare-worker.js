// compare-worker.js — Web Worker for DuckDB-WASM queries and stats.
//
// Responsibilities:
//   - Lazy-init DuckDB-WASM (~10 MB; one-time on first query).
//   - Register parquet HTTP-range readers for selected corpora.
//   - Run SQL to pull the (X, Y, codec, q, knob_json, feat_*) rows.
//   - If Y is a zensim variant not in the parquet, apply the 228→128→1
//     MLP forward pass against feat_* columns (using a bake .bin loaded
//     from R2 weights/).
//   - Compute per-band SROCC/KROCC/PLCC/RMSE.
//   - Bin by 5-unit X-step, compute median Y per bin.
//   - Stream progress messages back to the main thread.
//
// MVP STATE (2026-05-12): skeleton only. The DuckDB init is sketched but
// not exercised against real parquets yet. The MLP forward pass and
// statistics are left as TODOs; subsequent commits fill them in.

let db = null;
let baseUrl = "";

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

async function runQuery(msg) {
  const { corpora, x_metric, y_metric, codec_filter, version_filter } = msg;
  postMessage({ type: "progress", data: `query: corpora=${corpora.join(",")} X=${x_metric} Y=${y_metric}` });

  // PLACEHOLDER: until real parquets are uploaded to R2, return a synthetic
  // demo dataset so the UI round-trip is visible. Replaces all of this once
  // the upload+manifest step lands.
  const N = 2000;
  const rows = [];
  for (let i = 0; i < N; i++) {
    const x = Math.random() * 100;
    const y = 0.85 * x + (Math.random() - 0.5) * 25;
    rows.push({ x, y });
  }
  // step-5 binning
  const bins = new Map();
  for (const r of rows) {
    const b = Math.floor(r.x / 5) * 5;
    if (!bins.has(b)) bins.set(b, []);
    bins.get(b).push(r.y);
  }
  const step5 = Array.from(bins.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([x, ys]) => ({ x: x + 2.5, median_y: median(ys) }));

  // Stub bands (real per-band SROCC TODO once real metrics flow through)
  const bands = [
    { label: "B0 below medium",     range: "<50",       n: 0, srocc: NaN, krocc: NaN, plcc: NaN, rmse: NaN },
    { label: "B1 medium",            range: "[50,65)",   n: 0, srocc: NaN, krocc: NaN, plcc: NaN, rmse: NaN },
    { label: "B2 high",              range: "[65,90)",   n: 0, srocc: NaN, krocc: NaN, plcc: NaN, rmse: NaN },
    { label: "B3 visually-lossless", range: "≥90",       n: 0, srocc: NaN, krocc: NaN, plcc: NaN, rmse: NaN },
  ];

  postMessage({ type: "result", data: { rows, step5, bands } });
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
