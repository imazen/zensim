// compare.js — main thread for the interactive corpus comparison page.
//
// Responsibilities:
//   - Fetch _manifest.json (corpus list, metric names, R2 base URL).
//   - Render corpus checkboxes + X/Y axis dropdowns + filters.
//   - Forward "run query" to the Web Worker.
//   - Receive query results and render Plotly scatter + step-5 line +
//     per-band SROCC table.
//
// All CPU-heavy work (DuckDB SQL, MLP forward pass, statistics) happens
// in compare-worker.js. The main thread only does UI + plotting.
//
// MVP STATE (2026-05-12): skeleton only. _manifest.json is not yet
// uploaded; the page falls back to a stub corpus list so the UI is
// visible end-to-end. Real wiring lands in subsequent commits.

const R2_BASE = "https://zentrain.r2.dev/zensim-compare-site"; // placeholder; user to confirm public URL

const STUB_MANIFEST = {
  base_url: R2_BASE,
  corpora: [
    { id: "v13_zenjpeg",  label: "zenjpeg sweep (v13, 36k rows)",        codec: "zenjpeg" },
    { id: "v12_zenavif",  label: "zenavif sweep (v12, 4k rows)",          codec: "zenavif" },
    { id: "v12_zenjxl",   label: "zenjxl sweep (v12, 32k rows)",          codec: "zenjxl"  },
    { id: "v12_zenwebp",  label: "zenwebp sweep (v12, 1k rows)",          codec: "zenwebp" },
    { id: "v14_zenpng",   label: "zenpng sweep (v14, 2.4k rows)",         codec: "zenpng"  },
    { id: "v15r_zenjpeg", label: "zenjpeg sweep (v15r, 1.79M rows)",      codec: "zenjpeg" },
    { id: "cid22",        label: "CID22 (human MOS, 4292 rows) [TODO]",   codec: "*"       },
    { id: "kadid10k",     label: "KADID-10k (human DMOS, 10k rows) [TODO]", codec: "*"     },
    { id: "tid2013",      label: "TID2013 (human MOS, 3k rows) [TODO]",   codec: "*"       },
  ],
  metrics: [
    { id: "q",                        label: "q (codec quality)" },
    { id: "score_ssim2",              label: "ssim2 (sweep-time)" },
    { id: "score_butteraugli_max",    label: "butteraugli (max-norm)" },
    { id: "score_butteraugli_pnorm3", label: "butteraugli (3-norm)" },
    { id: "score_zensim",             label: "zensim (sweep-time bake)" },
    { id: "score_dssim",              label: "dssim (when present)" },
    // JS-MLP variants — bakes shipped under site/weights/*.bin.
    // Worker loads the bake and applies it to feat_0..feat_227.
    { id: "score_zensim_v0_4",        label: "zensim V0_4 (JS-MLP, 228→64→1, 2026-04-30)" },
    { id: "score_zensim_v0_16",       label: "zensim V0_16 SHIP (JS-MLP, 228→128→1, 2026-05-12)" },
    { id: "score_zensim_v0_20",       label: "zensim V0_20 seed 123 (JS-MLP)" },
    { id: "score_zensim_v0_22",       label: "zensim V0_22 konjnd_w=1 (JS-MLP)" },
    // Human-rated columns (corpus-dependent — only available when a
    // human-rated corpus like AIC-3/AIC-4/CID22 is selected).
    { id: "human_jnd",                label: "human JND / MOS (corpus-dependent)" },
    { id: "human_jnd_ci_lo",          label: "human JND — CI lower (AIC-4 only)" },
    { id: "human_jnd_ci_hi",          label: "human JND — CI upper (AIC-4 only)" },
    { id: "encoded_bytes",            label: "encoded bytes" },
    { id: "bpp",                      label: "bits per pixel (AIC-3 only for now)" },
  ],
};

const $ = (id) => document.getElementById(id);

function renderCorpusList(manifest) {
  const list = $("corpus-list");
  list.innerHTML = "";
  for (const c of manifest.corpora) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = c.id;
    cb.dataset.codec = c.codec;
    if (c.id === "v13_zenjpeg") cb.checked = true; // sensible default
    const span = document.createElement("span");
    span.textContent = c.label;
    label.appendChild(cb);
    label.appendChild(span);
    list.appendChild(label);
  }
}

function renderAxisDropdowns(manifest) {
  const xSel = $("x-axis"), ySel = $("y-axis");
  xSel.innerHTML = "";
  ySel.innerHTML = "";
  for (const m of manifest.metrics) {
    const oX = document.createElement("option");
    oX.value = m.id; oX.textContent = m.label;
    xSel.appendChild(oX);
    const oY = oX.cloneNode(true);
    ySel.appendChild(oY);
  }
  xSel.value = "q";
  ySel.value = "score_ssim2";
}

function selectedCorpora() {
  return Array.from(document.querySelectorAll("#corpus-list input:checked"))
              .map((cb) => cb.value);
}

function setProgress(msg, state = "") {
  const p = $("progress");
  p.textContent = msg;
  p.className = "progress" + (state ? " " + state : "");
}

let worker = null;
function initWorker() {
  worker = new Worker("js/compare-worker.js", { type: "module" });
  worker.onmessage = (e) => {
    const { type, data } = e.data;
    if (type === "ready")    setProgress("worker ready · DuckDB-WASM loaded", "ok");
    if (type === "progress") setProgress(data, "busy");
    if (type === "error")    setProgress("error: " + data, "busy");
    if (type === "result")   renderResult(data);
  };
  worker.onerror = (err) => setProgress("worker error: " + err.message, "busy");
  worker.postMessage({ type: "init", base_url: R2_BASE });
}

function renderResult(data) {
  // data: { rows: [{x, y, codec, q, knob_json}], bands: [{label, n, srocc, ...}], step5: [{x, median_y}] }
  setProgress(`rendered ${data.rows?.length ?? 0} rows`, "ok");

  if (data.rows?.length) {
    const trace = {
      x: data.rows.map((r) => r.x),
      y: data.rows.map((r) => r.y),
      mode: "markers", type: "scattergl",
      marker: { size: 3, opacity: 0.35 },
      name: "rows",
    };
    const step = data.step5?.length ? [{
      x: data.step5.map((r) => r.x),
      y: data.step5.map((r) => r.median_y),
      mode: "lines+markers", type: "scatter",
      line: { color: "#b1130c", width: 2 },
      name: "step-5 median",
    }] : [];
    Plotly.newPlot("scatter", [trace, ...step], {
      margin: { t: 30, r: 30, b: 50, l: 60 },
      xaxis: { title: $("x-axis").options[$("x-axis").selectedIndex].text },
      yaxis: { title: $("y-axis").options[$("y-axis").selectedIndex].text },
      hovermode: "closest",
    }, { responsive: true });
  }

  const tbody = document.querySelector("#band-table tbody");
  tbody.innerHTML = "";
  for (const b of (data.bands ?? [])) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${b.label}</td><td>${b.range}</td>
      <td class="num">${b.n}</td>
      <td class="num">${b.srocc?.toFixed(4) ?? "—"}</td>
      <td class="num">${b.krocc?.toFixed(4) ?? "—"}</td>
      <td class="num">${b.plcc ?.toFixed(4) ?? "—"}</td>
      <td class="num">${b.rmse ?.toFixed(3) ?? "—"}</td>`;
    tbody.appendChild(tr);
  }
}

function bindRun() {
  $("run").addEventListener("click", () => {
    const corpora = selectedCorpora();
    if (corpora.length === 0) { setProgress("pick at least one corpus", "busy"); return; }
    setProgress("querying…", "busy");
    worker.postMessage({
      type: "query",
      corpora,
      x_metric: $("x-axis").value,
      y_metric: $("y-axis").value,
      codec_filter: $("codec-filter").value || null,
      version_filter: $("version-filter").value || null,
    });
  });
}

async function main() {
  // Try to fetch the live manifest; fall back to stub if 404 (pre-upload phase).
  let manifest = STUB_MANIFEST;
  try {
    const r = await fetch(`${R2_BASE}/parquets/_manifest.json`);
    if (r.ok) manifest = await r.json();
  } catch (_) { /* offline / pre-upload — use stub */ }

  renderCorpusList(manifest);
  renderAxisDropdowns(manifest);
  bindRun();
  initWorker();
}

main();
