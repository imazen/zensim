// zensim Plotly.js site — loads data/*.json and renders per-band SROCC charts.

const PAPER_TABLE_3 = {
  // From docs/CID22_PAPER_PAGE_BY_PAGE_2026-05-11.md Page 20 (all 250 refs).
  SSIMULACRA2: { krcc: 0.6934, srcc: 0.882, pcc: 0.8601 },
  Butter2norm:  { krcc: -0.6575, srcc: -0.8455, pcc: -0.8089 },
  Butter3norm:  { krcc: -0.6547, srcc: -0.8387, pcc: -0.7903 },
  DSSIM:        { krcc: -0.6428, srcc: -0.8399, pcc: -0.7813 },
  VMAF:         { krcc:  0.6176, srcc:  0.8163, pcc:  0.7799 },
  FSIM:         { krcc:  0.6089, srcc:  0.8005, pcc:  0.7676 },
  "PSNR-HVS":   { krcc:  0.6076, srcc:  0.81,   pcc:  0.7559 },
};

const PAPER_HELDOUT_49 = {
  SSIMULACRA2: { krcc: 0.7033, srcc: 0.88541, pcc: 0.87448, mae: 4.97 },
};

async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return await r.json();
}

async function loadAllBakes() {
  const index = await fetchJSON('data/index.json');
  const bakes = await Promise.all(
    index.bakes.map(async (b) => ({ ...b, data: await fetchJSON(`data/${b.json}`) }))
  );
  return bakes;
}

function populateDatasetSelect(selectEl, datasets, defaultDs = 'CID22') {
  selectEl.innerHTML = '';
  for (const ds of datasets) {
    const o = document.createElement('option');
    o.value = ds;
    o.textContent = ds;
    if (ds === defaultDs) o.selected = true;
    selectEl.appendChild(o);
  }
}

function renderAggregate(bakes, ds) {
  // Bar chart: x = bake labels, y = SROCC. Bars per bake, plus ssim2 + butter
  // as reference lines.
  const labels = bakes.map(b => b.label);
  const v04Vals = bakes.map(b => b.data.aggregate[ds]?.v04);
  const v02Vals = bakes.map(b => b.data.aggregate[ds]?.v02);
  const ssim2 = bakes[0]?.data?.aggregate[ds]?.ssim2;
  const butter = bakes[0]?.data?.aggregate[ds]?.butter;
  const traces = [
    { x: labels, y: v04Vals, type: 'bar', name: 'V_X (MLP)', marker: { color: '#1f77b4' } },
    { x: labels, y: v02Vals, type: 'bar', name: 'V0_2 (linear)', marker: { color: '#7fb3d5' } },
  ];
  const layout = {
    title: `Aggregate SROCC on ${ds}`,
    yaxis: { title: 'SROCC' },
    barmode: 'group',
    shapes: [],
    annotations: [],
  };
  if (ssim2 != null) {
    layout.shapes.push({
      type: 'line', xref: 'paper', x0: 0, x1: 1, y0: ssim2, y1: ssim2,
      line: { color: '#b1130c', dash: 'dash', width: 2 },
    });
    layout.annotations.push({
      xref: 'paper', x: 0.02, y: ssim2, yshift: 10,
      text: `fast-ssim2: ${ssim2.toFixed(4)}`,
      showarrow: false, font: { color: '#b1130c' },
    });
  }
  if (butter != null) {
    layout.shapes.push({
      type: 'line', xref: 'paper', x0: 0, x1: 1, y0: butter, y1: butter,
      line: { color: '#888', dash: 'dot', width: 1 },
    });
    layout.annotations.push({
      xref: 'paper', x: 0.02, y: butter, yshift: 10,
      text: `butter: ${butter.toFixed(4)}`,
      showarrow: false, font: { color: '#888' },
    });
  }
  Plotly.newPlot('chart-aggregate', traces, layout, { responsive: true });
}

function renderPerBand(bakes, ds) {
  // For each band, group bars per bake. Stacks ssim2 as a comparison series.
  const bands = bakes[0]?.data?.per_band?.[ds] || [];
  const bandNames = bands.map(b => b.band);
  const traces = bakes.map((b, i) => ({
    x: bandNames,
    y: (b.data.per_band[ds] || []).map(r => r.v04),
    type: 'bar',
    name: b.label,
    error_y: {
      type: 'data',
      symmetric: false,
      array: (b.data.per_band[ds] || []).map(r => r.ci_hi - r.v04),
      arrayminus: (b.data.per_band[ds] || []).map(r => r.v04 - r.ci_lo),
      thickness: 1.2, width: 4,
    },
  }));
  // ssim2 series (same across bakes — use the first bake's row)
  traces.push({
    x: bandNames,
    y: (bakes[0]?.data?.per_band?.[ds] || []).map(r => r.ssim2),
    type: 'bar',
    name: 'fast-ssim2',
    marker: { color: '#b1130c' },
  });
  // butter series
  traces.push({
    x: bandNames,
    y: (bakes[0]?.data?.per_band?.[ds] || []).map(r => r.butter),
    type: 'bar',
    name: 'butter',
    marker: { color: '#888' },
  });
  const layout = {
    title: `Per-band SROCC on ${ds}`,
    yaxis: { title: 'SROCC' },
    xaxis: { tickfont: { size: 11 } },
    barmode: 'group',
  };
  Plotly.newPlot('chart-perband', traces, layout, { responsive: true });
}

function deltaCls(delta, eps = 0.002) {
  if (Math.abs(delta) < eps) return 'delta-near';
  return delta > 0 ? 'delta-good' : 'delta-bad';
}

function renderParityTable() {
  // Compare our reproduced numbers vs paper Table 3 — placeholder
  // until Goal 3 fills in our held-out 49-ref reproduction.
  const div = document.getElementById('parity-table');
  const rows = [
    `<tr><th>Metric</th><th>Paper Table 3 SRCC (250 refs)</th><th>Our SRCC (49-ref holdout)</th><th>Δ</th></tr>`,
  ];
  for (const [m, v] of Object.entries(PAPER_TABLE_3)) {
    const paper = v.srcc;
    // Placeholder: we don't yet have our per-metric reproduction.
    const ours = m === 'SSIMULACRA2' ? 0.8895 : null;
    const delta = ours != null ? (ours - paper) : null;
    rows.push(`<tr>
      <td>${m}</td>
      <td>${paper.toFixed(4)}</td>
      <td>${ours != null ? ours.toFixed(4) : '<span class="small">pending</span>'}</td>
      <td class="${ours != null ? deltaCls(delta) : ''}">${ours != null ? delta.toFixed(4) : '—'}</td>
    </tr>`);
  }
  div.innerHTML = `<table>${rows.join('')}</table>`;
}

async function loadParityTable() {
  try {
    return await fetchJSON('data/parity_table.json');
  } catch {
    return null;
  }
}

function renderKonjndTable(parity) {
  if (!parity) return;
  const div = document.getElementById('konjnd-table');
  if (!div) return;
  const r = parity.our_repro;
  const rows = [
    `<tr><th>Metric</th><th>Subset</th><th>Ours (mean ± stdev)</th><th>Paper Table 4</th><th>Δ</th></tr>`,
  ];
  const subsets = [
    { metric: 'fast-ssim2', subset: 'BPG',  ours: r.fast_ssim2_konjnd_bpg,  paper_mean: r.fast_ssim2_konjnd_bpg.paper_mean, paper_std: r.fast_ssim2_konjnd_bpg.paper_std },
    { metric: 'fast-ssim2', subset: 'JPEG', ours: r.fast_ssim2_konjnd_jpeg, paper_mean: r.fast_ssim2_konjnd_jpeg.paper_mean, paper_std: r.fast_ssim2_konjnd_jpeg.paper_std },
    { metric: 'butter 3-norm', subset: 'BPG',  ours: r.butter_konjnd_bpg,  paper_mean: r.butter_konjnd_bpg.paper_mean, paper_std: r.butter_konjnd_bpg.paper_std },
    { metric: 'butter 3-norm', subset: 'JPEG', ours: r.butter_konjnd_jpeg, paper_mean: r.butter_konjnd_jpeg.paper_mean, paper_std: r.butter_konjnd_jpeg.paper_std },
  ];
  for (const s of subsets) {
    const delta = s.ours.mean - s.paper_mean;
    const cls = deltaCls(delta, 0.6);  // within paper's reported stdev
    rows.push(`<tr>
      <td>${s.metric}</td><td>${s.subset}</td>
      <td>${s.ours.mean.toFixed(2)} ± ${s.ours.std.toFixed(2)}</td>
      <td>${s.paper_mean.toFixed(2)} ± ${s.paper_std.toFixed(2)}</td>
      <td class="${cls}">${delta >= 0 ? '+' : ''}${delta.toFixed(3)}</td>
    </tr>`);
  }
  div.innerHTML = `<table>${rows.join('')}</table>`;
}

// Non-mono q-step rates for bakes where we don't (yet) emit a field
// in the per-bake JSON. Values come from `score_unified_with_bake.py`
// on unified_v13_zenjpeg.parquet (the same harness CLAUDE.md references).
const NONMONO_PCT_BY_LABEL = {
  'V0_5_leaked': 5.36,
  'V0_6_clean_baseline': 5.94,
  'V0_7_seed0_initial': 5.67,
  'V0_7_seed1_TV10': 5.46,
  'V0_8_shipped': 5.87,
  'V0_10_perband_tv15_25_15_15': 2.40,
  'V0_11_flat_tv20': 2.33,
  'V0_12_b1_oversample': 1.68,
  'V0_16_shipped': 2.30,
  'V0_17_tv25_notship': 2.44,
  'V0_18_seed42': 2.01,
  'V0_19_seed7': 2.84,
  'V0_20_seed123': 2.65,
  'V0_21_butter_clean': 2.91,
  'V0_22_konjnd_w1': 1.96,
};

function renderParetoCid22Aic3(bakes) {
  // 2D Pareto: x = AIC-3 SROCC, y = CID22 SROCC. Both axes higher = better.
  // Each bake is a labelled point + ssim2 reference + ensemble points.
  const xs = []; const ys = []; const labels = []; const colors = [];
  for (const b of bakes) {
    const aic = b.data.aggregate?.['AIC-3 CTC']?.v04;
    const cid = b.data.aggregate?.CID22?.v04;
    if (aic == null || cid == null) continue;
    xs.push(aic); ys.push(cid); labels.push(b.label);
    if (b.label === 'V0_16_shipped') colors.push('#0a7d28');
    else if (b.label === 'V0_21_butter_clean') colors.push('#9467bd');
    else if (b.label.startsWith('V0_5') || b.label.startsWith('V0_6') || b.label.startsWith('V0_7') || b.label === 'V0_8_shipped')
      colors.push('#888');
    else colors.push('#1f77b4');
  }
  // Add hard-coded ensemble points (not in bakes index)
  const ensemblePoints = [
    { label: '{V0_16,V0_20} 2-bake (OPTIMUM)', aic: 0.8050, cid: 0.8910, color: '#d18811' },
    { label: '{V0_16,V0_21} 2-bake', aic: 0.8024, cid: 0.8911, color: '#d18811' },
    { label: '{V0_16,V0_20,V0_21} 3-bake', aic: 0.8051, cid: 0.8908, color: '#d18811' },
    { label: '5-bake ensemble', aic: 0.8012, cid: 0.8896, color: '#7fb3d5' },
  ];
  for (const p of ensemblePoints) {
    xs.push(p.aic); ys.push(p.cid); labels.push(p.label); colors.push(p.color);
  }
  const traces = [
    {
      x: xs, y: ys, text: labels,
      mode: 'markers+text', textposition: 'top center', type: 'scatter',
      marker: { size: 12, color: colors, line: { width: 1, color: '#222' } },
      name: 'V_X bakes + ensembles',
    },
    {
      x: [0.7965], y: [0.8895], text: ['fast-ssim2'],
      mode: 'markers+text', textposition: 'bottom center', type: 'scatter',
      marker: { size: 14, color: '#b1130c', symbol: 'diamond' },
      name: 'fast-ssim2',
    },
  ];
  const layout = {
    title: 'CID22 vs AIC-3 SROCC — V_X bakes + ensembles (target: upper-right of ssim2)',
    xaxis: { title: 'AIC-3 CTC SROCC' },
    yaxis: { title: 'CID22 SROCC' },
    shapes: [
      { type: 'line', xref: 'x', x0: 0.7965, x1: 0.7965, yref: 'paper', y0: 0, y1: 1, line: { color: '#b1130c', dash: 'dash', width: 1 } },
      { type: 'line', yref: 'y', y0: 0.8895, y1: 0.8895, xref: 'paper', x0: 0, x1: 1, line: { color: '#b1130c', dash: 'dash', width: 1 } },
    ],
    showlegend: false,
  };
  Plotly.newPlot('chart-pareto-cid22-aic3', traces, layout, { responsive: true });
}

function renderPareto(bakes) {
  // Scatter: x = non-mono %, y = CID22 SROCC.
  // Each bake is a labelled point. ssim2 reference as a separate marker.
  const xs = [];
  const ys = [];
  const labels = [];
  const colors = [];
  for (const b of bakes) {
    const nm = b.data.non_mono_q_step_pct ?? NONMONO_PCT_BY_LABEL[b.label];
    const srocc = b.data.aggregate?.CID22?.v04;
    if (nm == null || srocc == null) continue;
    xs.push(nm);
    ys.push(srocc);
    labels.push(b.label);
    // Color: ship (V0_8) green, archived gray, smoothness specialists blue
    if (b.label === 'V0_8_shipped') colors.push('#0a7d28');
    else if (b.label.startsWith('V0_5') || b.label.startsWith('V0_6') || b.label.startsWith('V0_7'))
      colors.push('#888');
    else colors.push('#1f77b4');
  }
  // ssim2 reference (CID22 0.8895, non-mono 5.08% per CLAUDE.md)
  const ssim2 = bakes[0]?.data?.aggregate?.CID22?.ssim2 ?? 0.8895;
  const traces = [
    {
      x: xs,
      y: ys,
      text: labels,
      mode: 'markers+text',
      textposition: 'top center',
      type: 'scatter',
      marker: { size: 14, color: colors, line: { width: 1, color: '#222' } },
      name: 'V_X bakes',
    },
    {
      x: [5.08],
      y: [ssim2],
      text: ['fast-ssim2'],
      mode: 'markers+text',
      textposition: 'bottom center',
      type: 'scatter',
      marker: { size: 16, color: '#b1130c', symbol: 'diamond' },
      name: 'fast-ssim2',
    },
  ];
  const layout = {
    title: 'Pareto: CID22 SROCC vs JPEG non-mono q-step rate',
    xaxis: {
      title: 'Non-mono q-step % (lower = smoother)',
      autorange: 'reversed',  // smoother on the right
    },
    yaxis: { title: 'CID22 SROCC (higher = more correlated with human MOS)' },
    showlegend: false,
    shapes: [
      {
        type: 'line', xref: 'paper', x0: 0, x1: 1, y0: ssim2, y1: ssim2,
        line: { color: '#b1130c', dash: 'dash', width: 1 },
      },
    ],
  };
  Plotly.newPlot('chart-pareto', traces, layout, { responsive: true });
}

async function loadStep5Bakes() {
  // Try to load step-5 band JSONs for the bakes we have data for.
  const labels = ['v0_16', 'v0_15', 'v0_17', 'v0_18', 'v0_19', 'v0_20', 'v0_21', 'v0_22', 'v0_8_tainted'];  // expand as more bakes get step-5 emitted
  const out = [];
  for (const lab of labels) {
    try {
      const data = await fetchJSON(`data/step5_bands/${lab}.json`);
      out.push(data);
    } catch (e) {
      console.warn(`step5 bands for ${lab} not found:`, e.message);
    }
  }
  return out;
}

const BAKE_COLORS = ['#0a7d28', '#1f77b4', '#9467bd', '#ff7f0e'];

function renderStep5(step5Bakes) {
  if (!step5Bakes || step5Bakes.length === 0) return;
  const div = document.getElementById('chart-step5');
  if (!div) return;
  const first = step5Bakes[0];
  const xs = first.bands.map(b => (b.bin_lo + b.bin_hi) / 2);
  const traces = [];
  // One V_X series per bake
  step5Bakes.forEach((bake, i) => {
    const bxs = bake.bands.map(b => (b.bin_lo + b.bin_hi) / 2);
    traces.push({
      x: bxs,
      y: bake.bands.map(b => b.srocc_v04),
      mode: 'lines+markers',
      type: 'scatter',
      name: bake.label,
      line: { color: BAKE_COLORS[i % BAKE_COLORS.length], width: 2 },
    });
  });
  // V0_2 + ssim2 + butter references (use first bake's data — same metrics for all)
  traces.push({
    x: xs,
    y: first.bands.map(b => b.srocc_v02),
    mode: 'lines+markers',
    type: 'scatter',
    name: 'V0_2 (linear baseline)',
    line: { color: '#7fb3d5', width: 1, dash: 'dashdot' },
  });
  traces.push({
    x: xs,
    y: first.bands.map(b => b.srocc_ssim2),
    mode: 'lines+markers',
    type: 'scatter',
    name: 'fast-ssim2',
    line: { color: '#b1130c', width: 2, dash: 'dash' },
  });
  traces.push({
    x: xs,
    y: first.bands.map(b => b.srocc_butter),
    mode: 'lines+markers',
    type: 'scatter',
    name: 'butter (3-norm)',
    line: { color: '#888', width: 1, dash: 'dot' },
  });
  const layout = {
    title: `Within-bin SROCC at step-5 MCOS bins (CID22)`,
    xaxis: { title: 'MCOS bin center (5-unit bins)', range: [0, 100] },
    yaxis: { title: 'SROCC (within bin)', range: [-1.05, 1.05] },
    shapes: [
      { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: '#bbb', width: 1 } },
    ],
    annotations: [
      {
        xref: 'paper', x: 0.01, y: -0.18, yanchor: 'top',
        text: `Per-bin n (${first.label}): ${first.bands.map(b => `[${b.bin_lo},${b.bin_hi}):${b.n}`).join(' • ')}`,
        showarrow: false,
        font: { size: 10, color: '#666' },
      },
    ],
  };
  Plotly.newPlot('chart-step5', traces, layout, { responsive: true });
}

async function loadScatter(label) {
  try {
    return await fetchJSON(`data/scatter/${label}.json`);
  } catch (e) {
    console.warn(`scatter ${label} not found:`, e.message);
    return null;
  }
}

const BAND_COLOR = ['#b1130c', '#d18811', '#0a7d28', '#1f77b4']; // B0 red, B1 orange, B2 green, B3 blue

function renderScatter(scatter, xKey, xLabel, divId) {
  if (!scatter) return;
  const pts = scatter.points;
  // One trace per band for colored legend
  const traces = [0, 1, 2, 3].map(band => ({
    x: pts.filter(p => p.band === band).map(p => p[xKey]),
    y: pts.filter(p => p.band === band).map(p => p.v),
    mode: 'markers',
    type: 'scatter',
    marker: { size: 4, color: BAND_COLOR[band], opacity: 0.5 },
    name: `B${band} (n=${pts.filter(p => p.band === band).length})`,
  }));
  const layout = {
    title: `${scatter.label} (V_X, y) vs ${xLabel} (x) — CID22 ${scatter.n} pairs, colored by MCOS band`,
    xaxis: { title: xLabel },
    yaxis: { title: 'V_X quality (= -distance)' },
    showlegend: true,
  };
  Plotly.newPlot(divId, traces, layout, { responsive: true });
}

async function main() {
  const bakes = await loadAllBakes();
  const parity = await loadParityTable();
  renderKonjndTable(parity);
  if (bakes.length === 0) {
    document.body.insertAdjacentHTML('beforeend',
      '<p class="red">No bake data found. Run scripts/v_next/build_site_data.py first.</p>');
    return;
  }
  const datasets = Array.from(new Set(
    bakes.flatMap(b => Object.keys(b.data.aggregate))
  )).sort();

  const aggSelect = document.getElementById('ds-aggregate');
  const pbSelect  = document.getElementById('ds-perband');
  populateDatasetSelect(aggSelect, datasets);
  populateDatasetSelect(pbSelect, datasets);

  renderAggregate(bakes, aggSelect.value);
  renderPerBand(bakes, pbSelect.value);
  renderParityTable();
  if (document.getElementById('chart-pareto')) renderPareto(bakes);
  if (document.getElementById('chart-pareto-cid22-aic3')) renderParetoCid22Aic3(bakes);
  const step5 = await loadStep5Bakes();
  if (document.getElementById('chart-step5')) renderStep5(step5);
  // Default scatter is V0_16; can swap via the selector
  const scatterLabels = ['v0_16', 'v0_15', 'v0_17', 'v0_18', 'v0_19', 'v0_20', 'v0_21', 'v0_22', 'v0_8_tainted'];
  const scatterCache = {};
  for (const lab of scatterLabels) scatterCache[lab] = await loadScatter(lab);
  function renderScatterTriple(label) {
    const s = scatterCache[label];
    if (!s) return;
    if (document.getElementById('chart-scatter-ssim2')) renderScatter(s, 's', 'fast-ssim2 score', 'chart-scatter-ssim2');
    if (document.getElementById('chart-scatter-butter')) renderScatter(s, 'b', '−butter (3-norm) — higher=better', 'chart-scatter-butter');
    if (document.getElementById('chart-scatter-human')) renderScatter(s, 'h', 'human MCOS', 'chart-scatter-human');
  }
  renderScatterTriple('v0_16');
  const scatterSel = document.getElementById('scatter-bake-select');
  if (scatterSel) {
    for (const lab of scatterLabels) {
      const o = document.createElement('option');
      o.value = lab; o.textContent = lab + (lab === 'v0_16' ? ' (current ship)' : (lab === 'v0_8_tainted' ? ' (archived, was tainted)' : ' (archived)'));
      if (lab === 'v0_16') o.selected = true;
      scatterSel.appendChild(o);
    }
    scatterSel.addEventListener('change', e => renderScatterTriple(e.target.value));
  }

  aggSelect.addEventListener('change', e => renderAggregate(bakes, e.target.value));
  pbSelect.addEventListener('change',  e => renderPerBand(bakes,  e.target.value));
}

main().catch(err => {
  console.error(err);
  document.body.insertAdjacentHTML('beforeend',
    `<p class="red">Error loading data: ${err.message}</p>`);
});
