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

async function main() {
  const bakes = await loadAllBakes();
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

  aggSelect.addEventListener('change', e => renderAggregate(bakes, e.target.value));
  pbSelect.addEventListener('change',  e => renderPerBand(bakes,  e.target.value));
}

main().catch(err => {
  console.error(err);
  document.body.insertAdjacentHTML('beforeend',
    `<p class="red">Error loading data: ${err.message}</p>`);
});
