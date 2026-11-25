// Interactive SVG charts, dot & density scatters, accessible table twins, and tooltips for zensim A evaluations.
// Uses DOM creation + textContent exclusively (never innerHTML with data strings).

const SVG_NS = "http://www.w3.org/2000/svg";

const MODEL_META = {
  A: { label: "zensim A", color: "var(--series-1)" },
  S2: { label: "SSIMULACRA2", color: "var(--series-2)" },
  BA: { label: "Butteraugli (−p3)", color: "var(--series-3)" },
  B: { label: "Research B (rev 5)", color: "var(--series-4)" },
  V2: { label: "Research by_v2fy (rev 4)", color: "var(--series-5)" },
};

const PEER_KEYS = ["A", "S2", "BA"];

const VIEW_META = {
  cid22_b: {
    title: "CID22-B",
    y_label: "MCOS (0–100, higher = better)",
    rmse_unit: "MCOS",
    polarity: "quality",
    is_subset: false,
    disclosure:
      "Approved 24-reference B split (2,100 pairs); first half of CID22 references trained A; prior frozen-model reads on 2026-10-04 and 2026-10-18.",
  },
  csiq: {
    title: "CSIQ",
    y_label: "1 − DMOS (higher = better)",
    rmse_unit: "1−DMOS",
    polarity: "quality",
    is_subset: false,
    disclosure:
      "Approved 866-pair population across 30 references; prior frozen-model reads on 2026-10-04 and 2026-10-18 (865 rows).",
  },
  live_all: {
    title: "LIVE Release 2 (full)",
    y_label: "1 − dmos_new/100 (higher = better)",
    rmse_unit: "1−dmos/100",
    polarity: "quality",
    is_subset: false,
    disclosure:
      "779 pairs across 29 references; 19 Kodak references overlap TID2013 training content. Prior external reads since 2026-10-03.",
  },
  live_clean10: {
    title: "LIVE Release 2 (clean10)",
    y_label: "1 − dmos_new/100 (higher = better)",
    rmse_unit: "1−dmos/100",
    polarity: "quality",
    is_subset: true,
    disclosure:
      "Fixed 2026-10-03 content-disjoint subset excluding the 19 Kodak/TID2013 references (267 pairs / 10 references); prior 2026-10-18 candidate read (267 rows), not an untouched or independent set.",
  },
  kadid_terminal: {
    title: "KADID-10k TERMINAL",
    y_label: "MOS (1–5, higher = better)",
    rmse_unit: "MOS",
    polarity: "quality",
    is_subset: false,
    disclosure:
      "Leads SSIMULACRA2 (0.893 vs 0.800) and Butteraugli (0.572), while failing both registered internal research-comparator confirmation rules (A − B = −0.049, A − by_v2fy = −0.040). First zensim model read; prior third-party label studies 2026-10-24..26.",
  },
  aic4: {
    title: "AIC-4 sample",
    y_label: "q_jnd distortion (lower = better)",
    rmse_unit: "q_jnd",
    polarity: "distortion",
    is_subset: false,
    disclosure:
      "Distortion-oriented q_jnd (negative SROCC = better quality alignment). Only 5 reference images (300 pairs); intervals are descriptive. Prior 2026-10-04 read.",
  },
  sdr25: {
    title: "SDR25 (AIC-4 JPEG-AI subset)",
    y_label: "q_jnd distortion (lower = better)",
    rmse_unit: "q_jnd",
    polarity: "distortion",
    is_subset: true,
    disclosure:
      "50-pair JPEG-AI subset inside AIC-4 (5 references), not an independent dataset; all three metrics reach −1.000 within-reference SROCC.",
  },
  mcljci_all: {
    title: "MCL-JCI (full50)",
    y_label: "Mean JND-chain count (lower = better)",
    rmse_unit: "JND count",
    polarity: "distortion",
    is_subset: false,
    disclosure:
      "Distortion-oriented JND count reconstructed from 30 subject chains per source (QF 1..100, QF100 = 0; original historical CSV lost). Shares 10 source images with KonFiG-IQA training images (earlier recipes); prior 2026-10-04 frozen-model read and 2026-10-24..25 third-party label studies.",
  },
  mcljci_clean40: {
    title: "MCL-JCI (clean40)",
    y_label: "Mean JND-chain count (lower = better)",
    rmse_unit: "JND count",
    polarity: "distortion",
    is_subset: true,
    disclosure:
      "Excludes the 10 sources overlapping KonFiG-IQA training images (1, 3, 6, 7, 9, 17, 28, 31, 45, 50) fixed before reading; subset of MCL-JCI, retaining earlier 2026-10-04 and 2026-10-24..25 exposure.",
  },
  konjnd_terminal: {
    title: "KonJND-1k JPEG TERMINAL",
    y_label: "Raw PJND threshold",
    rmse_unit: "PJND",
    polarity: "threshold",
    is_subset: false,
    disclosure:
      "Picturewise JND threshold instrument (100 coded threshold pairs across 100 distinct references), not suprathreshold MOS; within-reference SROCC is undefined (1 pair/ref). Prior 2026-10-04 R7/R7a frozen-model comparison.",
  },
};

const KADID_DISTORTION_META = {
  "01": ["Gaussian blur", "Blur"],
  "02": ["Lens blur", "Blur"],
  "03": ["Motion blur", "Blur"],
  "04": ["Color diffusion", "Color"],
  "05": ["Color shift", "Color"],
  "06": ["Color quantization", "Color"],
  "07": ["Color saturation 1 (HSV)", "Color"],
  "08": ["Color saturation 2 (Lab)", "Color"],
  "09": ["JPEG2000", "Compression"],
  "10": ["JPEG", "Compression"],
  "11": ["White Gaussian noise", "Noise"],
  "12": ["Color-component noise", "Noise"],
  "13": ["Impulse noise", "Noise"],
  "14": ["Multiplicative noise", "Noise"],
  "15": ["Denoise", "Noise"],
  "16": ["Brighten", "Brightness / contrast"],
  "17": ["Darken", "Brightness / contrast"],
  "18": ["Mean shift", "Brightness / contrast"],
  "19": ["Jitter", "Spatial"],
  "20": ["Non-eccentricity patch", "Spatial"],
  "21": ["Pixelate", "Spatial"],
  "22": ["Multi-threshold quantization", "Spatial"],
  "23": ["Color block", "Spatial"],
  "24": ["High sharpen", "Sharpness / contrast"],
  "25": ["Contrast change", "Brightness / contrast"],
};

const SPEED_BENCHMARK = {
  rows: [
    { threads: 1, size: "64²", a: [0.7502, 1.2536], fast_ssim2: [0.4001, 0.6974], margarine: [1.4754, 3.1743], ratios: [1.875, 0.509], rss_kib: [7876, 4736, 4752] },
    { threads: 1, size: "256²", a: [3.5328, 5.6097], fast_ssim2: [4.209, 6.3752], margarine: [6.0524, 11.5741], ratios: [0.839, 0.584], rss_kib: [14452, 11308, 11616] },
    { threads: 1, size: "1024²", a: [33.4841, 56.4965], fast_ssim2: [73.9088, 84.5089], margarine: [56.2368, 63.3265], ratios: [0.453, 0.595], rss_kib: [32656, 122688, 31408] },
    { threads: 1, size: "12 MP (12.0–12.6M px)", a: [350.9354, 376.8514], fast_ssim2: [1373.9451, 1498.0334], margarine: [666.2212, 747.9781], ratios: [0.255, 0.527], rss_kib: [179276, 1435000, 156628] },
    { threads: 16, size: "64² (pre-cutoff)", a: [2.1501, 3.4565], fast_ssim2: [0.4723, 1.7216], margarine: [1.6449, 2.6505], ratios: [4.552, 1.307], rss_kib: [7572, 5020, 6032] },
    { threads: 16, size: "256²", a: [2.6505, 5.6641], fast_ssim2: [4.7534, 7.06], margarine: [7.3119, 10.6318], ratios: [0.558, 0.362], rss_kib: [27704, 11532, 13144] },
    { threads: 16, size: "1024²", a: [11.2075, 16.8119], fast_ssim2: [33.8007, 42.6956], margarine: [48.4125, 55.039], ratios: [0.332, 0.232], rss_kib: [159920, 235020, 41204] },
    { threads: 16, size: "12 MP (12.0–12.6M px)", a: [105.3811, 119.9162], fast_ssim2: [357.4678, 396.6283], margarine: [550.1496, 587.3843], ratios: [0.295, 0.192], rss_kib: [598216, 2468936, 167168] },
  ],
  ols_fits: [
    { threads: 1, arm: "zensim A", alpha_ms: 1.59614, beta_ms_per_mp: 29.03716, r2: 0.99969 },
    { threads: 1, arm: "fast-ssim2 0.9.1", alpha_ms: -16.43188, beta_ms_per_mp: 116.27186, r2: 0.99855 },
    { threads: 1, arm: "margarine (in-process)", alpha_ms: 0.9729, beta_ms_per_mp: 55.54126, r2: 0.99969 },
    { threads: 16, arm: "zensim A", alpha_ms: 2.03585, beta_ms_per_mp: 8.6735, r2: 0.99637 },
    { threads: 16, arm: "fast-ssim2 0.9.1", alpha_ms: 2.04224, beta_ms_per_mp: 29.72196, r2: 0.99882 },
    { threads: 16, arm: "margarine (in-process)", alpha_ms: 2.1558, beta_ms_per_mp: 45.32139, r2: 0.99948 },
  ],
  cutoff_rows: [
    { threads: 16, size: "64²", before: [0.86, 1.85], after: [0.7, 0.7] },
    { threads: 16, size: "160²", before: [2.24, 2.06], after: [2.02, 1.98] },
    { threads: 16, size: "384²", before: [5.84, 5.91], after: [5.84, 5.03] },
    { threads: 16, size: "1024²", before: [30.6, 14.1], after: [30.7, 15.3] },
    { threads: 16, size: "12 MP", before: [345.7, 117.5], after: null },
    { threads: 4, size: "64²", before: [0.83, 1.95], after: [0.74, 0.74] },
    { threads: 4, size: "192²", before: [2.5, 2.46], after: [2.39, 2.29] },
    { threads: 4, size: "320²", before: [4.71, 4.81], after: [5.42, 4.28] },
  ],
  multihost_rows: [
    { cpu: "AMD Ryzen 9 9950X3D · v4x (AVX-512, CCD0)", threads: 16, s64: [0.39, 0.41], s256: [0.75, 0.93], s1024: [4.03, 4.87], s12m: [75.67, 77.42], beta: 6.40, rss1k: 115988, rss12m: 337180 },
    { cpu: "AMD Ryzen 9 9950X3D · v4x (AVX-512, CCD0)", threads: 1, s64: [0.39, 0.41], s256: [1.78, 1.87], s1024: [19.97, 22.46], s12m: [190.32, 194.07], beta: 15.70, rss1k: 32544, rss12m: 178192 },
    { cpu: "AMD Ryzen 9 9950X3D · v3 (AVX2 forced, CCD0)", threads: 16, s64: [0.40, 0.41], s256: [0.84, 1.03], s1024: [4.87, 6.01], s12m: [81.57, 85.18], beta: 6.88, rss1k: 120068, rss12m: 349104 },
    { cpu: "AMD Ryzen 9 9950X3D · v3 (AVX2 forced, CCD0)", threads: 1, s64: [0.40, 0.41], s256: [2.28, 2.38], s1024: [28.85, 30.09], s12m: [291.27, 293.45], beta: 24.13, rss1k: 32636, rss12m: 178036 },
    { cpu: "Intel Core Ultra 7 265K · v3 (AVX2, 8P+8E)", threads: 16, s64: [0.48, 0.69], s256: [1.84, 4.09], s1024: [11.81, 13.82], s12m: [167.44, 172.44], beta: 14.05, rss1k: 115836, rss12m: 338808 },
    { cpu: "Intel Core Ultra 7 265K · v3 (AVX2, P-core)", threads: 1, s64: [0.49, 0.55], s256: [2.81, 3.08], s1024: [35.84, 36.46], s12m: [382.08, 434.89], beta: 31.72, rss1k: 32844, rss12m: 178300 },
    { cpu: "Apple M6 · neon (ARM64, 2S+4P+6E)", threads: 12, s64: [1.07, 1.12], s256: [2.58, 2.86], s1024: [8.52, 11.90], s12m: [122.23, 125.20], beta: 10.21, rss1k: 147856, rss12m: 336864 },
    { cpu: "Apple M6 · neon (ARM64, 1T)", threads: 1, s64: [1.06, 1.10], s256: [7.15, 8.08], s1024: [38.48, 39.45], s12m: [467.41, 472.19], beta: 38.86, rss1k: 46304, rss12m: 215760 },
  ],
};

// Committed per-collection SROCC from zensim/README.md:46-56 + MODEL_A_REPORT.md:228-238 + e36_subset_test_2026-11-01.md:1620-1730
const SELECTION_EVAL = {
  collections: [
    { name: "CID:IQ", n: 210, s2: 0.757, ba: 0.592, a: [0.868, 0.869, 0.692, 7.8], f3: [0.863, 0.865, 0.684, 8.3] },
    { name: "CID:OA", n: 91, s2: 0.847, ba: 0.577, a: [0.911, 0.923, 0.753, 8.3], f3: [0.905, 0.921, 0.749, 8.7] },
    { name: "Colourlab (2022)", n: 192, s2: 0.604, ba: 0.373, a: [0.735, 0.728, 0.562, 10.7], f3: [0.689, 0.684, 0.516, 12.7] },
    { name: "IQA-ODI", n: 48, s2: 0.854, ba: 0.806, a: [0.867, 0.928, 0.679, 7.6], f3: [0.858, 0.924, 0.674, 7.5] },
    { name: "IRCCyN/IVC", n: 60, s2: 0.949, ba: 0.912, a: [0.919, 0.931, 0.752, 6.8], f3: [0.926, 0.942, 0.757, 6.4] },
    { name: "LIVE-MD", n: 180, s2: 0.913, ba: 0.827, a: [0.902, 0.926, 0.73, 7.0], f3: [0.892, 0.915, 0.716, 7.0] },
    { name: "MDID (other refs)", n: 480, s2: 0.787, ba: 0.572, a: [0.927, 0.931, 0.765, 6.1], f3: [0.886, 0.895, 0.705, 7.5] },
    { name: "NITS-IQA", n: 405, s2: 0.777, ba: 0.706, a: [0.734, 0.734, 0.542, 11.3], f3: [0.708, 0.709, 0.518, 12.2] },
    { name: "VCL@FER", n: 192, s2: 0.945, ba: 0.877, a: [0.95, 0.95, 0.8, 5.9], f3: [0.939, 0.937, 0.778, 6.6] },
  ],
  peers: [
    { metric: "zensim A", internal: false, eight: [8.18, 0.861], shared3: [8.27, 0.85], compression: [0.895, 6.95], noncodec: [0.79, 8.66] },
    { metric: "HaarPSI", internal: false, eight: [8.92, 0.843], shared3: [8.99, 0.828], compression: [0.888, 6.9], noncodec: [0.774, 9.76] },
    { metric: "SSIMULACRA2", internal: false, eight: [9.13, 0.831], shared3: [8.52, 0.859], compression: [0.896, 6.92], noncodec: [0.759, 10.59] },
    { metric: "CVVDP", internal: false, eight: [9.82, 0.823], shared3: [9.52, 0.829], compression: [0.876, 7.07], noncodec: [0.755, 10.76] },
    { metric: "FSIMc", internal: false, eight: [9.97, 0.83], shared3: [8.84, 0.864], compression: [0.906, 6.73], noncodec: [0.745, 10.79] },
    { metric: "FSIM (luma)", internal: false, eight: [10.06, 0.822], shared3: [8.86, 0.857], compression: [0.902, 6.67], noncodec: [0.734, 11.02] },
    { metric: "Butteraugli 3-norm", internal: false, eight: [11.57, 0.709], shared3: [10.47, 0.796], compression: [0.854, 8.61], noncodec: [0.661, 13.07] },
    { metric: "F3 (internal 11-MOS recipe)", internal: true, eight: [8.67, 0.848], shared3: [8.79, 0.835], compression: [0.895, 7.03], noncodec: [0.767, 9.28] },
  ],
  noncodec_sets: [
    { metric: "zensim A", internal: false, pooled: [0.79, 8.66], sets: [0.837, 0.714, 0.911, 0.915, 0.903, 0.672, 0.955] },
    { metric: "HaarPSI", internal: false, pooled: [0.774, 9.76], sets: [0.768, 0.708, 0.882, 0.976, 0.89, 0.672, 0.956] },
    { metric: "SSIMULACRA2", internal: false, pooled: [0.759, 10.59], sets: [0.698, 0.575, 0.847, 0.915, 0.921, 0.748, 0.96] },
    { metric: "CVVDP", internal: false, pooled: [0.755, 10.76], sets: [0.744, 0.579, 0.912, 0.915, 0.891, 0.699, 0.95] },
    { metric: "FSIMc", internal: false, pooled: [0.745, 10.79], sets: [0.657, 0.596, 0.889, 0.976, 0.896, 0.712, 0.938] },
    { metric: "FSIM (luma)", internal: false, pooled: [0.734, 11.02], sets: [0.645, 0.577, 0.887, 0.976, 0.891, 0.706, 0.907] },
    { metric: "Butteraugli 3-norm", internal: false, pooled: [0.661, 13.07], sets: [0.652, 0.316, 0.577, 0.794, 0.84, 0.749, 0.95] },
    { metric: "F3 (internal 11-MOS recipe)", internal: true, pooled: [0.767, 9.28], sets: [0.832, 0.664, 0.905, 0.915, 0.885, 0.643, 0.947] },
  ],
  confirmation_reads: [
    {
      name: "CLIC 2021 validation",
      scope: "4,812 2AFC questions / 315 crops (learned codecs; soft-vote agreement; read 3 times across program)",
      a: "0.642 (seed 0; 0.644 3-seed mean)",
      peers: "FSIM 0.638 [0.625, 0.652], FSIMc 0.637 [0.624, 0.652], SSIMULACRA2 0.626 [0.612, 0.640], Butteraugli 0.606 [0.592, 0.620], HaarPSI 0.603 [0.589, 0.618]",
      peers_f3: " (internal F3: 0.643 3-seed mean)",
    },
    {
      name: "MMSPG JPEG XR",
      scope: "186 rated conditions / 6 photographs (SROCC / within-ref SROCC / MedAE%)",
      a: "0.920 (seed 0; 3-seed 0.920–0.922 / 0.928–0.931 / 5.47–5.63%)",
      peers: "SSIMULACRA2 0.934 / 0.942 / 5.19%, FSIM 0.931 / 0.922 / 4.26%, FSIMc 0.929 / 0.912 / 4.69%, HaarPSI 0.921 / 0.923 / 5.64%, Butteraugli 0.902 / 0.896 / 6.58%",
      peers_f3: " (internal F3: 0.920 seed 0; 3-seed 0.920–0.922 / 0.925–0.928 / 5.43–5.47%)",
    },
    {
      name: "CIQA",
      scope: "7,902 2AFC questions / 6,596 photos (JPEG quality pairs)",
      a: "0.675",
      peers: "HaarPSI 0.676, SSIMULACRA2 0.675, FSIMc 0.675, FSIM 0.675, Butteraugli 0.675; 'higher JPEG q wins' = 0.675 vs 0.698 observer ceiling (non-separating)",
      peers_f3: " (internal F3: 0.675)",
    },
  ],
};

const RAMP_SEQ = [
  "var(--ramp-seq-0)",
  "var(--ramp-seq-1)",
  "var(--ramp-seq-2)",
  "var(--ramp-seq-3)",
  "var(--ramp-seq-4)",
  "var(--ramp-seq-5)",
  "var(--ramp-seq-6)",
];

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "text") node.textContent = String(v);
    else if (k === "className") node.className = v;
    else node.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (c !== null && c !== undefined) {
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
  }
  return node;
}

function svgEl(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "text") node.textContent = String(v);
    else node.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (c !== null && c !== undefined) node.appendChild(c);
  }
  return node;
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "undefined";
  return Number(v).toFixed(digits).replace("-", "−");
}

function fmtInt(v) {
  return Number(v).toLocaleString("en-US");
}

function fmtStatCell(arr, digits = 4) {
  if (!arr || arr[0] === null || arr[0] === undefined) {
    return el("td", { className: "num muted-3", text: "undefined" });
  }
  return el("td", { className: "num" }, [
    el("span", { text: fmt(arr[0], digits) }),
    el("span", {
      className: "ci-sub",
      text: `[${fmt(arr[1], digits)}, ${fmt(arr[2], digits)}]`,
    }),
  ]);
}

const tooltip = document.getElementById("chart-tooltip");

function showTooltip(evt, heading, rows) {
  tooltip.replaceChildren();
  if (heading) tooltip.appendChild(el("div", { className: "tt-head", text: heading }));
  for (const r of rows) {
    const row = el("div", { className: "tt-row" });
    if (r.color) {
      const key = el("span", { className: "swatch-line" });
      key.style.background = r.color;
      row.appendChild(key);
    }
    row.appendChild(el("strong", { text: r.value }));
    if (r.label) row.appendChild(el("span", { className: "tt-series", text: r.label }));
    tooltip.appendChild(row);
  }
  tooltip.hidden = false;
  positionTooltip(evt);
}

function positionTooltip(evt) {
  if (tooltip.hidden) return;
  const pad = 12;
  const x = evt.clientX ?? window.innerWidth / 2;
  const y = evt.clientY ?? window.innerHeight / 2;
  const rect = tooltip.getBoundingClientRect();
  let left = x + pad;
  let top = y + pad;
  if (left + rect.width > window.innerWidth - 8) left = Math.max(8, x - rect.width - pad);
  if (top + rect.height > window.innerHeight - 8) top = Math.max(8, y - rect.height - pad);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

function bindInteractiveMark(node, getTooltipSpec) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointerenter", (e) => {
    const spec = getTooltipSpec();
    showTooltip(e, spec.heading, spec.rows);
  });
  node.addEventListener("pointermove", positionTooltip);
  node.addEventListener("pointerleave", hideTooltip);
  node.addEventListener("focus", (e) => {
    const r = e.target.getBoundingClientRect();
    const spec = getTooltipSpec();
    showTooltip({ clientX: r.right, clientY: r.top }, spec.heading, spec.rows);
  });
  node.addEventListener("blur", hideTooltip);
}

function wireViewToggle(btnGroup, chartNode, tableNode, onModeChange) {
  const btns = btnGroup.querySelectorAll("button[data-mode]");
  btns.forEach((b) => {
    b.addEventListener("click", () => {
      const mode = b.dataset.mode;
      btns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      chartNode.hidden = mode !== "chart";
      tableNode.hidden = mode !== "table";
      if (onModeChange) onModeChange(mode);
    });
  });
}

function makeLegend(keys) {
  const row = el("div", { className: "legend-row" });
  for (const k of keys) {
    const m = MODEL_META[k];
    const sw = el("span", { className: "swatch-dot" });
    sw.style.background = m.color;
    row.appendChild(el("span", { className: "legend-item" }, [sw, m.label]));
  }
  return row;
}

const dotCache = new Map();

function decodeDots256(b64, dotGrid) {
  if (dotCache.has(b64)) return dotCache.get(b64);
  const bin = atob(b64);
  const len = bin.length;
  const pts = [];
  let pos = 0;
  let maxCnt = 1;
  while (pos < len) {
    const ix = bin.charCodeAt(pos);
    const nSingle = bin.charCodeAt(pos + 1);
    pos += 2;
    for (let i = 0; i < nSingle; i++) {
      const iy = bin.charCodeAt(pos++);
      pts.push([ix, iy, 1]);
    }
    const nMulti = bin.charCodeAt(pos++);
    for (let i = 0; i < nMulti; i++) {
      const iy = bin.charCodeAt(pos);
      const c0 = bin.charCodeAt(pos + 1);
      let cnt;
      if (c0 < 255) {
        cnt = c0;
        pos += 2;
      } else {
        cnt = bin.charCodeAt(pos + 2) | (bin.charCodeAt(pos + 3) << 8);
        pos += 4;
      }
      if (cnt > maxCnt) maxCnt = cnt;
      pts.push([ix, iy, cnt]);
    }
  }
  const out = { pts, maxCnt, grid: dotGrid };
  dotCache.set(b64, out);
  return out;
}

function renderFrozenChart(container, views) {
  container.replaceChildren();
  const groups = [
    {
      title: "Quality-oriented ratings (higher signed SROCC = stronger alignment)",
      polarity: "quality",
      domain: [0.5, 1.0],
      ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    {
      title: "Distortion-oriented JND targets (more negative signed SROCC = stronger alignment)",
      polarity: "distortion",
      domain: [-1.0, -0.65],
      ticks: [-1.0, -0.9, -0.8, -0.7],
    },
  ];

  const models = PEER_KEYS;
  container.appendChild(makeLegend(models));

  for (const grp of groups) {
    const subset = views.filter((v) => VIEW_META[v.id].polarity === grp.polarity);
    const W = 880;
    const left = 235;
    const right = 32;
    const top = 28;
    const rowH = 44;
    const bottom = 34;
    const H = top + subset.length * rowH + bottom;
    const plotW = W - left - right;
    const [d0, d1] = grp.domain;
    const xOf = (val) => left + ((val - d0) / (d1 - d0)) * plotW;

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      class: "chart",
      role: "img",
      "aria-label": grp.title,
    });
    svg.appendChild(
      svgEl("text", { x: 0, y: 16, fill: "var(--ink)", "font-weight": "600", text: grp.title })
    );

    const gridG = svgEl("g", { class: "grid" });
    for (const t of grp.ticks) {
      const tx = xOf(t);
      gridG.appendChild(svgEl("line", { x1: tx, y1: top, x2: tx, y2: H - bottom }));
      svg.appendChild(svgEl("text", { x: tx, y: H - 10, "text-anchor": "middle", text: fmt(t, 2) }));
    }
    svg.appendChild(gridG);

    subset.forEach((v, idx) => {
      const meta = VIEW_META[v.id];
      const cy = top + idx * rowH + rowH / 2;
      if (idx > 0) {
        svg.appendChild(
          svgEl("line", {
            x1: 0,
            y1: top + idx * rowH,
            x2: W,
            y2: top + idx * rowH,
            stroke: "var(--border)",
            "stroke-opacity": "0.5",
          })
        );
      }
      svg.appendChild(
        svgEl("text", { x: left - 12, y: cy - 3, "text-anchor": "end", fill: "var(--ink)", text: meta.title })
      );
      svg.appendChild(
        svgEl("text", {
          x: left - 12,
          y: cy + 12,
          "text-anchor": "end",
          fill: "var(--ink-3)",
          text: `${fmtInt(v.rows)} pairs / ${fmtInt(v.refs)} refs`,
        })
      );

      models.forEach((m, mi) => {
        const st = v.m[m]?.[0];
        if (!st) return;
        const yOff = cy + (mi - 1) * 10;
        const xVal = xOf(Math.max(d0, Math.min(d1, st[0])));
        const xLo = xOf(Math.max(d0, Math.min(d1, st[1])));
        const xHi = xOf(Math.max(d0, Math.min(d1, st[2])));
        const markG = svgEl("g", { style: "cursor:pointer" });
        markG.appendChild(
          svgEl("line", {
            x1: xLo,
            y1: yOff,
            x2: xHi,
            y2: yOff,
            stroke: MODEL_META[m].color,
            "stroke-width": "2",
          })
        );
        markG.appendChild(
          svgEl("circle", {
            cx: xVal,
            cy: yOff,
            r: 4.5,
            fill: MODEL_META[m].color,
            stroke: "var(--surface-1)",
            "stroke-width": "2",
          })
        );
        markG.appendChild(
          svgEl("rect", {
            x: Math.min(xLo, xVal) - 8,
            y: yOff - 12,
            width: Math.max(24, Math.abs(xHi - xLo) + 16),
            height: 24,
            fill: "transparent",
          })
        );
        bindInteractiveMark(markG, () => ({
          heading: `${meta.title} (${fmtInt(v.rows)} pairs / ${fmtInt(v.refs)} refs)`,
          rows: models.map((mk) => {
            const ms = v.m[mk][0];
            return {
              color: MODEL_META[mk].color,
              value: `${fmt(ms[0], 4)} [${fmt(ms[1], 4)}, ${fmt(ms[2], 4)}]`,
              label: MODEL_META[mk].label,
            };
          }),
        }));
        svg.appendChild(markG);
      });
    });

    container.appendChild(el("div", { className: "svg-wrap" }, [svg]));
  }
}

function renderFrozenTable(tbody, views, includeInternal) {
  tbody.replaceChildren();
  for (const v of views) {
    const meta = VIEW_META[v.id];
    const modelKeys = Object.keys(v.m).filter((k) => includeInternal || PEER_KEYS.includes(k));
    modelKeys.forEach((m, idx) => {
      const tuple = v.m[m];
      const tr = el("tr", { className: meta.is_subset ? "sub-row" : "" });
      if (idx === 0) {
        tr.appendChild(
          el("td", { rowspan: modelKeys.length }, [
            el("strong", { text: meta.title }),
            el("div", {
              className: "ci-sub",
              text: `${fmtInt(v.rows)} pairs / ${fmtInt(v.refs)} refs · ${meta.y_label}`,
            }),
            el("div", { className: "ci-sub", text: meta.disclosure }),
          ])
        );
      }
      const sw = el("span", { className: "swatch-dot" });
      sw.style.background = MODEL_META[m].color;
      tr.appendChild(el("td", {}, [el("span", { className: "legend-item" }, [sw, MODEL_META[m].label])]));
      tr.appendChild(fmtStatCell(tuple[0], 4));
      tr.appendChild(fmtStatCell(tuple[1], 4));
      tr.appendChild(fmtStatCell(tuple[2], 4));
      const rmseTd = fmtStatCell(tuple[3], 4);
      rmseTd.firstChild.textContent += ` ${meta.rmse_unit}`;
      tr.appendChild(rmseTd);
      tr.appendChild(fmtStatCell(tuple[4], 4));
      tbody.appendChild(tr);
    });
  }
}

function renderClicSection(chartWrap, tableBody, summaryBody, clic) {
  summaryBody.replaceChildren();
  const models = PEER_KEYS;
  for (const m of models) {
    const md = clic.models[m];
    const sw = el("span", { className: "swatch-dot" });
    sw.style.background = MODEL_META[m].color;
    summaryBody.appendChild(
      el("tr", {}, [
        el("td", {}, [el("span", { className: "legend-item" }, [sw, MODEL_META[m].label])]),
        fmtStatCell(md.maj, 6),
        fmtStatCell(md.soft, 6),
      ])
    );
  }

  const W = 760;
  const H = 290;
  const left = 54;
  const right = 26;
  const top = 22;
  const bottom = 42;
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const yMin = 0.5;
  const yMax = 1.0;
  const yOf = (v) => top + (1 - (v - yMin) / (yMax - yMin)) * plotH;
  const xOf = (idx, total) => left + (idx / Math.max(1, total - 1)) * plotW;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "chart",
    role: "img",
    "aria-label": "CLIC 2022 majority agreement across own-metric positive score-margin quantiles",
  });

  const gridG = svgEl("g", { class: "grid" });
  for (const yv of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
    const yy = yOf(yv);
    gridG.appendChild(svgEl("line", { x1: left, y1: yy, x2: W - right, y2: yy }));
    svg.appendChild(svgEl("text", { x: left - 8, y: yy + 4, "text-anchor": "end", text: fmt(yv, 2) }));
  }
  svg.appendChild(gridG);

  for (let q = 0; q < 10; q++) {
    const xx = xOf(q, 10);
    svg.appendChild(svgEl("text", { x: xx, y: H - 16, "text-anchor": "middle", text: `Q${q + 1}` }));
  }
  svg.appendChild(
    svgEl("text", {
      x: left + plotW / 2,
      y: H - 2,
      "text-anchor": "middle",
      fill: "var(--ink-3)",
      text: "Own-metric positive |Δscore| quantile bin (Q1 = smallest non-zero margin, Q10 = largest)",
    })
  );

  for (const m of models) {
    const posBins = clic.models[m].bins.filter((b) => b[1] > 0);
    const pts = posBins.map((b, i) => `${xOf(i, posBins.length)},${yOf(b[3])}`).join(" ");
    svg.appendChild(
      svgEl("polyline", {
        points: pts,
        fill: "none",
        stroke: MODEL_META[m].color,
        "stroke-width": "2",
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      })
    );
    posBins.forEach((b, i) => {
      const cx = xOf(i, posBins.length);
      const cy = yOf(b[3]);
      const g = svgEl("g", { style: "cursor:pointer" });
      g.appendChild(
        svgEl("circle", {
          cx,
          cy,
          r: 4.5,
          fill: MODEL_META[m].color,
          stroke: "var(--surface-1)",
          "stroke-width": "2",
        })
      );
      g.appendChild(svgEl("circle", { cx, cy, r: 12, fill: "transparent" }));
      bindInteractiveMark(g, () => ({
        heading: `${MODEL_META[m].label} · Positive margin Q${i + 1}`,
        rows: [
          { color: MODEL_META[m].color, value: fmt(b[3], 4), label: "majority agreement" },
          { value: `[${fmt(b[0], 3)}, ${fmt(b[1], 3)}]`, label: "own |Δscore| span" },
          { value: fmtInt(b[2]), label: "questions in bin" },
        ],
      }));
      svg.appendChild(g);
    });
  }

  chartWrap.replaceChildren(makeLegend(models), el("div", { className: "svg-wrap" }, [svg]));

  tableBody.replaceChildren();
  for (const m of models) {
    const bins = clic.models[m].bins;
    const hasTie = bins[0][0] === 0 && bins[0][1] === 0;
    bins.forEach((b, idx) => {
      const label = b[0] === 0 && b[1] === 0 ? "Exact tie (|Δ| = 0)" : `Positive Q${hasTie ? idx : idx + 1}`;
      tableBody.appendChild(
        el("tr", {}, [
          el("td", { text: idx === 0 ? MODEL_META[m].label : "" }),
          el("td", { text: label }),
          el("td", { className: "num", text: `[${fmt(b[0], 3)}, ${fmt(b[1], 3)}]` }),
          el("td", { className: "num", text: fmtInt(b[2]) }),
          el("td", { className: "num", text: fmt(b[3], 4) }),
        ])
      );
    });
  }
}

function renderScatters(stageNode, tableBody, disclosureNode, view, dotGrid, mode, showTrend) {
  const meta = VIEW_META[view.id];
  hideTooltip();
  stageNode.replaceChildren();
  tableBody.replaceChildren();
  disclosureNode.textContent = `${meta.title} (${fmtInt(view.rows)} pairs / ${fmtInt(view.refs)} references) · y-axis: ${meta.y_label} — ${meta.disclosure}`;

  const nb = 8;
  const [ymin, ymax] = view.yr;

  for (const m of PEER_KEYS) {
    const sc = view.sc[m];
    const sroccVal = view.m[m][0][0];
    const kroccVal = view.m[m][1][0];
    const [xmin, xmax] = sc.xr;
    const W = 360;
    const H = 310;
    const left = 52;
    const right = 16;
    const top = 16;
    const bottom = 42;
    const pw = W - left - right;
    const ph = H - top - bottom;
    const cw = pw / nb;
    const ch = ph / nb;

    const xOf = (val) => left + ((val - xmin) / Math.max(1e-9, xmax - xmin)) * pw;
    const yOf = (val) => top + (1 - (val - ymin) / Math.max(1e-9, ymax - ymin)) * ph;

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      class: "chart",
      role: "img",
      "aria-label": `${meta.title} ${MODEL_META[m].label} ${mode === "dots" ? "dot" : "8x8 density"} scatter`,
    });

    svg.appendChild(
      svgEl("rect", {
        x: left,
        y: top,
        width: pw,
        height: ph,
        fill: "var(--surface-1)",
        stroke: "var(--border)",
      })
    );

    const gridG = svgEl("g", { class: "grid" });
    for (const frac of [0.25, 0.5, 0.75]) {
      const gx = left + frac * pw;
      const gy = top + frac * ph;
      gridG.appendChild(svgEl("line", { x1: gx, y1: top, x2: gx, y2: top + ph }));
      gridG.appendChild(svgEl("line", { x1: left, y1: gy, x2: left + pw, y2: gy }));
    }
    svg.appendChild(gridG);

    let footerNote = "";

    if (mode === "dots") {
      const decoded = decodeDots256(sc.d, dotGrid);
      const gScale = Math.max(1, decoded.grid - 1);
      const isDense = view.rows >= 1500;
      const baseR = view.rows <= 150 ? 2.8 : isDense ? 1.85 : 2.25;
      const baseAlpha = view.rows <= 150 ? 0.72 : isDense ? 0.44 : 0.56;
      const logDenom = Math.log1p(Math.max(1, decoded.maxCnt - 1));

      const dotsG = svgEl("g", { "pointer-events": "none" });
      const screenPts = new Array(decoded.pts.length);

      for (let i = 0; i < decoded.pts.length; i++) {
        const [ix, iy, cnt] = decoded.pts[i];
        const fx = ix / gScale;
        const fy = iy / gScale;
        const cx = left + fx * pw;
        const cy = top + (1 - fy) * ph;
        const weight = cnt > 1 ? Math.log1p(cnt - 1) / logDenom : 0;
        const r = baseR + weight * 1.9;
        const alpha = Math.min(0.95, baseAlpha + weight * 0.46);
        dotsG.appendChild(
          svgEl("circle", {
            cx: cx.toFixed(1),
            cy: cy.toFixed(1),
            r: r.toFixed(2),
            fill: MODEL_META[m].color,
            "fill-opacity": alpha.toFixed(2),
          })
        );
        screenPts[i] = {
          cx,
          cy,
          cnt,
          xVal: xmin + fx * (xmax - xmin),
          yVal: ymin + fy * (ymax - ymin),
        };
      }
      svg.appendChild(dotsG);

      const hoverRing = svgEl("circle", {
        cx: 0,
        cy: 0,
        r: 5.5,
        fill: "none",
        stroke: "var(--ink)",
        "stroke-width": "1.8",
        "pointer-events": "none",
        visibility: "hidden",
      });

      const hitRect = svgEl("rect", {
        x: left,
        y: top,
        width: pw,
        height: ph,
        fill: "transparent",
        style: "cursor:crosshair",
      });

      hitRect.addEventListener("pointermove", (e) => {
        const box = svg.getBoundingClientRect();
        const sx = ((e.clientX - box.left) / Math.max(1, box.width)) * W;
        const sy = ((e.clientY - box.top) / Math.max(1, box.height)) * H;
        let best = null;
        let bestD2 = 28 * 28;
        for (let i = 0; i < screenPts.length; i++) {
          const p = screenPts[i];
          const dx = p.cx - sx;
          const dy = p.cy - sy;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) {
            bestD2 = d2;
            best = p;
          }
        }
        if (!best) {
          hoverRing.setAttribute("visibility", "hidden");
          hideTooltip();
          return;
        }
        hoverRing.setAttribute("cx", best.cx.toFixed(1));
        hoverRing.setAttribute("cy", best.cy.toFixed(1));
        hoverRing.setAttribute("visibility", "visible");
        const stepX = (xmax - xmin) / gScale;
        const stepY = (ymax - ymin) / gScale;
        showTooltip(e, `${MODEL_META[m].label} · display cell`, [
          {
            color: MODEL_META[m].color,
            value: fmt(best.xVal, stepX < 0.05 ? 3 : 2),
            label: `score (±${fmt(stepX / 2, stepX < 0.05 ? 3 : 2)})`,
          },
          {
            value: fmt(best.yVal, stepY < 0.05 ? 3 : 2),
            label: `${meta.rmse_unit} (±${fmt(stepY / 2, stepY < 0.05 ? 3 : 2)})`,
          },
          {
            value: `${fmtInt(best.cnt)} pair${best.cnt > 1 ? "s" : ""}`,
            label: `of ${fmtInt(view.rows)} total`,
          },
        ]);
      });
      hitRect.addEventListener("pointerleave", () => {
        hoverRing.setAttribute("visibility", "hidden");
        hideTooltip();
      });

      svg._hitRect = hitRect;
      svg._hoverRing = hoverRing;
      footerNote = `Observed x: [${fmt(xmin, 2)}, ${fmt(xmax, 2)}] · ${fmtInt(decoded.pts.length)} occupied cells (${fmtInt(view.rows)} pairs)`;
    } else {
      let maxCount = 1;
      for (let i = 1; i < sc.c.length; i += 2) {
        if (sc.c[i] > maxCount) maxCount = sc.c[i];
      }
      for (let k = 0; k < sc.c.length; k += 2) {
        const idx = sc.c[k];
        const cnt = sc.c[k + 1];
        const ix = Math.floor(idx / nb);
        const iy = idx % nb;
        const t = Math.log1p(cnt) / Math.log1p(maxCount);
        const rampIdx = Math.min(6, Math.floor(t * 6.999));
        const rx = left + ix * cw + 1;
        const ry = top + (nb - 1 - iy) * ch + 1;
        const xLo = xmin + (ix / nb) * (xmax - xmin);
        const xHi = xmin + ((ix + 1) / nb) * (xmax - xmin);
        const yLo = ymin + (iy / nb) * (ymax - ymin);
        const yHi = ymin + ((iy + 1) / nb) * (ymax - ymin);

        const rect = svgEl("rect", {
          x: rx,
          y: ry,
          width: Math.max(1, cw - 2),
          height: Math.max(1, ch - 2),
          rx: 2,
          fill: RAMP_SEQ[rampIdx],
          style: "cursor:pointer",
        });
        bindInteractiveMark(rect, () => ({
          heading: `${MODEL_META[m].label} · bin (${ix + 1}, ${iy + 1})`,
          rows: [
            { value: `${fmtInt(cnt)} pairs`, label: `of ${fmtInt(view.rows)}` },
            { value: `[${fmt(xLo, 2)}, ${fmt(xHi, 2)}]`, label: "score bin" },
            { value: `[${fmt(yLo, 2)}, ${fmt(yHi, 2)}]`, label: meta.rmse_unit },
          ],
        }));
        svg.appendChild(rect);
      }
      footerNote = `Observed x: [${fmt(xmin, 2)}, ${fmt(xmax, 2)}] · max 8×8 cell: ${fmtInt(maxCount)} pairs`;
    }

    if (showTrend && sc.q.length >= 2) {
      const upper = sc.q.map((q) => `${xOf(q[0])},${yOf(q[3])}`);
      const lower = [...sc.q].reverse().map((q) => `${xOf(q[0])},${yOf(q[2])}`);
      svg.appendChild(
        svgEl("polygon", {
          points: [...upper, ...lower].join(" "),
          fill: mode === "dots" ? "var(--ink)" : MODEL_META[m].color,
          "fill-opacity": mode === "dots" ? "0.11" : "0.18",
          "pointer-events": "none",
        })
      );
      svg.appendChild(
        svgEl("polyline", {
          points: sc.q.map((q) => `${xOf(q[0])},${yOf(q[1])}`).join(" "),
          fill: "none",
          stroke: "var(--surface-0)",
          "stroke-width": "3.8",
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
          "pointer-events": "none",
        })
      );
      svg.appendChild(
        svgEl("polyline", {
          points: sc.q.map((q) => `${xOf(q[0])},${yOf(q[1])}`).join(" "),
          fill: "none",
          stroke: mode === "dots" ? "var(--ink)" : MODEL_META[m].color,
          "stroke-width": "2",
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
          "pointer-events": "none",
        })
      );
    }

    sc.q.forEach((q, qi) => {
      if (showTrend) {
        const cx = xOf(q[0]);
        const cy = yOf(q[1]);
        if (mode === "dots") {
          svg.appendChild(
            svgEl("circle", {
              cx,
              cy,
              r: 3.0,
              fill: "var(--ink)",
              stroke: "var(--surface-1)",
              "stroke-width": "1.5",
              "pointer-events": "none",
            })
          );
        } else {
          const g = svgEl("g", { style: "cursor:pointer" });
          g.appendChild(
            svgEl("circle", {
              cx,
              cy,
              r: 3.4,
              fill: MODEL_META[m].color,
              stroke: "var(--surface-1)",
              "stroke-width": "1.6",
            })
          );
          g.appendChild(svgEl("circle", { cx, cy, r: 9, fill: "transparent" }));
          bindInteractiveMark(g, () => ({
            heading: `${MODEL_META[m].label} · score quantile bin ${qi + 1}/${sc.q.length}`,
            rows: [
              { color: MODEL_META[m].color, value: fmt(q[1], 3), label: `median ${meta.rmse_unit}` },
              { value: `[${fmt(q[2], 3)}, ${fmt(q[3], 3)}]`, label: "target IQR (p25–p75)" },
              { value: fmt(q[0], 2), label: "median score" },
              { value: `${fmtInt(q[4])} pairs`, label: "bin count" },
            ],
          }));
          svg.appendChild(g);
        }
      }

      tableBody.appendChild(
        el("tr", {}, [
          el("td", { text: qi === 0 ? MODEL_META[m].label : "" }),
          el("td", { text: `Bin ${qi + 1} / ${sc.q.length}` }),
          el("td", { className: "num", text: fmtInt(q[4]) }),
          el("td", { className: "num", text: fmt(q[0], 3) }),
          el("td", { className: "num", text: fmt(q[1], 3) }),
          el("td", { className: "num", text: `[${fmt(q[2], 3)}, ${fmt(q[3], 3)}]` }),
        ])
      );
    });

    if (svg._hitRect) {
      svg.appendChild(svg._hitRect);
      delete svg._hitRect;
    }
    if (svg._hoverRing) {
      svg.appendChild(svg._hoverRing);
      delete svg._hoverRing;
    }

    const xMid = (xmin + xmax) / 2;
    const yMid = (ymin + ymax) / 2;
    svg.appendChild(svgEl("text", { x: left, y: H - 23, "text-anchor": "start", text: fmt(xmin, 1) }));
    svg.appendChild(svgEl("text", { x: left + pw / 2, y: H - 23, "text-anchor": "middle", text: fmt(xMid, 1) }));
    svg.appendChild(svgEl("text", { x: W - right, y: H - 23, "text-anchor": "end", text: fmt(xmax, 1) }));
    svg.appendChild(
      svgEl("text", {
        x: left + pw / 2,
        y: H - 6,
        "text-anchor": "middle",
        fill: "var(--ink-3)",
        text: `${MODEL_META[m].label} quality score`,
      })
    );
    svg.appendChild(svgEl("text", { x: left - 6, y: top + 10, "text-anchor": "end", text: fmt(ymax, 2) }));
    svg.appendChild(svgEl("text", { x: left - 6, y: top + ph / 2 + 4, "text-anchor": "end", text: fmt(yMid, 2) }));
    svg.appendChild(svgEl("text", { x: left - 6, y: top + ph, "text-anchor": "end", text: fmt(ymin, 2) }));

    const sw = el("span", { className: "swatch-dot" });
    sw.style.background = MODEL_META[m].color;
    const card = el("div", { className: "scatter-panel" }, [
      el("div", { className: "scatter-panel-head" }, [
        el("strong", {}, [el("span", { className: "legend-item" }, [sw, MODEL_META[m].label])]),
        el("span", {
          className: "muted mono",
          text: `SROCC ${fmt(sroccVal, 3)} · KROCC ${fmt(kroccVal, 3)}`,
        }),
      ]),
      svg,
      el("div", {
        className: "muted-3",
        style: "font-size:11px",
        text: footerNote,
      }),
    ]);
    stageNode.appendChild(card);
  }
}

function renderKadidDiagTable(diagBody, diagMap, includeInternal) {
  diagBody.replaceChildren();
  const keys = includeInternal ? ["A", "S2", "BA", "B", "V2"] : PEER_KEYS;
  for (const m of keys) {
    const d = diagMap[m];
    const sw = el("span", { className: "swatch-dot" });
    sw.style.background = MODEL_META[m].color;
    diagBody.appendChild(
      el("tr", {}, [
        el("td", {}, [el("span", { className: "legend-item" }, [sw, MODEL_META[m].label])]),
        el("td", { className: "num", text: fmt(d.geo.mad, 4) }),
        el("td", { className: "num", text: fmt(d.geo.p99d, 4) }),
        el("td", { className: "num", text: fmt(d.geo.maxd, 4) }),
        el("td", { className: "num", text: fmt(d.geo.out4, 4) }),
        el("td", { className: "num", text: fmt(d.geo.clump, 4) }),
        el("td", { className: "num", text: fmt(d.raw.mad, 2) }),
        el("td", { className: "num", text: fmt(d.raw.residual_p99, 2) }),
        el("td", { className: "num", text: fmt(d.raw.clump, 4) }),
        el("td", { className: "num", text: `[${fmt(d.raw.pred_min, 1)}, ${fmt(d.raw.pred_max, 1)}]` }),
      ])
    );
  }
}

function renderKadidTypes(chartWrap, theadRow, tableBody, typesMap, groupFilter, includeInternal) {
  const rows = Object.keys(KADID_DISTORTION_META)
    .sort()
    .map((code) => ({
      code,
      name: KADID_DISTORTION_META[code][0],
      group: KADID_DISTORTION_META[code][1],
      models: typesMap[code],
    }))
    .filter((r) => groupFilter === "all" || r.group === groupFilter);

  const models = includeInternal ? ["A", "S2", "BA", "B", "V2"] : PEER_KEYS;
  const W = 880;
  const left = 240;
  const right = 28;
  const top = 22;
  const rowH = 30;
  const bottom = 32;
  const H = top + rows.length * rowH + bottom;
  const plotW = W - left - right;
  const d0 = -0.15;
  const d1 = 1.0;
  const xOf = (v) => left + ((Math.max(d0, Math.min(d1, v)) - d0) / (d1 - d0)) * plotW;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "chart",
    role: "img",
    "aria-label": "KADID-10k TERMINAL per-distortion signed SROCC",
  });

  const gridG = svgEl("g", { class: "grid" });
  for (const t of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
    const xx = xOf(t);
    gridG.appendChild(svgEl("line", { x1: xx, y1: top, x2: xx, y2: H - bottom }));
    svg.appendChild(svgEl("text", { x: xx, y: H - 10, "text-anchor": "middle", text: fmt(t, 1) }));
  }
  svg.appendChild(gridG);

  rows.forEach((row, idx) => {
    const cy = top + idx * rowH + rowH / 2;
    if (idx > 0) {
      svg.appendChild(
        svgEl("line", {
          x1: 0,
          y1: top + idx * rowH,
          x2: W,
          y2: top + idx * rowH,
          stroke: "var(--border)",
          "stroke-opacity": "0.4",
        })
      );
    }
    svg.appendChild(
      svgEl("text", {
        x: left - 10,
        y: cy + 4,
        "text-anchor": "end",
        fill: "var(--ink)",
        text: `${row.code} · ${row.name}`,
      })
    );

    const aSt = row.models.A;
    svg.appendChild(
      svgEl("line", {
        x1: xOf(aSt[1]),
        y1: cy,
        x2: xOf(aSt[2]),
        y2: cy,
        stroke: MODEL_META.A.color,
        "stroke-opacity": "0.45",
        "stroke-width": "2",
      })
    );

    const mid = (models.length - 1) / 2;
    models.forEach((m, mi) => {
      const st = row.models[m];
      const cx = xOf(st[0]);
      const yOff = cy + (mi - mid) * 3.6;
      const g = svgEl("g", { style: "cursor:pointer" });
      g.appendChild(
        svgEl("circle", {
          cx,
          cy: yOff,
          r: m === "A" ? 4.8 : 3.9,
          fill: MODEL_META[m].color,
          stroke: "var(--surface-1)",
          "stroke-width": "1.8",
        })
      );
      g.appendChild(svgEl("circle", { cx, cy, r: 12, fill: "transparent" }));
      bindInteractiveMark(g, () => ({
        heading: `KADID type ${row.code} · ${row.name} (80 pairs / 16 refs)`,
        rows: models.map((mk) => {
          const ms = row.models[mk];
          return {
            color: MODEL_META[mk].color,
            value: `${fmt(ms[0], 3)} [${fmt(ms[1], 3)}, ${fmt(ms[2], 3)}]`,
            label: MODEL_META[mk].label,
          };
        }),
      }));
      svg.appendChild(g);
    });
  });

  chartWrap.replaceChildren(makeLegend(models), el("div", { className: "svg-wrap" }, [svg]));

  theadRow.replaceChildren(
    el("th", { text: "Code" }),
    el("th", { text: "Distortion (imdist_generator.m)" }),
    el("th", { text: "Family" }),
    ...models.map((m) => el("th", { className: "num", text: `${MODEL_META[m].label} [95% CI]` }))
  );

  tableBody.replaceChildren();
  for (const row of rows) {
    tableBody.appendChild(
      el("tr", {}, [
        el("td", { className: "mono", text: row.code }),
        el("td", { text: row.name }),
        el("td", { className: "muted", text: row.group }),
        ...models.map((m) => fmtStatCell(row.models[m], 3)),
      ])
    );
  }
}

function renderSpeedCharts(latWrap, rssWrap, threads) {
  const rows = SPEED_BENCHMARK.rows.filter((r) => r.threads === Number(threads));
  const arms = [
    { key: "a", rssIdx: 0, label: "zensim A", color: "var(--series-1)" },
    { key: "fast_ssim2", rssIdx: 1, label: "fast-ssim2 0.9.1", color: "var(--series-2)" },
    { key: "margarine", rssIdx: 2, label: "margarine (in-process)", color: "var(--series-3)" },
  ];

  const legend = el(
    "div",
    { className: "legend-row" },
    arms.map((a) => {
      const sw = el("span", { className: "swatch-rect" });
      sw.style.background = a.color;
      return el("span", { className: "legend-item" }, [sw, a.label]);
    })
  );

  function buildRelativeRatioChart(title, getVals, unitFmt) {
    const W = 560;
    const left = 135;
    const right = 88;
    const top = 26;
    const groupH = 66;
    const barH = 15;
    const gap = 2;
    const H = top + rows.length * groupH + 28;
    const pw = W - left - right;

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      class: "chart",
      role: "img",
      "aria-label": title,
    });
    svg.appendChild(svgEl("text", { x: 0, y: 15, fill: "var(--ink)", "font-weight": "600", text: title }));

    const gridG = svgEl("g", { class: "grid" });
    for (const f of [0, 0.25, 0.5, 0.75, 1.0]) {
      const xx = left + f * pw;
      gridG.appendChild(svgEl("line", { x1: xx, y1: top, x2: xx, y2: H - 24 }));
      svg.appendChild(
        svgEl("text", { x: xx, y: H - 8, "text-anchor": "middle", text: `${Math.round(f * 100)}%` })
      );
    }
    svg.appendChild(gridG);

    rows.forEach((r, ri) => {
      const vals = arms.map((a) => getVals(r, a));
      const cellMax = Math.max(...vals.map((v) => v.main));
      const gy = top + ri * groupH + 8;
      svg.appendChild(
        svgEl("text", { x: left - 10, y: gy + 24, "text-anchor": "end", fill: "var(--ink)", text: r.size })
      );

      arms.forEach((a, ai) => {
        const v = vals[ai];
        const by = gy + ai * (barH + gap);
        const bw = Math.max(2, (v.main / cellMax) * pw);
        const g = svgEl("g", { style: "cursor:pointer" });
        g.appendChild(
          svgEl("rect", {
            x: left,
            y: by,
            width: bw,
            height: barH,
            rx: 3,
            fill: a.color,
          })
        );
        g.appendChild(
          svgEl("text", {
            x: left + bw + 6,
            y: by + 11.5,
            fill: "var(--ink-2)",
            text: unitFmt(v),
          })
        );
        g.appendChild(svgEl("rect", { x: left, y: by - 2, width: pw + right, height: barH + 4, fill: "transparent" }));
        bindInteractiveMark(g, () => ({
          heading: `${r.size} · ${threads} thread${threads > 1 ? "s" : ""}`,
          rows: arms.map((arm, idx) => ({
            color: arm.color,
            value: vals[idx].detail,
            label: arm.label,
          })),
        }));
        svg.appendChild(g);
      });
    });
    return svg;
  }

  const latSvg = buildRelativeRatioChart(
    `Single-call p50 latency (${threads}T, scaled to slowest arm per size)`,
    (r, a) => ({
      main: r[a.key][0],
      detail: `${fmt(r[a.key][0], 2)} ms p50 (${fmt(r[a.key][1], 2)} ms p95)`,
    }),
    (v) => `${fmt(v.main, 2)} ms`
  );

  const rssSvg = buildRelativeRatioChart(
    `Total-process peak RSS (${threads}T, scaled to largest RSS per size)`,
    (r, a) => {
      const kib = r.rss_kib[a.rssIdx];
      return {
        main: kib,
        detail: `${fmtInt(kib)} KiB (${fmt(kib / 1024, 1)} MiB)`,
      };
    },
    (v) => (v.main >= 102400 ? `${fmt(v.main / 1024, 0)} MiB` : `${fmt(v.main / 1024, 1)} MiB`)
  );

  latWrap.replaceChildren(legend.cloneNode(true), el("div", { className: "svg-wrap" }, [latSvg]));
  rssWrap.replaceChildren(legend, el("div", { className: "svg-wrap" }, [rssSvg]));
}

function renderSelectionSection(includeF3) {
  // 1. 9-collection peer chart + table
  const chartWrap = document.getElementById("eval9-chart-wrap");
  const theadRow = document.getElementById("eval9-thead-row");
  const evalBody = document.getElementById("eval9-tbody");

  const series = [
    { key: "a0", label: "zensim A", color: "var(--series-1)", get: (c) => c.a[0] },
    { key: "s2", label: "SSIMULACRA2", color: "var(--series-2)", get: (c) => c.s2 },
    { key: "ba", label: "Butteraugli (3-norm)", color: "var(--series-3)", get: (c) => c.ba },
  ];
  if (includeF3) {
    series.push({ key: "f3", label: "F3 (internal 11-MOS)", color: "var(--series-4)", get: (c) => c.f3[0] });
  }

  const legend = el(
    "div",
    { className: "legend-row" },
    series.map((s) => {
      const sw = el("span", { className: "swatch-dot" });
      sw.style.background = s.color;
      return el("span", { className: "legend-item" }, [sw, s.label]);
    })
  );

  const W = 560;
  const left = 148;
  const right = 22;
  const top = 14;
  const rowH = 31;
  const bottom = 30;
  const H = top + SELECTION_EVAL.collections.length * rowH + bottom;
  const pw = W - left - right;
  const d0 = 0.35;
  const d1 = 1.0;
  const xOf = (v) => left + ((Math.max(d0, Math.min(d1, v)) - d0) / (d1 - d0)) * pw;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "chart",
    role: "img",
    "aria-label": "9-collection SROCC comparison against SSIMULACRA2 and Butteraugli",
  });

  const gridG = svgEl("g", { class: "grid" });
  for (const t of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
    const xx = xOf(t);
    gridG.appendChild(svgEl("line", { x1: xx, y1: top, x2: xx, y2: H - bottom }));
    svg.appendChild(svgEl("text", { x: xx, y: H - 8, "text-anchor": "middle", text: fmt(t, 1) }));
  }
  svg.appendChild(gridG);

  const mid = (series.length - 1) / 2;
  SELECTION_EVAL.collections.forEach((c, idx) => {
    const cy = top + idx * rowH + rowH / 2;
    if (idx > 0) {
      svg.appendChild(
        svgEl("line", {
          x1: 0,
          y1: top + idx * rowH,
          x2: W,
          y2: top + idx * rowH,
          stroke: "var(--border)",
          "stroke-opacity": "0.4",
        })
      );
    }
    svg.appendChild(
      svgEl("text", { x: left - 10, y: cy + 4, "text-anchor": "end", fill: "var(--ink)", text: c.name })
    );

    series.forEach((s, si) => {
      const val = s.get(c);
      const cx = xOf(val);
      const yOff = cy + (si - mid) * 3.4;
      const g = svgEl("g", { style: "cursor:pointer" });
      g.appendChild(
        svgEl("circle", {
          cx,
          cy: yOff,
          r: s.key === "a0" ? 4.6 : 3.9,
          fill: s.color,
          stroke: "var(--surface-1)",
          "stroke-width": "1.7",
        })
      );
      g.appendChild(svgEl("circle", { cx, cy, r: 11, fill: "transparent" }));
      bindInteractiveMark(g, () => ({
        heading: `${c.name} (n = ${fmtInt(c.n)})`,
        rows: [
          ...series.map((sr) => ({
            color: sr.color,
            value: fmt(sr.get(c), 3),
            label: `${sr.label} SROCC`,
          })),
          { value: `${fmt(c.a[1], 3)} PLCC · ${fmt(c.a[3], 1)}% MedAE`, label: "zensim A fit" },
        ],
      }));
      svg.appendChild(g);
    });
  });

  chartWrap.replaceChildren(legend, el("div", { className: "svg-wrap" }, [svg]));

  const headCells = [
    el("th", { text: "Collection" }),
    el("th", { className: "num", text: "Rows" }),
    el("th", { className: "num", text: "zensim A SROCC" }),
    el("th", { className: "num", text: "SSIMULACRA2" }),
    el("th", { className: "num", text: "Butteraugli (3-norm)" }),
    el("th", { className: "num", text: "A: PLCC / KROCC / MedAE%" }),
  ];
  if (includeF3) {
    headCells.push(el("th", { className: "num", text: "F3: SROCC / MedAE%" }));
  }
  theadRow.replaceChildren(...headCells);

  evalBody.replaceChildren();
  for (const c of SELECTION_EVAL.collections) {
    const cells = [
      el("td", { text: c.name }),
      el("td", { className: "num", text: fmtInt(c.n) }),
      el("td", { className: "num", text: fmt(c.a[0], 3) }),
      el("td", { className: "num", text: fmt(c.s2, 3) }),
      el("td", { className: "num", text: fmt(c.ba, 3) }),
      el("td", {
        className: "num muted",
        text: `${fmt(c.a[1], 3)} / ${fmt(c.a[2], 3)} / ${fmt(c.a[3], 1)}%`,
      }),
    ];
    if (includeF3) {
      cells.push(el("td", { className: "num muted", text: `${fmt(c.f3[0], 3)} / ${fmt(c.f3[3], 1)}%` }));
    }
    evalBody.appendChild(el("tr", {}, cells));
  }

  // 2. Multi-peer summary table
  const peerEvalBody = document.getElementById("eval-peers-tbody");
  peerEvalBody.replaceChildren();
  for (const p of SELECTION_EVAL.peers) {
    if (p.internal && !includeF3) continue;
    peerEvalBody.appendChild(
      el("tr", {}, [
        el("td", { text: p.metric }),
        el("td", { className: "num", text: `${fmt(p.eight[0], 2)}% / ${fmt(p.eight[1], 3)}` }),
        el("td", { className: "num", text: `${fmt(p.shared3[0], 2)}% / ${fmt(p.shared3[1], 3)}` }),
        el("td", { className: "num", text: `${fmt(p.compression[0], 3)} / ${fmt(p.compression[1], 2)}%` }),
        el("td", { className: "num", text: `${fmt(p.noncodec[0], 3)} / ${fmt(p.noncodec[1], 2)}%` }),
      ])
    );
  }

  // 3. Non-codec 7-collection peer table
  const noncodecBody = document.getElementById("noncodec-peers-tbody");
  noncodecBody.replaceChildren();
  for (const r of SELECTION_EVAL.noncodec_sets) {
    if (r.internal && !includeF3) continue;
    noncodecBody.appendChild(
      el("tr", {}, [
        el("td", { text: r.metric }),
        el("td", { className: "num", text: `${fmt(r.pooled[0], 3)} / ${fmt(r.pooled[1], 2)}%` }),
        ...r.sets.map((s) => el("td", { className: "num", text: fmt(s, 3) })),
      ])
    );
  }

  // 4. Pre-final confirmation reads
  const confReadsBody = document.getElementById("conf-reads-tbody");
  confReadsBody.replaceChildren();
  for (const cr of SELECTION_EVAL.confirmation_reads) {
    confReadsBody.appendChild(
      el("tr", {}, [
        el("td", { text: cr.name }),
        el("td", { className: "muted", text: cr.scope }),
        el("td", { className: "mono", text: cr.a }),
        el("td", { className: "muted", text: cr.peers + (includeF3 ? cr.peers_f3 : "") }),
      ])
    );
  }
}

function populateStaticTables(data) {
  const confSpec = [
    ["B", "Research B (formula rev 5, codec_target)"],
    ["V2", "Research by_v2fy (formula rev 4, seed 0 epoch 119)"],
  ];
  const confBody = document.getElementById("kadid-conf-tbody");
  for (const [key, label] of confSpec) {
    const c = data.kadid_conf[key];
    const chip = el("span", {
      className: "chip",
      "data-severity": c[5] ? "note" : "critical",
      text: c[5] ? "✓ Passed" : "✕ Failed",
    });
    confBody.appendChild(
      el("tr", {}, [
        el("td", { text: `A minus ${label}` }),
        el("td", { className: "num", text: fmt(c[0], 6) }),
        el("td", { className: "num", text: fmt(c[1], 6) }),
        el("td", { className: "num", text: `≥ ${fmt(c[2], 6)}` }),
        el("td", { className: "num", text: `[${fmt(c[3], 6)}, ${fmt(c[4], 6)}]` }),
        el("td", {}, [chip]),
      ])
    );
  }

  const mhBody = document.getElementById("multihost-speed-tbody");
  if (mhBody) {
    for (const r of SPEED_BENCHMARK.multihost_rows) {
      mhBody.appendChild(
        el("tr", {}, [
          el("td", { text: r.cpu }),
          el("td", { className: "num", text: String(r.threads) }),
          el("td", { className: "num", text: `${fmt(r.s64[0], 2)} / ${fmt(r.s64[1], 2)}` }),
          el("td", { className: "num", text: `${fmt(r.s256[0], 2)} / ${fmt(r.s256[1], 2)}` }),
          el("td", { className: "num", text: `${fmt(r.s1024[0], 2)} / ${fmt(r.s1024[1], 2)}` }),
          el("td", { className: "num", text: `${fmt(r.s12m[0], 2)} / ${fmt(r.s12m[1], 2)}` }),
          el("td", { className: "num", text: fmt(r.beta, 2) }),
          el("td", { className: "num", text: fmtInt(r.rss1k) }),
          el("td", { className: "num", text: fmtInt(r.rss12m) }),
        ])
      );
    }
  }

  const speedBody = document.getElementById("speed-tbody");
  for (const r of SPEED_BENCHMARK.rows) {
    speedBody.appendChild(
      el("tr", {}, [
        el("td", { className: "num", text: String(r.threads) }),
        el("td", { text: r.size }),
        el("td", { className: "num", text: `${fmt(r.a[0], 2)} / ${fmt(r.a[1], 2)}` }),
        el("td", { className: "num", text: `${fmt(r.fast_ssim2[0], 2)} / ${fmt(r.fast_ssim2[1], 2)}` }),
        el("td", { className: "num", text: `${fmt(r.margarine[0], 2)} / ${fmt(r.margarine[1], 2)}` }),
        el("td", { className: "num", text: fmt(r.ratios[0], 3) }),
        el("td", { className: "num", text: fmt(r.ratios[1], 3) }),
        el("td", { className: "num", text: fmtInt(r.rss_kib[0]) }),
        el("td", { className: "num", text: fmtInt(r.rss_kib[1]) }),
        el("td", { className: "num", text: fmtInt(r.rss_kib[2]) }),
      ])
    );
  }

  const olsBody = document.getElementById("ols-tbody");
  for (const r of SPEED_BENCHMARK.ols_fits) {
    olsBody.appendChild(
      el("tr", {}, [
        el("td", { className: "num", text: String(r.threads) }),
        el("td", { text: r.arm }),
        el("td", { className: "num", text: fmt(r.alpha_ms, 3) }),
        el("td", { className: "num", text: fmt(r.beta_ms_per_mp, 3) }),
        el("td", { className: "num", text: fmt(r.r2, 5) }),
      ])
    );
  }

  const cutBody = document.getElementById("cutoff-tbody");
  for (const r of SPEED_BENCHMARK.cutoff_rows) {
    cutBody.appendChild(
      el("tr", {}, [
        el("td", { className: "num", text: String(r.threads) }),
        el("td", { text: r.size }),
        el("td", { className: "num", text: `${fmt(r.before[0], 2)} / ${fmt(r.before[1], 2)}` }),
        el("td", {
          className: "num",
          text: r.after ? `${fmt(r.after[0], 2)} / ${fmt(r.after[1], 2)}` : "not measured",
        }),
      ])
    );
  }
}

async function init() {
  const themeBtn = document.getElementById("theme");
  const savedTheme = localStorage.getItem("zensim-theme");
  if (savedTheme === "light" || savedTheme === "dark") {
    document.documentElement.dataset.theme = savedTheme;
  }
  const syncThemeLabel = () => {
    const cur = document.documentElement.dataset.theme || "dark";
    themeBtn.textContent = cur === "dark" ? "Light" : "Dark";
  };
  syncThemeLabel();
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("zensim-theme", next);
    syncThemeLabel();
  });

  const rampStrip = document.getElementById("scatter-ramp");
  for (const c of RAMP_SEQ) {
    const s = el("span");
    s.style.background = c;
    rampStrip.appendChild(s);
  }

  const resp = await fetch("./evaluations-data.json");
  if (!resp.ok) throw new Error(`Failed to load evaluations-data.json: ${resp.status}`);
  const data = await resp.json();
  const dotGrid = Number(data.dot_grid || 256);

  document.getElementById("foot-commits").textContent =
    `zensim ${data.commits.zensim.slice(0, 10)} · zensimx ${data.commits.zensimx.slice(0, 10)} · bake ${data.commits.bake_sha256.slice(0, 12)}…`;

  const frozenInternalWrap = document.getElementById("frozen-internal-wrap");
  const frozenInternalCheck = document.getElementById("frozen-internal-check");
  renderFrozenChart(document.getElementById("frozen-chart-wrap"), data.views);
  const syncFrozenTable = () =>
    renderFrozenTable(document.getElementById("frozen-tbody"), data.views, frozenInternalCheck.checked);
  syncFrozenTable();
  frozenInternalCheck.addEventListener("change", syncFrozenTable);
  wireViewToggle(
    document.getElementById("frozen-toggle"),
    document.getElementById("frozen-chart-wrap"),
    document.getElementById("frozen-table-wrap"),
    (mode) => {
      frozenInternalWrap.hidden = mode !== "table";
    }
  );

  renderClicSection(
    document.getElementById("clic-chart-wrap"),
    document.getElementById("clic-margin-tbody"),
    document.getElementById("clic-summary-tbody"),
    data.clic22
  );
  wireViewToggle(
    document.getElementById("clic-toggle"),
    document.getElementById("clic-chart-wrap"),
    document.getElementById("clic-table-wrap")
  );

  const popSelect = document.getElementById("scatter-select");
  for (const v of data.views) {
    popSelect.appendChild(
      el("option", { value: v.id, text: `${VIEW_META[v.id].title} (n = ${fmtInt(v.rows)})` })
    );
  }
  let scatterMode = "dots";
  const scatterTrendCheck = document.getElementById("scatter-trend-check");
  const scatterTrendWrap = document.getElementById("scatter-trend-wrap");
  const legendDots = document.getElementById("scatter-legend-dots");
  const legendDensity = document.getElementById("scatter-legend-density");
  const scatterStage = document.getElementById("scatter-stage");
  const scatterTableWrap = document.getElementById("scatter-table-wrap");

  const updateScatter = () => {
    const view = data.views.find((v) => v.id === popSelect.value) || data.views[0];
    const isTable = scatterMode === "table";
    scatterStage.hidden = isTable;
    scatterTableWrap.hidden = !isTable;
    scatterTrendWrap.hidden = isTable;
    legendDots.hidden = scatterMode !== "dots";
    legendDensity.hidden = scatterMode !== "density";
    renderScatters(
      scatterStage,
      document.getElementById("scatter-tbody"),
      document.getElementById("scatter-disclosure"),
      view,
      dotGrid,
      scatterMode === "density" ? "density" : "dots",
      scatterTrendCheck.checked
    );
  };
  popSelect.addEventListener("change", updateScatter);
  scatterTrendCheck.addEventListener("change", updateScatter);
  const scatterBtns = document.getElementById("scatter-toggle").querySelectorAll("button[data-mode]");
  scatterBtns.forEach((b) => {
    b.addEventListener("click", () => {
      scatterMode = b.dataset.mode;
      scatterBtns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      updateScatter();
    });
  });
  updateScatter();

  const kadidDiagInternal = document.getElementById("kadid-diag-internal-check");
  const syncKadidDiag = () =>
    renderKadidDiagTable(document.getElementById("kadid-diag-tbody"), data.kadid_diag, kadidDiagInternal.checked);
  kadidDiagInternal.addEventListener("change", syncKadidDiag);
  syncKadidDiag();

  const kadidSelect = document.getElementById("kadid-group-select");
  const kadidInternal = document.getElementById("kadid-internal-check");
  const updateKadid = () => {
    renderKadidTypes(
      document.getElementById("kadid-chart-wrap"),
      document.getElementById("kadid-types-thead-row"),
      document.getElementById("kadid-types-tbody"),
      data.kadid_types,
      kadidSelect.value,
      kadidInternal.checked
    );
  };
  kadidSelect.addEventListener("change", updateKadid);
  kadidInternal.addEventListener("change", updateKadid);
  updateKadid();
  wireViewToggle(
    document.getElementById("kadid-toggle"),
    document.getElementById("kadid-chart-wrap"),
    document.getElementById("kadid-table-wrap")
  );

  const threadSelect = document.getElementById("speed-threads-select");
  const updateSpeed = () => {
    renderSpeedCharts(
      document.getElementById("speed-lat-wrap"),
      document.getElementById("speed-rss-wrap"),
      Number(threadSelect.value)
    );
  };
  threadSelect.addEventListener("change", updateSpeed);
  updateSpeed();

  const evalF3Check = document.getElementById("eval-f3-check");
  const syncSelection = () => renderSelectionSection(evalF3Check.checked);
  evalF3Check.addEventListener("change", syncSelection);
  syncSelection();
  wireViewToggle(
    document.getElementById("eval9-toggle"),
    document.getElementById("eval9-chart-wrap"),
    document.getElementById("eval9-table-wrap")
  );

  populateStaticTables(data);
  window.evalsReady = true;
}

init().catch((err) => {
  window.evalsError = String(err?.message || err);
  console.error(err);
});
