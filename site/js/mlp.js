// mlp.js — ZNPR v2 binary parser + 228→128→1 forward pass in JS.
//
// Matches `zenpredict::inference::forward` exactly:
//   1. Scale inputs: x' = (x - mean) / safe_scale  (zero-scale → 1.0)
//   2. Per-layer: bias init → SAXPY matmul → activation
//   3. Activation: Identity / ReLU / LeakyReLU(α=0.01)
//
// Used by compare-worker.js to apply any V_X bake to the zensim per-pair
// features (feat_0..feat_227, the first 228 of the 300 in the parquet)
// pulled from a unified parquet via DuckDB-WASM.
//
// Format reference (canonical Rust impl):
//   /home/lilith/work/zen/zenanalyze/zenpredict/src/model.rs
//   /home/lilith/work/zen/zenanalyze/zenpredict/src/inference.rs

const ZNPR_MAGIC      = 0x52504e5a; // "ZNPR" little-endian
const HEADER_SIZE     = 128;
const LAYER_ENTRY_SIZE = 48;
const LEAKY_RELU_ALPHA = 0.01;

const ACT_IDENTITY  = 0;
const ACT_RELU      = 1;
const ACT_LEAKYRELU = 2;

const DTYPE_F32 = 0;
const DTYPE_F16 = 1;
const DTYPE_I8  = 2;

/**
 * Parse a ZNPR v2 binary into a model object.
 *
 * @param {ArrayBuffer | Uint8Array} bytes - raw bake file contents.
 * @returns {{
 *   nInputs: number,
 *   nOutputs: number,
 *   nLayers: number,
 *   schemaHash: bigint,
 *   scalerMean: Float32Array,
 *   scalerScale: Float32Array,
 *   layers: Array<{
 *     inDim: number, outDim: number, activation: number,
 *     weightDtype: number, weights: Float32Array, biases: Float32Array
 *   }>
 * }}
 */
export function parseZnpr(bytes) {
  const buf = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const dv = new DataView(buf);
  if (buf.byteLength < HEADER_SIZE) throw new Error(`bake too small: ${buf.byteLength} < ${HEADER_SIZE}`);
  const magic = dv.getUint32(0, true);
  if (magic !== ZNPR_MAGIC) throw new Error(`bad magic 0x${magic.toString(16)}, expected ZNPR`);

  const version    = dv.getUint16(4, true);
  /* flags */        dv.getUint16(6, true);
  const nInputs    = dv.getUint32(8, true);
  const nOutputs   = dv.getUint32(12, true);
  const nLayers    = dv.getUint32(16, true);
  // 20..24 = _pad0
  const schemaHash = dv.getBigUint64(24, true);

  // Sections: (offset:u32, len:u32). Header layout follows:
  //   32..40 scaler_mean
  //   40..48 scaler_scale
  //   48..56 layer_table
  //   56..64 feature_bounds
  //   64..72 metadata
  //   72..80 output_specs   (v3)
  //   80..88 discrete_sets  (v3)
  //   88..96 sparse_overrides (v3)
  //   96..128 reserved
  const sec = (off) => ({ offset: dv.getUint32(off, true), len: dv.getUint32(off + 4, true) });
  const scalerMeanSec  = sec(32);
  const scalerScaleSec = sec(40);
  const layerTableSec  = sec(48);

  const scalerMean  = readF32(buf, scalerMeanSec,  nInputs);
  const scalerScale = readF32(buf, scalerScaleSec, nInputs);

  // Layer table: nLayers × LAYER_ENTRY_SIZE bytes.
  // Each entry: in_dim u32, out_dim u32, activation u8, dtype u8,
  // flags u16, weights Section, scales Section, biases Section, reserved.
  const layers = [];
  for (let i = 0; i < nLayers; i++) {
    const base = layerTableSec.offset + i * LAYER_ENTRY_SIZE;
    const inDim    = dv.getUint32(base, true);
    const outDim   = dv.getUint32(base + 4, true);
    const activation  = dv.getUint8(base + 8);
    const weightDtype = dv.getUint8(base + 9);
    // 10..12 flags
    const weightsSec = { offset: dv.getUint32(base + 12, true), len: dv.getUint32(base + 16, true) };
    // scales section at base + 20 / + 24 — only used for I8 dtype, skipped for F32/F16.
    const biasesSec  = { offset: dv.getUint32(base + 28, true), len: dv.getUint32(base + 32, true) };

    if (weightDtype !== DTYPE_F32) {
      throw new Error(`layer ${i}: weight_dtype ${weightDtype} not implemented in JS (need F32; F16/I8 TODO)`);
    }
    const weights = readF32(buf, weightsSec, inDim * outDim);
    const biases  = readF32(buf, biasesSec,  outDim);
    layers.push({ inDim, outDim, activation, weightDtype, weights, biases });
  }

  return { version, nInputs, nOutputs, nLayers, schemaHash, scalerMean, scalerScale, layers };
}

function readF32(buf, sec, expectedLen) {
  const u8 = new Uint8Array(buf, sec.offset, sec.len);
  const bytes = expectedLen * 4;
  if (u8.byteLength !== bytes) {
    throw new Error(`section len ${u8.byteLength} ≠ ${bytes} (offset ${sec.offset}, expected ${expectedLen} × 4)`);
  }
  // Slice into an aligned buffer if needed (sec.offset may not be 4-aligned).
  if (sec.offset % 4 === 0) {
    return new Float32Array(buf, sec.offset, expectedLen);
  }
  const copy = new ArrayBuffer(bytes);
  new Uint8Array(copy).set(u8);
  return new Float32Array(copy);
}

/**
 * Run the forward pass. Bit-equivalent to zenpredict::inference::forward.
 *
 * @param {ReturnType<typeof parseZnpr>} model
 * @param {Float32Array | number[]} features - length must equal model.nInputs (228 for V_X).
 * @param {Float32Array} [scratchA] - optional reusable buffer ≥ max layer width.
 * @param {Float32Array} [scratchB] - optional reusable buffer ≥ max layer width.
 * @returns {Float32Array} new array of length model.nOutputs.
 */
export function predict(model, features, scratchA, scratchB) {
  const { nInputs, nOutputs, layers, scalerMean, scalerScale } = model;
  if (features.length !== nInputs) {
    throw new Error(`features length ${features.length} ≠ nInputs ${nInputs}`);
  }
  let maxWidth = nInputs;
  for (const l of layers) maxWidth = Math.max(maxWidth, l.outDim);
  if (!scratchA || scratchA.length < maxWidth) scratchA = new Float32Array(maxWidth);
  if (!scratchB || scratchB.length < maxWidth) scratchB = new Float32Array(maxWidth);

  // Scaler: x' = (x - mean) / safe_scale. Mirrors sklearn's StandardScaler
  // _handle_zeros_in_scale: scale==0 → 1.0 (column passes through as (x - mean)).
  for (let i = 0; i < nInputs; i++) {
    const s = scalerScale[i];
    const safe = (s === 0) ? 1.0 : s;
    scratchA[i] = (features[i] - scalerMean[i]) / safe;
  }

  let input = scratchA, output = scratchB;
  let finalOut = null;

  for (let li = 0; li < layers.length; li++) {
    const { inDim, outDim, activation, weights, biases } = layers[li];
    const lastLayer = (li === layers.length - 1);
    const dst = lastLayer ? new Float32Array(outDim) : output;

    // Init with biases (broadcast).
    for (let o = 0; o < outDim; o++) dst[o] = biases[o];

    // SAXPY matmul: dst[o] += src[i] * W[i*outDim + o].
    for (let i = 0; i < inDim; i++) {
      const s = input[i];
      if (s === 0.0) continue;
      const wRow = i * outDim;
      for (let o = 0; o < outDim; o++) dst[o] += s * weights[wRow + o];
    }

    applyActivation(dst, activation);

    if (lastLayer) {
      finalOut = dst;
    } else {
      // Swap input/output for next layer.
      const tmp = input; input = output; output = tmp;
      // Copy dst into the new "input" buffer for the next iteration.
      input.set(dst.subarray(0, outDim));
    }
  }

  return finalOut.subarray(0, nOutputs);
}

function applyActivation(arr, activation) {
  switch (activation) {
    case ACT_IDENTITY: break;
    case ACT_RELU:
      for (let i = 0; i < arr.length; i++) if (arr[i] < 0) arr[i] = 0;
      break;
    case ACT_LEAKYRELU:
      for (let i = 0; i < arr.length; i++) if (arr[i] < 0) arr[i] *= LEAKY_RELU_ALPHA;
      break;
    default:
      throw new Error(`unknown activation ${activation}`);
  }
}

// Convenience: parse + predict in one call (for one-shot use).
export function applyBake(bakeBytes, features) {
  const model = parseZnpr(bakeBytes);
  return predict(model, features);
}
