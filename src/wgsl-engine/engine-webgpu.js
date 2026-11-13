// Hand-written WebGPU (WGSL) encoder for granite-embedding-97m-multilingual-r2
// (ModernBERT, 12 layers, H=384, 12 heads × 32, GLU MLP 1536, alternating
// global / sliding-window(±64) attention, RoPE). Consumes the SAME model_q4.onnx
// bytes transformers.js caches (see onnx-weights.js) — no second artifact.
//
// Why this exists (the iPhone problem in one paragraph): on WebKit, ORT-Web's
// WebGPU path drags in a 14–24 MB jspi/asyncify wasm glue, a 4 GiB shared
// WebAssembly.Memory reservation, the ORT #26827 WASM-compile hang, a bucketed
// storage-buffer pool that grows with every distinct tensor shape, and a
// per-shape shader compile (hence the warmup grid). This engine has none of
// that: no wasm, sequence length is a UNIFORM (one pipeline per kernel, compiled
// once, any S), a fixed set of buffers allocated up front, and the 43 MB token
// embedding table never touches the GPU (rows are dequantized on the CPU).
//
// Numerics: fp32 activations, int4 weights dequantized in-register — the same
// math ORT does with accuracy_level=0, i.e. what ORT-WebGPU does on desktop.
//
// Limits: uses only WebGPU DEFAULT limits (16 KiB workgroup storage, 256
// invocations, 128 MiB storage binding) so it runs anywhere WebGPU does.

import { gatherEmbeddings } from './onnx-weights.js';

const WG = 64;      // matmul: output columns per workgroup
const TM = 8;       // matmul: rows per workgroup tile
const BLK = 32;     // MatMulNBits block size (asserted at load)

// ── WGSL ──────────────────────────────────────────────────────────────────
// Shared per-forward dims: rows = B*S, S = padded sequence length.
const DIMS = /* wgsl */ `
struct Dims { rows: u32, S: u32, B: u32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
`;

const LAYERNORM = DIMS + /* wgsl */ `
struct P { H: u32, eps: f32, rowStride: u32, pad: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  // rowStride lets the final norm visit only CLS rows (row b*S).
  let row = wid.x * p.rowStride;
  if (row >= dims.rows) { return; }
  let base = row * p.H;
  var s = 0.0;
  for (var i = lid.x; i < p.H; i += 128u) { s += x[base + i]; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let mu = red[0] / f32(p.H);
  workgroupBarrier();
  var v = 0.0;
  for (var i = lid.x; i < p.H; i += 128u) { let d = x[base + i] - mu; v += d * d; }
  red[lid.x] = v; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(p.H) + p.eps);
  let ob = wid.x * p.H;   // compact output when strided (CLS rows)
  for (var i = lid.x; i < p.H; i += 128u) { y[ob + i] = (x[base + i] - mu) * r * w[i]; }
}`;

// y[rows,N] (=|+=) x[rows,K] · Wᵀ, W = MatMulNBits u4 blocks of 32, zp 8.
// One thread per output column n, TM rows per workgroup: each weight block is
// read from memory ONCE per TM rows (the query path is weight-bandwidth bound,
// rows ≤ ~16). The X tile for the current K-block is staged in shared memory.
// inMode 1 = GLU fused into the load: x(s,k) = silu(u[s,k]) * u[s,K+k], with u
// the [rows, 2K] Wi output — so the gated activation is never materialized.
const MATMUL = DIMS + /* wgsl */ `
struct P { N: u32, K: u32, accumulate: u32, inMode: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> wq: array<u32>;
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> xt: array<f32, ${TM * BLK}>;
fn loadX(r: u32, k: u32) -> f32 {
  if (p.inMode == 1u) {
    let a = x[r * 2u * p.K + k];
    let g = x[r * 2u * p.K + p.K + k];
    return a / (1.0 + exp(-a)) * g;
  }
  return x[r * p.K + k];
}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * ${WG}u + lid.x;
  let r0 = wid.y * ${TM}u;
  let nb = p.K / ${BLK}u;
  var acc: array<f32, ${TM}>;
  for (var t = 0u; t < ${TM}u; t++) { acc[t] = 0.0; }
  for (var b = 0u; b < nb; b++) {
    // stage X[r0..r0+TM, b*32..b*32+32): 256 values, 64 threads × 4
    for (var i = lid.x; i < ${TM * BLK}u; i += ${WG}u) {
      let r = r0 + i / ${BLK}u;
      xt[i] = select(0.0, loadX(r, b * ${BLK}u + i % ${BLK}u), r < dims.rows);
    }
    workgroupBarrier();
    if (n < p.N) {
      let s = sc[n * nb + b];
      let wb = (n * nb + b) * 4u;
      for (var j = 0u; j < 4u; j++) {
        let word = wq[wb + j];
        for (var e = 0u; e < 8u; e++) {
          let wv = (f32((word >> (e * 4u)) & 15u) - 8.0) * s;
          let k = j * 8u + e;
          for (var t = 0u; t < ${TM}u; t++) { acc[t] += wv * xt[t * ${BLK}u + k]; }
        }
      }
    }
    workgroupBarrier();
  }
  if (n >= p.N) { return; }
  for (var t = 0u; t < ${TM}u; t++) {
    let r = r0 + t;
    if (r < dims.rows) {
      let o = r * p.N + n;
      if (p.accumulate == 1u) { y[o] += acc[t]; } else { y[o] = acc[t]; }
    }
  }
}`;

// Register-tiled variant for indexing-sized row counts, where MATMUL turns
// compute-bound (it re-decodes every weight once per 8 rows and does 8 FMAs per
// decode). Each 128-thread workgroup owns a 64-row × 64-column output tile; per
// K-block of 32 it stages X (64×32) and the dequantized W block (32×64) in
// shared memory (8 KiB each = the 16 KiB default limit), then every thread
// accumulates an 8-row × 4-column sub-tile: per k, 2+1 vec4 shared loads feed
// 32 FMAs. Requires N % 64 == 0 (true for every granite matmul).
const MT = 64, NT = 64;
const MATMUL_TILED = DIMS + /* wgsl */ `
struct P { N: u32, K: u32, accumulate: u32, inMode: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> wq: array<u32>;
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> xs: array<vec4f, ${BLK * MT / 4}>;   // [k][row/4]
var<workgroup> ws: array<vec4f, ${BLK * NT / 4}>;   // [k][col/4]
fn loadX(r: u32, k: u32) -> f32 {
  if (p.inMode == 1u) {
    let a = x[r * 2u * p.K + k];
    let g = x[r * 2u * p.K + p.K + k];
    return a / (1.0 + exp(-a)) * g;
  }
  return x[r * p.K + k];
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n0 = wid.x * ${NT}u;
  let r0 = wid.y * ${MT}u;
  let nb = p.K / ${BLK}u;
  let tr = lid.x / 16u;          // rows tr*8 .. tr*8+7
  let tc = lid.x % 16u;          // cols tc*4 .. tc*4+3
  var acc: array<vec4f, 8>;
  for (var i = 0u; i < 8u; i++) { acc[i] = vec4f(0.0); }
  for (var b = 0u; b < nb; b++) {
    // Every shared write is a WHOLE vec4 owned by one thread: a dynamic
    // component store (v[i] = …) can lower to a vec4 read-modify-write, which
    // races when neighbouring threads own other lanes of the same vec4.
    // X tile: 32 k × 16 row-quads; consecutive threads read consecutive k.
    for (var t = 0u; t < 4u; t++) {
      let idx = lid.x + t * 128u;
      let k = idx % ${BLK}u;
      let r4 = idx / ${BLK}u;
      let kg = b * ${BLK}u + k;
      var v = vec4f(0.0);
      for (var q = 0u; q < 4u; q++) {
        let r = r0 + r4 * 4u + q;
        if (r < dims.rows) { v[q] = loadX(r, kg); }
      }
      xs[k * ${MT / 4}u + r4] = v;
    }
    // W tile: 16 column-quads × 4 words (8 nibbles each) → 32 k × 64 cols.
    if (lid.x < 64u) {
      let c4 = lid.x / 4u;
      let j = lid.x % 4u;
      var words: vec4u;
      var s: vec4f;
      for (var q = 0u; q < 4u; q++) {
        let n = n0 + c4 * 4u + q;
        words[q] = wq[(n * nb + b) * 4u + j];
        s[q] = sc[n * nb + b];
      }
      for (var e = 0u; e < 8u; e++) {
        let nib = vec4f((words >> vec4u(e * 4u)) & vec4u(15u)) - vec4f(8.0);
        ws[(j * 8u + e) * ${NT / 4}u + c4] = nib * s;
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < ${BLK}u; k++) {
      let xa = xs[k * ${MT / 4}u + tr * 2u];
      let xb = xs[k * ${MT / 4}u + tr * 2u + 1u];
      let w = ws[k * ${NT / 4}u + tc];
      acc[0] += xa.x * w; acc[1] += xa.y * w; acc[2] += xa.z * w; acc[3] += xa.w * w;
      acc[4] += xb.x * w; acc[5] += xb.y * w; acc[6] += xb.z * w; acc[7] += xb.w * w;
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 8u; i++) {
    let r = r0 + tr * 8u + i;
    if (r < dims.rows) {
      let o = r * p.N + n0 + tc * 4u;
      for (var c = 0u; c < 4u; c++) {
        if (p.accumulate == 1u) { y[o + c] += acc[i][c]; } else { y[o + c] = acc[i][c]; }
      }
    }
  }
}`;

// In-place rotate_half RoPE on the q and k slices of qkv [rows, 3, NH, D].
// cos/sin tables are precomputed on the CPU in f64 → [maxS, D/2] per theta.
const ROPE = DIMS + /* wgsl */ `
struct P { NH: u32, D: u32, tableOff: u32, pad: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> cs: array<vec2f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let hlf = p.D / 2u;
  let perRow = 2u * p.NH * hlf;           // (q|k) × heads × half
  let idx = gid.x;
  if (idx >= dims.rows * perRow) { return; }
  let r = idx / perRow;
  let rem = idx % perRow;
  let which = rem / (p.NH * hlf);         // 0 = q, 1 = k
  let hd = (rem / hlf) % p.NH;
  let i = rem % hlf;
  let pos = r % dims.S;
  let t = cs[p.tableOff + pos * hlf + i];
  let H = p.NH * p.D;
  let base = r * 3u * H + which * H + hd * p.D;
  let a = qkv[base + i];
  let b = qkv[base + i + hlf];
  qkv[base + i] = a * t.x - b * t.y;
  qkv[base + i + hlf] = b * t.x + a * t.y;
}`;

// One workgroup per (query row, head). Scores over the (padding- and
// window-masked) keys live in shared memory; S ≤ MAXS_ATTN.
const MAXS_ATTN = 512;
const ATTN = DIMS + /* wgsl */ `
struct P { NH: u32, D: u32, window: u32, isGlobal: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> lens: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> sc: array<f32, ${MAXS_ATTN}>;
var<workgroup> red: array<f32, 64>;
var<workgroup> q: array<f32, 32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.x;                 // global row = b*S + i
  let hd = wid.y;
  let S = dims.S;
  let b = r / S;
  let i = r % S;
  let len = lens[b];
  let H = p.NH * p.D;
  let scale = inverseSqrt(f32(p.D));
  if (lid.x < p.D) { q[lid.x] = qkv[r * 3u * H + hd * p.D + lid.x] * scale; }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var j = lid.x; j < S; j += 64u) {
    let dist = select(j - i, i - j, i > j);
    let ok = j < len && (p.isGlobal == 1u || dist <= p.window);
    var s = -3.0e38;
    if (ok) {
      let kb = (b * S + j) * 3u * H + H + hd * p.D;
      var d = 0.0;
      for (var e = 0u; e < p.D; e++) { d += q[e] * qkv[kb + e]; }
      s = d;
    }
    sc[j] = s;
    mx = max(mx, s);
  }
  red[lid.x] = mx; workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] = max(red[lid.x], red[lid.x + k]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j < S; j += 64u) {
    let e = select(0.0, exp(sc[j] - m), sc[j] > -1.0e38);
    sc[j] = e; sum += e;
  }
  red[lid.x] = sum; workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  workgroupBarrier();
  // 64 threads = D(32) output dims × 2 key-parity halves, combined via red[].
  let d = lid.x % p.D;
  let par = lid.x / p.D;
  var acc = 0.0;
  // Masked keys have weight 0 but their V can be NaN (padded rows past the
  // window see no valid keys) — skip them so 0 × NaN never reaches acc.
  for (var j = par; j < S; j += 2u) {
    if (sc[j] > 0.0) { acc += sc[j] * qkv[(b * S + j) * 3u * H + 2u * H + hd * p.D + d]; }
  }
  red[lid.x] = acc; workgroupBarrier();
  if (lid.x < p.D) { o[r * H + hd * p.D + lid.x] = (red[lid.x] + red[lid.x + p.D]) * inv; }
}`;

// Tiled attention for indexing-sized inputs (flash-style, online softmax).
// One workgroup per (sequence, block of QB query rows, head); one thread per
// query row holding q and the output accumulator in registers. K/V tiles of KT
// keys are staged in shared memory ONCE per workgroup (vs once per query row in
// ATTN above), and sliding-window layers skip key tiles outside the block's
// ±window span instead of masking all S keys. Shared memory: 2 × KT × 32 f32 =
// 8 KiB, inside the 16 KiB default limit.
const QB = 64, KT = 32;
const ATTN_TILED = DIMS + /* wgsl */ `
struct P { NH: u32, D: u32, window: u32, isGlobal: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> lens: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> kt: array<f32, ${KT * 32}>;
var<workgroup> vt: array<f32, ${KT * 32}>;
@compute @workgroup_size(${QB})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let S = dims.S;
  let nqb = (S + ${QB - 1}u) / ${QB}u;
  let b = wid.x / nqb;
  let q0 = (wid.x % nqb) * ${QB}u;
  let hd = wid.y;
  let H = p.NH * 32u;
  let i = q0 + lid.x;
  let valid = i < S;
  let scale = inverseSqrt(32.0);
  var q: array<f32, 32>;
  var acc: array<f32, 32>;
  for (var e = 0u; e < 32u; e++) {
    acc[e] = 0.0;
    q[e] = select(0.0, qkv[(b * S + min(i, S - 1u)) * 3u * H + hd * 32u + e] * scale, valid);
  }
  var m = -3.0e38;
  var l = 0.0;
  // Workgroup-uniform key span: [0, len) for global layers; for local layers
  // the union of every row's ±window inside this query block.
  let len = min(lens[b], S);
  var jlo = 0u;
  var jhi = len;
  if (p.isGlobal == 0u) {
    jlo = select(0u, q0 - p.window, q0 > p.window);
    jhi = min(len, q0 + ${QB}u + p.window);
  }
  for (var j0 = jlo; j0 < jhi; j0 += ${KT}u) {
    for (var t = lid.x; t < ${KT * 32}u; t += ${QB}u) {
      let j = j0 + t / 32u;
      let base = (b * S + min(j, S - 1u)) * 3u * H + hd * 32u + t % 32u;
      kt[t] = select(0.0, qkv[base + H], j < jhi);
      vt[t] = select(0.0, qkv[base + 2u * H], j < jhi);
    }
    workgroupBarrier();
    if (valid) {
      let kend = min(${KT}u, jhi - j0);
      for (var jj = 0u; jj < kend; jj++) {
        let j = j0 + jj;
        let dist = select(j - i, i - j, i > j);
        if (p.isGlobal == 1u || dist <= p.window) {
          var s = 0.0;
          for (var e = 0u; e < 32u; e++) { s += q[e] * kt[jj * 32u + e]; }
          if (s > m) {
            let c = exp(m - s);
            l *= c;
            for (var e = 0u; e < 32u; e++) { acc[e] *= c; }
            m = s;
          }
          let w = exp(s - m);
          l += w;
          for (var e = 0u; e < 32u; e++) { acc[e] += w * vt[jj * 32u + e]; }
        }
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let inv = select(0.0, 1.0 / l, l > 0.0);
    let ob = (b * S + i) * H + hd * 32u;
    for (var e = 0u; e < 32u; e++) { o[ob + e] = acc[e] * inv; }
  }
}`;

// ── host ──────────────────────────────────────────────────────────────────
export class WgslEncoder {
    // weights: loadGraniteWeights() result. maxRows bounds B*S (activation
    // buffers are allocated once for it — the whole point is a FIXED footprint).
    static async create(weights, config, { device, maxRows = 128, maxS = 128 } = {}) {
        if (!device) {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) throw new Error('WebGPU adapter unavailable');
            device = await adapter.requestDevice();
        }
        const e = new WgslEncoder();
        e.init(device, weights, config, maxRows, maxS);
        return e;
    }

    init(device, weights, config, maxRows, maxS) {
        const H = config.hidden_size, NH = config.num_attention_heads, D = H / NH;
        if (D !== 32) throw new Error('attention kernel assumes head dim 32');
        if (maxS > MAXS_ATTN) throw new Error('maxS > ' + MAXS_ATTN);
        this.device = device; this.config = config; this.weights = weights;
        this.H = H; this.NH = NH; this.D = D; this.maxRows = maxRows; this.maxS = maxS;
        const I2 = weights.layers[0].Wi.N;   // 2 × intermediate
        const S_ = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
        let gpuBytes = 0;
        const buf = (size, usage) => { gpuBytes += size; return device.createBuffer({ size: Math.max(16, size), usage }); };
        const upload = (bytes, usage = S_ | CD) => {
            const b = buf(bytes.byteLength, usage);
            device.queue.writeBuffer(b, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
            return b;
        };
        const uni = (u32s) => {
            const b = buf(16, GPUBufferUsage.UNIFORM | CD);
            device.queue.writeBuffer(b, 0, new Uint32Array(u32s));
            return b;
        };
        const f32bits = (f) => new Uint32Array(new Float32Array([f]).buffer)[0];

        // activations (fixed footprint)
        this.dims = buf(16, GPUBufferUsage.UNIFORM | CD);
        this.x = buf(maxRows * H * 4, S_ | CD | CS);
        this.h = buf(maxRows * H * 4, S_ | CD);
        this.qkv = buf(maxRows * 3 * H * 4, S_);
        this.att = buf(maxRows * H * 4, S_);
        this.u = buf(maxRows * I2 * 4, S_);
        this.lens = buf(4 * maxRows, S_ | CD);
        this.cls = buf(maxRows * H * 4, S_ | CS);
        this.readback = device.createBuffer({ size: maxRows * H * 4, usage: GPUBufferUsage.MAP_READ | CD });
        const activationBytes = gpuBytes;

        // RoPE tables: [global | local], each [maxS, D/2] of (cos, sin), f64 → f32
        const half = D / 2, tbl = new Float32Array(2 * maxS * half * 2);
        [config.global_rope_theta, config.local_rope_theta].forEach((theta, ti) => {
            for (let pos = 0; pos < maxS; pos++) for (let i = 0; i < half; i++) {
                const f = pos / Math.pow(theta, (2 * i) / D);
                const o = ((ti * maxS + pos) * half + i) * 2;
                tbl[o] = Math.cos(f); tbl[o + 1] = Math.sin(f);
            }
        });
        this.rope = upload(new Uint8Array(tbl.buffer));

        // pipelines — one per kernel, compiled once, valid for every S
        const pipe = (code) => device.createComputePipeline({
            layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
        });
        const P = { ln: pipe(LAYERNORM), mm: pipe(MATMUL), rope: pipe(ROPE), attn: pipe(ATTN), attnT: pipe(ATTN_TILED), mmT: pipe(MATMUL_TILED) };
        const bg = (pl, bufs) => device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
        });

        // weights + one pre-built bind group per dispatch → forward() only encodes
        const eps = f32bits(config.norm_eps), win = config.local_attention / 2;
        const ops = [];   // { pipeline, bindGroup, grid: (rows) => [x,y,z] }
        const lnOp = (inBuf, w, outBuf) => ops.push({
            p: P.ln, g: bg(P.ln, [this.dims, uni([H, eps, 1, 0]), inBuf, upload(new Uint8Array(w.buffer)), outBuf]),
            grid: (rows) => [rows, 1, 1],
        });
        const mmOp = (inBuf, W, outBuf, accumulate, inMode = 0) => {
            if (W.bs !== BLK) throw new Error('block size ' + W.bs);
            if (W.N % NT !== 0) throw new Error('N % ' + NT + ' != 0: ' + W.N);
            const bufs = [this.dims, uni([W.N, W.K, accumulate, inMode]), inBuf, upload(W.q), upload(W.scales), outBuf];
            ops.push({
                p: P.mm, g: bg(P.mm, bufs),
                grid: (rows) => [Math.ceil(W.N / WG), Math.ceil(rows / TM), 1],
                // many rows (indexing batches) take the register-tiled kernel
                alt: { use: (rows) => rows >= this.tiledMinRows, p: P.mmT, g: bg(P.mmT, bufs),
                    grid: (rows) => [W.N / NT, Math.ceil(rows / MT), 1] },
            });
        };
        const ropeU = [uni([NH, D, 0, 0]), uni([NH, D, maxS * half, 0])];
        const attnU = [uni([NH, D, win, 1]), uni([NH, D, win, 0])];

        // x ← LN(emb)  (embeddings are written to h, normalized into x)
        lnOp(this.h, weights.embNorm, this.x);
        weights.layers.forEach((Lw, l) => {
            const glob = l % config.global_attn_every_n_layers === 0 ? 1 : 0;
            let attnIn = this.x;
            if (Lw.attnNorm) { lnOp(this.x, Lw.attnNorm, this.h); attnIn = this.h; }
            mmOp(attnIn, Lw.Wqkv, this.qkv, 0);
            ops.push({ p: P.rope, g: bg(P.rope, [this.dims, glob ? ropeU[0] : ropeU[1], this.qkv, this.rope]),
                grid: (rows) => [Math.ceil(rows * NH * D / 64), 1, 1] });
            const attnBufs = [this.dims, glob ? attnU[0] : attnU[1], this.qkv, this.lens, this.att];
            ops.push({ p: P.attn, g: bg(P.attn, attnBufs), grid: (rows) => [rows, NH, 1],
                // long sequences (indexing) take the tiled kernel; queries keep the per-row one
                alt: { use: (rows, S) => S >= this.tiledMinS, p: P.attnT, g: bg(P.attnT, attnBufs), grid: (rows, S, B) => [B * Math.ceil(S / QB), NH, 1] } });
            mmOp(this.att, Lw.Wo, this.x, 1);                 // x += attn · Woᵀ
            lnOp(this.x, Lw.mlpNorm, this.h);
            mmOp(this.h, Lw.Wi, this.u, 0);
            mmOp(this.u, Lw.Wo2, this.x, 1, 1);               // x += GLU(u) · Wo2ᵀ
        });
        // final norm on CLS rows only (rowStride = S, written per forward)
        this.finalU = buf(16, GPUBufferUsage.UNIFORM | CD);
        this.finalOp = { p: P.ln, g: bg(P.ln, [this.dims, this.finalU, this.x, upload(new Uint8Array(weights.finalNorm.buffer)), this.cls]) };
        this.ops = ops;
        this.tiledMinS = 64;      // S at/above which attention uses ATTN_TILED
        this.tiledMinRows = 192;  // B*S at/above which matmuls use MATMUL_TILED
        this.stats = { gpuBytes, activationBytes, weightBytes: gpuBytes - activationBytes, dispatches: ops.length + 1 };
    }

    // seqs: array of token-id arrays (B sequences). Returns B normalized
    // Float32Array(H) CLS vectors. Pads to the longest sequence (no bucket
    // ladder needed: S is a uniform, nothing recompiles per shape).
    async embed(seqs) {
        const { device, H } = this;
        const B = seqs.length, S = Math.max(...seqs.map((s) => s.length));
        const rows = B * S;
        if (S > this.maxS || rows > this.maxRows) throw new Error(`batch ${B}×${S} exceeds engine bounds ${this.maxRows} rows / ${this.maxS} seq`);
        const emb = new Float32Array(rows * H);
        seqs.forEach((ids, b) => emb.set(gatherEmbeddings(this.weights.emb, ids), b * S * H));
        device.queue.writeBuffer(this.h, 0, emb);
        device.queue.writeBuffer(this.dims, 0, new Uint32Array([rows, S, B, 0]));
        device.queue.writeBuffer(this.lens, 0, new Uint32Array(seqs.map((s) => s.length)));
        const eps = new Uint32Array(new Float32Array([this.config.norm_eps]).buffer)[0];
        device.queue.writeBuffer(this.finalU, 0, new Uint32Array([H, eps, S, 0]));

        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (const o of this.ops) {
            const op = o.alt && o.alt.use(rows, S) ? o.alt : o;
            pass.setPipeline(op.p); pass.setBindGroup(0, op.g);
            const [gx, gy, gz] = op.grid(rows, S, B);
            pass.dispatchWorkgroups(gx, gy, gz);
        }
        pass.setPipeline(this.finalOp.p); pass.setBindGroup(0, this.finalOp.g);
        pass.dispatchWorkgroups(B, 1, 1);
        pass.end();
        enc.copyBufferToBuffer(this.cls, 0, this.readback, 0, B * H * 4);
        device.queue.submit([enc.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ, 0, B * H * 4);
        const out = new Float32Array(this.readback.getMappedRange(0, B * H * 4).slice(0));
        this.readback.unmap();
        const vecs = [];
        for (let b = 0; b < B; b++) {
            const v = out.subarray(b * H, (b + 1) * H);
            let n = 0; for (const x of v) n += x * x;
            n = Math.sqrt(n) || 1;
            vecs.push(Float32Array.from(v, (x) => x / n));
        }
        return vecs;
    }

    destroy() { this.device.destroy(); }
}
