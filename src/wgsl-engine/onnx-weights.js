// Minimal ONNX weight reader for the ml97 GBQ4 export Seek already ships.
//
// Goal: the hand-written WebGPU encoder must consume the EXACT bytes that
// transformers.js caches today (onnx/model_q4.onnx at the pinned revision), so
// there is no second model artifact, no re-export and no mobile-vs-desktop
// split. We only need initializers (weights) and a few node attributes, so this
// is a ~150-line protobuf walker rather than a dependency on onnx/protobufjs.
//
// Zero-copy: every tensor is a Uint8Array VIEW into the model ArrayBuffer. The
// GPU uploader writes those views straight into GPUBuffers, and the token
// embedding table (34.5 MB int4 + 8.6 MB scales) is never copied at all: we
// dequantize only the rows a query actually uses, on the CPU.

// ── protobuf wire-format walker ────────────────────────────────────────────
function readVarint(buf, pos) {
    let result = 0, shift = 0, b;
    do {
        b = buf[pos++];
        // Values here (lengths, small ints) fit in 2^53; avoid bitwise overflow.
        result += (b & 0x7f) * 2 ** shift;
        shift += 7;
    } while (b & 0x80);
    return [result, pos];
}

// Calls fn(fieldNo, wireType, value, start, end) for each field in buf[start,end).
// value: number for varint/fixed, [s,e] byte range for length-delimited.
function walk(buf, start, end, fn) {
    let pos = start;
    while (pos < end) {
        let key; [key, pos] = readVarint(buf, pos);
        const field = Math.floor(key / 8), wt = key & 7;
        if (wt === 0) { let v; [v, pos] = readVarint(buf, pos); fn(field, wt, v); }
        else if (wt === 1) { fn(field, wt, null); pos += 8; }
        else if (wt === 5) { fn(field, wt, null); pos += 4; }
        else if (wt === 2) { let len; [len, pos] = readVarint(buf, pos); fn(field, wt, [pos, pos + len]); pos += len; }
        else throw new Error('unsupported wire type ' + wt + ' at ' + pos);
    }
}

const utf8 = new TextDecoder();
const str = (buf, [s, e]) => utf8.decode(buf.subarray(s, e));

function parseTensor(buf, range) {
    const t = { name: '', dims: [], dataType: 0, raw: null };
    walk(buf, range[0], range[1], (f, wt, v) => {
        if (f === 1) {
            if (wt === 0) t.dims.push(v);
            else walk(buf, v[0], v[1], (_f, _wt, d) => t.dims.push(d)); // packed
        } else if (f === 2) t.dataType = v;
        else if (f === 8) t.name = str(buf, v);
        else if (f === 9) t.raw = buf.subarray(v[0], v[1]);
    });
    if (!t.raw) throw new Error('tensor ' + t.name + ' has no raw_data (external data unsupported)');
    return t;
}

function parseNode(buf, range) {
    const n = { name: '', op: '', inputs: [], attrs: {} };
    walk(buf, range[0], range[1], (f, wt, v) => {
        if (f === 1) n.inputs.push(str(buf, v));
        else if (f === 3) n.name = str(buf, v);
        else if (f === 4) n.op = str(buf, v);
        else if (f === 5) {
            let an = '', ai = null;
            walk(buf, v[0], v[1], (af, awt, av) => {
                if (af === 1) an = str(buf, av);
                else if (af === 3) ai = av; // int attribute
            });
            if (ai !== null) n.attrs[an] = ai;
        }
    });
    return n;
}

function parseModel(bytes) {
    const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    let graph = null;
    walk(buf, 0, buf.length, (f, wt, v) => { if (f === 7) graph = v; });
    if (!graph) throw new Error('no graph in model');
    const inits = new Map(), nodes = [];
    walk(buf, graph[0], graph[1], (f, wt, v) => {
        if (f === 1) nodes.push(parseNode(buf, v));
        else if (f === 5) { const t = parseTensor(buf, v); inits.set(t.name, t); }
    });
    return { inits, nodes };
}

// Small tensors (norm weights) are copied into an aligned Float32Array —
// raw_data offsets inside the protobuf are not 4-byte aligned.
function f32(t) {
    const out = new Float32Array(t.raw.byteLength / 4);
    new Uint8Array(out.buffer).set(t.raw);
    return out;
}

// ── model-specific extraction ──────────────────────────────────────────────
// Returns:
//   emb:    { q: Uint8Array (int4, 2/byte, low nibble first, signed), scales: Uint8Array (f32 LE), vocab, hidden, blockSize }
//   embNorm, finalNorm: Float32Array(hidden)
//   layers[l]: { attnNorm (null for layer 0), mlpNorm, Wqkv, Wo, Wi, Wo2 }
//     where each W = { q: Uint8Array (N, K/bs, bs/2) u4 zp=8, scales: Uint8Array (N, K/bs) f32, N, K, bs }
export function loadGraniteWeights(bytes, config) {
    const { inits, nodes } = parseModel(bytes);
    const get = (name) => {
        const t = inits.get(name);
        if (!t) throw new Error('missing initializer ' + name);
        return t;
    };
    const L = config.num_hidden_layers, H = config.hidden_size;
    const layers = Array.from({ length: L }, () => ({}));
    let emb = null;
    for (const n of nodes) {
        if (n.op === 'MatMulNBits') {
            const m = /layers\.(\d+)\/(attn|mlp)\/(Wqkv|Wo|Wi)/.exec(n.name);
            if (!m) throw new Error('unrecognized MatMulNBits ' + n.name);
            if (n.inputs.length > 3 && n.inputs[3]) throw new Error('zero points not supported');
            if (n.attrs.bits !== 4) throw new Error('only 4-bit supported');
            const key = m[2] === 'mlp' && m[3] === 'Wo' ? 'Wo2' : m[3];
            layers[+m[1]][key] = {
                q: get(n.inputs[1]).raw, scales: get(n.inputs[2]).raw,
                N: n.attrs.N, K: n.attrs.K, bs: n.attrs.block_size,
            };
        } else if (n.op === 'GatherBlockQuantized') {
            const q = get(n.inputs[0]), s = get(n.inputs[2]);
            if (q.dataType !== 22) throw new Error('embedding table must be INT4');
            if (n.attrs.quantize_axis !== 1 || n.attrs.gather_axis !== 0) throw new Error('unexpected GBQ axes');
            emb = { q: q.raw, scales: s.raw, vocab: q.dims[0], hidden: q.dims[1], blockSize: n.attrs.block_size };
        }
    }
    if (!emb || emb.hidden !== H) throw new Error('embedding table not found / wrong width');
    for (let l = 0; l < L; l++) {
        const p = '0.auto_model.layers.' + l + '.';
        layers[l].attnNorm = l === 0 ? null : f32(get(p + 'attn_norm.weight'));
        layers[l].mlpNorm = f32(get(p + 'mlp_norm.weight'));
        for (const k of ['Wqkv', 'Wo', 'Wi', 'Wo2']) if (!layers[l][k]) throw new Error('layer ' + l + ' missing ' + k);
    }
    return {
        emb,
        embNorm: f32(get('0.auto_model.embeddings.norm.weight')),
        finalNorm: f32(get('0.auto_model.final_norm.weight')),
        layers,
    };
}

// Copy the embedding table out of the model buffer (43 MB) so the full 61 MB
// model ArrayBuffer can be released once the layer weights are on the GPU.
export function detachEmbedding(emb) {
    return { ...emb, q: emb.q.slice(), scales: emb.scales.slice() };
}

// Dequantize the embedding rows for `ids` into a Float32Array(S*hidden).
// CPU-side by design: a query touches ~3-30 rows of a 180k-row table, so
// uploading 43 MB to the GPU to gather from it would be pure waste.
export function gatherEmbeddings(emb, ids) {
    const { q, scales, hidden, blockSize } = emb;
    const sv = new DataView(scales.buffer, scales.byteOffset, scales.byteLength);
    const nBlocks = hidden / blockSize;
    const out = new Float32Array(ids.length * hidden);
    for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        const rowByte = (id * hidden) / 2;
        for (let b = 0; b < nBlocks; b++) {
            const s = sv.getFloat32((id * nBlocks + b) * 4, true);
            for (let j = 0; j < blockSize; j++) {
                const e = b * blockSize + j;
                const byte = q[rowByte + (e >> 1)];
                let v = (e & 1) ? (byte >> 4) : (byte & 0x0f);
                if (v >= 8) v -= 16; // signed int4
                out[i * hidden + e] = v * s;
            }
        }
    }
    return out;
}
