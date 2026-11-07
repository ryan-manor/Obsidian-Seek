// Runs INSIDE a dedicated srcdoc iframe (bundled to an IIFE string by
// esbuild.config.mjs → __WGSL_BENCH_SRC__). Same no-sandbox trick as the
// production runtime iframe: it inherits Obsidian's origin, so it shares the
// transformers.js Cache API partition (no re-download of model_q4.onnx) and can
// reach the HF CDN with CORS if the cache misses.
//
// Protocol: parent posts {type:'wgsl-bench-run', ...}; child replies
// {type:'wgsl-bench-result', result} (or {error}). One shot, then the parent
// removes the iframe — which destroys the GPUDevice and frees everything.

import { loadGraniteWeights, detachEmbedding } from './onnx-weights.js';
import { WgslEncoder } from './engine-webgpu.js';

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

async function fetchModelFile(repo, revision, file, cdnUrl) {
    const url = 'https://huggingface.co/' + repo + '/resolve/' + (revision || 'main') + '/' + file;
    const t0 = performance.now();
    let source = 'network';
    let resp = null;
    try {
        const cache = await caches.open('transformers-cache');
        resp = await cache.match(url);
        if (resp) source = 'cache';
    } catch (_) { /* no Cache API — fall through */ }
    if (!resp) resp = await fetch(url);
    if (!resp.ok) throw new Error('fetch ' + file + ' → HTTP ' + resp.status);
    const bytes = new Uint8Array(await resp.arrayBuffer());
    return { bytes, source, ms: performance.now() - t0, url };
}

async function run(msg) {
    const { repo, revision, cdnUrl, queries, reps } = msg;
    const out = { ua: navigator.userAgent, steps: {} };
    const T = (k, t0) => { out.steps[k] = Math.round(performance.now() - t0); };
    if (!navigator.gpu) throw new Error('navigator.gpu not present in this WebView');
    let t0 = performance.now();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('requestAdapter returned null');
    const info = adapter.info || {};
    out.adapter = { vendor: info.vendor || '', architecture: info.architecture || '', device: info.device || '', description: info.description || '' };
    const device = await adapter.requestDevice();
    let lost = null;
    device.lost.then((l) => { lost = l.reason + ': ' + l.message; });
    T('adapterAndDevice', t0);

    t0 = performance.now();
    const config = JSON.parse(new TextDecoder().decode((await fetchModelFile(repo, revision, 'config.json')).bytes));
    const model = await fetchModelFile(repo, revision, 'onnx/model_q4.onnx');
    out.modelSource = model.source;
    out.modelBytes = model.bytes.byteLength;
    T('fetchModel', t0);

    t0 = performance.now();
    const tx = await import(cdnUrl);   // tokenizer only — no ORT session is created
    tx.env.allowLocalModels = false;
    const tokenizer = await tx.AutoTokenizer.from_pretrained(repo, revision ? { revision } : {});
    T('tokenizer', t0);

    t0 = performance.now();
    let weights = loadGraniteWeights(model.bytes, config);
    T('parse', t0);

    t0 = performance.now();
    const enc = await WgslEncoder.create(weights, config, { device, maxRows: 128, maxS: 128 });
    T('engineInit', t0);
    // After the GPU upload only the embedding table is needed CPU-side; copy it
    // out so the 61 MB model ArrayBuffer can be garbage-collected.
    enc.weights = { emb: detachEmbedding(weights.emb) };
    weights = null;
    out.engine = enc.stats;

    const ids = queries.map((q) => tokenizer.encode(q));
    t0 = performance.now();
    await enc.embed([ids[0]]);
    T('firstEmbed', t0);   // includes any lazy pipeline/driver work

    out.queries = [];
    for (let i = 0; i < queries.length; i++) {
        const times = [];
        let vec = null;
        for (let r = 0; r < reps; r++) {
            const t = performance.now();
            const [v] = await enc.embed([ids[i]]);
            times.push(performance.now() - t);
            vec = v;
        }
        out.queries.push({ text: queries[i], tokens: ids[i].length, medianMs: median(times), minMs: Math.min(...times), vector: Array.from(vec) });
    }
    // Batched throughput probe (indexing-shaped): 8 × the longest query.
    const longest = ids.reduce((a, b) => (b.length > a.length ? b : a));
    const batch = Array(8).fill(longest);
    const bt = [];
    for (let r = 0; r < 3; r++) { const t = performance.now(); await enc.embed(batch); bt.push(performance.now() - t); }
    out.batch8 = { tokens: longest.length, medianMs: median(bt) };
    out.deviceLost = lost;
    enc.destroy();
    return out;
}

window.addEventListener('message', (ev) => {
    if (ev.source !== window.parent || !ev.data || ev.data.type !== 'wgsl-bench-run') return;
    run(ev.data).then(
        (result) => window.parent.postMessage({ type: 'wgsl-bench-result', result }, '*'),
        (e) => window.parent.postMessage({ type: 'wgsl-bench-result', error: String(e && e.stack || e) }, '*'),
    );
});
window.parent.postMessage({ type: 'wgsl-bench-ready' }, '*');
