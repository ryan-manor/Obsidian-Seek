// EXPERIMENTAL on-device benchmark for the hand-written WebGPU encoder
// (research/wgsl-encoder). Investigation branch only — not a product path.
//
// What it answers, on the device it runs on (aimed at iPhone):
//   1. Does a WKWebView WebGPU device come up at all without ORT-Web's wasm glue?
//   2. Cold start of the WGSL engine (parse the cached model_q4.onnx + upload +
//      compile 4 pipelines) and warm per-query latency.
//   3. Do its vectors agree with the production pipeline's (cosine)?
//   4. Production query latency on the same device, for the side-by-side.
//
// Sequencing is deliberate for the iPhone memory ceiling: the WGSL engine runs
// in its OWN srcdoc iframe first and that iframe is removed (GPUDevice + heap
// gone) BEFORE the production model is touched, so the two never co-reside.
// Results are written to a note at the vault root so they can be read on-phone.

import { normalizePath, type App } from 'obsidian';
import type { LocalEmbedder } from './embedder';
import { ACTIVE_MODEL_SPEC } from './model-registry';
import { TRANSFORMERS_VERSION } from './iframe-runner';

declare const __WGSL_BENCH_SRC__: string;

export const WGSL_BENCH_QUERIES = [
    'street photography',
    'Q3 roadmap',
    'wie konfiguriere ich obsidian sync',
    'motorcycle maintenance checklist before a long ride',
    'how do I tune BM25 field weights for title vs body',
    '東京の写真散歩ルート',
    'notes from the search relevance review with the ranking team about recency boosts and synonym expansion',
];
const REPS = 7;
const TIMEOUT_MS = 240_000;

interface WgslQueryResult { text: string; tokens: number; medianMs: number; minMs: number; vector: number[] }
interface WgslBenchResult {
    ua: string;
    adapter: Record<string, string>;
    modelSource: string;
    modelBytes: number;
    steps: Record<string, number>;
    engine: { gpuBytes: number; weightBytes: number; activationBytes: number; dispatches: number };
    queries: WgslQueryResult[];
    batch8: { tokens: number; medianMs: number };
    deviceLost: string | null;
}

function runInIframe(): Promise<WgslBenchResult> {
    return new Promise((resolve, reject) => {
        const iframe = window.document.createElement('iframe');
        iframe.addClass('seek-hidden');
        let done = false;
        const finish = (fn: () => void) => {
            if (done) return;
            done = true;
            window.removeEventListener('message', onMsg);
            window.clearTimeout(timer);
            iframe.remove();   // destroys the GPUDevice + the iframe's heap
            fn();
        };
        const onMsg = (ev: MessageEvent) => {
            if (ev.source !== iframe.contentWindow) return;
            const d = ev.data as { type?: string; result?: WgslBenchResult; error?: string };
            if (d?.type === 'wgsl-bench-ready') {
                iframe.contentWindow?.postMessage({
                    type: 'wgsl-bench-run',
                    repo: ACTIVE_MODEL_SPEC.repo,
                    revision: ACTIVE_MODEL_SPEC.revision,
                    cdnUrl: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}`,
                    queries: WGSL_BENCH_QUERIES,
                    reps: REPS,
                }, '*');
            } else if (d?.type === 'wgsl-bench-result') {
                if (d.error) finish(() => reject(new Error(d.error)));
                else finish(() => resolve(d.result as WgslBenchResult));
            }
        };
        const timer = window.setTimeout(() => finish(() => reject(new Error('WGSL bench timed out'))), TIMEOUT_MS);
        window.addEventListener('message', onMsg);
        // No sandbox attribute — same reasoning as the runtime iframe
        // (iframe-runner.ts): inherit the app origin → shared Cache API + CORS.
        window.document.body.appendChild(iframe);
        iframe.srcdoc = `<!DOCTYPE html><html><body><script type="module">${__WGSL_BENCH_SRC__}</script></body></html>`;
    });
}

const cos = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / Math.sqrt(na * nb);
};
const median = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

export async function runWgslBench(
    app: App,
    embedder: LocalEmbedder,
    ensureModelLoaded: () => Promise<void>,
    unloadModel: () => void,
): Promise<string> {
    const lines: string[] = [];
    const stamp = new Date().toISOString();
    lines.push(`# Seek WGSL encoder bench`, '', `- run: ${stamp}`, `- model: ${ACTIVE_MODEL_SPEC.repo}@${ACTIVE_MODEL_SPEC.revision}`, '');

    // Phase 1: WGSL engine alone (production model unloaded so they never co-reside).
    const wasLoaded = embedder.loaded;
    if (wasLoaded) unloadModel();
    let wgsl: WgslBenchResult | null = null;
    let wgslError: string | null = null;
    const t0 = performance.now();
    try { wgsl = await runInIframe(); }
    catch (e) { wgslError = String(e); }
    const wgslWall = performance.now() - t0;

    // Phase 2: production pipeline, same queries. Raw runner calls bypass the
    // query LRU so every rep is a real forward.
    const prod: { medianMs: number; vector: Float32Array }[] = [];
    let prodError: string | null = null;
    try {
        await ensureModelLoaded();
        const runner = (embedder as unknown as { runner: { embed(t: string): Promise<{ vector: Float32Array; latencyMs: number }> } }).runner;
        for (const q of WGSL_BENCH_QUERIES) {
            const times: number[] = [];
            let vec: Float32Array | null = null;
            for (let r = 0; r < REPS; r++) {
                const res = await runner.embed(q);
                times.push(res.latencyMs);
                vec = res.vector;
            }
            prod.push({ medianMs: median(times), vector: vec as Float32Array });
        }
    } catch (e) { prodError = String(e); }

    lines.push('## WGSL engine (hand-written WebGPU)', '');
    if (!wgsl) {
        lines.push(`**Failed:** \`${wgslError}\``, '');
    } else {
        lines.push(
            `- adapter: ${JSON.stringify(wgsl.adapter)}`,
            `- model bytes from: **${wgsl.modelSource}** (${(wgsl.modelBytes / 1e6).toFixed(1)} MB)`,
            `- GPU memory allocated: **${(wgsl.engine.gpuBytes / 1e6).toFixed(1)} MB** (weights ${(wgsl.engine.weightBytes / 1e6).toFixed(1)}, activations ${(wgsl.engine.activationBytes / 1e6).toFixed(1)}), ${wgsl.engine.dispatches} dispatches/forward`,
            `- cold-start steps (ms): ${Object.entries(wgsl.steps).map(([k, v]) => `${k}=${v}`).join(', ')}`,
            `- total iframe wall: ${wgslWall.toFixed(0)} ms`,
            `- batch 8 × ${wgsl.batch8.tokens} tok: ${wgsl.batch8.medianMs.toFixed(1)} ms (${(8000 / wgsl.batch8.medianMs).toFixed(1)} seq/s)`,
            `- device lost: ${wgsl.deviceLost ?? 'no'}`,
            '',
        );
    }
    lines.push('## Production pipeline', '');
    lines.push(prodError
        ? `**Failed:** \`${prodError}\``
        : `- device: **${embedder.device}** dtype ${embedder.dtype} glue ${embedder.glue ?? 'n/a'}`, '');

    lines.push('## Per query (median of ' + REPS + ')', '', '| query | tokens | WGSL ms | production ms | cosine |', '|---|---|---|---|---|');
    WGSL_BENCH_QUERIES.forEach((q, i) => {
        const w = wgsl?.queries[i];
        const p = prod[i];
        lines.push(`| ${q.slice(0, 40)} | ${w?.tokens ?? '?'} | ${w ? w.medianMs.toFixed(1) : '–'} | ${p ? p.medianMs.toFixed(1) : '–'} | ${w && p ? cos(w.vector, p.vector).toFixed(6) : '–'} |`);
    });
    lines.push('', `UA: \`${wgsl?.ua ?? navigator.userAgent}\``, '');

    const path = normalizePath(`Seek WGSL bench ${stamp.replace(/[:.]/g, '-')}.md`);
    await app.vault.create(path, lines.join('\n'));
    return path;
}
