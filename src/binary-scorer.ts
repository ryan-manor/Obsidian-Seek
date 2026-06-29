// Main-thread client for the off-thread stage-1 binary scorer (binary-worker.ts).
//
// Goal: keep the renderer thread free of the O(corpus) sign-bit scan during a
// search, so typing stays smooth as the vault grows. It is a JANK / scaling
// move, not a latency one — the scan is only a few ms today, so the win is that
// the main thread does BM25 + recency CONCURRENTLY while the worker scans (see
// binaryCandidatesAsync), and that the scan never blocks the UI at scale.
//
// It is purely additive: the worker computes the SAME binaryCandidates() the
// synchronous path computes, and ANY failure (no worker, creation error, message
// error, timeout, stale generation) resolves to null so the caller runs the
// identical synchronous fallback. So results are bit-identical whether or not
// the worker runs.
//
// Disabled on mobile: a second JS context + a per-generation copy of the packed
// index shares the iPhone WebContent memory budget with no headroom, and the
// scan is already cheap there — net-negative. The fallback is the shipped
// behaviour, so mobile loses nothing.
import { isMobilePlatform } from './platform';
import { binaryCandidates } from './binary';

// Bundled worker source (esbuild IIFE), injected via define. Empty under
// vitest/tsc — no DOM Worker there, and the worker stays disabled.
declare const __BINARY_WORKER_SRC__: string;
const WORKER_SRC: string =
    typeof __BINARY_WORKER_SRC__ !== 'undefined' ? __BINARY_WORKER_SRC__ : '';

// Defensive ceiling: the scan is ~ms, so this only fires if the worker truly
// hangs, in which case we fall back synchronously for that query.
const WORKER_TIMEOUT_MS = 1000;

// Worker → main reply (see binary-worker.ts): 'result' carries indices, 'stale'
// carries only the queryId. Structured clone hands us `any`, so we narrow.
interface WorkerReply {
    type?: string;
    queryId?: number;
    indices?: number[];
}

export class BinaryScorerWorker {
    private worker: Worker | null = null;
    private dead = false;
    private loadedGen = -1;
    private seq = 0;
    private pending = new Map<number, (idx: number[] | null) => void>();

    constructor() {
        // One line so a live smoke test can tell ACTIVE from (silent) fallback.
        const disabled = isMobilePlatform() ? 'mobile'
            : !WORKER_SRC ? 'no-src(dev/test)'
            : typeof Worker === 'undefined' ? 'no-Worker'
            : null;
        if (disabled) {
            this.dead = true;
            return;
        }
        try {
            const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'application/javascript' }));
            this.worker = new Worker(url);
            URL.revokeObjectURL(url);
            this.worker.onmessage = (e: MessageEvent) => this.onMessage(e.data as WorkerReply);
            this.worker.onerror = () => this.kill();
            this.worker.onmessageerror = () => this.kill();
        } catch {
            // e.g. renderer CSP blocking blob: workers → synchronous fallback.
            this.kill();
        }
    }

    get enabled(): boolean { return !this.dead && !!this.worker; }

    private onMessage(msg: WorkerReply): void {
        if (!msg || typeof msg.queryId !== 'number') return;
        const resolve = this.pending.get(msg.queryId);
        if (!resolve) return;
        this.pending.delete(msg.queryId);
        // 'result' → indices; 'stale'/anything else → null (caller falls back).
        resolve(msg.type === 'result' && Array.isArray(msg.indices) ? msg.indices : null);
    }

    private kill(): void {
        this.dead = true;
        try { this.worker?.terminate(); } catch { /* ignore */ }
        this.worker = null;
        for (const r of this.pending.values()) r(null);
        this.pending.clear();
    }

    // Resolve one query's binary candidate indices off-thread, or null on any
    // failure. NEVER rejects. `packed` is COPIED into the worker once per
    // generation (structured clone — the main-thread buffer is untouched so the
    // synchronous fallback stays available); per query only the small queryVec
    // (+ optional mask) crosses the boundary.
    score(
        generation: number, queryVec: Float32Array, packed: Uint8Array,
        n: number, bytesPerVec: number, topN: number, mask: boolean[] | null,
    ): Promise<number[] | null> {
        if (this.dead || !this.worker) return Promise.resolve(null);
        try {
            if (this.loadedGen !== generation) {
                // COPY, not transfer: the main thread keeps activePacked for the
                // fallback. postMessage without a transfer list structured-clones it.
                this.worker.postMessage({ type: 'load', generation, packed, n, bytesPerVec });
                this.loadedGen = generation;
            }
            const queryId = ++this.seq;
            const maskU8 = mask ? Uint8Array.from(mask, b => (b ? 1 : 0)) : null;
            const p = new Promise<number[] | null>(resolve => {
                this.pending.set(queryId, resolve);
                window.setTimeout(() => {
                    if (this.pending.delete(queryId)) resolve(null);
                }, WORKER_TIMEOUT_MS);
            });
            // queryVec is COPIED (do NOT transfer — it's the embedder's cached
            // vector, reused across queries). maskU8 we just built, so transfer it.
            this.worker.postMessage(
                { type: 'score', queryId, generation, queryVec, topN, mask: maskU8 },
                maskU8 ? [maskU8.buffer] : [],
            );
            return p;
        } catch {
            this.kill();
            return Promise.resolve(null);
        }
    }

    dispose(): void { this.kill(); }
}

// Stage-1 binary candidates via the worker when available, else synchronously.
// Identical result either way (binaryCandidates is the shared function); the
// Promise lets the caller overlap the scan with main-thread BM25/recency.
export function binaryCandidatesAsync(
    worker: BinaryScorerWorker | null,
    generation: number, queryVec: Float32Array, packed: Uint8Array,
    n: number, bytesPerVec: number, topN: number, mask: boolean[] | null,
): Promise<number[]> {
    const sync = (): number[] => binaryCandidates(queryVec, packed, n, bytesPerVec, topN, mask);
    if (!worker || !worker.enabled) return Promise.resolve(sync());
    return worker.score(generation, queryVec, packed, n, bytesPerVec, topN, mask)
        .then(idx => idx ?? sync());
}
