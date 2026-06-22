// Off-main-thread stage-1 binary candidate generation.
//
// Bundled by esbuild to a standalone IIFE string and instantiated from a Blob
// URL by binary-scorer.ts on the main thread (desktop only — see the client for
// the mobile-memory rationale). It holds the resident sign-bit index
// (activePacked) for the current dataGeneration, and per query runs the SAME
// binaryCandidates() the synchronous fallback runs — so its output is
// bit-identical: a different THREAD, never a different ranking.
//
// This bundle pulls in only binary.ts + select.ts (pure compute, no DOM, no
// obsidian), so it is tiny and safe to run in a worker.
//
// Protocol (main → worker):
//   { type: 'load',  generation, packed: Uint8Array, n, bytesPerVec }
//   { type: 'score', queryId, generation, queryVec: Float32Array, topN, mask: Uint8Array|null }
// Worker → main:
//   { type: 'result', queryId, indices: number[] }
//   { type: 'stale',  queryId }   // worker lacks this generation's index → caller falls back
import { binaryCandidates } from './binary';

// Minimal view of the worker global, avoiding a dependency on the WebWorker lib
// (the project compiles against the DOM lib).
const ctx = self as unknown as {
    onmessage: ((e: MessageEvent) => void) | null;
    postMessage: (message: unknown) => void;
};

let packed: Uint8Array | null = null;
let loadedGen = -1;
let n = 0;
let bytesPerVec = 0;

ctx.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (!msg) return;
    if (msg.type === 'load') {
        packed = msg.packed as Uint8Array;
        loadedGen = msg.generation as number;
        n = msg.n as number;
        bytesPerVec = msg.bytesPerVec as number;
        return;
    }
    if (msg.type === 'score') {
        const queryId = msg.queryId as number;
        // The index must match the query's generation — otherwise candidate
        // indices would point into a stale frame. Tell the caller to fall back.
        if (!packed || loadedGen !== msg.generation) {
            ctx.postMessage({ type: 'stale', queryId });
            return;
        }
        const maskU8 = msg.mask as Uint8Array | null;
        const mask = maskU8 ? Array.from(maskU8, (b: number) => b === 1) : null;
        const indices = binaryCandidates(
            msg.queryVec as Float32Array, packed, n, bytesPerVec, msg.topN as number, mask,
        );
        ctx.postMessage({ type: 'result', queryId, indices });
    }
};
