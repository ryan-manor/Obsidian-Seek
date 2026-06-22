import { describe, it, expect } from 'vitest';
import { packSignBits, concatPacked, scoreAsymmetric, topNIndices, binaryCandidates } from './binary';
import { BinaryScorerWorker, binaryCandidatesAsync } from './binary-scorer';

// The Worker transport (Blob URL, postMessage) can't run in node-vitest, so it
// gets a live in-app smoke test. What IS unit-tested here is the worker's COMPUTE
// — the shared binaryCandidates() it runs — and the exact mask round-trip that
// crosses the boundary, so a divergence between worker and fallback is caught.

const DIM = 384;
const BPV = (DIM + 7) >> 3;

function lcg(seed: number) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

function vec(rnd: () => number): Float32Array {
    const v = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = rnd() * 2 - 1;
    return v;
}

function buildPacked(n: number, seed: number): Uint8Array {
    const rnd = lcg(seed);
    const bufs: Uint8Array[] = [];
    for (let i = 0; i < n; i++) bufs.push(packSignBits(vec(rnd)));
    return concatPacked(bufs, BPV);
}

describe('binaryCandidates — the function shared by the worker and the synchronous fallback', () => {
    it('equals scoreAsymmetric + topNIndices (no mask)', () => {
        const n = 500;
        const packed = buildPacked(n, 1);
        const q = vec(lcg(2));
        for (const topN of [50, 200, n]) {
            const direct = topNIndices(scoreAsymmetric(q, packed, n, BPV), topN, null);
            expect(binaryCandidates(q, packed, n, BPV, topN, null)).toEqual(direct);
        }
    });

    it('survives the worker mask round-trip (boolean[] → Uint8Array → boolean[])', () => {
        const n = 400;
        const packed = buildPacked(n, 3);
        const q = vec(lcg(4));
        const rnd = lcg(5);
        const mask = Array.from({ length: n }, () => rnd() < 0.5);
        // Exactly what crosses the boundary: client packs to Uint8Array, worker unpacks.
        const maskU8 = Uint8Array.from(mask, b => (b ? 1 : 0));
        const reconstructed = Array.from(maskU8, b => b === 1);
        for (const topN of [50, 200]) {
            expect(binaryCandidates(q, packed, n, BPV, topN, reconstructed))
                .toEqual(binaryCandidates(q, packed, n, BPV, topN, mask));
        }
    });
});

describe('binaryCandidatesAsync — fallback resolves the identical synchronous result', () => {
    it('with a null worker', async () => {
        const n = 300;
        const packed = buildPacked(n, 7);
        const q = vec(lcg(8));
        const got = await binaryCandidatesAsync(null, 1, q, packed, n, BPV, 200, null);
        expect(got).toEqual(binaryCandidates(q, packed, n, BPV, 200, null));
    });

    it('with a disabled worker (no DOM Worker / empty src in vitest)', async () => {
        const w = new BinaryScorerWorker();
        expect(w.enabled).toBe(false); // constructor self-disables when WORKER_SRC is empty
        const n = 300;
        const packed = buildPacked(n, 9);
        const q = vec(lcg(10));
        const rnd = lcg(11);
        const mask = Array.from({ length: n }, () => rnd() < 0.6);
        const got = await binaryCandidatesAsync(w, 1, q, packed, n, BPV, 100, mask);
        expect(got).toEqual(binaryCandidates(q, packed, n, BPV, 100, mask));
        w.dispose();
    });
});
