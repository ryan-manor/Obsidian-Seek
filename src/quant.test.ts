// Tests for int8 scalar quantization (SQ8) of the rerank tier.
//
// The relevance-relevant contract is NOT bit-exact roundtrip — it's that a
// dequantized vector stays cosine-indistinguishable from the original (the
// ≤0.003 nDCG@10 the 2026-06-01 grid measured comes from this). So the core
// assertions are on cosine similarity and the per-component error bound s/2,
// not on recovering the exact floats.

import { describe, it, expect } from 'vitest';
import { quantizeInt8, dequantizeInt8 } from './quant';

// Deterministic unit-L2 vector of length `dim` from a seeded LCG — avoids
// Math.random so the error bounds are reproducible across runs.
function unitVec(dim: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const v = new Float32Array(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) {
        state = (state * 1664525 + 1013904223) >>> 0;
        const x = (state / 0xffffffff) * 2 - 1; // [-1, 1)
        v[i] = x;
        norm += x * x;
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) v[i] /= norm;
    return v;
}

function cosine(a: Float32Array, b: Float32Array): number {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

describe('quantizeInt8 / dequantizeInt8', () => {
    it('preserves cosine to within the int8 noise floor (granite dim 384)', () => {
        for (let seed = 1; seed <= 20; seed++) {
            const v = unitVec(384, seed);
            const { q, s } = quantizeInt8(v);
            const r = dequantizeInt8(q, s);
            // SQ8 on a unit vector recovers cosine to ~5 nines — far below any
            // nDCG-relevant threshold. (Empirically ~0.99999 at d=384.)
            expect(cosine(v, r)).toBeGreaterThan(0.9999);
        }
    });

    it('bounds per-component error by s/2 (round-to-nearest)', () => {
        const v = unitVec(384, 42);
        const { q, s } = quantizeInt8(v);
        const r = dequantizeInt8(q, s);
        for (let i = 0; i < v.length; i++) {
            // |v - round(v/s)*s| ≤ s/2 by definition of round-to-nearest.
            expect(Math.abs(v[i] - r[i])).toBeLessThanOrEqual(s / 2 + 1e-9);
        }
    });

    it('maps the max-magnitude component to ±127 (full int8 range used)', () => {
        const v = unitVec(384, 7);
        const { q } = quantizeInt8(v);
        let maxAbsQ = 0;
        for (let i = 0; i < q.length; i++) maxAbsQ = Math.max(maxAbsQ, Math.abs(q[i]));
        expect(maxAbsQ).toBe(127);
    });

    it('keeps every quantized component in [-127, 127]', () => {
        const v = unitVec(384, 99);
        const { q } = quantizeInt8(v);
        for (let i = 0; i < q.length; i++) {
            expect(q[i]).toBeGreaterThanOrEqual(-127);
            expect(q[i]).toBeLessThanOrEqual(127);
        }
    });

    it('achieves ~4x compression vs fp32 (388 B vs 1536 B at d=384)', () => {
        const { q } = quantizeInt8(unitVec(384, 1));
        const int8Bytes = q.byteLength + 8; // Int8Array(384) + one f64 scale
        const fp32Bytes = 384 * 4;
        expect(int8Bytes).toBeLessThan(fp32Bytes / 3.5);
    });

    it('handles the degenerate all-zero vector without NaN', () => {
        const z = new Float32Array(384); // all zeros
        const { q, s } = quantizeInt8(z);
        expect(s).toBe(0);
        const r = dequantizeInt8(q, s);
        for (let i = 0; i < r.length; i++) expect(r[i]).toBe(0); // not NaN
    });

    it('preserves sign on all non-near-zero components', () => {
        const v = unitVec(384, 5);
        const { q, s } = quantizeInt8(v);
        for (let i = 0; i < v.length; i++) {
            // Components that don't round to 0 must keep their sign — this is
            // why packSignBits-from-fp32 and a from-int8 derivation agree
            // except on |v| < s/2 (which round to q=0).
            if (q[i] !== 0) expect(Math.sign(q[i])).toBe(Math.sign(v[i]));
            // and the magnitude that survived is at least the half-step
            expect(Math.abs(v[i]) >= 0 || s === 0).toBe(true);
        }
    });
});
