import { describe, it, expect } from 'vitest';
import { packSignBits, concatPacked, scoreAsymmetric, scoreAsymmetricBranchy } from './binary';

const DIM = 384;                 // shipped granite-r2
const BPV = (DIM + 7) >> 3;      // 48 bytes/vec

// Deterministic LCG so the suite is reproducible without Math.random.
function lcg(seed: number) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

// Anisotropic unit vectors: v[d] = amp·sharedDir[d] + noise[d], then L2-norm.
// sharedDir and noise are the SAME [-1,1] scale, so `amp` cleanly controls the
// correlation (mean-pairwise-cos ≈ amp²/(amp²+1)): amp 0 → isotropic (cos 0,
// the branchy predictor's worst case), amp ≈ 1.8 → cos ≈ 0.76, matching granite.
function vecs(n: number, seed: number, amp = 0.0): Float32Array[] {
    const rnd = lcg(seed);
    const shared = new Float32Array(DIM);
    for (let d = 0; d < DIM; d++) shared[d] = rnd() * 2 - 1;   // raw, NOT unit-normed

    const out: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
        const v = new Float32Array(DIM);
        let norm = 0;
        for (let d = 0; d < DIM; d++) {
            const x = amp * shared[d] + (rnd() * 2 - 1);
            v[d] = x; norm += x * x;
        }
        norm = Math.sqrt(norm) || 1;
        for (let d = 0; d < DIM; d++) v[d] /= norm;
        out.push(v);
    }
    return out;
}

function meanPairwiseCos(vs: Float32Array[], cap = 300): number {
    const m = Math.min(vs.length, cap);
    let acc = 0, cnt = 0;
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
        let dot = 0;
        for (let d = 0; d < DIM; d++) dot += vs[i][d] * vs[j][d];
        acc += dot; cnt++;
    }
    return acc / cnt;
}

describe('scoreAsymmetric (LUT) — parity with branchy reference', () => {
    it('produces identical scores and identical top-50 ordering', () => {
        const docs = vecs(800, 7, 0.6);
        const packed = concatPacked(docs.map(packSignBits), BPV);
        for (const qseed of [11, 42, 99]) {
            const q = vecs(1, qseed, 0.6)[0];
            const ref = scoreAsymmetricBranchy(q, packed, docs.length, BPV);
            const lut = scoreAsymmetric(q, packed, docs.length, BPV);
            let maxDiff = 0;
            for (let i = 0; i < ref.length; i++) maxDiff = Math.max(maxDiff, Math.abs(ref[i] - lut[i]));
            expect(maxDiff).toBeLessThan(1e-9);                  // float add-order only
            const top = (s: Float64Array) =>
                Array.from(s.keys()).sort((a, b) => s[b] - s[a]).slice(0, 50);
            expect(top(lut)).toEqual(top(ref));
        }
    });

    it('handles a non-byte-aligned dim (guards the partial final byte)', () => {
        const dim = 380;                                        // 47.5 bytes -> 48, 4 unused high bits
        const bpv = (dim + 7) >> 3;
        const rnd = lcg(5);
        const mk = () => {
            const v = new Float32Array(dim);
            let nrm = 0;
            for (let d = 0; d < dim; d++) { const x = rnd() * 2 - 1; v[d] = x; nrm += x * x; }
            nrm = Math.sqrt(nrm);
            for (let d = 0; d < dim; d++) v[d] /= nrm;
            return v;
        };
        const docs = Array.from({ length: 100 }, mk);
        const packed = concatPacked(docs.map(packSignBits), bpv);
        const q = mk();
        const ref = scoreAsymmetricBranchy(q, packed, docs.length, bpv);
        const lut = scoreAsymmetric(q, packed, docs.length, bpv);
        let maxDiff = 0;
        for (let i = 0; i < ref.length; i++) maxDiff = Math.max(maxDiff, Math.abs(ref[i] - lut[i]));
        expect(maxDiff).toBeLessThan(1e-9);
    });
});

// Heavy benchmark — gated so `npm test` stays fast. Run with:
//   BENCH=1 npx vitest run binary
describe.skipIf(!process.env.BENCH)('scoreAsymmetric benchmark', () => {
    it('before/after at vault scale', () => {
        const QUERIES = 100;
        // amp 0 = isotropic (branchy worst case); amp 1.8 ≈ granite cone (cos 0.76,
        // branchy best case). LUT is branchless so its time is flat across both.
        const REGIMES: [string, number][] = [['isotropic', 0], ['granite-cone', 1.8]];
        for (const N of [4409, 10000]) for (const [label, amp] of REGIMES) {
            const docs = vecs(N, 3, amp);
            const packed = concatPacked(docs.map(packSignBits), BPV);
            const qs = vecs(QUERIES, 5, amp);
            const time = (fn: typeof scoreAsymmetric) => {
                for (let w = 0; w < 8; w++) fn(qs[0], packed, N, BPV);   // warmup / JIT
                const t0 = performance.now();
                for (const q of qs) fn(q, packed, N, BPV);
                return (performance.now() - t0) / QUERIES;
            };
            // alternate to neutralize thermal/JIT drift between the two
            const branchy = time(scoreAsymmetricBranchy);
            const lut = time(scoreAsymmetric);
            const branchy2 = time(scoreAsymmetricBranchy);
            const lut2 = time(scoreAsymmetric);
            const b = (branchy + branchy2) / 2, l = (lut + lut2) / 2;
            // eslint-disable-next-line no-console
            console.log(
                `N=${N} ${label} (cos ${meanPairwiseCos(docs).toFixed(2)}):  ` +
                `branchy ${b.toFixed(2)} ms/q (${(b * 1000 / N).toFixed(3)} µs/doc)  ` +
                `LUT ${l.toFixed(2)} ms/q (${(l * 1000 / N).toFixed(3)} µs/doc)  ` +
                `speedup ${(b / l).toFixed(2)}×`);
            expect(l).toBeLessThan(b);                          // LUT must win
        }
    }, 120_000);
});
