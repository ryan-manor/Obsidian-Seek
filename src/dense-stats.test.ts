// Tests for the corpus dense-cosine background statistics (calibration/gate).
//
// The contract is: (1) bgMeanFromSum reproduces the TRUE mean off-diagonal
// cosine without enumerating pairs (it's a closed form, so it must MATCH brute
// force, not merely approximate it); (2) bgStdSampled converges to the brute-
// force std given enough pairs and is deterministic under a seeded rng;
// (3) denseBgStats gates on MIN_BG_SAMPLE; (4) VecReservoir is uniform, capped,
// and copy-isolating so the sample can't pin a shared buffer or be mutated.

import { describe, it, expect } from 'vitest';
import {
    bgMeanFromSum,
    bgStdSampled,
    denseBgStats,
    calibratedConfidence,
    VecReservoir,
    matchStrength,
    MIN_BG_SAMPLE,
    CONF_Z0,
    CONF_ZSCALE,
} from './dense-stats';

// Seeded LCG in [0,1) — same generator family as quant.test, so reservoir and
// pair sampling are reproducible across runs (no Math.random in assertions).
function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
}

function unitVec(dim: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const v = new Float32Array(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) {
        state = (state * 1664525 + 1013904223) >>> 0;
        const x = (state / 0xffffffff) * 2 - 1;
        v[i] = x; norm += x * x;
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) v[i] /= norm;
    return v;
}

// An ANISOTROPIC corpus: each vector is a blend of a shared anchor and noise,
// so off-diagonal cosines bunch around a high mean with real spread — the
// regime these stats exist to characterize (mirrors granite-r2's cone).
function anisoCorpus(n: number, dim: number, seed: number, anchorMix = 0.6): Float32Array[] {
    const anchor = unitVec(dim, seed * 7 + 1);
    const out: Float32Array[] = [];
    for (let k = 0; k < n; k++) {
        const noise = unitVec(dim, seed * 100003 + k);
        const v = new Float32Array(dim);
        let norm = 0;
        for (let d = 0; d < dim; d++) {
            v[d] = anchorMix * anchor[d] + (1 - anchorMix) * noise[d];
            norm += v[d] * v[d];
        }
        norm = Math.sqrt(norm);
        for (let d = 0; d < dim; d++) v[d] /= norm;
        out.push(v);
    }
    return out;
}

function sumOf(vecs: Float32Array[]): Float64Array {
    const s = new Float64Array(vecs[0].length);
    for (const v of vecs) for (let d = 0; d < v.length; d++) s[d] += v[d];
    return s;
}

// Exhaustive off-diagonal mean/std — the ground truth the samplers approximate.
function bruteForce(vecs: Float32Array[]): { mean: number; std: number } {
    const n = vecs.length;
    let s = 0, s2 = 0, m = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            let dot = 0;
            for (let d = 0; d < vecs[i].length; d++) dot += vecs[i][d] * vecs[j][d];
            s += dot; s2 += dot * dot; m++;
        }
    }
    const mean = s / m;
    return { mean, std: Math.sqrt(Math.max(0, s2 / m - mean * mean)) };
}

describe('bgMeanFromSum (closed-form μ)', () => {
    it('equals the brute-force mean off-diagonal cosine', () => {
        const vecs = anisoCorpus(200, 24, 42);
        const got = bgMeanFromSum(sumOf(vecs), vecs.length);
        expect(got).toBeCloseTo(bruteForce(vecs).mean, 4);
    });

    it('= 1 for identical vectors (degenerate cone)', () => {
        const v = unitVec(16, 9);
        const vecs = [v, new Float32Array(v), new Float32Array(v)];
        expect(bgMeanFromSum(sumOf(vecs), 3)).toBeCloseTo(1, 6);
    });

    it('= 0 for an orthogonal pair (isotropic floor)', () => {
        const e1 = new Float32Array([1, 0]);
        const e2 = new Float32Array([0, 1]);
        expect(bgMeanFromSum(sumOf([e1, e2]), 2)).toBeCloseTo(0, 6);
    });

    it('returns 0 for n < 2 (no pairs exist)', () => {
        expect(bgMeanFromSum(new Float64Array(4), 1)).toBe(0);
        expect(bgMeanFromSum(new Float64Array(4), 0)).toBe(0);
    });
});

describe('bgStdSampled (sampled σ)', () => {
    it('converges to the brute-force std with enough pairs', () => {
        const vecs = anisoCorpus(150, 24, 7);
        const ref = bruteForce(vecs).std;
        const got = bgStdSampled(vecs, 50_000, lcg(123));
        expect(got).toBeCloseTo(ref, 2);   // within ~0.005
    });

    it('is deterministic under a seeded rng', () => {
        const vecs = anisoCorpus(80, 16, 3);
        expect(bgStdSampled(vecs, 4000, lcg(99))).toBe(bgStdSampled(vecs, 4000, lcg(99)));
    });

    it('returns 0 for n < 2', () => {
        expect(bgStdSampled([unitVec(8, 1)], 100, lcg(1))).toBe(0);
    });
});

describe('denseBgStats (index-time gate)', () => {
    it('returns null below MIN_BG_SAMPLE — calibration stays off', () => {
        const vecs = anisoCorpus(MIN_BG_SAMPLE - 1, 12, 5);
        expect(denseBgStats(sumOf(vecs), vecs.length, vecs, 2000, lcg(5))).toBeNull();
    });

    it('returns {mean,std} at/above MIN_BG_SAMPLE', () => {
        const vecs = anisoCorpus(MIN_BG_SAMPLE + 50, 12, 5);
        const stats = denseBgStats(sumOf(vecs), vecs.length, vecs, 4000, lcg(5));
        expect(stats).not.toBeNull();
        expect(stats!.mean).toBeCloseTo(bruteForce(vecs).mean, 3);
        expect(stats!.std).toBeGreaterThan(0);
    });
});

describe('calibratedConfidence (display-only)', () => {
    const mean = 0.76, std = 0.04;   // granite-r2-like background

    it('maps z = CONF_Z0 to exactly 50%', () => {
        const cosAtZ0 = mean + CONF_Z0 * std;
        expect(calibratedConfidence(cosAtZ0, mean, std)).toBeCloseTo(0.5, 6);
    });

    it('is monotonic increasing in the raw cosine', () => {
        let prev = -1;
        for (let cos = mean - 2 * std; cos <= mean + 8 * std; cos += std / 2) {
            const c = calibratedConfidence(cos, mean, std);
            expect(c).toBeGreaterThan(prev);
            prev = c;
        }
    });

    it('stays in (0,1) across an extreme cosine range', () => {
        for (const cos of [-1, 0, mean, 0.95, 1]) {
            const c = calibratedConfidence(cos, mean, std);
            expect(c).toBeGreaterThan(0);
            expect(c).toBeLessThan(1);
        }
    });

    it('puts ~75%/25% near ±1σ of CONF_ZSCALE around z0 (softness sanity)', () => {
        const hi = calibratedConfidence(mean + (CONF_Z0 + CONF_ZSCALE) * std, mean, std);
        const lo = calibratedConfidence(mean + (CONF_Z0 - CONF_ZSCALE) * std, mean, std);
        expect(hi).toBeCloseTo(1 / (1 + Math.exp(-1)), 6);   // ≈0.731
        expect(lo).toBeCloseTo(1 / (1 + Math.exp(1)), 6);    // ≈0.269
    });

    it('returns 0 when std is non-positive (no usable background)', () => {
        expect(calibratedConfidence(0.9, 0.76, 0)).toBe(0);
        expect(calibratedConfidence(0.9, 0.76, -1)).toBe(0);
    });
});

describe('VecReservoir', () => {
    it('keeps every vector when the stream fits the cap, as independent copies', () => {
        const r = new VecReservoir(10, lcg(1));
        const src = unitVec(4, 2);
        r.add(src);
        src[0] = 999;                       // mutate the source after retention
        expect(r.sample).toHaveLength(1);
        expect(r.sample[0][0]).not.toBe(999); // the copy is isolated
    });

    it('caps the sample size when the stream exceeds the cap', () => {
        const r = new VecReservoir(50, lcg(7));
        for (let k = 0; k < 5000; k++) r.add(unitVec(8, k + 1));
        expect(r.sample).toHaveLength(50);
    });

    it('is deterministic under a seeded rng', () => {
        const build = () => {
            const r = new VecReservoir(20, lcg(2024));
            for (let k = 0; k < 1000; k++) {
                const v = new Float32Array([k]);   // identity = arrival index
                r.add(v);
            }
            return r.sample.map(v => v[0]);
        };
        expect(build()).toEqual(build());
    });

    it('samples roughly uniformly across the stream (not just the head/tail)', () => {
        // 4000 items, cap 400: a uniform sample's mean index ≈ stream midpoint.
        const r = new VecReservoir(400, lcg(555));
        for (let k = 0; k < 4000; k++) r.add(new Float32Array([k]));
        const meanIdx = r.sample.reduce((a, v) => a + v[0], 0) / r.sample.length;
        expect(meanIdx).toBeGreaterThan(1500); // a head-biased "keep first K" would be ~200
        expect(meanIdx).toBeLessThan(2500);
    });
});

describe('matchStrength (display-only combine)', () => {
    const A = 0.8; // denseWeight

    it('is the α-linear blend of the two arms', () => {
        // 0.8·0.9 + 0.2·0.4 = 0.80
        expect(matchStrength(0.9, 0.4, A)).toBeCloseTo(0.8, 6);
    });

    it('returns null when the dense arm is uncalibrated (→ caller shows rank)', () => {
        expect(matchStrength(undefined, 0.95, A)).toBeNull();
        // Null gate is FIRST: a lexical-only row on a small corpus still shows
        // rank, so the list is rank-only UNIFORMLY, not a mix of % and rank.
        expect(matchStrength(undefined, 1.0, A, true)).toBeNull();
    });

    it('uses the lexical arm alone for a lexical-only chunk (calibrated corpus)', () => {
        // A title-only vector's cosine isn't a real semantic match → pLex is the
        // whole story; the (here low) dense confidence is ignored.
        expect(matchStrength(0.05, 0.92, A, true)).toBeCloseTo(0.92, 6);
    });

    it('lets a low dense confidence VETO a lexical false-positive (the noisy-OR fix)', () => {
        // The "Tal Shoer on shoes" case: bm25_norm pinned to 1.0 by a fuzzy match
        // to a rare token, but the dense arm knows it is junk (conf≈0). Linear
        // blend floors it near (1−α); noisy-OR would have read 100%.
        const s = matchStrength(0.02, 1.0, A);
        expect(s).toBeLessThan(0.25);
        expect(s).toBeCloseTo(A * 0.02 + (1 - A) * 1.0, 6);
    });

    it('reads high only when BOTH arms agree (a genuine strong match)', () => {
        expect(matchStrength(0.95, 0.9, A)).toBeGreaterThan(0.9);
    });

    it('stays within [0,1] even if bm25_norm overshoots 1', () => {
        expect(matchStrength(1, 1.4, A)).toBe(1);
        expect(matchStrength(0, 0, A)).toBe(0);
    });
});
