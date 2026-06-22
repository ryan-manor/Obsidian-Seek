// Stage-1 candidate-pool scaling (docs/seek-scaling.md "C"). Pins the √N curve,
// the per-arm floor/ceiling clamps, the flat recency arm, and — the load-bearing
// safety property — that at or below the anchor the caps equal the legacy
// 200/100/50 constants exactly, so shipping this is a no-op for current vaults.

import { describe, it, expect } from 'vitest';
import { poolCaps, scaledCap, POOL_FLOORS, POOL_CEILINGS, POOL_ANCHOR_N } from './pool';

describe('pool — corpus-scaled candidate-pool caps', () => {
    it('at or below the anchor, caps equal the legacy floors exactly (ship = no-op here)', () => {
        for (const n of [0, 1, 100, 1000, 4763 /* live vault today */, POOL_ANCHOR_N]) {
            expect(poolCaps(n)).toEqual({ binary: 200, bm25: 100, recency: 50 });
        }
    });

    it('grows binary + bm25 by √N above the anchor', () => {
        // round(floor · √(N/5000))
        expect(poolCaps(10_000)).toEqual({ binary: 283, bm25: 141, recency: 50 }); // √2
        expect(poolCaps(20_000)).toEqual({ binary: 400, bm25: 200, recency: 50 }); // √4 = 2×
        expect(poolCaps(50_000)).toEqual({ binary: 632, bm25: 316, recency: 50 }); // √10
    });

    it('clamps each scaled arm to its ceiling at large N', () => {
        // binary 800 and bm25 400 are both reached at N = 5000·4² = 80k
        expect(poolCaps(80_000).binary).toBe(POOL_CEILINGS.binary);
        expect(poolCaps(80_000).bm25).toBe(POOL_CEILINGS.bm25);
        const big = poolCaps(5_000_000);
        expect(big.binary).toBe(POOL_CEILINGS.binary);
        expect(big.bm25).toBe(POOL_CEILINGS.bm25);
    });

    it('holds recency flat at its floor regardless of corpus size', () => {
        for (const n of [0, 5_000, 50_000, 500_000]) {
            expect(poolCaps(n).recency).toBe(POOL_FLOORS.recency);
        }
    });

    it('binary + bm25 caps are monotonic non-decreasing in liveN', () => {
        let prevB = -1, prevM = -1;
        for (let n = 0; n <= 300_000; n += 2_500) {
            const c = poolCaps(n);
            expect(c.binary).toBeGreaterThanOrEqual(prevB);
            expect(c.bm25).toBeGreaterThanOrEqual(prevM);
            prevB = c.binary; prevM = c.bm25;
        }
    });

    it('scaledCap: floor below anchor, exact floor at anchor, ceiling at saturation', () => {
        expect(scaledCap(200, 800, 1_000)).toBe(200);              // < anchor → floor
        expect(scaledCap(200, 800, POOL_ANCHOR_N)).toBe(200);      // == anchor → floor
        expect(scaledCap(200, 800, 80_000)).toBe(800);             // 4× → ceiling
        expect(scaledCap(200, 800, 79_000)).toBeLessThan(800);     // just under saturation
        expect(scaledCap(200, 800, 1e9)).toBe(800);               // clamped, never exceeds
    });
});
