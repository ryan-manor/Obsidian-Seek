import { describe, it, expect } from 'vitest';
import { selectTopNIndices } from './select';
import { topNIndices } from './binary';

// Deterministic LCG (matches binary.test.ts) — reproducible, no Math.random.
function lcg(seed: number) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

// ── Parity oracle ──────────────────────────────────────────────────────────
// The EXACT pre-bounded-heap algorithm: build the eligible index list in
// ascending order, V8-stable-sort by key descending, slice the head. This is
// what `selectTopNIndices` must reproduce — members AND order, including the
// stable-sort tie-break (equal keys keep ascending index).
function referenceTopN(
    keys: number[],
    n: number,
    eligible?: (i: number) => boolean,
): number[] {
    const len = keys.length;
    let idx: number[];
    if (eligible) {
        idx = [];
        for (let i = 0; i < len; i++) if (eligible(i)) idx.push(i);
    } else {
        idx = Array.from({ length: len }, (_, i) => i);
    }
    idx.sort((a, b) => keys[b] - keys[a]); // V8 stable → equal keys keep idx asc
    return n >= idx.length ? idx : idx.slice(0, n);
}

function run(keys: number[], n: number, eligible?: (i: number) => boolean): number[] {
    return selectTopNIndices(keys.length, n, i => keys[i], eligible ?? null);
}

describe('selectTopNIndices — parity with full-sort oracle', () => {
    it('matches the oracle on random keys across N (members AND order)', () => {
        const rnd = lcg(12345);
        for (let trial = 0; trial < 200; trial++) {
            const len = 1 + Math.floor(rnd() * 400);
            const keys = Array.from({ length: len }, () => rnd() * 1000 - 500);
            for (const n of [1, 5, 50, 100, 200, len, len + 10]) {
                expect(run(keys, n)).toEqual(referenceTopN(keys, n));
            }
        }
    });

    // The load-bearing case: scoreAsymmetric is a 1-bit estimator with frequent
    // EXACT ties, and same-day notes tie on Date.parse. Draw keys from a tiny
    // alphabet so almost every value collides — this is where a naive heap that
    // drops the ascending-index tie-break diverges.
    it('matches the oracle under heavy exact ties', () => {
        const rnd = lcg(999);
        for (let trial = 0; trial < 300; trial++) {
            const len = 1 + Math.floor(rnd() * 300);
            const alphabet = 1 + Math.floor(rnd() * 3); // keys ∈ {0..0,1,2}
            const keys = Array.from({ length: len }, () => Math.floor(rnd() * alphabet));
            for (const n of [1, 3, 10, 50, 100, len]) {
                expect(run(keys, n)).toEqual(referenceTopN(keys, n));
            }
        }
    });

    it('matches the oracle with a mask (eligibility filter)', () => {
        const rnd = lcg(777);
        for (let trial = 0; trial < 200; trial++) {
            const len = 1 + Math.floor(rnd() * 400);
            const keys = Array.from({ length: len }, () => Math.floor(rnd() * 4)); // tie-heavy
            const mask = Array.from({ length: len }, () => rnd() < 0.5);
            const eligible = (i: number) => mask[i];
            for (const n of [1, 10, 50, 100, 200]) {
                expect(run(keys, n, eligible)).toEqual(referenceTopN(keys, n, eligible));
            }
        }
    });

    it('handles all-equal scores (every value ties)', () => {
        const keys = new Array(250).fill(7);
        for (const n of [0, 1, 50, 200, 250, 300]) {
            expect(run(keys, n)).toEqual(referenceTopN(keys, n));
        }
    });

    it('handles edge sizes: n=0, len=0, empty/all-false mask', () => {
        expect(run([1, 2, 3], 0)).toEqual([]);
        expect(selectTopNIndices(0, 5, () => 0)).toEqual([]);
        expect(run([1, 2, 3], 5, () => false)).toEqual([]);
        // all-true mask === no mask
        expect(run([3, 1, 2], 2, () => true)).toEqual(referenceTopN([3, 1, 2], 2));
    });

    it('handles negative and mixed-sign keys', () => {
        const keys = [-5, 3, -5, 3, 0, -1, 3];
        for (const n of [1, 2, 3, 7]) {
            expect(run(keys, n)).toEqual(referenceTopN(keys, n));
        }
    });

    // Mirrors topByRecency's wiring: a parallel buffer with NaN for ineligible
    // rows, eligibility = !isNaN. Confirms NaN keys never leak into selection
    // and the surviving order matches a date-desc oracle.
    it('matches a recency-style NaN-eligibility oracle', () => {
        const rnd = lcg(424242);
        for (let trial = 0; trial < 150; trial++) {
            const len = 1 + Math.floor(rnd() * 300);
            // tie-heavy "dates" (same-day collisions), ~40% ineligible (NaN)
            const dates = Array.from({ length: len }, () =>
                rnd() < 0.4 ? NaN : Math.floor(rnd() * 5) * 86_400_000);
            const eligible = (i: number) => !Number.isNaN(dates[i]);
            const ref = referenceTopN(dates, 50, eligible); // oracle skips NaN via eligible
            const got = selectTopNIndices(len, 50, i => dates[i], eligible);
            expect(got).toEqual(ref);
            // and never selects an ineligible (NaN) row
            for (const i of got) expect(Number.isNaN(dates[i])).toBe(false);
        }
    });
});

describe('topNIndices (binary.ts) — drop-in identical to the old full sort', () => {
    it('matches the oracle on Float64Array scores + boolean[] mask', () => {
        const rnd = lcg(2024);
        for (let trial = 0; trial < 150; trial++) {
            const len = 1 + Math.floor(rnd() * 500);
            const arr = new Float64Array(len);
            for (let i = 0; i < len; i++) arr[i] = Math.floor(rnd() * 5); // tie-heavy
            const keys = Array.from(arr);
            const useMask = rnd() < 0.5;
            const mask = useMask ? Array.from({ length: len }, () => rnd() < 0.6) : null;
            const eligible = mask ? (i: number) => mask[i] : undefined;
            for (const n of [100, 200]) {
                expect(topNIndices(arr, n, mask)).toEqual(referenceTopN(keys, n, eligible));
            }
        }
    });

    it('treats null/undefined mask as no filter', () => {
        const arr = new Float64Array([3, 1, 4, 1, 5, 9, 2, 6]);
        const keys = Array.from(arr);
        expect(topNIndices(arr, 3, null)).toEqual(referenceTopN(keys, 3));
        expect(topNIndices(arr, 3, undefined)).toEqual(referenceTopN(keys, 3));
    });
});
