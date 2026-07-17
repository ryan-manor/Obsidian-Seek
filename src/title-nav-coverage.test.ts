// Unit tests for titleNavCoverage (search-modal.ts) — the accessor that recovers
// [0,1] title COVERAGE from the WEIGHTED title_boost signal the ranker stores
// (title_boost = navTitleBoost · coverage, fusion.ts titleMatchBoost).
//
// It backs two surfaces: the debug score row's `title 0.00` figure, and the
// TITLE_NAV_COVERAGE_MIN gate that decides whether a click opens the note at its
// top (and, via seek:insert-link, whether a link targets [[Note]] or [[Note#Section]]).

import { describe, it, expect } from 'vitest';
import { titleNavCoverage } from './search-modal';
import type { ScoredChunk } from './types';

// Only ranking_signals.title_boost is read; the rest is scaffolding.
function chunk(titleBoost: number): ScoredChunk {
    return {
        chunk_id: 'c1',
        note_path: 'Note.md',
        content: '',
        score: 0,
        ranking_signals: {
            dense: 0, bm25: 0, hybrid: 0, recency: 0,
            title_boost: titleBoost,
            denseRaw: 0,
        },
    } as unknown as ScoredChunk;
}

describe('titleNavCoverage', () => {
    it('divides the weight back out to recover coverage', () => {
        // navTitleBoost 0.5 · coverage 0.4 = 0.2 stored → 0.4 recovered.
        expect(titleNavCoverage(chunk(0.2), 0.5)).toBeCloseTo(0.4, 10);
        // A full known-item title match round-trips to exactly 1.
        expect(titleNavCoverage(chunk(0.5), 0.5)).toBeCloseTo(1, 10);
        expect(titleNavCoverage(chunk(0.8), 0.8)).toBeCloseTo(1, 10);
    });

    it('returns 0 — never NaN — when the title bonus is Off', () => {
        // THE GUARD THIS FILE EXISTS FOR. With navTitleBoost 0, titleMatchBoost stores
        // `0 * coverage` = 0, so an unguarded divide is 0/0 = NaN and the score row
        // renders the literal string "title NaN" on every result. The `> 0` guard in
        // titleNavCoverage is load-bearing; do not "simplify" it to a bare division.
        const off = titleNavCoverage(chunk(0), 0);
        expect(Number.isNaN(off)).toBe(false);
        expect(off).toBe(0);
        expect(off.toFixed(2)).toBe('0.00');
    });

    it('reports no coverage for a chunk whose title did not match', () => {
        // titleMatchBoost gates on FULL query coverage, so a partial match stores 0.
        expect(titleNavCoverage(chunk(0), 0.5)).toBe(0);
    });
});
