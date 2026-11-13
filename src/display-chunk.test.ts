import { describe, it, expect } from 'vitest';
import { pickDisplayChunk, distinctTermHits, ownHeading, type DisplayCandidate } from './display-chunk';
import { buildPassageTerms } from './passage';

const terms = (q: string) => buildPassageTerms(q, () => 0);
const id = (s: string) => s;
const chunk = (chunk_id: string, heading_path: string[], content: string, score = 0.5): DisplayCandidate =>
    ({ chunk_id, heading_path, content, score });

describe('distinctTermHits', () => {
    it('counts each term once, word-boundary prefix', () => {
        const t = terms('infinity search prd');
        expect(distinctTermHits('infinity search infinity searching', t)).toBe(2);
        expect(distinctTermHits('reinfinity', t)).toBe(0);
    });
});

describe('ownHeading', () => {
    it('is the last heading segment, empty before the first heading', () => {
        expect(ownHeading({ heading_path: ['A', 'B'] })).toBe('B');
        expect(ownHeading({ heading_path: [] })).toBe('');
    });
});

describe('pickDisplayChunk', () => {
    const t = terms('infinity search prd');

    it('switches to a sibling with strictly more distinct terms', () => {
        const rank = chunk('r', ['Implementation Phases'], 'Phase-1 delivers infinity v1', 0.9);
        const sib = chunk('s', ['Background'], 'Infinity search queries stock backends', 0.7);
        expect(pickDisplayChunk(rank, [sib], t, id)).toBe(sib);
    });

    it('keeps the ranking chunk on a tie', () => {
        const rank = chunk('r', ['Related Pages'], 'Evolving Infinity Search', 0.9);
        const sib = chunk('s', ['Motivation'], 'mature infinity search into a system', 0.7);
        expect(pickDisplayChunk(rank, [sib], t, id)).toBe(rank);
    });

    it('counts the own heading', () => {
        const q = terms('brian guided creations');
        const rank = chunk('r', ['Notes'], 'Brian said no guided creations?', 0.9);
        const sib = chunk('s', ['Guided Creations'], 'Brian walked through the flow', 0.7);
        // both cover 3 → tie keeps rank; drop "brian" from rank to make sib win
        expect(pickDisplayChunk(rank, [sib], q, id)).toBe(rank);
        const weak = chunk('r2', ['Notes'], 'no guided creations?', 0.9);
        expect(pickDisplayChunk(weak, [sib], q, id)).toBe(sib);
    });

    it('among strictly-better siblings, prefers the higher fused score', () => {
        const rank = chunk('r', [], 'nothing relevant', 0.9);
        const a = chunk('a', [], 'infinity search', 0.4);
        const b = chunk('b', [], 'infinity search', 0.6);
        expect(pickDisplayChunk(rank, [a, b], t, id)).toBe(b);
    });

    it('ignores text the plain-text projection removes', () => {
        const strip = (s: string) => s.replace(/\(http[^)]*\)/g, '');
        const rank = chunk('r', [], 'infinity', 0.9);
        const sib = chunk('s', [], 'infinity (http://x/search)', 0.7);
        expect(pickDisplayChunk(rank, [sib], t, strip)).toBe(rank);
    });

    it('no-ops without terms or siblings', () => {
        const rank = chunk('r', [], 'x');
        expect(pickDisplayChunk(rank, [chunk('s', [], 'infinity')], [], id)).toBe(rank);
        expect(pickDisplayChunk(rank, [], t, id)).toBe(rank);
    });
});
