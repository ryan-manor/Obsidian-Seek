import { describe, it, expect } from 'vitest';
import { rank, DEFAULT_RANKING_CONFIG } from './ranker';
import type { Chunk } from './types';

// Minimal chunk factory — only the fields rank() reads matter.
function chunk(note_path: string, opts: { lexicalOnly?: boolean; created?: string | null; modified?: string | null } = {}): Chunk {
    return {
        chunk_id: note_path,
        title: note_path.replace(/\.md$/, '').split('/').pop()!,
        content: '',
        note_path,
        heading_path: [],
        metadata: { tags: [], aliases: [], pageType: '', created: opts.created ?? null, modified: opts.modified ?? null, properties: {} },
        start_line: 1,
        end_line: 1,
        ...(opts.lexicalOnly && { lexicalOnly: true }),
    };
}

// Pin the recency ε-tiebreaker to 0 so these assertions are about TM2C2
// dense/bm25 fusion only (the factory chunks carry no dates anyway, but the
// pin makes that independence explicit rather than incidental).
const cfg = { ...DEFAULT_RANKING_CONFIG, recencyEpsilon: 0 };

describe('rank — TM2C2 fixes the manufactured-winner (diluted gold below a noise distractor)', () => {
    // The wife's bug, reduced: an OOV/ID query where the true match's cosine is
    // DILUTED (long noisy chunk) BELOW a content-free distractor's, while only the
    // true match has a BM25 hit. Under per-query min-max at high alpha this misranks
    // (distractor's noise cosine stretched to 1.00 wins); TM2C2's fixed endpoints
    // keep the dense channel flat so BM25 decides. Verified in ~/eval-oov.
    const chunks = [
        chunk('Notes/2026-03-01 daily.md'),          // content-free distractor, no bm25
        chunk('Notes/Metrics Service requests.md'),  // gold: diluted cosine, bm25 hit
        chunk('Notes/Some Other Note.md'),           // noise
    ];
    const denseRaw = new Float64Array([0.794, 0.752, 0.781]); // gold (0.752) BELOW distractor (0.794)
    const bm25 = new Float64Array([0.0, 1.0, 0.0]);

    it('the BM25-bearing exact match wins at the shipped alpha (0.90)', () => {
        const { results } = rank(chunks, denseRaw, bm25, 'opaqueId', 3, cfg);
        expect(results[0].note_path).toBe('Notes/Metrics Service requests.md');
    });

    it('holds even at alpha 0.95 (where min-max sank the gold to rank 7+)', () => {
        const { results } = rank(chunks, denseRaw, bm25, 'opaqueId', 3, { ...cfg, alpha: 0.95 });
        expect(results[0].note_path).toBe('Notes/Metrics Service requests.md');
    });

    it('dense normalization is the fixed (cos+1)/2 map, reported in signals', () => {
        const { results } = rank(chunks, denseRaw, bm25, 'opaqueId', 3, cfg);
        const distractor = results.find(r => r.note_path.endsWith('2026-03-01 daily.md'))!;
        expect(distractor.ranking_signals.dense).toBeCloseTo((0.794 + 1) / 2, 10);
        expect(distractor.ranking_signals.denseRaw).toBeCloseTo(0.794, 10); // raw cosine preserved
    });
});

describe('rank — lexical-only floor (belt-and-suspenders)', () => {
    it('an empty-stub chunk is floored below the real candidates', () => {
        const chunks = [
            chunk('Notes/stub.md', { lexicalOnly: true }),
            chunk('Notes/real.md'),
        ];
        // Stub raw-cosines HIGH (0.95) but is flagged lexical-only.
        const { results } = rank(chunks, new Float64Array([0.95, 0.60]), new Float64Array([0, 0]), 'q', 2, cfg);
        const stub = results.find(r => r.note_path.endsWith('stub.md'))!;
        const real = results.find(r => r.note_path.endsWith('real.md'))!;
        // Floored to the real candidate's cosine, so it can't win on dense; raw preserved.
        expect(stub.ranking_signals.denseRaw).toBeCloseTo(0.95, 10);
        expect(real.score).toBeGreaterThanOrEqual(stub.score);
    });
});

describe('rank — no lexical-only: plain TM2C2 normalization', () => {
    it('denseRaw mirrors input cosines; dense is (cos+1)/2; bm25 divides by max', () => {
        const chunks = [chunk('a.md'), chunk('b.md')];
        const { results } = rank(chunks, new Float64Array([0.2, 0.8]), new Float64Array([1, 4]), 'q', 2, cfg);
        const a = results.find(r => r.note_path === 'a.md')!;
        const b = results.find(r => r.note_path === 'b.md')!;
        expect(a.ranking_signals.denseRaw).toBeCloseTo(0.2, 10);
        expect(a.ranking_signals.dense).toBeCloseTo(0.6, 10);   // (0.2+1)/2
        expect(b.ranking_signals.dense).toBeCloseTo(0.9, 10);   // (0.8+1)/2
        expect(a.ranking_signals.bm25).toBeCloseTo(0.25, 10);   // 1/4
        expect(b.ranking_signals.bm25).toBeCloseTo(1, 10);      // 4/4
    });
});

describe('rank — recency ε-tiebreaker (2026-06-11, [[Seek Rel]] §Recency Plan)', () => {
    // Dates relative to now: computeRecencyScore defaults referenceDateMs to
    // Date.now(), which is what ships — these tests exercise that path.
    const daysAgo = (d: number) => new Date(Date.now() - d * 86_400_000).toISOString();

    it('breaks an exact score tie newest-first (the dated-series case)', () => {
        // Two series siblings: identical dense + bm25 (the realistic near-tie:
        // same title coverage, near-identical vectors). Only ε·recency differs.
        const chunks = [
            chunk('Notes/Alex 1x1 2025-11-19.md', { created: daysAgo(204) }),
            chunk('Notes/Alex 1x1 2026-05-19.md', { created: daysAgo(23) }),
        ];
        const { results } = rank(chunks, new Float64Array([0.8, 0.8]), new Float64Array([1, 1]), 'zz', 2, DEFAULT_RANKING_CONFIG);
        expect(results[0].note_path).toBe('Notes/Alex 1x1 2026-05-19.md');
    });

    it('cannot flip a real score gap — ε is a tiebreaker, not a lean', () => {
        // The OLDER note is genuinely more relevant (dense 0.80 vs 0.70 → hybrid
        // gap 0.04 at α=0.80, double the entire ε budget). The 06-04 click study:
        // 50% of episodic clicks target an instance >90d old — newest must not
        // override real relevance.
        const chunks = [
            chunk('Notes/Older Better.md', { created: daysAgo(400) }),
            chunk('Notes/Newer Worse.md', { created: daysAgo(1) }),
        ];
        const { results } = rank(chunks, new Float64Array([0.80, 0.70]), new Float64Array([0, 0]), 'zz', 2, DEFAULT_RANKING_CONFIG);
        expect(results[0].note_path).toBe('Notes/Older Better.md');
    });

    it("recencyKey 'modified' ties break on mtime instead of created", () => {
        const chunks = [
            chunk('Notes/CreatedNew.md', { created: daysAgo(2), modified: daysAgo(300) }),
            chunk('Notes/EditedNew.md', { created: daysAgo(700), modified: daysAgo(2) }),
        ];
        const scores = { dense: new Float64Array([0.8, 0.8]), bm25: new Float64Array([1, 1]) };
        const byCreated = rank(chunks, scores.dense, scores.bm25, 'zz', 2, DEFAULT_RANKING_CONFIG).results;
        const byModified = rank(chunks, scores.dense, scores.bm25, 'zz', 2, { ...DEFAULT_RANKING_CONFIG, recencyKey: 'modified' }).results;
        expect(byCreated[0].note_path).toBe('Notes/CreatedNew.md');
        expect(byModified[0].note_path).toBe('Notes/EditedNew.md');
    });

    it('undated chunks get recency 0 — neutral, never penalized below their hybrid score', () => {
        const chunks = [chunk('Notes/Undated.md'), chunk('Notes/Dated.md', { created: daysAgo(10) })];
        const { results } = rank(chunks, new Float64Array([0.9, 0.5]), new Float64Array([0, 0]), 'zz', 2, DEFAULT_RANKING_CONFIG);
        // Real gap: undated-but-better still wins; its recency signal is 0.
        expect(results[0].note_path).toBe('Notes/Undated.md');
        expect(results[0].ranking_signals.recency).toBe(0);
    });
});
