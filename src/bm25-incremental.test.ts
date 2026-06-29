// Seek scaling A1 — incremental BM25 maintenance (add / remove / vacuum).
//
// The contract: keeping ONE MultiFieldBM25 alive and mutating it must produce
// the same RANKING a full refit over the same final corpus would — but NOT
// byte-identical scores. MiniSearch maintains avgFieldLength as an incremental
// running mean (add) / subtraction (remove), which diverges in the low bits from
// the batch summation a fresh fit() does, so calcBM25Score (which divides by
// avgFieldLength) differs ~1e-12..1e-9. So parity here is asserted BY chunk_id
// (row order also differs: fresh = dense, incremental = tombstone holes) with an
// epsilon. getQueryBound, by contrast, is pure df/n/idf arithmetic over integers
// and is exact once vacuum() has restored postings==df — which is the whole point
// of test 2 (the bound is wrong BETWEEN discard and vacuum).

import { describe, it, expect } from 'vitest';
import { MultiFieldBM25 } from './bm25';
import type { Chunk } from './types';

function makeChunk(id: string, title: string, content: string, tags: string[] = [], aliases: string[] = []): Chunk {
    return {
        chunk_id: id,
        title,
        content,
        note_path: `${id}.md`,
        heading_path: [],
        metadata: { tags, aliases, created: null, modified: null, properties: {} },
        start_line: 0,
        end_line: 0,
    };
}

function bodiesOf(chunks: Chunk[]): Map<string, string> {
    return new Map(chunks.map(c => [c.chunk_id, c.content]));
}

function fit(chunks: Chunk[]): MultiFieldBM25 {
    return new MultiFieldBM25().fit(chunks, bodiesOf(chunks));
}

// Map a BM25 index's per-query scores back to {chunk_id -> score}, since the row
// layout differs between a fresh fit and an incrementally-maintained index.
// Removed (tombstoned) ids have no row → excluded, which is correct.
function scoresById(bm: MultiFieldBM25, query: string, ids: string[]): Map<string, number> {
    const { scores } = bm.getScoresWithCoverage(query);
    const out = new Map<string, number>();
    for (const id of ids) {
        const row = bm.rowOf(id);
        if (row !== undefined) out.set(id, scores[row]);
    }
    return out;
}

// Two score maps agree by chunk_id within epsilon, same matched-id set.
function expectScoresClose(a: Map<string, number>, b: Map<string, number>, eps = 1e-6): void {
    const matched = (m: Map<string, number>) => [...m].filter(([, s]) => s > 0).map(([id]) => id).sort();
    expect(matched(a)).toEqual(matched(b));
    for (const [id, sa] of a) {
        const sb = b.get(id) ?? 0;
        expect(Math.abs(sa - sb)).toBeLessThan(eps);
    }
}

// Overlapping vocabulary so several terms have df>1 (idf is non-trivial) and the
// length norm varies across docs.
const CORPUS = (): Chunk[] => [
    makeChunk('c1', 'Alpha Note', 'alpha bravo charlie delta echo'),
    makeChunk('c2', 'Bravo Note', 'alpha bravo foxtrot'),
    makeChunk('c3', 'Charlie Note', 'alpha echo foxtrot golf hotel india'),
    makeChunk('c4', 'Delta Note', 'bravo golf hotel'),
    makeChunk('c5', 'Echo Note', 'charlie india juliet'),
];
const ALL_IDS = ['c1', 'c2', 'c3', 'c4', 'c5'];
const QUERIES = ['alpha bravo', 'charlie', 'golf hotel india', 'echo foxtrot'];

describe('MultiFieldBM25 incremental — add()', () => {
    it('fit(C-x) + add(x) ranks identically to fit(C), by chunk_id', () => {
        const all = CORPUS();
        const x = all[4]; // c5
        const fresh = fit(all);
        const incremental = fit(all.slice(0, 4));
        const row = incremental.add(x, x.content);

        // add() is the row authority: x lands at the next slot.
        expect(row).toBe(4);
        expect(incremental.size).toBe(5);
        expect(incremental.liveCount).toBe(5);
        expect(incremental.rowOf('c5')).toBe(4);

        for (const q of QUERIES) {
            expectScoresClose(scoresById(incremental, q, ALL_IDS), scoresById(fresh, q, ALL_IDS));
        }
        // Bound is exact (no avgFieldLength dependence).
        for (const q of QUERIES) {
            expect(incremental.getQueryBound(q)).toBeCloseTo(fresh.getQueryBound(q), 9);
        }
    });
});

describe('MultiFieldBM25 incremental — remove() + vacuum()', () => {
    it('fit(C) + remove(y) + vacuum() ranks identically to fit(C-y), by chunk_id', async () => {
        const all = CORPUS();
        const y = 'c1';
        const liveIds = ALL_IDS.filter(id => id !== y);
        const freshMinusY = fit(all.filter(c => c.chunk_id !== y));

        const incremental = fit(all);
        incremental.remove(y);
        await incremental.vacuum();

        // Tombstone leaves the row count monotonic; liveCount drops.
        expect(incremental.size).toBe(5);          // R unchanged (hole kept)
        expect(incremental.liveCount).toBe(4);
        expect(incremental.rowOf(y)).toBeUndefined();
        expect(incremental.dirtCount).toBe(0);     // vacuum reclaimed the tombstone

        for (const q of QUERIES) {
            expectScoresClose(scoresById(incremental, q, liveIds), scoresById(freshMinusY, q, liveIds));
            expect(incremental.getQueryBound(q)).toBeCloseTo(freshMinusY.getQueryBound(q), 9);
        }
    });

    it('remove() is idempotent and never throws on a missing / already-removed id', async () => {
        const incremental = fit(CORPUS());
        incremental.remove('c1');
        incremental.remove('c1');          // already gone
        incremental.remove('does-not-exist');
        await incremental.vacuum();
        expect(incremental.liveCount).toBe(4);
        expect(incremental.dirtCount).toBe(0);
    });
});

describe('MultiFieldBM25 incremental — getQueryBound is exact immediately, even pre-vacuum', () => {
    it('reflects a discard before vacuum (search-based bound honors tombstones synchronously)', async () => {
        const all = CORPUS();
        const y = 'c1'; // contains "charlie" (also in c5 → df=2 full, 1 after removal)
        const q = 'charlie';
        const freshMinusY = fit(all.filter(c => c.chunk_id !== y)).getQueryBound(q);

        const incremental = fit(all);
        incremental.remove(y);             // discard only — postings NOT yet physically cleaned
        const preVacuumBound = incremental.getQueryBound(q);

        // getQueryBound now scores through MiniSearch (the WAND/MaxScore UB is a
        // single-term search): MiniSearch excludes a discarded doc from results AND
        // updates its scoring stats (documentCount/df/avgFieldLength) the moment
        // discard() runs — only the physical posting cleanup is deferred to vacuum().
        // So the bound is exact pre-vacuum, no stale window. (The old internals-
        // reading bound read tombstoned df here and was stale until vacuum; this
        // assertion flipped from `.not.toBeCloseTo` when the MaxScore bound landed.)
        expect(preVacuumBound).toBeCloseTo(freshMinusY, 9);
        expect(incremental.dirtCount).toBeGreaterThan(0);   // tombstone still physically present

        await incremental.vacuum();
        expect(incremental.getQueryBound(q)).toBeCloseTo(freshMinusY, 9);
        expect(incremental.dirtCount).toBe(0);
    });
});

describe('MultiFieldBM25 incremental — change (remove old id + add new id)', () => {
    it('an edited chunk (new content-hash id) re-ranks like a fresh fit over the edited corpus', async () => {
        const all = CORPUS();
        // Simulate an edit to c2: old chunk c2 removed, new chunk c2b added.
        const c2b = makeChunk('c2b', 'Bravo Note', 'alpha bravo foxtrot kilo lima');
        const editedCorpus = [all[0], c2b, all[2], all[3], all[4]];
        const fresh = fit(editedCorpus);

        const incremental = fit(all);
        incremental.remove('c2');          // drop stale chunk first (reindexDelta order)
        incremental.add(c2b, c2b.content); // then append the new one
        await incremental.vacuum();

        const ids = ['c1', 'c2b', 'c3', 'c4', 'c5'];
        for (const q of [...QUERIES, 'kilo lima', 'bravo foxtrot']) {
            expectScoresClose(scoresById(incremental, q, ids), scoresById(fresh, q, ids));
            expect(incremental.getQueryBound(q)).toBeCloseTo(fresh.getQueryBound(q), 9);
        }
        expect(incremental.liveCount).toBe(5);
        expect(incremental.rowOf('c2')).toBeUndefined();
    });
});
