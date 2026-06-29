import { describe, it, expect } from 'vitest';
import { MultiFieldBM25, FUZZY_BY_LENGTH, PREFIX_LAST_TOKEN } from './bm25';
import { buildBm25Stamp, bm25StampMatches, type Bm25PersistStamp } from './search';
import type { Chunk, SeekSettings } from './types';
import type { MetaConfig } from './index-store';

function makeChunk(
    id: string, title: string, content: string,
    tags: string[] = [], aliases: string[] = [],
    heading_path: string[] = [], properties: Record<string, string> = {},
): Chunk {
    return {
        chunk_id: id,
        title,
        content,
        note_path: `${id}.md`,
        heading_path,
        metadata: { tags, aliases, created: null, modified: null, properties },
        start_line: 0,
        end_line: 0,
    };
}

function corpus(): Chunk[] {
    return [
        makeChunk('a', 'BMW M2 Competition', 'the bmw m2 is a compact sports coupe with a turbo inline six', ['cars', 'reviews'], ['m2'], ['Overview'], { year: '2023' }),
        makeChunk('b', 'Obsidian Plugins', 'building obsidian plugins with typescript and esbuild', ['dev'], ['plugin dev'], ['Setup', 'Build'], { lang: 'typescript' }),
        makeChunk('c', 'Granite Embeddings', 'granite r2 is a modernbert embedding model, cls pooled, 384 dims', ['ml', 'search'], [], ['Model'], {}),
        makeChunk('d', 'Recency Tiebreak', 'epsilon tiebreaker biases ties toward more recent notes', ['search'], [], [], {}),
        makeChunk('e', 'Sports Cars List', 'a list of compact sports coupes and roadsters', ['cars'], [], [], {}),
        makeChunk('f', 'Typescript Tips', 'typescript generics, narrowing, and const assertions', ['dev'], ['ts'], [], {}),
        makeChunk('g', 'Vault Search', 'hybrid search fuses dense cosine and bm25 over the vault', ['search', 'ml'], [], ['Fusion'], {}),
        makeChunk('h', 'Daily Note 2026-06-16', 'met with the team about the search roadmap', ['daily'], [], [], {}),
        makeChunk('i', 'Coupe Buyers Guide', 'how to buy a used sports coupe, what to inspect', ['cars', 'guides'], [], [], {}),
        makeChunk('j', 'Esbuild Config', 'bundling a plugin to a single main.js with esbuild define', ['dev'], [], [], {}),
        makeChunk('k', 'Embedding Quantization', 'int8 scalar quantization of unit-l2 vectors for the rerank tier', ['ml'], [], [], {}),
        makeChunk('l', 'Minisearch Notes', 'minisearch bm25 index, tojson and loadjson serialization', ['dev', 'search'], [], [], {}),
    ];
}

const QUERIES = [
    'bmw m2', 'obsidian plugin', 'typescript', 'sports coupe', 'plugins',
    'granite embedding', 'bm25', 'esbuild', 'search', 'coupe', 'minisearch loadjson',
    'compact sports', 'quantization int8', 'recency',
];

// Mirror the live per-call options (search.ts getScoresWithCoverage call) so the
// round-trip is exercised under the exact analyzer config that ships.
function scoresFor(m: MultiFieldBM25, q: string) {
    const r = m.getScoresWithCoverage(q, { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN });
    return { scores: Array.from(r.scores), coverage: Array.from(r.coverage), bound: r.bound };
}

describe('MultiFieldBM25 persist round-trip — fromJSON(toJSON) is scoring-identical to fit', () => {
    it('produces identical getScoresWithCoverage across queries (default fields)', () => {
        const chunks = corpus();
        const bodies = new Map(chunks.map(c => [c.chunk_id, c.content]));
        const fitted = new MultiFieldBM25().fit(chunks, bodies);
        const loaded = new MultiFieldBM25().fromJSON(fitted.toJSON(), chunks);
        for (const q of QUERIES) {
            expect(scoresFor(loaded, q)).toEqual(scoresFor(fitted, q));
        }
    });

    it('is identical with properties + headings fields enabled', () => {
        const chunks = corpus();
        const bodies = new Map(chunks.map(c => [c.chunk_id, c.content]));
        const opts = { searchableProperties: true, headingsField: true };
        const fitted = new MultiFieldBM25().fit(chunks, bodies, opts);
        const loaded = new MultiFieldBM25().fromJSON(fitted.toJSON(), chunks, opts);
        for (const q of QUERIES) {
            expect(scoresFor(loaded, q)).toEqual(scoresFor(fitted, q));
        }
    });

    it('throws if toJSON is called before fit', () => {
        expect(() => new MultiFieldBM25().toJSON()).toThrow();
    });
});

// The safety property the TOLERANT stamp gate (bm25StampMatches, 2026-06-20) relies
// on: loading a stale blob against a DRIFTED live chunk set is bounded-recall-stale,
// never wrong and never a throw. chunk_id is content-derived, so an edited chunk is a
// delete + an add: its old id lingers in the postings (skipped on load), its new id
// is absent from the postings (scores 0 until reconciled).
describe('MultiFieldBM25 tolerant load — fromJSON survives a drifted chunk set', () => {
    it('skips dead postings, scores new chunks 0, never throws', () => {
        const original = corpus();                          // ids a..l
        const bodies = new Map(original.map(c => [c.chunk_id, c.content]));
        const json = new MultiFieldBM25().fit(original, bodies).toJSON();

        // Drift: 'a' and 'b' deleted; two NEW chunks 'm','n' (content-derived new ids)
        // that were never in the postings. The consumer frame is the LIVE set.
        const drifted: Chunk[] = [
            ...original.filter(c => c.chunk_id !== 'a' && c.chunk_id !== 'b'),
            makeChunk('m', 'New Note Alpha', 'a freshly edited note about typescript decorators', ['dev']),
            makeChunk('n', 'New Note Beta', 'another new note on sports coupe maintenance', ['cars']),
        ];
        const idx = (id: string) => drifted.findIndex(c => c.chunk_id === id);

        const loaded = new MultiFieldBM25().fromJSON(json, drifted);

        // Sizing follows the CONSUMER frame, not the postings.
        for (const q of QUERIES) {
            const { scores } = loaded.getScoresWithCoverage(q, { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN });
            expect(scores.length).toBe(drifted.length);     // never throws, always consumer-sized
            expect(scores[idx('m')]).toBe(0);               // new chunk: absent from postings → 0
            expect(scores[idx('n')]).toBe(0);
        }

        // A surviving chunk still scores at its NEW (drifted) index.
        const ts = loaded.getScoresWithCoverage('typescript', { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN }).scores;
        expect(ts[idx('f')]).toBeGreaterThan(0);            // 'Typescript Tips' survived

        // A query whose ONLY match was a DELETED chunk scores nothing — the dead 'a'
        // posting is skipped, not mis-placed onto some live row.
        const bmw = loaded.getScoresWithCoverage('bmw m2', { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN }).scores;
        expect(Math.max(...bmw)).toBe(0);
    });

    // The freeze regression the adversarial review caught: a delta AFTER a tolerant
    // load must not throw (a throw aborts the incremental patch → full all-bodies
    // refit = the freeze, just deferred to the first relevant edit). Two collisions:
    // remove() of a live id absent from the (stale) postings, and add() of content
    // whose id still has a GHOST posting in the stale blob.
    it('a delta after a tolerant load does not throw (remove blob-missing id + add blob-ghost id)', () => {
        const original = corpus();                 // ids a..l
        const bodies = new Map(original.map(c => [c.chunk_id, c.content]));
        const json = new MultiFieldBM25().fit(original, bodies).toJSON();
        // Live frame: 'a','b' dropped since the blob; a NEW chunk 'm' added since.
        const mChunk = makeChunk('m', 'New Note', 'a fresh note about typescript decorators', ['dev']);
        const bm = new MultiFieldBM25().fromJSON(json, [...original.filter(c => c.chunk_id !== 'a' && c.chunk_id !== 'b'), mChunk]);

        // 'm' is in idToIdx (live) but NOT in the postings → mini.discard would throw pre-fix.
        expect(() => bm.remove('m')).not.toThrow();
        // 'a' is a GHOST (in the postings, not in idToIdx); re-adding content under id
        // 'a' collides with the ghost → mini.add would throw pre-fix. It must land.
        let row = -1;
        expect(() => { row = bm.add(makeChunk('a', 'BMW M2 Competition', 'the bmw m2 is a compact sports coupe with a turbo six'), 'the bmw m2 is a compact sports coupe with a turbo six'); }).not.toThrow();
        const scores = bm.getScores('bmw m2', { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN });
        expect(scores[row]).toBeGreaterThan(0);   // the re-added chunk is searchable at its new row
    });
});

describe('bm25 persist stamp — the relevance guard (TOLERANT: load on compatible analyzer/model/dim/shape; corpus size/timestamp tolerated)', () => {
    const meta: MetaConfig = { embeddingDim: 384, lastIndexedAt: '2026-06-16T10:00:00.000Z', schemaVersion: 7, modelId: 'granite-r2-ml97' };
    const settings = { searchableProperties: false, headingsField: false, boostedBm25: false } as unknown as SeekSettings;
    const live = buildBm25Stamp(meta, 1200, settings);

    it('matches an identical stamp', () => {
        expect(bm25StampMatches(buildBm25Stamp(meta, 1200, settings), live)).toBe(true);
    });

    it('TOLERATES a drifted corpus size/timestamp (the churn fields lastIndexedAt + chunkCount)', () => {
        // 2026-06-20: these two change on every delta/hydrate; gating on them was the
        // freeze. A drifted-but-compatible blob now LOADS (bounded recall staleness,
        // reconciled by the next delta/catch-up — never wrong text; chunk_id is
        // content-derived so a stale posting can't score for current text).
        expect(bm25StampMatches(buildBm25Stamp({ ...meta, lastIndexedAt: '2026-06-16T11:00:00.000Z' }, 1200, settings), live)).toBe(true);
        expect(bm25StampMatches(buildBm25Stamp(meta, 1201, settings), live)).toBe(true);
        expect(bm25StampMatches(buildBm25Stamp({ ...meta, lastIndexedAt: '2030-01-01T00:00:00.000Z' }, 9999, settings), live)).toBe(true);
    });

    it('still rejects a differing model / dim (corpus identity — a model swap drops chunks)', () => {
        expect(bm25StampMatches(buildBm25Stamp({ ...meta, modelId: 'other' }, 1200, settings), live)).toBe(false);
        expect(bm25StampMatches(buildBm25Stamp({ ...meta, embeddingDim: 512 }, 1200, settings), live)).toBe(false);
    });

    it('rejects a differing analyzer version', () => {
        expect(bm25StampMatches({ ...live, analyzerVersion: 'stale-hash' }, live)).toBe(false);
    });

    it('rejects a differing index shape (props / headings toggles)', () => {
        const propsOn = { ...settings, searchableProperties: true } as unknown as SeekSettings;
        const headingsOn = { ...settings, headingsField: true } as unknown as SeekSettings;
        expect(bm25StampMatches(buildBm25Stamp(meta, 1200, propsOn), live)).toBe(false);
        expect(bm25StampMatches(buildBm25Stamp(meta, 1200, headingsOn), live)).toBe(false);
    });

    it('treats boostedBm25 as implying headings (matches ensureBm25 gate)', () => {
        const boosted = { ...settings, boostedBm25: true } as unknown as SeekSettings;
        const s = buildBm25Stamp(meta, 1200, boosted);
        expect(s.headings).toBe(true);
        expect(bm25StampMatches(s, live)).toBe(false); // differs from headings:false live
    });

    it('rejects non-object / null / missing-field stamps', () => {
        expect(bm25StampMatches(null, live)).toBe(false);
        expect(bm25StampMatches('not a stamp', live)).toBe(false);
        expect(bm25StampMatches({}, live)).toBe(false);
        expect(bm25StampMatches({ ...live, headings: undefined } as Partial<Bm25PersistStamp>, live)).toBe(false);
    });
});
